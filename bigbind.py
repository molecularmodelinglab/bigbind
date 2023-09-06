import subprocess
import os
from glob import glob
import sqlite3
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import signal
import requests
from traceback import print_exc

from cfg_utils import get_output_dir
from workflow import Workflow
from task import file_task, simple_task, task
from downloads import StaticDownloadTask
from pdb_to_mol import load_components_dict, mol_from_pdb
from tanimoto_matrix import get_morgan_fps_parallel, get_tanimoto_matrix

def canonicalize(mol):

    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol = Chem.RenumberAtoms(mol, list(order))

    return mol

class timeout:
    def __init__(self, seconds, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class ProgressParallel(Parallel):
    def __call__(self, total, *args, **kwargs):
        with tqdm(total=total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

# first define all the things we need to download
download_chembl = StaticDownloadTask("download_chembl", "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_sqlite.tar.gz")
download_sifts = StaticDownloadTask("download_sifts", "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz")
download_crossdocked = StaticDownloadTask("download_crossdocked", "https://storage.googleapis.com/plantain_data/CrossDocked2022.tar.gz")
download_comp_dict = StaticDownloadTask("download_comp_dict", "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif")

# now we gotta unzip everything

@file_task("sifts.csv", local=True, max_runtime=0.5)
def unzip_sifts(cfg, out_filename, sifts_filename):
    subprocess.run(f"gunzip -c {sifts_filename} > {out_filename}", shell=True, check=True)

@file_task("CrossDocked2022", local=True, max_runtime=2)
def untar_crossdocked(cfg, out_filename, cd_filename):
    """ Hacky -- relies on out_filename being equal to the regular output of tar """
    out_dir = os.path.dirname(out_filename)
    cmd = f"tar -xf {cd_filename} --one-top-level={out_dir}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)

@file_task("chembl.db", local=True, max_runtime=200)
def untar_chembl(cfg, out_filename, chembl_filename):
    out_dir = os.path.dirname(out_filename)
    cmd = f"tar -xf {chembl_filename} --one-top-level={out_dir}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    # want to make this work for future chembl versions as well
    db_file = glob(os.path.join(out_dir, "chembl_*/chembl_*_sqlite/chembl_*.db"))[0]
    os.rename(db_file, out_filename)

@simple_task
def get_chembl_con(cfg, chembl_db_file):
    """ Gets the connection to the chembl sqlite database"""
    con = sqlite3.connect(chembl_db_file)
    return con

@simple_task
def load_sifts_into_chembl(cfg, con, sifts_csv):
    """ Loads SIFTS into the chembl sqlite database for easy sql queries. Note that
    this actually modifies the db file itself for caching purposes. Not ideal to have
    side effects but in this case it can't really hurt anything """
    sifts_df = pd.read_csv(sifts_csv, comment='#')
    cursor = con.cursor()

    # no need to insert if table exists
    cursor.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='sifts' ")
    if cursor.fetchone()[0]==1:
        return con
    
    cursor.execute("create table if not exists sifts (pdb text, chain text sp_primary text, res_beg integer, res_end integer, pdb_beg integer, pdb_end integer, sp_beg integer, sp_end integer)")
    cursor.fetchall()

    sifts_df.to_sql('sifts', con, if_exists='replace', index=False)

    return con

@task(max_runtime=0.1, local=True)
def get_crossdocked_rec_to_ligs(cfg, cd_dir):
    """ Get the pdb files associated with the crystal rec and lig files.
    (the crossdocked types file only lists gninatypes files). Returns a
    dict mapping rec files to a list of lig files that bind to the rec """

    ret = defaultdict(set)
    for pocket in tqdm(glob(f"{cd_dir}/*")):
        for rec_file in glob(pocket + "/*_rec.pdb"):
            for lig_file in glob(pocket + "/*_lig.pdb"):
                ret[rec_file].add(lig_file)
    return ret

@task(max_runtime=0.1)
def get_crossdocked_uniprots(cfg, con, cd_files):
    """ Return all the uniprot IDs from CrossDocked"""
    cd_pdbs = { f.split('/')[-1].split('_')[0] for f in cd_files.keys() }
    cd_chains = { f.split('/')[-1].split('_')[1] for f in cd_files.keys() }
    cd_chains_str = ", ".join(map(lambda s: f"'{s}'", cd_chains))
    cd_pdbs_str = ", ".join(map(lambda s: f"'{s}'", cd_pdbs))

    crossdocked_uniprots = pd.read_sql_query(f"select SP_PRIMARY from sifts where PDB in ({cd_pdbs_str})", con)

    return crossdocked_uniprots

@file_task("activities_chunked.csv", max_runtime=24)
def get_crossdocked_chembl_activities(cfg, out_filename, con, cd_uniprots):
    """ Get all activities (with some filtering for quality) of small
    molecules binding to proteins whose structures are in the crossdocked
    dataset """
    
    cd_uniprots_str = ", ".join(map(lambda s: f"'{s}'", cd_uniprots["SP_PRIMARY"]))
    
    query =   f"""

    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.potential_duplicate,
    COALESCE(act.data_validity_comment, 'valid') as data_validity_comment,
    a.confidence_score,
    td.chembl_id AS target_chembl_id,
    td.target_type,
    c.accession as protein_accession,
    a.chembl_id as assay_id
    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN target_type tt ON td.target_type = tt.target_type
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences c ON tc.component_id = c.component_id
    AND tt.parent_type  = 'PROTEIN' 
    AND act.pchembl_value IS NOT NULL
    AND c.accession in ({cd_uniprots_str});
    
    """
    with open(out_filename, 'w'): pass

    approx_tot = 1571668
    chunksize = 1000
    chunks = pd.read_sql_query(query,con,chunksize=chunksize)
    header = True
    for i, chunk in enumerate(tqdm(chunks, total=int(approx_tot/chunksize))):

        chunk.to_csv(out_filename, header=header, mode='a', index=False)

        header = False

    return pd.read_csv(out_filename)

@task(max_runtime=2)
def filter_activities(cfg, activities_unfiltered):
    """ Remove mixtures, bad assays, and remove duplicates"""
    # remove mixtures
    activities = activities_unfiltered[~activities_unfiltered["canonical_smiles"].str.contains("\.")].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query("potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='")

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(columns=["potential_duplicate", "data_validity_comment", "confidence_score", "target_chembl_id", "target_type", "assay_id"])

    # now we filter duplicates
    dup_indexes = activities.duplicated(keep=False, subset=['compound_chembl_id', 'protein_accession'])
    dup_df = activities[dup_indexes]

    dup_rows = defaultdict(list)
    for i, row in tqdm(dup_df.iterrows(), total=len(dup_df)):
        dup_rows[(row['compound_chembl_id'], row['protein_accession'])].append(row)

    activities = activities[~dup_indexes].reset_index(drop=True)

    new_data = {
        "canonical_smiles": [],
        "compound_chembl_id": [],
        "standard_type": [],
        "standard_relation": [],
        "standard_value": [],
        "standard_units": [],
        "pchembl_value": [],
        "protein_accession": []
    }
    
    uniprot2df = {} # we want to cache the ligand dfs we find for each protein

    # take the median value of duplicated measurements
    for (chembl_id, uniprot), rows in tqdm(dup_rows.items()):
        st_types = [ r.standard_type for r in rows ]
        pchembl_values = [ r.pchembl_value for r in rows ]
        
        
        final_pchembl = np.median(pchembl_values)
        final_st_type = "mixed"
        final_nM = 10**(9-final_pchembl)
        new_data["canonical_smiles"].append(rows[0].canonical_smiles)
        new_data["compound_chembl_id"].append(chembl_id)
        new_data["standard_type"].append(final_st_type)
        new_data["standard_relation"].append("=")
        new_data["standard_value"].append(final_nM)
        new_data["standard_units"].append('nM')
        new_data["pchembl_value"].append(final_pchembl)
        new_data["protein_accession"].append(uniprot)

    new_data_df = pd.DataFrame(new_data)
    activities = pd.concat([activities, new_data_df])

    return activities

@simple_task
def load_act_unfiltered(cfg, filename):
    return pd.read_csv(filename)

@task(max_runtime=0.1)
def save_activities_unfiltered(cfg, activities_unfiltered):
    activities_unfiltered.to_csv(get_output_dir(cfg) + "/activities_unfiltered.csv", index=False)

# ZINC yeets any molecule containing other elements, so shall we
allowed_atoms = { "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I" }
min_atoms_in_mol = 5
max_mol_weight = 1000

def save_mol_sdf(cfg, name, smiles, num_embed_tries=10, verbose=False, filename=None, ret_mol=False):
    """ Embed + UFF optimize single mol"""

    periodic_table = Chem.GetPeriodicTable()
    
    if filename is None:
        folder = get_output_dir(cfg) + "/chembl_structures"
        os.makedirs(folder, exist_ok=True)
        filename = folder + "/" + name + ".sdf"

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    if mol.GetNumAtoms() < min_atoms_in_mol: 
        if verbose:
            print(f"Rejecting {smiles} because it has only {mol.GetNumAtoms()} atoms")
        return False

    mw = Descriptors.MolWt(mol)
    if mw > max_mol_weight:
        if verbose:
            print(f"Rejecting {smiles} because it is too heavy {mw=}")
        return False

    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        sym = Chem.PeriodicTable.GetElementSymbol(periodic_table, num)
        if sym not in allowed_atoms:
            if verbose:
                print(f"Rejecting {smiles} because it contains {sym}.")
            return False

    # early exit cus this takes a while and Paul doesn't need the sdfs
    # return True

    try:
        with timeout(20):
            # try to come up with a 3d structure
            for i in range(num_embed_tries):
                conf_id = AllChem.EmbedMolecule(mol)
                if conf_id == 0:
                    try:
                        AllChem.UFFOptimizeMolecule(mol, 500)
                    except RuntimeError:
                        return False
                    break
            else:
                return False
    except TimeoutError:
        return False

    writer = Chem.SDWriter(filename)
    writer.write(mol)
    writer.close()

    if ret_mol:
        return next(Chem.SDMolSupplier(filename, sanitize=True))

    return True

SDF_N_CPU = 64
SDF_TOTAL_RUNTIME=64
@task(max_runtime=SDF_TOTAL_RUNTIME/SDF_N_CPU, n_cpu=SDF_N_CPU, mem=1*SDF_N_CPU)
def save_all_mol_sdfs(cfg, activities):
    """ Embed and UFF optimize each mol in BigBind"""
    unique_smiles = activities.canonical_smiles.unique()
    smiles2name = {}
    for i, smiles in enumerate(unique_smiles):
        smiles2name[smiles] = f"mol_{i}"

    smiles2filename = {}
    results = ProgressParallel(n_jobs=SDF_N_CPU)(len(smiles2name), (delayed(save_mol_sdf)(cfg, name, smiles) for smiles, name in smiles2name.items()))
    for result, (smiles, name) in zip(results, smiles2name.items()):
        if result:
            filename = f"chembl_structures/{name}.sdf"
            smiles2filename[smiles] = filename

    return smiles2filename

@task(max_runtime=0.2)
def add_sdfs_to_activities(cfg, activities, smiles2filename):
    filename_col = []
    for smiles in activities.canonical_smiles:
        if smiles in smiles2filename:
            filename_col.append(smiles2filename[smiles])
        else:
            filename_col.append("error")

    activities["lig_file"] = filename_col
    
    activities = activities.query("lig_file != 'error'").reset_index(drop=True)
    return activities

@task(max_runtime=0.2)
def get_chain_to_uniprot(cfg, con):
    """ Map PDB ids and chains to uniprot ids """
    sifts_df = pd.read_sql_query("SELECT * FROM SIFTS", con)
    chain2uniprot = {}
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
    return chain2uniprot

@task(max_runtime=0.3, num_outputs=6, local=True)
def get_uniprot_dicts(cfg, cd_files, chain2uniprot):
    """ Get a bunch of dictionaries mapping uniprot id to and from
    rec file, lig file, and pocketome pocket """
    uniprot2recs = defaultdict(set)
    uniprot2ligs = defaultdict(set)
    uniprot2pockets = defaultdict(set)
    pocket2recs = defaultdict(set)
    pocket2ligs = defaultdict(set)
    for rec_file, lig_files in tqdm(cd_files.items(), total=len(cd_files)):
        *rest, folder, file = rec_file.split("/")
        pocket = folder
        rec_id, rec_chain, *rest = file.split("_")
        key = (rec_id, rec_chain)
        for lig_file in lig_files:
            if key in chain2uniprot:
                uniprot = chain2uniprot[key]
                uniprot2recs[uniprot].add(rec_file)
                uniprot2ligs[uniprot].add(lig_file)
                uniprot2pockets[uniprot].add(pocket)

            pocket2recs[pocket].add(rec_file)
            pocket2ligs[pocket].add(lig_file)

    
    pocket2uniprots = defaultdict(set)
    for uniprot, pockets in uniprot2pockets.items():
        for pocket in pockets:
            pocket2uniprots[pocket].add(uniprot)

    return uniprot2recs,\
        uniprot2ligs,\
        uniprot2pockets,\
        pocket2uniprots,\
        pocket2recs,\
        pocket2ligs


def download_lig_sdf(cfg, name, lig):
    url = get_lig_url(lig)
    res = requests.get(url)
    return res.text

@task(max_runtime=10)
def download_all_lig_sdfs(cfg, uniprot2ligs, rigorous=False):
    """ Returns a mapping from lig file to text associated with
    its downloaded SDF file """
    ret = {}
    tot_ligs = 0
    lf_errors = []
    for uniprot in tqdm(uniprot2ligs):
        ligs = uniprot2ligs[uniprot]
        for lig in ligs:
            tot_ligs += 1
            try:
                ret[lig] = download_lig_sdf(cfg, lig.replace("/", "_"), lig)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error in {lig}")
                if rigorous:
                    raise
                else:
                    lf_errors.append(lig)
                    print_exc()
    print(f"Successfully downloaded {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/tot_ligs)*100}%)")
    return ret


def get_crystal_lig(cfg, name, lig_file, sdf_text, align_cutoff=2.0):
    """ Returns the RDKit mol object. The conformation is specified
    in the pdb lig_file and the connectivity is specified in sdf_text """
    lig = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_text.encode('utf-8'))))
    lig_untrans = deepcopy(lig)
    lig_pdb = Chem.MolFromPDBFile(lig_file)

    assert lig_pdb is not None
    assert lig is not None
    assert '.' not in Chem.MolToSmiles(lig)
    assert lig.GetNumAtoms() == lig_pdb.GetNumAtoms()
    
    conf = lig_pdb.GetConformer(0)
    lig.RemoveConformer(0)
    lig.AddConformer(conf)

    # sometimes the coordinates are in a different order?
    # for now just throw these out
    assert AllChem.AlignMol(lig_untrans, lig) < align_cutoff
    
    return lig

@task(max_runtime=1)
def get_all_crystal_ligs(cfg, comp_dict, lig_sdfs, rigorous=False, verbose=False):
    """ Returns a dict mapping ligand files to ligand mol objects """
    lf_errors = []
    ret = {}
    for lf, sdf in tqdm(lig_sdfs.items()):
        try:
            ret[lf] = mol_from_pdb(lf, comp_dict)
            # ret[lf] = get_lig(cfg, lf.replace("/", "_"), lf, sdf)
        except KeyboardInterrupt:
            raise
        except:
            try:
                ret[lf] = get_crystal_lig(cfg, lf.replace("/", "_"), lf, sdf)
            except KeyboardInterrupt:
                raise
            except:
                if verbose:
                    print(f"Error in {lf}")
                if rigorous:
                    raise
                else:
                    if verbose:
                        print_exc()
                    lf_errors.append(lf)
    print(f"Successfully obtained {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/len(lig_sdfs))*100}%)")
    return ret

@task(max_runtime=0.1, num_outputs=2)
def filter_uniprots(cfg, uniprot2pockets, pocket2uniprots):
    """ Return only the uniprot IDs and pocketome pockets that we're using. Only use
    proteins with a single pocket """
    filtered_uniprots = { uniprot for uniprot, pockets in uniprot2pockets.items() if len(pockets) == 1 }
    filtered_pockets = { pocket for pocket, uniprots in pocket2uniprots.items() if len(filtered_uniprots.intersection(uniprots)) }
    print(f"Found {len(filtered_uniprots)} proteins with only 1 pocket out of {len(uniprot2pockets)} (Success rate {100*len(filtered_uniprots)/len(uniprot2pockets)}%)")
    print(f"Found {len(filtered_pockets)} pockets with valid proteins out of {len(pocket2uniprots)} (Success rate {100*len(filtered_pockets)/len(pocket2uniprots)}%)")
    return filtered_uniprots, filtered_pockets

@task(max_runtime=3, num_outputs=4)
def save_all_structures(cfg,
                        final_pockets,
                        pocket2uniprots,
                        pocket2recs,
                        pocket2ligs,
                        ligfile2lig):
    """ Saves the pdb receptor and sdf ligand files. Returns new
    dicts that use _those_ filename instead of crossdocked's """

    folder = get_output_dir(cfg)
    
    my_pocket2recs = defaultdict(set)
    my_pocket2ligs = defaultdict(set)
    my_ligfile2lig = {}
    my_ligfile2uff_lig = {}
    
    for pocket in tqdm(pocket2uniprots): # tqdm(final_pockets):
        out_folder = folder + "/" + pocket
        
        recfiles = pocket2recs[pocket]
        ligfiles = pocket2ligs[pocket]

        ligs = set()
        for ligfile in ligfiles:
            if ligfile not in ligfile2lig: continue
            
            lig = ligfile2lig[ligfile]
            ligs.add(lig)
            
            os.makedirs(out_folder, exist_ok=True)
            out_file = out_folder + "/" + ligfile.split("/")[-1].split(".")[0] + ".sdf"
            my_pocket2ligs[pocket].add(out_file)
            my_ligfile2lig[out_file] = lig
            writer = Chem.SDWriter(out_file)
            writer.write(lig)
            writer.close()

            uff_filename = out_folder + "/" + ligfile.split("/")[-1].split(".")[0] + "_uff.sdf"
            # if os.path.exists(uff_filename):
            #     my_ligfile2uff_lig[out_file] = uff_filename
            uff = save_mol_sdf(cfg, None, Chem.MolToSmiles(lig), filename=uff_filename, ret_mol=True)
            if uff:
                uff_noh = canonicalize(Chem.RemoveHs(uff))
                lig_noh = canonicalize(Chem.RemoveHs(lig))
                if Chem.MolToSmiles(uff_noh, isomericSmiles=False) != Chem.MolToSmiles(lig_noh, isomericSmiles=False):
                    print(f"Error saving uff for {out_file}")
                    my_ligfile2uff_lig[out_file] = "none"
                else:
                    my_ligfile2uff_lig[out_file] = uff_filename
            else:
                my_ligfile2uff_lig[out_file] = "none"

        if len(ligs):
            for recfile in recfiles:
                out_file = out_folder + "/" + recfile.split("/")[-1]
                my_pocket2recs[pocket].add(out_file)
                shutil.copyfile(recfile, out_file)

    return my_pocket2recs, my_pocket2ligs, my_ligfile2lig, my_ligfile2uff_lig

@task(num_outputs=2)
def save_all_pockets(cfg, pocket2recs, pocket2ligs, ligfile2lig):
    """ """
    rec2pocketfile = {}
    rec2res_num = {}
    for pocket, recfiles in tqdm(pocket2recs.items()):
        ligfiles = pocket2ligs[pocket]
        ligs = { ligfile2lig[lf] for lf in ligfiles }
        pocket_files, res_numbers = save_pockets(recfiles, ligs, lig_dist_cutoff=5)
        for recfile, pocket_file in pocket_files.items():
            res_num = res_numbers[recfile]
            rec2pocketfile[recfile] = pocket_file
            rec2res_num[recfile] = res_num
    return rec2pocketfile, rec2res_num

@task(num_outputs=2)
def get_all_pocket_bounds(cfg, pocket2ligs, ligfile2lig, padding=4):
    """ Return the centroids and box sizes for each pocket """
    centers = {}
    sizes = {}
    for pocket, ligfiles in tqdm(pocket2ligs.items()):
        ligs = { ligfile2lig[lf] for lf in ligfiles }
        center, size = get_bounds(cfg, ligs, padding)
        centers[pocket] = center
        sizes[pocket] = size
    return centers, sizes


def make_bigbind_workflow():

    sifts_zipped = download_sifts()
    sifts_csv = unzip_sifts(sifts_zipped)

    crossdocked_tarred = download_crossdocked()
    cd_dir = untar_crossdocked(crossdocked_tarred)

    chembl_tarred = download_chembl()
    chembl_db_file = untar_chembl(chembl_tarred)

    con = get_chembl_con(chembl_db_file)
    con = load_sifts_into_chembl(con, sifts_csv)

    cd_rf2lf = get_crossdocked_rec_to_ligs(cd_dir)

    cd_uniprots = get_crossdocked_uniprots(con, cd_rf2lf)
    activities_unfiltered_fname = get_crossdocked_chembl_activities(con, cd_uniprots)
    activities_unfiltered = load_act_unfiltered(activities_unfiltered_fname)
    saved_act_unf = save_activities_unfiltered(activities_unfiltered)

    activities_filtered = filter_activities(activities_unfiltered)

    smiles2filename = save_all_mol_sdfs(activities_filtered)
    activities_filtered = add_sdfs_to_activities(activities_filtered, smiles2filename)

    chain2uniprot = get_chain_to_uniprot(con)
    
    uniprot2recs,\
    uniprot2ligs,\
    uniprot2pockets,\
    pocket2uniprots,\
    pocket2recs,\
    pocket2ligs = get_uniprot_dicts(cd_rf2lf, chain2uniprot)
    final_uniprots, final_pockets = filter_uniprots(uniprot2pockets,
                                                    pocket2uniprots)
    

    comp_dict_file = download_comp_dict()
    comp_dict = load_components_dict(comp_dict_file)
    
    lig_sdfs = download_all_lig_sdfs(uniprot2ligs)
    ligfile2lig = get_all_crystal_ligs(comp_dict, lig_sdfs)

    pocket2recs,\
    pocket2ligs,\
    ligfile2lig,\
    ligfile2uff = save_all_structures(final_pockets,
                                      pocket2uniprots,
                                      pocket2recs,
                                      pocket2ligs,
                                      ligfile2lig)
    
    rec2pocketfile, rec2res_num = save_all_pockets(pocket2recs, pocket2ligs, ligfile2lig)
    pocket_centers, pocket_sizes = get_all_pocket_bounds(pocket2ligs, ligfile2lig)

    lig_smi, lig_fps = get_morgan_fps_parallel(activities_filtered)
    lig_sim_mat = get_tanimoto_matrix(fps)

    return Workflow(
        pocket_centers,
        lig_sim_mat
    )

@task()
def error(cfg):
    raise Exception("!! This is a test !!")

if __name__ == "__main__":
    from cfg_utils import get_config
    workflow = make_bigbind_workflow()
    cfg = get_config("local")

    # workflow = Workflow(error())
    # workflow.run_node(cfg, workflow.nodes[0])

    # for node in workflow.nodes:
    #     print(node)

    print(workflow.run(cfg))

    # cd_nodes = workflow.out_nodes # find_nodes("untar_crossdocked")
    # levels = workflow.get_levels(cd_nodes)
    # print(levels)