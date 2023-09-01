import subprocess
import os
from glob import glob
import sqlite3
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import signal

from cfg_utils import get_output_dir
from workflow import Workflow
from task import file_task, simple_task, task
from downloads import StaticDownloadTask

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

@task(max_runtime=0.1)
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
@task(max_runtime=SDF_TOTAL_RUNTIME/SDF_N_CPU, n_cpu=SDF_N_CPU)
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

    return Workflow(
        saved_act_unf,
        activities_filtered,
    )

if __name__ == "__main__":
    from cfg_utils import get_config
    workflow = make_bigbind_workflow()
    cfg = get_config("local")

    workflow.run(cfg)

    # cd_nodes = workflow.out_nodes # find_nodes("untar_crossdocked")
    # levels = workflow.get_levels(cd_nodes)
    # print(levels)