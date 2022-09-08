import pandas as pd
import sqlite3
import os
import pickle
import shutil
import requests
from traceback import print_exc
import yaml
import io
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict

from bigbind import get_lig_url
from cache import cache, item_cache

import signal

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

def get_chembl_con(cfg):
    con = sqlite3.connect(cfg["chembl_file"])
    return con

def load_sifts_into_chembl(cfg, con):
    sifts_df = pd.read_csv(cfg["sifts_file"], comment='#')
    cursor = con.cursor()

    if cfg["cache"]:
        # no need to insert if table exists
        cursor.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='sifts' ")
        if cursor.fetchone()[0]==1:
            return
    
    cursor.execute("create table if not exists sifts (pdb text, chain text sp_primary text, res_beg integer, res_end integer, pdb_beg integer, pdb_end integer, sp_beg integer, sp_end integer)")
    cursor.fetchall()

    sifts_df.to_sql('sifts', con, if_exists='replace', index=False)

@cache
def load_crossdocked_files(cfg):
    """ Get the pdb files associated with the crystal rec and lig files.
    (the crossdocked types file only lists gninatypes files). Returns a
    dict mapping rec files to a list of lig files that bind to the rec """
    crossdocked_types = cfg["crossdocked_folder"] + "/types/it2_tt_v1.1_completeset_train0.types"
    
    ret = defaultdict(set)
    num_chunks = 225838
    for i, chunk in tqdm(enumerate(pd.read_csv(crossdocked_types, sep=' ', names=["label", "binding_affinity", "crystal_rmsd", "rec_file", "lig_file", "vina_score"], chunksize=100)), total=num_chunks):
        folder = chunk["rec_file"].apply(lambda rf: rf.split("/")[0])
        lig_prefix = chunk["lig_file"].apply(lambda lf: lf.split("_lig_")[0].split("_rec_")[1])
        lig_file = folder + lig_prefix.apply(lambda pre: "/" + pre + "_lig.pdb")
        lig_file = lig_file.apply(lambda lf: cfg["crossdocked_folder"] + "/" + lf)
        rec_file = chunk["rec_file"].apply(lambda lf: cfg["crossdocked_folder"] + "/" + "_".join(lf.split("_")[:-1]) + ".pdb")
        for rec_file, lig_file in zip(rec_file, lig_file):
            assert rec_file.endswith("_rec.pdb")
            assert lig_file.endswith("_lig.pdb")
            ret[rec_file].add(lig_file)

    return ret

@cache
def get_crossdocked_uniprots(cfg, con, cd_files):
    cd_pdbs = { f.split('/')[-1].split('_')[0] for f in cd_files.keys() }
    cd_chains = { f.split('/')[-1].split('_')[1] for f in cd_files.keys() }
    cd_chains_str = ", ".join(map(lambda s: f"'{s}'", cd_chains))
    cd_pdbs_str = ", ".join(map(lambda s: f"'{s}'", cd_pdbs))

    crossdocked_uniprots = pd.read_sql_query(f"select SP_PRIMARY from sifts where PDB in ({cd_pdbs_str})", con)

    return crossdocked_uniprots

"""
extra stuff we yeeted from the og query so we can filter later:
AND act.potential_duplicate = 0
AND a.confidence_score >= 8
AND act.data_validity_comment IS NULL
"""

@cache
def get_crossdocked_chembl_activities(cfg, con, cd_uniprots):
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
    
    filename = cfg["cache_folder"] + "/get_crossdocked_chembl_activities_chunked.csv"
    with open(filename, 'w'): pass

    chunks = pd.read_sql_query(query,con,chunksize=1000)
    header = True
    for i, chunk in enumerate(tqdm(chunks)):

        chunk.to_csv(filename, header=header, mode='a', index=False)

        header = False

    return pd.read_csv(filename)

@cache
def save_activities_unfiltered(cfg, activities_unfiltered):
    activities_unfiltered.to_csv(cfg["bigbind_folder"] + "/activities_unfiltered.csv", index=False)

@cache
def filter_activities(cfg, activities_unfiltered):
    # remove mixtures
    activities = activities_unfiltered[~activities_unfiltered["canonical_smiles"].str.contains("\.")].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query("potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='")

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(columns=["compound_chembl_id", "potential_duplicate", "data_validity_comment", "confidence_score", "target_chembl_id", "target_type", "assay_id"])

    # now we filter duplicates. For now, just use the median for all duplicated measurements
    dup_indexes = activities.duplicated(keep=False, subset=['canonical_smiles', 'protein_accession'])
    dup_df = activities[dup_indexes]

    dup_rows = defaultdict(list)
    for i, row in tqdm(dup_df.iterrows(), total=len(dup_df)):
        dup_rows[(row['canonical_smiles'], row['protein_accession'])].append(row)

    activities = activities[~dup_indexes].reset_index(drop=True)

    new_data = {
        "canonical_smiles": [],
        "standard_type": [],
        "standard_relation": [],
        "standard_value": [],
        "standard_units": [],
        "pchembl_value": [],
        "protein_accession": []
    }
    
    for (smiles, uniprot), rows in tqdm(dup_rows.items()):
        st_types = { r.standard_type for r in rows }
        if len(st_types) == 1:
            st_type = next(iter(st_types))
        else:
            st_type = "mixed"
        pchembl_values = [ r.pchembl_value for r in rows ]
        final_pchembl = np.median(pchembl_values)
        final_nM = 10**(9-final_pchembl)
        new_data["canonical_smiles"].append(smiles)
        new_data["standard_type"].append(st_type)
        new_data["standard_relation"].append("=")
        new_data["standard_value"].append(final_nM)
        new_data["standard_units"].append('nM')
        new_data["pchembl_value"].append(final_pchembl)
        new_data["protein_accession"].append(uniprot)

    new_data_df = pd.DataFrame(new_data)
    activities = pd.concat([activities, new_data_df])

    return activities

def save_mol_sdf(cfg, name, smiles, num_embed_tries=10, verbose=False):

    periodic_table = Chem.GetPeriodicTable()
    # ZINC yeets any molecule containing other elements, so shall we
    allowed_atoms = { "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I" }
    
    folder = cfg["bigbind_folder"] + "/chembl_structures"
    os.makedirs(folder, exist_ok=True)
    filename = folder + "/" + name + ".sdf"
    if cfg["cache"] and "save_mol_sdf" not in cfg["recalc"]:
        if os.path.exists(filename):
            return True
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        sym = Chem.PeriodicTable.GetElementSymbol(periodic_table, num)
        if sym not in allowed_atoms and verbose:
            print(f"Rejecting {smiles} because it contains {sym}.")
            return False

    try:
        with timeout(20):
            # try to come up with a 3d structure
            for i in range(num_embed_tries):
                conf_id = AllChem.EmbedMolecule(mol)
                if conf_id == 0:
                    break
            else:
                return False
    except TimeoutError:
        return False

    writer = Chem.SDWriter(filename)
    writer.write(mol)

    return True
        
@cache
def save_all_mol_sdfs(cfg, activities):
    unique_smiles = activities.canonical_smiles.unique()
    smiles2name = {}
    for i, smiles in enumerate(unique_smiles):
        smiles2name[smiles] = f"mol_{i}"

    smiles2filename = {}
    results = ProgressParallel(n_jobs=8)(len(smiles2name), (delayed(save_mol_sdf)(cfg, name, smiles) for smiles, name in smiles2name.items()))
    for result, (smiles, name) in zip(results, smiles2name.items()):
        if result:
            filename = f"chembl_structures/{name}.sdf"
            smiles2filename[smiles] = filename

    return smiles2filename

@cache
def add_sdfs_to_activities(cfg, activities, smiles2filename):
    filename_col = []
    for smiles in activities.canonical_smiles:
        if smiles in smiles2filename:
            filename_col.append(smiles2filename[smiles])
        else:
            filename_col.append("error")

    activities["structure_file"] = filename_col
    
    activities = activities.query("structure_file != 'error'").reset_index(drop=True)
    return activities
        
@cache
def get_chain_to_uniprot(cfg, con):
    """ Map PDB ids and chains to uniprot ids """
    sifts_df = pd.read_sql_query("SELECT * FROM SIFTS", con)
    chain2uniprot = {}
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
    return chain2uniprot

@cache
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

    return uniprot2recs,\
        uniprot2ligs,\
        uniprot2pockets,\
        pocket2recs,\
        pocket2ligs

@item_cache
def download_lig_sdf(cfg, name, lig):
    url = get_lig_url(lig)
    res = requests.get(url)
    return res.text

@cache
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
                # out_file = f"{lig}_untrans.sdf"
                # if c.calculate:
                #     url = get_lig_url(lig)
                #     res = requests.get(url)
                #     c.save(res.text)
                # with open(out_file, "w") as f:
                #     f.write(res.text)
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

@item_cache
def get_lig(cfg, name, lig_file, sdf_text):
    """ Returns the RDKit mol object. The conformation is specified
    in the pdb lig_file and the connectivity is specified in sdf_text """
    lig = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_text.encode('utf-8'))))
    lig_pdb = Chem.MolFromPDBFile(lig_file)

    assert lig_pdb is not None
    assert lig is not None
    assert '.' not in Chem.MolToSmiles(lig)
    
    return lig

def get_all_ligs(cfg, lig_sdfs, rigorous=False):
    """ Returns a dict mapping ligand files to ligand mol objects """
    lf_errors = []
    ret = {}
    for lf, sdf in lig_sdfs.items():
        try:
            ret[lf] = get_lig(cfg, lf.replace("/", "_"), lf, sdf)
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error in {lf}")
            if rigorous:
                raise
            else:
                print_exc()
                lf_errors.append(lf)
    print(f"Successfully obtained {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/len(lig_sdfs))*100}%)")
    return ret

def run(cfg):
    os.makedirs(cfg["bigbind_folder"], exist_ok=True)
    if cfg["cache"]:
        os.makedirs(cfg["cache_folder"], exist_ok=True)
        
    con = get_chembl_con(cfg)
    load_sifts_into_chembl(cfg, con)
    cd_files = load_crossdocked_files(cfg)
    cd_uniprots = get_crossdocked_uniprots(cfg, con, cd_files)
    activities_unfiltered = get_crossdocked_chembl_activities(cfg, con, cd_uniprots)
    save_activities_unfiltered(cfg, activities_unfiltered)
    activities_filtered = filter_activities(cfg, activities_unfiltered)
    smiles2filename= save_all_mol_sdfs(cfg, activities_filtered)
    activities_filtered = add_sdfs_to_activities(cfg, activities_filtered, smiles2filename)
    
    
    chain2uniprot = get_chain_to_uniprot(cfg, con)
    
    uniprot2recs,\
    uniprot2ligs,\
    uniprot2pockets,\
    pocket2recs,\
    pocket2ligs = get_uniprot_dicts(cfg, cd_files, chain2uniprot)
    
    lig_sdfs = download_all_lig_sdfs(cfg, uniprot2ligs)
    # ligs = get_all_ligs(cfg, lig_sdfs)
    # print(ligs)
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)




