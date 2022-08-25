import pandas as pd
import sqlite3
import os
import pickle
import shutil
import yaml
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from cache import cache

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
def load_crossdocked_prefixes(cfg):
    crossdocked_types = cfg["crossdocked_folder"] + "/types/it2_tt_v1.1_completeset_train0.types"

    all_rec_prefixes = set()
    all_lig_prefixes = set()
    num_chunks = 225838
    for i, chunk in tqdm(enumerate(pd.read_csv(crossdocked_types, sep=' ', names=["label", "binding_affinity", "crystal_rmsd", "rec_file", "lig_file", "vina_score"], chunksize=100)), total=num_chunks):
        lig_prefix = chunk["lig_file"].apply(lambda lf: "_".join(lf.split("_")[:-4]))
        rec_prefix = chunk["rec_file"].apply(lambda lf: "_".join(lf.split("_")[:-1]))
        for rec_prefix, lig_prefix in zip(rec_prefix, lig_prefix):
            all_rec_prefixes.add(rec_prefix)
            all_lig_prefixes.add(lig_prefix)

    return all_rec_prefixes, all_lig_prefixes

@cache
def get_crossdocked_uniprots(cfg, con, rec_prefixes):
    cd_pdbs = { f.split('/')[-1].split('_')[0] for f in rec_prefixes }
    cd_chains = { f.split('/')[-1].split('_')[1] for f in rec_prefixes }
    cd_chains_str = ", ".join(map(lambda s: f"'{s}'", cd_chains))
    cd_pdbs_str = ", ".join(map(lambda s: f"'{s}'", cd_pdbs))

    crossdocked_uniprots = pd.read_sql_query(f"select SP_PRIMARY from sifts where PDB in ({cd_pdbs_str})", con)

    return crossdocked_uniprots

@cache
def get_crossdocked_chembl_activities(cfg, con, cd_uniprots):
    cd_uniprots_str = ", ".join(map(lambda s: f"'{s}'", cd_uniprots["SP_PRIMARY"]))
    
    query =   f"""

    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
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
    AND act.potential_duplicate = 0
    AND a.confidence_score >= 8
    AND act.data_validity_comment IS NULL
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

def run(cfg):
    if cfg["cache"]:
        os.makedirs(cfg["cache_folder"], exist_ok=True)
    con = get_chembl_con(cfg)
    load_sifts_into_chembl(cfg, con)
    rec_prefixes, lig_prefixes = load_crossdocked_prefixes(cfg)
    cd_uniprots = get_crossdocked_uniprots(cfg, con, rec_prefixes)
    initial_activities = get_crossdocked_chembl_activities(cfg, con, cd_uniprots)
    print(initial_activities)
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)




