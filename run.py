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

def run(cfg):
    if cfg["cache"]:
        os.makedirs(cfg["cache_folder"], exist_ok=True)
    con = get_chembl_con(cfg)
    load_sifts_into_chembl(cfg, con)
    rec_prefixes, lig_prefixes = load_crossdocked_prefixes(cfg)
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)




