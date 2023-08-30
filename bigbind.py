import subprocess
import os
from glob import glob
import sqlite3
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from workflow import Workflow
from task import file_task, simple_task, task
from downloads import StaticDownloadTask

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

    return Workflow(cd_rf2lf)

if __name__ == "__main__":
    from cfg_utils import get_config
    workflow = make_bigbind_workflow()
    cfg = get_config("local")

    workflow.run(cfg)

    # cd_nodes = workflow.out_nodes # find_nodes("untar_crossdocked")
    # levels = workflow.get_levels(cd_nodes)
    # print(levels)