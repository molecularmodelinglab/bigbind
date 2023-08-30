import subprocess
import os
from glob import glob
from workflow import Workflow
from task import file_task
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

def make_bigbind_workflow():

    sifts_zipped = download_sifts()
    sifts_csv = unzip_sifts(sifts_zipped)

    crossdocked_tarred = download_crossdocked()
    cd_dir = untar_crossdocked(crossdocked_tarred)

    chembl_tarred = download_chembl()
    chembl_db_file = untar_chembl(chembl_tarred)

    return Workflow(sifts_csv, cd_dir, chembl_db_file)

if __name__ == "__main__":
    from cfg_utils import get_config
    workflow = make_bigbind_workflow()
    cfg = get_config("local")

    cd_nodes = workflow.out_nodes # find_nodes("untar_crossdocked")
    levels = workflow.get_levels(cd_nodes)
    print(levels)