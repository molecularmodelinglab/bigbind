import sys
import os
import shutil
import yaml
import subprocess
import pandas as pd
import numpy as np
import warnings

from glob import glob
from tqdm import tqdm

from traceback import print_exc
from multiprocessing import Pool

from Bio.PDB import get_surface, PDBParser, ShrakeRupley, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def run_fpocket(rec):
    fpocket_output = rec.split(".")[0] + "_out"
    if os.path.exists(fpocket_output): return
    cmd = ["./run_fpocket.sh", rec]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    try:
        proc.check_returncode()
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        
def find_all_pockets(cfg):
    recs = glob(cfg["bigbind_folder"] + "/*/*_rec.pdb")
    print(f"Processing {len(recs)} files")
    with Pool(8) as p:
        list(tqdm(p.imap(run_fpocket, recs), total=len(recs)))

def move_fpocket_output(cfg):
    fpoc_outs = glob(cfg["bigbind_folder"] + "/*/*_out")
    for fpoc in tqdm(fpoc_outs):
        out_folder = cfg["cache_folder"] + "/" + fpoc.split("/")[-1]
        shutil.move(fpoc, out_folder)

def update_structures_csv(cfg):
    struct_file = cfg["bigbind_folder"] + "/structures.csv"
    df = pd.read_csv(struct_file)
    fpocket_loadable = []
    rf_cache = {}
    for rf in tqdm(df["rec_file"]):
        if rf in rf_cache:
            loadable = rf_cache[rf]
        else:
            v2_pocket = cfg["bigbind_folder"] + "/" + rf[:-4] + "_v2.pdb"
            try:
                biopython_parser = PDBParser()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
                    structure = biopython_parser.get_structure('random_id', v2_pocket)
                    rec = structure[0]
                    loadable = True
            except KeyboardInterrupt:
                raise
            except:
                loadable = False
        rf_cache[rf] = loadable
        fpocket_loadable.append(loadable)
    fpocket_loadable = np.array(fpocket_loadable)
    print(f"% loadable: {sum(fpocket_loadable)/len(fpocket_loadable)}")
    df["fpocket_loadable"] = fpocket_loadable
    df.to_csv(struct_file)
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # find_all_pockets(cfg)
    # move_fpocket_output(cfg)
    update_structures_csv(cfg)
