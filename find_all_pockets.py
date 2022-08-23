import sys
import os
import yaml
import subprocess

from glob import glob
from tqdm import tqdm

from traceback import print_exc
from multiprocessing import Pool

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

if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    find_all_pockets(cfg)
