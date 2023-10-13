
import os
import subprocess

import numpy as np
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets

from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config
from utils.task import task
from utils.workflow import Workflow

def load_banana_scores(out_file):
    ret = []
    with open(out_file, "r") as f:
        for line in f:
            ret.append(float(line))
    return np.array(ret)

def run_banana(cfg, split, pocket):

    folder = get_bayesbind_dir(cfg) + f"/{split}/{pocket}"
    out_folder = get_baseline_dir(cfg, "banana", split, pocket)
    ret = {}

    pocket_file = folder + "/pocket.pdb"
    for name in [ "random", "actives" ]:
        smi_fname = folder + f"/{name}.smi"
        split, pocket = folder.split("/")[-2:]
        out_fname = os.path.abspath(f"{out_folder}/{name}.txt")

        if True: # not os.path.exists(out_fname):

            banana_cmd = f"cd {cfg.banana.dir} && conda run --no-capture-output -n {cfg.banana.env} python inference.py {smi_fname} {pocket_file} --out_file {out_fname}"
            # if not cfg.use_gpu:
            #     banana_cmd += " --no_gpu"

            print(f"Running:")
            print(banana_cmd)

            subprocess.run(banana_cmd, shell=True)

        ret[name] = load_banana_scores(out_fname)

    return ret

@task()
def run_all_banana(cfg):
    ret = {}
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        ret[pocket] = run_banana(cfg, split, pocket)
    return ret

def make_banana_workflow(cfg):
    return Workflow(cfg, run_all_banana())

if __name__ == "__main__":
    cfg = get_config("local")
    make_banana_workflow(cfg).run()