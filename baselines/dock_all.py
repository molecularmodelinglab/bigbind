
import random
import subprocess
import sys
from traceback import print_exc
import pandas as pd
from tqdm import tqdm
from utils.cfg_utils import get_config, get_docked_dir, get_output_dir
from utils.task import iter_task, task
from utils.workflow import Workflow

TIMEOUT = 60*10
VINA_GNINA_CPUS = 1
def run_full_gnina(cfg, args):
    """ Run either Vina or Gnina on a single ligand. Program
    is either 'vina' or 'gnina'. """

    try:

        index, split, cx, cy, cz, sx, sy, sz, lf, rf = args

        out_file = get_docked_dir(cfg, "gnina", split) + f"/{index}.sdf"

        # if os.path.exists(out_file):
        #     return out_file

        center = (cx, cy, cz)
        size = (sx, sy, sz)
        lig_file = get_output_dir(cfg) + "/" + lf

        rec_file = get_output_dir(cfg) + f"/{rf.replace('.pdb', '_nofix.pdb')}"

        cmd = [ "gnina", "--receptor", rec_file, "--ligand", lig_file, "--cpu", str(VINA_GNINA_CPUS) ]
        for c, s, ax in zip(center, size, ["x", "y", "z"]):
            cmd += ["--center_"+ax, str(c)]
            cmd += ["--size_"+ax, str(s)]
        cmd += [ "--out", out_file ]
        cmd += [ "--cnn", "crossdock_default2018" ]

        print("Docking with:", " ".join(cmd))
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(timeout=TIMEOUT)
        print(out.decode("utf-8"))
        print(err.decode("utf-8"))

        return out_file
    
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        return None

@task(max_runtime=0.1, force=False)
def get_single_rec_dfs(cfg):
    """ For each sna split, creates dataframes where each pocket has
    only a single rec file """
    ret = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(get_output_dir(cfg) + f"/activities_sna_1_{split}.csv")

        for pocket in tqdm(df.pocket.unique()):
            poc_df = df.query("pocket == @pocket")
            for key in [ "ex_rec_file", "ex_rec_pdb", "ex_rec_pocket_file", "num_pocket_residues",
                "pocket_center_x", "pocket_center_y", "pocket_center_z", 
                "pocket_size_x", "pocket_size_y", "pocket_size_z" ]:
                df.loc[poc_df.index, key] = poc_df[key].iloc[0]
        
        ret[split] = df
    return ret

@task(max_runtime=0.1, force=False)
def prepare_full_docking_inputs(cfg, split_dfs):
    ret = []
    for split, df in split_dfs.items():
        for i, row in df.iterrows():
            ret.append((i, split, row.pocket_center_x, row.pocket_center_y, row.pocket_center_z, row.pocket_size_x, row.pocket_size_y, row.pocket_size_z, row.lig_file, row.ex_rec_file))

    random.shuffle(ret)
    return ret

run_all_gnina_full = iter_task(600, 600*24*10, n_cpu=1, mem=4)(run_full_gnina)

def make_dock_workflow(cfg):
    split_dfs = get_single_rec_dfs()
    inputs = prepare_full_docking_inputs(split_dfs)
    docked = run_all_gnina_full(inputs)
    return Workflow(cfg, docked)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    make_dock_workflow(cfg).run()