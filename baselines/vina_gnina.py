
import os
import subprocess
import pandas as pd
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
from meeko import MoleculePreparation
from tqdm import tqdm

from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config, get_output_dir
from utils.task import iter_task, simple_task, task
from utils.workflow import Workflow

def get_cache_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "local", "vina_cache")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_all_bayesbind_splits_and_pockets(cfg):
    """ Returns a list of (split, pocket) tuples """
    ret = []
    for split in ["val", "test"]:
        for pocket in os.listdir(get_bayesbind_dir(cfg) + f"/{split}"):
            ret.append((split, pocket))
    return ret

def move_lig_to_center(lig, center):
    """ Ensure the ligand's conformer centroid is at center """
    center_pt = rdkit.Geometry.rdGeometry.Point3D(*center)
    conf = lig.GetConformer()
    lig_center = rdMolTransforms.ComputeCentroid(conf)
    for i in range(lig.GetNumAtoms()):
        og_pos = conf.GetAtomPosition(i)
        new_pos = og_pos + center_pt - lig_center
        conf.SetAtomPosition(i, new_pos)

def get_lig_size(lig, padding=3):
    """ Returns center and size of the molecule conformer """
    bounds = ComputeConfBox(lig.GetConformer(0))
    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return tuple(center), tuple(size)

@task()
def prepare_all_rec_pdbqts(cfg):
    """ Uses OpenBabel to prepare all receptor PDBQT files """
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        in_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/rec.pdb"
        out_file = get_baseline_dir(cfg, "vina", split, pocket) + "/rec.pdbqt"
        cmd = f"obabel -xr -ipdb {in_file} -opdbqt -O{out_file}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def prepare_lig_pdbqt(lig_file, center, size):
    """ Prepares a ligand PDBQT file. Returns a tuple
    lig_file, box_size """

    out_file = get_cache_dir(cfg) + "/" + lig_file.split("/")[-1].replace(".sdf", ".pdbqt")
    lig = next(Chem.SDMolSupplier(lig_file))
    lig = Chem.AddHs(lig, addCoords=True)

    move_lig_to_center(lig, center)
    _, lig_size = get_lig_size(lig)
    size = max(lig_size, size)

    preparator = MoleculePreparation(hydrate=False)
    preparator.prepare(lig)
    preparator.write_pdbqt_file(out_file)

    return out_file, size

def run_program(cfg, program, split, pocket, row, out_file):
    """ Run either Vina or Gnina on a single ligand. Program
    is either 'vina' or 'gnina'. """

    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)
    lig_file = get_output_dir(cfg) + "/" + row.lig_file

    if program == "vina":
        rec_file = get_baseline_dir(cfg, "vina", split, pocket) + "/rec.pdbqt"
        lig_file, size = prepare_lig_pdbqt(lig_file, center, size)
    else:
        rec_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/rec.pdb"

    cmd = [ program, "--receptor", rec_file, "--ligand", lig_file, "--cpu", "1" ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    print("Docking with:", " ".join(cmd))
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print(out.decode("utf-8"))
    print(err.decode("utf-8"))

    return out_file

def prepare_docking_inputs(cfg, rec_pdbqts, program):
    """ Prepares all inputs for docking. Returns a list of
    (split, pocket, row, out_file) tuples """

    ext = "sdf" if program == "gnina" else "pdbqt"

    ret = []
    for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
        for prefix in [ "actives", "random" ]:
            csv = prefix + ".csv"
            df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{csv}")
            for i, row in df.iterrows():
                out_file = get_baseline_dir(cfg, program, split, pocket) + f"/{prefix}_{i}.{ext}"
                ret.append((split, pocket, row, out_file))

    return ret

@task(max_runtime=0.1)
def prepare_vina_inputs(cfg, rec_pdbqts):
    return prepare_docking_inputs(cfg, rec_pdbqts, "vina")

def run_vina(cfg, args):
    return run_program(cfg, "vina", *args)

run_all_vina = iter_task(224, 48, n_cpu=1, mem=128)(run_vina)

def make_vina_workflow(cfg):

    rec_pdbqts = prepare_all_rec_pdbqts()
    vina_inputs = prepare_vina_inputs(rec_pdbqts)
    vina_outputs = run_all_vina(vina_inputs)

    return Workflow(cfg, vina_outputs)

if __name__ == "__main__":
    cfg = get_config("local")
    make_vina_workflow(cfg).run()