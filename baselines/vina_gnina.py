
import os
import random
import subprocess
from traceback import print_exc
import pandas as pd
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
from meeko import MoleculePreparation, PDBQTMolecule
from tqdm import tqdm, trange
from rdkit.Chem import PandasTools
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  

from utils.cfg_utils import get_baseline_dir, get_baseline_struct_dir, get_bayesbind_dir, get_bayesbind_struct_dir, get_config, get_output_dir
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

def get_all_bayesbind_struct_splits_and_pockets(cfg):
    """ Returns a list of (split, pocket) tuples """
    ret = []
    for split in ["val", "test"]:
        for pocket in os.listdir(get_bayesbind_struct_dir(cfg) + f"/{split}"):
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

@task(force=False)
def prepare_all_rec_pdbqts(cfg):
    """ Uses OpenBabel to prepare all receptor PDBQT files """
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        in_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/rec.pdb"
        out_file = get_baseline_dir(cfg, "vina", split, pocket) + "/rec.pdbqt"
        cmd = f"obabel -xr -ipdb {in_file} -opdbqt -O{out_file} -p 7"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def prepare_lig_pdbqt(cfg, lig_file, center, size):
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

TIMEOUT = None # 60*15
VINA_GNINA_CPUS = 1
def run_program(cfg, program, split, pocket, row, out_file, cpus=VINA_GNINA_CPUS):
    """ Run either Vina or Gnina on a single ligand. Program
    is either 'vina' or 'gnina'. """

    if os.path.exists(out_file):
        return out_file

    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)
    lig_file = get_output_dir(cfg) + "/" + row.lig_file

    if program == "vina":
        rec_file = get_baseline_dir(cfg, "vina", split, pocket) + "/rec.pdbqt"
        lig_file, size = prepare_lig_pdbqt(cfg, lig_file, center, size)
    else:
        rec_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/rec.pdb"

    cmd = [ program, "--receptor", rec_file, "--ligand", lig_file, "--cpu", str(cpus) ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    if program == "gnina":
        cmd += [ "--cnn", "crossdock_default2018" ]

    print("Docking with:", " ".join(cmd))
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(timeout=TIMEOUT)
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
            df = df.iloc[:cfg.baseline_max_ligands]
            for i, row in df.iterrows():
                out_file = get_baseline_dir(cfg, program, split, pocket) + f"/{prefix}_{i}.{ext}"
                ret.append((split, pocket, row, out_file))

    random.shuffle(ret)
    return ret

@task(max_runtime=0.1, force=False)
def prepare_vina_inputs(cfg, rec_pdbqts):
    return prepare_docking_inputs(cfg, rec_pdbqts, "vina")

def run_vina(cfg, args):
    try:
        return run_program(cfg, "vina", *args)
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        return None

run_all_vina = iter_task(600, 5*600*24, n_cpu=1, mem=4, force=True)(run_vina)

@task(max_runtime=0.1)
def prepare_gnina_inputs(cfg):
    return prepare_docking_inputs(cfg, None, "gnina")

def run_gnina(cfg, args):
    try:
        return run_program(cfg, "gnina", *args)
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        return None

run_all_gnina = iter_task(600, 10*600*24, n_cpu=1, mem=4, force=True)(run_gnina)

def make_vina_gnina_workflow(cfg):

    rec_pdbqts = prepare_all_rec_pdbqts()
    vina_inputs = prepare_vina_inputs(rec_pdbqts)
    vina_outputs = run_all_vina(vina_inputs)

    gnina_inputs = prepare_gnina_inputs()
    gnina_outputs = run_all_gnina(gnina_inputs)

    return Workflow(cfg, gnina_outputs)

def get_gnina_preds(cfg, split, pocket, prefix):
    """ Returns a numpy array of the gnina
    predictions for a given split, pocket, and prefix
    (either actives or random) """
    scores = []
    df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv")
    for i in range(len(df)):
        if i >= cfg.baseline_max_ligands:
            break
        fname = get_baseline_dir(cfg, "gnina", split, pocket) + f"/{prefix}_{i}.sdf"
        if os.path.exists(fname):
            sd_df = PandasTools.LoadSDF(fname, strictParsing=False, molColName=None)
            try:
                scores.append(sd_df.CNNaffinity[0])
            except AttributeError:
                print(f"Error loading {fname}")
                scores.append(-1000)
        else:
            scores.append(-1000)
    return np.asarray(scores)

def get_docked_scores_from_pdbqt(fname):
    return PDBQTMolecule.from_file(fname)._pose_data["free_energies"]

def get_vina_preds(cfg, split, pocket, prefix):
    """ Returns a numpy array of the vina
    predictions for a given split, pocket, and prefix
    (either actives or random) """
    scores = []
    df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv")
    for i in range(len(df)):
        if i >= cfg.baseline_max_ligands:
            break
        fname = get_baseline_dir(cfg, "vina", split, pocket) + f"/{prefix}_{i}.pdbqt"
        if os.path.exists(fname):
            cur_scores = get_docked_scores_from_pdbqt(fname)
            scores.append(-cur_scores[0])
        else:
            scores.append(-1000)
    return np.asarray(scores)

def collate_all_results(cfg):
    """ Collates the results of gnina + vina, writing all the actives 
    results for a given pocket to actives_results/results.csv 
    and the same for random  """
    
    for program in [ "vina", "gnina" ]:
        for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
            for prefix in [ "actives", "random" ]:
                csv = prefix + ".csv"
                df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{csv}")
                new_rows = []
                for i, row in df.iterrows():
                    if i >= cfg.baseline_max_ligands:
                        break
                    docked_fname = f"/{prefix}_{i}.sdf"
                    new_rows.append({
                        "smiles": row.lig_smiles,
                        "filaname": docked_fname,
                    })
                out_df = pd.DataFrame(new_rows)
                if program == "vina":
                    out_df["score"] = get_vina_preds(cfg, split, pocket, prefix)
                else:
                    out_df["score"] = get_gnina_preds(cfg, split, pocket, prefix)
                
                out_folder = get_baseline_dir(cfg, program, split, pocket) + f"/{prefix}_results"
                os.makedirs(out_folder, exist_ok=True)

                out_df.to_csv(out_folder + "/results.csv", index=False)

# def postproc_gnina(cfg):
#     """ Re-indexes gnina predictions according to valid_indexes """
#     for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
#         folder = get_baseline_dir(cfg, "gnina", split, pocket)
#         for prefix in [ "actives", "random" ]:
#             csv = prefix + ".csv"
#             df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{csv}")
#             if prefix == "actives":
#                 valid_indexes = df.query("standard_type != 'Potency'").index            
#             else:
#                 valid_indexes = df.index
#             with open(folder + f"/{prefix}.txt", "w") as f:
#                 for i, index in enumerate(valid_indexes):
#                     docked_fname = f"{prefix}_{index}.sdf"
#                     if os.path.exists(folder + "/" + docked_fname):
#                         f.write(docked_fname + "\n")
#                     else:
#                         f.write("\n")

def run_gnina_on_bayesbind_struct(cfg):
    for split, pocket in get_all_bayesbind_struct_splits_and_pockets(cfg):
        folder = get_bayesbind_struct_dir(cfg) + f"/{split}/{pocket}"
        out_folder = get_baseline_struct_dir(cfg, "gnina", split, pocket)
        df = pd.read_csv(folder + "/actives.csv")

        gnina_min_aff = []
        gnina_cnn_aff = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            rec_file = folder + "/" + row.redock_rec_file
            lig_file = folder + "/" + row.lig_crystal_file
            out_file = out_folder + f"/{i}.sdf"

            cmd = f"gnina --receptor {rec_file} --ligand {lig_file} --minimize --cnn crossdock_default2018 --out {out_file}"
            subprocess.run(cmd, shell=True, check=True)

            lig = Chem.SDMolSupplier(out_file)[0]
            gnina_min_aff.append(lig.GetProp("minimizedAffinity"))
            gnina_cnn_aff.append(lig.GetProp("CNNaffinity"))

        df["gnina_min_affinity"] = gnina_min_aff
        df["gnina_cnn_affinity"] = gnina_cnn_aff
        df.to_csv(out_folder + "/actives.csv", index=False)

if __name__ == "__main__":
    cfg = get_config("local")
    # run_gnina_on_bayesbind_struct(cfg)
    collate_all_results(cfg)
