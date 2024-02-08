# This is supposed to be cleaner, more maintainable code to use Glide

from collections import defaultdict
from glob import glob
from multiprocessing import Pool
import os
import shutil
import subprocess
import sys
from exceptiongroup import print_exc

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
from baselines.minimize_struct import minimize_protein

from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets, get_all_bayesbind_struct_splits_and_pockets
from utils.cfg_utils import get_baseline_dir, get_baseline_struct_dir, get_bayesbind_dir, get_bayesbind_struct_dir, get_config


def clean_glide_outputs(smi_file, docked_mae, out_folder, max_ligands=None):
    """ Cleans up the glide output mae file, extracting the scores
    to a new csv file with the (minimum) glide score per compound to
    {output_folder}/results.csv and saves the numbered sdf files to the
    folder as well. E.g. 1.sdf, 2.sdf, etc.
    
    :param smi_file: smi file we gave to glide for input. This can
        also be a list of smiles
    :param docked_mae: mae file output by glide
    :param out_folder: folder to put the cleaned outputs in
    :param max_ligands: The number of ligands we actually docked 
    (sometimes less than the total number)

    """

    os.makedirs(out_folder, exist_ok=True)

    # first convert the mae file to a big sdf file in out_folder
    big_sdf = out_folder + "/all_docked.sdf"
    cmd = f"$SCHRODINGER/utilities/structconvert {docked_mae} {big_sdf}"

    subprocess.run(cmd, shell=True, check=True)

    df = PandasTools.LoadSDF(big_sdf, removeHs=False)

    # load in the smiles
    if isinstance(smi_file, list):
        og_smiles = smi_file
    else:
        og_smiles = []
        with open(smi_file, "r") as f:
            for line in f:
                og_smiles.append(line.strip())
        if max_ligands is not None:
            og_smiles = og_smiles[:max_ligands]

    scores = np.zeros(len(og_smiles)) - 10000
    filenames = [None for _ in range(len(og_smiles))]

    # save individual sdf files
    scores_and_mols = defaultdict(list)
    for i, row in df.iterrows():
        # first row is the rec structure
        if i == 0: continue
        mol = row.ROMol

        # sometimes I'm docking things with only one ligand and without
        # any lig prep...
        if len(og_smiles) == 1:
            og_index = 0
        else:
            og_index = int(row.s_lp_Variant.split(":")[-1].split("-")[0]) - 1

        filename = f"{og_index}.sdf"
        filenames[og_index] = filename

        scores[og_index] = -float(row.r_i_docking_score)

        scores_and_mols[og_index].append((scores[og_index], mol))

    # order according to decreasing score
    for index, arr in scores_and_mols.items():
        filename = filenames[index]
        full_filename = out_folder + "/" + filename
        arr = list(sorted(arr, key=lambda x: -x[0]))

        writer = Chem.SDWriter(full_filename)
        for score, mol in arr:
            writer.write(mol)
        writer.close()

    # save results df
    out_df = pd.DataFrame({"smiles": og_smiles, "filename": filenames, "score": scores})
    out_df.to_csv(out_folder + "/results.csv", index=False)

def rec_to_mae(rec_file, out_file):
    if os.path.exists(out_file):
        return
    cmd = f"$SCHRODINGER/utilities/structconvert {rec_file} {out_file}"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)#, stderr=subprocess.DEVNULL)

def lig_to_mae(lig_file, out_file):
    if os.path.exists(out_file):
        return
    cmd = f"$SCHRODINGER/utilities/structconvert {lig_file} {out_file}"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)#, stderr=subprocess.DEVNULL)

def create_gridfile(row, rec_mae, gridfile):
    if os.path.exists(gridfile):
        return

    try:
        grid_in_file = gridfile.replace(".zip", ".in")
        # inner box is X times the size of outer box
        inner_scale = 0.5

        with open(grid_in_file, "w") as f:
            f.write(
    f"""INNERBOX {int(row.pocket_size_x*inner_scale)}, {int(row.pocket_size_y*inner_scale)},{int(row.pocket_size_z*inner_scale)}
    ACTXRANGE {row.pocket_size_x}
    ACTYRANGE {row.pocket_size_y}
    ACTZRANGE {row.pocket_size_z}
    OUTERBOX {row.pocket_size_x}, {row.pocket_size_y}, {row.pocket_size_z}
    GRID_CENTER {row.pocket_center_x}, {row.pocket_center_y}, {row.pocket_center_z}
    GRIDFILE {gridfile}
    RECEP_FILE {rec_mae}
    """
                        )
                    
        cmd = f"glide {grid_in_file}  -NOJOBID"
        print(f"Running {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        return True

    except subprocess.CalledProcessError:
        print(f"Failed to create gridfile for {gridfile}")
        return False

    finally:
        if os.path.exists(grid_in_file):
            os.remove(grid_in_file)

def glide_dock(gridfile, lig_mae, method="rigid"):
    """ Use 'mininplace' to only minimize + score"""

    try:

        dock_in_file = gridfile.replace(".zip", "_dock.in")

        with open(dock_in_file, "w") as f:
            f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_mae}
DOCKING_METHOD {method}
"""
            )

        output_folder = os.path.dirname(gridfile)
        os.chdir(output_folder)

        cmd = f"glide {dock_in_file} -NOJOBID -OVERWRITE"
        print(f"Running {cmd} from {os.path.abspath('.')}")
        subprocess.run(cmd, shell=True, check=True)

    except subprocess.CalledProcessError:
        print(f"Failed to dock {dock_in_file}")
        return False

    finally:
        if os.path.exists(dock_in_file):
            os.remove(dock_in_file)


MAX_LIGANDS = 1000
def clean_all_glide_results(cfg):
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        smi_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/random.smi"
        result_folder = get_baseline_dir(cfg, "glide", split, pocket) + "/random_results"
        docked_mae = result_folder + "/dock_random_pv.maegz"
        out_folder = result_folder
        clean_glide_outputs(smi_file, docked_mae, out_folder, MAX_LIGANDS)

def convert_glide_min_recs(cfg):
    split = "val"
    pocket = 'TTK_HUMAN_516_808_0'
    result_folder = get_baseline_dir(cfg, "glide_min", split, pocket) + "/random_results"
    for rec_file in tqdm(glob(result_folder + "/*.pdb")):
        out_file = rec_file.replace(".pdb", ".mae")
        rec_to_mae(rec_file, out_file)

def convert_glide_min_ligs(cfg):
    split = "val"
    pocket = 'TTK_HUMAN_516_808_0'
    prefix = "random"
    glide_folder = get_baseline_dir(cfg, "glide", split, pocket) + f"/{prefix}_results"
    result_folder = get_baseline_dir(cfg, "glide_min", split, pocket) + f"/{prefix}_results"
    
    inputs = []
    for lig_file in glob(glide_folder + "/*.sdf"):
        try:
            index = int(lig_file.split("/")[-1].split(".")[0])
        except ValueError: # it's "all_docked.sdf"
            continue
        out_file = result_folder + f"/lig_{index}_min.mae"
        inputs.append((lig_file, out_file))

    with Pool(12) as p:
        p.starmap(lig_to_mae, tqdm(inputs))

def create_all_gridfiles(cfg):
    split = "val"
    pocket = 'TTK_HUMAN_516_808_0'
    prefix = "random"
    csv_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv"
    result_folder = get_baseline_dir(cfg, "glide_min", split, pocket) + f"/{prefix}_results"

    df = pd.read_csv(csv_file)
    df = df.iloc[:MAX_LIGANDS]
    inputs = []
    for i, row in df.iterrows():
        rec_mae = result_folder + f"/rec_{i}_min.mae"
        if not os.path.exists(rec_mae):
            continue
        gridfile = result_folder + f"/rec_{i}_grid.zip"
        inputs.append((row, rec_mae, gridfile))

    with Pool(12) as p:
        p.starmap(create_gridfile, tqdm(inputs)) 


def dock_all_minimized(cfg):
    split = "val"
    pocket = 'TTK_HUMAN_516_808_0'
    prefix = "random"
    csv_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv"
    result_folder = get_baseline_dir(cfg, "glide_min", split, pocket) + f"/{prefix}_results"

    df = pd.read_csv(csv_file)
    df = df.iloc[:MAX_LIGANDS]
    inputs = []
    for i, row in df.iterrows():
        lig_mae = result_folder + f"/lig_{i}_min.mae"
        if not os.path.exists(lig_mae):
            continue
        gridfile = result_folder + f"/rec_{i}_grid.zip"
        if not os.path.exists(gridfile):
            continue
        inputs.append((gridfile, lig_mae, "mininplace"))

    with Pool(12) as p:
        p.starmap(glide_dock, tqdm(inputs)) 

def minimize_all_glide_structs(cfg):
    prefix = "random"
    split = "val"
    pocket = "TTK_HUMAN_516_808_0"

    out_folder = get_baseline_dir(cfg, "glide_min", split, pocket) + f"/{prefix}_results"
    os.makedirs(out_folder, exist_ok=True)

    rec_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/rec_hs.pdb"

    glide_folder = get_baseline_dir(cfg, "glide", split, pocket) + f"/{prefix}_results"
    for fname in tqdm(glob(glide_folder + "/*.sdf")):
        try:
            index = int(fname.split("/")[-1].split(".")[0])
        except ValueError: # it's "all_docked.sdf"
            continue
        out_file = out_folder + f"/rec_{index}_min.pdb"
        try:
            minimize_protein(rec_file, fname, out_file, False, force=True)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            continue

def make_glide_good_struct(cfg):
    """Coalesce the glide results into a much nice form factor """

    for split, pocket in get_all_bayesbind_struct_splits_and_pockets(cfg):
        bb_struct_dir = f"{get_bayesbind_struct_dir(cfg)}/{split}/{pocket}"
        bb_dir = f"{get_bayesbind_dir(cfg)}/{split}/{pocket}"
        glide_dir =f"{get_baseline_struct_dir(cfg, 'glide', split, pocket)}"
        glide_nostruct_dir =f"{get_baseline_dir(cfg, 'glide', split, pocket)}"
        glide_good_dir = f"{get_baseline_struct_dir(cfg, 'glide_good', split, pocket)}"

        act_df = pd.read_csv(f"{bb_struct_dir}/actives.csv")
        random_df = pd.read_csv(f"{bb_dir}/random.csv")

        # move the rec files over
        for fname in ["pocket.pdb", "rec.pdb", "rec_hs.pdb"]:
            shutil.copyfile(f"{bb_struct_dir}/{fname}", f"{glide_good_dir}/{fname}")

        mols = []
        scores = []

        for smi, pdb in zip(tqdm(act_df.lig_smiles), act_df.pdb):
            pdb_dir = f"{glide_dir}/{pdb}_crossdock"

            out_dir = f"{pdb_dir}/cleaned"
            docked_mae = f"{pdb_dir}/dock_{pdb}_crossdock_pv.maegz"
            if os.path.exists(docked_mae):
                clean_glide_outputs([smi], docked_mae, out_dir, MAX_LIGANDS)
                
                score_df = pd.read_csv(f"{out_dir}/results.csv")
                docked_sdf = f"{out_dir}/0.sdf"
                mols.append(Chem.SDMolSupplier(docked_sdf, removeHs=False)[0])
                scores.append(score_df.score[0])
            else:
                scores.append(-1000)

        out_active_sdf = f"{glide_good_dir}/actives.sdf"
        writer = Chem.SDWriter(out_active_sdf)
        for mol in mols:
            writer.write(mol)
        writer.close()

        act_df['score'] = scores
        act_df.to_csv(f"{glide_good_dir}/actives.csv", index=False)

        # now for random -- easier
        scores = pd.read_csv(f"{glide_nostruct_dir}/random_results/results.csv").score
        mols = []
        for i in range(len(scores)):
            docked_fname = f"{glide_nostruct_dir}/random_results/{i}.sdf"
            if os.path.exists(docked_fname):
                mols.append(Chem.SDMolSupplier(docked_fname, removeHs=False)[0])
        
        out_random_sdf = f"{glide_good_dir}/random.sdf"
        writer = Chem.SDWriter(out_random_sdf)
        for mol in mols:
            writer.write(mol)

        random_df['score'] = scores
        random_df.to_csv(f"{glide_good_dir}/random.csv", index=False)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    # clean_all_glide_results(cfg)
    make_glide_good_struct(cfg)
    # convert_glide_min_ligs(cfg)
    # minimize_all_glide_structs(cfg)
    # convert_glide_min_recs(cfg)
    # create_all_gridfiles(cfg)
    # dock_all_minimized(cfg)