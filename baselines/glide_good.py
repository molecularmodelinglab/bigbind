# This is supposed to be cleaner, more maintainable code to use Glide

from collections import defaultdict
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config


def clean_glide_outputs(smi_file, docked_mae, out_folder):
    """ Cleans up the glide output mae file, extracting the scores
    to a new csv file with the (minimum) glide score per compound to
    {output_folder}/results.csv and saves the numbered sdf files to the
    folder as well. E.g. 1.sdf, 2.sdf, etc.
    
    :param smi_file: smi file we gave to glide for input
    :param docked_mae: mae file output by glide
    :param out_folder: folder to put the cleaned outputs in
    
    """

    os.makedirs(out_folder, exist_ok=True)

    # first convert the mae file to a big sdf file in out_folder
    big_sdf = out_folder + "/all_docked.sdf"
    cmd = f"$SCHRODINGER/utilities/structconvert {docked_mae} {big_sdf}"

    subprocess.run(cmd, shell=True, check=True)

    df = PandasTools.LoadSDF(big_sdf)

    # load in the smiles
    og_smiles = []
    with open(smi_file, "r") as f:
        for line in f:
            og_smiles.append(line.strip())

    scores = np.zeros(len(og_smiles)) - 10000
    filenames = [None for _ in range(len(og_smiles))]

    # save individual sdf files
    scores_and_mols = defaultdict(list)

    for i, row in df.iterrows():
        # first row is the rec structure
        if i == 0: continue
        mol = row.ROMol
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

def clean_all_glide_results(cfg):
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        smi_file = get_bayesbind_dir(cfg) + f"/{split}/{pocket}/random.smi"
        result_folder = get_baseline_dir(cfg, "glide", split, pocket) + "/random_results"
        docked_csv =  result_folder + "/dock_random.csv"
        docked_mae = result_folder + "/dock_random_pv.maegz"
        out_folder = result_folder
        clean_glide_outputs(smi_file, docked_mae, out_folder)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    clean_all_glide_results(cfg)