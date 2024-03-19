from glob import glob
import subprocess
import sys
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import os
import shutil
import random

from tqdm import tqdm
from bigbind.similarity import LigSimilarity

from utils.cfg_utils import get_bayesbind_dir, get_config, get_final_bayesbind_dir, get_output_dir
from utils.task import task

SEED = 42
np.random.seed(SEED)

def save_smiles(smi_fname, smiles_list):
    with open(smi_fname, "w") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

def load_smiles(smi_fname):
    ret = []
    with open(smi_fname, "r") as f:
        for line in f:
            ret.append(line.strip())
    return ret

def make_bayesbind_dir(cfg, lig_sim, split, both_df, poc_df, pocket, num_random):
    folder = get_bayesbind_dir(cfg) + f"/{split}/{pocket}"
    os.makedirs(folder, exist_ok=True)

    rec_cluster = poc_df.rec_cluster[0]

    # poc_df["alpha"] = cd.get_alphas(poc_df.lig_smiles)

    for key in [ "ex_rec_file", "ex_rec_pdb", "ex_rec_pocket_file", "num_pocket_residues",
                 "pocket_center_x", "pocket_center_y", "pocket_center_z", 
                 "pocket_size_x", "pocket_size_y", "pocket_size_z" ]:
        poc_df[key] = poc_df[key][0]

    # poc_df.to_csv(folder + "/activities.csv", index=False)

    # take the molecule with median activity from each cluster
    clusters = poc_df.lig_cluster.unique()
    clustered_rows = []
    for cluster in tqdm(clusters):
        cluster_df = poc_df.query("lig_cluster == @cluster").reset_index(drop=True)
        median_idx = cluster_df.pchembl_value.sort_values().index[len(cluster_df) // 2]
        clustered_rows.append(cluster_df.loc[median_idx])
        # print(cluster_df.pchembl_value.sort_values(), cluster_df.loc[median_idx])

    clustered_df = pd.DataFrame(clustered_rows)
    clustered_df.to_csv(folder + "/actives.csv", index=False)

    save_smiles(folder + "/actives.smi", clustered_df.lig_smiles)

    # rec_file = get_output_dir(cfg) + "/" + poc_df.ex_rec_file[0]
    # shutil.copyfile(rec_file, folder + "/rec.pdb")

    rec_file_nofix = get_output_dir(cfg) + "/" + poc_df.ex_rec_file[0].replace(".pdb", "_nofix.pdb")
    shutil.copyfile(rec_file_nofix, folder + "/rec_nofix.pdb")

    # todo: put this in the main bigbind code!
    pdb_fix_cmd = f"pdbfixer {folder}/rec_nofix.pdb --output {folder}/rec.pdb --add-atoms=heavy --add-residues"
    print(f"Running: {pdb_fix_cmd}")
    subprocess.run(pdb_fix_cmd, shell=True, check=True)

    poc_file = get_output_dir(cfg) + "/" + poc_df.ex_rec_pocket_file[0]
    shutil.copyfile(poc_file, folder + "/pocket.pdb")

    rand_df = both_df.query("rec_cluster != @rec_cluster")
    other_smiles = rand_df.lig_smiles.unique()
    other_smiles_to_lf = { smi: lf for smi, lf in zip(rand_df.lig_smiles, rand_df.lig_file) }

    random.shuffle(other_smiles)
    X_rand = lig_sim.make_diverse_set(other_smiles, num_random)
    X_lf_rand = [ other_smiles_to_lf[smi] for smi in X_rand ]

    save_smiles(folder + "/random.smi", X_rand)

    X_rand_df = pd.DataFrame({ "lig_file": X_lf_rand, "lig_smiles": X_rand })
    for key in [ "ex_rec_file", "ex_rec_pdb", "ex_rec_pocket_file", "num_pocket_residues",
                 "pocket_center_x", "pocket_center_y", "pocket_center_z", 
                 "pocket_size_x", "pocket_size_y", "pocket_size_z" ]:
        X_rand_df[key] = poc_df[key][0]

    X_rand_df.to_csv(folder + "/random.csv", index=False)    

def make_bayesbind_split(cfg, lig_sim, split, df, both_df, poc_clusters, act_cutoff=6, cluster_cutoff=150):
    """ Makes benchmarks for all pockets with at least num_cutoff
    activities below act_cutoff. Num_random is the number of compounds
    to randomly sample from ChEMBL for each benchmark """
    
    poc2cluster = {}
    for cluster in poc_clusters:
        for poc in cluster:
            poc2cluster[poc] = cluster

    # the TOP1 pocket is known to be incorrect -- skip it
    # MCL and OPRK have too similar pockets in the train set, that neither TM score
    # nor ProBis can identify smh
    bad_pockets = [ "TOP1_HUMAN_202_765_0", "OPRK_HUMAN_55_347_TM_0", "MCL1_HUMAN_171_326_0" ]

    # print(split, len(df))

    # only take a single pocket per cluster
    seen = set()

    for pocket, num in df.query("pchembl_value < @act_cutoff").pocket.value_counts().items():
        if pocket not in seen and pocket not in bad_pockets and num >= cluster_cutoff:

            poc_df = df.query("pocket == @pocket").reset_index(drop=True)
            low_clusters = len(poc_df.query("pchembl_value < @act_cutoff").lig_cluster.unique())
            tot_clusters = len(poc_df.lig_cluster.unique())
            if low_clusters < cluster_cutoff:
                continue
            
            print(f"Running on {split}/{pocket} {low_clusters=} {tot_clusters=}")
            make_bayesbind_dir(cfg, lig_sim, split, both_df, poc_df, pocket, num_random=10000)

            for poc in poc2cluster[pocket]:
                seen.add(poc)

# force this!
@task(force=True)
def make_all_bayesbind(cfg, saved_act, lig_smi, lig_sim_mat, poc_clusters):
    shutil.rmtree(get_bayesbind_dir(cfg))

    print("Saving BayesBind set to {}".format(get_bayesbind_dir(cfg)))

    lig_sim = LigSimilarity(lig_smi, lig_sim_mat)

    val_df = pd.read_csv(get_output_dir(cfg) + f"/activities_val.csv")
    test_df = pd.read_csv(get_output_dir(cfg) + f"/activities_test.csv")
    both_df = pd.concat([val_df, test_df])

    for split, df in [ ("val", val_df), ("test", test_df) ]:
        make_bayesbind_split(cfg, lig_sim, split, df, both_df, poc_clusters)

# if __name__ == "__main__":
#     cfg = OmegaConf.load("cfg.yaml")
#     cd = ChemicalDiversity(cfg)

#     val_df = pd.read_csv(get_output_dir(cfg) + f"/activities_val.csv")
#     test_df = pd.read_csv(get_output_dir(cfg) + f"/activities_test.csv")
#     both_df = pd.concat([val_df, test_df])

#     for split in ["val", "test"]:
#         make_all_bayesbind(cfg, cd, split, both_df)

def postproc_bayesbind(cfg):
    from baselines.evaluate import get_all_valid_indexes
    valid_indexes = get_all_valid_indexes(cfg)
    for in_folder in glob(get_bayesbind_dir(cfg) + "/*/*"):
        split, poc = in_folder.split("/")[-2:]
        try:
            index = valid_indexes[poc]
        except KeyError:
            continue

        out_folder = os.path.join(get_final_bayesbind_dir(cfg), split, poc)
        os.makedirs(out_folder, exist_ok=True)

        act_df = pd.read_csv(in_folder + "/actives.csv")
        act_df = act_df.loc[index].reset_index(drop=True)

        act_df.to_csv(out_folder + "/actives.csv", index=False)
        save_smiles(out_folder + "/actives.smi", act_df.lig_smiles)

        for fname in ("pocket.pdb", "rec.pdb", "rec_nofix.pdb", "random.csv", "random.smi"):
            shutil.copyfile(os.path.join(in_folder, fname), os.path.join(out_folder, fname))

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    postproc_bayesbind(cfg)