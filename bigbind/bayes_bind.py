import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import os
import shutil
import random
from bigbind.similarity import LigSimilarity

from utils.cfg_utils import get_bayesbind_dir, get_output_dir
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

    poc_df.to_csv(folder + "/activities.csv", index=False)

    save_smiles(folder + "/actives.smi", poc_df.lig_smiles)

    rec_file = get_output_dir(cfg) + "/" + poc_df.ex_rec_file[0]
    shutil.copyfile(rec_file, folder + "/rec.pdb")

    poc_file = get_output_dir(cfg) + "/" + poc_df.ex_rec_pocket_file[0]
    shutil.copyfile(poc_file, folder + "/pocket.pdb")

    other_smiles = both_df.query("rec_cluster != @rec_cluster").lig_smiles.unique()
    random.shuffle(other_smiles)
    X_rand = lig_sim.make_diverse_set(other_smiles, num_random)

    save_smiles(folder + "/random.smi", X_rand)

def make_bayesbind_split(cfg, lig_sim, split, df, both_df, poc_clusters, act_cutoff=6, cluster_cutoff=150):
    """ Makes benchmarks for all pockets with at least num_cutoff
    activities below act_cutoff. Num_random is the number of compounds
    to randomly sample from ChEMBL for each benchmark """
    
    poc2cluster = {}
    for cluster in poc_clusters:
        for poc in cluster:
            poc2cluster[poc] = cluster

    # the TOP1 pocket is known to be incorrect -- skip it
    bad_pockets = [ "TOP1_HUMAN_202_765_0" ]

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
            make_bayesbind_dir(cfg, lig_sim, split, both_df, poc_df, pocket, num_random=20000)

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