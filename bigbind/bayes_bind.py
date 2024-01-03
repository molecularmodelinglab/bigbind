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
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from bigbind.add_seqres import add_seq_to_pdb
from bigbind.similarity import LigSimilarity

from utils.cfg_utils import get_bayesbind_dir, get_bayesbind_struct_dir, get_config, get_final_bayesbind_dir, get_output_dir
from utils.task import task
from utils.workflow import Workflow

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

    for key in [ "ex_rec_file", "ex_rec_pdb", "ex_rec_pocket_file", "num_pocket_residues",
                 "pocket_center_x", "pocket_center_y", "pocket_center_z", 
                 "pocket_size_x", "pocket_size_y", "pocket_size_z" ]:
        poc_df[key] = poc_df[key][0]

    rec_file = get_output_dir(cfg) + "/" + poc_df.ex_rec_file[0]
    rec_pdb = poc_df.ex_rec_pdb[0]
    add_seq_to_pdb(rec_file, rec_pdb, folder + "/rec.pdb")
    
    # add Hs and residues for MD-ready rec files
    pdb_fix_cmd = f"pdbfixer {folder}/rec.pdb --output {folder}/rec_hs.pdb --add-atoms=all --add-residues"
    print(f"Running: {pdb_fix_cmd}")
    subprocess.run(pdb_fix_cmd, shell=True, check=True)

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
    # pdb_fix_cmd = f"pdbfixer {folder}/rec_nofix.pdb --output {folder}/rec.pdb --add-atoms=heavy --add-residues"
    # print(f"Running: {pdb_fix_cmd}")
    # subprocess.run(pdb_fix_cmd, shell=True, check=True)

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

def make_bayesbind_split(cfg, lig_sim, split, df, both_df, poc_clusters, act_cutoff=5, cluster_cutoff=50):
    """ Makes benchmarks for all pockets with at least num_cutoff
    activities below act_cutoff. Num_random is the number of compounds
    to randomly sample from ChEMBL for each benchmark """
    
    struct_df = pd.read_csv(f"{get_output_dir(cfg)}/structures_{split}.csv")

    # TOP1 has a misannotated pocket
    bad_pockets = [ "TOP1_HUMAN_202_765_0" ]
    # gdi TOPK1 is a PPI
    bad_pockets.append("TOPK_HUMAN_20_320_0")
    # so is TLR8_HUMAN_26_831_0...
    bad_pockets.append("TLR8_HUMAN_26_831_0")

    # sort pockets according to number of ligand clusters with pchembl
    # value > act_cutoff, preferring more clusters in the _struct_ dataframe
    pocket_keys = []
    for pocket in df.pocket.unique():
        spoc_df = struct_df.query("pocket == @pocket")
        poc_df = df.query("pocket == @pocket")
        struct_clusters = len(spoc_df.query("pchembl_value > @act_cutoff").lig_cluster.unique())
        poc_clusters = len(poc_df.query("pchembl_value > @act_cutoff").lig_cluster.unique())
        pocket_keys.append((struct_clusters, poc_clusters, pocket))

    ordered_pockets = [p[2] for p in sorted(pocket_keys, reverse=True)]

    seen_clusters = set()

    for pocket in ordered_pockets:
        if pocket not in bad_pockets:

            poc_df = df.query("pocket == @pocket").reset_index(drop=True)
            poc_cluster = poc_df.rec_cluster[0]
            if poc_cluster in seen_clusters:
                continue

            # for rigorous benchmarking, they should all be
            # the same protein
            if len(poc_df.uniprot.unique()) != 1:
                continue
            
            active_clusters = len(poc_df.query("pchembl_value > @act_cutoff").lig_cluster.unique())
            if active_clusters < cluster_cutoff:
                continue

            seen_clusters.add(poc_cluster)

            tot_clusters = len(poc_df.lig_cluster.unique())
            
            print(f"Running on {split}/{pocket} {tot_clusters=} {active_clusters=} {poc_cluster=}")
            make_bayesbind_dir(cfg, lig_sim, split, both_df, poc_df, pocket, num_random=10000)


# force this!
@task(force=False)
def make_all_bayesbind(cfg, saved_act, lig_smi, lig_sim_mat, poc_clusters):
    # shutil.rmtree(get_bayesbind_dir(cfg))

    print("Saving BayesBind set to {}".format(get_bayesbind_dir(cfg)))

    lig_sim = LigSimilarity(lig_smi, lig_sim_mat)

    val_df = pd.read_csv(get_output_dir(cfg) + f"/activities_val.csv")
    test_df = pd.read_csv(get_output_dir(cfg) + f"/activities_test.csv")
    both_df = pd.concat([val_df, test_df])

    for split, df in [ ("val", val_df), ("test", test_df) ]:
        make_bayesbind_split(cfg, lig_sim, split, df, both_df, poc_clusters)

def make_bayesbind_struct_dir(cfg, split, pocket, struct_df):

    folder = get_bayesbind_struct_dir(cfg) + f"/{split}/{pocket}"
    os.makedirs(folder, exist_ok=True)

    bayesbind_folder = get_bayesbind_dir(cfg) + f"/{split}/{pocket}"

    shutil.copyfile(bayesbind_folder + "/rec.pdb", folder + "/rec.pdb")
    shutil.copyfile(bayesbind_folder + "/rec_hs.pdb", folder + "/rec_hs.pdb")
    shutil.copyfile(bayesbind_folder + "/pocket.pdb", folder + "/pocket.pdb")

    poc_df = struct_df.query(f"pocket == '{pocket}'").reset_index(drop=True)
    poc_df.lig_crystal_file = poc_df.lig_crystal_file.str.split("/").str[-1]
    poc_df.redock_rec_file = poc_df.redock_rec_file.str.split("/").str[-1]
    poc_df.crossdock_rec_file = poc_df.crossdock_rec_file.str.split("/").str[-1]
    poc_df.redock_rec_pocket_file = poc_df.redock_rec_pocket_file.str.split("/").str[-1]
    poc_df.crossdock_rec_pocket_file = poc_df.crossdock_rec_pocket_file.str.split("/").str[-1]

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
    save_smiles(folder + "/actives_no_cluster.smi", poc_df.lig_smiles)

    poc_df.to_csv(folder + "/actives_no_cluster.csv", index=False)

    for lig_file, rec_file, cd_rec_file in zip(poc_df.lig_crystal_file, poc_df.redock_rec_file, poc_df.crossdock_rec_file):
        shutil.copyfile(get_output_dir(cfg) + f"/{pocket}/{lig_file}", folder + f"/{lig_file}")
        shutil.copyfile(get_output_dir(cfg) + f"/{pocket}/{rec_file}", folder + f"/{rec_file}")
        shutil.copyfile(get_output_dir(cfg) + f"/{pocket}/{cd_rec_file}", folder + f"/{cd_rec_file}")

    for poc_file, cd_poc_file in zip(poc_df.redock_rec_pocket_file, poc_df.crossdock_rec_pocket_file):
        shutil.copyfile(get_output_dir(cfg) + f"/{pocket}/{poc_file}", folder + f"/{poc_file}")
        shutil.copyfile(get_output_dir(cfg) + f"/{pocket}/{cd_poc_file}", folder + f"/{cd_poc_file}")

    shutil.copyfile(bayesbind_folder + "/random.smi", folder + "/random.smi")


@task(force=False)
def make_all_bayesbind_struct(cfg, saved_bayesbind, cluster_cutoff=8, act_cutoff=5):

    struct_dfs = {
        "val": pd.read_csv(f"{get_output_dir(cfg)}/structures_val.csv"),
        "test": pd.read_csv(f"{get_output_dir(cfg)}/structures_test.csv"),
    }
    for key in struct_dfs:
        struct_dfs[key] = struct_dfs[key][struct_dfs[key].pchembl_value.notnull()]

    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        df = struct_dfs[split]
        poc_df = df.query(f"pocket == '{pocket}'").reset_index(drop=True)
        active_clusters = len(poc_df.query("pchembl_value > @act_cutoff").lig_cluster.unique())
        tot_clusters = len(poc_df.lig_cluster.unique())

        if active_clusters > 8:
            print(f"Running on {split}/{pocket} {tot_clusters=} {active_clusters=}")
            make_bayesbind_struct_dir(cfg, split, pocket, df)

if __name__ == "__main__":
    cfg = get_config("local")

    saved_bayesbind_struct = make_all_bayesbind_struct(None)

    workflow = Workflow(cfg, saved_bayesbind_struct)
    workflow.run()