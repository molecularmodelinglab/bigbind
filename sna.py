from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import random
from copy import deepcopy
import os

from cache import cache

def make_sna_df(cfg, df, big_clusters, smiles2filename, neg_ratio):
    """ Adds putative negative examples to the activity df.
    big_clusters are the set of clusters within which we assume
    things might bind (e.g. within the kinase cluster). Neg_ratio
    is the ratio of new negatives to the original dataset. """

    poc2cluster = {}
    for cluster in big_clusters:
        for pocket in cluster:
            poc2cluster[pocket] = cluster

    smiles2pockets = defaultdict(set)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smiles2pockets[row.lig_smiles].add(row.pocket)

    all_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            cluster = poc2cluster[row.pocket]
        except KeyError:
            # :eyes: not sure why this is happening...
            cluster = { row.pocket }
        all_rows.append(row)
        for i in range(neg_ratio):
            # find a ligand that isn't known to bind to anything
            # in the cluster
            while True:
                neg_smiles = random.choice(df.lig_smiles)
                # todo: should be unnecessary after I finish running
                try:
                    neg_filename = smiles2filename[neg_smiles]
                except KeyError:
                    continue
                other_pockets  = smiles2pockets[neg_smiles]
                if len(other_pockets.intersection(cluster)) == 0:
                    break
            new_row = deepcopy(row)
            new_row.lig_smiles = neg_smiles
            new_row.lig_file = neg_filename
            new_row.active = False
            all_rows.append(new_row)
    sna_df = pd.DataFrame(all_rows)

    # the non-boolean data no longer makes sense, so remove it
    to_remove = [ "pchembl_value" ] + [ key for key in sna_df if key.startswith("standard_") ]
    for key in to_remove:
        sna_df.drop(key, axis=1, inplace=True)
    
    return sna_df

@cache
def save_all_sna_dfs(cfg, big_clusters, smiles2filename, neg_ratio=1):
    for split in ["val", "test", "train" ]:
        df = pd.read_csv(cfg["bigbind_folder"] + f"/activities_{split}.csv")
        sna_df = make_sna_df(cfg, df, big_clusters, smiles2filename, neg_ratio)
        out_file = cfg["bigbind_folder"] + f"/activities_sna_{neg_ratio}_{split}.csv"
        print(f"Saving SNA activities to {out_file}")
        sna_df.to_csv(out_file, index=False)

def make_screen_df(cfg, df, pocket, poc2cluster, smiles2pockets, smiles2filename, tot_len=1000, max_actives=10):
    cluster = poc2cluster[pocket]
    poc_df = df.query("pocket == @pocket")
    # shuffle!
    poc_df = poc_df.sample(frac=1).reset_index(drop=True)
    active = poc_df.query("active").reset_index(drop=True)[:max_actives]
    if len(active) == 0:
        print(f"Skipping {pocket} because there are no actives")
    inactive = poc_df.query("not active")
    out = pd.concat([active, inactive]).reset_index(drop=True)
    seen_smiles = set(out.lig_smiles)
    to_add = tot_len - len(out)
    new_rows = []
    for i in range(to_add):
        row = out.loc[0]
        while True:
            neg_smiles = random.choice(df.lig_smiles)
            if neg_smiles in seen_smiles: continue
            try:
                neg_filename = smiles2filename[neg_smiles]
            except KeyError:
                continue
            other_pockets  = smiles2pockets[neg_smiles]
            if len(other_pockets.intersection(cluster)) == 0:
                break
        new_row = deepcopy(row)
        new_row.lig_smiles = neg_smiles
        new_row.lig_file = neg_filename
        new_row.active = False
        new_rows.append(new_row)
    out = pd.concat([out, pd.DataFrame(new_rows)])

    to_remove = [ "pchembl_value" ] + [ key for key in out if key.startswith("standard_") ]
    for key in to_remove:
        out.drop(key, axis=1, inplace=True)

    return out

@cache
def save_all_screen_dfs(cfg, big_clusters, smiles2filename):

    poc2cluster = {}
    for cluster in big_clusters:
        for pocket in cluster:
            poc2cluster[pocket] = cluster

    for split in ["val", "test" ]:
        out_folder = cfg["bigbind_folder"] + f"/{split}_screens"
        os.makedirs(out_folder, exist_ok=True)
        df = pd.read_csv(cfg["bigbind_folder"] + f"/activities_{split}.csv")

        smiles2pockets = defaultdict(set)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            smiles2pockets[row.lig_smiles].add(row.pocket)

        for pocket in tqdm(df.pocket.unique()):
            screen_df = make_screen_df(cfg, df, pocket, poc2cluster, smiles2pockets, smiles2filename)
            out_file = out_folder + "/" + pocket + ".csv"
            screen_df.to_csv(out_file, index=False)

if __name__ == "__main__":
    from run import *
    from probis import *
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    pocket2rep_rec = get_rep_recs(cfg)
    probis_scores = convert_inter_results_to_json(cfg)
    smiles2filename = save_all_mol_sdfs(cfg)
    # z cutoff of 3 clusters the kinases together
    big_clusters = get_clusters(cfg, pocket2rep_rec, probis_scores, z_cutoff=3.0)
    # save_all_sna_dfs(cfg, big_clusters, smiles2filename)
    save_all_screen_dfs(cfg, big_clusters, smiles2filename)