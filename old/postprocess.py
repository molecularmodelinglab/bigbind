import yaml
import random
import json
import numpy as np
from copy import copy
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from collections import defaultdict
from probis import *
from run import *
from analysis import *

import matplotlib.pyplot as plt

with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

#save probis clusters to train df

full_scores = convert_inter_results_to_json(cfg)
pocket2rep_rec = get_rep_recs(cfg)

clusters = get_clusters(cfg, pocket2rep_rec, full_scores)
big_clusters = get_clusters(cfg, pocket2rep_rec, full_scores, z_cutoff=3.0)

def get_pocket_to_cluster_index(clusters):
    ret = {}
    for idx, cluster in enumerate(clusters):
        for pocket in cluster:
            ret[pocket] = idx
    return ret

val_poc2train_sim = {}
for poc in val_pockets:
    val_poc2train_sim[poc] = get_train_probis_sim(full_scores, pocket2rep_rec, probis_neighbors, train_rep_recs, poc)

for neg_ratio in [ None, 1 ]:
    for split in [ "train", "val", "test" ]:

        if neg_ratio is None:
            out_file = cfg["bigbind_folder"] + f"/activities_{split}.csv"
        else:
            out_file = cfg["bigbind_folder"] + f"/activities_sna_{neg_ratio}_{split}.csv"

        df = pd.read_csv(out_file)

        poc2cluster_idx = get_pocket_to_cluster_index(clusters)
        poc2big_cluster_idx = get_pocket_to_cluster_index(big_clusters)
        clusters_30 = []
        clusters_35 = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            clusters_30.append(poc2big_cluster_idx[row.pocket])
            clusters_35.append(poc2cluster_idx[row.pocket])
        df["split_cluster"] = clusters_30
        df["sna_cluster"] = clusters_35
        df.to_csv(out_file, index=False)