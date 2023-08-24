import yaml
import pandas as pd
from run import *

def get_train_probis_sim(full_scores, pocket2rep_rec, probis_neighbors, train_rep_recs, val_poc):
    val_rep_rec = pocket2rep_rec[val_poc]
    train_neighbors = { rec for rec in probis_neighbors[val_rep_rec] if rec in train_rep_recs }
    max_score = 0.0
    for train_rec in train_neighbors:
        for key in ((val_rep_rec, train_rec), (train_rec, val_rep_rec)):
            if key in full_scores:
                score = full_scores[key]
                if score > max_score:
                    max_score = score
    return max_score

with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

full_scores = convert_inter_results_to_json(cfg)
pocket2rep_rec = get_rep_recs(cfg)

train_file = cfg["bigbind_folder"] + f"/activities_train.csv"
val_file = cfg["bigbind_folder"] + f"/activities_val.csv"

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)

train_pockets = train_df.pocket.unique()
val_pockets = val_df.pocket.unique()
train_rep_recs = { pocket2rep_rec[poc] for poc in train_pockets }
probis_neighbors = defaultdict(set)
for r1, r2 in full_scores:
    probis_neighbors[r1].add(r2)
    probis_neighbors[r2].add(r1)

val_poc2train_sim = {}
for poc in val_pockets:
    val_poc2train_sim[poc] = get_train_probis_sim(full_scores, pocket2rep_rec, probis_neighbors, train_rep_recs, poc)

print(val_poc2train_sim)