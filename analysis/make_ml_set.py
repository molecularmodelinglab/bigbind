
import os
import shutil
import sys
import pandas as pd

from analysis.bayesbind_analysis import get_bad_pockets, get_metric_df
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from utils.cfg_utils import get_bayesbind_dir, get_bayesbind_ml_dir, get_config


def make_ml_set(cfg):
    results_df = pd.read_csv("outputs/baseline_results.csv")
    metric_df = get_metric_df(results_df, force=False)

    bad_pockets = get_bad_pockets(metric_df)
    ml_pockets = set(metric_df.pocket.unique()) - bad_pockets

    ml_dir = get_bayesbind_ml_dir(cfg)
    print(f"Making ML set to {ml_dir}")
    bb_dir = get_bayesbind_dir(cfg)
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        split_dir = os.path.join(ml_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        if pocket in ml_pockets:
            og_pocket_dir = os.path.join(bb_dir, split, pocket)
            ml_pocket_dir = os.path.join(split_dir, pocket)
            shutil.copytree(og_pocket_dir, ml_pocket_dir)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    make_ml_set(cfg)