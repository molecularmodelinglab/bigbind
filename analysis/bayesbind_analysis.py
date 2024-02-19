
import os
import numpy as np

import pandas as pd
from tqdm import tqdm

from baselines.eef import calc_best_eef, calc_eef


def get_eef_df(results_df, force=False):
    """ Computes the EEFs at various cutoffs for all the models
    on all the targets """

    out_filename = "outputs/basesline_eefs.csv"
    if not force and os.path.exists(out_filename):
        return pd.read_csv(out_filename)

    models = ["knn", "banana", "glide", "gnina", "vina"]
    act_cutoff = 5.0
    select_fracs = [0.1, 0.01, 0.001]
    pockets = results_df.pocket.unique()

    eef_rows = []
    for pocket in tqdm(pockets):
        poc_df = results_df.query(f"pocket == @pocket").reset_index(drop=True)
        act_mask = poc_df.set == "actives"
        activities = np.array(poc_df.pchembl_value[act_mask])
        assert np.isnan(activities).sum() == 0

        for model in models:
            act_preds = np.array(poc_df[model][act_mask]) 
            rand_preds = np.array(poc_df[model][~act_mask])

            for select_frac in select_fracs:
                EEF, EEF_low, EEF_hi, EEF_pval = calc_eef(act_preds, rand_preds, activities, act_cutoff, select_frac)
                
                row = {
                    "split": poc_df.split[0],
                    "pocket": pocket,
                    "model": model,
                    "EEF": EEF,
                    "EEF_low": EEF_low,
                    "EEF_high": EEF_hi,
                    "EEF_pval": EEF_pval,
                    "act_cutoff": act_cutoff,
                    "select_frac": select_frac,
                    "EEF_max": False
                }
                eef_rows.append(row)
            
            for max_uncertainty_ratio in [None, 2.0, 1.0, 0.5, 0.1]:
                # now for EEF_max
                EEF, EEF_low, EEF_hi, EEF_pval, select_frac = calc_best_eef(act_preds, rand_preds, activities, act_cutoff, max_uncertainty_ratio)

                row = {
                    "split": poc_df.split[0],
                    "pocket": pocket,
                    "model": model,
                    "EEF": EEF,
                    "EEF_low": EEF_low,
                    "EEF_high": EEF_hi,
                    "EEF_pval": EEF_pval,
                    "act_cutoff": act_cutoff,
                    "select_frac": select_frac,
                    "EEF_max": True,
                    "max_uncertainty_ratio": max_uncertainty_ratio
                }
                eef_rows.append(row)

    eef_df = pd.DataFrame(eef_rows)
    eef_df.to_csv(out_filename, index=False)

    return eef_df

if __name__ == "__main__":
    results_df = pd.read_csv("outputs/baseline_results.csv")
    eef_df = get_eef_df(results_df, force=True)
    print(eef_df.head())
    print(eef_df.describe())