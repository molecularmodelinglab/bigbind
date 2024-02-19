
from glob import glob
import os
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from baselines.eef import calc_eef

from utils.cfg_utils import get_config


def calc_ef(active_scores, decoy_scores, frac):
    """ Calculate the enrichment factor (EF) for a given
    fraction of the dataset."""
    all_scores = np.concatenate([decoy_scores, active_scores])
    n_chosen = int(frac * len(all_scores))
    if n_chosen < 1:
        n_chosen = 1

    is_active = np.zeros(len(all_scores), dtype=bool)
    is_active[-len(active_scores):] = True

    indexes = np.argsort(-all_scores)
    chosen = indexes[:n_chosen]

    tot_active = np.sum(is_active)/len(is_active)
    chosen_active = np.sum(is_active[chosen])/len(chosen)

    EF = chosen_active/tot_active

    return EF

def get_all_dude_targets(cfg):
    """ Returns a list of all the targets in the DUD-E dataset """
    return [ f.split("/")[-1] for f in glob(f"{cfg.host.dude_vs_folder}/*") ]

def get_dude_scores(cfg, target, model):
    """ Returns the scores for the active and decoy ligands
    for a given target and model. Eg use newdefault_CNNaffinity
    for the default gnina model """
    
    default_file = f"{cfg.host.dude_vs_folder}/{target}/newdefault.summary"

    default_df = pd.read_csv(default_file, delim_whitespace=True)#, skiprows=1, names=["rank", "title", "vina", "target", "file", "gnina_affinity", "gnina_score"])

    mis_file = f"{cfg.host.dude_vs_folder}/{target}/missing.summary"
    mis_df = pd.read_csv(mis_file, delim_whitespace=True)
    dude_df = pd.concat([default_df, mis_df])

    # for some reason the vina scores are negative
    dude_df["Vina"] = -dude_df["Vina"]

    # max score per compound
    group = dude_df.groupby("Title")
    grouped_df = group[model].max()

    active_mask = group.File.first().str.contains("active")
    decoy_mask = group.File.first().str.contains("decoy")
    if not np.all(decoy_mask == ~active_mask):
        neither_mask = ~decoy_mask & ~active_mask
        print(target, grouped_df[neither_mask])
    assert np.all(decoy_mask == ~active_mask)

    active_scores = np.array(grouped_df[active_mask])
    decoy_scores = np.array(grouped_df[decoy_mask])

    random.shuffle(decoy_scores)
    random.shuffle(active_scores)

    assert len(active_scores) > 0
    assert len(decoy_scores) > 0

    return active_scores, decoy_scores


def get_gnina_dude_ef_df(cfg):
    """ Returns a dataframe containing the estimated EEFs and EFs
    for gnina on all the DUD-E targets """

    out_file = "outputs/gnina_dude_ef.csv"
    if os.path.exists(out_file):
        return pd.read_csv(out_file)

    model = "newdefault_CNNaffinity"
    targets = get_all_dude_targets(cfg)
    rows = []
    for target in tqdm(targets):
        active_scores, decoy_scores = get_dude_scores(cfg, target, model)
        fake_activities = np.ones(len(active_scores))

        for frac in [0.1, 0.01, 0.001, 0.0001]:
            EF = calc_ef(active_scores, decoy_scores, frac)
            EEF, EEF_lo, EEF_hi, pval = calc_eef(active_scores, decoy_scores, fake_activities, 0.0, frac)
            
            rows.append({
                "target": target,
                "model": model,
                "fraction": frac,
                "EF": EF,
                "EEF": EEF,
                "EEF_low": EEF_lo,
                "EEF_hi": EEF_hi,
                "EEF_p": pval
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)

    return df

def plot_eef_vs_ef_fig(df):
    """ Plot a figure showing the relationship between the
    EEF and the EF for GNINA on DUD-E """
    fracs = df.fraction.unique()

    fig, axes = plt.subplots(1, len(fracs), figsize=(5*len(fracs), 5))
    for i, (frac, ax) in enumerate(zip(fracs, axes)):
        frac_df = df[df.fraction == frac]

        max_ef = frac_df.EF.max()
        max_eef = frac_df.EEF.max()
        max_val = min(max_ef, max_eef)
        EEF_errs = np.array([frac_df.EEF - frac_df.EEF_low, frac_df.EEF_hi - frac_df.EEF])

        ax.errorbar(frac_df.EF, frac_df.EEF, yerr=EEF_errs, label="EEF", fmt='o')
        ax.plot([0, max_val], [0, max_val], color='black', linestyle='--')
        
        percent = frac * 100
        if percent == int(percent):
            percent = int(percent)
        
        ax.set_xlabel(f"EF$_{{{percent}\%}}$", fontsize=12)
        ax.set_ylabel(f"EEF$_{{{percent}\%}}$", fontsize=12)

    fig.tight_layout()
    fig.suptitle("EEF vs EF for GNINA on DUD-E", fontsize=14)
    fig.subplots_adjust(top=0.9)
    fig.savefig("outputs/gnina_dude_eef_vs_ef.pdf")

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    df = get_gnina_dude_ef_df(cfg)
    plot_eef_vs_ef_fig(df)