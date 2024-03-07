
from collections import defaultdict
from functools import lru_cache
from glob import glob
import os
import random
import sys
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaml import warnings
from baselines.efb import calc_best_efb, calc_efb
from baselines.metrics import calc_ef, compute_bootstrap_metrics, roc_auc, to_str

from utils.cfg_utils import get_config

def get_all_dude_targets(cfg):
    """ Returns a list of all the targets in the DUD-E dataset """
    return [ f.split("/")[-1] for f in glob(f"{cfg.host.dude_vs_folder}/*") ]

@lru_cache
def get_dude_df(cfg, target, summary_prefix):
    """ Loads the dude dataframe for a given target """
    default_file = f"{cfg.host.dude_vs_folder}/{target}/{summary_prefix}.summary"

    default_df = pd.read_csv(default_file, delim_whitespace=True)#, skiprows=1, names=["rank", "title", "vina", "target", "file", "gnina_affinity", "gnina_score"])

    mis_file = f"{cfg.host.dude_vs_folder}/{target}/missing.summary"
    mis_df = pd.read_csv(mis_file, delim_whitespace=True)
    dude_df = pd.concat([default_df, mis_df])

    to_collate = defaultdict(list)
    # take the average of all the seed values
    for key in dude_df.columns:
        if "seed" in key:
            splt = key.split("_")
            splt = splt[:-2] + splt[-1:]
            parent_key = "_".join(splt)
            to_collate[parent_key].append(key)
            
    for parent, keys in to_collate.items():
        dude_df[parent] = dude_df[keys].mean(axis=1)
        dude_df = dude_df.drop(columns=keys)

    # Vinardo and Vina are negative
    for neg_key in ["Vina", "Vinardo"]:
        if neg_key in dude_df.columns:
            dude_df[neg_key] = -dude_df[neg_key]

    return dude_df

def get_dude_scores(cfg, target, model, summary_prefix):
    """ Returns the scores for the active and decoy ligands
    for a given target and model. Eg use newdefault_CNNaffinity
    for the default gnina model """
    
    dude_df = get_dude_df(cfg, target, summary_prefix)

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
        active_scores, decoy_scores = get_dude_scores(cfg, target, model, "newdefault")
        fake_activities = np.ones(len(active_scores))

        for frac in [0.1, 0.01, 0.001, 0.0001]:
            EF = calc_ef(active_scores, decoy_scores, frac)
            EEF= calc_efb(active_scores, decoy_scores, frac)
            
            rows.append({
                "target": target,
                "model": model,
                "fraction": frac,
                "EF": EF,
                "EEF": EEF,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)

    return df

def dude_ef_row(args):
    target, model, active_scores, decoy_scores = args
    select_fracs = np.logspace(-1, -3, 15)
    metrics = {
        "EFB_max": calc_best_efb,
    }
    for frac in select_fracs:
        percent_str = to_str(frac*100, 3)
        metrics[f"EF_{percent_str}%"] = lambda act_preds, rand_preds: calc_ef(act_preds, rand_preds, frac)
        metrics[f"EFB_{percent_str}%"] = lambda act_preds, rand_preds: calc_efb(act_preds, rand_preds, frac)

    row = {
        "target": target,
        "model": model,
    }
    row.update(compute_bootstrap_metrics((active_scores, decoy_scores), metrics, method="percentile", n_resamples=1000))

    return row

def get_all_dude_ef_df(cfg, force=False):
    """ Computes the EFs and EF(B)s at various cutoffs for all the models
    on all the targets """

    out_filename = "outputs/dude_metrics.csv"
    if not force and os.path.exists(out_filename):
        return pd.read_csv(out_filename)

    models = {
        "Vina": ("Vina", "newdefault"),
        "Default (Affinity)": ("newdefault_CNNaffinity", "newdefault"),
        "Default (Pose)": ("newdefault_CNNscore", "newdefault"),
        "Vinardo": ("Vinardo", "vinardo"),
        "General (Affinity)": ("general_default2018_CNNaffinity", "sdsorter"),
        "General (Pose)": ("general_default2018_CNNscore", "sdsorter"),
        "Dense (Affinity)": ("dense_CNNaffinity", "sdsorter"),
        "Dense (Pose)": ("dense_CNNscore", "sdsorter"),
    }
    targets = get_all_dude_targets(cfg)

    # select_fracs = np.logspace(-1, -3, 15)
    select_fracs = np.logspace(0, -5, 21)

    efb_rows = []
    args = []
    for target in tqdm(targets):
        for model, (key, prefix) in (models.items()):
            active_scores, decoy_scores = get_dude_scores(cfg, target, key, prefix)
            args.append((target, model, active_scores, decoy_scores))

    with Pool(8) as p:
        efb_rows = list(tqdm(p.imap(dude_ef_row, args), total=len(args)))
        # ebf_rows = [dude_ef_row(*arg) for arg in args]

    efb_df = pd.DataFrame(efb_rows)
    efb_df.to_csv(out_filename, index=False)
    return efb_df

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
    
def make_dude_ef_table(df, sigfigs=2):
    """ Makes a table of the median EF and EFB for each model at various
    selection fractions. This prints out the LaTeX code for the table """
    
    tab_dict = defaultdict(dict)

    model_order = [
        "Vina",
        "Vinardo",
        "General (Affinity)",
        "General (Pose)",
        "Dense (Affinity)",
        "Dense (Pose)",
        "Default (Affinity)",
        "Default (Pose)"
    ]

    for model in model_order:
        for select_frac in [0.01, 0.001]:
            model_df = df.query(f"model == '{model}' and select_frac == {select_frac}").reset_index(drop=True)

            efbs = model_df.EFB.sort_values()
            med_idx = efbs.index[len(efbs) // 2]
            row = model_df.loc[med_idx]
            med_efb = f"{to_str(row.EFB, sigfigs)} [{to_str(row.EFB_low, sigfigs)}, {to_str(row.EFB_high, sigfigs)}]"
            med_ef = model_df.EF.median()

            percent = select_frac * 100
            if int(percent) == percent:
                percent = int(percent)

            tab_dict[model].update({
                f"$\\text{{EF}}_{{{percent}\%}}$": f"{to_str(med_ef, sigfigs)}",
                f"$\\text{{EF}}^B_{{{percent}\%}}$": med_efb,
            })
        melb_df = df.query(f"model == '{model}' and EFB_max == True").reset_index(drop=True)
        melb = melb_df.EFB_low.median()
        tab_dict[model].update({
            "MELB": to_str(melb, sigfigs),
        })



    tab_rows = []
    for key, val in tab_dict.items():
        tab_rows.append({"Model": key, **val})

    tab_df = pd.DataFrame(tab_rows)

    return tab_df

def get_mean_deocys_per_active(cfg):
    """ Returns the mean number of decoys per active
    for all the DUD-E targets"""
    targets = get_all_dude_targets(cfg)
    key, prefix = ("Vina", "newdefault") #arbitrary
    ratios = []
    for target in tqdm(targets):
        active_scores, decoy_scores = get_dude_scores(cfg, target, key, prefix)
        ratios.append(len(decoy_scores) / len(active_scores))

    return np.mean(ratios)

def plot_efb_vs_frac_fig(ef_df):
    """ Plot a figure showing the relationship between the
    EFB and the selection fraction for random model-target-pairs """

    targets = ef_df.target.unique()
    models = ef_df.model.unique()

    fig, axes = plt.subplots(3, 3)

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            target = random.choice(targets)
            model = random.choice(models)
            cur_df = ef_df[(ef_df.target == target) & (ef_df.model == model) & ~ef_df.EFB_max]
            melb = ef_df[(ef_df.target == target) & (ef_df.model == model) & ef_df.EFB_max].EFB_low.max()
            
            mask = np.array(cur_df.EFB > 0)
            high_vals = np.array(cur_df.EFB_high)
            max_upper = np.max(high_vals)
            for k in range(len(high_vals)):
                if high_vals[k] == max_upper and i < len(high_vals) - 1:
                    mask[k+1:] = False

            errs = np.array([cur_df.EFB[mask] - cur_df.EFB_low[mask], cur_df.EFB_high[mask] - cur_df.EFB[mask]])

            melb = cur_df.EFB_low.max()
            ax.errorbar(cur_df.select_frac[mask], cur_df.EFB[mask], yerr=errs, ecolor='grey', label="EF$^B$")
            ax.plot(cur_df.select_frac[mask], cur_df.EF[mask], label="EF")
            ax.axhline(melb, color='red', linestyle='--', label="MELB")
            ax.set_xscale("log")
            ax.invert_xaxis()
            if i == len(axes) - 1 and j == len(row)//2:
                ax.set_xlabel("$\chi$")
            if j == 0 and i == len(axes)//2:
                ax.set_ylabel(f"Enrichment")
            # if j == len(row) - 1 and i == len(axes) - 1:
            # if i == 0 and j == 0:
            #     ax.legend(prop={"size": 8})
        # decrease padding around the y and x labels
    fig.suptitle("EF$^B_\chi$ versus EF$_\chi$ for various $\chi$")
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    # df = get_gnina_dude_ef_df(cfg)
    # plot_eef_vs_ef_fig(df)
    all_df = get_all_dude_ef_df(cfg, force=True)
    # tab_df = make_dude_ef_table(all_df)
    # print(tab_df.to_latex(index=False))

    # fig = plot_efb_vs_frac_fig(all_df)
    # fig.savefig("outputs/efb_chi.pdf")