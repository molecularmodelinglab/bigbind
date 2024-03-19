
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import os
import warnings
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from baselines.efb import calc_best_efb, calc_efb

from baselines.metrics import calc_ef, compute_bootstrap_metrics, median_metric, roc_auc, to_str

ALL_MODELS = ["knn", "banana", "glide", "gnina", "vina"]
ACT_CUTOFF = 5.0

def get_pocket_metrics_df(args):
    """ Returns a dataframe of metrics for all the models on this pocket """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pocket, poc_df = args
        act_mask = poc_df.set == "actives"
        activities = np.array(poc_df.pchembl_value[act_mask])
        cutoff_mask = activities >= ACT_CUTOFF
        assert np.isnan(activities).sum() == 0

        rows = []
        for model in ALL_MODELS:

            act_preds = np.array(poc_df[model][act_mask])[cutoff_mask]
            rand_preds = np.array(poc_df[model][~act_mask])

            select_fracs = np.logspace(-1, -3, 15)
            metrics = {
                "EFB_max": calc_best_efb,
                "AUC": roc_auc,
            }
            for frac in select_fracs:
                percent_str = to_str(frac*100, 3)
                metrics[f"EF_{percent_str}%"] = partial(calc_ef, select_frac=frac)
                metrics[f"EFB_{percent_str}%"] = partial(calc_efb, select_frac=frac)
            row = {
                "split": poc_df.split[0],
                "model": model,
                "pocket": pocket,
            }
            row.update(compute_bootstrap_metrics((act_preds, rand_preds), metrics, n_resamples=1000, method="percentile"))
            rows.append(row)

        return pd.DataFrame(rows)

def get_metric_df(results_df, force=False):

    out_filename = "outputs/baseline_metrics.csv"
    if not force and os.path.exists(out_filename):
        df = pd.read_csv(out_filename)
    
    else:
        pockets = results_df.pocket.unique()
        poc_dfs = []
        for pocket in pockets:
            poc_df = results_df.query(f"pocket == @pocket").reset_index(drop=True)
            poc_dfs.append(poc_df)

        args = list(zip(pockets, poc_dfs))
        with Pool(8) as p:
            dfs = list(tqdm(p.imap(get_pocket_metrics_df, args), total=len(args)))

        df = pd.concat(dfs, ignore_index=True)

        for key in df.keys():
            if key.startswith("EFB") and "low" not in key and "high" not in key:
                # fixing issue with scipy bootstrap when the 95% CI lower bound
                # equals the the metric estimate
                for postfix in ["_low", "_high"]:
                    ci_key = key + postfix
                    nan_mask = np.isnan(df[ci_key])
                    df.loc[nan_mask, ci_key] = df.loc[nan_mask, key]

        df.to_csv(out_filename, index=False)

    return df

model_names = {
    "knn": "KNN",
    "banana": "BANANA",
    "vina": "Vina",
    "gnina": "GNINA",
    "glide": "Glide",
}
bb_ml_models = {"knn", "banana"}
model_order = ["Glide", "Vina", "GNINA", "BANANA", "KNN"]

def plot_efbs_on_ax(ax, cur_df, frac_str, only_low=False, logy=False, label_bottom=None, label_size=12):
    
    cur_df["Target"] = cur_df.pocket.str.split("_").str[0]
    cur_df["Model"] = cur_df.model.apply(lambda x: model_names[x])

    split2target = defaultdict(set)
    for row in cur_df.itertuples():
        split2target[row.split].add(row.pocket.split("_")[0])
    split2target = {k: sorted(v) for k, v in split2target.items()}
    sorted_targets = list(split2target["val"]) + list(split2target["test"])

    p_eef = cur_df.pivot(index="Target", columns="Model", values=f"EFB_{frac_str}")
    p_low = cur_df.pivot(index="Target", columns="Model", values=f"EFB_{frac_str}_low")
    p_high = cur_df.pivot(index="Target", columns="Model", values=f"EFB_{frac_str}_high")

    p_eef = p_eef.loc[sorted_targets]
    p_low = p_low.loc[sorted_targets]
    p_high = p_high.loc[sorted_targets]

    p_eef = p_eef[model_order[:len(p_eef.columns)]]
    p_low = p_low[model_order[:len(p_low.columns)]]
    p_high = p_high[model_order[:len(p_high.columns)]]

    err = []
    for col in p_low:
        err.append([p_eef[col].values - p_low[col].values, p_high[col].values - p_eef[col].values])
    err = np.abs(err)

    if only_low:
        p_low.plot(kind="bar", ax=ax, ylabel=f"$\\mathregular{{EF}}^B_\\mathregular{{{frac_str}}}$ lower CI bound", logy=logy)
    else:
        p_eef.plot(kind="bar", yerr=err, ax=ax, ylabel=f"$\\mathregular{{EF}}^B_\\mathregular{{{frac_str}}}$", logy=logy, ecolor="gray", capsize=2)

    ax.plot([-1, len(p_eef)], [1, 1], "k--")

    # separate the val and test sets
    ax.axvline(len(split2target["val"]) - 0.5, color="silver", linestyle="-")
    # label the val and test sets at the top
    if only_low:
        top_y = p_low.max().max()
    else:
        top_y = p_high.max().max()
    top_y *= 1.05
    val_x = (len(split2target["val"]) - 0.5) / 2
    

    if label_bottom is not None:
        top_y = label_bottom

    ax.text(val_x, top_y, "Validation", ha="center", va="center", fontsize=label_size)
    test_x = len(split2target["val"]) + (len(split2target["test"]) - 0.5) / 2
    # move the test label to the right if it's too close to the legend
    if test_x > len(p_eef)*0.7:
        test_x = len(p_eef)*0.7
    ax.text(test_x, top_y, "Test", ha="center", va="center", fontsize=label_size)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_efbs(*args, **kwargs):
    fig, ax = plt.subplots(figsize=(8,6))
    plot_efbs_on_ax(ax, *args, **kwargs)
    return fig, ax

def get_bad_pockets(metric_df, knn_cutoff=30):
    """ Returns the pockets to exclude from the ML subset"""
    query = "model == 'knn' and EFB_max > @knn_cutoff"
    return set(metric_df.query(query).pocket.unique())


def get_median_metric_df(results_df, metric_df, split, ml, force=False):
    """ Gets the (bootstrapped) median values for AUC,
    EFB_max, EFB_1%, and EFB_1% for each model over 
    all the pockets """

    out_filename = f"outputs/baseline_median_metrics_{split}_{ml}.csv"

    if not force and os.path.exists(out_filename):
        df = pd.read_csv(out_filename)
    else:
        if ml:
            bad_pockets = get_bad_pockets(metric_df)
        else:
            bad_pockets = set()
        results_df = results_df.query("pocket not in @bad_pockets and split == @split").reset_index(drop=True)
    

        model_preds = defaultdict(list)

        pockets = results_df.pocket.unique()
        for pocket in pockets:
            poc_df = results_df.query(f"pocket == @pocket").reset_index(drop=True)

            act_mask = poc_df.set == "actives"
            activities = np.array(poc_df.pchembl_value[act_mask])
            cutoff_mask = activities >= ACT_CUTOFF
            assert np.isnan(activities).sum() == 0

            rows = []
            for model in ALL_MODELS:

                act_preds = np.array(poc_df[model][act_mask])[cutoff_mask]
                rand_preds = np.array(poc_df[model][~act_mask])

                model_preds[model].append(act_preds)
                model_preds[model].append(rand_preds)

        rows = []
        for model, cur_preds in tqdm(model_preds.items()):
            metrics = {
                "AUC": roc_auc,
                "EFB_max": calc_best_efb,
                "EFB_1%": lambda act, rand: calc_efb(act, rand, 0.01),
                "EF_1%": lambda act, rand: calc_ef(act, rand, 0.01),
            }

            median_metrics = {
                metric: partial(median_metric, metrics[metric])
                for metric in metrics
            }

            row = {"model": model}
            row.update(compute_bootstrap_metrics(cur_preds, median_metrics, n_resamples=1000, method="percentile"))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out_filename, index=False)

    return df

def get_metric_tables(results_df, metric_df, use_ml_models, only_ml_pockets, force=False, sigfigs=2):
    """ Tables 2 and 3 in the paper """
    cur_models = ALL_MODELS if use_ml_models else [ model for model in ALL_MODELS if model not in bb_ml_models]

    med_dfs = {}
    for split in ["val", "test"]:
        df = get_median_metric_df(results_df, metric_df, split, only_ml_pockets, force=force)
        df = df.query("model in @cur_models").reset_index(drop=True)
        med_dfs[split] = df

    tab_rows = []
    for model in cur_models:
        row = {"Model": model_names[model]}
        for split, med_df in med_dfs.items():
            for metric in ["AUC", "EFB_1%", "EFB_max"]:
                val = med_df.query("model == @model")[metric].values[0]
                low = med_df.query("model == @model")[f"{metric}_low"].values[0]
                high = med_df.query("model == @model")[f"{metric}_high"].values[0]
                row[f"{split} {metric}"] = f"{to_str(val, sigfigs)} [{to_str(low, sigfigs)}, {to_str(high, sigfigs)}]"
        tab_rows.append(row)

    tab_df = pd.DataFrame(tab_rows)
    return tab_df

def plot_efbs_side_by_side(cur_df, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    plot_efbs_on_ax(axes[0], cur_df, "max")
    axes[0].get_legend().remove()
    plot_efbs_on_ax(axes[1], cur_df, "max", only_low=True)
    axes[1].legend(loc="upper right", fontsize=8, title="Model")
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_all_figs(metric_df):

    bad_pockets = get_bad_pockets(metric_df)

    ml_pockets = set(metric_df.pocket.unique()) - bad_pockets
    all_pockets = set(metric_df.pocket.unique())

    non_ml_models = { "glide", "vina", "gnina" }
    all_models = set(metric_df.model.unique())

    ml_query = f"pocket not in @bad_pockets"
    cur_df = metric_df.query(ml_query).reset_index(drop=True)
    plot_efbs_side_by_side(cur_df, "Model performance on the BayesBind ML set").savefig("outputs/efb_ml.pdf")
    # for low in [True, False]:
    #     fig, ax = plot_efbs(cur_df, "max", only_low=low, logy=False)
    #     ax.set_title(f"Model performance on the BayesBind ML set")
    #     fig.tight_layout()
    #     fig.savefig(f"outputs/efb_max_ml_low_{low}.pdf")

    full_non_ml_query = f"model not in @bb_ml_models"
    cur_df = metric_df.query(full_non_ml_query).reset_index(drop=True)
    plot_efbs_side_by_side(cur_df, "Model performance on the BayesBind full set").savefig("outputs/efb_full.pdf")
    # for low in [True, False]:
    #     fig, ax = plot_efbs(cur_df, "max", only_low=low, logy=False)
    #     ax.set_title(f"Model performance on the BayesBind full set")
    #     fig.tight_layout()
    #     fig.savefig(f"outputs/efb_max_full_non_ml_low_{low}.pdf")

    cur_df = metric_df
    plot_efbs_side_by_side(cur_df, "Model performance on the BayesBind full set").savefig("outputs/efb_full_full.pdf")
    # for low in [True, False]:
    #     fig, ax = plot_efbs(cur_df, "max", only_low=low, logy=False)
    #     ax.set_title(f"Model performance on the BayesBind full set")
    #     fig.tight_layout()
    #     fig.savefig(f"outputs/efb_max_full_full_low_{low}.pdf")

def save_poster_figure(metric_df):
    bad_pockets = get_bad_pockets(metric_df)

    ml_pockets = set(metric_df.pocket.unique()) - bad_pockets
    all_pockets = set(metric_df.pocket.unique())

    non_ml_models = { "glide", "vina", "gnina" }
    all_models = set(metric_df.model.unique())

    ml_query = f"pocket not in @bad_pockets"
    cur_df = metric_df.query(ml_query).reset_index(drop=True)
    low = False

    fig, ax = plot_efbs(cur_df, "max", only_low=low, logy=False, label_bottom=-85, label_size=20)
    ax.set_ylim([0, 1100])
    ax.set_xlabel("Protein target", fontsize=24)
    ax.set_ylabel("$\\mathregular{{EF}}^B_\\mathregular{max}$", fontsize=24)
    ax.xaxis.set_label_coords(0.5,-0.1)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.axhline(1000, color="grey", linestyle="--", linewidth=1)
    ax.legend(loc="upper right", fontsize=20)
    fig.set_size_inches(15, 15)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.98, bottom=0.12)
    # fig.tight_layout()
    fig.savefig("outputs/poster.png")#, dpi=200)

def main():
    results_df = pd.read_csv("outputs/baseline_results.csv")
    metric_df = get_metric_df(results_df, force=False)

    # save_poster_figure(metric_df)
    plot_all_figs(metric_df)

    full_tab = get_metric_tables(results_df, metric_df, use_ml_models=False, only_ml_pockets=False, force=False)
    full_full_tab = get_metric_tables(results_df, metric_df, use_ml_models=True, only_ml_pockets=False, force=False)
    ml_tab = get_metric_tables(results_df, metric_df, use_ml_models=True, only_ml_pockets=True, force=False)

    # just print out a bunch of numbers we can copy into the paper

    print("Full median results")
    print(full_tab.to_latex(index=False))

    print("ML median results")
    print(ml_tab.to_latex(index=False))

    print("Full median results (all models)")
    print(full_full_tab.to_latex(index=False))

    max_knn = metric_df.query("model == 'knn'").EFB_max.max()
    print(f"Max value achieved by KNN on any target: {max_knn}")

    med_knn_full = full_full_tab.query("Model == 'KNN'")
    med_knn_val = med_knn_full["val EFB_max"].values[0]
    med_knn_test = med_knn_full["test EFB_max"].values[0]
    print(f"median EFB_max for KNN val: {med_knn_val} and test: {med_knn_test}")

    best_model = "vina"
    best_pocket = "NR1H2_HUMAN_213_460_0"
    best_target = best_pocket.split("_")[0]
    best_row = metric_df.query("model == @best_model and pocket == @best_pocket").iloc[0]
    print(f"{best_model} EFB_max on {best_pocket}: {to_str(best_row.EFB_max, 2)} [{to_str(best_row.EFB_max_low, 2)}, {to_str(best_row.EFB_max_high, 2)}]")

if __name__ == "__main__":
    main()