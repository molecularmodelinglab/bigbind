
from collections import defaultdict
import os
import sys
from matplotlib import pyplot as plt
from meeko import PDBQTMolecule
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm, trange
from rdkit.Chem import PandasTools
from baselines.eef import calc_best_eef, calc_eef, get_all_bayesbind_activities, postprocess_preds
from baselines.knn import get_all_bayesbind_knn_preds
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from baselines.banana import run_all_banana

from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config
from utils.task import task
from utils.workflow import Workflow


def get_docked_scores_from_pdbqt(fname):
    return PDBQTMolecule.from_file(fname)._pose_data["free_energies"]

def get_vina_preds(cfg, split, pocket):
    ret = {}
    for prefix in ("actives", "random"):
        scores = []
        df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv")
        for i in trange(len(df)):
            fname = get_baseline_dir(cfg, "vina", split, pocket) + f"/{prefix}_{i}.pdbqt"
            if os.path.exists(fname):
                cur_scores = get_docked_scores_from_pdbqt(fname)
                scores.append(-cur_scores[0])
            else:
                scores.append(-1000)
        ret[prefix] = np.asarray(scores)
    return ret

@task(force=False)
def get_all_vina_scores(cfg):
    ret = {}
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        ret[pocket] = get_vina_preds(cfg, split, pocket)
    return ret

def get_gnina_preds(cfg, split, pocket):
    ret = {}
    for prefix in ("actives", "random"):
        scores = []
        df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{prefix}.csv")
        for i in trange(len(df)):
            fname = get_baseline_dir(cfg, "gnina", split, pocket) + f"/{prefix}_{i}.sdf"
            if os.path.exists(fname):
                sd_df = PandasTools.LoadSDF(fname, strictParsing=False, molColName=None)
                try:
                    scores.append(sd_df.CNNaffinity[0])
                except AttributeError:
                    print(f"Error loading {fname}")
                    scores.append(-1000)
            else:
                scores.append(-1000)
        ret[prefix] = np.asarray(scores)
    return ret

@task(force=False)
def get_all_gnina_scores(cfg):
    ret = {}
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        ret[pocket] = get_gnina_preds(cfg, split, pocket)
    return ret

def get_glide_scores(df, num_compounds):
    ret = np.zeros(num_compounds) - 10000
    og_index = df.title.str.split(":").apply(lambda x: int(x[-1]) - 1)
    for i in range(len(df)):
        ret[og_index[i]] = max(-df.r_i_docking_score[i], ret[og_index[i]])
    return ret

@task(force=False)
def get_all_glide_scores(cfg):
    preds = {}
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        folder = get_baseline_dir(cfg, "glide", split, pocket)
        act_csv = f"{folder}/actives_results/dock_actives.csv"
        rand_csv = f"{folder}/random_results/dock_random.csv"
        true_act_df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/actives.csv")
        true_rand_df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/random.csv")
        if os.path.exists(act_csv) and os.path.exists(rand_csv):
            preds[pocket] = {}
            rand_df = pd.read_csv(rand_csv)
            act_df = pd.read_csv(act_csv)
            preds[pocket]["actives"] = get_glide_scores(act_df, len(true_act_df))
            preds[pocket]["random"] = get_glide_scores(rand_df, len(true_rand_df))
        else:
            print(f"Missing {split} {pocket} {os.path.exists(act_csv)} {os.path.exists(rand_csv)} {len(true_act_df)}")
    return preds

def get_baseline_workflow(cfg):
    return Workflow(cfg,
                    get_all_bayesbind_knn_preds(),
                    get_all_vina_scores(),
                    get_all_gnina_scores(),
                    get_all_glide_scores(),
                    run_all_banana(),
    )

def get_results_df(cfg):

    out_fname = "outputs/full_baseline_results.csv"
    if os.path.exists(out_fname):
        return pd.read_csv(out_fname)

    all_preds = [ postprocess_preds(cfg, res) for res in get_baseline_workflow(cfg).run() ]
    model_preds = {
        "KNN": all_preds[0],
        "Vina": all_preds[1],
        "GNINA": all_preds[2],
        "Glide": all_preds[3],
        "BANANA": all_preds[4],
    }
    poc2activities = get_all_bayesbind_activities(cfg)

    fracs = [0.1, 0.01, 0.001, 0.0001]
    act_cutoff = 5

    all_results = []

    for model, predictions in model_preds.items():
        cur_results = { "Model": model }
        for pocket, preds in tqdm(predictions.items()):
            all_results.append({ "Model": model, "Target": pocket.split("_")[0] })
            for frac in fracs:
                eef, low, high, pval = calc_eef(preds["actives"], preds["random"], poc2activities[pocket], act_cutoff, select_frac=frac)
                all_results[-1].update({ f"EEF_{frac}": eef, f"EEF_{frac}_low": low, f"EEF_{frac}_high": high, f"EEF_{frac}_p_raw": pval })

            eef_max, high_max, low_max, p_max, frac_max, N_max = calc_best_eef(preds, poc2activities[pocket], act_cutoff)
            all_results[-1].update({
                "EEF_max": eef_max,
                "EEF_max_low": low_max,
                "EEF_max_high": high_max,
                "p_max_raw": p_max,
                "frac_max": frac_max,
                "N_max": N_max
            })

    all_results = pd.DataFrame(all_results)

    # find adjusted p values using Benjaminini-Yekutieli

    p_keys = [ key for key in all_results.keys() if "p_raw" in key ]
    ps = np.zeros((len(p_keys), len(all_results)))

    for i, key in enumerate(p_keys):
        ps[i] = all_results[key]

    ps_adj = scipy.stats.false_discovery_control(ps, method='by')

    for i, key in enumerate(p_keys):
        new_key = key.replace("p_raw", "p_adj")
        all_results[new_key] = ps_adj[i]

    all_results.to_csv(out_fname, index=False)

    return all_results

def eval_results(cfg):

    summary_df = []

    all_results = get_results_df(cfg)

    p_keys = [ key for key in all_results.keys() if "p_adj" in key ]
    model_sig_targets = {}

    split_targs = defaultdict(set)
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        split_targs[split].add(pocket.split("_")[0])

    for model in all_results.Model.unique():
        cur_row = { "Model": model }

        for split, targs in split_targs.items():
            cur_results = all_results.query("Model == @model and Target in @targs")

            for frac in [0.1, 0.01, 0.001, 0.0001, "max"]:
                eefs = cur_results[f"EEF_{frac}"].sort_values()
                med_idx = eefs.index[len(eefs) // 2]

                keys = [f"EEF_{frac}", f"EEF_{frac}_low", f"EEF_{frac}_high"]
                if frac == "max":
                    keys += ["frac_max", "N_max"]

                for key in keys:
                    cur_row[f"Median {key} ({split})"] = cur_results[key][med_idx]

            sig_targets = set()
            for i, row in cur_results.iterrows():
                target = row.Target
                for key in p_keys:
                    if row[key] < 0.05:
                        # ef_key = key.replace("_p_adj", "")
                        # print("significant", model, target, ef_key, row[ef_key], row[key])
                        sig_targets.add(target)
                        break

            cur_row[f"Significant Targets ({split})"] = len(sig_targets)
            model_sig_targets[(split, model)] = sig_targets

        summary_df.append(cur_row)

    summary_df = pd.DataFrame(summary_df)
    print(summary_df)

    for model, targets in model_sig_targets.items():
        print(f"Significant targets for {model}:")
        for target in targets:
            print(f"\t{target}")

    summary_df.to_csv("outputs/baseline_summary.csv", index=False)

    knn_order = []
    for targs in (split_targs["val"], split_targs["test"]):
        knn_order += list(all_results.query("Model == 'KNN' and Target in @targs").sort_values(by=f"EEF_max", ascending=False).Target)

    for i, row in summary_df.iterrows():
        row_str = f"{row.Model}"
        for split in ("val", "test"):
            frac = row[f'Median frac_max ({split})']
            med = row[f"Median EEF_max ({split})"]
            high = row[f"Median EEF_max_high ({split})"]
            low = row[f"Median EEF_max_low ({split})"]

            targets = split_targs[split]
            ef_10_targs = (all_results.query("Model == @row.Model and Target in @targets").EEF_max > 10).sum()

            # plus_err = row["Median EEF_max_high"] - row["Median EEF_max"]
            # minus_err = row["Median EEF_max"] - row["Median EEF_max_low"]
            # print(f"{row.Model} & ${med:.2f} \enskip \Vectorstack{{+{plus_err:.2f} -{minus_err:.2f}}} $ & {frac:.2f} \\\\")
            high_str = f"{high:.2g}" if high < 100 else f"{high:.3g}"
            med_str = f"{med:.1f}" if med < 10 else f"{med:.2g}"
            row_str += f" & {med_str}, CI = [{low:.2g}, {high_str}] & {frac*100:.2g}\% & {ef_10_targs}"
        row_str += " \\\\"
        print(row_str)

    # save the summary figures
    fracs = ["max"]
    for frac in fracs:
        model_order = ["KNN", "Vina", "GNINA", "Glide", "BANANA"]

        p_eef = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}").loc[knn_order][model_order]
        p_low = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}_low").loc[knn_order][model_order]
        p_high = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}_high").loc[knn_order][model_order]

        err = []
        for col in p_low:
            err.append([p_eef[col].values - p_low[col].values, p_high[col].values - p_eef[col].values])
        err = np.abs(err)

        frac_str = {
            0.1: "10\%",
            0.01: "1\%",
            0.001: "0.1\%",
            0.0001: "0.01\%",
            "max": "max",
        }[frac]

        fig, ax = plt.subplots(figsize=(8,1.75))
        p_eef.plot(kind='bar',yerr=err,ax=ax, capsize=2, rot=0, ylabel=f"$\\mathregular{{EEF}}_\\mathregular{{{frac_str}}}$")

        y_max = p_high.max().max()*1.05

        ax.axvline(3.5, color='k', linestyle='--', linewidth=1)
        ax.text(1.5, y_max, "Validation", ha='center', va='bottom', fontsize=10)
        ax.text(6, y_max, "Test", ha='center', va='bottom', fontsize=10)

        # remove top and right border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # force legend to be in the top right
        ax.legend(loc='upper right', bbox_to_anchor=(1.075, 1.0), fontsize=10)

        fig.savefig(f"outputs/baselines_{frac}.pdf", bbox_inches="tight", dpi=500)

    fracs = [0.1, 0.01, 0.001, 0.0001]
    fig, axes = plt.subplots(4, 1, figsize=(8,8))

    for ax, frac in zip(axes, fracs):
        first = frac == fracs[0]
        last = frac == fracs[-1]

        model_order = ["KNN", "Vina", "GNINA", "Glide", "BANANA"]

        p_eef = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}").loc[knn_order][model_order]
        p_low = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}_low").loc[knn_order][model_order]
        p_high = all_results.pivot(index="Target", columns="Model", values=f"EEF_{frac}_high").loc[knn_order][model_order]

        err = []
        for col in p_low:
            err.append([p_eef[col].values - p_low[col].values, p_high[col].values - p_eef[col].values])
        err = np.abs(err)

        frac_str = {
            0.1: "10\%",
            0.01: "1\%",
            0.001: "0.1\%",
            0.0001: "0.01\%",
            "max": "max",
        }[frac]

        # fig, ax = plt.subplots(figsize=(8,1.75))
        p_eef.plot(kind='bar',yerr=err,ax=ax, capsize=2, rot=0, ylabel=f"$\\mathregular{{EEF}}_\\mathregular{{{frac_str}}}$", legend=last, xlabel="" if not last else "Target", xticks=[] if not last else None)

        y_max = p_high.max().max()*1.05

        ax.axvline(3.5, color='k', linestyle='--', linewidth=1)

        if first:
            ax.text(1.5, y_max, "Validation", ha='center', va='bottom', fontsize=10)
            ax.text(6, y_max, "Test", ha='center', va='bottom', fontsize=10)

        # remove top and right border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    fig.tight_layout()
    fig.savefig(f"outputs/baselines_full.pdf", bbox_inches="tight", dpi=500)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    eval_results(cfg)