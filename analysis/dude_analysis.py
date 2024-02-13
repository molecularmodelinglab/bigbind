
import random
import sys
import numpy as np
import pandas as pd

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
    return [f.name for f in cfg.host.dude_vs_folder.iterdir() if f.is_dir()]

def get_dude_scores(cfg, target, model):
    """ Returns the scores for the active and decoy ligands
    for a given target and model. Eg use newdefault_CNNscore
    for the default gnina model """
    
    default_file = f"{cfg.host.dude_vs_folder}/{target}/newdefault.summary"

    default_df = pd.read_csv(default_file, delim_whitespace=True)#, skiprows=1, names=["rank", "title", "vina", "target", "file", "gnina_affinity", "gnina_score"])

    mis_file = f"{cfg.host.dude_vs_folder}/{target}/missing.summary"
    mis_df = pd.read_csv(mis_file, delim_whitespace=True)
    dude_df = pd.concat([default_df, mis_df])

    # for some reason the vina scores are negative
    dude_df["Vina"] = -dude_df["Vina"]

    # max score per compound
    grouped_df = dude_df.groupby("Title")[model].max()

    active_mask = grouped_df.index.str.contains("CHEMBL")
    decoy_mask = grouped_df.index.str.contains("ZINC")
    assert np.all(decoy_mask == ~active_mask)

    active_scores = np.array(grouped_df[active_mask])
    decoy_scores = np.array(grouped_df[decoy_mask])

    random.shuffle(decoy_scores)
    random.shuffle(active_scores)

    return active_scores, decoy_scores


def get_gnina_dude_ef_df(cfg):
    """ Returns a dataframe containing the estimated EEFs and EFs
    for gnina on all the DUD-E targets """

    pass

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    print(get_all_dude_targets(cfg))