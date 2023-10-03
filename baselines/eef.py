
from glob import glob
import numpy as np
import pandas as pd

from utils.cfg_utils import get_bayesbind_dir


def calc_eef(act_preds, rand_preds, activities, act_cutoff, select_num, select_frac=None):
    """" Computes expected enrichment factor (EEF)
    at a particular percentage. TODO: scale the probabilities
    by the activities. """

    if select_num is None:
        assert select_frac is not None
        select_num = int(select_frac * len(rand_preds))
    else:
        assert select_frac is None

    select = sorted(rand_preds)[select_num]
    is_active = activities >= act_cutoff

    P_rand = (rand_preds >= select).sum()/len(rand_preds)
    P_act = ((act_preds >= select) & is_active).sum()/is_active.sum()

    return P_act/P_rand

def get_all_metrics(cfg, predictions):
    
    poc2activities = {}
    for folder in glob(get_bayesbind_dir(cfg) + "/*/*"):
        pocket = folder.split("/")[-1]
        poc2activities[pocket] = pd.read_csv(folder + "/actives.csv").pchembl_value
    
    act_cutoff = 5
    N = 19999
    ret = {}
    all_eefs = []
    for pocket, preds in predictions.items():
        eef = calc_eef(preds["actives"], preds["random"], poc2activities[pocket], act_cutoff, N)
        all_eefs.append(eef)
        ret[f"{pocket}_EEF_{N}"] = eef
    ret[f"mean_EEF_{N}"] = np.mean(all_eefs)
    ret[f"median_EEF_{N}"] = np.median(all_eefs)
    return ret