
from glob import glob
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets

from utils.cfg_utils import get_bayesbind_dir

class CutoffExpDist:
    """Exponential distribution for X > C, undefined otherwise"""

    def __init__(self, data, cutoff, scale=None, p_gt_c=None):
        if data is None:
            assert scale is not None and p_gt_c is not None
            self.cutoff = cutoff
            self.p_gt_c = p_gt_c
            self.logp_gt_c = np.log(self.p_gt_c)
            self.dist = scipy.stats.expon(0.0, scale)
        else:
            mask = data > cutoff
            self.cutoff = cutoff
            self.p_gt_c = mask.sum()/len(data)
            self.logp_gt_c = np.log(self.p_gt_c)
            loc, scale = scipy.stats.expon.fit(data[mask] - self.cutoff)
            print(loc, scale, cutoff, self.p_gt_c)
            self.dist = scipy.stats.expon(loc, scale)

    def pdf(self, x):
        return self.p_gt_c*self.dist.pdf(x - self.cutoff)

    def logpdf(self, x):
        return self.logp_gt_c + self.dist.logpdf(x - self.cutoff)

    def cdf(self, x):
        return (1-self.p_gt_c) + self.p_gt_c*self.dist.cdf(x - self.cutoff)

    def logcdf(self, x):
        raise NotImplementedError

A_dist = CutoffExpDist(None, 3.5, 0.3, 0.05)

def calc_eef(act_preds, rand_preds, activities, act_cutoff, select_num=None, select_frac=None):
    """" Computes expected enrichment factor (EEF)
    at a particular percentage. TODO: scale the probabilities
    by the activities. """

    # assert select_num is None
    # select_num = int((1 - select_frac) * len(rand_preds)) + 1
    if select_num is None:
        assert select_frac is not None
        select_num = int((1 - select_frac) * len(rand_preds)) + 1
    else:
        assert select_frac is None

    select = sorted(rand_preds)[select_num-1]
    is_active = activities >= act_cutoff

    # weigh each active according to the probability of its activity
    # alphas = A_dist.pdf(activities[is_active])
    # alphas = alphas/alphas.sum()

    P_rand = (rand_preds >= select).sum()/len(rand_preds)
    
    K = ((act_preds >= select) & is_active).sum()
    N = is_active.sum()
    P_act = K/N
    # P_act = (alphas * (act_preds >= select)[is_active]).sum()

    ef_hat = P_act/P_rand

    bounds = binomtest(K, N, P_rand, alternative='two-sided').proportion_ci(0.95, method='wilson')
    pval = binomtest(K, N, P_rand, alternative='greater').pvalue
    low = bounds.low/P_rand
    high = bounds.high/P_rand

    return ef_hat, low, high, pval

def calc_best_eef(preds, true_act, act_cutoff):
    """ Compute the best possible EEF from the predictions.
    Returns a tuple (EEF, low, high, fraction selected, N (1/fraction selection)) """

    cur_best = None
    seen_fracs = set()
    for act in preds["actives"]:
        cur_frac = (preds["random"] >= act).sum()/len(preds["random"])
        if cur_frac in seen_fracs:
            continue
        if cur_frac == 0:
            cur_frac = 1/len(preds["random"])

        seen_fracs.add(cur_frac)
        cur_N = int(round(1/cur_frac))
        eef, low, high, pval = calc_eef(preds["actives"], preds["random"], true_act, act_cutoff, select_frac=cur_frac)
        if cur_best is None or eef > cur_best[0]:
            cur_best = (eef, high, low, pval, cur_frac, cur_N)
    return cur_best


ignore_pockets = { "AL1A1_HUMAN_4_501_0", "ESR1_HUMAN_300_553_0", "CBP_HUMAN_1079_1197_0" }
_valid_indexes = None
def get_all_valid_indexes(cfg):
    """ Alas, we created the bayesbind set (and docked everything) before realizing
    that we need to remove all potency (HTS) values from the activities. So now
    we postprocess everything smh"""
    global _valid_indexes
    if _valid_indexes is not None:
        return _valid_indexes
    
    _valid_indexes = {}
    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        if pocket in ignore_pockets:
            continue
        df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/actives.csv")
        if len(df.query("standard_type != 'Potency' and pchembl_value > 5")) > 30:
            _valid_indexes[pocket] = df.query("standard_type != 'Potency'").index
    return _valid_indexes

def postprocess_preds(cfg, preds):
    """ Postprocess predictions on old (including potency) bayesbind set so that
    they now only predict non-potency values """
    valid_indexes = get_all_valid_indexes(cfg)
    ret = {}
    for pocket in preds:
        if pocket in valid_indexes:
            ret[pocket] = {}
            ret[pocket]["random"] = preds[pocket]["random"]
            ret[pocket]["actives"] = preds[pocket]["actives"][valid_indexes[pocket]]
    return ret

def get_all_bayesbind_activities(cfg):
    valid_indexes = get_all_valid_indexes(cfg)
    poc2activities = {}
    for folder in glob(get_bayesbind_dir(cfg) + "/*/*"):
        pocket = folder.split("/")[-1]
        if pocket in valid_indexes:
            poc2activities[pocket] = pd.read_csv(folder + "/actives.csv").pchembl_value[valid_indexes[pocket]]
    return poc2activities

def get_all_metrics(cfg, predictions, act_cutoff=5, N=None, frac=None):
    poc2activities = get_all_bayesbind_activities(cfg)
    ret = {}
    all_eefs = []
    for pocket, preds in predictions.items():
        eef, low, high = calc_eef(preds["actives"], preds["random"], poc2activities[pocket], act_cutoff, N, frac)
        all_eefs.append(eef)
        ret[f"{pocket}_EEF_{N}"] = eef
        ret[f"{pocket}_EEF_{N}_high"] = high
        ret[f"{pocket}_EEF_{N}_low"] = low
    ret[f"mean_EEF_{N}"] = np.mean(all_eefs)
    ret[f"median_EEF_{N}"] = np.median(all_eefs)
    return ret