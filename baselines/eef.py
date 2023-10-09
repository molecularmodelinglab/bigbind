
from glob import glob
import numpy as np
import pandas as pd
import scipy

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
    P_act = ((act_preds >= select) & is_active).sum()/is_active.sum()
    # P_act = (alphas * (act_preds >= select)[is_active]).sum()

    return P_act/P_rand

def get_all_metrics(cfg, predictions, act_cutoff=5, N=10000):
    
    poc2activities = {}
    for folder in glob(get_bayesbind_dir(cfg) + "/*/*"):
        pocket = folder.split("/")[-1]
        poc2activities[pocket] = pd.read_csv(folder + "/actives.csv").pchembl_value
    
    ret = {}
    all_eefs = []
    for pocket, preds in predictions.items():
        eef = calc_eef(preds["actives"], preds["random"], poc2activities[pocket], act_cutoff, N)
        all_eefs.append(eef)
        ret[f"{pocket}_EEF_{N}"] = eef
    ret[f"mean_EEF_{N}"] = np.mean(all_eefs)
    ret[f"median_EEF_{N}"] = np.median(all_eefs)
    return ret