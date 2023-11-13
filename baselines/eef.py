
import numpy as np
import scipy
from scipy.stats import binomtest

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

def calc_eef(act_preds, rand_preds, activities, act_cutoff, select_frac):
    """" Computes expected enrichment factor (EEF) at a particular percentage.
    :param act_preds: predictions for the "active" compounds
        (though we reduce this number according to act_cutoff)
    :param rand_preds: predictions for the random compounds
    :param activities: the actual activities of the active compounds (pchembl_value)
    :param act_cutoff: the cutoff for what is considered an active compound (5 in the paper)
    :param select_frac: top fraction of compounds selected. E.g. 0.01 for EEF_1%
    :returns: Tuple of (EEF, lower bound, upper bound, p-value that EEF > 1)"""

    select_num = int((1 - select_frac) * len(rand_preds)) + 1

    select = sorted(rand_preds)[select_num-1]
    is_active = activities >= act_cutoff

    # We don't weigh each active according to the probability
    # of its activity (though this may be a good idea in the future )
    # alphas = A_dist.pdf(activities[is_active])
    # alphas = alphas/alphas.sum()
    # P_act = (alphas * (act_preds >= select)[is_active]).sum()

    P_rand = (rand_preds >= select).sum()/len(rand_preds)
    
    K = ((act_preds >= select) & is_active).sum()
    N = is_active.sum()
    P_act = K/N

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
        cur_frac = float((preds["random"] >= act).sum()/len(preds["random"]))
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