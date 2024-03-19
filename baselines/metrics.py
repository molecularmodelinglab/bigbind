
import math
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def calc_ef(active_scores, decoy_scores, select_frac):
    """ Calculate the enrichment factor (EF) for a given
    fraction of the dataset."""
    all_scores = np.concatenate([decoy_scores, active_scores])
    n_chosen = int(select_frac * len(all_scores))
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

def to_str(x, sigfigs):
    """ Converts a float to a string with a given number of significant figures, 
    without ever resorting to scientific notation """
    if x == 0.0:
        return "0." + "0"*(sigfigs-1)

    offset = math.ceil(math.log10(abs(x)))
    x = round(x, sigfigs-offset)
    if x >= 10**(sigfigs-1):
        dig = 10**offset
        ret = str(int(round(x/dig, sigfigs)*dig))
        if len(ret) == sigfigs and ret[-1] == "0":
            ret = ret + "."
        return ret
        
    ret = f"{x:.{sigfigs}}"
    if "." in ret:
        cur_sigfigs = len(ret) - 1
        if ret.startswith("0."):
            cur_sigfigs -= 1
        while cur_sigfigs < sigfigs:
            ret += "0"
            cur_sigfigs += 1
    return ret

def compute_bootstrap_metrics(inputs, metrics, **kwargs):
    """ Returns a dict that, for every key and
    metric in the metrics dict, has {key}, {key}_low,
    and {key}_high, where the latter two are the
    95% confidence interval of the metric computed
    by scipy.stats.bootstrap. """

    results = {}
    for key, metric in metrics.items():
        val = metric(*inputs)
        res = stats.bootstrap(inputs, metric, **kwargs)
        low = res.confidence_interval.low
        high = res.confidence_interval.high
        results[key] = val
        results[key + "_low"] = low
        results[key + "_high"] = high
    return results

def median_metric(metric, *cur_preds):
    """ Returns the median value of a metric; cur preds is a list of
    active scores interleaved with random scores. """
    all_act_preds = cur_preds[::2]
    all_rand_preds = cur_preds[1::2]
    metric_vals = []
    for act_preds, rand_preds in zip(all_act_preds, all_rand_preds):
        metric_vals.append(metric(act_preds, rand_preds))
    return np.median(metric_vals)


def roc_auc(act_preds, rand_preds):
    """ AUC for separate active and random predictions."""
    all_preds = np.concatenate([act_preds, rand_preds])
    all_labels = np.concatenate([np.ones_like(act_preds), np.zeros_like(rand_preds)])
    return roc_auc_score(all_labels, all_preds)