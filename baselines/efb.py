import numpy as np
import jax.numpy as jnp
import jax

def calc_efb(act_preds, rand_preds, select_frac, sorted=False):
    """" Computes EFB at a particular percentage.
    :param act_preds: predictions for the active compounds
    :param rand_preds: predictions for the random compounds
    :param select_frac: top fraction of compounds selected. E.g. 0.01 for EEF_1%
    :returns: EFB"""

    select_num = round((1 - select_frac) * len(rand_preds))
    if select_num > len(rand_preds) - 1:
        print("!")

    if sorted:
        select = rand_preds[select_num-1]
    else:
        select = np.sort(rand_preds)[select_num-1]

    P_rand = (rand_preds >= select).sum()/len(rand_preds)
    
    K = (act_preds >= select).sum()
    N = len(act_preds)
    P_act = K/N

    if P_act == 0:
        return 0.0

    efb = P_act/P_rand

    return efb

def calc_best_efb(act_preds, rand_preds, sorted=False):
    """ Compute the best possible EFB from the predictions. (EFB_max) """

    # only need to sort once
    rand_preds = np.sort(rand_preds)

    cur_best = None
    seen_fracs = set()

    # we don't need much precision here -- it takes a while
    # to get through everything smh
    cutoffs = np.unique(act_preds.astype(np.float16))

    for act in cutoffs:
        cur_frac = float((rand_preds >= act).sum()/len(rand_preds))
        if cur_frac in seen_fracs:
            continue
        if cur_frac == 0:
            cur_frac = 1/len(rand_preds)

        seen_fracs.add(cur_frac)
        efb = calc_efb(act_preds, rand_preds, cur_frac, sorted=True)

        if cur_best is None or efb > cur_best:
            cur_best = efb

    return cur_best