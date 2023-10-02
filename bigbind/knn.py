

from collections import defaultdict
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from bigbind.similarity import LigSimilarity
from utils.task import task


def get_optimal_lig_rec_coefs(tan_cutoffs, tm_cutoffs, prob_ratios):
    """ Fits a linear model to predict log(prob_ratio) from the ligand
    and rec similarities. We assume these are the optimal coefficients
    for a KNN model."""
    model = linear_model.LinearRegression()
    X = np.array([tan_cutoffs.reshape(-1), tm_cutoffs.reshape(-1)]).T
    Y = np.log(prob_ratios.reshape((-1, 1)))
    model.fit(X, Y)
    tan_coef, tm_coef = model.coef_[0]
    return tan_coef, tm_coef

def get_knn_preds(target_df, train_df, lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, tm_cutoffs, prob_ratios, K=1):
    """ Uses poc sim and lig sim to get predictions for the target_df
    using a (K=1) KNN """

    K = 1
    default_act = 3

    tan_coef, tm_coef = get_optimal_lig_rec_coefs(tan_cutoffs, tm_cutoffs, prob_ratios)

    lig_sim = LigSimilarity(lig_smi, lig_sim_mat)

    train_lig_idxs = set()
    lig_idx2train_idx = defaultdict(set)
    for i, smi in enumerate(train_df.lig_smiles):
        lig_idx = lig_sim.smi2idx[smi]
        train_lig_idxs.add(lig_idx)
        lig_idx2train_idx[lig_idx].add(i)

    targ_lig_idxs = set()
    lig_idx2targ_idx = defaultdict(set)
    for i, smi in enumerate(target_df.lig_smiles):
        lig_idx = lig_sim.smi2idx[smi]
        targ_lig_idxs.add(lig_idx)
        lig_idx2targ_idx[lig_idx].add(i)

    train_lig_idxs = np.array(list(train_lig_idxs))
    targ_lig_idxs = np.array(list(targ_lig_idxs))

    # find all the edges between the target and train ligands
    targ_mask = np.in1d(lig_sim_mat.row, targ_lig_idxs)
    train_mask = np.in1d(lig_sim_mat.col, train_lig_idxs)
    mask = targ_mask & train_mask

    cur_row = lig_sim_mat.row[mask]
    cur_col = lig_sim_mat.col[mask]
    cur_data = lig_sim_mat.data[mask]

    preds = np.zeros(len(target_df)) + default_act
    sims = np.zeros(len(target_df))

    # now actually get the predictions (only works for K = 1 rn)
    assert K == 1

    for i, j, tan_sim in zip(tqdm(cur_row), cur_col, cur_data):

        targ_idxs = lig_idx2targ_idx[i]
        train_idxs = lig_idx2train_idx[j]
        # expand to _all_ the edges between the target and train sets
        for targ_idx in targ_idxs:
            targ_poc = target_df.pocket[targ_idx]
            for train_idx in train_idxs:
                train_poc = train_df.pocket[train_idx]
                
                tm_sim = poc_sim.get_similarity(targ_poc, train_poc)
                total_sim = tan_coef * tan_sim + tm_coef * tm_sim

                if total_sim > sims[targ_idx]:
                    sims[targ_idx] = total_sim
                    preds[targ_idx] = train_df.pchembl_value[train_idx]
    return sims

@task(force=False)
def compare_probis_and_pocket_tm(cfg,
                                 tm_clusters,
                                 probis_clusters,
                                 split2act,
                                 split2sna,
                                 split2act_probis,
                                 split2sna_probis,
                                 lig_smi, 
                                 lig_sim_mat,
                                 poc_sim,
                                 poc_sim_probis,
                                 tan_cutoffs,
                                 tm_cutoffs,
                                 probis_cutoffs,
                                 prob_ratios_tm,
                                 prob_ratios_probis):
    """ Runs a KNN using both probis and pocket TM scores, using
    both probis and pocket TM to split the dataset """

    print("Number of ProBis clusters: ", len(probis_clusters))
    print("Number of TM clusters: ", len(tm_clusters))

    target_split = "val"
    # TM on TM splits
    tm_tm_preds = get_knn_preds(split2sna[target_split], split2act["train"], lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, tm_cutoffs, prob_ratios_tm)
    tm_tm_auc = roc_auc_score(split2sna[target_split].active, tm_tm_preds)
    print("AUC for TM on TM: ", tm_tm_auc)

    # TM on ProBis splits
    tm_probis_preds = get_knn_preds(split2sna_probis[target_split], split2act_probis["train"], lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, tm_cutoffs, prob_ratios_tm)
    tm_probis_auc = roc_auc_score(split2sna_probis[target_split].active, tm_probis_preds)
    print("AUC for TM on ProBis: ", tm_probis_auc)

    # ProBis on TM splits
    probis_tm_preds = get_knn_preds(split2sna[target_split], split2act["train"], lig_smi, lig_sim_mat, poc_sim_probis, tan_cutoffs, probis_cutoffs, prob_ratios_probis)
    probis_tm_auc = roc_auc_score(split2sna[target_split].active, probis_tm_preds)
    print("AUC for ProBis on TM: ", probis_tm_auc)
    
    # ProBis on ProBis splits
    probis_probis_preds = get_knn_preds(split2sna_probis[target_split], split2act_probis["train"], lig_smi, lig_sim_mat, poc_sim_probis, tan_cutoffs, probis_cutoffs, prob_ratios_probis)
    probis_probis_auc = roc_auc_score(split2sna_probis[target_split].active, probis_probis_preds)
    print("AUC for ProBis on ProBis: ", probis_probis_auc)

    return tm_tm_preds, tm_probis_preds, probis_tm_preds, probis_probis_preds