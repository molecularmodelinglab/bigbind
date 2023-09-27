from collections import defaultdict
import os
import random
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import networkx as nx

from utils.cache import cache
from bigbind.tanimoto_matrix import get_tanimoto_matrix, get_morgan_fps_parallel
from old.probis import convert_inter_results_to_json, get_rep_recs
from utils.cfg_utils import get_figure_dir
from utils.task import iter_task, simple_task, task


class PocketSimilarityProbis:
    """Replace with TM score when those come in"""

    def __init__(self, probis_scores, pocket2rep_rec):
        # cfg = {
        #     "bigbind_folder": "/home/boris/Data/BigBindV1",
        #     "cache_folder": "/home/boris/Data/BigBindCache",
        #     "cache": True,
        #     "recalc": [],
        # }

        # probis_scores = convert_inter_results_to_json(cfg)
        # pocket2rep_rec = get_rep_recs(cfg)

        rep_rec2pocket = {}
        for poc, rec in pocket2rep_rec.items():
            rep_rec2pocket[rec] = poc

        self.poc2pocs = defaultdict(list)
        self.poc2scores = defaultdict(list)
        self.all_scores = {}

        for (r1, r2), score in probis_scores.items():
            p1 = rep_rec2pocket[r1]
            p2 = rep_rec2pocket[r2]
            # if p1 == p2: continue

            self.all_scores[(p1, p2)] = score
            self.all_scores[(p2, p1)] = score
            # self.poc2pocs[p1].append(p2)
            # self.poc2scores[p1].append(score)
            # self.poc2pocs[p2].append(p1)
            # self.poc2scores[p2].append(score)

        seen = set()
        for (p1, p2), score in tqdm(self.all_scores.items()):
            if (p2, p1) in seen:
                continue
            seen.add((p1, p2))
            self.poc2pocs[p1].append(p2)
            self.poc2scores[p1].append(score)
            if p1 != p2:
                self.poc2pocs[p2].append(p1)
                self.poc2scores[p2].append(score)

    def get_similarity(self, p1, p2):
        return self.all_scores.get((p1, p2), 0.0)


class PocketSimilarityTM:
    """Replace with TM score when those come in"""

    def __init__(self, valid_scores):
        self.poc2pocs = defaultdict(list)
        self.poc2scores = defaultdict(list)
        self.all_scores = {}

        for (r1, r2), score in tqdm(valid_scores.items()):
            p1 = r1.split("/")[0]
            p2 = r2.split("/")[0]
            # if p1 == p2: continue

            # take the max score
            if (p1, p2) in self.all_scores and self.all_scores[(p1, p2)] > score:
                continue

            self.all_scores[(p1, p2)] = score
            self.all_scores[(p2, p1)] = score

        seen = set()
        for (p1, p2), score in tqdm(self.all_scores.items()):
            if (p2, p1) in seen:
                continue
            seen.add((p1, p2))
            self.poc2pocs[p1].append(p2)
            self.poc2scores[p1].append(score)
            if p1 != p2:
                self.poc2pocs[p2].append(p1)
                self.poc2scores[p2].append(score)

    def get_similarity(self, p1, p2):
        score = self.all_scores.get((p1, p2), 0.0)
        if np.isnan(score):
            return 0.0
        return score


class LigSimilarity:
    def __init__(self, lig_smi, lig_sim_mat):
        self.smi_list = lig_smi
        self.tanimoto_mat = lig_sim_mat
        # self.smi_list, _ = get_morgan_fps_parallel.get_output(cfg)
        # self.tanimoto_mat = get_tanimoto_matrix.get_output(cfg)
        self.smi2idx = {smi: idx for idx, smi in enumerate(self.smi_list)}

    def get_similarity(self, smi1, smi2):
        try:
            i1 = self.smi2idx[smi1]
            i2 = self.smi2idx[smi2]
            col = self.tanimoto_mat.col[self.tanimoto_mat.row == i1]
            mask = col == i2
            if mask.sum() == 0:
                return 0.0
            return self.tanimoto_mat.data[self.tanimoto_mat.row == i1][mask][0]
        except KeyError:
            print("This shouldn't happen...")
            return 0.0

    def get_nx_graph(self, smi_list, tan_cutoff=0.7):
        """Returns dict mapping smiles to neighbor smiles"""
        idxs = np.array([self.smi2idx[smi] for smi in smi_list])
        mask = np.logical_and(
            np.in1d(self.tanimoto_mat.row, idxs), np.in1d(self.tanimoto_mat.col, idxs)
        )
        cur_row = self.tanimoto_mat.row[mask]
        cur_col = self.tanimoto_mat.col[mask]

        graph = nx.Graph()
        for row, col in zip(cur_row, cur_col):
            graph.add_edge(self.smi_list[row], self.smi_list[col])

        return graph
    
    def get_edge_dict(self, smi_list):
        """ Returns dict mapping smiles to neighbor smiles"""
        idxs = np.array([ self.smi2idx[smi] for smi in smi_list])
        mask = np.logical_and(np.in1d(self.tanimoto_mat.row, idxs), np.in1d(self.tanimoto_mat.col, idxs))
        cur_row = self.tanimoto_mat.row[mask]
        cur_col = self.tanimoto_mat.col[mask]

        edge_dict = defaultdict(set)
        print("Getting edge dict")
        for row, col in zip(tqdm(cur_row), cur_col):
            smi1 = self.smi_list[row]
            smi2 = self.smi_list[col]
            if smi1 == smi2: continue
            edge_dict[smi1].add(smi2)
            edge_dict[smi2].add(smi1)

        return edge_dict

    
    def make_diverse_set(self, smi_list, max_size=None, edge_dict=None):
        """ Greedly find a diverse subset of smiles_list """
        if edge_dict is None:
            edge_dict = self.get_edge_dict(smi_list)
        diverse_set = set()
        to_search = set(smi_list)
        while len(to_search):
            smi = next(iter(to_search))
            diverse_set.add(smi)
            if max_size is not None and len(diverse_set) == max_size:
                break
            to_search.remove(smi)
            to_search -= edge_dict[smi]

        return list(diverse_set)


@task(max_runtime=0.2, force=False)
def get_pocket_indexes(cfg, activities):
    """Returns a dictionary mapping pockets to the indexes of all
    rows in the activities dataframe with the pocket"""
    poc_indexes = {}
    for p1 in tqdm(activities.pocket.unique()):
        poc_indexes[p1] = np.array(activities.index[activities.pocket == p1], dtype=int)
    return poc_indexes


# force this!
@task(max_runtime=0.2, force=False)
def get_pocket_similarity(cfg, pocket_tm_scores):
    return PocketSimilarityTM(pocket_tm_scores)

@task(max_runtime=0.2, force=True)
def get_pocket_similarity_probis(cfg, probis_scores, pocket2rep_rec):
    return PocketSimilarityProbis(probis_scores, pocket2rep_rec)

def get_edge_nums(
    tanimoto_mat, poc_sim, poc_indexes, tan_min, tan_max, probis_min, probis_max
):
    """Returns a tuple of the number of pairs satisfying both:
    tan_max > tanimoto_sim >= tan_min,
    probis_max > probis_sim >= probis_min
    and than the number of pairs satisfying only the first and only
    the second condition, respectively"""

    tan_mask = np.logical_and(tanimoto_mat.data >= tan_min, tanimoto_mat.data < tan_max)
    tan_mask = np.logical_and(tan_mask, tanimoto_mat.row != tanimoto_mat.col)
    # cur_tan_data = tanimoto_mat.data[tan_mask]
    cur_tan_row = tanimoto_mat.row[tan_mask]
    cur_tan_col = tanimoto_mat.col[tan_mask]

    # probis_max being none implied we're only looking at ligands
    # _within the same pocket_
    if probis_max is None:
        cur_poc2pocs = {poc: [poc] for poc in poc_sim.poc2pocs.keys()}
    elif probis_max == 0.0:
        all_pocs = set(poc_sim.poc2pocs.keys())
        cur_poc2pocs = {
            poc: list(all_pocs - set(poc_sim.poc2pocs[poc])) for poc in poc_sim.poc2pocs
        }
    else:
        cur_poc2pocs = {
            poc: [
                p
                for p, s in zip(poc_sim.poc2pocs[poc], poc_sim.poc2scores[poc])
                if s < probis_max and s >= probis_min
            ]
            for poc in poc_sim.poc2pocs
        }

    both_edges = 0
    lig_edges = tan_mask.sum() // 2
    rec_edges = 0

    seen = set()

    for p1, p2s in tqdm(cur_poc2pocs.items()):
        # print(tan_min, tan_max, probis_min, probis_max, p1)

        p1_idx = poc_indexes.get(p1, [])
        if len(p1_idx) == 0:
            continue

        p1_mask = np.in1d(cur_tan_row, p1_idx)
        # ct_row = cur_tan_row[p1_mask]
        ct_col = cur_tan_col[p1_mask]

        for p2 in p2s:
            if (p2, p1) in seen:
                continue
            seen.add((p1, p2))

            p2_idx = poc_indexes.get(p2, [])
            if len(p2_idx) == 0:
                continue
            if p1 == p2:
                rec_edges += (len(p1_idx) ** 2) // 2
            else:
                rec_edges += len(p2_idx) * len(p1_idx)

            p2_mask = np.in1d(ct_col, p2_idx)

            both_edges += p2_mask.sum()

    return both_edges, lig_edges, rec_edges


def compute_edge_nums(cfg, args):
    return get_edge_nums(*args)

# force this!
compute_all_edge_nums = iter_task(56, 1, force=False)(compute_edge_nums)

def compute_edge_nums_probis(cfg, args):
    return get_edge_nums(*args)
compute_all_edge_nums_probis = iter_task(56, 1, force=True)(compute_edge_nums_probis)

num_tan = 5
num_tm = 15
@simple_task
def get_edge_num_inputs(cfg, full_lig_sim_mat, poc_sim, poc_indexes):
    """Returns a list of arguments to be passed to compute_all_edge_nums"""
    tan_cutoffs = np.linspace(0.4, 1.0, num_tan + 1)
    tm_cutoffs = np.array([0.0] + list(np.linspace(0.0, 1.0, num_tm)))
    arg_list = [
        (full_lig_sim_mat, poc_sim, poc_indexes, t1, t2, p1, p2)
        for t1, t2 in zip(tan_cutoffs, tan_cutoffs[1:])
        for p1, p2 in zip(tm_cutoffs, tm_cutoffs[1:])
    ]
    return arg_list

@simple_task
def get_edge_num_inputs_probis(cfg, full_lig_sim_mat, poc_sim, poc_indexes):
    """Returns a list of arguments to be passed to compute_all_edge_nums"""
    tan_cutoffs = np.linspace(0.4, 1.0, num_tan + 1)
    tm_cutoffs = np.linspace(2.0, 4.0, num_tm + 1)
    arg_list = [
        (full_lig_sim_mat, poc_sim, poc_indexes, t1, t2, p1, p2)
        for t1, t2 in zip(tan_cutoffs, tan_cutoffs[1:])
        for p1, p2 in zip(tm_cutoffs, tm_cutoffs[1:])
    ]
    return arg_list

@simple_task
def postproc_prob_ratios(cfg, edge_results, activities, arg_list):
    shape = (num_tan, num_tm)

    print(edge_results)

    tans = np.array([0.5 * (t1 + t2) for *rest, t1, t2, p1, p2 in arg_list]).reshape(
        shape
    )
    tms = np.array([0.5 * (p1 + p2) for *rest, t1, t2, p1, p2 in arg_list]).reshape(
        shape
    )

    possible_edges = (len(activities) ** 2) // 2
    prob_ratios = []
    for both_edges, lig_edges, rec_edges in edge_results:
        p_both = both_edges / possible_edges
        p_rec = rec_edges / possible_edges
        p_lig = lig_edges / possible_edges
        ratio = p_both / (p_rec * p_lig)
        prob_ratios.append(ratio)

    prob_ratios = np.array(prob_ratios).reshape(shape)
    return tans, tms, prob_ratios

postproc_prob_ratios.num_outputs = 3

@simple_task
def postproc_prob_ratios_probis(cfg, edge_results, activities, arg_list):
    shape = (num_tan, num_tm)

    print(edge_results)

    tans = np.array([0.5 * (t1 + t2) for *rest, t1, t2, p1, p2 in arg_list]).reshape(
        shape
    )
    tms = np.array([0.5 * (p1 + p2) for *rest, t1, t2, p1, p2 in arg_list]).reshape(
        shape
    )

    possible_edges = (len(activities) ** 2) // 2
    prob_ratios = []
    for both_edges, lig_edges, rec_edges in edge_results:
        p_both = both_edges / possible_edges
        p_rec = rec_edges / possible_edges
        p_lig = lig_edges / possible_edges
        ratio = p_both / (p_rec * p_lig)
        prob_ratios.append(ratio)

    prob_ratios = np.array(prob_ratios).reshape(shape)
    return tans, tms, prob_ratios

postproc_prob_ratios_probis.num_outputs = 3

def get_lig_rec_edge_prob_ratios(activities, full_lig_sim_mat, poc_sim, poc_indexes):
    """Returns a tuple of (tanimoto_cutoffs, probis_cutoffs, prob_ratios)"""

    arg_list = get_edge_num_inputs(full_lig_sim_mat, poc_sim, poc_indexes)
    results = compute_all_edge_nums(arg_list)
    prob_ratios = postproc_prob_ratios(results, activities, arg_list)

    return prob_ratios

def get_lig_rec_edge_prob_ratios_probis(activities, full_lig_sim_mat, poc_sim, poc_indexes):
    """Returns a tuple of (tanimoto_cutoffs, probis_cutoffs, prob_ratios)"""

    arg_list = get_edge_num_inputs_probis(full_lig_sim_mat, poc_sim, poc_indexes)
    results = compute_all_edge_nums_probis(arg_list)
    prob_ratios = postproc_prob_ratios_probis(results, activities, arg_list)

    return prob_ratios

# force this!
@task(force=False)
def plot_prob_ratios(cfg, tans, tms, prob_ratios):

    print("tans", tans)
    print("tms", tms)
    print("prob_ratios", prob_ratios)

    fig, ax = plt.subplots()
    contour = ax.contourf(tans, tms, prob_ratios)
    fig.colorbar(contour, ax=ax)
    ax.set_title("P(L,R)/(P(L)*P(R))")
    ax.set_xlabel("L (Tanimoto similarity)")
    ax.set_ylabel("R (Pocket TM score)")

    fname = os.path.join(get_figure_dir(cfg), "prob_ratios.png")
    print("Saving figure to", fname)
    fig.savefig(fname)

@task(force=True)
def plot_prob_ratios_probis(cfg, tans, tms, prob_ratios):

    print("tans", tans)
    print("probis scores", tms)
    print("prob_ratios", prob_ratios)

    fig, ax = plt.subplots()
    contour = ax.contourf(tans, tms, prob_ratios)
    fig.colorbar(contour, ax=ax)
    ax.set_title("P(L,R)/(P(L)*P(R))")
    ax.set_xlabel("L (Tanimoto similarity)")
    ax.set_ylabel("R (Probis score)")

    fname = os.path.join(get_figure_dir(cfg), "prob_ratios_probis.png")
    print("Saving figure to", fname)
    fig.savefig(fname)


# force this!
@task(num_outputs=2, force=False)
def get_pocket_clusters(cfg, activities, tms, prob_ratios, poc_sim, poc_indexes, cutoff_ratio=1.5):
    """Finds the optimal TM cutoff and clusters the pockets according
    to this cutoff -- two pockets are in the same cluster if their TM
    score is above the cutoff. Returns a tuple of (cutoff, clusters)"""

    # compute optimal TM cutoff
    cutoff_idx = num_tm - 1
    for ratio in reversed(prob_ratios.max(axis=0)):
        if ratio < cutoff_ratio:
            break
        cutoff_idx -= 1
        
    cutoff = tms[0, cutoff_idx]
    print("Optimal TM cutoff:", cutoff)

    # now let's find the pocket clusters
    G = nx.Graph()
    for poc in activities.pocket.unique():
        G.add_node(poc)

    for (p1, p2), score in poc_sim.all_scores.items():
        if score > cutoff:
            G.add_edge(p1, p2)

    clusters = list(nx.connected_components(G))
    print("Found", len(clusters), "clusters")

    biggest_cluster = list(sorted(clusters, key=lambda x: -len(x)))[0]

    num_idxs = 0 
    for poc in biggest_cluster:
        if poc in poc_indexes:
            num_idxs += len(poc_indexes[poc])

    print("Biggest cluster has", len(biggest_cluster), "pockets and", num_idxs, "datapoints")

    return cutoff, clusters


@simple_task
def get_tan_cluster_inputs(cfg, full_lig_sim_mat, tms, prob_ratios, poc_sim, poc_indexes):
    
    # compute optimal TM cutoff
    cutoff_idx = 0
    for ratio in prob_ratios.max(axis=0):
        if ratio > 1.0:
            continue
        cutoff_idx += 1
    cutoff = tms[0, cutoff_idx]
    print("Optimal TM cutoff:", cutoff)

    args = []
    for poc in poc_indexes:
        args.append((poc, poc_sim, cutoff, full_lig_sim_mat, poc_indexes))

    random.shuffle(args)

    return args

def get_tan_cluster_edges(cfg, arg):
    """ Creates an edge between two pockets if their TM score is above the high cutoff
    OR their tanimoto similarity is above the tan cutoff and their poc simiilarity is
    above the low cutoff """

    p1, poc_sim, low_cutoff, full_lig_sim_mat, poc_indexes = arg

    high_cutoff = 0.9
    tan_cutoff = 0.4
    assert tan_cutoff == 0.4

    edges = []
    p1_mask = np.in1d(full_lig_sim_mat.row, poc_indexes[p1])

    for p2, score in zip(poc_sim.poc2pocs[p1], poc_sim.poc2scores[p1]):
        if p2 in poc_indexes and score > low_cutoff:
            if score > high_cutoff:
                edges.append((p1, p2))
            else:
                p2_mask = np.in1d(full_lig_sim_mat.row, poc_indexes[p2])
                both_mask = p1_mask & p2_mask
                if both_mask.sum() > 0:
                    edges.append((p1, p2))

    return edges

# force this!
get_all_tan_cluster_edges = iter_task(4, 1, force=False)(get_tan_cluster_edges)

@simple_task
def postproc_tan_cluster_edges(cfg, edge_results, poc_indexes):

    # now let's find the pocket clusters
    G = nx.Graph()
    for poc in poc_indexes:
        G.add_node(poc)

    for edges in edge_results:
        for p1, p2 in edges:
            G.add_edge(p1, p2)

    clusters = list(nx.connected_components(G))
    print("Found", len(clusters), "clusters (w/ tanimoto))")

    biggest_cluster = list(sorted(clusters, key=lambda x: -len(x)))[0]

    num_idxs = 0 
    for poc in biggest_cluster:
        if poc in poc_indexes:
            num_idxs += len(poc_indexes[poc])

    print("Biggest cluster has", len(biggest_cluster), "pockets and", num_idxs, "datapoints (w/ tanimoto)")

    return clusters

def get_pocket_clusters_with_tanimoto(
    full_lig_sim_mat, tms, prob_ratios, poc_sim, poc_indexes
):
    inputs = get_tan_cluster_inputs(full_lig_sim_mat, tms, prob_ratios, poc_sim, poc_indexes)
    results = get_all_tan_cluster_edges(inputs)
    return postproc_tan_cluster_edges(results, poc_indexes)

# @cache(lambda cfg: "")
# def get_early_poc_tm_sims(cfg):
#     import yaml
#     import numpy as np
#     import pandas as pd
#     import scipy
#     from glob import glob
#     import random
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm, trange
#     from collections import defaultdict
#     from functools import lru_cache
#     import pickle
#     from sklearn.preprocessing import PolynomialFeatures
#     from sklearn import linear_model
#     import networkx as nx
#     from multiprocessing import Pool, shared_memory

#     from array import array
#     from utils.cache import cache
#     from utils.cfg_utils import get_config
#     from bigbind.pocket_tm_score import (
#         get_alpha_and_beta_coords,
#         get_all_pocket_tm_scores,
#         get_struct,
#     )
#     from bigbind.bigbind import make_bigbind_workflow

#     workflow = make_bigbind_workflow()

#     node = workflow.find_node("load_act_unfiltered")[0]
#     activities = workflow.run_node(cfg, node)

#     node = workflow.find_node("get_tanimoto_matrix")[0]
#     lig_sim_mat = workflow.run_node(cfg, node)

#     node = workflow.find_node("get_crossdocked_maps")[0]
#     (
#         uniprot2rfs,
#         uniprot2lfs,
#         uniprot2pockets,
#         pocket2uniprots,
#         pocket2rfs_nofix,
#         pocket2lfs,
#     ) = workflow.run_node(cfg, node)

#     node = workflow.find_node("untar_crossdocked")[0]
#     cd_dir = workflow.run_node(cfg, node)

#     @cache(lambda cfg: "", disable=False)
#     def get_valid_rfs(cfg):
#         valid_rfs = set()
#         for uniprot, rfs in tqdm(uniprot2rfs.items()):
#             for rf in rfs:
#                 full_rf = os.path.join(cd_dir, rf)
#                 try:
#                     get_alpha_and_beta_coords(get_struct(full_rf))
#                     valid_rfs.add(rf)
#                 except KeyError:
#                     pass
#         return valid_rfs

#     valid_rfs = get_valid_rfs(cfg)

#     @cache(lambda cfg: "", disable=False)
#     def get_valid_act(cfg):
#         valid_uniprots = set()
#         for uniprot, rfs in uniprot2rfs.items():
#             for rf in rfs:
#                 if rf in valid_rfs:
#                     valid_uniprots.add(uniprot)
#                     break
#         return activities.query("protein_accession in @valid_uniprots").reset_index(
#             drop=True
#         )

#     valid_act = get_valid_act(cfg)

#     og_dir = "/home/boris/Data/BigBindScratch/test/global/"
#     with open(f"{og_dir}/save_all_pockets/output.pkl", "rb") as f:
#         og_rec2pocfile, _ = pickle.load(f)

#     idx2rf = {}
#     rf2idx = {}
#     valid_indexes = set()
#     for i, rf in enumerate(og_rec2pocfile.keys()):
#         rf = "/".join(rf.split("/")[-2:])
#         idx2rf[i] = rf
#         rf2idx[rf] = i
#         if rf in valid_rfs:
#             valid_indexes.add(i)

#     @cache(lambda cfg: "", disable=False)
#     def get_valid_scores(cfg):
#         valid_scores = {}
#         for fname in tqdm(glob(f"{og_dir}compute_rec_tm_score_*/output.pkl")):
#             with open(fname, "rb") as f:
#                 res = pickle.load(f)
#                 for item in res:
#                     for (i1, i2), score in item.items():
#                         if i1 in valid_indexes and i2 in valid_indexes:
#                             valid_scores[idx2rf[i1], idx2rf[i2]] = score
#                 del res
#         return valid_scores

#     valid_scores = get_valid_scores(cfg)

#     @cache(lambda cfg: "", disable=False)
#     def get_poc_indexes(cfg):
#         ret = {}
#         for rf in tqdm(valid_rfs):
#             poc = rf.split("/")[0]
#             uniprots = pocket2uniprots[poc]
#             poc_indexes = activities.query("protein_accession in @uniprots").index
#             ret[poc] = poc_indexes
#         return ret

#     poc_indexes = get_poc_indexes(cfg)

#     smi_list = list(activities.canonical_smiles.unique())
#     smi2idx = {smi: idx for idx, smi in enumerate(smi_list)}
#     idx2act_idx = defaultdict(set)
#     for act_idx, smi in enumerate(activities.canonical_smiles):
#         idx = smi2idx[smi]
#         idx2act_idx[idx].add(act_idx)

#     # make new tanimoto matrix indexes by activities_unfiltered, not ligand id
#     @cache(lambda cfg: "")
#     def get_new_lig_sim(cfg):
#         smi_list = list(activities.canonical_smiles.unique())
#         smi2idx = {smi: idx for idx, smi in enumerate(smi_list)}
#         idx2act_idx = defaultdict(set)
#         for act_idx, smi in enumerate(activities.canonical_smiles):
#             idx = smi2idx[smi]
#             idx2act_idx[idx].add(act_idx)

#         new_row = array("I")
#         new_col = array("I")
#         new_data = array("f")
#         for i, j, data in zip(tqdm(lig_sim_mat.row), lig_sim_mat.col, lig_sim_mat.data):
#             for i2 in idx2act_idx[i]:
#                 for j2 in idx2act_idx[j]:
#                     new_row.append(i)
#                     new_col.append(j)
#                     new_data.append(data)
#         return np.array(new_row), np.array(new_col), np.array(new_data)

#     new_row, new_col, new_data = get_new_lig_sim(cfg)

#     new_tan_mat = scipy.sparse.coo_array(
#         (new_data, (new_row, new_col)), shape=lig_sim_mat.shape, copy=False
#     )

#     poc_sim = PocketSimilarityTM(valid_scores)

#     return new_tan_mat, poc_sim, poc_indexes, len(valid_act)


# if __name__ == "__main__":
#     import sys
#     from multiprocessing import Pool
#     from utils.cfg_utils import get_config

#     cfg_name = sys.argv[1]
#     pool_size = int(sys.argv[2])

#     cfg = get_config(cfg_name)
#     new_tan_mat, poc_sim, poc_indexes, valid_act_len = get_early_poc_tm_sims(cfg)

#     def f(args):
#         # print("Running", args)
#         return get_edge_nums(new_tan_mat, poc_sim, poc_indexes, *args)

#     num_tan = 8
#     tan_cutoffs = np.linspace(0.4, 1.0, num_tan + 1)
#     num_prob = 12
#     tm_cutoffs = [
#         0.0,
#         0.0,
#         0.2,
#         0.4,
#         0.6,
#         0.8,
#         1.0,
#     ]  # np.linspace(0.0, 1.0, num_prob+1)
#     arg_list = [
#         (t1, t2, p1, p2)
#         for t1, t2 in zip(tan_cutoffs, tan_cutoffs[1:])
#         for p1, p2 in zip(tm_cutoffs, tm_cutoffs[1:])
#     ]
#     # zero_result = f(arg_list[0])
#     # f(arg_list[10])
#     # results = list(map(f, arg_list))
#     with Pool(pool_size) as p:
#         results = list(p.imap(f, arg_list))

#     print("results")
#     print(results)
#     print("tan_cutoffs")
#     print(list(tan_cutoffs))
#     print("tm_cutoffs")
#     print(list(tm_cutoffs))
