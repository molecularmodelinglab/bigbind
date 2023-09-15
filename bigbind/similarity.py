from collections import defaultdict
from tqdm import tqdm
import numpy as np

from bigbind.tanimoto_matrix import get_tanimoto_matrix, get_morgan_fps_parallel
from old.probis import convert_inter_results_to_json, get_rep_recs

class PocketSimilarity:
    """ Replace with TM score when those come in"""

    def __init__(self):
        cfg = {
            "bigbind_folder": "/home/boris/Data/BigBindV1",
            "cache_folder": "/home/boris/Data/BigBindCache",
            "cache": True,
            "recalc": [],
        }

        probis_scores = convert_inter_results_to_json(cfg)
        pocket2rep_rec = get_rep_recs(cfg)

        rep_rec2pocket = {}
        for poc, rec in pocket2rep_rec.items():
            rep_rec2pocket[rec] = poc

        self.poc2pocs = defaultdict(list)
        self.poc2scores = defaultdict(list)
        self.all_scores = {}

        for (r1, r2), score in probis_scores.items():
            p1 = rep_rec2pocket[r1]
            p2 = rep_rec2pocket[r2]
            if p1 == p2: continue

            self.all_scores[(p1, p2)] = score
            self.all_scores[(p2, p1)] = score
            self.poc2pocs[p1].append(p2)
            self.poc2scores[p1].append(score)
            self.poc2pocs[p2].append(p1)
            self.poc2scores[p2].append(score)

    def get_similarity(self, p1, p2):
        return self.all_scores.get((p1, p2), 0.0)

class LigSimilarity:

    def __init__(self, cfg):
        self.smi_list, _ = get_morgan_fps_parallel.get_output(cfg)
        self.tanimoto_mat = get_tanimoto_matrix.get_output(cfg)
        self.smi2idx = { smi: idx for idx, smi in enumerate(self.smi_list) }

    def get_similarity(self, smi1, smi2):
        try:
            i1 = self.smi2idx[smi1]
            i2 = self.smi2idx[smi2]
            col = self.tanimoto_mat.col[self.tanimoto_mat.row == i1]
            mask = col == i2
            if mask.sum() == 0: return 0.0
            return self.tanimoto_mat.data[self.tanimoto_mat.row == i1][mask][0]
        except KeyError:
            print("This shouldn't happen...")
            return 0.0

def get_edge_nums(df, tanimoto_mat, poc_sim, poc_indexes, tan_min, tan_max, probis_min, probis_max):
    """ Returns a tuple of the number of pairs satisfying both:
        tan_max > tanimoto_sim >= tan_min,
        probis_max > probis_sim >= probis_min
        and than the number of pairs satisfying only the first and only
        the second condition, respectively """

    tan_mask = np.logical_and(tanimoto_mat.data >= tan_min, tanimoto_mat.data < tan_max)
    tan_mask = np.logical_and(tan_mask, tanimoto_mat.row != tanimoto_mat.col)
    cur_tan_data = tanimoto_mat.data[tan_mask]
    cur_tan_row = tanimoto_mat.row[tan_mask]
    cur_tan_col = tanimoto_mat.col[tan_mask]

    # probis_max being none implied we're only looking at ligands
    # _within the same pocket_
    if probis_max is None:
        cur_poc2pocs = { poc: [ poc ] for poc in df.pocket.unique() }
    else:
        cur_poc2pocs = { poc: [ p for p, s in zip(poc_sim.poc2pocs[poc], poc_sim.poc2scores[poc]) if s < probis_max and s >= probis_min ] for poc in poc_sim.poc2pocs }
        
    both_edges = 0
    lig_edges = tan_mask.sum()//2
    rec_edges = 0

    seen = set()

    for p1, p2s in tqdm(cur_poc2pocs.items()):

        p1_idx = poc_indexes.get(p1, [])
        if len(p1_idx) == 0: continue

        p1_mask = np.in1d(cur_tan_row, p1_idx)
        # ct_row = cur_tan_row[p1_mask]
        ct_col = cur_tan_col[p1_mask]

        for p2 in p2s:
            if (p2, p1) in seen: continue
            seen.add((p1, p2))

            p2_idx = poc_indexes.get(p2, [])
            if len(p2_idx) == 0: continue
            rec_edges += len(p2_idx)

            p2_mask = np.in1d(ct_col, p2_idx)

            both_edges += p2_mask.sum()

    return both_edges, lig_edges, rec_edges
    