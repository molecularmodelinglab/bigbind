from collections import defaultdict
from tqdm import tqdm

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
