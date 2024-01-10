
import os
import shutil
import sys
import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from baselines.evaluate import get_all_glide_scores_struct
from baselines.vina_gnina import get_all_bayesbind_struct_splits_and_pockets
from bigbind.tanimoto_matrix import batch_tanimoto, get_morgan_fps, get_tanimoto_matrix_impl
from utils.cfg_utils import get_baseline_dir, get_bayesbind_small_dir, get_bayesbind_struct_dir, get_config, get_output_dir
from utils.workflow import Workflow

ACT_CUTOFF = 5
NUM_ACT = 4
NUM_RAND = 16
RAND_DOCK_METHOD = "glide"
def make_bayesbind_small_dir(cfg, split, pocket, scores):
    out_folder = get_bayesbind_small_dir(cfg) + f"/{pocket}"
    bb_folder = get_bayesbind_struct_dir(cfg) + f"/{split}/{pocket}"

    df = pd.read_csv(bb_folder + "/actives.csv")
    # ensure that glide successfully docked all of these
    # and that the pchembl value is above the cutoff
    mask = np.logical_and(df.pchembl_value > ACT_CUTOFF, scores > -1000)
    df = df.loc[mask]# .iloc[:NUM_ACT]

    # get tanimoto similarity matrix
    fps = get_morgan_fps(cfg, df)
    tan_rows = []
    for i in range(len(fps)):
        tan_rows.append(batch_tanimoto(fps[i], fps))
    tan_mat = np.vstack(tan_rows)

    # find the min tanimoto cutoff such that
    # we can find at least NUM_ACT clusters
    cutoffs = np.linspace(0.1, 1, 30)
    for cutoff in cutoffs:
        G = nx.Graph()
        for i in range(tan_mat.shape[0]):
            G.add_node(i)
        for i in range(tan_mat.shape[0]):
            for j in range(i+1, tan_mat.shape[1]):
                if tan_mat[i, j] > cutoff:
                    G.add_edge(i, j)
        components = list(nx.connected_components(G))
        if len(components) >= NUM_ACT:
            break
    else:
        # print(f"Could not find a cutoff that would give at least {NUM_ACT} clusters")
        return False

    os.makedirs(out_folder, exist_ok=True)
    print(f"Making {out_folder}")

    shutil.copy(bb_folder + "/rec.pdb", out_folder + "/rec.pdb")
    shutil.copy(bb_folder + "/rec_hs.pdb", out_folder + "/rec_hs.pdb")


    # choose a single ligand from each cluster (up to NUM_ACT)
    chosen = []
    for component in components[:NUM_ACT]:
        chosen.append(list(component)[0])
    chosen = np.array(chosen)
    df = df.iloc[chosen]
    
    df.to_csv(out_folder + "/actives.csv", index=False)

    # save indexes of actives
    np.savetxt(out_folder + "/actives_indexes.txt", df.index.values, fmt="%d")

    writer = Chem.SDWriter(out_folder + "/actives.sdf")
    for lig_file in df.lig_crystal_file:
        lf = get_output_dir(cfg) + f"/{pocket}/{lig_file}"
        mol = Chem.SDMolSupplier(lf)[0]
        writer.write(mol)
    writer.close()

    rand_folder = get_baseline_dir(cfg, RAND_DOCK_METHOD, split, pocket) + "/random_results"
    rand_mols = []
    for i in range(10000):
        lf = rand_folder + f"/{i}.sdf"
        if os.path.exists(lf):
            mol = Chem.SDMolSupplier(lf)[0]
            rand_mols.append(mol)
        if len(rand_mols) >= NUM_RAND:
            break

    writer = Chem.SDWriter(out_folder + "/random.sdf")
    for mol in rand_mols:
        writer.write(mol)
    writer.close()

    return True

CHOSEN_POCKETS = [
    "RORG_HUMAN_262_508_0",
    "CBP_HUMAN_1079_1197_0",
    "TTK_HUMAN_516_808_0"
]
def make_all_bayesbind_small(cfg):
    scores = Workflow(cfg, {
        "glide": get_all_glide_scores_struct()
    }[RAND_DOCK_METHOD]).run()[0]
    for split, pocket in get_all_bayesbind_struct_splits_and_pockets(cfg):
        if pocket not in CHOSEN_POCKETS:
            continue
        assert make_bayesbind_small_dir(cfg, split, pocket, np.array(scores[pocket]["actives"]))

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    shutil.rmtree(get_bayesbind_small_dir(cfg), ignore_errors=True)
    make_all_bayesbind_small(cfg)