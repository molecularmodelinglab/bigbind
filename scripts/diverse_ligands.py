import yaml
import pandas as pd
import random
import re
import json
import scipy
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from scipy import sparse
from google.cloud import storage

from utils.cfg_utils import get_config
from bigbind.pocket_tm_score import * 
from bigbind.bigbind import *

from graph_tool.topology import max_independent_vertex_set
from bigbind.similarity import LigSimilarity
from utils.cfg_utils import get_config, get_output_dir

def main():
    """ This (currently very rushed) script generates a diverse set
    of ligands from the BigBind set, such that no two ligands are
    within 0.4 Tanimoto similarity of each other. """
    cfg = get_config("local")
    workflow = make_bigbind_workflow(cfg)
    lig_smi, lig_fps = workflow.run_node_from_name(cfg, "get_morgan_fps_parallel")
    lig_sim_mat = workflow.run_node_from_name(cfg, "get_tanimoto_matrix")

    lig_sim = LigSimilarity(lig_smi, lig_sim_mat)
    graph = lig_sim.get_gt_graph(lig_smi)

    diverse_mask = np.array(list(max_independent_vertex_set(graph)), dtype=bool)

    df = pd.DataFrame(lig_smi, columns=["lig_smiles"])
    df["bigbind_index"] = df.index
    df_diverse = df[diverse_mask]
    df_diverse.to_csv("outputs/diverse_ligands.csv", index=False)

main()