import numpy as np
import os
import warnings
import pickle
from collections import defaultdict
from traceback import print_exc
from tqdm import tqdm

import yaml
from glob import glob
import scipy.spatial as spa

from Bio.PDB import get_surface, PDBParser, ShrakeRupley, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem

from bigbind import get_rec_coords, PocketSelect

class Pocket:
    
    def __init__(self):
        self.coords = []
        self.radii = []
        
    def add_sphere(self, x, y, z, radius):
        self.coords.append(np.array([x, y, z]))
        self.radii.append(radius)
        
    def get_residues(self, res_indexes, res_coords, dist_cutoff):
        mask = np.zeros(len(res_coords), dtype=bool)
        for coord, radius in zip(self.coords, self.radii):
            mask |= spa.distance.cdist(res_coords, np.array([coord])).min(axis=1) < radius + dist_cutoff
        included = set(res_indexes[mask])
        return included

    def get_num_overlaps(self, lig_coords):
        ret = 0
        for coord, radius in zip(self.coords, self.radii):
            ret += sum(spa.distance.cdist(lig_coords, np.array([coord])).min(axis=1) < radius)
        return ret

def get_pockets(pqr_fname):
    pocket_dict = defaultdict(Pocket)
    pqr = open(pqr_fname).read()
    for line in pqr.splitlines():
        if line.startswith("ATOM"):
            try:
                idx = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                r = float(line[66:71])
                # _, _, _, _, idx, x, y, z, _, r = line.split()
                pocket_dict[int(idx)].add_sphere(float(x), float(y), float(z), float(r))
            except ValueError:
                print(line)
                raise
    return pocket_dict

def get_pqr_file(rf):
    rec_file = rf.split("/")[-1].split(".")[0]
    return rf.split(".")[0] + "_out/" + rec_file + "_pockets.pqr"

def get_lig_pocket_overlaps(ligs, pocket_dict):
    all_lig_coords = []
    for lig in ligs:
        lig_coords = list(lig.GetConformer().GetPositions())
        all_lig_coords += lig_coords
    all_lig_coords = np.array(all_lig_coords)
    overlap_dict = {}
    for idx, pocket in pocket_dict.items():
        overlaps = pocket.get_num_overlaps(all_lig_coords)
        overlap_dict[idx] = overlaps
    return overlap_dict

def get_rec_pocket(cfg, rf, ligs):

    rec_file = rf.split("/")[-1].split(".")[0]
    out_file = rf.split('.')[0] + "_pocket_v2.pdb"

    if os.path.exists(out_file):
        return
    
    pqr = get_pqr_file(rf)
    if not os.path.exists(pqr): return
    pocket_dict = get_pockets(pqr)
    overlaps = get_lig_pocket_overlaps(ligs, pocket_dict)
    best_poc = pocket_dict[max(overlaps, key=overlaps.get)]
    
    overlap_file = f"{cfg['cache_folder']}/{rec_file}_pocket_overlaps.pkl"
    with open(overlap_file, "wb") as fh:
        pickle.dump(overlaps, fh)
        
    biopython_parser = PDBParser()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rf)
        rec = structure[0]
        res_indexes, coords = get_rec_coords(rec)
        included = best_poc.get_residues(res_indexes, coords, 0)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_file, PocketSelect(included))

def get_all_pockets(cfg):
    for rec_folder in tqdm(glob(cfg["bigbind_folder"] + "/*")):
        rec_files = glob(rec_folder + "/*_rec.pdb")
        lig_files = glob(rec_folder + "/*_lig.sdf")
        ligs = []
        for lf in lig_files:
            lig = next(Chem.SDMolSupplier(lf))
            ligs.append(lig)
        for rf in rec_files:
            try:
                get_rec_pocket(cfg, rf, ligs)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error handling {rf}")
                print_exc()
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    get_all_pockets(cfg)
