import numpy as np
import os
import warnings
import pickle
from collections import defaultdict
from traceback import print_exc
from tqdm import tqdm
import random

import scipy.spatial as spa

import yaml
from glob import glob

from Bio.PDB import get_surface, PDBParser, ShrakeRupley, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem

from bigbind import get_rec_coords, PocketSelect

class SphereVolume:
    
    def __init__(self, coords, radii):
        """ coords is list of x, y, z sphere centers
        radii is list of sphere radii
        sphere_res is the lat, lon resultion of the uv spheres. Should
        be pretty low since it slows it down a lot and we don't need
        high res """
        self.coords = np.array(coords)
        self.radii = np.array(radii)

    def union(self, other):
        return SphereVolume(np.concatenate((self.coords, other.coords)),
                            np.concatenate((self.radii, other.radii)))
        
    @staticmethod
    def from_mols(mols):
        """ Constructs a volume composed of all the (non-H) atoms
        of all the molecules (uses van der Waals radii) """
        coords = []
        radii = []
        for mol in mols:
            for i, atom in enumerate(mol.GetAtoms()):
                pos = mol.GetConformer().GetAtomPosition(i)
                r = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
                coord = [ pos.x, pos.y, pos.z ]
                coords.append(coord)
                radii.append(r)
        return SphereVolume(coords, radii)

    def get_bounding_box(self):
        max_r = max(self.radii)
        c1 = np.min(self.coords, 0) - max_r
        c2 = np.max(self.coords, 0) + max_r
        return c1, c2

    def get_union_bounding_box(self, other):
        """ returns the bounding box (min coord, max coord) of the
        union of the two volumes """
        c11, c12 = self.get_bounding_box()
        c21, c22 = other.get_bounding_box()
        all_coords = np.array([c11, c12, c21, c22])
        return np.min(all_coords, 0), np.max(all_coords, 0)

    def are_coords_in_volume(self, coords, dist_cutoff=0.0):
        """ Returns a boolean np array specifying if each coord
        is within the volume """
        mask = np.zeros(len(coords), dtype=bool)
        for my_coord, radius in zip(self.coords, self.radii):
            mask |= spa.distance.cdist(coords, np.array([my_coord])).min(axis=1) < radius + dist_cutoff
        return mask

    def intersect_volume(self, other, num_samples = 10000):
        """ use monte carlo method to estimate area of intersection
        of two volumes """

        
        c11, c12 = self.get_bounding_box()
        c21, c22 = other.get_bounding_box()

        # if the bounding boxes don't intersect, volume must be 0
        for i in range(3):
            if (c11[i] > c22[i]) or (c21[i] > c12[i]):
                return 0.0
        
        all_coords = np.array([c11, c12, c21, c22])
        c1, c2 =  np.min(all_coords, 0), np.max(all_coords, 0)
        
        coords = []
        for n in range(num_samples):
            coords.append(np.random.uniform(c1, c2))
        coords = np.array(coords)
        both_mask = self.are_coords_in_volume(coords) & other.are_coords_in_volume(coords)
        tot_vol = np.prod(c2 - c1)
        return (sum(both_mask)/len(both_mask))*tot_vol
        
    def get_residues(self, res_indexes, res_coords, dist_cutoff):
        """ Returns a set of indexes of residue numbers which are
        within dist_cutoff of this volume """
        mask = self.are_coords_in_volume(res_coords, dist_cutoff)
        included = set(res_indexes[mask])
        return included

def get_pockets(pqr_fname):
    """ Reads all the pockets from the pqr file into a dict mapping
    pocket id to a spherevolume representing the cavity """
    pocket_coords = defaultdict(list)
    pocket_radii = defaultdict(list)
    pqr = open(pqr_fname).read()
    for line in pqr.splitlines():
        if line.startswith("ATOM"):
            idx = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            r = float(line[66:71])
            coord = [x, y, z]
            pocket_coords[idx].append(coord)
            pocket_radii[idx].append(r)
    pocket_dict = {}
    for idx, coords in pocket_coords.items():
        radii = pocket_radii[idx]
        pocket_dict[idx] = SphereVolume(coords, radii)

    return pocket_dict

def get_pqr_file(cfg, rf):
    rec_file = rf.split("/")[-1].split(".")[0]
    return cfg["cache_folder"] + "/" + rec_file + "_out/" + rec_file + "_pockets.pqr"

def get_lig_pocket_intersections(ligs, pocket_dict):
    """ Returns dict mapping pocket ids to volumes of
    intersections with the ligands """
    lig_volume = SphereVolume.from_mols(ligs)
    inter_dict = {}
    for idx, poc, in pocket_dict.items():
        inter_dict[idx] = lig_volume.intersect_volume(poc)
    return inter_dict

def get_union_pocket(pocket_dict, inter_dict, frac_cutoff=0.5):
    """ Returns the union of all the pockets whose intersection
    volume is greater than the cutoff volume (the best intersection
    volume times the cutoff fraction) """
    best_vol = max(inter_dict.values())
    cutoff_vol = best_vol*frac_cutoff
    ret = None
    for idx, vol in inter_dict.items():
        poc = pocket_dict[idx]
        if vol > cutoff_vol:
            if ret is None:
                ret = poc
            else:
                ret = ret.union(poc)
    return ret
    
        
def get_rec_pocket(cfg, rf, ligs):

    rec_file = rf.split("/")[-1].split(".")[0]
    out_file = rf.split('.')[0] + "_pocket_v3.pdb"

    if os.path.exists(out_file):
       return
    
    pqr = get_pqr_file(cfg, rf)
    if not os.path.exists(pqr): return
    pocket_dict = get_pockets(pqr)
    inter_dict = get_lig_pocket_intersections(ligs, pocket_dict)
    final_poc = get_union_pocket(pocket_dict, inter_dict)
    
    inter_file = f"{cfg['cache_folder']}/{rec_file}_pocket_intersections.pkl"
    with open(inter_file, "wb") as fh:
        pickle.dump(inter_dict, fh)
        
    biopython_parser = PDBParser()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rf)
        rec = structure[0]
        res_indexes, coords = get_rec_coords(rec)
        included = final_poc.get_residues(res_indexes, coords, 0)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_file, PocketSelect(included))
    print(out_file)

def get_all_pockets(cfg):
    for rec_folder in tqdm(glob(cfg["bigbind_folder"] + "/*")):
        rec_files = glob(rec_folder + "/*_rec.pdb")
        lig_files = glob(rec_folder + "/*_lig.sdf")
        for rf in rec_files:
            if "19gs_B" in rf: break
        else:
            continue
        # print(lig_files)
        # break
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
                raise
                print(f"Error handling {rf}")
                print_exc()
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    get_all_pockets(cfg)
