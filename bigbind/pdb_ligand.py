import warnings
import os
import shutil
from traceback import print_exc

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import sqlite3
from tqdm import tqdm
import numpy as np
import scipy.spatial as spa

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from utils.cfg_utils import get_output_dir

def to_elem(s):
    return s[0] + s[1:].lower()

def get_rec_coords(rec):
    res_indexes = []
    coords = []
    for i, chain in enumerate(rec):
        for res_idx, residue in enumerate(chain):
            # ain't nobody got time for waters
            if residue.get_resname() == "HOH": continue
            for atom in residue:
                elem = to_elem(atom.element)
                # atomic_num = Chem.Atom(elem).GetAtomicNum()
                coords.append(list(atom.get_vector()))
                res_indexes.append(str(residue.get_full_id()))
    return np.array(res_indexes), np.array(coords)

def get_lig_coords(lig_file):
    lig = next(Chem.SDMolSupplier(lig_file, sanitize=False))
    # assert lig is not None
    # assert '.' not in Chem.MolToSmiles(lig)
    conf = lig.GetConformer()
    return list(conf.GetPositions())

class PocketSelect(Select):

    def __init__(self, included):
        super().__init__()
        self.included = included

    def accept_residue(self, residue):
        return str(residue.get_full_id()) in self.included

biopython_parser = PDBParser()

def get_lig_url(lig_file):
    pdb_id = lig_file.split("/")[-1].split("_")[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        struct = biopython_parser.get_structure('random_id', lig_file)
        rec = struct[0]
        chain = next(iter(rec))
        asym_id = chain.id
        res = next(iter(chain))
        seq_id = res.id[1]
        return f"https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={seq_id}&auth_asym_id={asym_id}&encoding=sdf&filename=lig.sdf"

def save_pockets(cfg, rec_files, ligs, lig_dist_cutoff):

    out_pockets = {}
    res_numbers = {}
    
    all_lig_coords = []
    for lig in ligs:
        conf = lig.GetConformer()
        all_lig_coords += list(conf.GetPositions())
        
    all_lig_coords = np.array(all_lig_coords)
    
    for rec_file in rec_files:
        file_pre = rec_file.split(".")[0]
        outfile = os.path.join(get_output_dir(cfg), f"{file_pre}_pocket.pdb")
        full_rec_file = os.path.join(get_output_dir(cfg), rec_file)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            structure = biopython_parser.get_structure('random_id', full_rec_file)
            rec = structure[0]
            res_indexes, coords = get_rec_coords(rec)
            mask = spa.distance.cdist(coords, all_lig_coords).min(axis=1) < lig_dist_cutoff
            included = set(res_indexes[mask])
            io = PDBIO()
            io.set_structure(structure)
            io.save(outfile, PocketSelect(included))
        out_pockets[rec_file] = "/".join(outfile.split("/")[-2:])
        res_numbers[rec_file] = len(included)

    return out_pockets, res_numbers
            
