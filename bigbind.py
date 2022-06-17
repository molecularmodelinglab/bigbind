import warnings
import os
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import sqlite3
from tqdm import tqdm
import numpy as np
import scipy.spatial as spa

from Bio.PDB import get_surface, PDBParser, ShrakeRupley, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def get_crossdocked_df():
    crossdocked_types = "/home/boris/Data/CrossDocked/types/it2_tt_v1.1_completeset_test0.types_filtered.types"
    in_df = pd.read_csv(crossdocked_types, sep=' ', names=["label", "binding_affinity", "crystal_rmsd", "rec_file", "lig_file", "vina_score"])
    return in_df

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
                atomic_num = Chem.Atom(elem).GetAtomicNum()
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
def get_pocket(uniprot, rec_files, lig_files, pdb_lig_df, lig_dist_cutoff=8):

    
    out_folder = f"/home/boris/Data/BigBind/{uniprot}"
    os.makedirs(out_folder, exist_ok=True)

    out_lig_files = []
    out_rec_files = []
    res_numbers = []
    
    all_lig_coords = []
    for lig_file in lig_files:
        *folder, file = lig_file.split("/")
        file_pre = file.split(".")[0]
        outfile = f"{out_folder}/{file_pre}.sdf"

        resname = file.split("_")[-2].upper()
        smiles = pdb_lig_df['smiles'][resname]
        template = Chem.MolFromSmiles(smiles)

        lig = Chem.MolFromPDBFile(lig_file)
        assert lig is not None
        lig = AllChem.AssignBondOrdersFromTemplate(template, lig)
        assert '.' not in Chem.MolToSmiles(lig)
        conf = lig.GetConformer()

        writer = Chem.SDWriter(outfile)
        writer.write(lig)
        writer.close()
        # shutil.copyfile(lig_file, outfile)
        out_lig_files.append(outfile)
        all_lig_coords += list(conf.GetPositions())
        
    all_lig_coords = np.array(all_lig_coords)
    
    # for rec_file in rec_files:
    #     *folder, file = rec_file.split("/")
    #     file_pre = file.split(".")[0]
    #     outfile = f"{out_folder}/{file_pre}_pocket.pdb"
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings("ignore", category=PDBConstructionWarning)
    #         structure = biopython_parser.get_structure('random_id', rec_file)
    #         rec = structure[0]
    #         res_indexes, coords = get_rec_coords(rec)
    #         mask = spa.distance.cdist(coords, all_lig_coords).min(axis=1) < lig_dist_cutoff
    #         included = set(res_indexes[mask])
    #         io = PDBIO()
    #         io.set_structure(structure)
    #         out_file = f"{out_folder}/{file_pre}_pocket.pdb"
    #         io.save(out_file, PocketSelect(included))
    #         out_rec_files.append(out_file)
    #         res_numbers.append(len(included))
    return out_lig_files, out_rec_files, res_numbers
            
