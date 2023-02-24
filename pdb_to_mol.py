from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from cache import cache

@dataclass
class PDBBond:
    atom1: str
    atom2: str
    order: str
    aromatic: bool

@dataclass
class PDBChemical:
    code: str # pdb three letter code
    bonds: List[PDBBond]

@cache
def load_components_dict(cfg):
    cur_chemical = None
    in_bond_section = False

    chemicals = {}

    with open(cfg["pdb_components_file"], "r") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line.startswith("data_"):
                code = line.split("_")[-1]
                cur_chemical = PDBChemical(code, [])
                chemicals[code] = cur_chemical
            elif line == "_chem_comp_bond.pdbx_ordinal":
                in_bond_section = True
            elif in_bond_section:
                try:
                    code, atom1, atom2, order, aromatic, *rest = line.split()
                    assert cur_chemical is not None
                    aromatic = {"Y": True, "N": False}[aromatic]
                    cur_chemical.bonds.append(PDBBond(atom1, atom2, order, aromatic))
                except ValueError:
                    in_bond_section = False
    return chemicals