from dataclasses import dataclass
from traceback import print_exc
from typing import List
from collections import defaultdict
import warnings
from tqdm import tqdm
# from Bio.PDB import PDBParser
# from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Geometry import Point3D

from utils.task import task

@dataclass
class PDBBond:
    atom1: str
    atom2: str
    order: str
    aromatic: bool

@dataclass
class PDBAtom:
    name: str
    element: str
    charge: int

@dataclass
class PDBChemical:
    code: str # pdb three letter code
    atoms: List[PDBAtom]
    bonds: List[PDBBond]

@task(max_runtime=0.25)
def load_components_dict(cfg, comp_file):
    """ Load the chemical component dictionary (from https://www.wwpdb.org/data/ccd)
    into an actual dict so we can easily determine correct bond orders from pdb files """
    cur_chemical = None
    in_atom_section = False
    in_bond_section = False

    chemicals = {}

    with open(comp_file, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line.startswith("data_"):
                code = line.split("_")[-1]
                cur_chemical = PDBChemical(code, [], [])
                chemicals[code] = cur_chemical
            elif line == "_chem_comp_atom.pdbx_ordinal":
                in_atom_section = True
            elif line == "_chem_comp_bond.pdbx_ordinal":
                in_bond_section = True
            elif in_atom_section:
                try:
                    code, name, alt, elem, charge, *rest = line.split()
                    elem = elem[0] + elem[1:].lower()
                    assert cur_chemical is not None
                    cur_chemical.atoms.append(PDBAtom(name, elem, int(charge)))
                except ValueError:
                    in_atom_section = False
            elif in_bond_section:
                try:
                    code, atom1, atom2, order, aromatic, *rest = line.split()
                    assert cur_chemical is not None
                    aromatic = {"Y": True, "N": False}[aromatic]
                    cur_chemical.bonds.append(PDBBond(atom1, atom2, order, aromatic))
                except ValueError:
                    in_bond_section = False
                    # try:
                    #     mol_from_pdb_mol(cur_chemical)
                    # except:
                    #     print(f"Error processing {cur_chemical.code}")
                    #     print_exc()
                    #     return chemicals
                    # print(Chem.MolToSmiles(mol_from_pdb_mol(cur_chemical)))
                    # return chemicals
    return chemicals

def mol_from_pdb_mol(pdb_mol: PDBChemical):
    mol = Chem.RWMol()
    name2idx = {}
    for i, pdb_atom in enumerate(pdb_mol.atoms):
        atom = Chem.Atom(pdb_atom.element)
        atom.SetFormalCharge(pdb_atom.charge)
        mol.AddAtom(atom)
        name2idx[pdb_atom.name] = i
    for pdb_bond in pdb_mol.bonds:
        i1 = name2idx[pdb_bond.atom1]
        i2 = name2idx[pdb_bond.atom2]
        order = {
            "SING": Chem.BondType.SINGLE,
            "DOUB": Chem.BondType.DOUBLE,
            "TRIP": Chem.BondType.TRIPLE
        }[pdb_bond.order]
        mol.AddBond(i1, i2, order)
    Chem.SanitizeMol(mol)
    return mol

def mol_from_pdb(pdb_file, comp_dict):
    pdb_mol = None
    mol = Chem.RWMol()
    name2idx = {}
    name2pdb_atom = {}
    points = []

    obabel_bonds = defaultdict(lambda: defaultdict(int))

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                code = line[17:20].strip()

                if pdb_mol is None:
                    pdb_mol = comp_dict[code]
                    for pdb_atom in pdb_mol.atoms:
                        name2pdb_atom[pdb_atom.name] = pdb_atom

                name = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                points.append(Point3D(x, y, z))

                if code == "UNL":
                    elem = line[76:78].strip()
                    atom = Chem.Atom(elem)
                    if line[78] != ' ':
                        charge = int(line[78])
                        if line[79] == '-':
                            charge = -charge
                        atom.SetFormalCharge(int(charge))
                    mol.AddAtom(atom)
                else:
                    if name not in name2pdb_atom:
                        name = "\"" + name + "\""

                    name2idx[name] = len(name2idx)

                    pdb_atom = name2pdb_atom[name]
                    
                    atom = Chem.Atom(pdb_atom.element)
                    atom.SetFormalCharge(pdb_atom.charge)
                    mol.AddAtom(atom)

            elif line.startswith("CONECT") and code == "UNL":
                idx1 = int(line[6:11]) - 1
                idx2_counts = defaultdict(int)
                for idx2_str in [line[11:16], line[16:21], line[21:26], line[26:31]]:
                    if idx2_str.strip() == '': continue
                    idx2 = int(idx2_str)-1
                    obabel_bonds[idx1][idx2] += 1
                    
    seen = set()
    for idx1 in list(obabel_bonds.keys()):
        for idx2 in obabel_bonds[idx1].keys():
            if (idx2, idx1) in seen: continue
            count = max(obabel_bonds[idx1][idx2], obabel_bonds[idx2][idx1])
            order = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE][count-1]
            mol.AddBond(idx1, idx2, order)
            seen.add((idx1, idx2))
                    
    for pdb_bond in pdb_mol.bonds:
        if pdb_bond.atom1 in name2idx and pdb_bond.atom2 in name2idx:
            i1 = name2idx[pdb_bond.atom1]
            i2 = name2idx[pdb_bond.atom2]
            order = {
                "SING": Chem.BondType.SINGLE,
                "DOUB": Chem.BondType.DOUBLE,
                "TRIP": Chem.BondType.TRIPLE
            }[pdb_bond.order]
            mol.AddBond(i1, i2, order)

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, point in enumerate(points):
        conformer.SetAtomPosition(i, point)

    Chem.SanitizeMol(mol)

    mol.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(mol)

    return mol