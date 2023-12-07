
import sys
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from chembl_structure_pipeline import standardize_mol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

from utils.cfg_utils import get_config, get_output_dir
from utils.task import task

@task()
def add_canonical_smiles_to_structures(cfg, struct_df):
    """ Uses Chembl's standardize_mol to canonicalize smiles strings
    so that we can find chembl activities in this dataframe """
    canon_smi = []
    for i, row in tqdm(struct_df.iterrows(), total=len(struct_df)):
        mol = Chem.MolFromSmiles(row.lig_smiles)
        assert mol is not None
        mol = standardize_mol(mol)
        if mol is None:
            print("nooo")
            canon_smi.append(None)
        else:
            canon_smi.append(Chem.MolToSmiles(mol))
    struct_df["canonical_smiles"] = canon_smi
    return struct_df

@task()
def get_chembl_crossdocked_affinities(cfg, act_df):
    """ Returns dict mapping pocket and canonical smiles to pchembl_value """

    chembl_pks = {}
    for i, row in tqdm(act_df.iterrows(), total=len(act_df)):
        pocket = row["pocket"]
        smi = row["lig_smiles"]
        key = (pocket, smi)
        pk = row["pchembl_value"]
        chembl_pks[key] = pk
    return chembl_pks

def get_crossdocked_affinities(cfg, crossdocked_folder):
    """ Returns dict mapping lig_file, rec_file to CrossDocked pK"""
    
    pks = {}

    types_files = [ 
        f"{crossdocked_folder}/types/it2_redocked_tt_v1.1_completeset_train0.types",
        f"{crossdocked_folder}/types/it2_redocked_tt_v1.1_completeset_test0.types",
    ]
    for fname in types_files:
        with open(fname, "r") as f:
            for line in f:
                lab, pk, rmsd, rec, lig, vina = line.split(" ")
                if "lig_tt_docked" not in lig: continue
                pk = abs(float(pk))
                if pk == 0: continue
                pocket = rec.split("/")[0]
                rf = "_".join(rec.split("_")[:-1]) + ".pdb"
                lf = pocket + "/" + ("_".join(lig.split("_")[-6:-3])) + ".sdf"
                key = (pocket, rf, lf)
                if key in pks:
                    assert abs(pks[key] - pk) < 0.1
                else:
                    pks[key] = pk
    
    return pks

@task()
def add_affinities_to_structures(cfg, struct_df, chembl_pks):
    """ Combines CrossDocked and ChEMBL affinities into one dataframe.
    If that values differ by >1.5 pK units, we throw out the value.
    Else we take the average of the two values. """

    # todo: putting crossdocked_dir in cfg is pretty jank
    crossdocked_pks = get_crossdocked_affinities(cfg, cfg.host.crossdocked_folder)

    pchembl_value = []
    for i, row in tqdm(struct_df.iterrows(), total=len(struct_df)):
        pocket = row["pocket"]
        rf = row["redock_rec_file"]
        lf = row["lig_crystal_file"]
        smi = row["canonical_smiles"]
        cd_key = (pocket, rf, lf)
        chembl_key = (pocket, smi)

        if cd_key in crossdocked_pks:
            pk = crossdocked_pks[cd_key]
            if chembl_key in chembl_pks:
                pk2 = chembl_pks[chembl_key]
                if abs(pk - pk2) > 1.5:
                    pk = None
                else:
                    pk = (pk + pk2) / 2
        elif chembl_key in chembl_pks:
            pk = chembl_pks[chembl_key]
        else:
            pk = None

        pchembl_value.append(pk)

    struct_df["pchembl_value"] = pchembl_value
    return struct_df