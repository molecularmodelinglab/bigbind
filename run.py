import random
import pandas as pd
from glob import glob
import sqlite3
import os
import pickle
import shutil
import requests
import subprocess
from traceback import print_exc
from copy import copy
import yaml
import io
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
from rdkit import RDLogger
from rdkit.Chem import AllChem
from collections import defaultdict
from copy import deepcopy

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

RDLogger.DisableLog('rdApp.*')

from pdb_ligand import get_lig_url, save_pockets
from cache import cache, item_cache
from probis import *
from sna import *

import signal

class timeout:
    def __init__(self, seconds, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class ProgressParallel(Parallel):
    def __call__(self, total, *args, **kwargs):
        with tqdm(total=total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def get_chembl_con(cfg):
    """ Get the SQLite connection to the chembl database"""
    con = sqlite3.connect(cfg["chembl_file"])
    return con

def load_sifts_into_chembl(cfg, con):
    """ Loads SIFTS into the chembl sqlite database for easy sql queries. Note that
    this actually modifies the db file itself for caching purposes. Not ideal to have
    side effects but in this case it can't really hurt anything """
    sifts_df = pd.read_csv(cfg["sifts_file"], comment='#')
    cursor = con.cursor()

    if cfg["cache"]:
        # no need to insert if table exists
        cursor.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='sifts' ")
        if cursor.fetchone()[0]==1:
            return
    
    cursor.execute("create table if not exists sifts (pdb text, chain text sp_primary text, res_beg integer, res_end integer, pdb_beg integer, pdb_end integer, sp_beg integer, sp_end integer)")
    cursor.fetchall()

    sifts_df.to_sql('sifts', con, if_exists='replace', index=False)

@cache
def load_crossdocked_files(cfg):
    """ Get the pdb files associated with the crystal rec and lig files.
    (the crossdocked types file only lists gninatypes files). Returns a
    dict mapping rec files to a list of lig files that bind to the rec """

    ret = defaultdict(set)
    for pocket in tqdm(glob(cfg["crossdocked_folder"] + "/*")):
        for rec_file in glob(pocket + "/*_rec.pdb"):
            for lig_file in glob(pocket + "/*_lig.pdb"):
                ret[rec_file].add(lig_file)
    return ret

@cache
def get_crossdocked_uniprots(cfg, con, cd_files):
    """ Return all the uniprot IDs from CrossDocked """
    cd_pdbs = { f.split('/')[-1].split('_')[0] for f in cd_files.keys() }
    cd_chains = { f.split('/')[-1].split('_')[1] for f in cd_files.keys() }
    cd_chains_str = ", ".join(map(lambda s: f"'{s}'", cd_chains))
    cd_pdbs_str = ", ".join(map(lambda s: f"'{s}'", cd_pdbs))

    crossdocked_uniprots = pd.read_sql_query(f"select SP_PRIMARY from sifts where PDB in ({cd_pdbs_str})", con)

    return crossdocked_uniprots

# ZINC yeets any molecule containing other elements, so shall we
allowed_atoms = { "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I" }
min_atoms_in_mol = 5
max_mol_weight = 1000

@cache
def get_crossdocked_chembl_activities(cfg, con, cd_uniprots):
    """ Get all activities (with some filtering for quality) of small
    molecules binding to proteins whose structures are in the crossdocked
    dataset """
    
    cd_uniprots_str = ", ".join(map(lambda s: f"'{s}'", cd_uniprots["SP_PRIMARY"]))
    
    query =   f"""

    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.potential_duplicate,
    COALESCE(act.data_validity_comment, 'valid') as data_validity_comment,
    a.confidence_score,
    td.chembl_id AS target_chembl_id,
    td.target_type,
    c.accession as protein_accession,
    a.chembl_id as assay_id
    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN target_type tt ON td.target_type = tt.target_type
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences c ON tc.component_id = c.component_id
    AND tt.parent_type  = 'PROTEIN' 
    AND act.pchembl_value IS NOT NULL
    AND c.accession in ({cd_uniprots_str});
    
    """
    
    filename = cfg["cache_folder"] + "/get_crossdocked_chembl_activities_chunked.csv"
    with open(filename, 'w'): pass

    approx_tot = 1571668
    chunksize = 1000
    chunks = pd.read_sql_query(query,con,chunksize=chunksize)
    header = True
    for i, chunk in enumerate(tqdm(chunks, total=int(approx_tot/chunksize))):

        chunk.to_csv(filename, header=header, mode='a', index=False)

        header = False

    return pd.read_csv(filename)

@cache
def save_activities_unfiltered(cfg, activities_unfiltered):
    activities_unfiltered.to_csv(cfg["bigbind_folder"] + "/activities_unfiltered.csv", index=False)

@cache
def filter_activities(cfg, activities_unfiltered):
    # remove mixtures
    activities = activities_unfiltered[~activities_unfiltered["canonical_smiles"].str.contains("\.")].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query("potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='")

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(columns=["compound_chembl_id", "potential_duplicate", "data_validity_comment", "confidence_score", "target_chembl_id", "target_type", "assay_id"])

    # now we filter duplicates
    dup_indexes = activities.duplicated(keep=False, subset=['canonical_smiles', 'protein_accession'])
    dup_df = activities[dup_indexes]

    dup_rows = defaultdict(list)
    for i, row in tqdm(dup_df.iterrows(), total=len(dup_df)):
        dup_rows[(row['canonical_smiles'], row['protein_accession'])].append(row)

    activities = activities[~dup_indexes].reset_index(drop=True)

    new_data = {
        "canonical_smiles": [],
        "standard_type": [],
        "standard_relation": [],
        "standard_value": [],
        "standard_units": [],
        "pchembl_value": [],
        "protein_accession": []
    }
    
    uniprot2df = {} # we want to cache the ligand dfs we find for each protein

    for (smiles, uniprot), rows in tqdm(dup_rows.items()):
        st_types = [ r.standard_type for r in rows ]
        pchembl_values = [ r.pchembl_value for r in rows ]
        
        # if there are multiple values for same protein-ligand pair, take the median

        final_pchembl = np.median(pchembl_values)
        final_st_type = "mixed"
        final_nM = 10**(9-final_pchembl)
        new_data["canonical_smiles"].append(smiles)
        new_data["standard_type"].append(final_st_type)
        new_data["standard_relation"].append("=")
        new_data["standard_value"].append(final_nM)
        new_data["standard_units"].append('nM')
        new_data["pchembl_value"].append(final_pchembl)
        new_data["protein_accession"].append(uniprot)

    new_data_df = pd.DataFrame(new_data)
    activities = pd.concat([activities, new_data_df])

    return activities

# alas, this takes up too much memory. need to create mols on the fly
@cache
def get_smiles_to_mol(cfg, activities):
    """ Creates an rdkit mol for each ligand in activities. Returns mapping
    from canonical_smiles to mol """
    ret = {}
    for smiles in tqdm(activities.canonical_smiles):
        if smiles in ret: continue
        mol = Chem.MolFromSmiles(smiles)
        ret[smiles] = mol
    return ret

def save_mol_sdf(cfg, name, smiles, num_embed_tries=10, verbose=True):
    """ Embed a single molecule in 3D, UFF optimize it, and save the structure
    to an SDF file. Returns None if it violates some of our molecule conditions
    -- invalid atoms, too large molecular weight, too few atoms. Really those
    filters should be in their own function... This also returns None if 
    the code times or can't embed the molecule in num_embed_tries tries"""

    periodic_table = Chem.GetPeriodicTable()
    
    folder = cfg["bigbind_folder"] + "/chembl_structures"
    os.makedirs(folder, exist_ok=True)
    filename = folder + "/" + name + ".sdf"
    if cfg["cache"] and "save_mol_sdf" not in cfg["recalc"]:
        if os.path.exists(filename):
            return True
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    if mol.GetNumAtoms() < min_atoms_in_mol: 
        if verbose:
            print(f"Rejecting {smiles} because it has only {mol.GetNumAtoms()} atoms")
        return False

    mw = Descriptors.MolWt(mol)
    if mw > max_mol_weight:
        if verbose:
            print(f"Rejecting {smiles} because it is too heavy {mw=}")
        return False

    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        sym = Chem.PeriodicTable.GetElementSymbol(periodic_table, num)
        if sym not in allowed_atoms:
            if verbose:
                print(f"Rejecting {smiles} because it contains {sym}.")
            return False

    # early exit cus this takes a while and Paul doesn't need the sdfs
    # return True

    try:
        with timeout(20):
            # try to come up with a 3d structure
            for i in range(num_embed_tries):
                conf_id = AllChem.EmbedMolecule(mol)
                if conf_id == 0:
                    try:
                        AllChem.UFFOptimizeMolecule(mol, 500)
                    except RuntimeError:
                        return False
                    break
            else:
                return False
    except TimeoutError:
        return False

    writer = Chem.SDWriter(filename)
    writer.write(mol)

    return True
        
@cache
def save_all_mol_sdfs(cfg, activities):
    """ Save all the sdfs for the bigbind molecules. In parellel! This takes a while"""
    unique_smiles = activities.canonical_smiles.unique()
    smiles2name = {}
    for i, smiles in enumerate(unique_smiles):
        smiles2name[smiles] = f"mol_{i}"

    smiles2filename = {}
    results = ProgressParallel(n_jobs=8)(len(smiles2name), (delayed(save_mol_sdf)(cfg, name, smiles) for smiles, name in smiles2name.items()))
    for result, (smiles, name) in zip(results, smiles2name.items()):
        if result:
            filename = f"chembl_structures/{name}.sdf"
            smiles2filename[smiles] = filename

    return smiles2filename

@cache
def add_sdfs_to_activities(cfg, activities, smiles2filename):
    """ Add SDF filenames to current activities dataframe"""
    filename_col = []
    for smiles in activities.canonical_smiles:
        if smiles in smiles2filename:
            filename_col.append(smiles2filename[smiles])
        else:
            filename_col.append("error")

    activities["lig_file"] = filename_col
    
    activities = activities.query("lig_file != 'error'").reset_index(drop=True)
    return activities
        
@cache
def get_chain_to_uniprot(cfg, con):
    """ Map PDB ids and chains to uniprot ids """
    sifts_df = pd.read_sql_query("SELECT * FROM SIFTS", con)
    chain2uniprot = {}
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
    return chain2uniprot

@cache
def get_uniprot_dicts(cfg, cd_files, chain2uniprot):
    """ Get a bunch of dictionaries mapping uniprot id to and from
    rec file, lig file, and pocketome pocket """
    uniprot2recs = defaultdict(set)
    uniprot2ligs = defaultdict(set)
    uniprot2pockets = defaultdict(set)
    pocket2recs = defaultdict(set)
    pocket2ligs = defaultdict(set)
    for rec_file, lig_files in tqdm(cd_files.items(), total=len(cd_files)):
        *rest, folder, file = rec_file.split("/")
        pocket = folder
        rec_id, rec_chain, *rest = file.split("_")
        key = (rec_id, rec_chain)
        for lig_file in lig_files:
            if key in chain2uniprot:
                uniprot = chain2uniprot[key]
                uniprot2recs[uniprot].add(rec_file)
                uniprot2ligs[uniprot].add(lig_file)
                uniprot2pockets[uniprot].add(pocket)

            pocket2recs[pocket].add(rec_file)
            pocket2ligs[pocket].add(lig_file)

    
    pocket2uniprots = defaultdict(set)
    for uniprot, pockets in uniprot2pockets.items():
        for pocket in pockets:
            pocket2uniprots[pocket].add(uniprot)

    return uniprot2recs,\
        uniprot2ligs,\
        uniprot2pockets,\
        pocket2uniprots,\
        pocket2recs,\
        pocket2ligs

@item_cache
def download_lig_sdf(cfg, name, lig):
    """ Download the sdf file associated with the ligand pdb
    file from the PDB"""
    url = get_lig_url(lig)
    res = requests.get(url)
    return res.text

@cache
def download_all_lig_sdfs(cfg, uniprot2ligs, rigorous=False):
    """ Returns a mapping from lig file to text associated with
    its downloaded SDF file """
    ret = {}
    tot_ligs = 0
    lf_errors = []
    for uniprot in tqdm(uniprot2ligs):
        ligs = uniprot2ligs[uniprot]
        for lig in ligs:
            tot_ligs += 1
            try:
                ret[lig] = download_lig_sdf(cfg, lig.replace("/", "_"), lig)
                # out_file = f"{lig}_untrans.sdf"
                # if c.calculate:
                #     url = get_lig_url(lig)
                #     res = requests.get(url)
                #     c.save(res.text)
                # with open(out_file, "w") as f:
                #     f.write(res.text)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error in {lig}")
                if rigorous:
                    raise
                else:
                    lf_errors.append(lig)
                    print_exc()
    print(f"Successfully downloaded {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/tot_ligs)*100}%)")
    return ret

@item_cache
def get_lig(cfg, name, lig_file, sdf_text, align_cutoff=2.0):
    """ Returns the RDKit mol object. The conformation is specified
    in the pdb lig_file and the connectivity is specified in sdf_text """
    lig = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_text.encode('utf-8'))))
    lig_untrans = deepcopy(lig)
    lig_pdb = Chem.MolFromPDBFile(lig_file)

    assert lig_pdb is not None
    assert lig is not None
    assert '.' not in Chem.MolToSmiles(lig)
    assert lig.GetNumAtoms() == lig_pdb.GetNumAtoms()
    
    conf = lig_pdb.GetConformer(0)
    lig.RemoveConformer(0)
    lig.AddConformer(conf)

    # sometimes the coordinates are in a different order?
    # for now just throw these out
    assert AllChem.AlignMol(lig_untrans, lig) < align_cutoff
    
    return lig

@cache
def get_all_ligs(cfg, lig_sdfs, rigorous=False, verbose=False):
    """ Returns a dict mapping ligand SDF files to ligand mol objects.
    Just for speed and convenience """
    lf_errors = []
    ret = {}
    for lf, sdf in tqdm(lig_sdfs.items()):
        try:
            ret[lf] = get_lig(cfg, lf.replace("/", "_"), lf, sdf)
        except KeyboardInterrupt:
            raise
        except:
            if verbose:
                print(f"Error in {lf}")
            if rigorous:
                raise
            else:
                if verbose:
                    print_exc()
                lf_errors.append(lf)
    print(f"Successfully obtained {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/len(lig_sdfs))*100}%)")
    return ret

@cache
def filter_uniprots(cfg, uniprot2pockets, pocket2uniprots):
    """ Filter down the proteins to just those with only one binding site """
    filtered_uniprots = { uniprot for uniprot, pockets in uniprot2pockets.items() if len(pockets) == 1 }
    filtered_pockets = { pocket for pocket, uniprots in pocket2uniprots.items() if len(filtered_uniprots.intersection(uniprots)) }
    print(f"Found {len(filtered_uniprots)} proteins with only 1 pocket out of {len(uniprot2pockets)} (Success rate {100*len(filtered_uniprots)/len(uniprot2pockets)}%)")
    print(f"Found {len(filtered_pockets)} pockets with valid proteins out of {len(pocket2uniprots)} (Success rate {100*len(filtered_pockets)/len(pocket2uniprots)}%)")
    return filtered_uniprots, filtered_pockets
    
@cache
def save_all_structures(cfg,
                        final_pockets,
                        pocket2uniprots,
                        pocket2recs,
                        pocket2ligs,
                        ligfile2lig):
    """ Saves the pdb receptor and sdf ligand files. Returns new
    dicts that use _those_ filename instead of crossdocked's """

    my_pocket2recs = defaultdict(set)
    my_pocket2ligs = defaultdict(set)
    my_ligfile2lig = {}
    
    for pocket in tqdm(final_pockets):
        out_folder = cfg["bigbind_folder"] + "/" + pocket
        
        recfiles = pocket2recs[pocket]
        ligfiles = pocket2ligs[pocket]

        ligs = set()
        for ligfile in ligfiles:
            if ligfile not in ligfile2lig: continue
            
            lig = ligfile2lig[ligfile]
            ligs.add(lig)
            
            os.makedirs(out_folder, exist_ok=True)
            out_file = out_folder + "/" + ligfile.split("/")[-1].split(".")[0] + ".sdf"
            my_pocket2ligs[pocket].add(out_file)
            my_ligfile2lig[out_file] = lig
            writer = Chem.SDWriter(out_file)
            writer.write(lig)

        if len(ligs):
            for recfile in recfiles:
                out_file = out_folder + "/" + recfile.split("/")[-1]
                my_pocket2recs[pocket].add(out_file)
                shutil.copyfile(recfile, out_file)

    return my_pocket2recs, my_pocket2ligs, my_ligfile2lig

@cache
def save_all_pockets(cfg, pocket2recs, pocket2ligs, ligfile2lig):
    """ Find the binding site for each receptor file and save it """
    rec2pocketfile = {}
    rec2res_num = {}
    for pocket, recfiles in tqdm(pocket2recs.items()):
        ligfiles = pocket2ligs[pocket]
        ligs = { ligfile2lig[lf] for lf in ligfiles }
        pocket_files, res_numbers = save_pockets(recfiles, ligs, lig_dist_cutoff=5)
        for recfile, pocket_file in pocket_files.items():
            res_num = res_numbers[recfile]
            rec2pocketfile[recfile] = pocket_file
            rec2res_num[recfile] = res_num
        # print(pocket_files)
        # print(res_numbers)
        # return
    return rec2pocketfile, rec2res_num

def get_bounds(cfg, ligs, padding):
    """ Returns the bounfds of the protein binding site from the 3D coords
    of all the aligned ligands """
    bounds = None
    for lig in ligs:
        box = ComputeConfBox(lig.GetConformer(0))
        if bounds is None:
            bounds = box
        else:
            bounds = ComputeUnionBox(box, bounds)

    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return center, size

@cache
def get_all_pocket_bounds(cfg, pocket2ligs, ligfile2lig, padding=4):
    """ Return the centroids and box sizes for each pocket """
    centers = {}
    sizes = {}
    for pocket, ligfiles in tqdm(pocket2ligs.items()):
        ligs = { ligfile2lig[lf] for lf in ligfiles }
        center, size = get_bounds(cfg, ligs, padding)
        centers[pocket] = center
        sizes[pocket] = size
    return centers, sizes

@cache
def create_struct_df(cfg,
                     pocket2recs,
                     pocket2ligs,
                     ligfile2lig,
                     rec2pocketfile,
                     rec2res_num,
                     pocket_centers,
                     pocket_sizes):
    """ Creates the dataframe from all the struct data we have """
    lig_smiles_series = []
    ligfile_series = []
    lig_pdb_series = []
    pocket_series = []
    recfile_series = []
    rec_pocket_file_series = []
    rec_pdb_series = []
    pocket_residue_series = []
    center_series = []
    bounds_series = []
    for pocket, ligfiles in tqdm(pocket2ligs.items()):
        recfiles = pocket2recs[pocket]
        rec_pdb2rec = { recfile.split("/")[-1].split("_")[0]: recfile for recfile in recfiles }
        current_rec_pdbs = set(rec_pdb2rec.keys())
        for ligfile in ligfiles:
            lig_pdb = ligfile.split("/")[-1].split("_")[0]
            lig = ligfile2lig[ligfile]
            smiles = Chem.MolToSmiles(lig)
            
            if len(current_rec_pdbs) > 1:
                rec_pdb = random.choice(list(current_rec_pdbs - { lig_pdb }))
            else:
                # we only do re-docking if there are no structures to
                # crossdock to
                rec_pdb = next(iter(current_rec_pdbs))
            # lol doesn't work cuz some receptors have multiple ligands
            # bound to the same site...
            # current_rec_pdbs = current_rec_pdbs - { rec_pdb }
            recfile = rec_pdb2rec[rec_pdb]
            rec_pocket_file = rec2pocketfile[recfile]
            poc_res_num = rec2res_num[recfile]
            center = pocket_centers[pocket]
            bounds = pocket_sizes[pocket]

            lig_smiles_series.append(smiles)
            ligfile_series.append("/".join(ligfile.split("/")[-2:]))
            lig_pdb_series.append(lig_pdb)
            pocket_series.append(pocket)
            recfile_series.append("/".join(recfile.split("/")[-2:]))
            rec_pdb_series.append(rec_pdb)
            rec_pocket_file_series.append(("/".join(rec_pocket_file.split("/")[-2:])))
            pocket_residue_series.append(poc_res_num)
            center_series.append(center)
            bounds_series.append(bounds)

    center_series = np.array(center_series)
    bounds_series = np.array(bounds_series)

    df_dict = {
        "lig_smiles": lig_smiles_series,
        "lig_file": ligfile_series,
        "lig_pdb": lig_pdb_series,
        "pocket": pocket_series,
        "ex_rec_file": recfile_series,
        "ex_rec_pdb": rec_pdb_series,
        "ex_rec_pocket_file": rec_pocket_file_series,
        "num_pocket_residues": pocket_residue_series
    }
    for name, num in zip("xyz", [0,1,2]):
        df_dict[f"pocket_center_{name}"] = center_series[:,num]
    for name, num in zip("xyz", [0,1,2]):
        df_dict[f"pocket_size_{name}"] = bounds_series[:,num]
            
    return pd.DataFrame(df_dict)

max_pocket_size = 42
max_pocket_residues = 5
@cache
def filter_struct_df(cfg, struct_df):
    """ Remove all the proteins with too large a binding site or too few residues from the structures dataframe. """
    return struct_df.query("num_pocket_residues >= @max_pocket_residues and lig_pdb != ex_rec_pdb and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)

@cache
def filter_act_df_final(cfg, act_df):
    """ Remove all the proteins with too large a binding site or too few residues from the activities dataframe. """
    ret = act_df.query("num_pocket_residues >= @max_pocket_residues and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)
    mask = []
    for i, row in tqdm(ret.iterrows(), total=len(ret)):
        lig_file = cfg["bigbind_folder"] + "/" + row.lig_file
        if not os.path.exists(lig_file):
            # print(f"! Something is up... {lig_file} doesn't exist...")
            mask.append(False)
        else:
            mask.append(True)
    ret = ret[mask].reset_index(drop=True)
    return ret


def get_clusters(cfg, pocket2rep_rec, probis_scores, z_cutoff=3.5):
    """ cluster pockets according to z-score. A pocket is added to a cluster
    if its rec. rep has a z_score of >= the z cutoff to any other rep. recs
    of the pockets in the cluster. The cutoff of 3.5 was used in 
    the og. CrossDocked paper """
    clusters = set()
    for pocket, rec in pocket2rep_rec.items():
        found_clusters = []
        for cluster in clusters:
            for pocket2 in cluster:
                rec2 = pocket2rep_rec[pocket2]
                key = (rec, rec2)
                if key in probis_scores and probis_scores[key] >= z_cutoff:
                    found_clusters.append(cluster)
                    break
        for cluster in found_clusters:
            clusters.remove(cluster)
        new_cluster = frozenset({pocket}.union(*found_clusters))
        clusters.add(new_cluster)
    return clusters

@cache
def get_lit_pcba_pockets(cfg, con, uniprot2pockets):

    pcba_files = glob(cfg["lit_pcba_folder"] + "/*/*.mol2")
    pcba_pdbs = defaultdict(set) # { f.split("/")[-1].split("_")[0] for f in pcba_files }
    for f in pcba_files:
        target = f.split("/")[-2]
        pdb = f.split("/")[-1].split("_")[0]
        pcba_pdbs[target].add(pdb)
    targ2pockets = {}
    for target, pdbs in pcba_pdbs.items():
        pdbs_str = ", ".join(map(lambda s: f"'{s}'", pdbs))
        uniprot_df = pd.read_sql_query(f"select SP_PRIMARY from sifts where PDB in ({pdbs_str})", con)
        pcba_uniprots = set(uniprot_df.SP_PRIMARY)
        pcba_pockets = set()
        for uniprot in pcba_uniprots:
            pcba_pockets = pcba_pockets.union(uniprot2pockets[uniprot])
        targ2pockets[target] = pcba_pockets
    return targ2pockets

@cache
def get_splits(cfg, clusters, lit_pcba_pockets, val_test_frac=0.1):
    """ Returns a dict mapping split name (train, val, test) to pockets.
    Both val and test splits are approx. val_test_frac of total. """
    assert val_test_frac < 0.5
    split_fracs = {
        "train": 1.0 - 2*val_test_frac,
        "val": val_test_frac,
        "test": val_test_frac,
    }
    all_pcba_pockets = set().union(*lit_pcba_pockets.values())
    clusters = list(clusters)
    splits = {}
    tot_clusters = 0
    for split, frac in split_fracs.items():
        cur_pockets = set()
        num_clusters = int(len(clusters)*frac)
        for cluster in clusters[tot_clusters:tot_clusters+num_clusters]:
            if split != "test" and len(cluster.intersection(all_pcba_pockets)) > 0:
                continue
            for pocket in cluster:
                cur_pockets.add(pocket)
        splits[split] = cur_pockets
        tot_clusters += num_clusters

    # re-add all the LIT-PCBA pockets to test
    for cluster in clusters:
        if len(cluster.intersection(all_pcba_pockets)) > 0:
            for pocket in cluster:
                splits["test"].add(pocket)

    # assert splits as disjoint
    for s1, p1 in splits.items():
        for s2, p2 in splits.items():
            if s1 == s2: continue
            assert len(p1.intersection(p2)) == 0
    return splits

@cache
def make_final_activities_df(cfg, 
                             activities,
                             uniprot2pockets,
                             pocket2recs,
                             rec2pocketfile,
                             rec2res_num,
                             pocket_centers,
                             pocket_sizes,
                             act_cutoff = 5.0):
    """ Add all the receptor stuff to the activities df. Code is
    mostly copied and pasted from the struct_df creation. Act_cutoff
    is the pchembl value at which we label a compound as active. Default
    is 5.0 (10 uM) """
    
    # final_uniprots ain't so final after all...
    # in future, this belongs elsewhere
    final_uniprots = set()
    for uniprot, pockets in uniprot2pockets.items():
        if len(pockets) > 1: continue
        pocket = next(iter(pockets))
        recs = pocket2recs[pocket]
        if len(recs) > 0:
            final_uniprots.add(uniprot)

    activities["active"] = activities["pchembl_value"] > act_cutoff
    
    # rename columns to jive w/ structure naming convention
    col_order = ["lig_smiles", "lig_file", "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value", "active", "uniprot"]
    new_act = activities.rename(columns={"canonical_smiles": "lig_smiles", "protein_accession": "uniprot" })[col_order]
    new_act = new_act.query("uniprot in @final_uniprots").reset_index(drop=True)

    pocket_series = []
    recfile_series = []
    rec_pocket_file_series = []
    rec_pdb_series = []
    pocket_residue_series = []
    center_series = []
    bounds_series = []
    for uniprot in new_act["uniprot"]:
        pockets = uniprot2pockets[uniprot]
        assert len(pockets) == 1
        pocket = next(iter(pockets))
        recfiles = list(pocket2recs[pocket])
        recfile = random.choice(recfiles)
        rec_pdb = recfile.split("/")[-1].split("_")[0]
        
        rec_pocket_file = rec2pocketfile[recfile]
        poc_res_num = rec2res_num[recfile]
        center = pocket_centers[pocket]
        bounds = pocket_sizes[pocket]

        pocket_series.append(pocket)
        recfile_series.append("/".join(recfile.split("/")[-2:]))
        rec_pdb_series.append(rec_pdb)
        rec_pocket_file_series.append(("/".join(rec_pocket_file.split("/")[-2:])))
        pocket_residue_series.append(poc_res_num)
        center_series.append(center)
        bounds_series.append(bounds)

    center_series = np.array(center_series)
    bounds_series = np.array(bounds_series)
    
    df_dict = {
        "pocket": pocket_series,
        "ex_rec_file": recfile_series,
        "ex_rec_pdb": rec_pdb_series,
        "ex_rec_pocket_file": rec_pocket_file_series,
        "num_pocket_residues": pocket_residue_series
    }
    for name, num in zip("xyz", [0,1,2]):
        df_dict[f"pocket_center_{name}"] = center_series[:,num]
    for name, num in zip("xyz", [0,1,2]):
        df_dict[f"pocket_size_{name}"] = bounds_series[:,num]
        
    for name, series in df_dict.items():
        new_act[name] = series
    return new_act

@cache
def save_clustered_structs(cfg, struct_df, splits):
    """ Split the structures dataframe into train, val, and test """
    struct_df.to_csv(cfg["bigbind_folder"] + "/structures_all.csv", index=False)
    for split, pockets in splits.items():
        split_struct = struct_df.query("pocket in @pockets").reset_index(drop=True)
        split_struct.to_csv(cfg["bigbind_folder"] + f"/structures_{split}.csv", index=False)

@cache
def save_clustered_activities(cfg, activities, splits):
    """ Split the activities dataframe into train, val, and test """
    activities.to_csv(cfg["bigbind_folder"] + f"/activities_all.csv", index=False)
    for split, pockets in splits.items():
        split_act = activities.query("pocket in @pockets").reset_index(drop=True)
        split_act.to_csv(cfg["bigbind_folder"] + f"/activities_{split}.csv", index=False)


def tarball_everything(cfg):
    *folders_out, folder = cfg["bigbind_folder"].split("/")
    os.chdir("/".join(folders_out))
    tar_out = folder + ".tar.bz2"
    cmd = ["tar", "-cjf", tar_out, folder]
    print(f"Running '{' '.join(cmd)}'")
    proc = subprocess.run(cmd)
    proc.check_returncode()
    print(f"Final BigBind Dataset saved to {tar_out}. Bon appetit!")
        
def run(cfg):
    os.makedirs(cfg["bigbind_folder"], exist_ok=True)
    if cfg["cache"]:
        os.makedirs(cfg["cache_folder"], exist_ok=True)
        
    con = get_chembl_con(cfg)
    load_sifts_into_chembl(cfg, con)
    cd_files = load_crossdocked_files(cfg)
    cd_uniprots = get_crossdocked_uniprots(cfg, con, cd_files)
    activities_unfiltered = get_crossdocked_chembl_activities(cfg, con, cd_uniprots)
    save_activities_unfiltered(cfg, activities_unfiltered)

    activities_filtered = filter_activities(cfg, activities_unfiltered)
    smiles2filename = save_all_mol_sdfs(cfg, activities_filtered)
    activities_filtered = add_sdfs_to_activities(cfg, activities_filtered, smiles2filename)
    
    chain2uniprot = get_chain_to_uniprot(cfg, con)
    
    uniprot2recs,\
    uniprot2ligs,\
    uniprot2pockets,\
    pocket2uniprots,\
    pocket2recs,\
    pocket2ligs = get_uniprot_dicts(cfg, cd_files, chain2uniprot)
    final_uniprots, final_pockets = filter_uniprots(cfg,
                                                    uniprot2pockets,
                                                    pocket2uniprots)
    
    lig_sdfs = download_all_lig_sdfs(cfg, uniprot2ligs)
    ligfile2lig = get_all_ligs(cfg, lig_sdfs)
    pocket2recs,\
    pocket2ligs,\
    ligfile2lig = save_all_structures(cfg,
                                      final_pockets,
                                      pocket2uniprots,
                                      pocket2recs,
                                      pocket2ligs,
                                      ligfile2lig)

    rec2pocketfile, rec2res_num = save_all_pockets(cfg, pocket2recs, pocket2ligs, ligfile2lig)
    pocket_centers, pocket_sizes = get_all_pocket_bounds(cfg, pocket2ligs, ligfile2lig)

    struct_df = create_struct_df(cfg,
                                 pocket2recs,
                                 pocket2ligs,
                                 ligfile2lig,
                                 rec2pocketfile,
                                 rec2res_num,
                                 pocket_centers,
                                 pocket_sizes)
    struct_df = filter_struct_df(cfg, struct_df)

    # probis stuff
    
    rec2srf = create_all_probis_srfs(cfg, rec2pocketfile)
    rep_srf2nosql = find_representative_rec(cfg, pocket2recs, rec2srf)
    rep_scores = convert_intra_results_to_json(cfg, rec2srf, rep_srf2nosql)
    pocket2rep_rec = get_rep_recs(cfg, pocket2recs, rep_scores)
    srf2nosql = find_all_probis_distances(cfg, pocket2rep_rec, rec2srf)
    full_scores = convert_inter_results_to_json(cfg, rec2srf, srf2nosql)

    # cluster and save everything
    lit_pcba_pockets = get_lit_pcba_pockets(cfg, con, uniprot2pockets)
    clusters = get_clusters(cfg, pocket2rep_rec, full_scores)
    splits = get_splits(cfg, clusters, lit_pcba_pockets)

    save_clustered_structs(cfg, struct_df, splits)

    activities = make_final_activities_df(cfg,\
                                          activities_filtered,\
                                          uniprot2pockets,\
                                          pocket2recs,\
                                          rec2pocketfile,\
                                          rec2res_num,\
                                          pocket_centers,\
                                          pocket_sizes)
    activities = filter_act_df_final(cfg, activities)

    save_clustered_activities(cfg, activities, splits)

    # SNA!

    # z cutoff of 3 clusters the kinases together
    big_clusters = get_clusters(cfg, pocket2rep_rec, full_scores, z_cutoff=3.0)
    save_all_sna_dfs(cfg, big_clusters, smiles2filename)
    save_all_screen_dfs(cfg, big_clusters, smiles2filename)

    tarball_everything(cfg)

SEED = 49
random.seed(SEED)
np.random.seed(SEED)
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
