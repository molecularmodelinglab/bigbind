from copy import deepcopy
import io
import subprocess
import os
from glob import glob
import sqlite3
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import signal
import networkx as nx
import requests
from traceback import print_exc
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
import random
from bigbind.bayes_bind import make_all_bayesbind
from bigbind.knn import compare_probis_and_pocket_tm, get_optimal_tan_tm_coefs
from bigbind.similarity import LigSimilarity, get_lig_rec_edge_prob_ratios, get_lig_rec_edge_prob_ratios_probis, get_pocket_clusters, get_pocket_clusters_probis, get_pocket_clusters_with_tanimoto, get_pocket_clusters_with_tanimoto_probis, get_pocket_indexes, get_pocket_similarity, get_pocket_similarity_probis, plot_prob_ratios, plot_prob_ratios_probis
from bigbind.probis import convert_inter_results_to_json, convert_intra_results_to_json, create_all_probis_srfs, find_all_probis_distances, find_representative_rec, get_rep_recs

from utils.cfg_utils import get_output_dir
from utils.workflow import Workflow
from utils.task import file_task, simple_task, task, iter_task
from utils.downloads import StaticDownloadTask
from bigbind.pdb_to_mol import load_components_dict, mol_from_pdb
from bigbind.tanimoto_matrix import get_full_tanimoto_matrix, get_morgan_fps_parallel, get_tanimoto_matrix
from bigbind.pocket_tm_score import get_all_pocket_tm_scores
from bigbind.pdb_ligand import get_lig_url, save_pockets

SEED = 49
random.seed(SEED)
np.random.seed(SEED)


def canonicalize(mol):
    order = tuple(
        zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))]))
    )[1]
    mol = Chem.RenumberAtoms(mol, list(order))

    return mol


class timeout:
    def __init__(self, seconds, error_message="Timeout"):
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


# first define all the things we need to download
download_chembl = StaticDownloadTask(
    "download_chembl",
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_sqlite.tar.gz",
)
download_sifts = StaticDownloadTask(
    "download_sifts",
    "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz",
)
download_crossdocked = StaticDownloadTask(
    "download_crossdocked",
    "https://storage.googleapis.com/plantain_data/CrossDocked2022.tar.gz",
)
download_comp_dict = StaticDownloadTask(
    "download_comp_dict", "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif"
)

download_lit_pcba = StaticDownloadTask(
    "download_lit_pcba",
    "https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz"
)

# now we gotta unzip everything


@file_task("sifts.csv", max_runtime=0.5)
def unzip_sifts(cfg, out_filename, sifts_filename, prev_output=None):
    # if prev_output is not None:
    #     print(f"Using previous data from unzip_sifts")
    #     shutil.copyfile(prev_output, out_filename)
    subprocess.run(
        f"gunzip -c {sifts_filename} > {out_filename}", shell=True, check=True
    )


@file_task("CrossDocked2022", max_runtime=2, local=True)
def untar_crossdocked(cfg, out_filename, cd_filename):
    """Hacky -- relies on out_filename being equal to the regular output of tar"""
    out_dir = os.path.dirname(out_filename)
    cmd = f"tar -xf {cd_filename} --one-top-level={out_dir}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)

@file_task("LIT_PCBA", max_runtime=2, local=True)
def untar_lit_pcba(cfg, out_filename, lit_pcba_filename):
    """Hacky -- relies on out_filename being equal to the regular output of tar"""
    out_dir = out_filename
    os.makedirs(out_dir, exist_ok=True)
    cmd = f"tar -xf {lit_pcba_filename} --one-top-level={out_dir}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)

@file_task("chembl.db", max_runtime=200, local=True)
def untar_chembl(cfg, out_filename, chembl_filename):
    out_dir = os.path.dirname(out_filename)
    cmd = f"tar -xf {chembl_filename} --one-top-level={out_dir}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    # want to make this work for future chembl versions as well
    db_file = glob(os.path.join(out_dir, "chembl_*/chembl_*_sqlite/chembl_*.db"))[0]
    os.rename(db_file, out_filename)


@simple_task
def get_chembl_con(cfg, chembl_db_file):
    """Gets the connection to the chembl sqlite database"""
    con = sqlite3.connect(chembl_db_file)
    return con


@simple_task
def load_sifts_into_chembl(cfg, con, sifts_csv):
    """Loads SIFTS into the chembl sqlite database for easy sql queries. Note that
    this actually modifies the db file itself for caching purposes. Not ideal to have
    side effects but in this case it can't really hurt anything"""
    sifts_df = pd.read_csv(sifts_csv, comment="#")
    cursor = con.cursor()

    # no need to insert if table exists
    cursor.execute(
        " SELECT count(name) FROM sqlite_master WHERE type='table' AND name='sifts' "
    )
    if cursor.fetchone()[0] == 1:
        return con

    cursor.execute(
        "create table if not exists sifts (pdb text, chain text sp_primary text, res_beg integer, res_end integer, pdb_beg integer, pdb_end integer, sp_beg integer, sp_end integer)"
    )
    cursor.fetchall()

    sifts_df.to_sql("sifts", con, if_exists="replace", index=False)

    return con


@task(max_runtime=0.1)
def get_crossdocked_rf_to_lfs(cfg, cd_dir, prev_output=None):
    """Get the pdb files associated with the crystal rec and lig files.
    (the crossdocked types file only lists gninatypes files). Returns a
    dict mapping rec files to a list of lig files that bind to the rec"""

    ret = defaultdict(set)
    for pocket in tqdm(glob(f"{cd_dir}/*")):
        for rec_file in glob(pocket + "/*_rec.pdb"):
            rec_file = "/".join(rec_file.split("/")[-2:])
            for lig_file in glob(pocket + "/*_lig.pdb"):
                lig_file = "/".join(lig_file.split("/")[-2:])
                ret[rec_file].add(lig_file)
    return ret


@task(max_runtime=0.1)
def get_crossdocked_uniprots(cfg, con, cd_files):
    """Return all the uniprot IDs from CrossDocked"""
    cd_pdbs = {f.split("/")[-1].split("_")[0] for f in cd_files.keys()}
    cd_chains = {f.split("/")[-1].split("_")[1] for f in cd_files.keys()}
    cd_chains_str = ", ".join(map(lambda s: f"'{s}'", cd_chains))
    cd_pdbs_str = ", ".join(map(lambda s: f"'{s}'", cd_pdbs))

    crossdocked_uniprots = pd.read_sql_query(
        f"select SP_PRIMARY from sifts where PDB in ({cd_pdbs_str})", con
    )

    return crossdocked_uniprots


@file_task("activities_chunked.csv", max_runtime=24)
def get_crossdocked_chembl_activities(
    cfg, out_filename, con, cd_uniprots, prev_output=None
):
    """Get all activities (with some filtering for quality) of small
    molecules binding to proteins whose structures are in the crossdocked
    dataset"""

    if prev_output is not None:
        return prev_output

    cd_uniprots_str = ", ".join(map(lambda s: f"'{s}'", cd_uniprots["SP_PRIMARY"]))

    query = f"""

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
    with open(out_filename, "w"):
        pass

    approx_tot = 1571668
    chunksize = 1000
    chunks = pd.read_sql_query(query, con, chunksize=chunksize)
    header = True
    for i, chunk in enumerate(tqdm(chunks, total=int(approx_tot / chunksize))):
        chunk.to_csv(out_filename, header=header, mode="a", index=False)

        header = False

    return pd.read_csv(out_filename)


@task(max_runtime=2)
def filter_activities(cfg, activities_unfiltered, prev_output=None):
    """Remove mixtures, bad assays, and remove duplicates"""

    if prev_output is not None:
        return prev_output

    # remove mixtures
    activities = activities_unfiltered[
        ~activities_unfiltered["canonical_smiles"].str.contains("\.")
    ].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query(
        "potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='"
    )

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(
        columns=[
            "potential_duplicate",
            "data_validity_comment",
            "confidence_score",
            "target_chembl_id",
            "target_type",
            "assay_id",
        ]
    )

    # now we filter duplicates
    dup_indexes = activities.duplicated(
        keep=False, subset=["compound_chembl_id", "protein_accession"]
    )
    dup_df = activities[dup_indexes]

    dup_rows = defaultdict(list)
    for i, row in tqdm(dup_df.iterrows(), total=len(dup_df)):
        dup_rows[(row["compound_chembl_id"], row["protein_accession"])].append(row)

    activities = activities[~dup_indexes].reset_index(drop=True)

    new_data = {
        "canonical_smiles": [],
        "compound_chembl_id": [],
        "standard_type": [],
        "standard_relation": [],
        "standard_value": [],
        "standard_units": [],
        "pchembl_value": [],
        "protein_accession": [],
    }

    uniprot2df = {}  # we want to cache the ligand dfs we find for each protein

    # take the median value of duplicated measurements
    for (chembl_id, uniprot), rows in tqdm(dup_rows.items()):
        st_types = [r.standard_type for r in rows]
        pchembl_values = [r.pchembl_value for r in rows]

        final_pchembl = np.median(pchembl_values)
        final_st_type = "mixed"
        final_nM = 10 ** (9 - final_pchembl)
        new_data["canonical_smiles"].append(rows[0].canonical_smiles)
        new_data["compound_chembl_id"].append(chembl_id)
        new_data["standard_type"].append(final_st_type)
        new_data["standard_relation"].append("=")
        new_data["standard_value"].append(final_nM)
        new_data["standard_units"].append("nM")
        new_data["pchembl_value"].append(final_pchembl)
        new_data["protein_accession"].append(uniprot)

    new_data_df = pd.DataFrame(new_data)
    activities = pd.concat([activities, new_data_df])

    return activities


@simple_task
def load_act_unfiltered(cfg, filename):
    return pd.read_csv(filename)


@task(max_runtime=0.1)
def save_activities_unfiltered(cfg, activities_unfiltered):
    activities_unfiltered.to_csv(
        get_output_dir(cfg) + "/activities_unfiltered.csv", index=False
    )


# ZINC yeets any molecule containing other elements, so shall we
allowed_atoms = {"H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I"}
min_atoms_in_mol = 5
max_mol_weight = 1000


def save_mol_sdf(
    cfg, name, smiles, num_embed_tries=10, verbose=False, filename=None, ret_mol=False
):
    """Embed + UFF optimize single mol"""

    periodic_table = Chem.GetPeriodicTable()

    if filename is None:
        folder = get_output_dir(cfg) + "/chembl_structures"
        os.makedirs(folder, exist_ok=True)
        filename = folder + "/" + name + ".sdf"

    if os.path.exists(filename):
        if ret_mol:
            return next(Chem.SDMolSupplier(filename, sanitize=True))
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
    writer.close()

    if ret_mol:
        return next(Chem.SDMolSupplier(filename, sanitize=True))

    return True


SDF_N_CPU = 64
SDF_TOTAL_RUNTIME = 64


@task(max_runtime=SDF_TOTAL_RUNTIME / SDF_N_CPU, n_cpu=SDF_N_CPU, mem=1 * SDF_N_CPU)
def save_all_mol_sdfs(cfg, activities, prev_output=None):
    """Embed and UFF optimize each mol in BigBind. Returns a dict mapping smiles
    to filenames relative to bigbind output folder"""

    if prev_output is not None:
        return prev_output

    unique_smiles = activities.canonical_smiles.unique()
    smiles2name = {}
    for i, smiles in enumerate(unique_smiles):
        smiles2name[smiles] = f"mol_{i}"

    smiles2filename = {}
    results = ProgressParallel(n_jobs=SDF_N_CPU)(
        len(smiles2name),
        (
            delayed(save_mol_sdf)(cfg, name, smiles)
            for smiles, name in smiles2name.items()
        ),
    )
    for result, (smiles, name) in zip(results, smiles2name.items()):
        if result:
            filename = f"chembl_structures/{name}.sdf"
            smiles2filename[smiles] = filename

    return smiles2filename


@task(max_runtime=0.2)
def add_sdfs_to_activities(cfg, activities, smiles2filename):
    filename_col = []
    for smiles in activities.canonical_smiles:
        if smiles in smiles2filename:
            filename_col.append(smiles2filename[smiles])
        else:
            filename_col.append("error")

    activities["lig_file"] = filename_col

    activities = activities.query("lig_file != 'error'").reset_index(drop=True)
    return activities


@task(max_runtime=0.2)
def get_chain_to_uniprot(cfg, con):
    """Map PDB ids and chains to uniprot ids"""
    sifts_df = pd.read_sql_query("SELECT * FROM SIFTS", con)
    chain2uniprot = {}
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
    return chain2uniprot


@task(max_runtime=0.3, num_outputs=6)
def get_crossdocked_maps(cfg, cd_files, chain2uniprot):
    """Get a bunch of dictionaries mapping uniprot id to and from
    rec file, lig file, and pocketome pocket"""
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

    return (
        uniprot2recs,
        uniprot2ligs,
        uniprot2pockets,
        pocket2uniprots,
        pocket2recs,
        pocket2ligs,
    )


def download_lig_sdf(cfg, name, lig):
    url = get_lig_url(lig)
    res = requests.get(url)
    return res.text


@task(max_runtime=10)
def download_all_lig_sdfs(cfg, uniprot2ligs, rigorous=False):
    """Returns a mapping from lig file to text associated with
    its downloaded SDF file"""
    ret = {}
    tot_ligs = 0
    lf_errors = []
    for uniprot in tqdm(uniprot2ligs):
        ligs = uniprot2ligs[uniprot]
        for lig in ligs:
            tot_ligs += 1
            try:
                ret[lig] = download_lig_sdf(cfg, lig.replace("/", "_"), lig)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error in {lig}")
                if rigorous:
                    raise
                else:
                    lf_errors.append(lig)
                    print_exc()
    print(
        f"Successfully downloaded {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/tot_ligs)*100}%)"
    )
    return ret


def get_crystal_lig(cfg, name, lig_file, sdf_text, align_cutoff=2.0):
    """Returns the RDKit mol object. The conformation is specified
    in the pdb lig_file and the connectivity is specified in sdf_text"""
    lig = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_text.encode("utf-8"))))
    lig_untrans = deepcopy(lig)
    lig_pdb = Chem.MolFromPDBFile(lig_file)

    assert lig_pdb is not None
    assert lig is not None
    assert "." not in Chem.MolToSmiles(lig)
    assert lig.GetNumAtoms() == lig_pdb.GetNumAtoms()

    conf = lig_pdb.GetConformer(0)
    lig.RemoveConformer(0)
    lig.AddConformer(conf)

    # sometimes the coordinates are in a different order?
    # for now just throw these out
    assert AllChem.AlignMol(lig_untrans, lig) < align_cutoff

    return lig


@task(max_runtime=1)
def get_all_crystal_ligs(
    cfg, comp_dict, lig_sdfs, rigorous=False, verbose=False, prev_output=None
):
    """Returns a dict mapping ligand files to ligand mol objects"""

    if prev_output is not None:
        ret = {}
        for lf, mol in prev_output.items():
            lf = "/".join(lf.split("/")[-2:])
            ret[lf] = mol
        return ret

    lf_errors = []
    ret = {}
    for lf, sdf in tqdm(lig_sdfs.items()):
        try:
            ret[lf] = mol_from_pdb(lf, comp_dict)
            # ret[lf] = get_lig(cfg, lf.replace("/", "_"), lf, sdf)
        except KeyboardInterrupt:
            raise
        except:
            try:
                ret[lf] = get_crystal_lig(cfg, lf.replace("/", "_"), lf, sdf)
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
    print(
        f"Successfully obtained {len(ret)} ligands and had {len(lf_errors)} errors (success rate {(len(ret)/len(lig_sdfs))*100}%)"
    )
    return ret


@task(max_runtime=0.1, num_outputs=2)
def filter_uniprots(cfg, uniprot2pockets, pocket2uniprots):
    """Return only the uniprot IDs and pocketome pockets that we're using. Only use
    proteins with a single pocket."""
    filtered_uniprots = set()
    filtered_pockets = set()
    for uniprot, pockets in uniprot2pockets.items():
        if len(pockets) > 1:
            continue
        pocket = next(iter(pockets))
        filtered_uniprots.add(uniprot)
        filtered_pockets.add(pocket)
    # filtered_uniprots = { uniprot for uniprot, pockets in uniprot2pockets.items() if len(pockets) == 1 }
    # filtered_pockets = { pocket for pocket, uniprots in pocket2uniprots.items() if len(filtered_uniprots.intersection(uniprots)) }
    print(
        f"Found {len(filtered_uniprots)} proteins with only 1 pocket out of {len(uniprot2pockets)} (Success rate {100*len(filtered_uniprots)/len(uniprot2pockets)}%)"
    )
    print(
        f"Found {len(filtered_pockets)} pockets with valid proteins out of {len(pocket2uniprots)} (Success rate {100*len(filtered_pockets)/len(pocket2uniprots)}%)"
    )
    return filtered_uniprots, filtered_pockets


@task(max_runtime=3, num_outputs=4)
def save_all_structures(
    cfg, cd_dir, pocket2uniprots, pocket2recs, pocket2ligs, ligfile2lig
):
    """Saves the pdb receptor and sdf ligand files. Returns new
    dicts that use _those_ filename instead of crossdocked's"""

    folder = get_output_dir(cfg)

    my_pocket2recs = defaultdict(set)
    my_pocket2ligs = defaultdict(set)
    my_ligfile2lig = {}
    my_ligfile2uff_lig = {}

    for pocket in tqdm(pocket2uniprots):
        out_folder = folder + "/" + pocket

        recfiles = pocket2recs[pocket]
        ligfiles = pocket2ligs[pocket]

        ligs = set()
        for ligfile in ligfiles:
            if ligfile not in ligfile2lig:
                continue

            lig = ligfile2lig[ligfile]
            ligs.add(lig)

            os.makedirs(out_folder, exist_ok=True)
            out_file = out_folder + "/" + ligfile.split("/")[-1].split(".")[0] + ".sdf"
            my_pocket2ligs[pocket].add("/".join(out_file.split("/")[-2:]))
            my_ligfile2lig["/".join(out_file.split("/")[-2:])] = lig
            writer = Chem.SDWriter(out_file)
            writer.write(lig)
            writer.close()

            uff_filename = (
                out_folder + "/" + ligfile.split("/")[-1].split(".")[0] + "_uff.sdf"
            )
            # if os.path.exists(uff_filename):
            #     my_ligfile2uff_lig[out_file] = uff_filename
            uff = save_mol_sdf(
                cfg, None, Chem.MolToSmiles(lig), filename=uff_filename, ret_mol=True
            )
            if uff:
                uff_noh = canonicalize(Chem.RemoveHs(uff))
                lig_noh = canonicalize(Chem.RemoveHs(lig))
                if Chem.MolToSmiles(uff_noh, isomericSmiles=False) != Chem.MolToSmiles(
                    lig_noh, isomericSmiles=False
                ):
                    print(f"Error saving uff for {out_file}")
                    my_ligfile2uff_lig["/".join(out_file.split("/")[-2:])] = "none"
                else:
                    my_ligfile2uff_lig[
                        "/".join(out_file.split("/")[-2:])
                    ] = uff_filename
            else:
                my_ligfile2uff_lig["/".join(out_file.split("/")[-2:])] = "none"

        if len(ligs):
            for recfile in recfiles:
                full_recfile = cd_dir + "/" + recfile
                out_file = (
                    out_folder
                    + "/"
                    + recfile.split("/")[-1].split(".")[0]
                    + "_nofix.pdb"
                )
                my_pocket2recs[pocket].add("/".join(out_file.split("/")[-2:]))
                shutil.copyfile(full_recfile, out_file)

    return my_pocket2recs, my_pocket2ligs, my_ligfile2lig, my_ligfile2uff_lig


@task(max_runtime=3, num_outputs=2)
def save_all_pockets(cfg, pocket2recs, pocket2ligs, ligfile2lig, prev_output=None):
    # if prev_output is not None:
    #     r2p, r2r = prev_output
    #     r2p_fixed = { "/".join(f.split("/")[-2:]): "/".join(val.split("/")[-2:]) for f, val in r2p.items() }
    #     r2r_fixed = { "/".join(f.split("/")[-2:]): val for f, val in r2r.items() }
    #     return r2p_fixed, r2r_fixed

    rec2pocketfile = {}
    rec2res_num = {}
    for pocket, recfiles in tqdm(pocket2recs.items()):
        ligfiles = pocket2ligs[pocket]
        ligs = {ligfile2lig[lf] for lf in ligfiles}
        pocket_files, res_numbers = save_pockets(cfg, recfiles, ligs, lig_dist_cutoff=5)
        for recfile, pocket_file in pocket_files.items():
            res_num = res_numbers[recfile]
            rec2pocketfile[recfile] = pocket_file
            rec2res_num[recfile] = res_num
    return rec2pocketfile, rec2res_num


def get_bounds(cfg, ligs, padding):
    bounds = None
    for lig in ligs:
        box = ComputeConfBox(lig.GetConformer(0))
        if bounds is None:
            bounds = box
        else:
            bounds = ComputeUnionBox(box, bounds)

    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5 * (bounds_min + bounds_max)
    size = (bounds_max - center + padding) * 2
    return center, size


@task(num_outputs=2)
def get_all_pocket_bounds(cfg, pocket2ligs, ligfile2lig, padding=4):
    """Return the centroids and box sizes for each pocket"""
    centers = {}
    sizes = {}
    for pocket, ligfiles in tqdm(pocket2ligs.items()):
        ligs = {ligfile2lig[lf] for lf in ligfiles}
        center, size = get_bounds(cfg, ligs, padding)
        centers[pocket] = center
        sizes[pocket] = size
    return centers, sizes


# pdb_parser = PDBParser(QUIET=True)
# @task(max_runtime=2)
# def get_recfile_to_struct(cfg, rec_to_pocketfile):
#     ret = {}
#     for rec in tqdm(rec_to_pocketfile.keys()):
#         ret[rec] = pdb_parser.get_structure("1", rec)
#     return ret


# def get_single_pqr_file(cfg, pdb_file):
#     pdb_file_full = get_output_dir(cfg) + "/" + pdb_file
#     out_filename = get_output_dir(cfg) + "/" + pdb_file.replace("_nofix.pdb", ".pdb")

#     # if os.path.exists(out_filename):
#     #     return "/".join(out_filename.split("/")[-2:])

#     # old_name = "/home/boris/Data/BigBindScratch/test/global/output/" + pdb_file.replace("_nofix.pdb", ".pqr")
#     # if os.path.exists(old_name):
#     #     shutil.copyfile(old_name, out_filename)
#     #     return out_filename

#     cmd = f"pdb2pqr --titration-state-method propka -q {pdb_file_full} {out_filename}"
#     # print(f"Running {cmd}")
#     try:
#         subprocess.run(
#             cmd,
#             shell=True,
#             check=True,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#         )
#     except subprocess.CalledProcessError:
#         return None
#     return "/".join(out_filename.split("/")[-2:])


# compute_pqr_files = iter_task(1, 16, n_cpu=8)(get_single_pqr_file)


# @simple_task
# def preproc_pdb2pqr(cfg, pocket2rfs):
#     ret = []
#     for poc, rfs in pocket2rfs.items():
#         ret += rfs
#     return ret


# @simple_task
# def postproc_pdb2pqr(cfg, pocket2rfs, pqr_files):
#     ret = defaultdict(set)
#     idx = 0
#     for poc, rfs in pocket2rfs.items():
#         for rf in rfs:
#             fixed_rf = pqr_files[idx]
#             idx += 1
#             if fixed_rf is not None:
#                 ret[poc].add(fixed_rf)
#     return ret


# def fix_all_recfiles(pocket2rfs):
#     inputs = preproc_pdb2pqr(pocket2rfs)
#     outputs = compute_pqr_files(inputs)
#     return postproc_pdb2pqr(pocket2rfs, outputs)

def fix_single_pdb_file(cfg, pdb_file):
    pdb_file_full = get_output_dir(cfg) + "/" + pdb_file
    out_filename = get_output_dir(cfg) + "/" + pdb_file.replace("_nofix.pdb", ".pdb")

    # if os.path.exists(out_filename):
    #     return "/".join(out_filename.split("/")[-2:])

    # old_name = "/home/boris/Data/BigBindScratch/test/global/output/" + pdb_file.replace("_nofix.pdb", ".pqr")
    # if os.path.exists(old_name):
    #     shutil.copyfile(old_name, out_filename)
    #     return out_filename

    cmd = f"pdbfixer {pdb_file_full} --output {out_filename}  --add-atoms=heavy --add-residues"
    # print(f"Running {cmd}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    return "/".join(out_filename.split("/")[-2:])


fix_pdb_files = iter_task(1, 16, n_cpu=8)(fix_single_pdb_file)


@simple_task
def preproc_fix_pdb(cfg, pocket2rfs):
    ret = []
    for poc, rfs in pocket2rfs.items():
        ret += rfs
    return ret


@simple_task
def postproc_fix_pdb(cfg, pocket2rfs, pdb_files):
    ret = defaultdict(set)
    idx = 0
    for poc, rfs in pocket2rfs.items():
        for rf in rfs:
            fixed_rf = pdb_files[idx]
            idx += 1
            if fixed_rf is not None:
                ret[poc].add(fixed_rf)
    return ret

def fix_all_recfiles(pocket2rfs):
    inputs = preproc_fix_pdb(pocket2rfs)
    outputs = fix_pdb_files(inputs)
    return postproc_fix_pdb(pocket2rfs, outputs)


max_pocket_size = 42
max_pocket_residues = 5


@task(max_runtime=0.2)
def make_final_activities_df(
    cfg,
    activities,
    uniprot2pockets,
    pocket2recs,
    rec2pocketfile,
    rec2res_num,
    pocket_centers,
    pocket_sizes,
    act_cutoff=5.0,
):
    """Add all the receptor stuff to the activities df. Code is
    mostly copied and pasted from the struct_df creation. Act_cutoff
    is the pchembl value at which we label a compound as active. Default
    is 5.0 (10 uM)"""

    final_uniprots = set()
    for uniprot, pockets in uniprot2pockets.items():
        if len(pockets) > 1:
            continue
        pocket = next(iter(pockets))
        recs = pocket2recs[pocket]
        if len(recs) > 0:
            final_uniprots.add(uniprot)

    activities["active"] = activities["pchembl_value"] > act_cutoff

    # rename columns to jive w/ structure naming convention
    col_order = [
        "lig_smiles",
        "lig_file",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "active",
        "uniprot",
    ]
    new_act = activities.rename(
        columns={"canonical_smiles": "lig_smiles", "protein_accession": "uniprot"}
    )[col_order]
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
        # print(pockets)
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
        "num_pocket_residues": pocket_residue_series,
    }
    for name, num in zip("xyz", [0, 1, 2]):
        df_dict[f"pocket_center_{name}"] = center_series[:, num]
    for name, num in zip("xyz", [0, 1, 2]):
        df_dict[f"pocket_size_{name}"] = bounds_series[:, num]

    for name, series in df_dict.items():
        new_act[name] = series

    ret = new_act.query(
        "num_pocket_residues >= @max_pocket_residues and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size"
    ).reset_index(drop=True)
    mask = []
    for i, row in tqdm(ret.iterrows(), total=len(ret)):
        lig_file = get_output_dir(cfg) + "/" + row.lig_file
        if not os.path.exists(lig_file):
            # print(f"! Something is up... {lig_file} doesn't exist...")
            mask.append(False)
        else:
            mask.append(True)
    ret = ret[mask].reset_index(drop=True)
    return ret

@simple_task
def get_fake_pocket_tm_scores(cfg, rf2pocketfile):
    rfs1 = random.sample(list(rf2pocketfile.keys()), 100)
    rfs2 = random.sample(list(rf2pocketfile.keys()), 100)
    ret = {}
    for rf1 in rfs1:
        for rf2 in rfs2:
            ret[(rf1, rf2)] = random.random()
    return ret

@task(max_runtime=0.2)
def get_lit_pcba_pockets(cfg, con, lit_pcba_dir, uniprot2pockets):

    pcba_files = glob(lit_pcba_dir + "/*/*.mol2")
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

# force this!
@task(max_runtime=0.2, force=False)
def get_splits(cfg, activities, clusters, lit_pcba_pockets, pocket_indexes, val_test_frac=0.15):
    """ Returns a dict mapping split name (train, val, test) to pockets.
    Both val and test splits are approx. val_test_frac of total. """
    assert val_test_frac < 0.5
    split_fracs = {
        "train": 1.0 - 2*val_test_frac,
        "val": val_test_frac,
        "test": val_test_frac,
    }
    all_pcba_pockets = set().union(*lit_pcba_pockets.values())
    
    # rescale the split fracs so that the fracs are correct after adding the pcba pockets
    pcba_clusters = [ cluster for cluster in clusters if len(cluster.intersection(all_pcba_pockets)) > 0 ]
    pcba_indexes = 0
    for cluster in pcba_clusters:
        cluster_size = 0
        for poc in cluster:
            if poc in pocket_indexes:
                pcba_indexes += len(pocket_indexes[poc])
                cluster_size += len(pocket_indexes[poc])
        # print(cluster)
        # print(len(cluster), cluster_size)

    pcba_frac = pcba_indexes / len(activities)
    test_res = split_fracs["test"] - pcba_frac
    no_pcba_fracs = {
        "train": split_fracs["train"]*(1+test_res),
        "val": split_fracs["val"]*(1+test_res),
        "test": max(test_res, 0.0),
    }
    cur_clusters = [ cluster for cluster in sorted(clusters, key=lambda c: len(c), reverse=True) if cluster not in pcba_clusters ]

    splits = {}
    for split, frac in no_pcba_fracs.items():
        cur_pockets = set()
        desired_len = int(len(activities)*frac)
        total_clusters = 0
        total_idxs = 0
        for cluster in cur_clusters:
            for pocket in cluster:
                cur_pockets.add(pocket)
                if pocket in pocket_indexes:
                    total_idxs += len(pocket_indexes[pocket])
            total_clusters += 1
            if total_idxs >= desired_len and split != "test":
                break
        # print(total_clusters, total_idxs, desired_len, len(cur_clusters))
        cur_clusters = cur_clusters[total_clusters:]
        splits[split] = cur_pockets

    # re-add all the LIT-PCBA pockets to test
    for cluster in clusters:
        if len(cluster.intersection(all_pcba_pockets)) > 0:
            for pocket in cluster:
                splits["test"].add(pocket)

    # assert splits as disjoint
    for s1, p1 in splits.items():
        num_idxs = 0
        for poc in p1:
            if poc in pocket_indexes:
                num_idxs += len(pocket_indexes[poc])
        print(f"{s1} has {len(p1)} pockets and {num_idxs} datapoints")
        for s2, p2 in splits.items():
            if s1 == s2: continue
            assert len(p1.intersection(p2)) == 0

    return splits

# force this!
@task(max_runtime=0.2, force=False)
def get_splits_probis(cfg, activities, clusters, lit_pcba_pockets, pocket_indexes, val_test_frac=0.15):
    """ Returns a dict mapping split name (train, val, test) to pockets.
    Both val and test splits are approx. val_test_frac of total. """
    assert val_test_frac < 0.5
    split_fracs = {
        "train": 1.0 - 2*val_test_frac,
        "val": val_test_frac,
        "test": val_test_frac,
    }
    all_pcba_pockets = set().union(*lit_pcba_pockets.values())
    
    # rescale the split fracs so that the fracs are correct after adding the pcba pockets
    pcba_clusters = [ cluster for cluster in clusters if len(cluster.intersection(all_pcba_pockets)) > 0 ]
    pcba_indexes = 0
    for cluster in pcba_clusters:
        for poc in cluster:
            if poc in pocket_indexes:
                pcba_indexes += len(pocket_indexes[poc])

    pcba_frac = pcba_indexes / len(activities)
    test_res = split_fracs["test"] - pcba_frac
    no_pcba_fracs = {
        "train": split_fracs["train"]*(1+test_res),
        "val": split_fracs["val"]*(1+test_res),
        "test": max(test_res, 0.0),
    }

    cur_clusters = [ cluster for cluster in sorted(clusters, key=lambda c: len(c), reverse=True) if cluster not in pcba_clusters ]

    splits = {}
    for split, frac in no_pcba_fracs.items():
        cur_pockets = set()
        desired_len = int(len(activities)*frac)
        total_clusters = 0
        total_idxs = 0
        for cluster in cur_clusters:
            for pocket in cluster:
                cur_pockets.add(pocket)
                if pocket in pocket_indexes:
                    total_idxs += len(pocket_indexes[pocket])
            total_clusters += 1
            if total_idxs >= desired_len and split != "test":
                break
        # print(total_clusters, total_idxs, desired_len, len(cur_clusters))
        cur_clusters = cur_clusters[total_clusters:]
        splits[split] = cur_pockets

    # re-add all the LIT-PCBA pockets to test
    for cluster in clusters:
        if len(cluster.intersection(all_pcba_pockets)) > 0:
            for pocket in cluster:
                splits["test"].add(pocket)

    # assert splits as disjoint
    for s1, p1 in splits.items():
        num_idxs = 0
        for poc in p1:
            if poc in pocket_indexes:
                num_idxs += len(pocket_indexes[poc])
        print(f"{s1} has {len(p1)} pockets and {num_idxs} datapoints")
        for s2, p2 in splits.items():
            if s1 == s2: continue
            assert len(p1.intersection(p2)) == 0

    return splits

@task(max_runtime=2)
def get_lig_clusters(cfg, activities, poc_indexes, lig_smi, lig_sim_mat):
    """ Clusters all the ligands for a given pocket with tanimoto cutoff > 0.4
    Returns an array of cluster indexes for each ligand  in activities"""
    lig_sim = LigSimilarity(lig_smi, lig_sim_mat)
    ret_clusters = np.zeros(len(activities), dtype=np.int32) - 1
    for poc, indexes in tqdm(poc_indexes.items()):
        poc_smi = activities.lig_smiles[indexes]
        G = lig_sim.get_nx_graph(poc_smi)

        clusters = list(nx.connected_components(G))
        lig2cluster = {}
        for i, cluster in enumerate(clusters):
            for lig in cluster:
                lig2cluster[lig] = i

        # print(poc, len(list(clusters)), len(indexes))

        for idx, smi in zip(indexes, poc_smi):
            ret_clusters[idx] = lig2cluster[smi]

    assert (ret_clusters >= 0).all()

    return ret_clusters

# force this!
@task(max_runtime=0.1, force=False)
def add_all_clusters_to_act(cfg, activities, lig_cluster_idxs, poc_indexes, poc_clusters):
    
    activities["lig_cluster"] = lig_cluster_idxs

    rec_cluster_idxs = np.zeros(len(activities), dtype=np.int32) - 1
    for i, cluster in enumerate(poc_clusters):
        for poc in cluster:
            if poc in poc_indexes:
                rec_cluster_idxs[np.array(poc_indexes[poc], dtype=int)] = i

    assert (rec_cluster_idxs >= 0).all()

    activities["rec_cluster"] = rec_cluster_idxs

    return activities

@task(max_runtime=0.5)
def create_struct_df(cfg,
                     pocket2recs,
                     pocket2ligs,
                     ligfile2lig,
                     ligfile2uff,
                     rec2pocketfile,
                     rec2res_num,
                     pocket_centers,
                     pocket_sizes):
    """ Creates the dataframe from all the struct data we have """

    df_list = []

    for pocket, ligfiles in tqdm(pocket2ligs.items()):
        recfiles = pocket2recs[pocket]
        rec_pdb2rec = { recfile.split("/")[-1].split("_")[0]: recfile for recfile in recfiles }
        current_rec_pdbs = set(rec_pdb2rec.keys())
        for ligfile in ligfiles:
            lig_uff_file = ligfile2uff[ligfile]
            lig_pdb = ligfile.split("/")[-1].split("_")[0]
            lig = ligfile2lig[ligfile]
            smiles = Chem.MolToSmiles(lig)
            
            # smh sometimes crossdocked has the ligand pdb but not the associated rec pdb
            if lig_pdb not in rec_pdb2rec: continue

            redock_pdb = lig_pdb
            if len(current_rec_pdbs) > 1:
                crossdock_pdb = random.choice(list(current_rec_pdbs - { lig_pdb }))
            else:
                crossdock_pdb = "none"

            cur_row = {
                "pdb": lig_pdb,
                "pocket": pocket,
                "lig_smiles": smiles,
                "lig_crystal_file": "/".join(ligfile.split("/")[-2:]),
                "lig_uff_file": "/".join(lig_uff_file.split("/")[-2:]),
            }
            df_list.append(cur_row)

            for prefix, rec_pdb in (("redock", redock_pdb), ("crossdock", crossdock_pdb)):
                recfile = rec_pdb2rec[rec_pdb] if rec_pdb != "none" else "none"
                rec_pocket_file = rec2pocketfile[recfile] if rec_pdb != "none" else "none"
                cur_row[f"{prefix}_rec_file"] = "/".join(recfile.split("/")[-2:]) if rec_pdb != "none" else "none"
                cur_row[f"{prefix}_rec_pocket_file"] = "/".join(rec_pocket_file.split("/")[-2:]) if rec_pdb != "none" else "none"
                cur_row[f"{prefix}_num_pocket_residues"] = rec2res_num[recfile] if rec_pdb != "none" else 0

            center = pocket_centers[pocket]
            bounds = pocket_sizes[pocket]
            for name, num in zip("xyz", [0,1,2]):
                cur_row[f"pocket_center_{name}"] = center[num]
            for name, num in zip("xyz", [0,1,2]):
                cur_row[f"pocket_size_{name}"] = bounds[num]

    struct_df = pd.DataFrame(df_list)

    # now filter
    folder = get_output_dir(cfg)

    struct_df = struct_df.query("lig_uff_file != 'none' and redock_num_pocket_residues >= @max_pocket_residues and crossdock_num_pocket_residues >= @max_pocket_residues and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)
    mask = []
    for i, row in tqdm(struct_df.iterrows(), total=len(struct_df)):
        uff_file = folder + "/" + row.lig_uff_file
        if not os.path.exists(uff_file):
            print("UFF FILE DOESNT EXIST")
            mask.append(False)
            continue

        lig = next(Chem.SDMolSupplier(uff_file, sanitize=True))
        lig = Chem.RemoveHs(lig)

        uff_smiles = Chem.MolToSmiles(lig, False)
        reg_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(row.lig_smiles), False)

        if uff_smiles != reg_smiles:
            print("UFF FILE GOT SMILES WRONG")
            mask.append(False)
            continue

        mask.append(True)
    
    struct_df = struct_df[mask].reset_index(drop=True)
    return struct_df

# force this!
@task(max_runtime=0.1, force=False)
def save_clustered_structs(cfg, struct_df, splits,):
    folder = get_output_dir(cfg)
    struct_df.to_csv(folder + "/structures_all.csv", index=False)
    for split, pockets in splits.items():
        split_struct = struct_df.query("pocket in @pockets").reset_index(drop=True)
        split_struct.to_csv(folder + f"/structures_{split}.csv", index=False)

# force this!
@task(max_runtime=0.1, force=False)
def save_clustered_activities(cfg, activities, splits):
    folder = get_output_dir(cfg)
    activities.to_csv(folder+ f"/activities_all.csv", index=False)
    split2df = {}
    for split, pockets in splits.items():
        split_act = activities.query("pocket in @pockets").reset_index(drop=True)
        split_act.to_csv(folder + f"/activities_{split}.csv", index=False)
        split2df[split] = split_act
    return split2df

@task(max_runtime=0.1, force=False)
def get_clustered_activities_probis(cfg, activities, splits):
    split2df = {}
    for split, pockets in splits.items():
        split_act = activities.query("pocket in @pockets").reset_index(drop=True)
        split2df[split] = split_act
    return split2df

def make_sna_df(df, neg_ratio, smiles2clusters):
    """ Adds (putative) negative examples to the dataframe. Neg_ratio
    ratio of putative inactives to the original df length """

    all_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        all_rows.append(row)
        my_clusters = smiles2clusters[row.lig_smiles]
        for i in range(neg_ratio):
            # find a ligand that isn't known to bind to anything
            # in the cluster
            while True:
                neg_idx = random.randint(0, len(df)-1)
                neg_row = df.iloc[neg_idx]
                neg_clusters = smiles2clusters[neg_row.lig_smiles]
                if len(neg_clusters.intersection(my_clusters)) == 0:
                    break
            new_row = deepcopy(row)
            new_row.lig_smiles = neg_row.lig_smiles
            new_row.lig_file = neg_row.lig_file
            new_row.standard_value = np.nan
            new_row.pchembl_value = np.nan
            new_row.lig_cluster = np.nan
            new_row.active = False
            all_rows.append(new_row)
    sna_df = pd.DataFrame(all_rows).reset_index(drop=True)
    return sna_df

@task(max_runtime=0.1, force=False)
def get_smiles_to_clusters(cfg, activities):
    smiles2clusters = defaultdict(set)
    for i, row in tqdm(activities.iterrows(), total=len(activities)):
        smiles2clusters[row.lig_smiles].add(row.rec_cluster)
    return smiles2clusters

@task(max_runtime=0.1, force=False)
def get_smiles_to_clusters_probis(cfg, activities):
    smiles2clusters = defaultdict(set)
    for i, row in tqdm(activities.iterrows(), total=len(activities)):
        smiles2clusters[row.lig_smiles].add(row.rec_cluster)
    return smiles2clusters

@task(force=False)
def save_all_sna_dfs(cfg, smiles2clusters, split2df, neg_ratio=1):

    out_split2df = {}
    for split, df in split2df.items():
        sna_df = make_sna_df(df, neg_ratio, smiles2clusters)
        out_file = get_output_dir(cfg) + f"/activities_sna_{neg_ratio}_{split}.csv"
        print(f"Saving SNA activities to {out_file}")
        sna_df.to_csv(out_file, index=False)
        out_split2df[split] = sna_df

    return out_split2df

@task(force=False)
def get_all_sna_dfs_probis(cfg, smiles2clusters, split2df, neg_ratio=1):

    out_split2df = {}
    for split, df in split2df.items():
        sna_df = make_sna_df(df, neg_ratio, smiles2clusters)
        out_split2df[split] = sna_df

    return out_split2df

@task(max_runtime=0.1, force=False)
def remove_potencies_from_act(cfg, activities):
    ret = activities.query("standard_type != 'Potency'").reset_index(drop=True)
    print(f"Len before potencies removed: {len(activities)}, len after: {len(ret)}")
    return ret

def make_bigbind_workflow(cfg):
    sifts_zipped = download_sifts()
    sifts_csv = unzip_sifts(sifts_zipped)

    crossdocked_tarred = download_crossdocked()
    cd_dir = untar_crossdocked(crossdocked_tarred)

    lit_pcba_tarred = download_lit_pcba()
    lit_pcba_dir = untar_lit_pcba(lit_pcba_tarred)

    chembl_tarred = download_chembl()
    chembl_db_file = untar_chembl(chembl_tarred)

    con = get_chembl_con(chembl_db_file)
    con = load_sifts_into_chembl(con, sifts_csv)

    cd_rf2lfs_nofix = get_crossdocked_rf_to_lfs(cd_dir)

    cd_uniprots = get_crossdocked_uniprots(con, cd_rf2lfs_nofix)
    activities_unfiltered_fname = get_crossdocked_chembl_activities(con, cd_uniprots)
    activities_unfiltered = load_act_unfiltered(activities_unfiltered_fname)
    saved_act_unf = save_activities_unfiltered(activities_unfiltered)

    activities_filtered = filter_activities(activities_unfiltered)

    smiles2filename = save_all_mol_sdfs(activities_filtered)
    activities_filtered = add_sdfs_to_activities(activities_filtered, smiles2filename)

    chain2uniprot = get_chain_to_uniprot(con)

    (
        uniprot2rfs_nofix,
        uniprot2lfs,
        uniprot2pockets,
        pocket2uniprots,
        pocket2rfs_nofix,
        pocket2lfs,
    ) = get_crossdocked_maps(cd_rf2lfs_nofix, chain2uniprot)

    comp_dict_file = download_comp_dict()
    comp_dict = load_components_dict(comp_dict_file)

    lig_sdfs = download_all_lig_sdfs(uniprot2lfs)
    ligfile2lig = get_all_crystal_ligs(comp_dict, lig_sdfs)

    pocket2rfs_nofix, pocket2lfs, ligfile2lig, ligfile2uff = save_all_structures(
        cd_dir, pocket2uniprots, pocket2rfs_nofix, pocket2lfs, ligfile2lig
    )

    # pocket2rfs_nofix = defaultdict(list)
    # for rf in glob(get_output_dir(cfg) + "/*/*_nofix.pdb"):
    #     pocket = rf.split("/")[-2]
    #     pocket2rfs_nofix[pocket].append("/".join(rf.split("/")[-2:]))

    pocket2rfs = fix_all_recfiles(pocket2rfs_nofix)

    rf2pocketfile, rf2res_num = save_all_pockets(pocket2rfs, pocket2lfs, ligfile2lig)
    pocket_centers, pocket_sizes = get_all_pocket_bounds(pocket2lfs, ligfile2lig)

    lig_smi, lig_fps = get_morgan_fps_parallel(activities_filtered)
    lig_sim_mat = get_tanimoto_matrix(lig_fps)

    activities = make_final_activities_df(
        activities_filtered,
        uniprot2pockets,
        pocket2rfs,
        rf2pocketfile,
        rf2res_num,
        pocket_centers,
        pocket_sizes,
    )

    # pocket_tm_scores = get_fake_pocket_tm_scores(rf2pocketfile)
    pocket_tm_scores = get_all_pocket_tm_scores(rf2pocketfile)

    # now let's find the optimal splits
    full_lig_sim_mat = get_full_tanimoto_matrix(activities, lig_smi, lig_sim_mat)
    pocket_indexes = get_pocket_indexes(activities)
    poc_sim = get_pocket_similarity(pocket_tm_scores)
    
    tan_cutoffs, tm_cutoffs, prob_ratios = get_lig_rec_edge_prob_ratios(activities, full_lig_sim_mat, poc_sim, pocket_indexes)
    plotted_prob_ratios = plot_prob_ratios(tan_cutoffs, tm_cutoffs, prob_ratios)

    tm_cutoff, poc_clusters_no_tan = get_pocket_clusters(activities, tm_cutoffs, prob_ratios, poc_sim, pocket_indexes)
    poc_clusters = get_pocket_clusters_with_tanimoto(full_lig_sim_mat, tm_cutoffs, tan_cutoffs, prob_ratios, poc_sim, pocket_indexes)


    lit_pcba_pockets = get_lit_pcba_pockets(con, lit_pcba_dir, uniprot2pockets)
    splits = get_splits(activities, poc_clusters, lit_pcba_pockets, pocket_indexes)

    # ew -- yes this should go waaayyy earlier but I don't have time to recompute
    # all the docking results for BayesBind

    lig_cluster_idxs = get_lig_clusters(activities, pocket_indexes, lig_smi, lig_sim_mat)
    activities = add_all_clusters_to_act(activities, lig_cluster_idxs, pocket_indexes, poc_clusters)


    activities_no_pot = remove_potencies_from_act(activities)

    struct_df = create_struct_df(pocket2rfs,
                                 pocket2lfs,
                                 ligfile2lig,
                                 ligfile2uff,
                                 rf2pocketfile,
                                 rf2res_num,
                                 pocket_centers,
                                 pocket_sizes)
    
    split2act_df = save_clustered_activities(activities_no_pot, splits)
    saved_struct = save_clustered_structs(struct_df, splits)

    # SNA
    smiles2clusters = get_smiles_to_clusters(activities)
    split2sna_df = save_all_sna_dfs(smiles2clusters, split2act_df, neg_ratio=1)

    # Probis stuff
    rec2srf = create_all_probis_srfs(rf2pocketfile)
    rep_srf2nosql = find_representative_rec(pocket2rfs, rec2srf)
    rep_scores = convert_intra_results_to_json(rec2srf, rep_srf2nosql)
    pocket2rep_rec = get_rep_recs(pocket2rfs, rep_scores)
    srf2nosql = find_all_probis_distances(pocket2rep_rec, rec2srf)
    probis_scores = convert_inter_results_to_json(rec2srf, srf2nosql)

    poc_sim_probis = get_pocket_similarity_probis(probis_scores, pocket2rep_rec)
    tan_cutoffs_probis, probis_cutoffs, prob_ratios_probis = get_lig_rec_edge_prob_ratios_probis(activities, full_lig_sim_mat, poc_sim_probis, pocket_indexes)
    plotted_prob_ratios_probis = plot_prob_ratios_probis(tan_cutoffs_probis, probis_cutoffs, prob_ratios_probis)

    probis_cutoff, poc_clusters_probis_no_tan = get_pocket_clusters_probis(activities, probis_cutoffs, prob_ratios_probis, poc_sim_probis, pocket_indexes)
    poc_clusters_probis = get_pocket_clusters_with_tanimoto_probis(full_lig_sim_mat, probis_cutoffs, tan_cutoffs, prob_ratios_probis, poc_sim_probis, pocket_indexes)
    
    splits_probis = get_splits_probis(activities, poc_clusters_probis, lit_pcba_pockets, pocket_indexes)
    split2act_df_probis = get_clustered_activities_probis(activities, splits_probis)
    smiles2clusters_probis = get_smiles_to_clusters_probis(activities)
    split2sna_df_probis = get_all_sna_dfs_probis(smiles2clusters_probis, split2act_df_probis, neg_ratio=1)

    # Finally, compare probis vs pocket tm splits by seeing KNN performance
    tm_vs_probis = compare_probis_and_pocket_tm(poc_clusters,
                                                poc_clusters_probis,
                                                split2act_df,
                                                split2sna_df,
                                                split2act_df_probis,
                                                split2sna_df_probis,
                                                lig_smi, 
                                                lig_sim_mat,
                                                poc_sim,
                                                poc_sim_probis,
                                                tan_cutoffs,
                                                tm_cutoffs,
                                                probis_cutoffs,
                                                prob_ratios,
                                                prob_ratios_probis)

    # BayesBind!

    saved_bayesbind = make_all_bayesbind(split2act_df, lig_smi, lig_sim_mat, poc_clusters)

    tan_tm_coefs = get_optimal_tan_tm_coefs(tan_cutoffs, tm_cutoffs, prob_ratios)

    return Workflow(
        cfg,
        rf2pocketfile,
        saved_act_unf,
        plotted_prob_ratios,
        plotted_prob_ratios_probis,
        split2act_df,
        split2sna_df,
        saved_struct,
        saved_bayesbind,
        plotted_prob_ratios_probis,
        tm_vs_probis,
        tan_tm_coefs,
    )


@task()
def error(cfg):
    raise Exception("!! This is a test !!")


@task()
def task1(cfg):
    print("!!!1!!!")


@task()
def task2(cfg, x):
    print("!!!2!!!")


if __name__ == "__main__":
    import sys
    from utils.cfg_utils import get_config

    cfg = get_config(sys.argv[1])
    workflow = make_bigbind_workflow(cfg)
    # for node in workflow.nodes:
    #     print(node)

    # workflow = Workflow(error())
    # workflow.run_node(cfg, workflow.nodes[0])

    # for node in workflow.nodes:
    #     print(node)

    workflow.prev_run_name = "v2_fixed"
    workflow.run()

    # cd_nodes = workflow.out_nodes # find_nodes("untar_crossdocked")
    # levels = workflow.get_levels(cd_nodes)
    # print(levels)
