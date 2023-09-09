from Bio import SeqIO
from Bio import pairwise2
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder, is_aa
import numpy as np
import sys
from functools import reduce, lru_cache
from tqdm import tqdm
from traceback import print_exc

from utils.task import task, iter_task, simple_task
from utils.cache import cache

def get_all_res_nums(pocket_file):
    """ Return the set of all residue numbers in the pocket """
    ret = set()
    with open(pocket_file, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                resn = int(line[22:26])
                ret.add(resn)

    return ret

def get_alpha_and_beta_coords(structure):
    """ Returns the coords (as np arrays) of both the alpha carbons
    and (virtual) beta carbons for the structure """
    alpha_carbon_coordinates = []
    beta_coords = []

    ideal_bond_length = 1.52 

    # Iterate through atoms and extract alpha carbon coordinates
    # also get idealized beta carbon coords (to include glycine)
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue and "N" in residue and "C" in residue:
                    alpha_carbon = residue["CA"]  # Standard amino acids
                    alpha_carbon_coordinates.append(alpha_carbon.get_coord())

                    # Calculate virtual C-beta coordinates for glycine
                    n_atom = residue["N"]
                    c_atom = residue["C"]
                    ca_atom = residue["CA"]
                    
                    # Calculate unit vector
                    nc_vector = n_atom.get_vector() - c_atom.get_vector()
                    nc_vector.normalize()
                    
                    # Compute C-beta position
                    cb_coord = np.array(ca_atom.get_vector().get_array()) + np.array(nc_vector.get_array()) * ideal_bond_length
                    beta_coords.append(cb_coord)

    # Convert the coordinates list to a NumPy array
    alpha_carbon_coordinates_array = np.array(alpha_carbon_coordinates)
    beta_coords = np.array(beta_coords)

    # Print the NumPy array of alpha carbon coordinates
    return alpha_carbon_coordinates_array, beta_coords

def get_aligned_coords(ref, other, idx):

    cur_ref = ref[idx]
    cur_other = other[idx]

    # Calculate centroids
    centroid_ref = np.mean(cur_ref, axis=0)
    centroid_other = np.mean(cur_other, axis=0)

    # Calculate cross-covariance matrix
    cross_covariance = np.dot((cur_other - centroid_other).T, cur_ref - centroid_ref)

    # Perform SVD
    U, _, Vt = np.linalg.svd(cross_covariance)
    rotation_matrix = np.dot(Vt.T, U.T)

    return np.dot(rotation_matrix, (other - centroid_other).T).T + centroid_ref

pdb_parser = PDBParser(QUIET=True)
def get_struct(rf):
    return pdb_parser.get_structure("1", rf)

@simple_task
# @task(max_runtime=3, mem=128, num_outputs=2)
def get_all_structs_and_res_nums(cfg, rec2pocketfile):
    structs = {}
    res_nums = {}
    for rf, pf in tqdm(rec2pocketfile.items()):
        structs[rf] = get_struct(rf)
        res_nums[pf] = get_all_res_nums(pf)
    return structs, res_nums

OVERLAP_CUTOFF = 5
# @cache(lambda cfg, r1, r2, r1_poc_file, r2_poc_file: (r1, r2), disable=True, version=1.0)
def pocket_tm_score(cfg, protein1_pdb, protein2_pdb, poc1_res, poc2_res):
    """ Aligns just the pockets of r1 and r2 and returns the TM score
    of the pocket residues (using Calpha and idealized Cbeta coords) """

    # chatGPT helped with this lol

    # # Load PDB structures using PDB IDs
    # pdb_parser = PDBParser(QUIET=True)
    # protein1_pdb = get_struct(r1)
    # protein2_pdb = get_struct(r2)

    # Align PDB structures based on the aligned sequences
    ppb = PPBuilder()
    pp_list1 = ppb.build_peptides(protein1_pdb[0])
    pp_list2 = ppb.build_peptides(protein2_pdb[0])

    # Create a dictionary mapping residue number to sequence index for each peptide
    # seq_index_mapping1 = {res.id[1]: i for i, res in enumerate(pp_list1[0]) if is_aa(res)}
    # seq_index_mapping2 = {res.id[1]: i for i, res in enumerate(pp_list2[0]) if is_aa(res)}

    seq_index_mapping1 = {}
    i = 0
    for pp in pp_list1:
        for res in pp:
            if is_aa(res):
                seq_index_mapping1[res.id[1]] = i
                i += 1

    seq_index_mapping2 = {}
    i = 0
    for pp in pp_list2:
        for res in pp:
            if is_aa(res):
                seq_index_mapping2[res.id[1]] = i
                i += 1

    p1_seq = reduce(lambda x, y: x+y, [p.get_sequence() for p in pp_list1])
    p2_seq = reduce(lambda x, y: x+y, [p.get_sequence() for p in pp_list2])

    alignment = pairwise2.align.globalxx(p1_seq, p2_seq)
    best_alignment = alignment[0]  # Get the best alignment

    # Extract the aligned sequences and alignment score
    aligned_seq1 = best_alignment[0]
    aligned_seq2 = best_alignment[1]

    # mapping between r1 residues and r2 residues
    rec1_to_rec2 = {}
    rec2_to_rec1 = {}
    s1 = 0
    s2 = 0
    for res1, res2 in zip(aligned_seq1, aligned_seq2):
        if res1 != '-' and s2 != '-':
            rec1_to_rec2[int(s1)] = int(s2)
            rec2_to_rec1[int(s2)] = int(s1)
        if res1 != '-' :
            s1 += 1
        if res2 != '-':
            s2 += 1

    # poc1_res = get_all_res_nums(r1_poc_file)
    # poc2_res = get_all_res_nums(r2_poc_file)
    poc1_res = { seq_index_mapping1[i] for i in poc1_res }
    poc2_res = { seq_index_mapping2[i] for i in poc2_res }

    all_rec1_indexes = poc1_res.intersection({rec2_to_rec1[idx] for idx in poc2_res if idx in rec2_to_rec1})
    all_rec2_indexes = poc2_res.intersection({rec1_to_rec2[idx] for idx in poc1_res if idx in rec1_to_rec2})
    all_rec1_indexes = { idx for idx in all_rec1_indexes if idx in rec1_to_rec2 and rec1_to_rec2[idx] in all_rec2_indexes }
    all_rec2_indexes = { idx for idx in all_rec2_indexes if idx in rec2_to_rec1 and rec2_to_rec1[idx] in all_rec1_indexes }

    assert len(all_rec1_indexes) == len(all_rec2_indexes)


    # yeet outta here if the pockets don't overlap
    if len(all_rec1_indexes) < OVERLAP_CUTOFF:
        return 0.0
    
    all_rec1_indexes = np.array(sorted(all_rec1_indexes))
    all_rec2_indexes = np.array([rec1_to_rec2[idx] for idx in all_rec1_indexes])

    rec1_coords, rec1_beta = get_alpha_and_beta_coords(protein1_pdb)
    rec2_coords, rec2_beta = get_alpha_and_beta_coords(protein2_pdb)

    # r1 = rec1_coords[all_rec1_indexes]
    # r2 = rec2_coords[all_rec2_indexes]

    r1 = np.concatenate([rec1_coords[all_rec1_indexes], rec1_beta[all_rec1_indexes]], 0)
    r2 = np.concatenate([rec2_coords[all_rec2_indexes], rec2_beta[all_rec2_indexes]], 0)

    align_idx = np.ones(len(r1), dtype=bool)

    for i in range(10):
        aligned_r2 = get_aligned_coords(r1, r2, align_idx)

        L = r1.shape[0]
        d = np.sqrt(((r1 - aligned_r2)**2).sum(-1))

        d0 = (L**0.39)*np.sqrt(1 - 0.42 + 0.05*L*np.exp(-L/4.7) - 0.63*np.exp(-L/37)) - 0.75
        # the algorithm recommends only realigning the atoms below d0 distance. That can
        # result in very few atoms to align, so we align the top third atoms
        low_d = max(np.sort(d)[int(len(d)*0.3)], d0)

        score = (1/(1 + (d/d0)**2)).sum()/L

        new_align_idx = d < low_d
        if np.all(align_idx == new_align_idx):
            break
        align_idx = new_align_idx

    return score

@task()
def get_all_pocket_tm_scores(cfg, rec2pocketfile):
    all_recs = list(rec2pocketfile.keys())
    all_pairs = [ (all_recs[i], all_recs[j]) for j in range(len(all_recs)) for i in range(j) ]
    ret = {}
    for r1, r2 in tqdm(all_pairs):
        p1 = rec2pocketfile[r1]
        p2 = rec2pocketfile[r2]
        try:
            ret[(r1, r2)] = pocket_tm_score(r1, r2, p1, p2)
        except:
            print(f"Error computing TM score bwteen {r1} and {r2}")
            print_exc()
    return ret

@simple_task
def get_all_rec_pairs(cfg, rec2pocketfile, recfile2struct, pocfile2res_num):
    all_recs = list(rec2pocketfile.keys())
    return [ (all_recs[i],
              all_recs[j],
              recfile2sturct[all_recs[i]], 
              recfile2struct[all_recs[j]],
              pocfile2res_num[rec2pocketfile[all_recs[i]]],
              pocfile2res_num[rec2pocketfile[all_recs[j]]]) for j in range(len(all_recs)) for i in range(j) ]

@simple_task
def postproc_tm_outputs(cfg, all_pairs, tm_scores):
    ret = {}
    for (r1, r2, *rest), score in zip(all_pairs, tm_scores):
        ret[(r1, r2)] = score

# @iter_task(600, 24*4*600, n_cpu=16, mem=32)
def compute_single_tm_score(cfg, item):
    r1, r2, s1, s2, p1, p2 = item
    try:
        return pocket_tm_score(cfg, s1, s2, p1, p2)
    except KeyboardInterrupt:
        raise
    except:
        print(f"Error computing TM score bewteen {r1} and {r2}", file=sys.stderr)
        print_exc()
        return 0

# compute_all_tm_scores = iter_task(1000, 48*1000, n_cpu=8, mem=32)(compute_single_tm_score)
compute_all_tm_scores = iter_task(1, 48, n_cpu=224, mem=128)(compute_single_tm_score)

def get_all_pocket_tm_scores(rec2pocketfile, recfile2struct, pocfile2res_num):
    pairs = get_all_rec_pairs(rec2pocketfile, recfile2struct, pocfile2res_num)
    scores = compute_all_tm_scores(pairs)
    return postproc_tm_outputs(pairs, scores)

