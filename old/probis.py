import subprocess
import yaml
import json
import os
from tqdm import tqdm
from traceback import print_exc

from old.cache import cache

def get_all_res_nums(pocket_file):
    """ Return the set of all residue numbers in the pocket """
    ret = set()
    with open(pocket_file, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                resn = int(line[22:26])
                ret.add(resn)

    return ret

@cache
def create_all_probis_srfs(cfg, rec2pocketfile, rigorous=True):
    """ Create probis surface files for all pockets for faster execution """
    rec2srf = {}
    for rec, pocketfile in tqdm(rec2pocketfile.items()):
        # if not ("PK3CG_HUMAN_143_1102_0" in rec or "BEV1A_BETPN_2_160_0" in rec): continue
        pocket, file_pre = pocketfile.split(".")[0].split("/")[-2:]
        chain = rec.split("_")[-2]
        outfile = cfg["cache_folder"] + "/" + file_pre + "_" + pocket + ".srf"
        res_numbers = get_all_res_nums(pocketfile)
        motif_str = f"[:{chain} and ({','.join(map(str, res_numbers))})]"
        # probis -extract -f1 $f -c1 A -srffile $f.srf
        cmd = ["probis", "-extract", "-motif", motif_str, "-f1", rec, "-c1", chain, "-srffile", outfile]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        try:
            proc.check_returncode()
        except KeyboardInterrupt:
            raise
        except:
            if rigorous:
                raise
            print_exc()
        rec2srf[rec] = outfile
    return rec2srf

@cache
def find_representative_rec(cfg, pocket2recs, rec2srf, n_cpu=8, rigorous=True):
    """ For each pocket, find a receptor most representative of the pocket
    (that is, a receptor file whose probis distance from the other
    receptors of the same pocket)
    """

    srf2nosql = {}
    for pocket, recs in tqdm(pocket2recs.items()):
        srf_fname = cfg["cache_folder"] + f"/find_representative_rec_{pocket}.txt"
        srfs = { rec2srf[rec] for rec in recs }
        with open(srf_fname, "w") as f:
            for rec in recs:
                srf = rec2srf[rec]
                chain = rec.split("_")[-2]
                f.write(srf + " " + chain + "\n")


        for rec in recs:
            srf = rec2srf[rec]
            chain = rec.split("_")[-2]
            nosql_file = srf.split(".")[0] + "find_representative_rec.nosql"

            cmd = [ "probis", "-ncpu", str(n_cpu), "-surfdb", "-local", "-sfile", srf_fname, "-f1", srf, "-c1", chain, "-nosql", nosql_file ]
            # print(cmd)
            # continue
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            try:
                proc.check_returncode()
            except KeyboardInterrupt:
                raise
            except:
                if rigorous:
                    raise
                print_exc()
                
            srf2nosql[srf] = nosql_file

    return srf2nosql

@cache
def find_all_probis_distances(cfg, pocket2rep_rec, rec2srf, n_cpu=12, rigorous=True):
    """ For each pair of srf files, use probis to get the pocket
    similarity between them """
    srf2nosql = {}
    srf_fname = cfg["cache_folder"] + "/find_all_probis_distances_srfs.txt"
    with open(srf_fname, "w") as f:
        for pocket, rec in pocket2rep_rec.items():
            srf = rec2srf[rec]
            chain = rec.split("_")[-2]
            f.write(srf + " " + chain + "\n")
    # command from the probis tutorial pdf:
    # ./probis -ncpu 8 -surfdb -sfile srfs.txt -f1 1phrA.srf -c1 A -nosql example.nosql

    for pocket, rec in tqdm(pocket2rep_rec.items()):
        srf = rec2srf[rec]
        chain = rec.split("_")[-2]
        nosql_file = srf.split(".")[0] + ".nosql"
        if os.path.exists(nosql_file):
            srf2nosql[srf] = nosql_file
            continue
        cmd = [ "probis", "-ncpu", str(n_cpu), "-surfdb", "-local", "-sfile", srf_fname, "-f1", srf, "-c1", chain, "-nosql", nosql_file ]
        # print(cmd)
        # continue
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        try:
            proc.check_returncode()
        except KeyboardInterrupt:
            raise
        except:
            if rigorous:
                raise
            print_exc()
        srf2nosql[srf] = nosql_file

    return srf2nosql

def get_scores_from_probis_json(probis_results):
    """ Input: mapping from rec filename to probis json result
    output: dict mapping (rec1, rec2) pairs to the z-score between them """
    pdb_chain2rec = {}
    for rec in probis_results.keys():
        pdb, chain = rec.split("/")[-1].split("_")[-3:-1]
        pdb_chain2rec[(pdb, chain)] = rec

    scores = {}
    for rec, fname in tqdm(probis_results.items()):
        with open(fname, "r") as f:
            items = json.load(f)
        for item in items:
            pdb = item["pdb_id"]
            chain = item["chain_id"]
            try:
                rec2 = pdb_chain2rec[(pdb, chain)]
            except KeyError:
                continue
            zscores = []
            for align in item["alignment"]:
                zscores.append(align["scores"]["z_score"])
            scores[(rec, rec2)] = zscores[0]
            
    return scores

def convert_probis_results_to_json(cfg, rec2srf, srf2nosql, rigorous):
    """ probis outputs a weird nosql file format, need to covert it to
    json """
    rec2json = {}
    for rec, srf in tqdm(rec2srf.items()):
        if srf not in srf2nosql: continue
        nosql = srf2nosql[srf]
        json_out = nosql.split(".")[0] + ".json"
        if os.path.exists(json_out):
            # smhhh the full json list is too long. Just use filename
            rec2json[rec] = json_out
            # with open(json_out, "r") as f:
            #     rec2json[rec] = json.load(f)
            continue
        
        chain = rec.split("_")[-2]
        # probis -results -f1 1phr.pdb -c1 A -nosql example.nosql -json example.json
        cmd = [ "probis", "-results", "-f1", rec, "-c1", chain, "-nosql", nosql, "-json", json_out, "z_score", "0.0" ]
        # print(" ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        try:
            proc.check_returncode()
            if os.path.exists(json_out):
                rec2json[rec] = json_out
            # with open(json_out, "r") as f:
            #     rec2json[rec] = json.load(f)
        except KeyboardInterrupt:
            raise
        except:
            if rigorous:
                raise
            print_exc()

    return get_scores_from_probis_json(rec2json)

# we use probis twice in total -- once to find the representative rec,
# and once to use the rep. recs to find the inter-pocket distances
@cache
def convert_intra_results_to_json(cfg, rec2srf, srf2nosql, rigorous=False):
    return convert_probis_results_to_json(cfg, rec2srf, srf2nosql, rigorous=rigorous)

@cache
def convert_inter_results_to_json(cfg, rec2srf, srf2nosql, rigorous=False):
    return convert_probis_results_to_json(cfg, rec2srf, srf2nosql, rigorous=rigorous)

@cache
def get_rep_recs(cfg, pocket2recs, scores):
    """ Returns map from pocket to rep. rec """
    invalid_pockets = set()
    ret = {}
    for pocket, recs in pocket2recs.items():
        rec2scores = {}
        for rec in recs:
            score_list = []
            for rec2 in recs:
                key = (rec, rec2)
                # probis doesn't always give us a number if
                # the alignment sucks. So we assume z-score is 0
                score = scores[key] if key in scores else 0
                score_list.append(score)
            rec2scores[rec] = sum(score_list)
        rep_rec, score = max(rec2scores.items(), key=lambda x: x[1])
        # print(rep_rec, score)
        if score > 0:
            ret[pocket] = rep_rec
    return ret

if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    from run import save_all_pockets, save_all_structures
    pocket2recs,\
        pocket2ligs,\
        ligfile2lig = save_all_structures(cfg)
    rec2pocketfile, rec2res_num = save_all_pockets(cfg)
    rec2srf = create_all_probis_srfs(cfg, rec2pocketfile)
    rep_srf2nosql = find_representative_rec(cfg, pocket2recs, rec2srf)
    rep_scores = convert_intra_results_to_json(cfg, rec2srf, rep_srf2nosql)
    pocket2rep_rec = get_rep_recs(cfg, pocket2recs, rep_scores)
    srf2nosql = find_all_probis_distances(cfg, pocket2rep_rec, rec2srf)
    full_scores = convert_inter_results_to_json(cfg, rec2srf, srf2nosql)
    
