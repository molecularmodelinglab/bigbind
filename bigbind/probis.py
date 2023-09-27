import json
import os
import subprocess
from tqdm import tqdm
from traceback import print_exc
from utils.cfg_utils import get_output_dir

from utils.task import simple_task, task

def get_probis_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "probis")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_all_res_nums(pocket_file):
    """ Return the set of all residue numbers in the pocket """
    ret = set()
    with open(pocket_file, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                resn = int(line[22:26])
                ret.add(resn)

    return ret

@task()
def create_all_probis_srfs(cfg, rec2pocketfile, rigorous=True):
    """ Create probis surface files for all pockets for faster execution """
    rec2srf = {}
    for rec, pocketfile in tqdm(rec2pocketfile.items()):
        # guess probis can't handle pqr files
        full_rec = os.path.join(get_output_dir(cfg), rec).replace(".pdb", "_nofix.pdb")
        full_pocket = os.path.join(get_output_dir(cfg), pocketfile)
        pocket, file_pre = pocketfile.split(".")[0].split("/")[-2:]
        chain = rec.split("_")[-2]
        outfile = get_probis_dir(cfg) + "/" + file_pre + "_" + pocket + ".srf"
        res_numbers = get_all_res_nums(full_pocket)
        motif_str = f"[:{chain} and ({','.join(map(str, res_numbers))})]"
        # probis -extract -f1 $f -c1 A -srffile $f.srf

        cmd = ["probis", "-extract", "-motif", motif_str, "-f1", full_rec, "-c1", chain, "-srffile", outfile]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        try:
            proc.check_returncode()
        except KeyboardInterrupt:
            raise
        except:
            if rigorous:
                raise
            print_exc()
        rec2srf[rec] = outfile.split("/")[-1]
    return rec2srf

rep_rec_cpus = 8
@task(n_cpu=rep_rec_cpus)
def find_representative_rec(cfg, pocket2recs, rec2srf, rigorous=True):
    """ For each pocket, find a receptor most representative of the pocket
    (that is, a receptor file whose probis distance from the other
    receptors of the same pocket)
    """

    srf2nosql = {}
    for pocket, recs in tqdm(pocket2recs.items()):
        srf_fname = get_probis_dir(cfg) + f"/find_representative_rec_{pocket}.txt"
        # srfs = { rec2srf[rec] for rec in recs }
        with open(srf_fname, "w") as f:
            for rec in recs:
                srf = get_probis_dir(cfg) + "/" + rec2srf[rec]
                chain = rec.split("_")[-2]
                f.write(srf + " " + chain + "\n")


        for rec in recs:
            srf = get_probis_dir(cfg) + "/" + rec2srf[rec]
            chain = rec.split("_")[-2]
            nosql_file = srf.split(".")[0] + "find_representative_rec.nosql"

            cmd = [ "probis", "-ncpu", str(rep_rec_cpus), "-surfdb", "-local", "-sfile", srf_fname, "-f1", srf, "-c1", chain, "-nosql", nosql_file ]
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
                
            srf2nosql[srf.split("/")[-1]] = nosql_file.split("/")[-1]

    return srf2nosql

probis_cpus = 48
@task(n_cpu=rep_rec_cpus, force=False)
def find_all_probis_distances(cfg, pocket2rep_rec, rec2srf, n_cpu=probis_cpus, rigorous=True):
    """ For each pair of srf files, use probis to get the pocket
    similarity between them """
    srf2nosql = {}
    srf_fname = get_probis_dir(cfg) + "/find_all_probis_distances_srfs.txt"
    with open(srf_fname, "w") as f:
        for pocket, rec in pocket2rep_rec.items():
            srf = get_probis_dir(cfg) + "/" + rec2srf[rec]
            chain = rec.split("_")[-2]
            f.write(srf + " " + chain + "\n")
    # command from the probis tutorial pdf:
    # ./probis -ncpu 8 -surfdb -sfile srfs.txt -f1 1phrA.srf -c1 A -nosql example.nosql

    for pocket, rec in tqdm(pocket2rep_rec.items()):
        srf = get_probis_dir(cfg) + "/" + rec2srf[rec]
        chain = rec.split("_")[-2]
        nosql_file = srf.split(".")[0] + ".nosql"
        # if os.path.exists(nosql_file):
        #     srf2nosql[srf] = nosql_file
        #     continue
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
        srf2nosql[srf.split("/")[-1]] = nosql_file.split("/")[-1]

    return srf2nosql

def get_scores_from_probis_json(cfg, probis_results):
    """ Input: mapping from rec filename to probis json result
    output: dict mapping (rec1, rec2) pairs to the z-score between them """
    pdb_chain2rec = {}
    for rec in probis_results.keys():
        pdb, chain = rec.split("/")[-1].split("_")[-3:-1]
        pdb_chain2rec[(pdb, chain)] = rec

    scores = {}
    for rec, fname in tqdm(probis_results.items()):
        fname = get_probis_dir(cfg) + "/" + fname
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
        nosql = get_probis_dir(cfg) + "/" + srf2nosql[srf]
        json_out = nosql.split(".")[0] + ".json"
        if os.path.exists(json_out):
            # smhhh the full json list is too long. Just use filename
            rec2json[rec] = json_out.split("/")[-1]
            # with open(json_out, "r") as f:
            #     rec2json[rec] = json.load(f)
            # continue
        
        full_rec = os.path.join(get_output_dir(cfg), rec).replace(".pdb", "_nofix.pdb")
        chain = rec.split("_")[-2]
        # probis -results -f1 1phr.pdb -c1 A -nosql example.nosql -json example.json
        cmd = [ "probis", "-results", "-f1", full_rec, "-c1", chain, "-nosql", nosql, "-json", json_out, "z_score", "0.0" ]
        # print(" ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        try:
            proc.check_returncode()
            if os.path.exists(json_out):
                rec2json[rec] = json_out.split("/")[-1]
            # with open(json_out, "r") as f:
            #     rec2json[rec] = json.load(f)
        except KeyboardInterrupt:
            raise
        except:
            if rigorous:
                raise
            print_exc()

    return get_scores_from_probis_json(cfg, rec2json)

@task(max_runtime=0.2, force=False)
def convert_intra_results_to_json(cfg, rec2srf, srf2nosql, rigorous=False):
    return convert_probis_results_to_json(cfg, rec2srf, srf2nosql, rigorous=rigorous)

@task(max_runtime=0.2, force=False)
def convert_inter_results_to_json(cfg, rec2srf, srf2nosql, rigorous=False):
    return convert_probis_results_to_json(cfg, rec2srf, srf2nosql, rigorous=rigorous)

@task(max_runtime=0.2, force=False)
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
