import yaml
import random
import json
import numpy as np
import requests
import json
import re
from copy import copy
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from collections import defaultdict
from cache import cache, item_cache
from probis import *
from run import *

@item_cache
def get_protein_family(cfg, uniprot):
    url = f'https://rest.uniprot.org/uniprotkb/search?query=accession:{uniprot}'
    results = requests.get(url).text
    res_json = json.loads(results)
    ret = None
    if "results" not in res_json or len(res_json["results"]) == 0 or "comments" not in res_json["results"][0]: return "other"
    for comment in res_json["results"][0]["comments"]:
        if "texts" not in comment: continue
        for text in comment["texts"]:
            val = text["value"]
            re_res = re.search("Belongs to the (.+) family", val)
            if re_res is not None:
                if ret is not None:
                    print("WARNING: this protein has two families: ", ret, re_res.groups(0)[0])
                ret = re_res.groups(0)[0]
    return ret if ret is not None else "other"

@cache
def get_uniprot_to_family(cfg):
    uniprots = set(get_crossdocked_uniprots(cfg).SP_PRIMARY)
    uniprot2family = {}
    for uniprot in tqdm(uniprots):
        uniprot2family[uniprot] = get_protein_family(cfg, uniprot)
    return uniprot2family

@cache
def get_family_to_num(cfg):
    activities = add_sdfs_to_activities(cfg)
    uniprot2family  = get_uniprot_to_family(cfg)
    family2num = defaultdict(int)
    for uniprot in tqdm(uniprot2family.keys()):
        family2num[uniprot2family[uniprot]] += sum(activities.protein_accession == uniprot)
    return family2num

def analysis(cfg):
    fam2num = get_family_to_num(cfg)
    print(fam2num.keys())

if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    analysis(cfg)