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
import matplotlib.pyplot as plt

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
    activities = pd.read_csv(cfg["bigbind_folder"] + "/activities_all.csv")
    uniprot2family  = get_uniprot_to_family(cfg)
    family2num = defaultdict(int)
    for uniprot in tqdm(uniprot2family.keys()):
        fam = uniprot2family[uniprot]
        family2num[uniprot2family[uniprot]] += sum(activities.uniprot == uniprot)
    return family2num

def get_class_to_num(cfg, fam2num):
    class2num = defaultdict(int)
    for fam, num in fam2num.items():
        if "kinase" in fam:
            cls = "Kinase"
        elif "G-protein coupled receptor" in fam:
            cls = "GPCR"
        elif "cytochrome P450" in fam:
            cls = "CYP450"
        else:
            cls = "Other"
        class2num[cls] += num
    return class2num

def analysis(cfg):
    fam2num = get_family_to_num(cfg)
    class2num = get_class_to_num(cfg, fam2num)

    df = pd.read_csv(cfg["bigbind_folder"] + "/activities_all.csv")

    comp_targets = defaultdict(int)
    for smiles in tqdm(df.lig_smiles):
        comp_targets[smiles] += 1
    comp_targets = list(comp_targets.values())

    poc_compounds = []
    for pocket in tqdm(df.pocket.unique()):
        poc_compounds.append(len(df.query("pocket == @pocket")))

    print(f"Number of activities: {len(df)}")
    print(f"Number of pockets: {len(df.pocket.unique())}")
    print(f"Number of 3D pocket structures: {len(df.ex_rec_file.unique())}")
    print(f"Average number of structures per pocket: {len(df.ex_rec_file.unique())/len(df.pocket.unique()):.2f}")
    print(f"Number of unique compounds: {len(df.lig_smiles.unique())}")
    print(f"pChEMBL mean: {df.pchembl_value.mean():.2f}")
    print(f"pChEMBL std: {df.pchembl_value.std():.2f}")
    print(f"targets per compound mean: {np.mean(comp_targets):.2f}")
    print(f"targets per compound std: {np.std(comp_targets):.2f}")
    print(f"compounds per target mean: {np.mean(poc_compounds):.2f}")
    print(f"compounds per target std: {np.std(poc_compounds):.2f}")

    comp_with_single_target = sum(np.array(comp_targets) == 1)/len(comp_targets)
    print(f"fraction of compounds with a single target ({comp_with_single_target*100:.0f}")

    tot = sum(class2num.values())
    print("Target family percentages:")
    for cls, num in class2num.items():
        frac = num/tot
        print(f"  {cls}: {frac*100:.0f}")

    fig, axs = plt.subplots(2,2)
    axs[0,0].hist(df.pchembl_value)
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel('pChEMBL value')
    axs[0,0].set_ylabel('Number of compounds')

    bins = np.logspace(0, 2, 10)
    axs[1,1].hist(comp_targets, bins=bins)
    axs[1,1].set_yscale('log')
    axs[1,1].set_xscale('log')
    axs[1,1].set_ylabel('Number of compounds')
    axs[1,1].set_xlabel('Targets per compound')

    bins = np.logspace(0, 5, 10)
    axs[1,0].hist(poc_compounds, bins=bins)
    axs[1,0].set_yscale('log')
    axs[1,0].set_xscale('log')
    axs[1,0].set_xlabel('Compounds per target')
    axs[1,0].set_ylabel('Number of targets')

    axs[0,1].pie(class2num.values(), labels=class2num.keys())
    axs[0,1].axis('equal')
    axs[0,1].set_xlabel("Target protein families", labelpad=10)

    fig.tight_layout()
    fig.set_size_inches(6, 4)

    os.makedirs("./outputs", exist_ok=True)
    out_file = "./outputs/histograms.pdf"
    print(f"Writing figure to {out_file}")
    fig.savefig(out_file, dpi=300)

if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    analysis(cfg)