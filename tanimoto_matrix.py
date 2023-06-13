from multiprocessing import Pool
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from cache import cache


@cache
def get_morgan_fps(cfg, df, radius=4, bits=1024):
    fps = []
    for smi in tqdm(df.lig_smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=bits)
        fps.append(np.array(fp, dtype=bool))
    fps = np.asarray(fps)


def get_fp(smi, radius=4, bits=1024):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=bits)
    return np.array(fp, dtype=bool)

# like get_morgan_fps, but parallelized
@cache
def get_morgan_fps_parallel(cfg, df):
    fps = []
    with Pool(cfg['num_cpus']) as p:
        fps = list(tqdm(p.imap(get_fp, df.lig_smiles), total=len(df)))
    fps = np.asarray(fps)
    return fps
    
def main(cfg):
    df = pd.read_csv(f"{cfg['bigbind_folder']}/activities_all.csv")
    fps = get_morgan_fps_parallel(cfg, df)
    print(len(fps))

SEED = 49
random.seed(SEED)
np.random.seed(SEED)
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
