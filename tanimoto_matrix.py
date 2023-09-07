from functools import partial
from multiprocessing import Pool, shared_memory
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import yaml
from task import task

def get_morgan_fps(cfg, df, radius=3, bits=2048):
    fps = []
    for smi in tqdm(df.lig_smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=bits)
        fps.append(np.array(fp, dtype=bool))
    fps = np.asarray(fps)


def get_fp(smi, radius=4, bits=2048):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=bits)
    return np.array(fp, dtype=bool)

# like get_morgan_fps, but parallelized
MORGAN_FP_CPUS = 32
@task(max_runtime=1, n_cpu=MORGAN_FP_CPUS, num_outputs=2, mem=32)
def get_morgan_fps_parallel(cfg, df):
    smi_list = list(df.canonical_smiles.unique())
    fps = []
    with Pool(MORGAN_FP_CPUS) as p:
        fps = list(tqdm(p.imap(get_fp, smi_list), total=len(smi_list)))
    fps = np.asarray(fps)
    return smi_list, fps

def batch_tanimoto(fp, fps):
    inter = np.logical_and(fp, fps)
    union = np.logical_or(fp, fps)
    sims = inter.sum(-1)/union.sum(-1)
    return sims

def get_bytes(a):
    return a.data.nbytes + a.row.nbytes + a.col.nbytes

def batch_tanimoto_faster(fp_shape, fp_shm_name, fp_sum_shape, fp_sum_shm_name, idx):
    fp_shm = shared_memory.SharedMemory(name=fp_shm_name)
    fps = np.ndarray(fp_shape, dtype=bool, buffer=fp_shm.buf)
    fp_sum_shm = shared_memory.SharedMemory(name=fp_sum_shm_name)
    fp_sum = np.ndarray(fp_sum_shape, dtype=int, buffer=fp_sum_shm.buf)

    fp = fps[idx]
    inter = np.logical_and(fp, fps)
    inter_sum = inter.sum(-1)
    sims = inter_sum/(fp_sum + fp.sum() - inter_sum)

    # print(idx, (sims < 0.2).sum()/len(sims))
    
    sims[sims < 0.4] = 0.0
    ssim = sparse.coo_matrix(sims)

    # print(ssim.data.shape, get_bytes(ssim))

    return ssim

TANIMOTO_CPUS = 16
@task(max_runtime=48, n_cpu=TANIMOTO_CPUS, mem=32)
def get_tanimoto_matrix(cfg, fps):
    try:
        fp_sum = fps.sum(-1)

        fp_shm = shared_memory.SharedMemory(create=True, size=fps.nbytes)
        fps_shared = np.ndarray(fps.shape, dtype=bool, buffer=fp_shm.buf)
        fps_shared[:] = fps[:]
        
        fp_sum_shm = shared_memory.SharedMemory(create=True, size=fp_sum.nbytes)
        fp_sum_shared = np.ndarray(fp_sum.shape, dtype=int, buffer=fp_sum_shm.buf)
        fp_sum_shared[:] = fp_sum[:]

        sim_func = partial(batch_tanimoto_faster, fps.shape, fp_shm.name, fp_sum.shape, fp_sum_shm.name)
        with Pool(TANIMOTO_CPUS) as p:
            cols = list(tqdm(p.imap(sim_func, range(len(fps))), total=len(fps)))
        return sparse.vstack(cols)
    finally:
        fp_shm.close()
        fp_shm.unlink()
        fp_sum_shm.close()
        fp_sum_shm.unlink()

def main(cfg):
    df = pd.read_csv(f"{cfg['bigbind_folder']}/activities_all.csv")
    fps = get_morgan_fps_parallel(cfg, df)
    tm = get_tanimoto_matrix(cfg, fps)
    print(tm.shape)

SEED = 49
random.seed(SEED)
np.random.seed(SEED)
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
