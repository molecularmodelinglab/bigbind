import yaml
import os
import subprocess
import random
import numpy as np
import sys

from tqdm import tqdm
from glob import glob
from traceback import print_exc
from multiprocessing import Pool

from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule
from rdkit import Chem
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox

def prepare_recs(cfg):

    recs = glob(cfg["bigbind_folder"] + "/*/*_rec.pdb")

    for rec in tqdm(recs):
        rec_folder, imm_rec_file = rec.split("/")[-2:]
        out_folder = cfg["docked_folder"] + "/" + rec_folder
        out_file = out_folder + "/" + imm_rec_file + "qt"
        
        os.makedirs(out_folder, exist_ok=True)

        if os.path.exists(out_file): continue
        
        prep_cmd = f"{cfg['adfr_folder']}/bin/prepare_receptor"
        proc = subprocess.run([prep_cmd, "-r", rec, "-o", out_file])
        try:
            proc.check_returncode()
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            
def prepare_ligs(cfg):
    ligs = glob(cfg["bigbind_folder"] + "/*/*_lig.sdf")
    for lig_file in tqdm(ligs):
        lig_folder, imm_lig_file = lig_file.split("/")[-2:]
        out_folder = cfg["docked_folder"] + "/" + lig_folder
        out_file = out_folder + "/" + imm_lig_file.split(".")[0] + ".pdbqt"
        # todo: only for debugging
        if not os.path.exists(out_folder): continue
        lig = Chem.SDMolSupplier(lig_file)[0]

        preparator = MoleculePreparation(hydrate=False) # macrocycles flexible by default since v0.3.0
        preparator.prepare(lig)
        # preparator.show_setup()
        try:
            preparator.write_pdbqt_file(out_file)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()

def get_bounds(cfg, lig_files, padding=4):
    bounds = None
    for lig_file in lig_files:
        try:
            lig = PDBQTMolecule.from_file(lig_file).export_rdkit_mol()
            box = ComputeConfBox(lig.GetConformer(0))
            if bounds is None:
                bounds = box
            else:
                bounds = ComputeUnionBox(box, bounds)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()

    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return center, size

def get_pdb(filename):
    return filename.split("/")[-1].split("_")[0]

def choose_rec_file(lig_file, rec_files, redock):
    """ get a receptor file that cooresponds to a different pdb entry
    than the lig file, so we ensure we are cross docking """

    possible = []
    lig_pdb = get_pdb(lig_file)
    for rec_file in rec_files:
        rec_pdb = get_pdb(rec_file)
        if (rec_pdb == lig_pdb) == redock:
            possible.append(rec_file)
    if len(possible) == 0: return None
    return random.choice(possible)

def dock(lig_file, rec_file, center, size, redock, n_poses=16):

    # os.makedirs("logs", exist_ok=True)
    # sys.stdout = open("logs/" + str(os.getpid()) + ".out", "w")
    # sys.stderr = open("logs/" + str(os.getpid()) + ".err", "w")
    
    folder = "/".join(lig_file.split("/")[:-1])
    lig_name = lig_file.split("/")[-1].split(".")[0]
    rec_name = rec_file.split("/")[-1].split(".")[0]
    if redock:
        out_file = f"{folder}/{lig_name}_{rec_name}_redocked.pdbqt"
    else:
        out_file = f"{folder}/{lig_name}_{rec_name}_docked.pdbqt"

    print(f"Docking {lig_file} with {rec_file} to {out_file}")
    try:
        v = Vina(sf_name='vina', verbosity=1)
 
        v.set_receptor(rec_file)
        v.set_ligand_from_file(lig_file)
        
        v.compute_vina_maps(center=center, box_size=size)
        
        energy_minimized = v.optimize()
        # print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
        # v.write_pose('test_minimized.pdbqt', overwrite=True)

        # Dock the ligand
        # add n_workers?
        v.dock(exhaustiveness=n_poses, n_poses=n_poses, min_rmsd=1)#, max_evals=10)
        v.write_poses(out_file, n_poses=n_poses, energy_range=1e10, overwrite=True)
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
    
    print(f"Finished docking {lig_file} with {rec_file} to {out_file}")
        
def dock_all(cfg, redock, num_procs=1):
    # pool = Pool(num_procs, maxtasksperchild=1)
    results = []
    for folder in glob(cfg["docked_folder"] + "/*"):
        lig_files = glob(folder + "/*_lig.pdbqt")
        rec_files = glob(folder + "/*_rec.pdbqt")
        center, size = get_bounds(cfg, lig_files)
        for lig_file in lig_files:
            rec_file = choose_rec_file(lig_file, rec_files, redock)
            if rec_file is None: continue
            # results.append(pool.apply_async(dock, (lig_file, rec_file, center, size)))
            dock(lig_file, rec_file, center, size, redock)
            # todo: only for debugging
            break
    # for res in results:
    #    res.get()
        
    # pool.close()
    # pool.join()
    # pool.terminate()

def clean_docked(cfg, redock):
    if redock:
        files = glob(cfg["docked_folder"] + "/*/*_redocked.pdbqt")
    else:
        files = glob(cfg["docked_folder"] + "/*/*_docked.pdbqt")
    for docked in files:
        os.remove(docked)
        
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    # prepare_recs(cfg)
    # prepare_ligs(cfg)
    redock = True
    clean_docked(cfg, redock)
    dock_all(cfg, redock)
