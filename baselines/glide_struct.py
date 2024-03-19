from multiprocessing import Pool
import sys
from functools import partial
from traceback import print_exc
from omegaconf import OmegaConf
from glob import glob
import pandas as pd
import os
import subprocess
from tqdm import tqdm
from baselines.vina_gnina import get_all_bayesbind_struct_splits_and_pockets

from utils.cfg_utils import get_baseline_dir, get_baseline_struct_dir, get_bayesbind_struct_dir, get_config, get_parent_baseline_dir, get_parent_baseline_struct_dir

def prep_ligs_no_struct(cfg, out_folder):
    """ Run ligprep on all the actives"""
    for folder in glob(get_bayesbind_struct_dir(cfg) + "/*/*"):
        smi_file = folder + "/actives.smi"
        out_file = out_folder + "/" + "/".join(smi_file.split("/")[-3:]).split(".")[0] + ".mae"

        if os.path.exists(out_file): continue        
        os.makedirs("/".join(out_file.split("/")[:-1]), exist_ok=True)

        cmd = f"ligprep -ismi {smi_file} -omae {out_file}"
        print("Running " + cmd)
        subprocess.run(cmd, shell=True, check=True)

def prep_single_lig(out_folder, folder):
    """ Use individual components of ligprep to protonate the ligands"""
    for lig_file in tqdm(glob(folder + "/*_lig.sdf")):
        out_file = out_folder + "/" + "/".join(lig_file.split("/")[-3:]).split(".")[0] + ".mae"
        
        # temp files for hydrogens and neutralization
        out_file_h = out_file.replace(".mae", "_h.mae")
        out_file_n = out_file.replace(".mae", "_n.mae")
        bad_file = out_file.replace(".mae", "_bad.mae")
        
        if os.path.exists(out_file): continue
        os.makedirs("/".join(out_file.split("/")[:-1]), exist_ok=True)

        # use obabel to add hydrogens -- nothing else needed
        cmd = f"$SCHRODINGER/utilities/applyhtreat {lig_file} {out_file_h} && $SCHRODINGER/utilities/neutralizer {out_file_h} {out_file_n} && $SCHRODINGER/utilities/ionizer -i {out_file_n} -o {out_file} -b {bad_file} && rm -f {out_file_h} {out_file_n} {bad_file}"
        print("Running " + cmd)
        subprocess.run(cmd, shell=True, check=True)

def prep_ligs(cfg, out_folder):
    folders = glob(get_bayesbind_struct_dir(cfg) + "/*/*")
    with Pool(8) as p:
        p.map(partial(prep_single_lig, out_folder), folders)


def prep_single_rec(out_folder, folder):
    for rec_file in tqdm(glob(folder + "/*_rec.pdb")):
        out_file = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        
        if os.path.exists(out_file): continue
        cmd = f"$SCHRODINGER/utilities/prepwizard -fix {rec_file} {out_file} -NOJOBID"
        print("Running " + cmd)
        subprocess.run(cmd, shell=True, check=True)

def prep_recs(cfg, out_folder):
    """ Run prepwizard on all the rec files """
    folders = glob(get_bayesbind_struct_dir(cfg) + "/*/*")
    f = partial(prep_single_rec, out_folder)
    with Pool(8) as p:
        p.map(partial(prep_single_rec, out_folder), folders)

def single_grid_and_dock(out_folder, folder, force=True):
    df = pd.read_csv(folder + "/actives.csv")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        rec_file = folder + "/" + row.redock_rec_file
        rec_mae = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        cur_folder = "/".join(rec_mae.split("/")[:-1])
        os.makedirs(cur_folder, exist_ok=True)
        
        gridfile = rec_mae.replace(".mae", "_grid.zip")
        in_file = rec_mae.replace(".mae", "_grid.in")


        if not os.path.exists(gridfile) or not force:

            # inner box is X times the size of outer box
            inner_scale = 0.5

            print(f"Writing grid params to {in_file}")
            with open(in_file, "w") as f:
                f.write(
f"""INNERBOX {int(row.pocket_size_x*inner_scale)}, {int(row.pocket_size_y*inner_scale)},{int(row.pocket_size_z*inner_scale)}
ACTXRANGE {row.pocket_size_x}
ACTYRANGE {row.pocket_size_y}
ACTZRANGE {row.pocket_size_z}
OUTERBOX {row.pocket_size_x}, {row.pocket_size_y}, {row.pocket_size_z}
GRID_CENTER {row.pocket_center_x}, {row.pocket_center_y}, {row.pocket_center_z}
GRIDFILE {gridfile}
RECEP_FILE {rec_mae}
"""
                )
            
            cmd = f"glide {in_file}  -NOJOBID"
            print(f"Running {cmd}")
            subprocess.run(cmd, shell=True, check=True)


        dock_in_file = cur_folder + "/dock_" + row.pdb + ".in"
        output_folder = cur_folder + "/" + row.pdb
        os.makedirs(output_folder, exist_ok=True)


        lig_file = folder + "/" + row.lig_crystal_file
        lig_mae = out_folder + "/" + "/".join(lig_file.split("/")[-3:]).split(".")[0] + ".mae"


        out_file = output_folder + f"/dock_{row.pdb}.csv"
        if os.path.exists(out_file) and not force:
            # print("Already ran glide for " + out_file)
            continue

        # print(f"Writing docking params to {in_file}")
        with open(dock_in_file, "w") as f:
            f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_mae}
DOCKING_METHOD mininplace
"""
            )

        os.chdir(output_folder)
        cmd = f"glide {dock_in_file} -NOJOBID -OVERWRITE"
        print(f"Running {cmd} from {os.path.abspath('.')}")
        subprocess.run(cmd, shell=True, check=True)


def grid_and_dock_all(cfg, out_folder):
    folders = glob(get_bayesbind_struct_dir(cfg) + "/*/*")
    with Pool(8) as p:
        p.map(partial(single_grid_and_dock, out_folder), folders)

def single_grid_and_dock_crossdock(cfg, out_folder, folder):
    """ Crossdocks to the same rec mae file used by non-struct
        BayesBind. A more accurate comparison """
    split, pocket = folder.split("/")[-2:]
    df = pd.read_csv(folder + "/actives.csv")
    for i, row in tqdm(df.iterrows(), total=len(df)):

        gridfile = get_baseline_dir(cfg, "glide", split, pocket) + "/grid.zip"

        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)
        dock_in_file = cur_folder + "/dock_" + row.pdb + "_crossdock.in"
        output_folder = cur_folder + "/" + row.pdb + "_crossdock"
        os.makedirs(output_folder, exist_ok=True)


        lig_file = folder + "/" + row.lig_crystal_file
        lig_mae = out_folder + "/" + "/".join(lig_file.split("/")[-3:]).split(".")[0] + ".mae"


        out_file = output_folder + f"/dock_{row.pdb}_crossdock_pv.maegz"
        if os.path.exists(out_file):
            # print("Already ran glide for " + out_file)
            continue

        # print(f"Writing docking params to {in_file}")
        with open(dock_in_file, "w") as f:
            f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_mae}
DOCKING_METHOD mininplace
"""
            )

        os.chdir(output_folder)
        cmd = f"glide {dock_in_file} -NOJOBID "
        print(f"Running {cmd} from {os.path.abspath('.')}")
        subprocess.run(cmd, shell=True, check=True)

def grid_and_dock_all_crossdock(cfg, out_folder):
    folders = glob(get_bayesbind_struct_dir(cfg) + "/*/*")
    with Pool(8) as p:
        p.map(partial(single_grid_and_dock_crossdock, cfg, out_folder), folders)

def dock_all_no_struct(cfg, out_folder):

    abs_path = os.path.abspath(".")
    
    for i, folder in enumerate(glob(get_bayesbind_struct_dir(cfg) + f"/*/*")):

        split, pocket = folder.split("/")[-2:]
        gridfile = get_baseline_dir(cfg, "glide", split, pocket) + "/grid.zip"

        os.chdir(abs_path)

        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)
        dock_in_file = cur_folder + "/dock_no_struct.in"

        lig_file = cur_folder + "/actives.mae"
        output_folder = cur_folder + "/no_struct_results"
        os.makedirs(output_folder, exist_ok=True)

        out_file = output_folder + f"/dock_no_struct.csv"
        if os.path.exists(out_file):
            # print("Already ran glide for " + out_file)
            continue

        # print(f"Writing docking params to {in_file}")
        with open(dock_in_file, "w") as f:
            f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_file}
"""
            )

        os.chdir(output_folder)
        cmd = f"glide {dock_in_file}"
        print(f"Running {cmd} from {os.path.abspath('.')}")
            
        subprocess.run(cmd, shell=True, check=True)

def convert_crossdocked(cfg):
    # pdb
    for split, pocket in tqdm(get_all_bayesbind_struct_splits_and_pockets(cfg)):
        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)
        for cd_mae in glob(cur_folder + "/*_crossdock/dock_*_pv.maegz"):
            out_file = cd_mae.replace("maegz", "pdb")
            out_folder = os.path.dirname(out_file)
            if len(glob(out_folder + "/*.pdb")) >= 2:
                continue
            cmd = f"cd {out_folder} && {os.environ['SCHRODINGER']}/utilities/structconvert {cd_mae} {out_file}"
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)

    # sdf
    for split, pocket in tqdm(get_all_bayesbind_struct_splits_and_pockets(cfg)):
        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)
        for cd_mae in glob(cur_folder + "/*_crossdock/dock_*_pv.maegz"):
            # os.remove(out_file)
            out_folder = os.path.dirname(cd_mae)

            # first split the mae
            cmd = f"cd {out_folder} && {os.environ['SCHRODINGER']}/utilities/structconvert {cd_mae} {cd_mae.replace('.maegz', '_split.mae')} -split-nstructures 1"
            # print(cmd)
            if len(glob(out_folder + "/*split-*.mae")) > 0:
                subprocess.run(cmd, shell=True, check=True)
            # now convert to sdf
            for mae in glob(out_folder + "/*split-*.mae"):
                out_file = mae.replace("mae", "sdf")
                if os.path.exists(out_file): continue
                cmd = f"cd {out_folder} && {os.environ['SCHRODINGER']}/utilities/structconvert {mae} {out_file}"
                # print(cmd)
                subprocess.run(cmd, shell=True, check=True)
                os.remove(mae)

def single_grid_and_dock_full_min(out_folder, folder):
    df = pd.read_csv(folder + "/actives.csv")
    *_, split, pocket = folder.split("/")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        pdb = row.pdb
        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)

        recfile = cur_folder + f"/{pdb}_crossdock/rec_0_full_min.pdb"
        if not os.path.exists(recfile):
            continue

        rec_mae = recfile.replace(".pdb", ".mae")
        if not os.path.exists(rec_mae):
            cmd = f"$SCHRODINGER/utilities/structconvert {recfile} {rec_mae}"
            print("Running " + cmd)
            subprocess.run(cmd, shell=True, check=True)

        gridfile = recfile.replace(".pdb", "_grid.zip")

        if not os.path.exists(gridfile):

            grid_in_file = gridfile.replace(".zip", ".in")
            # inner box is X times the size of outer box
            inner_scale = 0.5

            print(f"Writing grid params to {grid_in_file}")
            with open(grid_in_file, "w") as f:
                f.write(
f"""INNERBOX {int(row.pocket_size_x*inner_scale)}, {int(row.pocket_size_y*inner_scale)},{int(row.pocket_size_z*inner_scale)}
ACTXRANGE {row.pocket_size_x}
ACTYRANGE {row.pocket_size_y}
ACTZRANGE {row.pocket_size_z}
OUTERBOX {row.pocket_size_x}, {row.pocket_size_y}, {row.pocket_size_z}
GRID_CENTER {row.pocket_center_x}, {row.pocket_center_y}, {row.pocket_center_z}
GRIDFILE {gridfile}
RECEP_FILE {rec_mae}
"""
                )
            
            cmd = f"glide {grid_in_file}  -NOJOBID"
            print(f"Running {cmd}")
            subprocess.run(cmd, shell=True, check=True)


        dock_in_file = cur_folder + "/dock_" + row.pdb + "_full_min.in"
        output_folder = cur_folder + "/" + row.pdb + "_full_min"
        os.makedirs(output_folder, exist_ok=True)

        lig_file = folder + "/" + row.lig_crystal_file
        lig_mae = out_folder + "/" + "/".join(lig_file.split("/")[-3:]).split(".")[0] + ".mae"

        out_file = output_folder + f"/dock_{row.pdb}_full_min.csv"
        if os.path.exists(out_file):
            # print("Already ran glide for " + out_file)
            continue

        # print(f"Writing docking params to {in_file}")
        with open(dock_in_file, "w") as f:
            f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_mae}
DOCKING_METHOD mininplace
"""
            )

        os.chdir(output_folder)
        cmd = f"glide {dock_in_file} -NOJOBID -OVERWRITE"
        print(f"Running {cmd} from {os.path.abspath('.')}")
        subprocess.run(cmd, shell=True, check=True)


def grid_and_dock_all_full_min(cfg, out_folder):
    folders = glob(get_bayesbind_struct_dir(cfg) + "/val/*")
    with Pool(8) as p:
        p.map(partial(single_grid_and_dock_full_min, out_folder), folders)

def single_grid_and_dock_min(out_folder, folder):
    try:
        df = pd.read_csv(folder + "/actives.csv")
        *_, split, pocket = folder.split("/")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pdb = row.pdb
            cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)

            recfile = cur_folder + f"/{pdb}_crossdock/rec_0_esp.pdb"
            if not os.path.exists(recfile):
                continue

            rec_mae = recfile.replace(".pdb", ".mae")
            if not os.path.exists(rec_mae):
                # cmd = f"$SCHRODINGER/utilities/prepwizard -fix {recfile} {rec_mae} -NOJOBID"
                cmd = f"$SCHRODINGER/utilities/structconvert {recfile} {rec_mae}"
                print("Running " + cmd)
                subprocess.run(cmd, shell=True, check=True)

            gridfile = recfile.replace(".pdb", "_grid.zip")

            if not os.path.exists(gridfile):

                grid_in_file = gridfile.replace(".zip", ".in")
                # inner box is X times the size of outer box
                inner_scale = 0.5

                print(f"Writing grid params to {grid_in_file}")
                with open(grid_in_file, "w") as f:
                    f.write(
f"""INNERBOX {int(row.pocket_size_x*inner_scale)}, {int(row.pocket_size_y*inner_scale)},{int(row.pocket_size_z*inner_scale)}
ACTXRANGE {row.pocket_size_x}
ACTYRANGE {row.pocket_size_y}
ACTZRANGE {row.pocket_size_z}
OUTERBOX {row.pocket_size_x}, {row.pocket_size_y}, {row.pocket_size_z}
GRID_CENTER {row.pocket_center_x}, {row.pocket_center_y}, {row.pocket_center_z}
GRIDFILE {gridfile}
RECEP_FILE {rec_mae}
"""
                    )
                
                cmd = f"glide {grid_in_file}  -NOJOBID"
                print(f"Running {cmd}")
                subprocess.run(cmd, shell=True, check=True)


            dock_in_file = cur_folder + "/dock_" + row.pdb + "_min.in"
            output_folder = cur_folder + "/" + row.pdb + "_min"
            os.makedirs(output_folder, exist_ok=True)

            lig_file = folder + "/" + row.lig_crystal_file
            lig_mae = out_folder + "/" + "/".join(lig_file.split("/")[-3:]).split(".")[0] + ".mae"

            out_file = output_folder + f"/dock_{row.pdb}_min.csv"
            if os.path.exists(out_file):
                # print("Already ran glide for " + out_file)
                continue

            # print(f"Writing docking params to {in_file}")
            with open(dock_in_file, "w") as f:
                f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_mae}
DOCKING_METHOD mininplace
"""
                )

            os.chdir(output_folder)
            cmd = f"glide {dock_in_file} -NOJOBID -OVERWRITE"
            print(f"Running {cmd} from {os.path.abspath('.')}")
            subprocess.run(cmd, shell=True, check=True)
    except:
        print_exc()

def grid_and_dock_all_min(cfg, out_folder):
    folders = glob(get_bayesbind_struct_dir(cfg) + "/*/*")
    with Pool(8) as p:
        p.map(partial(single_grid_and_dock_min, out_folder), folders)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    
    out_folder = get_parent_baseline_struct_dir(cfg) + "/glide/"
    os.makedirs(out_folder, exist_ok=True)
    os.chdir(out_folder)
    
    # out_folder = "baseline_data/glide"
    # if not os.path.exists("baseline_data"):
    #     subprocess.run(f"ln -s {get_parent_baseline_dir(cfg)} baseline_data", shell=True, check=True)

    grid_and_dock_all_min(cfg, out_folder)
    # prep_recs(cfg, out_folder)
    # make_grids(cfg, out_folder)
    # grid_and_dock_all(cfg, out_folder)

            
    # for folder in glob(get_parent_baseline_dir(cfg) + "/glide/*/*"):
    #     cmd = f"ls {folder}/*_crossdock"
    #     print(cmd)
    #     subprocess.run(cmd, shell=True, check=True)
