import sys
from omegaconf import OmegaConf
from glob import glob
import pandas as pd
import os
import subprocess

from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config, get_parent_baseline_dir

HOST = "\"general:6\""

def prep_ligs(cfg, out_folder):
    """ Run ligprep on all the actives and random smi files"""
    for folder in glob(get_bayesbind_dir(cfg) + "/*/*"):
        for smi_file in [ folder + "/actives.smi", folder + "/random.smi" ]:
            out_file = out_folder + "/" + "/".join(smi_file.split("/")[-3:]).split(".")[0] + ".sdf"
            if os.path.exists(out_file): continue
            os.makedirs("/".join(out_file.split("/")[:-1]), exist_ok=True)
            cmd = f"ligprep -ismi {smi_file} -osd {out_file}"
            print("Running " + cmd)
            subprocess.run(cmd, shell=True)

def prep_recs(cfg, out_folder):
    """ Run prepwizard on all the rec files """
    for i, folder in enumerate(glob(get_bayesbind_dir(cfg) + f"/*/*")):
        rec_file = folder + "/rec.pdb"
        # out_file = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        # for some godforsaken reason there's a memory error if I try to 
        # output the final file directly. But a tmp file and copying works
        out_file = f"rec_{i}.mae"
        if os.path.exists(out_file): continue
        # os.makedirs("/".join(out_file.split("/")[:-1]), exist_ok=True)
        cmd = f"$SCHRODINGER/utilities/prepwizard -fix {rec_file} {out_file}"
        print("Running " + cmd)
        subprocess.run(cmd, shell=True)

def finalize_rec_prep(cfg, out_folder):
    for i, folder in enumerate(glob(get_bayesbind_dir(cfg) + f"/*/*")):
        rec_file = folder + "/rec.pdb"
        old_file = f"rec_{i}.mae"
        new_file = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        os.rename(old_file, new_file)

def make_grids(cfg, out_folder):
    for i, folder in enumerate(glob(get_bayesbind_dir(cfg) + f"/*/*")):
        rec_file = folder + "/rec.pdb"
        rec_mae = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        cur_folder = "/".join(rec_mae.split("/")[:-1])
        os.makedirs(cur_folder, exist_ok=True)
        
        gridfile = cur_folder + "/grid.zip"
        in_file = cur_folder + "/grid.in"
        df = pd.read_csv(folder + "/actives.csv")
        row = df.iloc[0]

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
        
        cmd = f"glide {in_file}"
        print(f"Running {cmd}")
        subprocess.run(cmd, shell=True)

def dock_all(cfg, out_folder):

    abs_path = os.path.abspath(".")
    
    for i, folder in enumerate(glob(get_bayesbind_dir(cfg) + f"/*/*")):

        os.chdir(abs_path)

        rec_file = folder + "/rec.pdb"
        rec_mae = out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"

        cur_folder = "/".join(rec_mae.split("/")[:-1])
        os.makedirs(cur_folder, exist_ok=True)

        gridfile = cur_folder + "/grid.zip"


        for prefix in ["actives", "random"]:
            lig_file = cur_folder + "/" + prefix + ".sdf"
            in_file = cur_folder + "/dock_" + prefix + ".in"
            output_folder = cur_folder + "/" + prefix + "_results"
            os.makedirs(output_folder, exist_ok=True)

            out_file = output_folder + f"/dock_{prefix}.csv"
            if os.path.exists(out_file):
                # print("Already ran glide for " + out_file)
                continue

            print(f"Writing docking params to {in_file}")
            with open(in_file, "w") as f:
                f.write(
f"""GRIDFILE {gridfile}
LIGANDFILE {lig_file}
"""
                )

            os.chdir(output_folder)
            cmd = f"glide {in_file} -HOST {HOST} "
            print(f"Running {cmd} from {os.path.abspath('.')}")
            subprocess.run(cmd, shell=True)

def glide_to_sdf(cfg, out_folder):

    abs_path = os.path.abspath(".")
    
    for i, folder in enumerate(glob(get_bayesbind_dir(cfg) + "/*/*")):

        os.chdir(abs_path)

        rec_file = abs_path + "/" + folder + "/rec.pdb"
        rec_mae = abs_path + "/" + out_folder + "/" + "/".join(rec_file.split("/")[-3:]).split(".")[0] + ".mae"
        cur_folder = "/".join(rec_mae.split("/")[:-1])
        os.makedirs(cur_folder, exist_ok=True)

        for prefix in ["actives", "random"]:
            lig_file = cur_folder + "/" + prefix + ".sdf"
            in_file = cur_folder + "/dock_" + prefix + ".in"
            output_folder = cur_folder + "/" + prefix + "_results"

            mae_file = output_folder + f"/{prefix}_pv.maegz"
            if not os.path.exists(mae_file): continue
            out_file = output_folder + f"/{prefix}_pv.sdf"
            cmd = f"$SCHRODINGER/utilities/sdconvert -imae {mae_file} -osf {out_file}"
            print(f"Running {cmd}")
            subprocess.run(cmd, shell=True, check=True)



if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    
    out_folder = get_parent_baseline_dir(cfg) + "/glide/"
    os.makedirs(out_folder, exist_ok=True)
    os.chdir(out_folder)
    
    # out_folder = "baseline_data/glide"
    # if not os.path.exists("baseline_data"):
    #     subprocess.run(f"ln -s {get_parent_baseline_dir(cfg)} baseline_data", shell=True, check=True)

    # prep_ligs(cfg, out_folder)
    # prep_recs(cfg, out_folder)
    # finalize_rec_prep(cfg, out_folder)
    # make_grids(cfg, out_folder)
    dock_all(cfg, out_folder)
    # glide_to_sdf(cfg, out_folder)
            
