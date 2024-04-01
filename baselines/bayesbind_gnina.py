
import os
import shutil
import pandas as pd
from tqdm import tqdm, trange
from rdkit import Chem

from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config, get_output_dir

def get_bayesbind_gnina_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "bayesbind_gnina")
    os.makedirs(ret, exist_ok=True)
    return ret

def make_bayesbind_gnina_dir(cfg, split, pocket):

    folder = get_bayesbind_gnina_dir(cfg) + f"/{split}/{pocket}"
    print(f"Making BayesBind GNINA to {folder}")
    os.makedirs(folder, exist_ok=True)

    bayesbind_folder = get_bayesbind_dir(cfg) + f"/{split}/{pocket}"
    baseline_folder = get_baseline_dir(cfg, "gnina", split, pocket)

    shutil.copyfile(bayesbind_folder + "/rec.pdb", folder + "/rec.pdb")
    shutil.copyfile(bayesbind_folder + "/rec_hs.pdb", folder + "/rec_hs.pdb")
    shutil.copyfile(bayesbind_folder + "/pocket.pdb", folder + "/pocket.pdb")
    
    for prefix in ["actives", "random"]:
        shutil.copyfile(bayesbind_folder + f"/{prefix}.smi", folder + f"/{prefix}.smi")

        df = pd.read_csv(bayesbind_folder + f"/{prefix}.csv")[:cfg.baseline_max_ligands]
        score_df = pd.read_csv(baseline_folder + f"/{prefix}_results/results.csv")
        df["gnina_score"] = score_df["score"]

        sdf_indices = []
        cur_sdf_index = 0
        writer = Chem.SDWriter(folder + f"/{prefix}.sdf")
        for i in trange(len(df)):
            out_sdf = baseline_folder + f"/{prefix}_{i}.sdf"
            try:
                lig = Chem.SDMolSupplier(out_sdf, removeHs=False)[0]
            except OSError:
                lig = None
            if lig is None:
                sdf_indices.append(None)
                continue
            sdf_indices.append(cur_sdf_index)
            cur_sdf_index += 1
            writer.write(lig)
        writer.close()

        df["sdf_index"] = sdf_indices
        df.to_csv(folder + f"/{prefix}.csv", index=False)

def make_all_bayesbind_gnina(cfg):

    struct_dfs = {
        "val": pd.read_csv(f"{get_output_dir(cfg)}/structures_val.csv"),
        "test": pd.read_csv(f"{get_output_dir(cfg)}/structures_test.csv"),
    }
    for key in struct_dfs:
        struct_dfs[key] = struct_dfs[key][struct_dfs[key].pchembl_value.notnull()]

    for split, pocket in get_all_bayesbind_splits_and_pockets(cfg):
        make_bayesbind_gnina_dir(cfg, split, pocket)

if __name__ == "__main__":
    cfg = get_config("local")
    make_all_bayesbind_gnina(cfg)