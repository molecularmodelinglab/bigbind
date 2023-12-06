
import sys
import pandas as pd
from tqdm import tqdm

from utils.cfg_utils import get_config, get_output_dir


def generate_crossdocked_affinities(cfg, crossdocked_folder):
    """ Returns a dataframe in the bigbind format with the
        pK values from the crossdocked dataset """
    
    pks = {}

    types_files = [ 
        f"{crossdocked_folder}/types/it2_redocked_tt_v1.1_completeset_train0.types",
        f"{crossdocked_folder}/types/it2_redocked_tt_v1.1_completeset_test0.types",
    ]
    for fname in types_files:
        with open(fname, "r") as f:
            for line in f:
                lab, pk, rmsd, rec, lig, vina = line.split(" ")
                if "lig_tt_docked" not in lig: continue
                pk = abs(float(pk))
                if pk == 0: continue
                pocket = rec.split("/")[0]
                rf = "_".join(rec.split("_")[:-1]) + ".pdb"
                lf = pocket + "/" + ("_".join(lig.split("_")[-6:-3])) + ".sdf"
                key = (pocket, rf, lf)
                if key in pks:
                    assert abs(pks[key] - pk) < 0.1
                else:
                    pks[key] = pk
    
    df = []
    for (pocket, rf, lf), pk in pks.items():
        df.append({
            "pocket": pocket,
            "rec_file": rf,
            "lig_file": lf,
            "pchembl_value": pk,
        })
    df = pd.DataFrame(df)

    return df

def save_crossdocked_affinities(cfg, crossdocked_folder):
    df = generate_crossdocked_affinities(cfg, crossdocked_folder)
    df.to_csv(f"{get_output_dir(cfg)}/structures_pK_all.csv", index=False)
    for split in tqdm(["train", "val", "test"]):
        act_df = pd.read_csv(f"{get_output_dir(cfg)}/activities_{split}.csv")
        splits_pocs = act_df.pocket.unique()
        split_df = df.query("pocket in @splits_pocs").reset_index(drop=True)
        out_fname = f"{get_output_dir(cfg)}/structures_pK_{split}.csv"
        split_df.to_csv(out_fname, index=False)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    cd_folder = sys.argv[2]
    save_crossdocked_affinities(cfg, cd_folder)
        