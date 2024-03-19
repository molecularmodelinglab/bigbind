
import sys
import pandas as pd
from tqdm import tqdm
from baselines.banana import make_banana_workflow
from baselines.knn import make_knn_bayesbind_workflow
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from utils.cfg_utils import get_baseline_dir, get_bayesbind_dir, get_config


def collate_baseline_results(cfg, out_file):
    """ Saves all the results from the baselines into a
    single CSV file """

    all_preds = {}
    workflow = make_knn_bayesbind_workflow(cfg)
    all_preds["knn"] = workflow.run()[0]

    workflow = make_banana_workflow(cfg)
    all_preds["banana"] = workflow.run()[0]

    ret = []
    for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
        for set_name in [ "random", "actives" ]:
            df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{set_name}.csv")[:cfg.baseline_max_ligands]
            df["split"] = [split] * len(df)
            df["pocket"] = [pocket] * len(df)
            df["set"] = [set_name] * len(df)

            for baseline in ["knn", "banana", "glide", "gnina", "vina"]:
                if baseline in all_preds:
                    df[baseline] = all_preds[baseline][pocket][set_name][:cfg.baseline_max_ligands]
                else:
                    baseline_df = pd.read_csv(get_baseline_dir(cfg, baseline, split, pocket) + f"/{set_name}_results/results.csv")
                    df[baseline] = baseline_df["score"]

            ret.append(df)

    ret = pd.concat(ret)
    ret.to_csv(out_file, index=False)

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    collate_baseline_results(cfg, "outputs/baseline_results.csv")