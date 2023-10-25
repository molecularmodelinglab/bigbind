import pandas as pd
from tqdm import tqdm
from baselines.eef import get_all_metrics
from baselines.vina_gnina import get_all_bayesbind_splits_and_pockets
from bigbind.bigbind import make_bigbind_workflow
from bigbind.knn import get_knn_preds
from utils.cfg_utils import get_analysis_dir, get_bayesbind_dir, get_config, get_output_dir
from utils.task import task
from utils.workflow import Workflow

@task(force=False)
def get_all_bayesbind_knn_preds(cfg):
    
    train_df = pd.read_csv(get_output_dir(cfg) + "/activities_train.csv")
    
    workflow = make_bigbind_workflow(cfg)
    lig_sim_mat = workflow.run_node_from_name(cfg, "get_tanimoto_matrix")
    lig_smi, _ = workflow.run_node_from_name(cfg, "get_morgan_fps_parallel")
    poc_sim = workflow.run_node_from_name(cfg, "get_pocket_similarity")
    tan_cutoffs, tm_cutoffs, prob_ratios = workflow.run_node_from_name(cfg, "postproc_prob_ratios")

    preds = {}
    for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
        preds[pocket] = {}
        for prefix in [ "actives", "random" ]:
            csv = prefix + ".csv" 
            df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{csv}")
            preds[pocket][prefix] = get_knn_preds(df, train_df, lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, tm_cutoffs, prob_ratios, target_pocket=pocket)

    return preds

@task(force=False)
def get_all_bayesbind_knn_preds_probis(cfg):
    
    train_df = pd.read_csv(get_output_dir(cfg) + "/activities_train.csv")
    
    workflow = make_bigbind_workflow(cfg)
    lig_sim_mat = workflow.run_node_from_name(cfg, "get_tanimoto_matrix")
    lig_smi, _ = workflow.run_node_from_name(cfg, "get_morgan_fps_parallel")
    poc_sim = workflow.run_node_from_name(cfg, "get_pocket_similarity_probis")
    tan_cutoffs, probis_cutoffs, prob_ratios = workflow.run_node_from_name(cfg, "postproc_prob_ratios_probis")

    preds = {}
    for split, pocket in tqdm(get_all_bayesbind_splits_and_pockets(cfg)):
        preds[pocket] = {}
        for prefix in [ "actives", "random" ]:
            csv = prefix + ".csv" 
            df = pd.read_csv(get_bayesbind_dir(cfg) + f"/{split}/{pocket}/{csv}")
            preds[pocket][prefix] = get_knn_preds(df, train_df, lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, probis_cutoffs, prob_ratios, target_pocket=pocket)

    return preds

def make_knn_bayesbind_workflow(cfg):
    preds = get_all_bayesbind_knn_preds()
    preds_probis = get_all_bayesbind_knn_preds_probis()
    return Workflow(cfg, preds, preds_probis)

@task(force=False)
def get_all_bigbind_knn_preds(cfg):
    
    train_df = pd.read_csv(get_output_dir(cfg) + "/activities_train.csv")
    
    workflow = make_bigbind_workflow(cfg)
    lig_sim_mat = workflow.run_node_from_name(cfg, "get_tanimoto_matrix")
    lig_smi, _ = workflow.run_node_from_name(cfg, "get_morgan_fps_parallel")
    poc_sim = workflow.run_node_from_name(cfg, "get_pocket_similarity")
    tan_cutoffs, tm_cutoffs, prob_ratios = workflow.run_node_from_name(cfg, "postproc_prob_ratios")

    preds = {}
    for split in ["val", "test"]:
        df = pd.read_csv(get_output_dir(cfg) + f"/activities_{split}.csv")
        preds[split] = get_knn_preds(df, train_df, lig_smi, lig_sim_mat, poc_sim, tan_cutoffs, tm_cutoffs, prob_ratios)

    return preds

def make_knn_bigbind_workflow(cfg):
    preds = get_all_bigbind_knn_preds()
    return Workflow(cfg, preds)


if __name__ == "__main__":
    cfg = get_config("local")
    
    # print("Calculating KNN preds for BayesBind")
    # workflow = make_knn_bayesbind_workflow(cfg)
    # knn_preds, knn_preds_probis = workflow.run()
    # for name, preds in ("TM", knn_preds), ("ProBis", knn_preds_probis):
    #     print(f"KNN results for {name}")
    #     for key, val in get_all_metrics(cfg, preds).items():
    #         print("  ", key, val)

    print("Calculating KNN preds for BigBind val and test")
    workflow = make_knn_bigbind_workflow(cfg)
    knn_preds = workflow.run()[0]
    for split, preds in knn_preds.items():
        out_file = get_analysis_dir(cfg) + f"/knn_preds_{split}.txt"
        print(f"Writing KNN preds for {split} to {out_file}")
        with open(out_file, "w") as f:
            for p in preds:
                f.write(str(p) + "\n")

