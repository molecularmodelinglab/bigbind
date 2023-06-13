import os
from run import *

@cache
def get_crossdocked_test_clusters(cfg, clusters):
    types_file = cfg["crossdocked_folder"] + "/types/it2_tt_v1.3_0_test0.types"

    test_pockets = set()
    with open(types_file, "r") as f:
        for line in f.readlines():
            pocket = line.split(" ")[3].split("/")[0]
            test_pockets.add(pocket)

    test_clusters = set()
    for pocket in test_pockets:
        for cluster in clusters:
            if pocket in cluster:
                test_clusters.add(cluster)
                break

    return test_clusters

@cache
def get_crossdocked_splits(cfg, clusters, cd_test_clusters):
    """ We take all the clusters from the crossdocked train set 
    (that is, all clusters not in cd_test_clusters) and put them
    in the train split. We then evenly devide the cd_test_clusters
    into train, val and test """
    test_split_fracs = {
        "train": 1/3,
        "val": 1/3,
        "test": 1/3,
    }

    cd_test_clusters = list(cd_test_clusters)
    splits = {}
    tot_clusters = 0
    for split, frac in test_split_fracs.items():
        cur_pockets = set()
        num_clusters = round(len(cd_test_clusters)*frac)
        for cluster in cd_test_clusters[tot_clusters:tot_clusters+num_clusters]:
            for pocket in cluster:
                cur_pockets.add(pocket)
        splits[split] = cur_pockets
        tot_clusters += num_clusters

    for cluster in clusters:
        if cluster not in cd_test_clusters:
            for pocket in cluster:
                splits["train"].add(pocket)

        # make sure we haven't forgotten any
        for pocket in cluster:
            for split in splits.values():
                if pocket in split:
                    break
            else:
                raise AssertionError

    # assert splits as disjoint
    for s1, p1 in splits.items():
        for s2, p2 in splits.items():
            if s1 == s2: continue
            assert len(p1.intersection(p2)) == 0
    return splits


def run_plantain_crossdocked(cfg):
    os.makedirs(cfg["crossdocked_plantain_folder"], exist_ok=True)
    if cfg["cache"]:
        os.makedirs(cfg["cache_folder"], exist_ok=True)
        
    con = get_chembl_con(cfg)
    load_sifts_into_chembl(cfg, con)
    cd_files = load_crossdocked_files(cfg)
    cd_uniprots = get_crossdocked_uniprots(cfg, con, cd_files)
    
    chain2uniprot = get_chain_to_uniprot(cfg, con)
    
    uniprot2recs,\
    uniprot2ligs,\
    uniprot2pockets,\
    pocket2uniprots,\
    pocket2recs,\
    pocket2ligs = get_uniprot_dicts(cfg, cd_files, chain2uniprot)
    final_uniprots, final_pockets = filter_uniprots(cfg,
                                                    uniprot2pockets,
                                                    pocket2uniprots)
    

    comp_dict = load_components_dict(cfg)
    
    lig_sdfs = download_all_lig_sdfs(cfg, uniprot2ligs)
    ligfile2lig = get_all_ligs(cfg, comp_dict, lig_sdfs)
    pocket2recs,\
    pocket2ligs,\
    ligfile2lig,\
    ligfile2uff = save_all_structures(cfg,
                                      final_pockets,
                                      pocket2uniprots,
                                      pocket2recs,
                                      pocket2ligs,
                                      ligfile2lig,
                                      cfg["crossdocked_plantain_folder"])


    rec2pocketfile, rec2res_num = save_all_pockets(cfg, 
                                                   pocket2recs, 
                                                   pocket2ligs, 
                                                   ligfile2lig)
    pocket_centers, pocket_sizes = get_all_pocket_bounds(cfg, pocket2ligs, ligfile2lig)

    # probis stuff
    
    rec2srf = create_all_probis_srfs(cfg, rec2pocketfile)
    rep_srf2nosql = find_representative_rec(cfg, pocket2recs, rec2srf)
    rep_scores = convert_intra_results_to_json(cfg, rec2srf, rep_srf2nosql)
    pocket2rep_rec = get_rep_recs(cfg, pocket2recs, rep_scores)
    srf2nosql = find_all_probis_distances(cfg, pocket2rep_rec, rec2srf)
    full_scores = convert_inter_results_to_json(cfg, rec2srf, srf2nosql)

    struct_df = create_struct_df(cfg,
                                 pocket2recs,
                                 pocket2ligs,
                                 ligfile2lig,
                                 ligfile2uff,
                                 rec2pocketfile,
                                 rec2res_num,
                                 pocket_centers,
                                 pocket_sizes)
    struct_df = filter_struct_df(cfg, struct_df, cfg["crossdocked_plantain_folder"])

    # cluster and save everything
    clusters = get_clusters(cfg, pocket2rep_rec, full_scores)
    cd_test_clusters = get_crossdocked_test_clusters(cfg, clusters)
    splits = get_crossdocked_splits(cfg, clusters, cd_test_clusters)

    save_clustered_structs(cfg, struct_df, splits, cfg["crossdocked_plantain_folder"])
    # SNA!

SEED = 49
random.seed(SEED)
np.random.seed(SEED)
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_plantain_crossdocked(cfg)
