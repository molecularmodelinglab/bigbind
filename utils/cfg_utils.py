import os
from omegaconf import OmegaConf

def get_output_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "output")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_bayesbind_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "bayesbind")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_bayesbind_ml_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "bayesbind_ml")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_bayesbind_struct_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "bayesbind_struct")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_bayesbind_small_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "bayesbind_small")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_final_bayesbind_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "BayesBindV1")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_figure_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "figures")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_parent_baseline_dir(cfg):
    return os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines")

def get_parent_baseline_struct_dir(cfg):
    return os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines_struct")


def get_baseline_dir(cfg, program_name, split, pocket):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines", program_name, split, pocket)
    os.makedirs(ret, exist_ok=True)
    return ret

def get_baseline_struct_dir(cfg, program_name, split, pocket):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines_struct", program_name, split, pocket)
    os.makedirs(ret, exist_ok=True)
    return ret

def get_docked_dir(cfg, program_name, split):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "docked", program_name, split)
    os.makedirs(ret, exist_ok=True)
    return ret

def get_analysis_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "analysis")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_config(host_name):
    cfg = OmegaConf.load("configs/cfg.yaml")
    hosts = OmegaConf.load("configs/hosts.yaml")
    cfg["host"] = hosts[host_name]
    cfg.host.name = host_name
    return cfg