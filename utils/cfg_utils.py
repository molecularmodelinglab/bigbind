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

def get_figure_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "figures")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_parent_baseline_dir(cfg):
    return os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines")

def get_baseline_dir(cfg, program_name, split, pocket):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "baselines", program_name, split, pocket)
    os.makedirs(ret, exist_ok=True)
    return ret

def get_docked_dir(cfg, program_name, split):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "docked", program_name, split)
    os.makedirs(ret, exist_ok=True)
    return ret

def get_config(host_name):
    cfg = OmegaConf.load("configs/cfg.yaml")
    hosts = OmegaConf.load("configs/hosts.yaml")
    cfg["host"] = hosts[host_name]
    cfg.host.name = host_name
    return cfg