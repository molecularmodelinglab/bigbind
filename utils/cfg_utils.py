import os
from omegaconf import OmegaConf

def get_output_dir(cfg):
    ret = os.path.join(cfg.host.work_dir, cfg.run_name, "global", "output")
    os.makedirs(ret, exist_ok=True)
    return ret

def get_config(host_name):
    cfg = OmegaConf.load("configs/cfg.yaml")
    hosts = OmegaConf.load("configs/hosts.yaml")
    cfg["host"] = hosts[host_name]
    cfg.host.name = host_name
    return cfg