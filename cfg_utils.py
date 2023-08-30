from omegaconf import OmegaConf

def get_config(host_name):
    cfg = OmegaConf.load("configs/cfg.yaml")
    hosts = OmegaConf.load("configs/hosts.yaml")
    cfg["host"] = hosts[host_name]
    cfg.host.name = host_name
    return cfg