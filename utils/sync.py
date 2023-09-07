import sys
import os
import subprocess
from utils.cfg_utils import get_config

def get_rsync_dir(cfg):
    dir = os.path.join(cfg.host.work_dir, cfg.run_name, "global")
    hostname = None
    for key in ["network_hostname", "hostname"]:
        if key in cfg.host:
            hostname = cfg.host[key]
    if hostname is None:
        return dir
    return f"{cfg.host.user}@{hostname}:{dir}"

if __name__ == "__main__":
    host1 = sys.argv[1]
    host2 = sys.argv[2]
    cfg1 = get_config(host1)
    cfg2 = get_config(host2)

    dir1 = get_rsync_dir(cfg1)
    dir2 = get_rsync_dir(cfg2)
    dir2 = os.path.dirname(dir2)

    cmd = f"rsync -r {dir1} {dir2}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)
