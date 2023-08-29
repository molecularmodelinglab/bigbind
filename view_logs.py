import sys
import os
from termcolor import colored
from fabric.connection import Connection
from cfg_utils import get_config

def print_log(c, sftp, filename, color):
    with sftp.open(filename) as file:
        for line in file:
            print(colored(line, color), end="")

def print_all_logs(cfg, task_name, hostname_key="hostname"):

    log_dir = os.path.join(cfg.host.work_dir, "logs")
    out_file = os.path.join(log_dir, task_name + ".out")
    err_file = os.path.join(log_dir, task_name + ".err")

    with Connection(cfg.host[hostname_key], cfg.host.user) as c, c.sftp() as sftp:
        try:
            print("------------------")
            print(f"stdout for {task_name}")
            print("------------------")
            print_log(c, sftp, out_file, "green")
            
            print("------------------")
            print(f"stderr for {task_name}")
            print("------------------")
            print_log(c, sftp, err_file, "red")
        except FileNotFoundError:
            print(f"No logs available for {task_name}")

if __name__ == "__main__":
    host_name = "longleaf"
    task_name = sys.argv[1]
    cfg = get_config(host_name)
    print_all_logs(cfg, task_name, "network_hostname")