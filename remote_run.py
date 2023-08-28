from omegaconf import OmegaConf
import subprocess

def get_cur_branch():
    # get current branch name
    proc = subprocess.Popen("git rev-parse --abbrev-ref HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return  out.decode("utf-8").strip()

def transfer_git(host):
    """ Syncs host's git status with current status """
    
    try:
        cur_branch = get_cur_branch()
        # make sure remote ain't in the transfer branch
        subprocess.run(f"ssh -t {host.user}@{host.hostname} 'cd {host.repo_dir} && git checkout {cur_branch}'", shell=True)

        # push to remote
        new_branch = "transfer"
        subprocess.run(f"git checkout -b {new_branch}", shell=True)
        subprocess.run(f"git add . && git commit -am 'Transferring'", shell=True)
        subprocess.run(f"git push -f {host.git_remote} {new_branch}", shell=True)
        # subprocess.run(f"git push -f --mirror {host.git_remote}", shell=True)

        # copy over hosts file and make sure we're on transfer branch on remote
        subprocess.run(f"ssh -t {host.user}@{host.hostname} 'mkdir -p {host.repo_dir}/configs && cd {host.repo_dir} && git checkout {new_branch}'", shell=True)
        subprocess.run(f"scp configs/hosts.yaml {host.user}@{host.hostname}:{host.repo_dir}/configs/hosts.yaml", shell=True)
    finally:
        # back to og branch
        subprocess.run(f"git symbolic-ref HEAD refs/heads/{cur_branch} && git reset", shell=True)
        subprocess.run(f"git branch -D {new_branch}", shell=True)

def remote_run(host, host_name, func_name, hostname_key="hostname"):
    remote_cmds = []
    if "pre_command" in host:
        remote_cmds.append(host.pre_command)
    remote_cmds.append(f"cd {host.repo_dir}")
    remote_cmds.append(f"nohup python -m local_run {host_name} {func_name} 1>/dev/null 2>/dev/null &")
    cmd = f"printf '{' ; '.join(remote_cmds)} exit' | ssh {host.user}@{host[hostname_key]}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/hosts.yaml")
    host_name = "longleaf"
    func_name = "test_f"
    host = cfg[host_name]
    transfer_git(host)

    remote_run(host, host_name, func_name, 'network_hostname')