from omegaconf import OmegaConf
import subprocess

def transfer_git(host):
    """ Syncs host's git status with current status """
    
    # get current branch name
    proc = subprocess.Popen("git rev-parse --abbrev-ref HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    cur_branch = out.decode("utf-8").strip()
    
    # push to remote
    new_branch = "transfer"
    subprocess.run(f"git checkout -b {new_branch}", shell=True)
    subprocess.run(f"git add . && git commit -am 'Transferring'", shell=True)
    subprocess.run(f"git push -f {host.git_remote} {new_branch}", shell=True)

    # back to og branch
    subprocess.run(f"git symbolic-ref HEAD refs/heads/{cur_branch} && git reset", shell=True)
    subprocess.run(f"git branch -D {new_branch}", shell=True)

    # copy over hosts file
    subprocess.run(f"ssh -t {host.user}@{host.hostname} mkdir -p {host.repo_dir}/configs", shell=True)
    subprocess.run(f"scp configs/hosts.yaml {host.user}@{host.hostname}:{host.repo_dir}/configs/hosts.yaml", shell=True)

def remote_run(host):
    remote_cmds = []
    if "pre_command" in host:
        remote_cmds.append(host.pre_command)
    remote_cmds.append(f"cd {host.repo_dir}")
    remote_cmds.append("echo 'hello world'")
    cmd = f"ssh -t {host.user}@{host.hostname} {'&&'.join(remote_cmds)}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/hosts.yaml")
    host = cfg.unc_data
    transfer_git(host)
    remote_run(host)