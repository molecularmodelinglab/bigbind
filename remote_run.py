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

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/hosts.yaml")
    host = cfg.unc_data
    transfer_git(host)