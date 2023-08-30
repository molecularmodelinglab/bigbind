from datetime import timedelta
import subprocess
import re
import os

sbatch_regex = re.compile("Submitted batch job (.+)")

def submit_slurm_task(cfg, workflow, node):
    """ Submits job to SLURM and returns job id """
    index = workflow.nodes.index(node)
    runtime = timedelta(hours=node.task.max_runtime)
    
    hours, remainder = divmod(runtime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    log_dir = os.path.join(cfg.host.work_dir, cfg.run_name, "logs")
    out_file = os.path.join(log_dir, node.task.name + ".out")
    err_file = os.path.join(log_dir, node.task.name + ".err")

    sbatch_args = []
    sbatch_args.append(f"-J {node.task.name}")
    sbatch_args.append(f"-t {runtime.days:02d}-{int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    sbatch_args.append(f"--cpus-per-task={node.task.n_cpu}")
    sbatch_args.append(f"--mem={node.task.mem}G")
    sbatch_args.append(f"--output={out_file}")
    sbatch_args.append(f"--error={err_file}")
    sbatch_args = " ".join(sbatch_args)

    run_cmds = []
    if "pre_command" in cfg.host:
        run_cmds.append(cfg.host.pre_command)
    run_cmds.append(f"cd {cfg.host.repo_dir}")

    run_cmds.append(f"python local_run.py {cfg.host.name} -n {index}")
    run_cmds = " && ".join(run_cmds)

    cmd = f"cd ~ && echo '#!/bin/bash\n {run_cmds}' | sbatch {sbatch_args}"
    print(f" Running {cmd}")

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
    return sbatch_regex.match(result.stdout.decode('utf-8')).groups()[0]
