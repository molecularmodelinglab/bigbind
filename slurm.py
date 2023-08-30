from datetime import timedelta
import subprocess
import os

def submit_slurm_task(cfg, workflow, node):
    index = workflow.nodes.index(node)
    runtime = timedelta(hours=node.task.max_runtime)
    
    hours, remainder = divmod(runtime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    log_dir = os.path.join(cfg.host.work_dir, "logs")
    out_file = os.path.join(log_dir, task_name + ".out")
    err_file = os.path.join(log_dir, task_name + ".err")

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

    run_cmds.append(f"python local_run.py {cfg.host.name} -n {index}")
    run_cmds = " && ".join(run_cmds)

    cmd = f"echo '#!/bin/bash\n {run_cmds}' | sbatch {sbatch_args}"
    print(f" Running {cmd}")

    subprocess.run(cmd, shell=True, check=True)