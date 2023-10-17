import sys
import os
import argparse
import asyncio
from baselines.dock_all import make_dock_workflow

from utils.cfg_utils import get_config
from bigbind.bigbind import make_bigbind_workflow
from baselines.vina_gnina import make_vina_gnina_workflow
from utils.slurm import submit_slurm_task
from utils.sync import sync_to

def run_single_node(cfg, workflow, node_index):
    node = workflow.nodes[node_index]
    workflow.run_node(cfg, node)

def submit_single_task(cfg, workflow, node):
    """ Submit a single task (currently to SLURM) """
    submit_slurm_task(cfg, workflow, node)
    

async def submit_task(cfg, workflow, node, prereq_procs):
    await asyncio.gather(*prereq_procs)

    index = workflow.nodes.index(node)

    log_dir = os.path.join(cfg.host.work_dir, cfg.run_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_file = os.path.join(log_dir, node.task.name + ".out")
    err_file = os.path.join(log_dir, node.task.name + ".err")

    run_cmds = []
    if "pre_command" in cfg.host:
        run_cmds.append(cfg.host.pre_command)
    run_cmds.append(f"cd {cfg.host.repo_dir}")
    run_cmds.append(f"python -m utils.local_run {cfg.host.name} -n {index}")
    run_cmds = " && ".join(run_cmds)

    cmd = f" ( {run_cmds} ) 1> {out_file} 2> {err_file} "
    print(f" Running {cmd}")

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    return await proc.communicate()

async def submit_tasks(cfg, workflow, task_names):
    """ Submits (either by running directly or using SLURM) the task
    and all its ancestor tasks, respecting dependencies """
    final_nodes = []
    for name in task_names:
        final_nodes += workflow.find_node(name)

    ancestors = set()
    for node in final_nodes:
        ancestors.add(node)
        ancestors = ancestors.union(node.get_all_ancestors(cfg))

    to_search = { node for node in ancestors if node.can_submit(cfg) }
    
    node2job_id = {}

    while len(to_search):
        for node in to_search:
            cur_anc = { n for n in node.get_all_ancestors(cfg) if n.can_submit(cfg) }
            try:
                job_ids = { node2job_id[anc] for anc in cur_anc }
                break
            except KeyError:
                pass

        print("\n----")
        print(f"Running {node} with dependencies {job_ids}")
        print("----")

        if "slurm" in cfg.host:
            node2job_id[node] = submit_slurm_task(cfg, workflow, node, job_ids)
        else:
            node2job_id[node] = asyncio.create_task(submit_task(cfg, workflow, node, job_ids))
        to_search.remove(node)

    if "slurm" not in cfg.host:
        await asyncio.gather(*node2job_id.values())

    # for level in workflow.get_levels(final_nodes):
    #     for node in level:
    #         if not node.can_submit(cfg): continue
    #         submit_single_task(cfg, workflow, node)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="Name of host in hosts file")
    parser.add_argument("-t", "--task-name")
    parser.add_argument("-n", "--node-index", type=int)
    parser.add_argument("-s", "--submit", action="store_true")
    parser.add_argument("--sync", help="sync with gs bucket before running", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.host)
    # workflow = make_bigbind_workflow(cfg)
    workflow = make_dock_workflow(cfg)

    if args.sync:
        sync_to(cfg)

    if args.node_index is not None:
        if args.submit:
            raise NotImplementedError
        else:
            run_single_node(cfg, workflow, args.node_index)
    
    else:
        task_names = workflow.get_output_task_names() if args.task_name is None else [ args.task_name ]

        if args.submit:
            asyncio.run(submit_tasks(cfg, workflow, task_names))
        else:
            raise NotImplementedError
