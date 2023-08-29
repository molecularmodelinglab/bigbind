import sys
from cfg_utils import get_config
from task import Task
import downloads

if __name__ == "__main__":
    host_name = sys.argv[1]
    task_name = sys.argv[2]
    if len(sys.argv) > 3:
        force = bool(sys.argv[3])
    else:
        force = False

    cfg = get_config(host_name)
    task = Task.ALL_TASKS[task_name]
    task.full_run(cfg, force=force)