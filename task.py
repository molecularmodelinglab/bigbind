from functools import wraps
import os
from typing import List, Dict, Any

class WorkNode:
    
    task: "Task"
    args: List[Any]
    kwargs: Dict[str, Any]

    def __init__(self, task, args, kwargs):
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"WorkNode[{self.task.name}]"

class Task:
    ALL_TASKS = {}

    def __init__(self, name, out_filename_rel, 
                 local=False,
                 # slurm stuff
                 max_runtime=1, # hours
                 n_cpu=1,
                 mem=2, # GB
                 ):
        self.name = name
        self._out_filename_rel = out_filename_rel
        self.local = local

        self.max_runtime = max_runtime
        self.n_cpu = n_cpu
        self.mem = mem

        if self.name in Task.ALL_TASKS:
            raise Exception(f"Trying to define another Task with name {name}")
        Task.ALL_TASKS[self.name] = self

    def get_out_folder(self, cfg):
        prefix = "local" if self.local else "global"
        return os.path.join(cfg.host.work_dir, cfg.run_name, prefix, self.name)

    def get_out_filename(self, cfg):
        """ Returns the final output filename for the task.
        Also makes sure the directory exists """
        ret = os.path.join(self.get_out_folder(cfg), self._out_filename_rel)
        dir = os.path.dirname(ret)
        os.makedirs(dir, exist_ok=True)
        return ret

    def run(self, cfg, *args, **kwargs):
        """ The idea is that run will input the host info (and all
        the input data) and write to the out_filename"""
        raise NotImplementedError

    def get_completed_filename(self, cfg):
        return os.path.join(self.get_out_folder(cfg), "completed.txt")

    def is_finished(self, cfg):
        """ Returns true if we can just call get_output directly,
        using the cached output """
        try:
            completed_filename = self.get_completed_filename(cfg)
            with open(completed_filename, "r") as f:
                if f.readlines()[0] == "completed\n":
                    return True
        except OSError:
            return False

    def full_run(self, cfg, args, kwargs, force=False):
        """ Checks to see if we already ran it -- if so, we're good!"""

        completed_filename = self.get_completed_filename(cfg)

        if self.is_finished(cfg):
            if force:
                os.remove(completed_filename)
            else:
                print(f"Using cached data from {self.name}")
                return self.get_output(cfg)
        
        self.run(cfg, *args, **kwargs)
        dir = os.path.dirname(completed_filename)
        os.makedirs(dir, exist_ok=True)
        with open(completed_filename, "w") as f:
            f.write("completed\n")

        return self.get_output(cfg)

    def get_output(self, cfg):
        """ By default, return the output filename. But subclasses
        can also load the output file contents as well """
        return self.get_out_filename(cfg)

    def __call__(self, *args, **kwargs):
        """ Lazily create Workflow graph """
        return WorkNode(self, args, kwargs)

class FileTask(Task):
    """ Tasks that return nothing but output a particular file"""

    def __init__(self, out_filename_rel, func, *args, **kwargs):
        super().__init__(func.__name__, out_filename_rel, *args, **kwargs)
        self.func = func

    def run(self, cfg, *args, **kwargs):
        return self.func(cfg, self.get_out_filename(cfg), *args, **kwargs)

def file_task(out_filename_rel, *args, **kwargs):
    def wrapper(f):
        return wraps(f)(FileTask(out_filename_rel, f, *args, **kwargs))
    return wrapper

