from functools import wraps
import os
from typing import List, Dict, Any
import pickle
from utils import recursive_map

class WorkNode:
    
    task: "Task"
    args: List[Any]
    kwargs: Dict[str, Any]

    def __init__(self, task, args, kwargs):
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def can_submit(self, cfg):
        """ Returns true if you can submit the task to e.g. slurm.
        Returns false when the task has already completed or is simple """
        if self.task.is_finished(cfg) or self.task.simple:
            return False
        return True

    def get_parents(self):
        """ Returns the immediate parents of the task,
        from the args and kwargs"""

        parents = set()

        def add_parent(item):
            nonlocal parents
            if isinstance(item, WorkNode):
                parents.add(item)
        
        recursive_map(add_parent, self.args)
        recursive_map(add_parent, self.kwargs)

        return parents

    def get_all_ancestors(self, cfg):
        """ Returns a list of all the ancestors of the node, excluding
        all the tasks that have already run """
        
        ancestors = set()

        def add_ancestor(item):
            nonlocal ancestors
            if isinstance(item, WorkNode) and not item.task.is_finished(cfg):
                ancestors.add(item)
                ancestors = ancestors.union(item.get_all_ancestors(cfg))
        
        recursive_map(add_ancestor, self.args)
        recursive_map(add_ancestor, self.kwargs)

        return ancestors

    def __repr__(self):
        return f"WorkNode[{self.task.name}]"

    def __getitem__(self, index):
        # super hacky way to make a task on demand
        if index >= self.task.num_outputs:
            raise IndexError
        f = lambda cfg, s, i: s[i]
        f.__name__ = f"getitem_{self.task.name}_{index}"
        f = simple_task(f)

        f_index = lambda cfg: index
        f_index.__name__ = f"index_{self.task.name}_{index}"
        f_index = simple_task(f_index)

        return f(self, f_index())

class Task:
    ALL_TASKS = {}

    def __init__(self,
                 name,
                 out_filename_rel, 
                 local=False,
                 # slurm stuff
                 max_runtime=1, # hours
                 n_cpu=1,
                 mem=2, # GB
                 simple=False, # is it so simple you don't need to cache?
                 num_outputs=1,
                 ):
        self.name = name
        self._out_filename_rel = out_filename_rel
        self.local = local

        self.max_runtime = max_runtime
        self.n_cpu = n_cpu
        self.mem = mem

        self.simple = simple
        self.num_outputs = num_outputs

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
        if self.simple:
            return False
        try:
            completed_filename = self.get_completed_filename(cfg)
            with open(completed_filename, "r") as f:
                if f.readlines()[0] == "completed\n":
                    return True
        except OSError:
            return False

    def full_run(self, cfg, args, kwargs, force=False):
        """ Checks to see if we already ran it -- if so, we're good!"""

        if self.simple:
            return self.run(cfg, *args, **kwargs)

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
        assert not self.simple # should never call this on simple task
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

class SimpleTask(Task):
    """ Tasks that either run very quickly or do their own caching """

    def __init__(self, func):
        super().__init__(func.__name__, None, simple=True)
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def simple_task(f):
    return wraps(f)(SimpleTask(f))

class PickleTask(Task):
    """ The default task. Saves a pickle of the function output """

    def __init__(self, func, *args, **kwargs):
        super().__init__(func.__name__, "output.pkl", *args, **kwargs)
        self.func = func

    def run(self, cfg, *args, **kwargs):
        ret = self.func(cfg, *args, **kwargs)
        with open(self.get_out_filename(cfg), "wb") as f:
            pickle.dump(ret, f)

    def get_output(self, cfg):
        with open(self.get_out_filename(cfg), "rb") as f:
            return pickle.load(f)

def task(*args, **kwargs):
    def wrapper(f):
        return wraps(f)(PickleTask(f, *args, **kwargs))
    return wrapper