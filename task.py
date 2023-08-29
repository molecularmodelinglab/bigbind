from functools import wraps
import os

class Task:
    ALL_TASKS = {}

    def __init__(self, name, out_filename_rel, _global=True):
        self.name = name
        self._out_filename_rel = out_filename_rel
        self._global = _global
        if self.name in Task.ALL_TASKS:
            raise Exception(f"Trying to define another Task with name {name}")
        Task.ALL_TASKS[self.name] = self

    def get_out_folder(self, cfg):
        prefix = "global" if self._global else "local"
        return os.path.join(cfg.host.work_dir, cfg.run_name, prefix, self.name)

    def get_out_filename(self, cfg):
        """ Returns the final output filename for the task.
        Also makes sure the directory exists """
        ret = os.path.join(self.get_out_folder(cfg), self._out_filename_rel)
        dir = os.path.dirname(ret)
        os.makedirs(dir, exist_ok=True)
        return ret

    def run(self, cfg):
        """ The idea is that run will input the host info (and all
        the input data) and write to the out_filename"""
        raise NotImplementedError

    def get_completed_filename(self, cfg):
        return os.path.join(self.get_out_folder(cfg), "completed.txt")

    def is_finished(self, cfg):
        try:
            completed_filename = self.get_completed_filename(cfg)
            with open(completed_filename, "r") as f:
                if f.readlines()[0] == "completed\n":
                    return True
        except OSError:
            return False

    def full_run(self, cfg, force=False):
        """ Checks to see if we already ran it -- if so, we're good!"""

        completed_filename = self.get_completed_filename(cfg)

        if self.is_finished(cfg):
            if force:
                os.remove(completed_filename)
            else:
                print(f"Using cached data from {self.name}")
                return
        self.run(cfg)
        dir = os.path.dirname(completed_filename)
        os.makedirs(dir, exist_ok=True)
        with open(completed_filename, "w") as f:
            f.write("completed\n")

# class FuncTask:

#     def __init__(self, func):
#         super().__init__(func.__name__)
#         self.run = func

# def task(func):
#     return wraps(func)(FuncTask(func))

