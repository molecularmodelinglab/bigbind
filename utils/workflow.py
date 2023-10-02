from typing import Dict, Any, Callable
import networkx as nx
from collections import defaultdict
from traceback import format_exc
import requests
from copy import deepcopy
from google.cloud import storage

from utils.task import Task, WorkNode
from utils.utils import recursive_map, can_take_kwarg

gs_bucket = None

class Workflow:

    out_nodes: WorkNode
    node_cache: Dict[WorkNode, Any]

    def __init__(self, cfg, *out_nodes):
        self.cfg = cfg
        self.out_nodes = out_nodes
        self.node_cache = {}
        self.graph = self.get_graph()
        self.nodes = self.get_all_nodes()

        self.name_to_nodes = defaultdict(list)
        for node in self.nodes:
            name = node.task.name
            self.name_to_nodes[name].append(node)

        self.prev_run_name = None

    def get_output_task_names(self):
        return [ node.task.name for node in self.out_nodes ]

    def find_node(self, task_name: str):
        """ Returns a list of all nodes with the task_name """
        return self.name_to_nodes[task_name]
    
    def run_node_from_name(self, cfg, task_name: str, force=False):
        """ Runs all nodes with task_name """
        nodes = self.find_node(task_name)
        assert len(nodes) == 1, f"Found {len(nodes)} nodes with name {task_name}"
        return self.run_node(cfg, nodes[0], force)

    def run_node(self, cfg, node: WorkNode, force=False):
        """ Run a single node"""
        try:
            
            if node.task.force:
                force = True

            if node in self.node_cache:
                return self.node_cache[node]

            if not force and node.task.is_finished(cfg):
                return node.task.get_output(cfg)

            def maybe_run_node(x):
                if isinstance(x, WorkNode):
                    return self.run_node(cfg, x)
                return x

            # try to find previous outputs for this task if they exist
            # (and we want them)
            if self.prev_run_name is not None:
                fake_cfg = deepcopy(cfg)
                fake_cfg.run_name = self.prev_run_name
                # print(node, node.task.is_finished(fake_cfg))
                if node.task.is_finished(fake_cfg) and (hasattr(node.task, "func") and can_take_kwarg(node.task.func, "prev_output")): 
                    try:
                        prev_output = node.task.get_output(fake_cfg)
                        node.kwargs["prev_output"] = prev_output
                    except:
                        raise
                        # pass
        
            args = recursive_map(maybe_run_node, node.args)
            kwargs = recursive_map(maybe_run_node, node.kwargs)

            print(f"Running {node}")
            ret = node.task.full_run(cfg, args, kwargs, force)

            self.node_cache[node] = ret

            # sync everything to google cloud
            if "gcloud" in cfg and not node.task.simple and False:
                global gs_bucket

                if gs_bucket is None:
                    storage_client = storage.Client.from_service_account_json('configs/gcloud_key.json')
                    gs_bucket = storage_client.bucket(cfg.gcloud.bucket)

                fake_cfg = deepcopy(cfg)
                fake_cfg.host.work_dir = cfg.gcloud.work_dir
                for func in [ "get_out_filename", "get_completed_filename" ]:
                    from_fname = getattr(node.task, func)(cfg)
                    to_fname = getattr(node.task, func)(fake_cfg)

                    print(f"Uploading {from_fname} to gs {to_fname}")

                    blob = gs_bucket.blob(to_fname)
                    blob.upload_from_filename(from_fname)

        except:
            if "hooks" in cfg and "on_crash" in cfg.hooks:
                if "slack" in cfg.hooks.on_crash:
                    msg = f"Error running {node.task.name} on {cfg.host.name} (run_name={cfg.run_name}):\n\n"
                    msg += format_exc()
                    url = cfg.hooks.on_crash.slack
                    requests.post(url, json={"text": msg})
            raise

        return ret

    def run(self, force=False):
        ret = []
        for out_node in self.out_nodes:
            ret.append(self.run_node(self.cfg, out_node, force))
        return ret

    def get_graph(self):
        """ Return a nx graph representing the Workflow DAG. An (directed) edge
        exists between two nodes if one not is in the args or kwargs of another.
        Also returns a list of nodes in a defined order """
        graph = nx.DiGraph()
        nodes = []

        def add_edges(node):
            if node.task.is_finished(self.cfg):
                return
            nodes.append(node)
            def add_single_edge(x):
                if isinstance(x, WorkNode):
                    add_edges(x)
                    graph.add_edge(node, x)
            recursive_map(add_single_edge, node.args)
            recursive_map(add_single_edge, node.kwargs)
        
        for out_node in self.out_nodes:
            add_edges(out_node)

        return graph, nodes
    
    def get_all_nodes(self):
        """ Returns a list of all nodes in the workflow """
        nodes = []

        def add_nodes(node):
            if node in nodes:
                return
            nodes.append(node)
            def add_single_node(x):
                if isinstance(x, WorkNode):
                    add_nodes(x)
            recursive_map(add_single_node, node.args)
            recursive_map(add_single_node, node.kwargs)
        
        for out_node in self.out_nodes:
            add_nodes(out_node)

        return nodes

    def get_levels(self, nodes):
        """ Runs BFS on the DAG graph starting from nodes. Returns a list
        of lists of nodes, corresponding to the levels from the search.
        The list is reversed from normal BFS. """
        graph = self.graph

        levels = []
        curr_level = nodes
        while len(curr_level) > 0:
            next_level = []
            for node in curr_level:
                next_level += list(graph.successors(node))
            levels.append(curr_level)
            curr_level = next_level
        return list(reversed(levels))

if __name__ == "__main__":
    from utils.task import iter_task, simple_task
    from utils.cfg_utils import get_config

    @simple_task
    def test_input(cfg):
        return list(range(100))

    @iter_task(11, 5)
    def test_iter(cfg, x):
        return x*10

    cfg = get_config("local")
    workflow = Workflow(cfg, test_iter(test_input()))
    print(workflow.run(force=True))
