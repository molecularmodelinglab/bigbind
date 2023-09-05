from typing import Dict, Any, Callable
from task import Task, WorkNode
import networkx as nx
from collections import defaultdict
from utils import recursive_map
from traceback import format_exc
import requests

class Workflow:

    out_nodes: WorkNode
    node_cache: Dict[WorkNode, Any]

    def __init__(self, *out_nodes):
        self.out_nodes = out_nodes
        self.node_cache = {}
        self.graph, self.nodes = self.get_graph()

        self.name_to_nodes = defaultdict(list)
        for node in self.graph.nodes:
            name = node.task.name
            self.name_to_nodes[name].append(node)

    def get_output_task_names(self):
        return [ node.task.name for node in self.out_nodes ]

    def find_node(self, task_name: str):
        """ Returns a list of all nodes with the task_name """
        return self.name_to_nodes[task_name]

    def run_node(self, cfg, node: WorkNode):
        """ Run a single node"""
        try:
            # print(node, node in self.node_cache, node.task.is_finished(cfg))

            if node in self.node_cache:
                return self.node_cache[node]

            if node.task.is_finished(cfg):
                return node.task.get_output(cfg)

            def maybe_run_node(x):
                if isinstance(x, WorkNode):
                    return self.run_node(cfg, x)
            
            args = recursive_map(maybe_run_node, node.args)
            kwargs = recursive_map(maybe_run_node, node.kwargs)
            ret = node.task.full_run(cfg, args, kwargs)

            self.node_cache[node] = ret
        except:
            if "hooks" in cfg and "on_crash" in cfg.hooks:
                if "slack" in cfg.hooks.on_crash:
                    msg = f"Error running {node.task.name} on {cfg.host.name} (run_name={cfg.run_name}):\n\n"
                    msg += format_exc()
                    url = cfg.hooks.on_crash.slack
                    requests.post(url, json={"text": msg})
            raise

        return ret

    def run(self, cfg):
        ret = []
        for out_node in self.out_nodes:
            ret.append(self.run_node(cfg, out_node))
        return ret

    def get_graph(self):
        """ Return a nx graph representing the Workflow DAG. An (directed) edge
        exists between two nodes if one not is in the args or kwargs of another.
        Also returns a list of nodes in a defined order """
        graph = nx.DiGraph()
        nodes = []

        def add_edges(node):
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