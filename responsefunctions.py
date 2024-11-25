# -*- coding: utf-8 -*-


import random
import graphviz
import itertools
import numpy as np
import pandas as pd
import gurobipy as gp
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, count
from networkx.drawing.nx_agraph import to_agraph
from generate_synthetic_data import *
from causalgraph import *

class ResponseFunctions:
    def __init__(self, CausalGraph):
        # Replace 'CausalGraph' by 'cg'
        self.CausalGraph = CausalGraph
        self.cg = self.CausalGraph
        assert nx.is_directed_acyclic_graph(CausalGraph.dag) == True
        states = CausalGraph.states
        self.response_functions = []
        self.response_functions_combinations = []
        self.rf_names = ['R_{}'.format(i) for i in CausalGraph.observed_index]
        for node in CausalGraph.nodes:
            if nx.ancestors(CausalGraph.dag,node) == set():
                self.response_functions.append(CausalGraph.nodes_dict.get(node))
                self.response_functions_combinations.append(tuple(CausalGraph.domain_dict.get(node)))
            else:
                self.response_functions.append(CausalGraph.nodes_dict.get(node)**np.prod([CausalGraph.nodes_dict[k] for k in sorted(list(CausalGraph.dag.predecessors(node)))])) # Correct
                self.response_functions_combinations.append(list(product(range(CausalGraph.nodes_dict.get(node)), repeat =np.prod([CausalGraph.nodes_dict[nodes] for nodes in sorted(list(CausalGraph.dag.predecessors(node)))]))))
        self.rf_dict = dict(zip(CausalGraph.nodes, self.response_functions_combinations))
        self.numRF_dict = dict(zip(CausalGraph.nodes, self.response_functions))
        self.rf_nodes = ['R_{}'.format(i) for i in CausalGraph.observed_index]
        self.rf_nodes_dict = dict(zip(self.rf_nodes, CausalGraph.latent_index))
        self.rf_conf_edges = []
        for (a, b) in CausalGraph.latent_edges:
            (x, y) = (CausalGraph.observed_dict[a], CausalGraph.observed_dict[b])
            self.rf_conf_edges.append([list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(x)],
                                        list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(y)]])
        self.rf_conf_edges = [tuple(l) for l in self.rf_conf_edges]
        self.rf_edges = [(self.rf_nodes[i], CausalGraph.nodes[i]) for i in range(len(CausalGraph.nodes))]
        self.rf_observed_nodes_dict = dict(zip(self.rf_nodes, CausalGraph.nodes))

    def RF_graph(self):

        dot = graphviz.Digraph()
        for node in self.cg.nodes:
            dot.node(node, node, {"shape": "ellipse"})

        for a, b in self.cg.edges:
            if a in self.cg.nodes and b in self.cg.nodes:
                dot.edge(a, b)

        for name in self.rf_nodes:
            dot.node(name, name, {"shape": "rectangle", "color": "grey"})

        for (a, b) in self.rf_edges:
            dot.edge(a, b)

        for (a, b) in self.rf_conf_edges:
            dot.edge(a, b)
        return dot

    def Indicator(self, node, par, R, observation):
        states = self.cg.nodes_dict.get(node)
        if nx.ancestors(self.cg.dag,node)==set():
            Resp = self.numRF_dict.get(node)
            lst = [self.cg.domain_dict.get(node)]
            names = list(product(range(0,self.cg.nodes_dict[node]),repeat=0))
            final = dict(zip(names,lst))
        else:
            parents= sorted(list(self.cg.dag.predecessors(node)))
            index = np.prod([self.cg.nodes_dict[nodes] for nodes in sorted(list(self.cg.dag.predecessors(node)))])
            Resp = self.numRF_dict.get(node)
            lst = [self.rf_dict.get(node)[response][i] for i in range(0,index) for response in range(0,Resp)]
            A = [lst[i:i + Resp] for i in range(0, len(lst), Resp)]
            names = list(product(*[range(self.cg.nodes_dict[node]) for node in parents]))
            final = dict(zip(names,A))
        return 1 if observation == final.get(par)[R] else 0

    def par_by_node(self, node):
        states = self.cg.nodes_dict.get(node)
        if nx.ancestors(self.cg.dag,node)==set():
            Resp = self.numRF_dict.get(node)
            lst = [self.cg.domain_dict.get(node)]
            names = list(product(range(0,self.cg.nodes_dict[node]),repeat=0))
            final = dict(zip(names,lst))
        else:
            parents= sorted(list(self.cg.dag.predecessors(node)))
            index = np.prod([self.cg.nodes_dict[nodes] for nodes in sorted(list(self.cg.dag.predecessors(node)))])
            Resp = self.numRF_dict.get(node)
            lst = [self.rf_dict.get(node)[response][i] for i in range(0,index) for response in range(0,Resp)]
            A = [lst[i:i + Resp] for i in range(0, len(lst), Resp)]
            names = list(product(*[range(self.cg.nodes_dict[node]) for node in parents]))
            final = dict(zip(names,A))
        return final.keys()
