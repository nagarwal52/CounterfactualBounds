import random
import graphviz
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, count
from networkx.drawing.nx_agraph import to_agraph
from generate_synthetic_data import *

class CausalGraph:
    def __init__(self, data, edges, latent_edges):
        self.data = data
        self.edges = edges
        self.latent_edges = latent_edges
        self.dag = nx.DiGraph()
        self.nodes = list(self.data.keys())
        type_of_actions = ['non-actionable', 'actionable', 'actionable']
        A = dict(zip(self.nodes, type_of_actions))
        #attrs = {'X_1': {"attr1": 20}, 'X_2': {"attr2": 3}, 'X_3': {'attr3':7}}
        nx.set_node_attributes(self.dag, A , name='types')
        #groups = set(nx.get_node_attributes(self.dag,'types').values())

        self.observed_index = [self.nodes.index(node)+1 for node in self.nodes]
        self.observed_dict = dict(zip(self.nodes,self.observed_index))
        self.latent_nodes = ['U_{}'.format(i) for i in self.observed_index] 
        #self.rf_nodes = ['R_{}'.format(i) for i in self.observed_index] 
        self.latent_index = [self.latent_nodes.index(lat_node)+1 for lat_node in self.latent_nodes]
        #self.nodes_names_dict = dict(zip(self.rf_nodes,self.nodes))
        self.latent_dict = dict(zip(self.latent_nodes,self.latent_index))
        #self.rf_nodes_dict = dict(zip(self.rf_nodes,self.latent_index))
        self.node_labels = {'X_1': 'actionable',
                   'X_2': 'actionable',
                   'X_3':'actionable',
                   'X_4':'non-actionable',
                   'X_5':'non-actionable'}
        self.edges = edges
        self.dimension = len(list(self.data.values())[0])
        #self.states = len(np.unique(list(self.data.values())[0]))
        self.dag.add_nodes_from(self.nodes)
        self.dag.add_edges_from(self.edges)
        self.data_nd = np.array([list(self.data.values())[i] for i in range(len(self.nodes))])
        self.domain = [list(np.unique(data.get(node))) for node in self.nodes]
        self.domain_dict = dict(zip(self.nodes,self.domain))
        self.dimension = len(list(data.values())[0])
        self.states = [len(np.unique(self.data.get(node))) for node in self.nodes]
        self.nodes_dict = dict(zip(self.nodes,self.states))
        self.nodes_with_latent = dict(zip(self.nodes,self.latent_nodes))
        self.boundary_edges = [(self.nodes_with_latent.get(node),node) for node in self.nodes] 
        #self.dag.add_nodes_from(self.rf_nodes)
        self.model = gp.Model('bilinear')
        self.conf_edges = []
        for (a,b) in self.latent_edges:
            (x,y) = (self.observed_dict[a],self.observed_dict[b])
            self.conf_edges.append([list(self.latent_dict.keys())[list(self.latent_dict.values()).index(x)],
                    list(self.latent_dict.keys())[list(self.latent_dict.values()).index(y)]])
        self.conf_edges = [tuple(l) for l in self.conf_edges]

        #self.lg = self.ResponseFunctionsons()
        
    def classifier(self):
        np.random.seed(SEED)
        #return np.random.random(self.states)
        return np.random.uniform(0,1, tuple(self.states))
    
    def draw(self):
        dot = graphviz.Digraph()
        for i, node in enumerate(self.nodes):
            #dot.node(node, node, {"shape": "ellipse", "color": colors[i]}, texmode="math")
            dot.node(node, node, {"shape": "ellipse"}, texmode="math")

        for a, b in self.edges:
            if a in self.nodes and b in self.nodes:
                dot.edge(a,b)

        for name in self.latent_nodes:
            dot.node(name, name, {"shape":"circle", "color":"grey"})

        for i in range(0,len(self.nodes)):
            dot.edge(self.latent_nodes[i],self.nodes[i])

        for (a,b) in self.conf_edges:
            dot.edge(a,b,style='dashed',dir='both')

        return dot
    def distribution(self):
        jointProbs, _ = np.histogramdd(self.data_nd.T, bins=tuple(self.states))
        jointProbs /= jointProbs.sum()
        p_vec = jointProbs.flatten()
        num_variable= len(self.nodes)
        joint_event = list(product(*[range(self.nodes_dict[node]) for node in self.nodes]))
        prob_value_dict = dict(zip(joint_event,p_vec))
        return p_vec, prob_value_dict, joint_event
