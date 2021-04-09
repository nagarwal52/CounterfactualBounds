import networkx as nx
import graphviz
import itertools
from itertools import combinations, chain, repeat, zip_longest, permutations
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from operator import add
import cvxpy as cp
from cvxpy import reshape, matmul, vec, hstack, vstack
import time
from cvxpy.reductions.solvers.solver import Solver
from functools import reduce, partial
from operator import mul

np.random.seed(2021)

class CausalGraph:
  def __init__(self, data, edges, latent_edges):
    
    """
    data = {node: data points corresponding to the node}
    edges = [e1,e2,e3,...], i.e.
    latent_edges = []
    Input Arguments
    """
    self.data = data
    self.edges = edges
    self.latent_edges = latent_edges

    # Extracting information
    self.dag = nx.DiGraph()
    self.nodes = list(self.data.keys())
    
    self.observed_index = [self.nodes.index(node)+1 for node in self.nodes]
    self.observed_dict = dict(zip(self.nodes,self.observed_index))
    self.latent_nodes = ['U_{}'.format(i) for i in self.observed_index] 
    #self.rf_nodes = ['R_{}'.format(i) for i in self.observed_index] 
    self.latent_index = [self.latent_nodes.index(lat_node)+1 for lat_node in self.latent_nodes]
    #self.nodes_names_dict = dict(zip(self.rf_nodes,self.nodes))
    self.latent_dict = dict(zip(self.latent_nodes,self.latent_index))
    #self.rf_nodes_dict = dict(zip(self.rf_nodes,self.latent_index))

    self.edges = edges
    self.dimension = len(list(self.data.values())[0])
    #self.states = len(np.unique(list(self.data.values())[0]))
    self.dag.add_nodes_from(self.nodes)
    self.dag.add_edges_from(self.edges)
    self.data_nd = np.array(tuple(self.data.values()))
    self.domain = [list(np.unique(data.get(node))) for node in self.nodes]
    self.domain_dict = dict(zip(self.nodes,self.domain))
    self.dimension = len(list(data.values())[0])
    self.states = [len(np.unique(self.data.get(node))) for node in self.nodes]
    self.nodes_dict = dict(zip(self.nodes,self.states))
    self.nodes_with_latent = dict(zip(self.nodes,self.latent_nodes))
    self.boundary_edges = [(self.nodes_with_latent.get(node),node) for node in self.nodes] 
    #self.dag.add_nodes_from(self.rf_nodes)

    self.dag.add_nodes_from(self.latent_nodes)
    self.dag.add_edges_from(self.boundary_edges)
    self.conf_edges = []
    for (a,b) in self.latent_edges:
      (x,y) = (self.observed_dict[a],self.observed_dict[b])
      self.conf_edges.append([list(self.latent_dict.keys())[list(self.latent_dict.values()).index(x)],
                    list(self.latent_dict.keys())[list(self.latent_dict.values()).index(y)]])
      
    self.conf_edges = [tuple(l) for l in self.conf_edges]
    # Draw 
    
  def draw(self):

    dot = graphviz.Digraph()
    for node in self.nodes:
      dot.node(node, node, {"shape": "ellipse"})

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
    
    # Probability distribution
  def distribution(self):
    jointProbs, _ = np.histogramdd(self.data_nd.T, bins=tuple(self.states))
    jointProbs /= jointProbs.sum()
    p_vec = jointProbs.flatten()
    num_variable= len(self.nodes)
    joint_event = list(product(*[range(self.nodes_dict[node]) for node in self.nodes]))
    prob_value_dict = dict(zip(joint_event,p_vec))

    return p_vec, prob_value_dict, joint_event

  def classifier(self):
    return np.random.uniform(0,1, tuple(self.states)).flatten()

  def RF_formulation(self):
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    # Check for cylclicity
    assert nx.is_directed_acyclic_graph(self.dag) == True
    states = self.states
    response_functions = []
    response_functions_combinations = []
    for node in self.nodes:
      if nx.ancestors(self.dag,node) == set():
        response_functions.append(self.nodes_dict.get(node))
        response_functions_combinations.append(tuple(self.domain_dict.get(node)))
      else:
        response_functions.append(self.nodes_dict.get(node)**np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))]))
        response_functions_combinations.append(list(product(range(self.nodes_dict.get(node)), repeat =np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))]))))
    rf_names = ['R_{}'.format(i) for i in self.observed_index] 
    rf_dict = dict(zip(self.nodes, response_functions_combinations))
    numRF_dict = dict(zip(self.nodes,response_functions))
    return response_functions, numRF_dict, rf_dict

  def draw_RFgraph(self):
    self.rf_nodes = ['R_{}'.format(i) for i in self.observed_index] 
    self.rf_nodes_dict = dict(zip(self.rf_nodes,self.latent_index))
    self.rf_conf_edges = [('R_1','R_3'),('R_2','R_3')]
    """
    self.rf_conf_edges = []
    for (a,b) in self.latent_edges:
      (x,y) = (self.observed_dict[a], self.observed_dict[b])
      self.rf_conf_edges.append([list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(x)],
      list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(y)]])
    self.rf_conf_edges = [tuple(l) for l in self.rf_conf_edges]
    """
    self.rf_edges = [(self.rf_nodes[i], self.nodes[i]) for i in range(len(self.nodes))]
    self.rf_nodes_dict = dict(zip(self.rf_nodes, self.nodes))
    dot = graphviz.Digraph()
    for node in self.nodes:
      dot.node(node, node, {"shape": "ellipse"})

    for a, b in self.edges:
      if a in self.nodes and b in self.nodes:
        dot.edge(a,b)
        
    for name in self.rf_nodes:
      dot.node(name, name, {"shape":"rectangle", "color":"grey"})
      
    for (a,b) in self.rf_edges:
      dot.edge(a,b)
      
    for (a,b) in self.rf_conf_edges:
      dot.edge(a,b)
      
    return dot
   
  def Indicator(self,node, par, R, observation):
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    states = self.nodes_dict.get(node)
    if nx.ancestors(self.dag,node)==set():
      Resp = self.RF_formulation()[1].get(node)
      lst = [self.domain_dict.get(node)]
      names = list(product(range(0,self.nodes_dict[node]),repeat=0))
      final = dict(zip(names,lst))
    else:
      parents= sorted(list(self.dag.predecessors(node)))
      index = np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))])
      Resp = self.RF_formulation()[1].get(node)
      lst = [self.RF_formulation()[2].get(node)[response][i] for i in range(0,index) for response in range(0,Resp)]
      A = [lst[i:i + Resp] for i in range(0, len(lst), Resp)]
      names = list(product(*[range(self.nodes_dict[node]) for node in parents]))
      final = dict(zip(names,A))
    return 1 if observation == final.get(par)[R] else 0

  def par_value_by_node(self,node):
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    states = self.nodes_dict.get(node)
    if nx.ancestors(self.dag,node)==set():
      Resp = self.RF_formulation()[1].get(node)
      lst = [self.domain_dict.get(node)]
      names = list(product(range(0,self.nodes_dict[node]),repeat=0))
      final = dict(zip(names,lst))
    else:
      parents= sorted(list(self.dag.predecessors(node)))
      index = np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))])
      Resp = self.RF_formulation()[1].get(node)
      lst = [self.RF_formulation()[2].get(node)[response][i] for i in range(0,index) for response in range(0,Resp)]
      A = [lst[i:i + Resp] for i in range(0, len(lst), Resp)]
      names = list(product(*[range(self.nodes_dict[node]) for node in parents]))
      final = dict(zip(names,A))
    return final.keys()

  def Q_matrix(self):
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    # Finding all the combinations of r, observation, par_value
    r = list(product(*[range(self.RF_formulation()[1].get(node)) for node in self.nodes]))*np.prod(self.states)
    joint_event = self.distribution()[2]
    observation = []
    for i in range(0, len(joint_event)):
      observation = observation + [joint_event[i]]*np.prod(self.RF_formulation()[0])
      
    par_tmp = []
    for j,node in enumerate(self.nodes):
      tmp_lst = list((self.par_value_by_node(node)))
      par_tmp = par_tmp + [sorted(tmp_lst*(int(len(observation)/len(tmp_lst))))]
    par = list(zip(*par_tmp))
    lst = [[] for _ in range(np.prod(self.states)*np.prod(self.RF_formulation()[0]))]
    for i in range(0,len(lst)):
      for j, node in enumerate(self.nodes):
        tmp = (node,par[i][j]) + (r[i][j],) + (observation[i][j],)
        lst[i].append(tmp)
        
    Q_tmp = np.zeros((np.prod(self.states)*np.prod(self.RF_formulation()[0]),len(self.nodes)),dtype=int)
    for i in range(0,(np.prod(self.states)*np.prod(self.RF_formulation()[0]))):
      for j,(a,b,c,d) in enumerate(lst[i]):
        Q_tmp[i,j] = self.Indicator(a,b,c,d)
    New = np.prod(Q_tmp, axis = 1)
    Q = New.reshape(np.prod(self.states),np.prod(self.RF_formulation()[0]))
    return Q 
      

