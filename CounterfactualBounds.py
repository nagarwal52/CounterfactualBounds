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
from itertools import count
from networkx.drawing.nx_agraph import to_agraph
import random
from prettytable import PrettyTable
import gurobipy as gp
from gurobipy import GRB
SEED_VALUE = 2021
np.random.seed(SEED_VALUE )

class CounterfactualBounds:
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
    #b = to_agraph(self.dag)
    #texcode = dot2tex.dot2tex(b.to_string(), format='tikz', crop=True)
    # Draw 
  def classifier(self):
    np.random.seed(SEED_VALUE )
    return np.random.random(self.states)
    
  def draw(self):

    dot = graphviz.Digraph()
    #for node in self.nodes:
      #dot.node(node, node, {"shape": "ellipse", "color": colors[]})

    #colors = ['red', 'yellow', 'green']
    colors = []
    """

    for i in range(len(self.nodes)):
      if type_of_actions[i] == 'actionable':
        colors.append('green')
      else:
        colors.append('red')
    """

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
    
    # Probability distribution
  def distribution(self):
    jointProbs, _ = np.histogramdd(self.data_nd.T, bins=tuple(self.states))
    jointProbs /= jointProbs.sum()
    p_vec = jointProbs.flatten()
    num_variable= len(self.nodes)
    joint_event = list(product(*[range(self.nodes_dict[node]) for node in self.nodes]))
    prob_value_dict = dict(zip(joint_event,p_vec))

    return p_vec, prob_value_dict, joint_event

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
        response_functions.append(self.nodes_dict.get(node)**np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))])) # Correct
        response_functions_combinations.append(list(product(range(self.nodes_dict.get(node)), repeat =np.prod([self.nodes_dict[nodes] for nodes in sorted(list(self.dag.predecessors(node)))]))))
    rf_names = ['R_{}'.format(i) for i in self.observed_index]
    rf_dict = dict(zip(self.nodes, response_functions_combinations))
    numRF_dict = dict(zip(self.nodes,response_functions))

    self.rf_nodes = ['R_{}'.format(i) for i in self.observed_index]
    self.rf_nodes_dict = dict(zip(self.rf_nodes,self.latent_index))
    #example.rf_conf_edges = [('R_1','R_3'),('R_2','R_3')]

    self.rf_conf_edges = []
    for (a,b) in self.latent_edges:
      (x,y) = (self.observed_dict[a],self.observed_dict[b])
      self.rf_conf_edges.append([list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(x)],
              list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(y)]])
    self.rf_conf_edges = [tuple(l) for l in self.rf_conf_edges]

    self.rf_edges = [(self.rf_nodes[i],self.nodes[i]) for i in range(len(self.nodes))]
    self.rf_nodes_dict = dict(zip(self.rf_nodes,self.nodes))
    return response_functions, numRF_dict, rf_dict

  #response_functions = RF_formulation()[0]
  #numRF_dict = RF_formulation()[1]
  #self.rf_nodes = RF_formulation()[2]

  def RF_graph(self):
    self.rf_nodes = ['R_{}'.format(i) for i in self.observed_index]
    self.rf_nodes_dict = dict(zip(self.rf_nodes, self.latent_index))
    # example.rf_conf_edges = [('R_1','R_3'),('R_2','R_3')]

    self.rf_conf_edges = []
    for (a, b) in self.latent_edges:
      (x, y) = (self.observed_dict[a], self.observed_dict[b])
      self.rf_conf_edges.append([list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(x)],
                                    list(self.rf_nodes_dict.keys())[list(self.rf_nodes_dict.values()).index(y)]])
    self.rf_conf_edges = [tuple(l) for l in self.rf_conf_edges]

    self.rf_edges = [(self.rf_nodes[i], self.nodes[i]) for i in range(len(self.nodes))]
    self.rf_nodes_dict = dict(zip(self.rf_nodes, self.nodes))
    # example.rf_index = [example.rf_nodes.index(node)+1 for node in example.rf_nodes]
    # example.rf_dict = dict(zip(example.rf_nodes,example.rf_index))

    dot = graphviz.Digraph()
    for node in self.nodes:
      dot.node(node, node, {"shape": "ellipse"})

    for a, b in self.edges:
      if a in self.nodes and b in self.nodes:
        dot.edge(a, b)

    for name in self.rf_nodes:
      dot.node(name, name, {"shape": "rectangle", "color": "grey"})

    for (a, b) in self.rf_edges:
      dot.edge(a, b)

    for (a, b) in self.rf_conf_edges:
      dot.edge(a, b)
    return dot
  
  def Indicator(self, node, par, R, observation):
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    states = self.nodes_dict.get(node)
    if nx.ancestors(self.dag,node)==set():
      Resp = self.RF_formulation()[1].get(node)
      lst = [self.domain_dict.get(node)]
      names = list(product(range(0,self.nodes_dict[node]),repeat=0))
      final = dict(zip(names,lst))
    else:
      # TO-DO : Clean this
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
    # Removing latent edges
    self.dag.remove_edges_from(self.boundary_edges)
    self.dag.remove_nodes_from(self.latent_nodes)
    numRF_dict = self.RF_formulation()[1]
    response_functions = self.RF_formulation()[0]

    # Finding all the combinations of r, observation, par_value
    r = list(product(*[range(numRF_dict.get(node)) for node in self.nodes]))*np.prod(self.states)
    joint_event = self.distribution()[2]
    observation = []
    for i in range(0, len(joint_event)):
      observation = observation + [joint_event[i]]*np.prod(response_functions)

    par_tmp = []
    for j,node in enumerate(self.nodes):
      tmp_lst = list((self.par_value_by_node(node)))
      par_tmp = par_tmp + [sorted(tmp_lst*(int(len(observation)/len(tmp_lst))))]
    par = list(zip(*par_tmp))

    # All the combination of nodes,r,observation, par_value
    lst = [[] for _ in range(np.prod(self.states)*np.prod(response_functions))]
    for i in range(0,len(lst)):
      for j, node in enumerate(self.nodes):
        tmp = (node,par[i][j]) + (r[i][j],) + (observation[i][j],)
        lst[i].append(tmp)

    # P matrix
    A_check = np.zeros((np.prod(self.states)*np.prod(response_functions),len(self.nodes)),dtype=int)
    for i in range(0,(np.prod(self.states)*np.prod(response_functions))):
      for j,(a,b,c,d) in enumerate(lst[i]):
        A_check[i,j] = self.Indicator(a,b,c,d)
    New = np.prod(A_check, axis = 1)
    Q_fin = New.reshape(np.prod(self.states),np.prod(response_functions))
    return Q_fin

  def bounds(self, confounding_type):
    t = PrettyTable(['Intervened node','Factual observation (xF)','Lower Bound', 'Upper Bound', 'Decision'])
    #classifier = (example.states).flatten()
    joint_event = self.distribution()[2] # Remove this later
    h = self.classifier().flatten()
    h_dict = dict(zip(joint_event, h))
    xF = [i for i in joint_event if h_dict[i]<0.5]
    numRF_dict = self.RF_formulation()[1]
    response_functions = self.RF_formulation()[0]
    p = self.distribution()[0].reshape(np.prod(self.states),1)
    names = list(product(*[range(numRF_dict.get(node)) for node in self.nodes]))
    q = cp.Variable(shape=(np.prod(response_functions),1))
    Constraints = [q >= 10e-10, sum(q) == 1, p == self.Q_matrix() @ q]
    q_dict = dict(zip(names,q))
    for n in self.nodes:
      if sorted(list(nx.descendants(self.dag, n))) != []:
        self.dag.remove_nodes_from(self.rf_nodes)
        self.dag.remove_edges_from(self.rf_conf_edges)
        self.dag.remove_edges_from(self.rf_edges)
        intervened_node = n
        descendant_nodes = sorted(list(nx.descendants(self.dag, intervened_node))) # List of all descendants
        nd_set_inclusive = sorted(list(set(nx.topological_sort(self.dag)).difference(nx.descendants(self.dag,intervened_node))))
        nd_set_exclude = sorted(list(set(nx.topological_sort(self.dag)).difference(nx.descendants(self.dag,intervened_node)).symmetric_difference(set([intervened_node]))))
        all_nodes = self.nodes + sorted(descendant_nodes)
        for factual in xF:
          xF_tup = factual
          xF_lst = list(factual)
          xF_dict = dict(zip(self.nodes,xF_lst))
          xF_intervened_node = factual[self.observed_dict[intervened_node]-1]
          domain_copy = self.domain_dict[intervened_node].copy()
          domain_copy.remove(xF_intervened_node)
          theta_lst = domain_copy
          for th in theta_lst:
            theta = (th,)
            r = list(product(*[range(numRF_dict.get(node)) for node in self.nodes]))*\
            np.prod([self.nodes_dict.get(node) for node in descendant_nodes])

            r_des = list(product(*[range(numRF_dict.get(node)) for node in sorted(descendant_nodes)]))*\
            np.prod([numRF_dict[node] for node in nd_set_inclusive])*\
            np.prod([self.nodes_dict.get(node) for node in descendant_nodes])

            assert len(r) == len(r_des) # Sanity check
            r_lst = [r[i]+r_des[i] for i in range(len(r))] 
            
            # observation
            joint = list(product(*[range(self.nodes_dict[nodes]) for nodes in sorted(descendant_nodes)]))
            obs_des = [item for sublist in [list(repeat(joint[i],np.prod(response_functions))) for i in range(len(joint))] for item in sublist]
            xf_obs = [xF_tup]*len(obs_des)
            observed = [xf_obs[i]+ obs_des[i] for i in range(len(obs_des))]
            obs_des_dict = dict(zip(descendant_nodes,list(zip(*obs_des))))

            # par
            par_set = [list(self.dag.predecessors(node)) for i,node in enumerate(self.nodes)]
            d_par = [[] for _ in range(len(par_set[1:]))]
            for i in range(len(par_set[1:])):
              for j, node in enumerate(sorted(par_set[1:][i])):
                tmp = xF_dict[node]
                d_par[i].append(tmp)

            par_xf = list(repeat([()] + [tuple(l) for l in d_par], np.prod([self.nodes_dict.get(node) for node in descendant_nodes])*np.prod(response_functions)))

            par_d_set = [sorted(list(self.dag.predecessors(node))) for i,node in enumerate(descendant_nodes)]
            d_par_des = [[] for _ in range(len(par_d_set))]
            TMP = [item for sublist in par_d_set for item in sublist]
            TMP_d = [[] for _ in range(len(TMP))]
            len_parentsofdescendants = [len(list(self.dag.predecessors(node))) for node in descendant_nodes]
            for j, node in enumerate(TMP):
              if node in intervened_node:
                TMP_d[j].append(list(repeat(list(theta)[0], len(par_xf))))
              elif node in nd_set_exclude:
                #TMP_d[j].append(xF_dict[node])
                TMP_d[j].append(list(repeat(xF_dict[node], len(par_xf))))
              else:
                #TMP_d[j].append(list(itertools.chain.from_iterable(itertools.repeat(x, np.prod(response_functions)) for x in example.domain_dict[node])))
                TMP_d[j].append(obs_des_dict[node])
            TMP_d = [item for sublist in TMP_d for item in sublist]
            A = list(zip(*TMP_d))
            tmp = []
            for i in range(len(A)):
              it = iter(A[i])
              tmp.append([tuple(next(it) for _ in range(size)) for size in len_parentsofdescendants])
            all_par = [par_xf[i]+tmp[i] for i in range(len(par_xf))]

            lst = [[] for _ in range(np.prod([self.nodes_dict.get(node) for node in descendant_nodes])*np.prod(response_functions))]
            for i in range(0,len(lst)):
              for j, node in enumerate(all_nodes):
                tmp = (node,all_par[i][j],) + (r_lst[i][j],) + (observed[i][j],)
                lst[i].append(tmp)  

            # Find out the value of h
            h_nd = [tuple(l) for l in list(repeat([(xF_lst[self.observed_dict[n]-1]) for n in nd_set_exclude],len(joint)))]
            h_int = list(repeat(theta,len(joint)))
            h_des = joint
            h_indexes = [h_nd[i] + h_int[i] + h_des[i] for i in range(len(joint))]
            #h_values = np.asarray([h_dict.get(i) for i in h_indexes]).reshape(1,-1)
            h_values = [h_dict.get(i) for i in h_indexes] 

            

            # Indicator function value for factual observation and descendant variables
            B_mat = np.zeros((np.prod([self.nodes_dict.get(node) for node in descendant_nodes])*np.prod(response_functions),len(all_nodes)),dtype=int)
            for i in range(len(B_mat)):
              for j, (a,b,c,d) in enumerate(lst[i]):
                B_mat[i,j] = self.Indicator(a,b,c,d)
            Ind_mat = np.prod(B_mat, axis = 1).reshape(np.prod([self.nodes_dict.get(node) for node in descendant_nodes]),np.prod(response_functions)).T
      
            Query = (q.T @ Ind_mat).T
            #print(Query.shape, len(h_values))
            Objective = (h_values @ Query)*(1/self.distribution()[1].get(xF_tup))
            if confounding_type == 'full':
              minimum = cp.Problem(cp.Minimize(Objective),Constraints)
              maximum = cp.Problem(cp.Maximize(Objective),Constraints)
              assert minimum.is_dqcp()
              assert maximum.is_dqcp()
              LB = minimum.solve(verbose=False)
              UB = maximum.solve(verbose=False)
              if 0.5 < LB and UB > 0.5:
                result = 'Informative bound, i.e., action is recommeneded'
              elif 0.5 > LB and 0.5 > UB : 
                result = 'Non-informative bound'
              elif 0.5 > LB and UB > 0.5:
                result = 'Informative bound but not certain about the action'
              assert LB >=0 and UB <=1
              t.hrules = 1
              t.add_row([intervened_node + str('(')+ 'do(theta=' + str(theta[0]) + str(')'), str(xF_lst), LB,UB,result])
            elif confounding_type == 'partial':
              print('To add')

    print(t)
    print('NOTE: If a different classifier is used, bounds (and therefore decision) may change.\n To change the output of classifier used here (as an example), change the "SEED_VALUE".')
      

