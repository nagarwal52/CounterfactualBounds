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

 
class Bounds:
    def __init__(self,CausalGraph,ResponseFunctions):
        self.CausalGraph = CausalGraph
        self.CausalGraph.ResponseFunctions = ResponseFunctions
        self.RF = self.CausalGraph.ResponseFunctions
        self.cg = self.CausalGraph 
        self.results = {'Intervened_node':[],
                            'factual_obs':[],
                        'thetas':[],
                        'Lower_bound_linear':[],
                        'Upper_bound_linear':[],
                        'Lower_bound_MIP':[],
                        'Upper_bound_MIP':[]}

    def Q_matrix(self):
        # Finding all the combinations of r, observation, par_value
        r = list(product(*[range(self.RF.numRF_dict.get(node)) for node in self.cg.nodes]))*np.prod(self.cg.states)
        joint_event = self.cg.distribution()[2]
        observation = []
        for i in range(0, len(joint_event)):
            observation = observation + [joint_event[i]]*np.prod(self.RF.response_functions)

        par_tmp = []
        for j,node in enumerate(self.cg.nodes):
            tmp_lst = list((self.RF.par_by_node(node))) # Change 
            par_tmp = par_tmp + [sorted(tmp_lst*(int(len(observation)/len(tmp_lst))))]
        par = list(zip(*par_tmp))

        # All the combination of nodes,r,observation, par_value
        lst = [[] for _ in range(np.prod(self.cg.states)*np.prod(self.RF.response_functions))]
        for i in range(0,len(lst)):
            for j, node in enumerate(self.cg.nodes):
                tmp = (node, par[i][j]) + (r[i][j],) + (observation[i][j],)
                lst[i].append(tmp)

        # Q matrix
        A_check = np.zeros((np.prod(self.cg.states)*np.prod(self.RF.response_functions),len(self.cg.nodes)),dtype=int)
        for i in range(0,(np.prod(self.cg.states)*np.prod(self.RF.response_functions))):
            for j,(a,b,c,d) in enumerate(lst[i]):
                A_check[i,j] = self.RF.Indicator(a,b,c,d)
        New = np.prod(A_check, axis = 1)
        Q = New.reshape(np.prod(self.cg.states),np.prod(self.RF.response_functions))
        return Q

    def create_variables_and_get_constraints(self):
        p = self.cg.distribution()[0].reshape(np.prod(example.states),1) 
        q = cp.Variable(shape=(np.prod(self.RF.response_functions),1))
        names = list(product(*[range(self.RF.numRF_dict.get(node)) for node in self.cg.nodes]))
        q_dict = dict(zip(names, q))   
        Constraints = [q >= 10e-10, sum(q) == 1, p == self.Q_matrix() @ q]
        return q, Constraints

    def check(self):
        #return self.CausalGraph.ResponseFunctions.rf_nodes, self.CausalGraph.ResponseFunctions.rf_nodes
        return self.create_variables('ALM')[1]

    def reduce(self,A,var,j):
        if len(A) == 2:
            var['u'+ str(j)] = [A[0],A[1]]
            return var

        n = len(A)
        if n % 2 != 0:
            last = n-1 
        else:
            last = n
        temp = []
        for i in range(0,last,2):
            var['u'+ str(j)] = [A[i],A[i+1]]
            temp.append('u'+ str(j))

            j += 1
        if last < n:
            temp.append((A[last]))

        return self.reduce(temp,var,j), var

    def objective(self,classifier_type,confounding_type):
        self.cg.dag.remove_nodes_from(self.RF.rf_nodes)
        self.cg.dag.remove_edges_from(self.RF.rf_conf_edges)
        self.cg.dag.remove_edges_from(self.RF.rf_edges)
        joint_event = self.cg.distribution()[2] # Remove this later

        if classifier_type == 'user':
            h = self.cg.classifier().flatten()
        else:
            print('Toadd')    
        h_dict = dict(zip(joint_event, h))
        xF = [i for i in joint_event if h_dict[i]<0.5]
        actionable_nodes = []
        for node in self.cg.nodes:
            if self.cg.node_labels[node] == 'actionable':
                actionable_nodes.append(node)
        for n in actionable_nodes:
            if sorted(list(nx.descendants(self.cg.dag, n))) != []:
                intervened_node = n
                self.results['Intervened_node'].append(intervened_node)
                descendant_nodes = sorted(list(nx.descendants(self.cg.dag, intervened_node))) # List of all descendants
                nd_set_inclusive = sorted(list(set(nx.topological_sort(self.cg.dag)).difference(nx.descendants(self.cg.dag,intervened_node))))
                nd_set_exclude = sorted(list(set(nx.topological_sort(self.cg.dag)).difference(nx.descendants(self.cg.dag,intervened_node)).symmetric_difference(set([intervened_node]))))
                all_nodes = self.cg.nodes + sorted(descendant_nodes)
                for factual in xF:
                    xF_tup = factual
                    xF_lst = list(factual)
                    xF_dict = dict(zip(self.cg.nodes,xF_lst))
                    xF_intervened_node = factual[self.cg.observed_dict[intervened_node]-1]
                    domain_copy = self.cg.domain_dict[intervened_node].copy()
                    domain_copy.remove(xF_intervened_node)
                    theta_lst = domain_copy
                    self.results['factual_obs'].append(factual)
                    for th in theta_lst:
                        theta = (th,)
                        self.results['thetas'].append(th)
                        response_combinations_for_obsnodes = list(product(*[range(self.RF.numRF_dict.get(node)) for node in self.cg.nodes]))*\
                            np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes])

                        response_combinations_for_desnodes = list(product(*[range(self.RF.numRF_dict.get(node)) for node in sorted(descendant_nodes)]))*\
                            np.prod([self.RF.numRF_dict[node] for node in nd_set_inclusive])*\
                            np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes]) 

                        assert len(response_combinations_for_obsnodes) == len(response_combinations_for_desnodes)
                        #r_lst = [r[i]+r_des[i] for i in range(len(r))] 
                        response_combinations_for_allnodes = [list(itertools.chain.from_iterable(i)) for i in zip(response_combinations_for_obsnodes,response_combinations_for_desnodes)]

                        # observation
                        events_for_desnodes = list(product(*[range(self.cg.nodes_dict[nodes]) for nodes in sorted(descendant_nodes)])) # CHANGE THE NAME 
                        observation_for_desnodes = [item for sublist in [list(repeat(events_for_desnodes[i],np.prod(self.RF.response_functions))) for i in range(len(events_for_desnodes))] for item in sublist]
                        observation_for_factualvalue = [xF_tup]*len(observation_for_desnodes)
                        observation_combinations_for_allnodes = [list(itertools.chain.from_iterable(i)) for i in zip(observation_for_factualvalue,observation_for_desnodes)]
                        #observed = [xf_obs[i]+ obs_des[i] for i in range(len(obs_des))]
                        observation_for_desnodes_dict = dict(zip(descendant_nodes,list(zip(*observation_for_desnodes))))



                        setofparents_for_obsnodes = [list(self.cg.dag.predecessors(node)) for i,node in enumerate(self.cg.nodes)]
                        parents_of_obsnodes = [[] for _ in range(len(setofparents_for_obsnodes[1:]))]
                        for i in range(len(setofparents_for_obsnodes[1:])):
                            for j, node in enumerate(sorted(setofparents_for_obsnodes[1:][i])):
                                tmp = xF_dict[node]
                                parents_of_obsnodes[i].append(tmp)

                        parents_of_obsnodes = list(repeat([()] + [tuple(l) for l in parents_of_obsnodes], 
                                            np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes])*np.prod(self.RF.response_functions)))
                        setofparents_for_desnodes = [sorted(list(self.cg.dag.predecessors(node))) for i,node in enumerate(descendant_nodes)]
                        setofparents_for_allnodes = [item for sublist in setofparents_for_desnodes for item in sublist]
                        tmp_parents_of_desnodes = [[] for _ in range(len(setofparents_for_allnodes))]
                        len_parentsofdescendants = [len(list(self.cg.dag.predecessors(node))) for node in descendant_nodes]
                        for j, node in enumerate(setofparents_for_allnodes):
                            if node in intervened_node:
                                tmp_parents_of_desnodes[j].append(list(repeat(list(theta)[0], len(parents_of_obsnodes))))
                            elif node in nd_set_exclude:
                                tmp_parents_of_desnodes[j].append(list(repeat(xF_dict[node], len(parents_of_obsnodes))))
                            else:
                                tmp_parents_of_desnodes[j].append(observation_for_desnodes_dict[node])
                        tmp_parents_of_desnodes = [item for sublist in tmp_parents_of_desnodes for item in sublist]
                        tmp_parents_of_desnodes = list(zip(*tmp_parents_of_desnodes))

                        parents_of_desnodes = []
                        for i in range(len(tmp_parents_of_desnodes)):
                            it = iter(tmp_parents_of_desnodes[i])
                            parents_of_desnodes.append([tuple(next(it) for _ in range(size)) for size in len_parentsofdescendants])
                        parents_of_allnodes = [parents_of_obsnodes[i]+parents_of_desnodes[i] for i in range(len(parents_of_obsnodes))]

                        combinations_for_allnodes = [[] for _ in range(np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes])*np.prod(self.RF.response_functions))]
                        for i in range(0,len(combinations_for_allnodes)):
                            for j, node in enumerate(all_nodes):
                                tmp = (node, parents_of_allnodes[i][j],) + (response_combinations_for_allnodes[i][j],) + (observation_combinations_for_allnodes[i][j],)
                                combinations_for_allnodes[i].append(tmp)

                        # Indicator function value for factual observation and descendant variables
                        Q_for_allnodes = np.zeros((np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes])*np.prod(self.RF.response_functions),len(all_nodes)),dtype=int)
                        for i in range(len(Q_for_allnodes)):
                            for j, (a,b,c,d) in enumerate(combinations_for_allnodes[i]):
                                Q_for_allnodes[i,j] = self.RF.Indicator(a,b,c,d)

                        Q_for_allnodes = np.prod(Q_for_allnodes, axis = 1).reshape(np.prod([self.cg.nodes_dict.get(node) for node in descendant_nodes]),np.prod(self.RF.response_functions)).T
                        h_nd = [tuple(l) for l in list(repeat([(xF_lst[self.cg.observed_dict[n]-1]) for n in nd_set_exclude],len(events_for_desnodes)))]
                        h_des = events_for_desnodes
                        h_int = list(repeat(theta,len(events_for_desnodes)))              
                        h_indexes = [h_nd[i] + h_int[i] + h_des[i] for i in range(len(events_for_desnodes))]
                        h_values = [h_dict.get(i) for i in h_indexes]
                        if confounding_type =='linear':
                            p = self.cg.distribution()[0].reshape(np.prod(self.cg.states),1) 
                            names = list(product(*[range(self.RF.numRF_dict.get(node)) for node in self.cg.nodes]))
                            q = cp.Variable(shape=(np.prod(self.RF.response_functions),1))
                            Constraints = [q >= 10e-10, sum(q) == 1, p == self.Q_matrix() @ q]
                            Query = (q.T @ Q_for_allnodes ).T
                            Objective =  (h_values @ Query)*(1/self.cg.distribution()[1].get(xF_tup))
                            minimum = cp.Problem(cp.Minimize(Objective),Constraints)
                            maximum = cp.Problem(cp.Maximize(Objective),Constraints)
                            LB = minimum.solve(verbose=False)
                            UB = maximum.solve(verbose=False)
                            self.results['Lower_bound_linear'].append(LB)
                            self.results['Upper_bound_linear'].append(UB)
                            print(LB,intervened_node, factual, theta) 
                            #print(round(LB, 6),round(UB, 6), intervened_node, factual, theta) 
                            
                        else:
                            test_lb = []
                            # Define bilinear model
                            model = gp.Model('bilinear')
                            # Add latent nodes and edges
                            self.cg.dag.add_nodes_from(self.RF.rf_nodes)
                            self.cg.dag.add_edges_from(self.RF.rf_conf_edges)
                            self.cg.dag.add_edges_from(self.RF.rf_edges)

                            successors = []
                            s = []
                            NUM_variable = []
                            simplex = []
                            simplex_cond = []
                            free_parameters = []
                            for i, node in enumerate(self.RF.rf_nodes):
                                succ = list(self.cg.dag.successors(node))
                                succ.remove(self.RF.rf_observed_nodes_dict[node])
                                NUM  = int(self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[node]]*np.prod([self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[nodes]] 
                                                                            for nodes in list(self.cg.dag.predecessors(node))]))
                                NUM_variable.append(NUM)
                                s.append(model.addVars(NUM, lb=0, ub=1, name='a{}'.format(i)).select()); model.update()
                                N_dict = dict(zip(self.RF.rf_nodes,NUM_variable))
                                S_dict = dict(zip(self.RF.rf_nodes,s))
                                successors.append(succ)
                                if list(self.cg.dag.predecessors(node)) == []:
                                    simplex.append(model.addConstr(sum(S_dict[node])==1,'prob_constr_{}'.format(i+1))) ; model.update()
                                    free_parameters.append(len(S_dict[node])-1)
                                else:
                                    tmp = [S_dict[node][k:k+self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[node]]] for k in range(0, len(S_dict[node]),self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[node]])]
                                    for i in range(len(tmp)):
                                        simplex_cond.append(model.addConstr(sum(tmp[i])==1,'prob_constr_'+str(self.cg.observed_dict[self.RF.rf_observed_nodes_dict[node]])+str(i))) ; model.update()
                                        free_parameters.append(len(tmp[i])-1)
                            Successors_dict = dict(zip(self.RF.rf_nodes, successors)) 
                            if len(self.RF.rf_nodes) == 2:
                                b = self.reduce(self.RF.rf_nodes,{},0) 
                            else:
                                _,b =  self.reduce(self.RF.rf_nodes,{},0)
                            pairs = sorted(list(b.values()))
                            variables = sorted(list(b.keys()))
                            len_u = []
                            u = []
                            constraints = []
                            u_dict = {}
                            pairwise_rf = [np.prod([self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[pairs[:int(len(self.RF.rf_nodes)/2)][k][i]]] for i in range(2)]) for k in range(int(len(self.RF.rf_nodes)/2))]
                            u_rf_dict = dict(zip(variables[:3],pairwise_rf))
                            for i in range(int(len(self.RF.rf_nodes)/2)): # Total nodes/2
                                if (Successors_dict[pairs[i][0]] == [] and list(self.cg.dag.predecessors(pairs[i][1])) == []) or (Successors_dict[pairs[i][0]] != [] and list(self.cg.dag.predecessors(pairs[i][1])) == []):
                                    tmp = N_dict[pairs[i][0]]*N_dict[pairs[i][1]]
                                    len_u.append(tmp)
                                    u.append(model.addVars(tmp, lb=0, ub=1, name='u{}'.format(i)).select())  ; model.update()
                                    constraints.append(model.addConstrs(u[i][j]== [np.prod(list(product(S_dict[pairs[i][0]],S_dict[pairs[i][1]]))[k]) for k in range(tmp)][j] for j in range(tmp)).select())
                                elif list(self.cg.dag.predecessors(pairs[i][0])) !=[] and list(self.cg.dag.predecessors(pairs[i][1])) !=[]:
                                    print('8')
                                    tmp = 64*2 # Generalize it
                                    len_u.append(tmp)
                                    u.append(model.addVars(tmp, lb=0, ub=1, name='u{}'.format(i)).select())  ; model.update()
                                    constraints.append(model.addConstrs(u[i][j]== [np.prod(list(product(S_dict[pairs[i][0]],S_dict[pairs[i][1]]))[k]) for k in range(tmp)][j] for j in range(tmp)).select()) # Have to check
                                else:
                                    print('is')
                                    tmp = N_dict[pairs[i][1]]
                                    len_u.append(tmp)
                                    u.append(model.addVars(tmp, lb=0, ub=1, name='u{}'.format(i)).select())
                                    tmp_c = list(itertools.chain.from_iterable(itertools.repeat(x, self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[pairs[i][1]]]) for x in S_dict[pairs[i][0]]))*int(len_u[i]/pairwise_rf[i]) 
                                    constraints.append(model.addConstrs(u[i][j]== [a*b for a,b in zip(tmp_c,S_dict[pairs[i][1]])][j] for j in range(tmp)).select())
                            u_dict = dict(zip(variables[:int(len(self.RF.rf_nodes)/2)],u))
                            init_pairs = pairs.copy()
                            init_variables = variables.copy()
                            rem_pairs = init_pairs[int(len(self.RF.rf_nodes)/2):]
                            rem_variables = init_variables[int(len(self.RF.rf_nodes)/2):]
                            # for first pair
                            comb_firstpair = []
                            if Successors_dict[pairs[0][0]] == [] and list(self.cg.dag.predecessors(pairs[0][1])) == []: 
                                comb_firstpair.append(list(itertools.chain.from_iterable(itertools.repeat(x, int(np.prod(self.RF.response_functions)/len(u[0]))) for x in list(u[0]))))
                            else:
                                tmp = []
                                for k in range(0,len(u[0]),self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[pairs[0][1]]]):
                                    li = u_dict['u0'][k:k+self.RF.numRF_dict[self.RF.rf_observed_nodes_dict[pairs[0][1]]]]
                                    tmp.append((list(itertools.chain.from_iterable(itertools.repeat(x, int(np.prod(self.RF.response_functions)/len(u[0]))) for x in li))))
                                tmp = [item for sublist in tmp for item in sublist]
                                assert len(tmp) == np.prod(self.RF.response_functions)
                                comb_firstpair.append(tmp)

                                # For rest of the pairs
                                comb_rempair = [[] for _ in range(int(len(self.RF.rf_nodes)/2) - 1)]
                                for i,k in enumerate(list(u_dict.keys())[1:]):
                                    if (Successors_dict[pairs[i+1][0]] == [] and list(self.cg.dag.predecessors(pairs[i+1][1])) == []) and (Successors_dict[pairs[i+1][1]] == [] and list(self.cg.dag.predecessors(pairs[i+1][0])) == []): # All nodes are independent
                                        print('1')
                                        tmp = list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in u[i+1]))*int(np.prod(self.RF.response_functions)/len(u[i+1]))
                                        comb_rempair[i].append(tmp)
                                        assert len(tmp) == np.prod(self.RF.response_functions)
                                    elif Successors_dict[pairs[i+1][0]] != [] and set(Successors_dict[pairs[i+1][0]]) <= set(pairs[1]) and list(self.cg.dag.predecessors(pairs[i+1][0]))==[]: # Pairs are not dependent
                                        print('2')
                                        comb_rempair[i].append(list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in u[i+1]))*int(np.prod(self.RF.response_functions)/len(u[i+1])))
                                        assert len(tmp) == np.prod(self.RF.response_functions)
                                    elif list(self.cg.dag.predecessors(pairs[i+1][0])) !=[] and list(self.cg.dag.predecessors(pairs[i+1][1])) ==[]:  # Inter pair dependencies
                                        print('3')
                                        tmp = []
                                        for m in range(0,len(u_dict[k]),u_rf_dict[k]):
                                            index = u_rf_dict[k]
                                            li = u_dict[k][m:m+index]
                                            #tmp.append(list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in li))*int(len(u_dict[k])/u_rf_dict[k]))
                                            tmp.append(list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in li))*int(np.prod(self.response_functions)/len(u_dict['u1'])))
                                        tmp = [item for sublist in tmp for item in sublist]
                                        assert len(tmp) == np.prod(self.response_functions)
                                        comb_rempair[i].append(tmp)
                                    elif list(self.cg.dag.predecessors(pairs[i+1][0])) !=[] and list(self.cg.dag.predecessors(pairs[i+1][1])) !=[]:  # Inter pair dependencies as well as in pair dependencies  
                                        print('4')
                                        tmp = []
                                        for m in range(0,len(u_dict[k]),u_rf_dict[k]):
                                            index = u_rf_dict[k]
                                            li = u_dict[k][m:m+index]
                                            tmp.append(list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in li))*int(len(u_dict[k])/u_rf_dict[k]))
                                        tmp = [item for sublist in tmp for item in sublist]
                                        assert len(tmp) == np.prod(self.RF.response_functions)
                                        comb_rempair[i].append(tmp)
                                comb_rempair = [item for sublist in comb_rempair for item in sublist]

                                comb = [comb_firstpair,comb_rempair]
                                comb = [item for sublist in comb for item in sublist]
                            
                                self.cg.dag.remove_nodes_from(self.RF.rf_nodes)
                                self.cg.dag.remove_edges_from(self.RF.rf_conf_edges)
                                self.cg.dag.remove_edges_from(self.RF.rf_edges)
                                model.update()

                                if len(self.RF.rf_nodes) ==2:
                                    qs = comb[0]
                                elif len(self.RF.rf_nodes) % 2 != 0:
                                    last_node_comb = list(itertools.chain.from_iterable(itertools.repeat(x, 1) for x in S_dict[self.RF.rf_nodes[-1]]))*int(np.prod(self.RF.response_functions)/len(S_dict[self.RF.rf_nodes[-1]]))
                                    qs = [a*b for a,b in zip(comb[-1], last_node_comb)]
                                else:
                                    qs = [a*b for a,b in zip(comb[-2], comb[-1])]
                                
                                prob = self.cg.distribution()[0]
                                constraints.append(model.addConstrs(prob[i] - (self.Q_matrix() @ qs)[i] == 0 for i in range(len(prob))).select())
                                Query = qs @ Q_for_allnodes
                                Objective =  (h_values @ Query)*(1/self.cg.distribution()[1].get(xF_tup))
                                softlimit = 2
                                hardlimit = 60 # TO CHANGE
                                model.params.NonConvex = 2
                                model.setObjective(Objective, GRB.MAXIMIZE)
                                def softtime(model, where):
                                    if where == GRB.Callback.MIP:
                                        runtime = model.cbGet(GRB.Callback.RUNTIME)
                                        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                                        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                                        gap = abs((objbst - objbnd) / objbst)

                                        if runtime > softlimit and gap < 0.5:
                                            model.terminate()
                                model.Params.LogToConsole = 0
                                model.setParam('TimeLimit', hardlimit)
                                model.optimize(softtime)
                                
                                
                                print('Upper Bound: %g' % model.objVal) 
            
                                model.setObjective(Objective, GRB.MINIMIZE)
                                #def softtime(model, where):
                                    #if where == GRB.Callback.MIP:
                                        #runtime = model.cbGet(GRB.Callback.RUNTIME)
                                        #objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                                        #objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                                        #gap = abs((objbst - objbnd) / objbst)

                                        #if runtime > hardlimit and gap < 0.1:
                                            #model.terminate()
                                #model.Params.LogToConsole = 0
                                #model.setParam('TimeLimit', hardlimit)
                                model.optimize()
                                
                                print('Lower Bound: %g' % model.objVal) 

