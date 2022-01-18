

import numpy as np
import matplotlib.pyplot as plt

def Indicator(x,y):
    if x == y :
        return x 
    else:
        return max(x,y) - x 

def synthetic_data(edges, states, n, N):
    NUM_NODES = n
    NUM_SAMPLES = N
    name_nodes= ['X_{}'.format(i+1) for i in range(NUM_NODES)] 
    parents = [[] for _ in range(NUM_NODES)]
    for j, node in enumerate(name_nodes):
      for i in range(len(edges)):
        if edges[i][1] == node:
          parents[j].append(edges[i][0])

    parents = dict(zip(name_nodes, parents))
    data = {}
    for node in name_nodes:
      if parents[node]== []:
        data[node] = np.random.choice(states[node],NUM_SAMPLES)
      else:
        if len(parents[node]) >=2:
          tmp = [data[parents[node][k]] for k in range(len(parents[node]))]
          print(tmp)
          x = np.random.choice(3,NUM_SAMPLES) 
          data[node] = np.array([Indicator(x[i],tmp[i]) for i in range(N)]) 
        else:
          print(node,states[node])
          x = np.random.choice(states[node],NUM_SAMPLES)
          data[node] = np.array([Indicator(x[i],data[parents[node][0]][i]) for i in range(N)]) 
    return data

#edges = [("X_1","X_2"), ("X_2","X_3"),("X_3","X_4")]
#states = {"X_1":3, "X_2":3,"X_3":3,"X_4":3}
#n = 4
#N = 50

#data = synthetic_data(edges, states, n, N)
#data

