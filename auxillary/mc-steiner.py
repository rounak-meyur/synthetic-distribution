# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:07:54 2021

@author: rouna
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


edge_cost = {(1,2):0.01, (2,3):0.02, (3,4):0.3, (3,5):0.4, (2,7):0.05, (3,6):0.06,
             (1,7):1.0, (2,6):0.4, (4,5):0.05, (5,6): 1.0}
nodepos = {1:(0,0), 2:(1,0), 3:(2,0), 4:(3,1), 5:(3,-1), 6:(2,-1),7:(1,-1)}

graph = nx.Graph()
graph.add_edges_from(edge_cost.keys())
nx.set_edge_attributes(graph,edge_cost,'cost')
nx.set_node_attributes(graph,nodepos,'cord')

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
nx.draw_networkx(graph,pos=nodepos,ax=ax1,with_labels=False)

def random_successor(G,n):
    neighbors = list(nx.neighbors(G,n))
    cost = sum([np.exp(-G[n][i]['cost']) for i in neighbors])
    prob = [np.exp(-G[n][i]['cost'])/cost for i in neighbors]
    return np.random.choice(neighbors,p=prob)


def compute_hops(u,link,hop):
    if u not in hop:
        return compute_hops(link[u],link,hop)+1
    else:
        return hop[u]


def random_spanning_tree(graph,root,dmax=3):
    link = {}
    in_tree = []
    in_tree.append(root)
    nodelist = [n for n in graph if n!=root]
    nodes = []
    hop = {root:0}
    
    while len(nodelist)!=0:
        u = nodelist[0]
        while u not in in_tree:
            link[u] = random_successor(graph,u)
            u = link[u]
        u = nodelist[0]
        while u not in in_tree:
            h = compute_hops(u,link,hop)
            if h<=dmax:
                hop[u] = h
                in_tree.append(u)
                u = link[u]
                flag = 1
            else:
                flag = 0
                break
        if flag == 1: nodes.append(nodelist.pop(0))
    edgelist = [(link[n],n) for n in nodes]
    G = nx.Graph()
    G.add_edges_from(edgelist)
    return G


tree = random_spanning_tree(graph,1)
print(tree.edges())
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
nx.draw_networkx(graph,pos=nodepos,ax=ax1,with_labels=False)
nx.draw_networkx(tree,pos=nodepos,ax=ax2,with_labels=False)
























