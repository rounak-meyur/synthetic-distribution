# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:04:41 2022

Author: Rounak Meyur

Description: Consists of methods and attributes to measure resilience of networks
to random events of failures
"""

from scipy.special import comb
import networkx as nx


#%% k-path motif and k-star motif
def path(tree,k):
    P = {}
    for v in tree:
        P[(v,1)] = [[u,v] for u in list(nx.neighbors(tree,v))]
    
    # Dynamic programing
    for j in range(2,k):
        for v in tree:
            all_path = []
            path = []
            for u in P[(v,j-1)]:
                all_path.extend([x+u[1:] for x in P[(u[0],1)]])
            for p in all_path:
                if len(list(set(p))) == len(p):
                    path.append(p)
            P[(v,j)] = path
    
    # Count the motifs
    motif = 0
    for v in tree:
        motif += len(P[(v,k-1)])
    return int(motif/2)

def star(tree,k):
    node_deg = {n:nx.degree(tree,n) for n in tree if nx.degree(tree,n)>=k-1}
    return sum([comb(node_deg[n],k-1) for n in node_deg])