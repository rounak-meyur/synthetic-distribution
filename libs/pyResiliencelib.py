# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:04:41 2022

Author: Rounak Meyur

Description: Consists of methods and attributes to measure resilience of networks
to random events of failures
"""

import sys
import itertools
import networkx as nx


#%% Graph motifs
def count_motif(g,t=1):
    if t == 1:
        target = nx.Graph()
        nx.add_path(target,['A','B','C','D'])
    elif t == 2:
        target = nx.Graph()
        nx.add_path(target,['A','B','C'])
        target.add_edge('B','D')
    elif type(t) == nx.Graph:
        edgelist = list(t.edges())
        target = nx.Graph()
        target.add_edges_from(edgelist)
    else:
        print("Target ID not Valid. Define the target")
        sys.exit(0)
    count = 0
    for sub_nodes in itertools.combinations(g.nodes(),len(target.nodes())):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            count += 1
    return count