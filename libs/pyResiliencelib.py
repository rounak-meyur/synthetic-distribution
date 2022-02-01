# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:04:41 2022

Author: Rounak Meyur

Description: Consists of methods and attributes to measure resilience of networks
to random events of failures
"""

import itertools
import networkx as nx


#%% Graph motifs
def count_motifs(g,target):
    count = 0
    for sub_nodes in itertools.combinations(g.nodes(),len(target.nodes())):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            count += 1
    return count