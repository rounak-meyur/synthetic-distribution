# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx


import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/results-dec12/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network
from pyValidationlib import create_base, degree_dist, hop_dist

#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()
G = create_base(csvPath,filename='hethwood-ver2')


# sys.exit(0)

#%% Hop and degree distributions
# for sub in [34780,34810,34816,28228,28235]:
#     dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)
#     nodelab = nx.get_node_attributes(dist_net,'label')
#     sec_nodes = [n for n in nodelab if nodelab[n]=='H']
#     for n in sec_nodes:
#         dist_net.remove_node(n)
        
#     bfs_edges = list(nx.dfs_edges(dist_net,source=sub))[:G.number_of_nodes()]
#     M = dist_net.edge_subgraph(bfs_edges).copy()
#     degree_dist(M,G,sub,figPath)
#     hop_dist(M,G,sub,figPath)

#%% Ensemble
sub = 24664
for sub in [28228,28235,34810,34816,34780]:
    fname = str(sub)+'-network'
    dist_net = read_network(tmpPath+fname+'.txt',homes)
    nodelab = nx.get_node_attributes(dist_net,'label')
    sec_nodes = [n for n in nodelab if nodelab[n]=='H']
    for n in sec_nodes:
        dist_net.remove_node(n)
        
    degree_dist(dist_net,G,sub,figPath)
    hop_dist(dist_net,G,sub,figPath)