# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak

Description: This program plots the motifs in distribution network of Virginia.
"""

import sys,os
import geopandas as gpd
import networkx as nx
from scipy.special import comb
from shapely.geometry import Point
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import itertools

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
distpath = workpath + "/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
print("Imported modules")


sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
k = 5

#%% k-path motif
def count_k_path(tree,k):
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

#%% k-path motif

data = ''
for sub in sublist:
    dist = GetDistNet(distpath,sub)
    motif = count_k_path(dist,k)
    num_nodes = dist.number_of_nodes()
    data += '\t'.join([str(sub), str(num_nodes), str(motif)])+'\n'


with open(workpath+"/out/"+str(k)+"path-motif.txt",'w') as f:
    f.write(data)


#%% k-star motif

data = ''
for sub in sublist:
    dist = GetDistNet(distpath,sub)
    node_deg = {n:nx.degree(dist,n) for n in dist if nx.degree(dist,n)>=k-1}
    count = sum([comb(node_deg[n],k-1) for n in node_deg])
    data += '\t'.join([str(sub), str(dist.number_of_nodes()), str(int(count))])+'\n'

with open(workpath+"/out/"+str(k)+"star-motif.txt",'w') as f:
    f.write(data)