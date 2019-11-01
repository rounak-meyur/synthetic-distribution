# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes.
This program creates the spider network using heuristic based as well as power
flow based optimization setup and compares them to better understand the 
differences.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider

#from pyMapElementslib import MapLink
#MapLink(roads).map_points(homes,path=csvPath,name='home')

#%%
def get_neighbors(graph,u,v,hops=2):
    """
    """
    nlist = [u,v]
    for i in range(hops):
        temp = []
        for n in nlist:
            temp.extend(list(graph.neighbors(n)))
        nlist = list(set(temp))
    return nlist


#%% Initialization of data sets and mappings
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()

df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home

#%% Check for a random link
import random
links = [l for l in L2Home if 20<=len(L2Home[l])<=45]
link = random.choice(links)
#link = (171514360, 979565325)
link = (171524810, 918459968)
homelist = L2Home[link]

#%% Plot the Delaunay Triangulation
import numpy as np
from scipy.spatial import Delaunay
H,T = spider_obj.get_nodes(link,minsep=50)
points = np.array([[homes.cord[h][0],homes.cord[h][1]] for h in H])
tri = Delaunay(points)
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax,
                       width=2.5,edge_color='k')
ax.scatter([homes.cord[h][0] for h in H],[homes.cord[h][1] for h in H],c='r',
           s=25.0,marker='^')
ax.scatter([t[0] for t in list(T.values())],[t[1] for t in list(T.values())],
            c='b',s=35.0,marker='s')
ax.triplot(points[:,0], points[:,1], tri.simplices.copy())
ax.set_xlabel("Longitude",fontsize=15)
ax.set_ylabel("Latitude",fontsize=15)
ax.set_title("Residences mapped to a link and probable locations of transformers along the link",fontsize=15)
#sys.exit(0)
#%% Create secondary distribution network as a forest of disconnected trees
#forest = spider_obj.generate_optimalpf_topology(link,minsep=50)
#pos_nodes = nx.get_node_attributes(forest,'cords')
#fig1 = plt.figure(figsize=(15,8))
#ax1 = fig1.add_subplot(111)
#nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax1,
#                       edge_color='k',width=2.5)
#nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
#                 ax=ax1,edge_color='r',width=1.0,with_labels=False,
#                 node_size=25.0)
#
#ax1.set_xlabel("Longitude",fontsize=15)
#ax1.set_ylabel("Latitude",fontsize=15)
#ax1.set_title("Secondary distribution network generated for the link with power flow constraints",fontsize=15)
#
#sys.exit(0)
#%% Create secondary distribution network as a forest of disconnected trees
forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,k=2,hops=5)
pos_nodes = nx.get_node_attributes(forest,'cords')
fig2 = plt.figure(figsize=(15,8))
ax2 = fig2.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax2,
                       edge_color='k',width=2.5)
nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                 ax=ax2,edge_color='r',width=1.5,with_labels=False,
                 node_size=25.0)
ax2.set_xlabel("Longitude",fontsize=15)
ax2.set_ylabel("Latitude",fontsize=15)
ax2.set_title("Secondary distribution network generated for the link with heuristics"+\
              "(maximum degree of 2, maximum leg length of 5)",fontsize=15)

plt.show()
