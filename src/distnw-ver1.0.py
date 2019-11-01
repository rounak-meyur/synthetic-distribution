# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes.This program generates 
spider without considering power flow in the network.
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
links = [l for l in L2Home if 50<=len(L2Home[l])<=100]
link = random.choice(links)
#link = (171514360, 979565325)
#link = (171524810, 918459968)
homelist = L2Home[link]
forest,roots = spider_obj.generate_optimal_topology(link,sep=40,k=2,hops=5)

#%% Create secondary distribution network as a forest of disconnected trees
pos_nodes = nx.get_node_attributes(forest,'cords')
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax1,
                       edge_color='k',width=2.0)
nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                 ax=ax1,edge_color='r',width=1.0,with_labels=False,
                 node_size=5.0)
plt.show()


#%% Check ratings of transformers
#output = steiner_obj.road_to_home
#ratings = {k:output[k] for k in output if output[k]!=0.0 and output[k]<=21e3}
#ratings = [r/1000.0 for r in list(ratings.values())]
#total_load = sum(list(homes.average.values()))
#plt.show()
#
#
#f=plt.figure(figsize=(10,6))
#ax=f.add_subplot(111)
#ax.hist(ratings,color='seagreen',ec='black')
#ax.set_xlabel('Rating of transformer (kVA)',fontsize=15)
#ax.set_ylabel('Number of distribution transformers',fontsize=15)
#ax.set_title('Histogram of distribution transformer ratings',fontsize=15)
#ax.tick_params(axis='x',labelsize=15)
#ax.tick_params(axis='y',labelsize=15)