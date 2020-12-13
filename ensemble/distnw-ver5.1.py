# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program tries to generate ensemble of synthetic networks by varying
parameters in the optimization problem.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import random
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider

# from pyMapElementslib import MapLink
# MapLink(roads).map_point(homes,path=csvPath,name='home')
# sys.exit(0)
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
roads = q_object.GetRoads(level=[1,2,3,4,5])


df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home



#%% Check for a random link
links = [l for l in L2Home if 20<len(L2Home[l])<=25]
# sys.exit(0)
# link = random.choice(links)
# link = (171535026, 171535137)
# link = (171514360, 979565325)
link = (171528302, 171526182)

#%% Plot the base network / Delaunay Triangulation

H,T = spider_obj.get_nodes(link,minsep=50)
points = np.array([[homes.cord[h][0],homes.cord[h][1]] for h in H])
tri = Delaunay(points)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax,
                        width=2.5,edge_color='k')
ax.scatter([homes.cord[h][0] for h in H],[homes.cord[h][1] for h in H],c='r',
            s=25.0,marker='*')
ax.scatter([t[0] for t in list(T.values())],[t[1] for t in list(T.values())],
            c='b',s=50.0,marker='*')
# ax.triplot(points[:,0], points[:,1], tri.simplices.copy())
ax.set_xlabel("Longitude",fontsize=15)
ax.set_ylabel("Latitude",fontsize=15)
ax.set_title("Residences to be connected by network",fontsize=20)
leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='*',markersize=10),
            Line2D([0], [0], color='white', markerfacecolor='blue', marker='*',markersize=10),
            Line2D([0], [0], color='white', markerfacecolor='red', marker='*',markersize=10)]
ax.legend(leglines,['road link','probable local transformers',
                    'individual residential consumers'],
          loc='best',ncol=2,prop={'size': 13})
ax.autoscale(tight=True)
fig.savefig("{}{}.png".format(figPath,'secnet'))

#%% Create secondary distribution network as a forest of disconnected trees
for i,c in enumerate([0.1,1,10,100]):
    forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,penalty=c)
    pos_nodes = nx.get_node_attributes(forest,'cord')
    
    # Display the secondary network
    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.add_subplot(111)
    nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax2,
                           edge_color='k',width=1.5)
    nodelist = list(forest.nodes())
    colors = ['red' if n not in roots else 'blue' for n in nodelist]
    # shapes = ['*' if n not in roots else 's' for n in nodelist]
    nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                     ax=ax2,edge_color='crimson',width=1,with_labels=False,
                     node_size=20.0,node_shape='*',node_color=colors)
    
    ax2.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
    ax2.set_xlabel("Longitude",fontsize=15)
    ax2.set_ylabel("Latitude",fontsize=15)
    ax2.set_title("penalty factor ="+str(c),fontsize=20)
    
    
    leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='*',markersize=10),
                Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='*',markersize=10),
                Line2D([0], [0], color='white', markerfacecolor='blue', marker='*',markersize=10),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',markersize=10)]
    ax2.legend(leglines,['road link','secondary network','local transformers','residences'],
              loc='best',ncol=2,prop={'size': 15})
    ax2.autoscale(tight=True)
    fig2.savefig("{}{}.png".format(figPath,'secnet-'+str(i+1)))
