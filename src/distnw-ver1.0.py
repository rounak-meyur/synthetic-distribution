# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes.

This program creates the spider network using heuristic based as well as power
flow based optimization setup.
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


#%% Initialization of data sets and mappings
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads(level=[1,2,3,4,5])


df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home



#%% Check for a random link
links = [l for l in L2Home if len(L2Home[l])==20]
# sys.exit(0)
link = links[16]
# link = (171535026, 171535137)
# link = (171514360, 979565325)
# link = (171528302, 171526182)

# Plot the base network / Delaunay Triangulation

H,T = spider_obj.get_nodes(link,minsep=180)
points = np.array([[homes.cord[h][0],homes.cord[h][1]] for h in H])

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax,
                        width=0.8,edge_color='k',style='dashed')
ax.scatter([homes.cord[h][0] for h in H],[homes.cord[h][1] for h in H],c='red',
            s=80.0,marker='*')
ax.scatter([t[0] for t in list(T.values())],[t[1] for t in list(T.values())],
            c='green',s=120.0,marker='*')

# tri = Delaunay(points)
# ax.triplot(points[:,0], points[:,1], tri.simplices.copy(),color='b',linewidth=1.5,linestyle='dashed')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# ax.set_xlabel("Longitude",fontsize=25)
# ax.set_ylabel("Latitude",fontsize=25)

# ax.set_title("Residences mapped to a road link",fontsize=20)
ax.set_title("Probable transformers along road link",fontsize=20)
# ax.set_title("Possible edges obtained from Delaunay triangulation",fontsize=20)
leglines = [Line2D([0], [0], color='black', markerfacecolor='c', marker='*',markersize=0,linestyle='dashed'),
            Line2D([0], [0], color='white', markerfacecolor='green', marker='*',markersize=15),
            Line2D([0], [0], color='white', markerfacecolor='red', marker='*',markersize=15),
            # Line2D([0], [0], color='b', markerfacecolor='b', marker='*',markersize=0,linestyle='dashed')
            ]
ax.legend(leglines,['road link',
                    'possible transformers',
                    'residential consumers',
                    # 'possible edges'
                    ],
          loc='best',ncol=1,prop={'size': 15})
ax.set_xlim(left=-80.485,right=-80.467)
# fig.savefig("{}{}.png".format(figPath,'secnet_org'),bbox_inches='tight')
fig.savefig("{}{}.png".format(figPath,'secnet_base'),bbox_inches='tight')
# fig.savefig("{}{}.png".format(figPath,'secnet_delaunay'),bbox_inches='tight')
sys.exit(0)
#%% Create secondary distribution network as a forest of disconnected trees
forest,roots = spider_obj.generate_optimal_topology(link,minsep=180)
pos_nodes = nx.get_node_attributes(forest,'cord')

# Display the secondary network
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax2,
                       edge_color='k',width=1.0,style='dashed')
nodelist = list(forest.nodes())
colors = ['red' if n not in roots else 'green' for n in nodelist]
# shapes = ['*' if n not in roots else 's' for n in nodelist]
nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                 ax=ax2,edge_color='crimson',width=1,with_labels=False,
                 node_size=80.0,node_shape='*',node_color=colors)

ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# ax2.set_xlabel("Longitude",fontsize=15)
# ax2.set_ylabel("Latitude",fontsize=15)
ax2.set_title("Secondary network creation for road link",fontsize=20)


leglines = [Line2D([0], [0], color='black', markerfacecolor='green', marker='*',markersize=0,linestyle='dashed'),
            Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='*',markersize=0),
            Line2D([0], [0], color='white', markerfacecolor='green', marker='*',markersize=15),
            Line2D([0], [0], color='white', markerfacecolor='red', marker='*',markersize=15)]
ax2.legend(leglines,['road link','secondary network','local transformers','residences'],
          loc='best',ncol=2,prop={'size': 15})
ax2.set_xlim(left=-80.485,right=-80.467)

fig2.savefig("{}{}.png".format(figPath,'secnet-out'),bbox_inches='tight')

#%% Compare voltages at different nodes when heuristic choices are varied
dict_vol = {h:[] for h in H}
for hop in range(4,12):
    forest,tsfr = spider_obj.generate_optimal_topology(link,minsep=50,hops=hop)
    volts = spider_obj.checkpf(forest,tsfr)
    for h in H: dict_vol[h].append(volts[h])

# Plot variation in voltages at nodes
data = np.array(list(dict_vol.values()))
homeID = [str(h) for h in list(dict_vol.keys())]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.plot(data,'o-')
ax.set_xticks(range(len(H)))
ax.set_xticklabels(homeID)
ax.tick_params(axis='x',rotation=90)
ax.set_xlabel("Residential Building IDs",fontsize=15)
ax.set_ylabel("Voltage level in pu",fontsize=15)
ax.legend(labels=['max depth='+str(i) for i in range(4,12)])
ax.set_title("Voltage profile at residential nodes in the generated forest",
             fontsize=15)

print("DONE")