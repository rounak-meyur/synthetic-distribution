# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:20:45 2019

Author: Dr Anil Vullikanti
        Rounak Meyur
        
Description: Generates primary distribution network for Montgomery county
"""


import sys,os
import time
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Steiner
from pyMapElementslib import MapLink

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


#%% Main function goes here
    
start = time.time()
q_object = Query(csvPath)
subs = q_object.GetSubstations()
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()


#obj_MapLink = MapLink(roads)
#obj_MapLink.map_point(homes,path=csvPath,name='home')
df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])

steiner_obj = Steiner(homes,roads,H2Link)
L2Home = steiner_obj.link_to_home


#%% Check on a sample link
#links = [k for k in list(L2Home.keys()) if len(L2Home[k])>0]
#
#n1,n2 = links[np.random.choice(range(len(links)))]
#rnodes = get_neighbors(roads.graph,n1,n2)
#subgraph = nx.subgraph(roads.graph,rnodes)
#edges = [e for e in list(subgraph.edges()) \
#         if e in L2Home or (e[1],e[0]) in L2Home]

edges = [k for k in list(L2Home.keys()) if len(L2Home[k])>0]

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=edges,ax=ax,
                       edge_color='k',width=0.5)

for sample_link in edges:
    sample_homes = L2Home[sample_link] \
        if sample_link in L2Home else L2Home[(sample_link[1],sample_link[0])]

    ga = steiner_obj.create_dummy_graph(sample_link,group=0,penalty=0.5)
    gb = steiner_obj.create_dummy_graph(sample_link,group=1,penalty=0.5)

    node_posa = nx.get_node_attributes(ga,'cord')
    nx.draw_networkx(ga,pos=node_posa,ax=ax,edge_color='r',with_labels=False,
                 node_size=1,node_color='r')
    node_posb = nx.get_node_attributes(gb,'cord')
    nx.draw_networkx(gb,pos=node_posb,ax=ax,edge_color='b',with_labels=False,
                 node_size=1,node_color='b')

end = time.time()
print ("Time taken:",end-start)
sys.exit(0)

#%%
output = steiner_obj.road_to_home
ratings = {k:output[k] for k in output if output[k]!=0.0 and output[k]<=21e3}
ratings = [r/1000.0 for r in list(ratings.values())]
total_load = sum(list(homes.average.values()))
plt.show()


f=plt.figure(figsize=(10,6))
ax=f.add_subplot(111)
ax.hist(ratings,color='seagreen',ec='black')
ax.set_xlabel('Rating of transformer (kVA)',fontsize=15)
ax.set_ylabel('Number of distribution transformers',fontsize=15)
ax.set_title('Histogram of distribution transformer ratings',fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)








