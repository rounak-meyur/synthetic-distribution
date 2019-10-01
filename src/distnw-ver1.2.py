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
from pyBuildNetworklib import Spider

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
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()

#%% Create the mapping between homes and transformers/links
#from pyMapElementslib import Cluster
#cluster_obj = Cluster(homes,roads)
#cluster_obj.get_tsfr(path=csvPath)
#print("Clustering done in",cluster_obj.cluster[3],"iterations.")
#f = cluster_obj.plot_clusters(path=figPath)

#%% Create network connection spider networks
tsfrs = q_object.get_tsfr_to_link()
df_hmap = pd.read_csv(csvPath+'home2tsfr.csv')
H2Tsfr = dict([(t.HID, t.TID) for t in df_hmap.itertuples()])

edges = list(tsfrs.link.values())

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=edges,ax=ax,
                       edge_color='k',width=0.5)


spider_obj = Spider(homes,tsfrs,roads,H2Tsfr)
for tsf in list(tsfrs.cord.keys()):
    S = spider_obj.generate_spider(tsf)
    node_pos = nx.get_node_attributes(S,'cord')
    nx.draw_networkx(S,pos=node_pos,ax=ax,edge_color='r',with_labels=False,
                 node_size=1,node_color='r')

plt.show()
#%%
end = time.time()
print ("Time taken:",end-start)
sys.exit(0)