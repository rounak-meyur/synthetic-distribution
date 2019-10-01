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
from pyBuildNetworklib import Spider,Steiner

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

#%% Create network connection spider networks
tsfrs = q_object.get_tsfr_to_link()
df_hmap = pd.read_csv(csvPath+'home2tsfr.csv')
H2Tsfr = dict([(t.HID, t.TID) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,tsfrs,roads,H2Tsfr)

tsfr_list = list(tsfrs.cord.keys())
tsf = tsfr_list[np.random.choice(range(len(tsfr_list)))]
link = [tsfrs.link[tsf]]

fig = plt.figure(figsize=(24,12))
ax1 = fig.add_subplot(121)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=link,ax=ax1,
                       edge_color='k',width=0.5)
S = spider_obj.generate_spider(tsf)
node_pos = nx.get_node_attributes(S,'cord')
nx.draw_networkx(S,pos=node_pos,ax=ax1,edge_color='r',with_labels=False,
                 node_size=10,node_color='r')


#%% Create network connection Steiner connections
df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])

steiner_obj = Steiner(homes,roads,H2Link)
L2Home = steiner_obj.link_to_home


ax2 = fig.add_subplot(122)
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=link,ax=ax2,
                       edge_color='k',width=0.5)

sample_link = link[0]
sample_homes = L2Home[sample_link]\
        if sample_link in L2Home else L2Home[(sample_link[1],sample_link[0])]

ga = steiner_obj.create_dummy_graph(sample_link,group=0,penalty=0.5)
gb = steiner_obj.create_dummy_graph(sample_link,group=1,penalty=0.5)

node_posa = nx.get_node_attributes(ga,'cord')
nx.draw_networkx(ga,pos=node_posa,ax=ax2,edge_color='r',with_labels=False,
             node_size=10,node_color='r')
node_posb = nx.get_node_attributes(gb,'cord')
nx.draw_networkx(gb,pos=node_posb,ax=ax2,edge_color='r',with_labels=False,
             node_size=10,node_color='r')

plt.show()
























