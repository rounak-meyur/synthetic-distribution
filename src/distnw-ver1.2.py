# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes. The program creates the
network using design heuristics and checks power flow to identify voltage limit
violations. The voltage profile is plotted for different heuristic choices.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


#%% Create secondary distribution network as a forest of disconnected trees
dict_vol = {h:[] for h in homelist}
for hop in range(4,12):
    forest,tsfr = spider_obj.generate_optimal_topology(link,minsep=50,k=2,hops=hop)
    volts = spider_obj.checkpf(forest,tsfr)
    for h in homelist: dict_vol[h].append(volts[h])


#%%
data = np.array(list(dict_vol.values()))
homeID = [str(h) for h in list(dict_vol.keys())]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.plot(data,'o-')
ax.set_xticks(range(len(homelist)))
ax.set_xticklabels(homeID)
ax.tick_params(axis='x',rotation=90)
ax.set_xlabel("Residential Building IDs",fontsize=15)
ax.set_ylabel("Voltage level in pu",fontsize=15)
ax.legend(labels=['max depth='+str(i) for i in range(4,12)])
ax.set_title("Voltage profile at residential nodes in the generated forest",
             fontsize=15)

print("DONE")