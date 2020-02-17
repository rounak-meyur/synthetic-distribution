# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network. The first step is to identify Voronoi cells based on
network distance. This partitions the large connected graph into a number of small
components which can be solved separately.

This program displays th
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Initialize_Primary as init
from pyBuildNetworklib import InvertMap as imap


#%% Get transformers and store them in csv
q_object = Query(csvPath)
roads = q_object.GetRoads()
subs = q_object.GetSubstations()
tsfr = q_object.GetTransformers()

df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
L2Home = imap(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])<=70]


#%% Primary Network Generation
color_code = ['black','lightcoral','red','chocolate','darkorange','goldenrod',
              'olive','chartreuse','palegreen','seagreen','springgreen',
              'darkslategray','darkturquoise','deepskyblue','dodgerblue',
              'cornflowerblue','midnightblue','blue','mediumslateblue',
              'darkviolet','violet','magenta','deeppink','crimson','lightpink']

G,S2Node = init(subs,roads,tsfr,links)
nodepos = nx.get_node_attributes(G,'cord')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
for i,s in enumerate(list(S2Node.keys())):
    ax.scatter(subs.cord[s][0],subs.cord[s][1],s=100.0,c=color_code[i],label=str(i+1),
               marker='D')
    xpts = [nodepos[r][0] for r in S2Node[s]]
    ypts = [nodepos[r][1] for r in S2Node[s]]
    ax.scatter(xpts,ypts,s=1.0,c=color_code[i])
ax.legend(loc='best',ncol=5,prop={'size': 20})
ax.set_xlabel('Longitude',fontsize=20.0)
ax.set_ylabel('Latitude',fontsize=20.0)
ax.set_title('Voronoi partitioning of nodes based on shortest-path distance metric',
             fontsize=20.0)


#%% Inset figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 4.0, loc=4)
i = 21
s = list(S2Node.keys())[i]
axins.scatter(subs.cord[s][0],subs.cord[s][1],s=100.0,c=color_code[i],label=str(i+1),
              marker='D')
xpts = [nodepos[r][0] for r in S2Node[s]]
ypts = [nodepos[r][1] for r in S2Node[s]]
axins.scatter(xpts,ypts,s=1.0,c=color_code[i])

axins.set_xlim(min(xpts),max(xpts))
axins.set_ylim(min(ypts),max(ypts))
axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
