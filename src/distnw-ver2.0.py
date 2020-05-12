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
from collections import namedtuple as nt


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
from pyBuildNetworklib import plot_graph
from pyBuildNetworklib import Primary

#%% Get transformers and store them in csv
q_object = Query(csvPath,inpPath)
homes,roads = q_object.GetDataset(fislist=[161,770,775])
subs = q_object.GetSubstations(fis=161)
tsfr = q_object.GetTransformers(fis=161)

fiscode = '%03.f'%(161)
df_hmap = pd.read_csv(csvPath+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])
L2Home = imap(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]


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
    xpts = [nodepos[r][0] for r in S2Node[s]]
    ypts = [nodepos[r][1] for r in S2Node[s]]
    ax.scatter(xpts,ypts,s=1.0,c=color_code[i])
    ax.scatter(subs.cord[s][0],subs.cord[s][1],s=500.0,c='green',
               marker='*')
    ax.scatter(subs.cord[s][0],subs.cord[s][1],s=100.0,c=color_code[i],
               label=str(i+1),marker='D')
ax.legend(loc='best',ncol=6,prop={'size': 10})
# ax.set_xlabel('Longitude',fontsize=20.0)
# ax.set_ylabel('Latitude',fontsize=20.0)
ax.set_title('Voronoi partitioning of nodes based on shortest-path distance metric',
             fontsize=20.0)
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

sys.exit(0)
#%% Inset figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(ax, 0.5, loc=4)
i = 0
s = list(S2Node.keys())[i]
axins.scatter(subs.cord[s][0],subs.cord[s][1],s=100.0,c=color_code[i],label=str(i+1),
              marker='D')
axins.set_aspect(1.2)
xpts = [nodepos[r][0] for r in S2Node[s]]
ypts = [nodepos[r][1] for r in S2Node[s]]
axins.scatter(xpts,ypts,s=1.0,c=color_code[i])

axins.set_xlim(min(xpts),max(xpts))
axins.set_ylim(min(ypts),max(ypts))
axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

# sys.exit(0)
#%% Plot the substations
def display_data(ax,roads,homes,subs,showhome=True,showsub=True):
    """
    Displays the road network and residences in the given region.
    Parameters
    ----------
    ax    : TYPE: matplotlib axis object
        DESCRIPTION.
    roads : TYPE: named tuple with road network data
        DESCRIPTION.
    homes : TYPE: named tuple with residential data
        DESCRIPTION.
    colors: TYPE: list, optional
        DESCRIPTION. The default is ['royalblue','seagreen']. 
        Color specifications for roads and residences

    Returns
    -------
    None.
    """
    nx.draw_networkx(roads.graph,node_size=0.1,color='blue',with_labels=False,
                     ax=ax,pos=roads.cord,edge_color='blue',style="dashed",width=0.5)
    if showhome:
        hpts = list(homes.cord.values())
        xpts = [h[0] for h in hpts]
        ypts = [h[1] for h in hpts]
        ax.scatter(xpts,ypts,c='lightgreen',s=0.1)
    
    if showsub:
        spts = list(subs.cord.values())
        xpts = [s[0] for s in spts]
        ypts = [s[1] for s in spts]
        ax.scatter(xpts,ypts,c='crimson',s=50)
    return ax

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
ax = display_data(ax,roads,homes,subs,showhome=False)


#%% Generate partitions in each cluster and plot it
sub = 146410
substation = nt("local_substation",field_names=["id","cord","nodes"])
sub_data = substation(id=sub,cord=subs.cord[sub],nodes=S2Node[sub])

P = Primary(sub_data,homes,G)
plot_graph(P.graph,subdata=sub_data,path=figPath,filename=str(sub)+'-master',
           rcol=color_code[1:nx.number_connected_components(P.graph)+1])