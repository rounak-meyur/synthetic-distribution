# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program plots the distribution network of Virginia state.
"""

import sys,os
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyDrawNetworklib import DrawNodes, DrawEdges
print("Imported modules")




sublist = [int(x.strip("-prim-dist.gpickle")) for x in os.listdir(distpath)]


state = 'VA'
state_file="states.shp"
block_file = "tl_2018_51_tabblock10.shp"
data_states = gpd.read_file(inppath+"census/"+state_file)
data_blocks = gpd.read_file(inppath+"census/"+block_file)
state_polygon = list(data_states[data_states.STATE_ABBR == 
                         state].geometry.items())[0][1]


colors = ['blue' if data_blocks.iloc[i]['UR10']=='R' else 'orangered'\
          for i in range(len(data_blocks))]

data_blocks['color'] = colors

# sub_file="eia/Electric_Substations.shp"
# data_substations = gpd.read_file(inppath+sub_file)
# subs = data_substations.loc[data_substations.geometry.within(state_polygon)]

# sw_subs = subs.loc[subs.LONGITUDE<-82.5686]["ID"].values

#%% Plot the network

DPI = 72    
fig = plt.figure(figsize=(5000/DPI, 2500/DPI), dpi=DPI)
ax = fig.add_subplot(111)
for pol in state_polygon:
    ax.plot(*pol.exterior.xy)
data_blocks.plot(ax=ax,color=data_blocks["color"])


def plot_network(ax,sublist,with_secnet=False):
    for code in sublist:
        print(code)
        net = GetDistNet(distpath,code)
        # Draw nodes
        DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
        DrawNodes(net,ax,label='T',color='green',size=25)
        DrawNodes(net,ax,label='R',color='black',size=2.0)
        if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
        # Draw edges
        DrawEdges(net,ax,label='P',color='black',width=2.0)
        DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
        if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)


#%%
donelist = [f.strip("-prim-dist.gpickle") for f in os.listdir(distpath)]
with open(inppath+"urban-sublist.txt") as f:
    urban_sublist = f.readlines()[0].strip('\n').split(' ')
with open(inppath+"rural-sublist.txt") as f:
    rural_sublist = f.readlines()[0].strip('\n').split(' ')

rural_sub = [s for s in donelist if s in rural_sublist]
urban_sub = [s for s in donelist if s in urban_sublist]


# mont_sub = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

# sw_sub = [109418, 109426, 109430, 109487, 113962, 113968, 113969]


import numpy as np


plot_network(ax,rural_sub)
plot_network(ax,urban_sub)



hops1 = []
hops2 = []
dist1 = []
dist2 = []
for s in rural_sub:
    net = GetDistNet(distpath,s)
    nodes = [n for n in net if net.nodes[n]['label']!='H']
    hops1.extend([nx.shortest_path_length(net,n,int(s)) for n in net.nodes \
            if net.nodes[n]['label']!='H'])
    dist1.extend([nx.shortest_path_length(net,n,int(s),'geo_length') for n in net.nodes \
            if net.nodes[n]['label']!='H'])

w1 = np.ones_like(hops1)/float(len(hops1))

for s in urban_sub:
    net = GetDistNet(distpath,s)
    nodes = [n for n in net if net.nodes[n]['label']!='H']
    hops2.extend([nx.shortest_path_length(net,n,int(s)) for n in net.nodes \
            if net.nodes[n]['label']!='H'])
    dist2.extend([nx.shortest_path_length(net,n,int(s),'geo_length') for n in net.nodes \
            if net.nodes[n]['label']!='H'])

w2 = np.ones_like(hops2)/float(len(hops2))

hops = [hops1,hops2]
w = [w1,w2]

#%% Hop distribution
from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{0:.1f}".format(100*y)
    return s

colors = ['blue','orangered']
labels = ['Rural Areas','Urban Areas']
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.hist(hops,weights=w,label=labels,color=colors)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of nodes",fontsize=20)
ax.set_xlabel("Hops from root node",fontsize=20)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize=20)
fig.savefig("{}{}.png".format(figpath,'hop-comp'),bbox_inches='tight')

#%% Distance
dist = [[d/1e3 for d in dist1],[d/1.6e3 for d in dist2]]
w = [w1,w2]


from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{0:.1f}".format(100*y)
    return s

colors = ['blue','orangered']
labels = ['Rural Areas','Urban Areas']
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.hist(dist,weights=w,label=labels,color=colors)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of nodes",fontsize=20)
ax.set_xlabel("Distance (in miles) from root node",fontsize=20)
ax.legend(prop={'size': 20})
ax.tick_params(axis='both', labelsize=20)
fig.savefig("{}{}.png".format(figpath,'dist-comp'),bbox_inches='tight')