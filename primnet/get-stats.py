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



# plot_network(ax,rural_sub)
# plot_network(ax,urban_sub)



#%% Statistical Properties
deg1 = []
deg2 = []
hops1 = []
hops2 = []
dist1 = []
dist2 = []


for i,sub in enumerate(rural_sub):
    net = GetDistNet(distpath,sub)
    hops1.extend([nx.shortest_path_length(net,n,sub) for n in net])
    dist1.extend([nx.shortest_path_length(net,n,sub,'length') for n in net])
    deg1.extend([nx.degree(net,n) for n in net])
    print(str(i+1)+" out of " + str(len(rural_sub)))



for i,sub in enumerate(urban_sub):
    net = GetDistNet(distpath,sub)
    hops2.extend([nx.shortest_path_length(net,n,sub) for n in net])
    dist2.extend([nx.shortest_path_length(net,n,sub,'length') for n in net])
    deg2.extend([nx.degree(net,n) for n in net])
    print(str(i+1)+" out of " + str(len(urban_sub)))



# Save the data

line1 = ' '.join([str(x) for x in deg1])
line2 = ' '.join([str(x) for x in deg2])
line3 = ' '.join([str(x) for x in hops1])
line4 = ' '.join([str(x) for x in hops2])
line5 = ' '.join([str(x) for x in dist1])
line6 = ' '.join([str(x) for x in dist2])

data = '\n'.join([line1,line2,line3,line4,line5,line6])

with open(workpath + "/out/network-stats.txt",'w') as f:
    f.write(data)