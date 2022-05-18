# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import geopandas as gpd
import matplotlib.pyplot as plt


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
grbpath = workpath + "/out/gurobi/"
shappath = rootpath + "/output/optimal/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet

print("Imported modules")

#%% Functions to draw 3 phase network

def DrawNodes(graph,ax,nodelist,color='red',size=25,alpha=1.0,label=''):
    d = {'nodes':nodelist,
         'geometry':[graph.nodes[n]['geometry'] for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,label=label)
    return ax

def DrawEdges(graph,ax,phase='ABC',color='black',width=2.0,style='solid',
              alpha=1.0,label=''):
    """
    """
    edgelist = [e for e in graph.edges if graph.edges[e]['phase'] == phase]
    d = {'edges':edgelist,
         'geometry':[graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,linestyle=style,
                  alpha=alpha,label=label)
    return
    

#%% Get 3 phase network
sub = 121144
# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

new_dist = GetDistNet(distpath,sub)
t_nodes = [n for n in new_dist if new_dist.nodes[n]['label']=='T']
s_nodes = [n for n in new_dist if new_dist.nodes[n]['label']=='S']
res_A = [n for n in new_dist if new_dist.nodes[n]['phase']=='A']
res_B = [n for n in new_dist if new_dist.nodes[n]['phase']=='B']
res_C = [n for n in new_dist if new_dist.nodes[n]['phase']=='C']





#%% Plot the network
fig = plt.figure(figsize=(40,40), dpi=72)
ax = fig.add_subplot(111)

DrawNodes(new_dist,ax,res_A,color='red',size=10,label='Residence Phase A')
DrawNodes(new_dist,ax,res_B,color='blue',size=10,label='Residence Phase B')
DrawNodes(new_dist,ax,res_C,color='green',size=10,label='Residence Phase C')
DrawNodes(new_dist,ax,t_nodes,color='black',size=25,label='Transformer')
DrawNodes(new_dist,ax,s_nodes,color='dodgerblue',size=200,label='Substation')

DrawEdges(new_dist,ax,phase='ABC',color='black',width=3.0,label='3 phase')
DrawEdges(new_dist,ax,phase='A',color='red',width=2.0,label='Phase A')
DrawEdges(new_dist,ax,phase='B',color='blue',width=2.0,label='Phase B')
DrawEdges(new_dist,ax,phase='C',color='green',width=2.0,label='Phase C')

ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax.legend(loc='best',ncol=1,fontsize=30,markerscale=3)

















