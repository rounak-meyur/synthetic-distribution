# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import geopandas as gpd
from shapely.geometry import Point,LineString
import networkx as nx
import gurobipy as grb
import numpy as np
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
from pyGeometrylib import Link

print("Imported modules")

#%% Functions to label 3 phase network
def get_edges(t_node,nodes,phase):
    branch = {'A':[t_node],'B':[t_node],'C':[t_node]}
    for n in nodes:
        branch[phase[n]].append(n)
    g = nx.Graph()
    for ph in branch:
        nx.add_path(g,branch[ph])
    return list(g.edges)
    


def DrawNodes(graph,ax,nodelist,color='red',size=25,alpha=1.0,label=''):
    d = {'nodes':nodelist,
         'geometry':[Point(graph.nodes[n]['cord']) for n in nodelist]}
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
    

#%% Assign phases to edges
sub = 121144
# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

with open(workpath+'/out/phase.txt') as f:
    lines = f.readlines()

phase = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    phase[int(temp[0])] = temp[1]


dist = GetDistNet(distpath,sub)
sec_edges = [e for e in dist.edges if dist.edges[e]['label']=='S']
prm_edges = [e for e in dist.edges if dist.edges[e]['label']=='P']
hvf_edges = [e for e in dist.edges if dist.edges[e]['label']=='E']
h_nodes = [n for n in dist if dist.nodes[n]['label'] == 'H']
s_nodes = [n for n in dist if dist.nodes[n]['label'] == 'S']
t_nodes = [n for n in dist if dist.nodes[n]['label'] == 'T']
res_A = [n for n in h_nodes if phase[n] == 'A']
res_B = [n for n in h_nodes if phase[n] == 'B']
res_C = [n for n in h_nodes if phase[n] == 'C']


tree = nx.dfs_tree(dist,sub)
rem_edges = [e for e in tree.edges if dist.edges[e]['label']!='S']
tree.remove_edges_from(rem_edges)

# Create new 3 phase network
new_sec_edges = []
for t in t_nodes:
    t_child = list(tree.successors(t))
    dfs_nodes = list(nx.dfs_preorder_nodes(tree, source=t))
    c_index = [dfs_nodes.index(c) for c in t_child]
    
    for idx in range(len(t_child)):
        if idx == len(t_child) - 1:
            nodelist = dfs_nodes[c_index[idx]:]
        else:
            nodelist = dfs_nodes[c_index[idx]:c_index[idx+1]]
        
        new_sec_edges += get_edges(t,nodelist,phase)
        

#%% Reconstruct the secondary network for main synthetic network
new_dist = nx.Graph()
new_dist.add_edges_from(hvf_edges+prm_edges+new_sec_edges)
# node attributes
for n in new_dist:
    new_dist.nodes[n]['cord'] = dist.nodes[n]['cord']
    new_dist.nodes[n]['load'] = dist.nodes[n]['load']
    new_dist.nodes[n]['label'] = dist.nodes[n]['label']
    if new_dist.nodes[n]['label'] == 'H':
        new_dist.nodes[n]['phase'] = phase[n]
    else:
        new_dist.nodes[n]['phase'] = 'ABC'
# edge attributes
for e in new_dist.edges:
    if (e in hvf_edges+prm_edges) or ((e[1],e[0]) in hvf_edges+prm_edges):
        new_dist.edges[e]['geometry'] = dist.edges[e]['geometry']
        new_dist.edges[e]['length'] = dist.edges[e]['length']
        new_dist.edges[e]['label'] = dist.edges[e]['label']
        new_dist.edges[e]['phase'] = 'ABC'
    else:
        new_dist.edges[e]['geometry'] = LineString([dist.nodes[e[0]]['cord'],
                                                    dist.nodes[e[1]]['cord']])
        new_dist.edges[e]['length'] = Link(new_dist.edges[e]['geometry']).geod_length
        new_dist.edges[e]['label'] = 'S'
        new_dist.edges[e]['phase'] = [phase[n] for n in e if dist.nodes[n]['label']=='H'][0]




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
















