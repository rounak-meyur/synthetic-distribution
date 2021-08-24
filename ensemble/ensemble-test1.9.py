# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program creates a schematic of the ensemble network generation
process.
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.lines import Line2D
from pyqtree import Index

workpath = os.getcwd()
figpath = workpath + "/figs/"

def geodist(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates.
    pt1: shapely point geometry of point 1
    pt2: shapely point geometry of point 2
    '''
    lon1,lat1 = pt1.x,pt1.y
    lon2,lat2 = pt2.x,pt2.y
    return np.sqrt((lon1-lon2)**2+(lat1-lat2)**2)

def bounds(pt,radius):
    """
    Returns the bounds for a point geometry. The bound is a square around the
    point with side of 2*radius units.
    
    pt:
        TYPE: shapely point geometry
        DESCRIPTION: the point for which the bound is to be returned
    
    radius:
        TYPE: floating type 
        DESCRIPTION: radius for the bounding box
    """
    return (pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)

def find_nearest_node(center_cord,node_cord):
    """
    Computes the nearest node in the dictionary 'node_cord' to the point denoted
    by the 'center_cord'
    
    center_cord: 
        TYPE: list of two entries
        DESCRIPTION: geographical coordinates of the center denoted by a list
                     of two entries
    
    node_cord: 
        TYPE: dictionary 
        DESCRIPTION: dictionary of nodelist with values as the geographical 
                     coordinate
    """
    xmin,ymin = np.min(np.array(list(node_cord.values())),axis=0)
    xmax,ymax = np.max(np.array(list(node_cord.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    nodes = []
    for i,n in enumerate(list(node_cord.keys())):
        node_geom = Point(node_cord[n])
        node_bound = bounds(node_geom,0.0)
        idx.insert(i,node_bound)
        nodes.append((node_geom, node_bound, n))
    
    pt_center = Point(center_cord)
    center_bd = bounds(pt_center,2)
    matches = idx.intersect(center_bd)
    closest_node = min(matches,key=lambda i: geodist(nodes[i][0],pt_center))
    return nodes[closest_node][-1]


def get_additional_edge(graph,edge):
    new_graph = graph.__class__()
    new_graph.add_nodes_from(graph.nodes)
    new_graph.add_edges_from(graph.edges)
    
    new_graph.remove_edge(*edge)
    if nx.has_path(new_graph,0,edge[0]):
        end_node = edge[0]
        other_node = edge[1]
    else:
        end_node = edge[1]
        other_node = edge[0]
    
    comps = list(nx.connected_components(new_graph))
    connected_nodes = list(comps[0]) \
            if end_node in list(comps[0]) else list(comps[1])
    dict_node = {n:graph.nodes[n]['cord'] for n in connected_nodes \
                 if n!=end_node}
    center_node = graph.nodes[other_node]['cord']
    near_node = find_nearest_node(center_node,dict_node)
    new_edge = (near_node,other_node)
    return new_edge
    
    

G = nx.Graph()
edgelist = [(0,1),(0,2),(1,3),(1,4),(3,5),(2,6),(6,7),(6,8),(6,9),(7,10),(10,11)]
G.add_edges_from(edgelist)


cord = {0:[0,4],1:[-1,3],2:[1,3],3:[-2,2],4:[0,2],5:[-2,1],
        6:[1,2],7:[1,1],8:[2,1],9:[2,2],10:[-1,0],11:[-2,0]}

label = {0:'S',1:'R',2:'R',3:'R',4:'T',5:'T',6:'R',
         7:'T',8:'T',9:'T',10:'R',11:'T'}

nx.set_node_attributes(G,cord,'cord')
nx.set_node_attributes(G,label,'label')

edges = [(1,3),(1,4),(3,5),(2,6),(6,7),(6,8),(6,9),(7,10),(10,11)]
new_edges = [get_additional_edge(G, edge) for edge in edges]
add_edges = []
for e in new_edges:
    if (e[1],e[0]) not in add_edges and e not in add_edges:
        add_edges.append(e)

# Node colors
n_col = []
for n in G:
    if label[n]=='T':
        n_col.append('green')
    elif label[n]=='R':
        n_col.append('orange')
    else:
        n_col.append('dodgerblue')

leghands1 = [Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=15,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='orange', 
                   marker='o',markersize=15,label='road node'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=15,label='substation')]

leghands2 = [Line2D([0], [0], color='crimson', markerfacecolor='crimson',linestyle='dashed', 
                   marker='o',markersize=0,label='deleted edge'),
            Line2D([0], [0], color='green', markerfacecolor='green', 
                   linestyle='dashed',markersize=0,label='added edge'),
            Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=15,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='orange', 
                   marker='o',markersize=15,label='road node'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=15,label='substation')]

#%% Figure 1: Optimal Network G0=(V,Ep)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,node_size=800,width=2)
ax.legend(handles=leghands1,loc='best',ncol=1,prop={'size': 12})
fig.savefig("{}{}.png".format(figpath,'optimal-g0'),bbox_inches='tight')

#%% Figure 2: Transition from G0 to G1
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

edges = [(0,1),(0,2),(1,3),(3,5),(2,6),(6,7),(6,8),(6,9),(7,10),(10,11)]
del_edge = [(1,4)]
add_edge = [(4,6)]
nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,with_labels=True,edgelist=edges,
                 node_size=800)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='solid',edgelist=edges,
                       edge_color='black',width=2)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=del_edge,
                       edge_color='crimson',width=2.5)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=add_edge,
                       edge_color='green',width=2.5)
ax.legend(handles=leghands2,loc='best',ncol=1,prop={'size': 12})
fig.savefig("{}{}.png".format(figpath,'transition-g0-g1'),bbox_inches='tight')

#%% Figure 3: Variant Network 1 G1=(V,Ep')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

G.add_edges_from(add_edge)
G.remove_edge(*del_edge[0])

nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,node_size=800,width=2)
ax.legend(handles=leghands1,loc='best',ncol=1,prop={'size': 12})
fig.savefig("{}{}.png".format(figpath,'variant-g1'),bbox_inches='tight')

#%% Figure 4: Transition from G1 to G2
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)

# edges = [(0,1),(0,2),(1,3),(3,5),(2,6),(4,6),(6,7),(6,9),(7,10),(10,11)]
# del_edge = [(6,8)]
# add_edge = [(7,8)]
# nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,with_labels=True,edgelist=edges,
#                  node_size=800)
# nx.draw_networkx_edges(G,ax=ax,pos=cord,style='solid',edgelist=edges,
#                        edge_color='black',width=2)
# nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=del_edge,
#                        edge_color='crimson',width=2)
# nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=add_edge,
#                        edge_color='green',width=2)
# ax.legend(handles=leghands2,loc='best',ncol=1,prop={'size': 12})
# fig.savefig("{}{}.png".format(figpath,'transition-g1-g2'),bbox_inches='tight')

#%% Figure 5: Variant Network 1 G2=(V,Ep'')
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)

# G.add_edges_from(add_edge)
# G.remove_edge(*del_edge[0])

# nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,node_size=800,width=2)
# ax.legend(handles=leghands1,loc='best',ncol=1,prop={'size': 12})
# fig.savefig("{}{}.png".format(figpath,'variant-g2'),bbox_inches='tight')

#%% Figure 7: Transition from G1 to G3
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

edges = [(0,1),(0,2),(1,3),(3,5),(2,6),(4,6),(6,7),(6,8),(7,10),(10,11)]
del_edge = [(6,9)]
add_edge = [(2,9)]
nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,with_labels=True,edgelist=edges,
                 node_size=800)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='solid',edgelist=edges,
                       edge_color='black',width=2)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=del_edge,
                       edge_color='crimson',width=2.5)
nx.draw_networkx_edges(G,ax=ax,pos=cord,style='dashed',edgelist=add_edge,
                       edge_color='green',width=2.5)
ax.legend(handles=leghands2,loc='best',ncol=1,prop={'size': 12})
fig.savefig("{}{}.png".format(figpath,'transition-g1-g3'),bbox_inches='tight')

#%% Figure 8: Variant Network 1 G3=(V,Ep''')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

G.add_edges_from(add_edge)
G.remove_edge(*del_edge[0])

nx.draw_networkx(G,ax=ax,pos=cord,node_color=n_col,node_size=800,width=2)
ax.legend(handles=leghands1,loc='best',ncol=1,prop={'size': 12})
fig.savefig("{}{}.png".format(figpath,'variant-g3'),bbox_inches='tight')