# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:44:07 2021

Author: Rounak

Description: Removes road nodes from the created synthetic network and just 
creates a network consisting of transformer nodes. 
"""

import sys,os
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyBuildPrimNetlib import assign_linetype

print("Imported modules")

#%% Load network
sub = 121144


synth_net = GetDistNet(distpath,sub)
tnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
rnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='R']
graph = nx.Graph()
nodelist = [sub]
edgelist = []

for t in tnodes:
    if t not in nodelist:
        nodes = [v for v in nx.shortest_path(synth_net,sub,t) if v not in rnodes]
        edges = [(nodes[i],nodes[i+1]) for i,_ in enumerate(nodes[:-1])]
        nodelist.extend(nodes[1:])
        edgelist.extend(edges)

graph = nx.Graph()
graph.add_edges_from(edgelist)


#%% Add edge geometries
edge_geom = {}
for edge in edgelist:
    path = nx.shortest_path(synth_net,edge[0],edge[1])
    path_geom = MultiLineString([synth_net[path[i]][path[i+1]]['geometry'] \
                 for i,_ in enumerate(path[:-1])])
    out_coords = [list(i.coords) for i in path_geom]
    edge_geom[edge] = LineString([i for sublist in out_coords for i in sublist])
    
        


























        