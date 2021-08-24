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
from pyGeometrylib import Link

print("Imported modules")



def create_final_network(path,sub):
    # Load network and remove unnecessary nodes
    synth_net = GetDistNet(path,sub)
    reg_nodes = list(nx.neighbors(synth_net, sub))
    tnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
    rnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='R' and n not in reg_nodes]
    sec_edges = [e for e in synth_net.edges if synth_net.edges[e]['label']=='S']
    
    nodelist = [sub]
    prim_edges = []
    
    for t in tnodes:
        if t not in nodelist:
            nodes = [v for v in nx.shortest_path(synth_net,sub,t) if v not in rnodes]
            edges = [(nodes[i],nodes[i+1]) for i,_ in enumerate(nodes[:-1])]
            nodelist.extend(nodes[1:])
            prim_edges.extend(edges)
    
    graph = nx.Graph()
    graph.add_edges_from(prim_edges+sec_edges)
    
    
    # Add edge and node properties
    for edge in prim_edges:
        path = nx.shortest_path(synth_net,edge[0],edge[1])
        path_geom = MultiLineString([synth_net[path[i]][path[i+1]]['geometry'] \
                     for i,_ in enumerate(path[:-1])])
        out_coords = [list(i.coords) for i in path_geom]
        
        # Edge data
        graph.edges[edge]['geometry'] = LineString([i for sublist in out_coords for i in sublist])
        graph.edges[edge]['geo_length'] = Link(graph.edges[edge]['geometry']).geod_length
        if sub in edge:
            graph.edges[edge]['label'] = 'E'
            graph.edges[edge]['r'] = 1e-12 * graph.edges[edge]['geo_length']
            graph.edges[edge]['x'] = 1e-12 * graph.edges[edge]['geo_length']
        else:
            graph.edges[edge]['label'] = 'P'
            graph.edges[edge]['r'] = 0.8625/39690 * graph.edges[edge]['geo_length']
            graph.edges[edge]['x'] = 0.4154/39690 * graph.edges[edge]['geo_length']
    
    for edge in sec_edges:
        graph.edges[edge]['geometry'] = synth_net.edges[edge]['geometry']
        graph.edges[edge]['label'] = 'S'
        graph.edges[edge]['geo_length'] = Link(graph.edges[edge]['geometry']).geod_length
        graph.edges[edge]['r'] = 0.81508/57.6 * graph.edges[edge]['geo_length']
        graph.edges[edge]['x'] = 0.3496/57.6 * graph.edges[edge]['geo_length']
        
    for n in graph.nodes:
        graph.nodes[n]['cord'] = synth_net.nodes[n]['cord']
        graph.nodes[n]['label'] = synth_net.nodes[n]['label']
        graph.nodes[n]['load'] = synth_net.nodes[n]['load']
    return graph
    

#%% 
sub = 121144    

    
    
    
        


























        