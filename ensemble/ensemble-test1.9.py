# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program uses a Markov chain to create synthetic networks which
are solutions of the optimization program.
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point,LineString
from matplotlib.lines import Line2D
from pyqtree import Index
import threading

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
tsfrpath = rootpath + "/secnet/out/osm-prim-road/"
distpath = rootpath + "/primnet/out/osm-primnet-final/"
outpath = workpath + "/out/osm-ensemble-mc/"
sys.path.append(libpath)

from pyGeometrylib import geodist,Link
from pyExtractDatalib import GetDistNet,GetPrimRoad
from pyBuildPrimNetlib import powerflow

#%% Create new networks
def create_network(graph,master):
    sec_edges = [e for e in master.edges() \
                 if master[e[0]][e[1]]['label']=='S']
    graph.add_edges_from(sec_edges)
    
    # node attributes
    nodepos = {n:master.nodes[n]['cord'] for n in graph}
    nodelab = {n:master.nodes[n]['label'] for n in graph}
    nodeload = {n:master.nodes[n]['load'] for n in graph}
    nx.set_node_attributes(graph,nodepos,'cord')
    nx.set_node_attributes(graph,nodelab,'label')
    nx.set_node_attributes(graph,nodeload,'load')
    
    # edge attributes
    edge_geom = {}
    edge_label = {}
    edge_r = {}
    edge_x = {}
    glength = {}
    for e in list(graph.edges()):
        if e in master.edges():
            edge_geom[e] = master[e[0]][e[1]]['geometry']
            edge_label[e] = master[e[0]][e[1]]['label']
            edge_r[e] = master[e[0]][e[1]]['r']
            edge_x[e] = master[e[0]][e[1]]['x']
            glength[e] = master[e[0]][e[1]]['geo_length']   
        else:
            edge_geom[e] = LineString((nodepos[e[0]],nodepos[e[1]]))
            glength[e] = Link(edge_geom[e]).geod_length
            edge_label[e] = 'P'
            length = glength[e] if glength[e] != 0.0 else 1e-12
            edge_r[e] = 0.8625/39690 * length
            edge_x[e] = 0.4154/39690 * length
            
    nx.set_edge_attributes(graph, edge_geom, 'geometry')
    nx.set_edge_attributes(graph, edge_label, 'label')
    nx.set_edge_attributes(graph, edge_r, 'r')
    nx.set_edge_attributes(graph, edge_x, 'x')
    nx.set_edge_attributes(graph, glength,'geo_length')
    return


def get_new(graph,road):
    """
    Gets a new network from the original synthetic distribution network by 
    swapping an edge with a minimum length reconnection. The created network is
    rejected if power flow is not satisfied.
    """
    prim_nodes = [n for n in graph if graph.nodes[n]['label']!='H']
    prim_edges = [e for e in graph.edges() \
                  if graph[e[0]][e[1]]['label']!='S']
    sub = [n for n in graph if graph.nodes[n]['label']=='S'][0]
    
    # Reconstruct primary network
    new_graph = graph.__class__()
    new_graph.add_nodes_from(prim_nodes)
    new_graph.add_edges_from(prim_edges)
    
    # Get edgelist for random sampling and remove an edge
    edgelist = [e for e in prim_edges if graph.nodes[e[0]]['label']=='T' \
                and graph.nodes[e[1]]['label']!='T']
    rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
    new_graph.remove_edge(*rand_edge)
    
    # Get node labels for the edge: end_node is connected to root
    if nx.has_path(new_graph,sub,rand_edge[0]):
        end_node = rand_edge[0]
        other_node = rand_edge[1]
    else:
        end_node = rand_edge[1]
        other_node = rand_edge[0]
    
    # Analyze the two disconnected components
    comps = list(nx.connected_components(new_graph))
    connected_nodes = list(comps[0]) \
            if end_node in list(comps[0]) else list(comps[1])
    # dict_node = {n:graph.nodes[n]['cord'] for n in connected_nodes \
    #              if n!=end_node}
    # center_node = graph.nodes[other_node]['cord']
    # near_node = find_nearest_node(center_node,dict_node)
    # new_edge = (near_node,other_node)
    # new_graph.add_edge(*new_edge)
    # create_network(new_graph,graph)
    
    # print("Checking power flow result...")
    powerflow(new_graph)
    voltage = [new_graph.nodes[n]['voltage'] for n in new_graph \
               if new_graph.nodes[n]['label']!='H']
    low_voltage_nodes = [v for v in voltage if v<=0.87]
    check = (len(low_voltage_nodes)/len(voltage))*100.0
    if check>5.0:
        print("Many nodes have low voltage. Percentage:",check)
        return graph,0
    else:
        print("Acceptable power flow results. Percentage:",check)
        return new_graph,1


sub = 121143
synth_net = GetDistNet(distpath,sub)
road_net = GetPrimRoad(tsfrpath,sub)

#%% Create networks
# Markov chain initialized with the same graph M times 
# and traversed over N iterations

# def process(items, start, end):
#     for item in items[start:end]:
#         iterations = 20
#         try:
#             count = 0
#             while count < iterations:
#                 if count == 0:
#                     new_graph,flag = get_new(synth_net,sub)
#                 else:
#                     new_graph,flag = get_new(new_graph,sub)
#                 count += flag
#             # save the network
#             nx.write_gpickle(new_graph,
#                              outpath+str(sub)+'-ensemble-'+str(item+1)+'.gpickle')
#         except Exception:
#             print('error with item')

# def split_processing(items, num_splits=4):
#     split_size = len(items) // num_splits
#     threads = []
#     for i in range(num_splits):
#         # determine the indices of the list this thread will handle
#         start = i * split_size
#         # special case on the last chunk to account for uneven splits
#         end = None if i+1 == num_splits else (i+1) * split_size
#         # create the thread
#         threads.append(
#             threading.Thread(target=process, args=(items, start, end)))
#         threads[-1].start() # start the thread we just created

#     # wait for all threads to finish
#     for t in threads:
#         t.join()



# numnets = 20
# items = range(numnets)
# split_processing(items)