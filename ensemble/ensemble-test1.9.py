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
distpath = rootpath + "/primnet/out/osm-primnet/"
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
    for n in graph:
        graph.nodes[n]['cord'] = master.nodes[n]['cord']
        graph.nodes[n]['load'] = master.nodes[n]['load']
        graph.nodes[n]['label'] = master.nodes[n]['label']
    
    # edge attributes
    for e in list(graph.edges()):
        if e in master.edges():
            graph.edges[e]['geometry'] = master[e[0]][e[1]]['geometry']
            graph.edges[e]['geo_length'] = master[e[0]][e[1]]['geo_length']
            graph.edges[e]['label'] = master[e[0]][e[1]]['label']
            graph.edges[e]['r'] = master[e[0]][e[1]]['r']
            graph.edges[e]['x'] = master[e[0]][e[1]]['x']
        else:
            graph.edges[e]['geometry'] = LineString((master.nodes[e[0]]['cord'],
                                                     master.nodes[e[1]]['cord']))
            graph.edges[e]['geo_length'] = Link(graph.edges[e]['geometry']).geod_length
            graph.edges[e]['label'] = 'P'
            length = graph.edges[e]['geo_length'] if graph.edges[e]['geo_length'] != 0.0 else 1e-12
            graph.edges[e]['r'] = 0.8625/39690 * length
            graph.edges[e]['x'] = 0.4154/39690 * length
    return


def get_new(graph,road):
    """
    Gets a new network from the original synthetic distribution network by 
    swapping an edge with a minimum length reconnection. The created network is
    rejected if power flow is not satisfied.
    """
    prim_nodes = [n for n in graph if graph.nodes[n]['label']!='H']
    reg_nodes = [n for n in graph if graph.nodes[n]['label']=='R']
    prim_edges = [e for e in graph.edges() \
                  if graph[e[0]][e[1]]['label']!='S']
    sub = [n for n in graph if graph.nodes[n]['label']=='S'][0]
    
    # Reconstruct primary network to alter without modifying original
    new_graph = graph.__class__()
    new_graph.add_nodes_from(prim_nodes)
    new_graph.add_edges_from(prim_edges)
    
    # Get edgelist for random sampling and remove an edge
    edgelist = [e for e in prim_edges if graph.nodes[e[0]]['label']=='T' \
                and graph.nodes[e[1]]['label']=='T']
    rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
    rem_geom = graph.edges[rand_edge]['geometry']
    new_graph.remove_edge(*rand_edge)
    
    # Analyze the two disconnected components
    target = rand_edge[1] if nx.has_path(new_graph,sub,rand_edge[0]) else rand_edge[0]
    comps = list(nx.connected_components(new_graph))
    if sub in list(comps[0]):
        not_connected_nodes = list(comps[1])
        connected_nodes = list(comps[0])
    else:
        not_connected_nodes = list(comps[0])
        connected_nodes = list(comps[1])
    
    # Create a dummy road network
    new_road = road.__class__()
    new_road.add_nodes_from(road.nodes)
    new_road.add_edges_from(road.edges)
    for e in new_road.edges:
        new_road.edges[e]['geometry'] = LineString([road.nodes[e[0]]['cord'],
                                                    road.nodes[e[1]]['cord']])
        new_road.edges[e]['length'] = Link(new_road.edges[e]['geometry']).geod_length
        
    # Delete the same edge/path from the dummy road network
    path = [rand_edge[0],rand_edge[1]]
    if len(rem_geom.coords)>2:
        path_coords = list(rem_geom.coords)[1:-1]
        for cord in path_coords:
            neighbor_dist = {n: geodist(cord,road.nodes[n]['cord']) \
                             for n in nx.neighbors(road,path[-2])}
            node_in_path = min(neighbor_dist,key=neighbor_dist.get)
            path.insert(-1,node_in_path)
    rem_road_edge = [(path[i],path[i+1]) for i,_ in enumerate(path[:-1])]
    for edge in rem_road_edge:
        new_road.remove_edge(*edge)
    
    # Get edgeset
    for n in not_connected_nodes:
        circle = Point(graph.nodes[n]['cord']).buffer(0.01)
        nearby_nodes = [m for m in connected_nodes \
                        if Point(graph.nodes[m]['cord']).within(circle)]
        
        
    
    sys.exit(0)
    
    # Get the set of edges between the disconnected components
    # edge_set = []
    # for e in new_road:
    #     if (e[0] in comps[0] and e[1] in comps[1]) or (e[0] in comps[1] and e[1] in comps[0]):
            
    
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


#%% Grow the network
def base(graph,road):
    edges = [e for e in graph.edges if graph.edges[e]['label']=='E']
    sub = [n for n in graph if graph.nodes[n]['label']=='S'][0]
    net = graph.__class__()
    net.add_edges_from(edges)
    
    # Edge data
    for e in net.edges:
        net.edges[e]['geometry'] = graph[e[0]][e[1]]['geometry']
        net.edges[e]['geo_length'] = graph[e[0]][e[1]]['geo_length']
        net.edges[e]['label'] = graph[e[0]][e[1]]['label']
        net.edges[e]['r'] = graph[e[0]][e[1]]['r']
        net.edges[e]['x'] = graph[e[0]][e[1]]['x']
    
    # Node data
    for n in net.nodes:
        net.nodes[n]['cord'] = graph.nodes[n]['cord']
        net.nodes[n]['load'] = graph.nodes[n]['load']
        net.nodes[n]['label'] = graph.nodes[n]['label']
    
    # BFS sort of nodes
    S = list(nx.bfs_tree(graph,sub))
    return net,S


def grow(graph,road,nodelist):
    
    return

#%% Run the program


sub = 121143
synth_net = GetDistNet(distpath,sub)
road_net = GetPrimRoad(tsfrpath,sub)

sys.exit(0)

get_new(synth_net,road_net)

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