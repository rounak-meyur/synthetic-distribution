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
from shapely.geometry import Point
from matplotlib.lines import Line2D
from pyqtree import Index
import threading

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/osm-primnet/"
outpath = workpath + "/out/osm-ensemble-mc/"
sys.path.append(libpath)

from pyGeometrylib import geodist
from pyExtractDatalib import GetDistNet

#%% Create new networks
def get_new(graph,sub):
    """
    Gets a new network from the original synthetic distribution network by 
    swapping an edge with a minimum length reconnection. The created network is
    rejected if power flow is not satisfied.
    """
    prim_nodes = [n for n in graph if graph.nodes[n]['label']!='H']
    prim_edges = [e for e in graph.edges() \
                  if graph[e[0]][e[1]]['label']!='S']
    
    # Reconstruct primary network
    new_graph = graph.__class__()
    new_graph.add_nodes_from(prim_nodes)
    new_graph.add_edges_from(prim_edges)
    
    # Get edgelist for random sampling and remove an edge
    edgelist = [e for e in prim_edges if graph[e[0]][e[1]]['label']!='E']
    rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
    new_graph.remove_edge(*rand_edge)
    
    # Get node labels for the edge: end_node is connected to root
    if nx.has_path(new_graph,sub,rand_edge[0]):
        end_node = rand_edge[0]
    else:
        end_node = rand_edge[1]
    other_node = rand_edge[0] if end_node==rand_edge[1] else rand_edge[1]
    
    if graph.nodes[end_node]['label']=='R':
        # print("Leaf node is road node. Choose another edge.")
        return graph,0
    else:
        # print("Found a suitable edge to remove. Proceeding...")
        comps = list(nx.connected_components(new_graph))
        connected_nodes = list(comps[0]) \
                if end_node in list(comps[0]) else list(comps[1])
        dict_node = {n:graph.nodes[n]['cord'] for n in connected_nodes \
                     if n!=end_node}
        center_node = graph.nodes[other_node]['cord']
        near_node = find_nearest_node(center_node,dict_node)
        new_edge = (near_node,other_node)
        new_graph.add_edge(*new_edge)
        create_network(new_graph,graph)
        
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


sub = int(sys.argv[1])
synth_net = GetDistNet(distpath,sub)

#%% Create networks
# Markov chain initialized with the same graph M times 
# and traversed over N iterations

def process(items, start, end):
    for item in items[start:end]:
        iterations = 20
        try:
            count = 0
            while count < iterations:
                if count == 0:
                    new_graph,flag = get_new(synth_net,sub)
                else:
                    new_graph,flag = get_new(new_graph,sub)
                count += flag
            # save the network
            nx.write_gpickle(new_graph,
                             outpath+str(sub)+'-ensemble-'+str(item+1)+'.gpickle')
        except Exception:
            print('error with item')

def split_processing(items, num_splits=4):
    split_size = len(items) // num_splits
    threads = []
    for i in range(num_splits):
        # determine the indices of the list this thread will handle
        start = i * split_size
        # special case on the last chunk to account for uneven splits
        end = None if i+1 == num_splits else (i+1) * split_size
        # create the thread
        threads.append(
            threading.Thread(target=process, args=(items, start, end)))
        threads[-1].start() # start the thread we just created

    # wait for all threads to finish
    for t in threads:
        t.join()



numnets = 20
items = range(numnets)
split_processing(items)