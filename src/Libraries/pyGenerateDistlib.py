# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:23:18 2019

Authors: Dr Anil Vullikanti
         Dr Henning Mortveit
         Rounak Meyur
"""


import networkx as nx
import csv
from scipy.spatial import cKDTree


import numpy as np
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt


def ReadRoadNetwork(pathname,road_network_fname,road_coord_fname):
    '''
    Creates a networkx graph for the road network with only the edges between
    Level 3, Level 4 and Level 5 nodes. The isolated nodes are removed from the
    graph.
    
    Inputs: road_network_fname: file name for the road network edgelist data
            road_coord_fname: file name for the road network coordinates data
    
    Outputs: RoadNW_coord: dictionary of road network nodes and their coordinates
             G: Networkx graph representation of the road network
    '''
    # Create the road network graph
    road_f_reader = csv.reader(open(pathname+road_network_fname), delimiter = '\t')
    G=nx.Graph()
    V={}
    for line in road_f_reader:
        if ('#' in line[0]): continue
        v = int(line[0]); w=int(line[1])
        if (v not in V): 
            G.add_node(v)
            V[v]=1
        if (w not in V): 
            G.add_node(w)
            V[w]=1
        if (line[2] == '3' or line[2] == '4' or line[2] == '5'): G.add_edge(v, w, weight=line[2])
        
    
    # Remove isolated nodes
    isolated_nodes = [nd for nd in G.nodes() if G.degree(nd)==0]
    for v in isolated_nodes:
        G.remove_node(v)
    
    # Create the road network coordinate dictionary
    RoadNW_coord={}
    coord_f_reader = csv.reader(open(pathname+road_coord_fname), delimiter = '\t')
    for line in coord_f_reader:
        if ('#' in line[0]): continue
        RoadNW_coord[int(line[0])] = float(line[1]), float(line[2])
  
    return RoadNW_coord, G


###############################################################################
# Functions to map road network with houses
def map_Homes_to_RoadNW(Home_coord, RoadNW_coord):
    '''
    Mapping between road network nodes and home/activity locations.
    
    Inputs:     Home_coord: dictionary of home/activity location coordinates
                RoadNW_coord: dictionary of road network node coordinates
    
    Outputs:    H2R: mapping from home/activity IDs to road network nodes
                R2H: mapping from road network nodes to home/activity IDs
    '''
    # Form Voronoi regions with the road nodes as centers
    voronoi_kdtree = cKDTree(np.array(RoadNW_coord.values()))
    regions = voronoi_kdtree.query(np.array(Home_coord.values()), k=1)
    
    # Map homes to the road nodes
    R2H = {}
    H2R = {Home_coord.keys()[k]:RoadNW_coord.keys()[regions[1][k]] for k in range(len(Home_coord))}
    for h in Home_coord:
        if H2R[h] in R2H: R2H[H2R[h]].append(h)
        else: R2H[H2R[h]] = [h]
    return H2R, R2H

    



