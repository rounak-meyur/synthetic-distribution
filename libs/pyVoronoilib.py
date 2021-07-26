# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:01:00 2020

Author: Rounak

A library with methods/attributes for creating Voronoi partitions in the given
region. The centers are the substations and the transformer/road network nodes
are partioned into connected graphs.
"""

import networkx as nx
import numpy as np
from pyqtree import Index
from scipy.spatial import cKDTree
from shapely.geometry import Point
import time


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
    center_bd = bounds(pt_center,0.1)
    matches = idx.intersect(center_bd)
    closest_node = min(matches,key=lambda i: nodes[i][0].distance(pt_center))
    return nodes[closest_node][-1]

def get_nearest_road(subs,graph):
    """
    Get list of nodes mapped in the Voronoi cell of the substation. The Voronoi 
    cell is determined on the basis of geographical distance.
    Returns: dictionary of substations with list of nodes mapped to it as values
    """
    # Get the Voronoi centers and data points
    centers = list(subs.cord.keys())
    center_pts = [subs.cord[s] for s in centers]
    nodes = list(graph.nodes())
    nodepos = nx.get_node_attributes(graph,'cord')
    nodelabel = nx.get_node_attributes(graph,'label')
    node_pts = [nodepos[n] for n in nodes]
    
    # Find number of road nodes mapped to each substation
    voronoi_kdtree = cKDTree(center_pts)
    _, node_regions = voronoi_kdtree.query(node_pts, k=1)
    sub_map = {s:node_regions.tolist().count(s) for s in range(len(centers))}
    
    # Compute new centers of Voronoi regions
    centers = [centers[s] for s in sub_map if sub_map[s]>50]
    center_pts = [subs.cord[s] for s in centers]
    
    # Recompute the Voronoi regions and generate the final map
    voronoi_kdtree = cKDTree(center_pts)
    _, node_regions = voronoi_kdtree.query(node_pts, k=1)
    
    # Index the region and assign the nodes to the substation
    indS2N = {i:np.argwhere(i==node_regions)[:,0]\
              for i in np.unique(node_regions)}
    S2Node = {centers[i]:[nodes[j] for j in indS2N[i]] for i in indS2N}
    
    # Compute nearest node to substation
    S2Near = {}
    for s in S2Node:
        nodes_partition = [n for n in S2Node[s] if nodelabel[n]=='R']
        nodecord = {n: nodepos[n] for n in nodes_partition}
        S2Near[s] = find_nearest_node(subs.cord[s],nodecord)
    return S2Near

def get_partitions(S2Near,graph):
    """
    Get list of nodes mapped in the Voronoi cell of the substation. The Voronoi 
    cell is determined on the basis of shortest path distance from each node to
    the nearest node to the substation.
    Returns: dictionary of substations with list of nodes mapped to it as values
    """
    # Compute Voronoi cells with network distance 
    centers = list(S2Near.values())
    cells = nx.voronoi_cells(graph, centers, 'length')
    
    # Recompute Voronoi cells for larger primary networks
    centers = [c for c in centers if len(cells[c])>100]
    cells = nx.voronoi_cells(graph, centers, 'length')
    
    # Recompute S2Near and S2Node
    S2Near = {s:S2Near[s] for s in S2Near if S2Near[s] in centers}
    S2Node = {s:list(cells[S2Near[s]]) for s in S2Near}
    return S2Node

def create_voronoi(subs,graph):
    """
    Initialization function to generate master graph for primary network generation
    and node to substation mapping. The function is called before calling the class
    object to optimize the primary network.
    
    Inputs: 
        subs: named tuple for substations
        roads: named tuple for road network
        tsfr: named tuple for local transformers
        links: list of road links along which transformers are placed.
    Returns:
        graph: master graph from which the optimal network would be generated.
        S2Node: mapping between substation and road/transformer nodes based on shortest
        path distance in the master network.
    """
    start = time.time()
    S2Near = get_nearest_road(subs,graph)
    print(time.time()-start)
    
    start = time.time()
    S2Node = get_partitions(S2Near,graph)
    print(time.time()-start)
    return S2Near,S2Node