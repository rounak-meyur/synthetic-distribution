# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:01:00 2020

Author: Rounak

A library with methods/attributes for creating Voronoi partitions in the given
region. The centers are the substations and the transformer/road network nodes
are partioned into connected graphs.
"""

from geographiclib.geodesic import Geodesic
import networkx as nx
import numpy as np
from pyqtree import Index
from scipy.spatial import cKDTree
from shapely.geometry import Point
import time

def MeasureDistance(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates. The format of each point 
    is (longitude,latitude).
    pt1: (longitude,latitude) of point 1
    pt2: (longitude,latitude) of point 2
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

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
    

def create_master_graph(roads,tsfr,links):
    """
    Creates the master graph consisting of all possible edges from the road
    network links. Each node in the graph has a spatial attribute called
    cord.
    
    roads:
        TYPE: namedtuple
        DESCRIPTION: road network data
    tsfr:
        TYPE: namedtuple
        DESCRIPTION: local transformer data
    
    """
    road_edges = list(roads.graph.edges())
    tsfr_edges = list(tsfr.graph.edges())
    for edge in links:
        if edge in road_edges:
            road_edges.remove(edge)
        elif (edge[1],edge[0]) in road_edges:
            road_edges.remove((edge[1],edge[0]))
    edgelist = road_edges + tsfr_edges
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    nodelist = list(graph.nodes())
    
    # Coordinates of nodes in network
    nodepos = {n:roads.cord[n] if n in roads.cord else tsfr.cord[n] for n in nodelist}
    nx.set_node_attributes(graph,nodepos,name='cord')
    
    # Length of edges
    edge_length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                    for e in edgelist}
    nx.set_edge_attributes(graph,edge_length,name='length')
    
    # Label the nodes in network
    node_label = {n:'T' if n in tsfr.cord else 'R' for n in list(graph.nodes())}
    nx.set_node_attributes(graph,node_label,'label')
    
    # Add load at local transformer nodes
    node_load = {n:tsfr.load[n] if node_label[n]=='T' else 0.0 \
                 for n in list(graph.nodes())}
    nx.set_node_attributes(graph,node_load,'load')
    
    # Add the associated secondary network for each local transformer
    sec_net = {n:tsfr.secnet[n] if node_label[n]=='T' else nx.Graph() \
                 for n in list(graph.nodes())}
    nx.set_node_attributes(graph,sec_net,'secnet')
    return graph

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