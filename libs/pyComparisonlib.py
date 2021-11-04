# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:07:26 2020

Author: Rounak Meyur
"""

import networkx as nx
import numpy as np
import threading
import itertools
from pyGeometrylib import Link,geodist
from pygeodesy import hausdorff_


# Distribution of edges
def edge_dist(grid,edge_geom):    
    m_within = {}
    m_cross = {}    
    for g in grid:
        m_within[g] = sum([geom.within(g) for geom in edge_geom])
        m_cross[g] = sum([geom.intersects(g.exterior) for geom in edge_geom])
    return m_within,m_cross

#%% Haussdorff Distance between networks

# process: 
def process(data, edges1, edges2, items, start, end):
    s = 10
    for grid in items[start:end]:
        edges_pts1 = [p for geom in edges1 \
                      for p in Link(geom).InterpolatePoints(sep=s) \
                          if geom.within(grid) or geom.intersects(grid.exterior)]
        edges_pts2 = [p for geom in edges2 \
                      for p in Link(geom).InterpolatePoints(sep=s) \
                          if geom.within(grid) or geom.intersects(grid.exterior)]
        
        if len(edges_pts1) != 0 and len(edges_pts2) != 0:
            data[grid] = hausdorff_(edges_pts1,edges_pts2,
                                      distance=geodist).hd
        else:
            data[grid] = np.nan
        # print(data[grid])
        

def compute_hausdorff(items, eset1, eset2, num_splits=5):
    split_size = len(items) // num_splits
    threads = []
    D = {}
    for i in range(num_splits):
        # determine the indices of the list this thread will handle
        start = i * split_size
        # special case on the last chunk to account for uneven splits
        end = None if i+1 == num_splits else (i+1) * split_size
        # create the thread
        threads.append(
            threading.Thread(target=process, 
                             args=(D, eset1, eset2, items, start, end)))
        threads[-1].start() # start the thread we just created

    # wait for all threads to finish
    for t in threads:
        t.join()
    return D

#%% Graph motifs
def count_motifs(g,target):
    count = 0
    for sub_nodes in itertools.combinations(g.nodes(),len(target.nodes())):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            count += 1
    return count

    
