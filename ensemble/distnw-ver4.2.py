# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx

import matplotlib.pyplot as plt

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/results-dec12/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network
from pyValidationlib import create_base


import random


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()
G = create_base(csvPath)
sub = 34780
dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)
nodelab = nx.get_node_attributes(dist_net,'label')
sec_nodes = [n for n in nodelab if nodelab[n]=='H']
for n in sec_nodes:
    dist_net.remove_node(n)

fig1 = plt.figure(figsize=(10,6))
ax1 = fig1.add_subplot(111)
pos = hierarchy_pos(G,8)
nx.draw_networkx(G,pos=pos,ax=ax1,with_labels=False,node_size=1,node_color='blue',
                 edge_color='blue')

fig2 = plt.figure(figsize=(10,6))
ax2 = fig2.add_subplot(111)
pos = hierarchy_pos(dist_net,sub)    
nx.draw_networkx(dist_net,pos=pos,ax=ax2,with_labels=False,node_size=1,node_color='blue',
                 edge_color='blue')











# for sub in [34780,34810,34816,28228,28235]:
#     dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)
#     nodelab = nx.get_node_attributes(dist_net,'label')
#     sec_nodes = [n for n in nodelab if nodelab[n]=='H']
#     for n in sec_nodes:
#         dist_net.remove_node(n)
    