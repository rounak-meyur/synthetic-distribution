# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates the primary distribution network of the area
in one single step. Use this for areas with small number of transformers and
roads.

System inputs: fiscode of county
"""

import sys
import networkx as nx



# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
tmpPath = scratchPath + "/temp/"
dirname = 'osm-prim-master/'
target = 'osm-prim-road/'

#%% Read master graph
sub = sys.argv[1]
graph = nx.read_gpickle(tmpPath+dirname+sub+'-master.gpickle')
    

# Select the required data for network geometry
road = graph.__class__()
road.add_nodes_from(graph.nodes)
road.add_edges_from(graph.edges)

for n in road.nodes:
    road.nodes[n]['cord'] = graph.nodes[n]['cord']
    road.nodes[n]['label'] = graph.nodes[n]['label']


nx.write_gpickle(road,tmpPath+'osm-prim-road/'+str(sub)+'-road.gpickle')