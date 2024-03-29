# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates the primary distribution network of the area
in one single step. Use this for areas with small number of transformers and
roads.

System inputs: fiscode of county
"""

import sys,os
import networkx as nx



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import GetSubstations
from pyVoronoilib import create_voronoi as vorpart

# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"
dirname = 'osm-master-graph/'

#%% Pre-processing step

# Extract the arealist
with open(inpPath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

# Extract all substations in the region
subs = GetSubstations(inpPath)

# Compose the total master graph of road and transformer nodes
G = nx.Graph()
for a in areas:
    g = nx.read_gpickle(tmpPath+dirname+a+'-graph.gpickle')
    G = nx.compose(G,g)

# Create the partition
S2Near,S2Node = vorpart(subs,G)


#%% Initialize Primary Network Generation Process

for sub in S2Node:
    sub_graph = G.subgraph(S2Node[sub])
    nx.write_gpickle(sub_graph,tmpPath+'osm-prim-master/'+str(sub)+'-master.gpickle')
    
    # Save the substation data
    data = '\t'.join([str(x) for x in [sub,S2Near[sub],subs.cord[sub][0],
                                       subs.cord[sub][1]]]) + '\n'
    with open(tmpPath + "subdata.txt",'a') as f:
        f.write(data)
    
    

