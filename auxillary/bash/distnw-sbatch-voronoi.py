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
from collections import namedtuple as nt



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

#%% Pre-processing step

# Extract the arealist
with open(inpPath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

# Extract all substations in the region
subs = GetSubstations(inpPath,areas=areas)

# Compose the total master graph of road and transformer nodes
G = nx.Graph()
for a in areas:
    g = nx.read_gpickle(tmpPath+'master-graph/'+a+'-graph.gpickle')
    G = nx.compose(G,g)

# Create the partition
S2Near,S2Node = vorpart(subs,G)
substation = nt("local_substation",field_names=["id","cord","nodes","nearest"])


#%% Initialize Primary Network Generation Process
def get_subgraph(graph,subdata):
    near = subdata.nearest
    nodelist = subdata.nodes
    
    # Get and set attributes
    nodelabel = nx.get_node_attributes(graph,'label')
    
    # Get the distance from the nearest substation
    hvpath = {r:nx.shortest_path(graph,source=near,target=r,weight='length') \
              if nodelabel[r]=='R' else [] for r in nodelist}
    hvdist = {r:sum([graph[hvpath[r][i]][hvpath[r][i+1]]['length']\
                     for i in range(len(hvpath[r])-1)]) for r in nodelist}
    
    sgraph = graph.subgraph(subdata.nodes)
    nx.set_node_attributes(sgraph,hvpath,'feedpath')
    nx.set_node_attributes(sgraph,hvdist,'distance')
    return sgraph


for sub in S2Node:
    sub_data = substation(id=sub,cord=subs.cord[sub],
                          nodes=S2Node[sub],nearest=S2Near[sub])
    sub_graph = get_subgraph(G,sub_data)
    nx.write_gpickle(sub_graph,tmpPath+'prim-master/'+str(sub)+'-master.gpickle')
    

