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
from pyExtractDatalib import Query
from pyBuildNetworklib import Initialize_Primary as init
from pyBuildNetworklib import MeasureDistance as dist

# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

#%% Get transformers and store them in csv
with open(inpPath+'fislist.txt') as f:
    all_areas = f.readlines()[0].strip('\n').split(' ')
with open(inpPath+'arealist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

q_object = Query(csvPath,inpPath)
roads = q_object.GetAllRoads(all_areas)
subs = q_object.GetAllSubstations()
tsfr = q_object.GetAllTransformers(areas)

set_tsfr = set(list(tsfr.cord.keys()))
set_road = set(list(roads.graph.nodes()))
print(len(list(set_tsfr.intersection(set_road))))


sys.exit(0)
links = q_object.GetAllMappings(areas)



#%% Initialize Primary Network Generation Process
def get_subgraph(graph,subdata):
    # Get and set attributes
    nodepos = nx.get_node_attributes(graph,'cord')
    
    # Get the distance from the nearest substation
    hvdist = {r:dist(subdata.cord,nodepos[r]) \
              for r in subdata.nodes}
    
    sgraph = graph.subgraph(subdata.nodes)
    nx.set_node_attributes(sgraph,hvdist,'distance')
    return sgraph


G,S2Node = init(subs,roads,tsfr,links)
substation = nt("local_substation",field_names=["id","cord","nodes"])


for sub in S2Node:
    sub_data = substation(id=sub,cord=subs.cord[sub],nodes=S2Node[sub])
    sub_graph = get_subgraph(G,sub_data)
    edgelist = list(sub_graph.edges())
    nodepos = nx.get_node_attributes(sub_graph,'cord')
    nodedist = nx.get_node_attributes(sub_graph,'distance')
    nodeload = nx.get_node_attributes(sub_graph,'load')
    nodelabel = nx.get_node_attributes(sub_graph,'label')
    edgedata = '\n'.join(
        ['\t'.join([str(edge[0]),nodelabel[edge[0]],str(nodepos[edge[0]][0]),
                    str(nodepos[edge[0]][1]),str(nodeload[edge[0]]),str(nodedist[edge[0]]),
                    str(edge[1]),nodelabel[edge[1]],str(nodepos[edge[1]][0]),
                    str(nodepos[edge[1]][1]),str(nodeload[edge[1]]),str(nodedist[edge[1]])])\
         for edge in edgelist])
    with open(tmpPath+'prim-master/'+str(sub)+'-master.txt','w') as f:
        f.write(edgedata)
    

