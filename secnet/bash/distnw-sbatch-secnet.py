# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the secondary distribution network in a given county.
"""

import sys,os
import pandas as pd
import networkx as nx
import datetime,time

workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
tmpPath = scratchPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import GetRoads,GetHomes
from pyBuildSecNetlib import MapLink,SecNet
from pyBuildSecNetlib import MeasureDistance as dist

#%% Data Extraction
# Extract residence and road network information for the fiscode
fiscode = sys.argv[1]
roads = GetRoads(inpPath,fis=fiscode)
homes = GetHomes(inpPath,fis=fiscode)

#%% Mapping
# Map the residence to nearest road network link
MapLink(roads).map_point(homes,path=tmpPath+'sec-network/',fiscode=fiscode)
print("DONE MAPPING")

# Extract the mapping data
df_hmap = pd.read_csv(tmpPath+'sec-network/'+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])

#%% Creation of Network
secnet_obj = SecNet(homes,roads,H2Link)
L2Home = secnet_obj.link_to_home

links = [l for l in L2Home if 0<len(L2Home[l])]
linkdata = '\n'.join([','.join([str(x) for x in link])+'\t'+','.join([str(h) for h in L2Home[link]]) for link in links])
with open(tmpPath+'sec-network/'+fiscode+'-link2home.txt','w') as f:
    f.write(linkdata)

#%% Create secondary network
prefix = '51'+fiscode
suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
count = 5000001

for link in links:    
    # Solve the optimization problem and generate forest
    start_time = time.time()
    if len(L2Home[link])>150: 
        forest,roots = secnet_obj.generate_optimal_topology(link,minsep=50,hops=80,
                                tsfr_max=100,heuristic=5,path=tmpPath)
    elif len(L2Home[link])>80: 
        forest,roots = secnet_obj.generate_optimal_topology(link,minsep=50,hops=40,
                                tsfr_max=60,heuristic=15,path=tmpPath)
    else:
        forest,roots = secnet_obj.generate_optimal_topology(link,minsep=50,hops=10,
                                tsfr_max=25,heuristic=None,path=tmpPath)
    end_time = time.time()
    
    # Additional network data
    cord = nx.get_node_attributes(forest,'cord')
    load = nx.get_node_attributes(forest,'load')
    
    # Iterate over nodes to record transformers
    tsfr_data = ''
    dict_tsfr = {}
    rename = {} # to rename the transformers with node IDs
    tsfrlist = [] # to list the transformers
    for n in sorted(list(forest.nodes())):
        if n in roots:
            dict_tsfr[int(prefix+str(count))] = [cord[n][0],cord[n][1],load[n]]
            rename[n] = int(prefix+str(count))
            tsfrlist.append(int(prefix+str(count)))
            tsfr_data += ','.join([str(rename[n]),str(cord[n][0]),str(cord[n][1]),
                                  str(load[n])])+'\n'
            count += 1
    
    # Get edgelist for primary network design
    start_tsf = dict_tsfr[tsfrlist[0]][:2]
    if dist(start_tsf,roads.cord[link[0]])<dist(start_tsf,roads.cord[link[1]]):
        start = link[0]; end = link[1]
    else:
        start = link[1]; end = link[0]
    nodes = [start]+tsfrlist+[end]
    g = nx.Graph()
    nx.add_path(g,nodes)
    edgelist = list(g.edges())
    edgelist_data = '\n'.join([','.join([str(x) for x in edge]) \
                               for edge in edgelist])+'\n'
    
    # Iterate over edges to record secondary network
    network = ''
    for e in list(forest.edges()):
        node1 = str(rename[e[0]]) if e[0] in roots else str(e[0])
        type1 = 'T' if e[0] in roots else 'H'
        cord1a = str(cord[e[0]][0])
        cord1b = str(cord[e[0]][1])
        node2 = str(rename[e[1]]) if e[1] in roots else str(e[1])
        type2 = 'T' if e[1] in roots else 'H'
        cord2a = str(cord[e[1]][0])
        cord2b = str(cord[e[1]][1])
        network += '\t'.join([node1,type1,cord1a,cord1b,
                             node2,type2,cord2a,cord2b])+'\n'
    
    # Store time data
    time_data = str(len(L2Home[link]))+'\t'+str(end_time-start_time)+'\n'
    
    # Append network data to file
    dirname = 'sec-network/'
    with open(tmpPath+dirname+fiscode+"-sec-dist.txt",'a') as f_network:
        f_network.write(network)
    # Append edge data to file
    with open(tmpPath+dirname+fiscode+"-tsfr-net.csv",'a') as f_edge:
        f_edge.write(edgelist_data)
    # Append transformer data to file
    with open(tmpPath+dirname+fiscode+"-tsfr-data.csv",'a') as f_tsfr:
        f_tsfr.write(tsfr_data)
    # Append transformer data to file
    with open(tmpPath+dirname+fiscode+"-time-stat.txt",'a') as f_time:
        f_time.write(time_data)

