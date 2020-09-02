# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the secondary distribution network in a given county.
"""

import sys,os
import shutil
import pandas as pd
import networkx as nx
import datetime,time

workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
tmpPath = scratchPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider
from pyMapElementslib import MapLink
from pyBuildNetworklib import MeasureDistance as dist

# Create mappings
fiscode = sys.argv[1]
q_object = Query(csvPath,inpPath)
homes,roads = q_object.GetDataset(fis=fiscode)
print("Total number of links:",roads.graph.number_of_edges(),
      "\nTotal number of homes:",len(homes.cord))
print(nx.number_connected_components(roads.graph))

MapLink(roads).map_point(homes,path=csvPath,fiscode=fiscode)
print("DONE MAPPING")

df_hmap = pd.read_csv(csvPath+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home

links = [l for l in L2Home if 0<len(L2Home[l])]
linkdata = '\n'.join([','.join([str(x) for x in link])+'\t'+','.join([str(h) for h in L2Home[link]]) for link in links])
with open(csvPath+fiscode+'-link2home.txt','w') as f:
    f.write(linkdata)
print("Total number of links:",len(links))

#%% Create secondary network

prefix = '51'+fiscode
suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
count = 5000001

for link in links:    
    # Solve the optimization problem and generate forest
    start_time = time.time()
    if len(L2Home[link])>80: 
        forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,hops=40,
                                tsfr_max=60,followroad=True,heuristic=15,path=tmpPath)
    else:
        forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,hops=10,
                                tsfr_max=25,followroad=True,heuristic=None,path=tmpPath)
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
    with open(tmpPath+fiscode+"-sec-dist.txt",'a') as f_network:
        f_network.write(network)
    # Append edge data to file
    with open(tmpPath+fiscode+"-tsfr-net.csv",'a') as f_edge:
        f_edge.write(edgelist_data)
    # Append transformer data to file
    with open(tmpPath+fiscode+"-tsfr-data.csv",'a') as f_tsfr:
        f_tsfr.write(tsfr_data)
    # Append transformer data to file
    with open(tmpPath+fiscode+"-time-stat.txt",'a') as f_time:
        f_time.write(time_data)



os.mkdir(csvPath+fiscode+'-data')
newdir = csvPath+fiscode+'-data/'
shutil.move(tmpPath+fiscode+"-sec-dist.txt",newdir+fiscode+"-sec-dist.txt")
shutil.move(tmpPath+fiscode+"-tsfr-net.csv",newdir+fiscode+"-tsfr-net.csv")
shutil.move(tmpPath+fiscode+"-tsfr-data.csv",newdir+fiscode+"-tsfr-data.csv")
shutil.move(tmpPath+fiscode+"-time-stat.txt",newdir+fiscode+"-time-stat.txt")
shutil.move(csvPath+fiscode+'-home2link.csv',newdir+fiscode+"-home2link.csv")
shutil.move(csvPath+fiscode+'-home-load.csv',newdir+fiscode+"-home-load.csv")