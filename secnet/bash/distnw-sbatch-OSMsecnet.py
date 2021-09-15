# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the secondary distribution network in a given county.
"""

import sys,os
import networkx as nx
import datetime,time

workpath = os.getcwd()
libpath = workpath + "/Libraries/"
# Load scratchpath
scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inppath = scratchpath + "/input/"
tmppath = scratchpath + "/temp/" 


sys.path.append(libpath)
from pyExtractDatalib import GetOSMRoads,GetHomes,GetTransformers,GetMappings
from pyBuildSecNetlib import MapOSM
from pyMiscUtilslib import groups
from pyBuildSecNetlib import MeasureDistance as dist
from pyBuildSecNetlib import generate_optimal_topology as generate
from pyBuildSecNetlib import create_master_graph_OSM

#%% Data Extraction
# Extract residence and road network information for the fiscode
fiscode = sys.argv[1]

dirname = 'osm-sec-network/'
roads = GetOSMRoads(inppath,fis=fiscode)
homes = GetHomes(inppath,fis=fiscode)

# Store the road network in cache memory
nx.write_gpickle(roads, tmppath+"cache/"+fiscode+'-road.gpickle')

#%% Mapping
# Map the residence to nearest road network link
H2Link = MapOSM(roads).map_point(homes,path=tmppath+dirname,fiscode=fiscode)
print("DONE MAPPING")

#%% Reverse Mapping
L2Home = groups(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]
linkdata = '\n'.join([','.join([str(x) for x in link])+'\t'+','.join([str(h) for h in L2Home[link]]) for link in links])
with open(tmppath+dirname+fiscode+'-link2home.txt','w') as f:
    f.write(linkdata)

#%% Create secondary network
prefix = '51'+fiscode
suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
count = 5000001

for link in links:
    # Get the link geometry and home data
    linkgeom = roads.edges[link]['geometry']
    dict_homes = {h:{'cord':homes.cord[h],'load':homes.average[h]} \
              for h in L2Home[link]}
    # Solve the optimization problem and generate forest
    start_time = time.time()
    homelist = L2Home[link]
    if len(homelist)>350:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=100,
                                tsfr_max=150,heuristic=3,path=tmppath)
    elif len(homelist)>150:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=80,
                                tsfr_max=100,heuristic=5,path=tmppath)
    elif len(homelist)>80: 
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=40,
                                tsfr_max=60,heuristic=15,path=tmppath)
    else:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=10,
                                tsfr_max=25,heuristic=None,path=tmppath)
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
    road_cord1 = [roads.nodes[link[0]]['x'],roads.nodes[link[0]]['y']]
    road_cord2 = [roads.nodes[link[1]]['x'],roads.nodes[link[1]]['y']]
    if dist(start_tsf,road_cord1)<dist(start_tsf,road_cord2):
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
    
    with open(tmppath+dirname+fiscode+"-sec-dist.txt",'a') as f_network:
        f_network.write(network)
    # Append edge data to file
    with open(tmppath+dirname+fiscode+"-tsfr-net.csv",'a') as f_edge:
        f_edge.write(edgelist_data)
    # Append transformer data to file
    with open(tmppath+dirname+fiscode+"-tsfr-data.csv",'a') as f_tsfr:
        f_tsfr.write(tsfr_data)
    # Append transformer data to file
    with open(tmppath+dirname+fiscode+"-time-stat.txt",'a') as f_time:
        f_time.write(time_data)


#%% Create the transformer and road network
tsfr = GetTransformers(tmppath+dirname,fiscode,homes)
links = GetMappings(tmppath+dirname,fiscode)

target = "osm-master-graph/"
g = create_master_graph_OSM(roads,tsfr,links)
nx.write_gpickle(g, tmppath+target+fiscode+'-graph.gpickle')