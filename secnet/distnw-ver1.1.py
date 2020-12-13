# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes.

The output of the program are the following:
    1. secondary network which is a forest of trees. The graph is output in a 
    .txt file in the temp directory
    2. list of transformers with the load and spatial coordinates in csv dir
    3. list of edges in the primary network formed by road nodes and transformers
"""

import sys,os
import pandas as pd
import networkx as nx
import datetime


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/121-data/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query,getareas
from pyBuildNetworklib import Spider
from pyBuildNetworklib import MeasureDistance as dist


#%% Initialization of data sets and mappings
fiscode = '121'
q_object = Query(csvPath,inpPath)
areas = getareas(inpPath,fiscode)
homes,roads = q_object.GetDataset(fislist=areas)


df_hmap = pd.read_csv(csvPath+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home

links = [l for l in L2Home if 0<len(L2Home[l])]
sys.exit(0)

#%% Create secondary network
import time

prefix = '51'+fiscode
suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
network = ''
count = 5000001
dict_tsfr = {}
edgelist = []
dict_time = {}

for link in links[1812:1813]:
    # initialize
    start_time = time.time()
    rename = {} # to rename the transformers with node IDs
    tsfrlist = [] # to list the transformers
    
    # Solve the optimization problem and generate forest
    if len(L2Home[link])>80:
        print("HERE")
        forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,hops=40,
                                tsfr_max=60,followroad=True,heuristic=10,path=tmpPath)
    else:
        forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,hops=10,
                                tsfr_max=25,followroad=True,heuristic=None,path=tmpPath)
    
    cord = nx.get_node_attributes(forest,'cord')
    load = nx.get_node_attributes(forest,'load')
    
    # Iterate over nodes to record transformers
    for n in sorted(list(forest.nodes())):
        if n in roots:
            dict_tsfr[int(prefix+str(count))] = [cord[n][0],cord[n][1],load[n]]
            rename[n] = int(prefix+str(count))
            tsfrlist.append(int(prefix+str(count)))
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
    edgelist += list(g.edges())
    
    # Iterate over edges to record secondary network
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
    
    # Store end time
    end_time = time.time()
    dict_time[link]=end_time-start_time

#%% Save required data in the temp directory
# This is to account for the program being run at different times
# suffix='-1'
f_network = open(tmpPath+fiscode+"-sec-dist"+suffix+".txt",'w')
f_network.write(network)
f_network.close()
df = pd.DataFrame(edgelist,columns=['source','target'])
df.to_csv(tmpPath+fiscode+'-tsfr-net'+suffix+'.csv',index=False)

df = pd.DataFrame.from_dict(dict_tsfr,orient='index',
                            columns=['long','lat','load'])
df.to_csv(tmpPath+fiscode+'-tsfr-data'+suffix+'.csv')

f=open(tmpPath+fiscode+"-time-stat"+suffix+".txt",'w')
times = '\n'.join(['\t'.join([str(k[0]),str(k[1]),str(dict_time[k])]) \
                   for k in dict_time])+'\n'
f.write(times)
f.close()

print("DONE")
sys.exit(0)


#%% Update saved csv/txt files with new secondary network data
def update_file(dirname,mainfile,updatefile,header=True):
    """
    Updates the csv files which have headers or txt files which have no headers
    
    dirname: name of directory where the files are present
    mainfile: name of main file
    updatefile: name of the file which is to be appended
    header: optional, default is True. If header is present or not
    """
    f = open(dirname+updatefile)
    if header: data_update = f.readlines()[1:]
    else: data_update = f.readlines()
    f.close()
    data = ''.join(data_update)
    f = open(dirname+mainfile,'a')
    f.write(data)
    f.close()
    return

update_file(tmpPath,"161-sec-dist.txt","161-sec-dist-1.txt",header=False)
update_file(tmpPath,"161-time-stat.txt","161-time-stat-1.txt",header=False)
update_file(tmpPath,"161-tsfr-net.csv","161-tsfr-net-1.csv")
update_file(tmpPath,"161-tsfr-data.csv","161-tsfr-data-1.csv")