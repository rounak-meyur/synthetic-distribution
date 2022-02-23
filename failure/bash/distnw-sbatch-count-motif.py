# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:56:39 2022

Author: Rounak Meyur

Description: This program computes the number of 4-node motifs for different
primary networks
"""

import sys,os
import networkx as nx
import itertools
import threading

workpath = os.getcwd()
libpath = workpath + "/Libraries/"


scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
tmppath = scratchpath + "/temp/"
distpath = scratchpath + "/temp/osm-prim-network/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet


#%% User defined functons
def count_motifs(g,count,i):
    nodelist = [n for n in g.nodes if g.nodes[n]['label']!='H']
    for quad in itertools.combinations(nodelist,4):
        sub_g = nx.subgraph(g,quad)
        if sub_g.number_of_edges() == 3:
            if nx.is_isomorphic(sub_g, nx.path_graph(4)):
                count[i]['path'] += 1
            else:
                count[i]['star'] += 1
    return



def counter(g):
    threads = []
    motif_counter = {}
    for i,feed_nodes in enumerate(nx.connected_components(g)):
        # create the thread
        motif_counter[i] = {'path':0,'star':0}
        g_feeder = g.subgraph(feed_nodes)
        threads.append(threading.Thread(target=count_motifs, 
                                        args=(g_feeder,motif_counter,i)))
        threads[-1].start() # start the thread we just created

    # wait for all threads to finish
    for t in threads:
        t.join()
    return motif_counter


#%% Main code
sub = int(sys.argv[1])
dist = GetDistNet(distpath,sub)
dist.remove_node(sub)


C = counter(dist)
MOTIF_COUNTER = {'path':sum([C[i]['path'] for i in C]),
                 'star':sum([C[i]['star'] for i in C])}
print(MOTIF_COUNTER)

data = str(sub)+"\t"+str(MOTIF_COUNTER['path'])+" "+str(MOTIF_COUNTER['star'])+"\n"
with open(tmppath+"motif-count.txt",'a') as f:
    f.write(data)