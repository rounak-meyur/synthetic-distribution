# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:10:18 2021

Author: Rounak
"""

import sys
import networkx as nx
from geographiclib.geodesic import Geodesic
from itertools import combinations


#%% Functions
def MeasureDistance(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates. The format of each point 
    is (longitude,latitude).
    pt1: (longitude,latitude) of point 1
    pt2: (longitude,latitude) of point 2
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

# Efficiency computation
def compute_efficiency(net):
    nodes = [n for n in net.nodes if net.nodes[n]['label']=='T']
    node_pairs = list(combinations(nodes, 2))
    eff = 0.0
    for pair in node_pairs:
        length = nx.shortest_path_length(net,pair[0],
                                    pair[1],'geo_length')
        distance = MeasureDistance(net.nodes[pair[0]]['cord'],
                               net.nodes[pair[1]]['cord'])
        eff += distance/length if length != 0 else 0.0
    return eff/len(node_pairs)


#%% Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
tmppath = scratchPath + "/temp/"
distpath = tmppath + "osm-prim-network/"
enspath = tmppath + "osm-ensemble/"

sub = sys.argv[1]
opt_net = nx.read_gpickle(distpath+str(sub)+'-prim-dist.gpickle')


#%% Main code
all_eff = [compute_efficiency(opt_net)]
for i in range(20):
    print("Network number: ",i+1)
    graph = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
    all_eff.append(compute_efficiency(graph))

data = sub + '\t' +','.join([str(x) for x in all_eff]) + '\n'
with open(tmppath+"osm-ensemble-efficiency.txt",'a') as f:
    f.write(data)
