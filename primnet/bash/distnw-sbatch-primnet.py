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
import time

def remove_cycle(graph):
    try:
        cycle = nx.find_cycle(graph)
        print("Cycles found:",cycle)
        nodes = list(set([c[0] for c in cycle] + [c[1] for c in cycle]))
        nodes = [n for n in nodes if graph.nodes[n]['label']=='R']
        print("Number of nodes:",graph.number_of_nodes())
        print("Removing cycles...")
        for n in nodes:
            graph.remove_node(n)
        print("After removal...Number of nodes:",graph.number_of_nodes())
        print("Number of comps.",nx.number_connected_components(graph))
        remove_cycle(graph)
    except:
        print("No cycles found!!!")
        return



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import GetSubstations
from pyBuildPrimNetlib import Primary,create_final_network



# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"
dirname = 'osm-prim-master/'


# Extract all substations in the region
subs = GetSubstations(inpPath)


sub=int(sys.argv[1])
substation = nt("local_substation",field_names=["id","cord"])
sub_data = substation(id=sub,cord=subs.cord[sub])

# Generate primary distribution network by partitions
start_time = time.time()
P = Primary(sub_data,tmpPath+dirname)


prim_net = P.get_sub_network(grbpath=tmpPath)
end_time = time.time()
time_taken = end_time - start_time

remove_cycle(prim_net)

synth_net = create_final_network(prim_net)
with open(tmpPath+'osm-prim-time.txt','a') as f:
    f.write(sys.argv[1]+'\t'+str(time_taken)+'\n')



nx.write_gpickle(synth_net,tmpPath+'osm-prim-network/'+str(sub)+'-prim-dist.gpickle')