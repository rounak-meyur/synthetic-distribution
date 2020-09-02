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



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Primary
from pyBuildNetworklib import Display


# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

q_object = Query(csvPath,inpPath)
subs = q_object.GetAllSubstations()


#%% Get subgraph
sub=int(sys.argv[1])
substation = nt("local_substation",field_names=["id","cord"])
sub_data = substation(id=sub,cord=subs.cord[sub])

# Generate primary distribution network by partitions
master_graph_path = tmpPath+'prim-master/'
start_time = time.time()
P = Primary(sub_data,master_graph_path,max_node=700)
prim_net = P.get_sub_network(flowmax=1000,feedermax=5,grbpath=tmpPath)
end_time = time.time()
time_taken = end_time - start_time
with open(tmpPath+'prim-time.txt','a') as f:
    f.write(sys.argv[1]+'\t'+str(time_taken)+'\n')

D = Display(prim_net)
D.save_network(tmpPath+'prim-network/',str(sub)+'-primary')
D.plot_primary(figPath,str(sub)+'-primary')

try:
    c = nx.find_cycle(prim_net)
    print("Cycles found:",c)
except:
    print("No cycles found!!!")