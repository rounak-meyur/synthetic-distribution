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
from pyExtractDatalib import GetSubstations
from pyBuildPrimNetlib import Primary,powerflow



# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

# Extract the arealist
with open(inpPath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

# Extract all substations in the region
subs = GetSubstations(inpPath,areas=areas)


sub=int(sys.argv[1])
substation = nt("local_substation",field_names=["id","cord"])
sub_data = substation(id=sub,cord=subs.cord[sub])

# Generate primary distribution network by partitions
start_time = time.time()
P = Primary(sub_data,tmpPath)
print("Master graph loaded and partitioned")
prim_net = P.get_sub_network(grbpath=tmpPath)
end_time = time.time()
time_taken = end_time - start_time
prim_net = powerflow(prim_net)
with open(tmpPath+'prim-time.txt','a') as f:
    f.write(sys.argv[1]+'\t'+str(time_taken)+'\n')

try:
    c = nx.find_cycle(prim_net)
    print("Cycles found:",c)
except:
    print("No cycles found!!!")

nx.write_gpickle(prim_net,tmpPath+'prim-network/'+str(sub)+'-prim-dist.gpickle')