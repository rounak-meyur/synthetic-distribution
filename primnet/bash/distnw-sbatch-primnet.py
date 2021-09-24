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
import time



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)

from pyBuildPrimNetlib import Primary
from pyMiscUtilslib import powerflow,assign_linetype



# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"
dirname = 'osm-prim-master/'


# Extract all substations in the region
with open(tmpPath+"subdata.txt") as f:
    lines = f.readlines()

data = [temp.strip('\n').split('\t') for temp in lines]
subs = {int(d[0]):{"id":int(d[0]),"near":int(d[1]),
                   "cord":[float(d[2]),float(d[3])]} for d in data}


sub=int(sys.argv[1])
sub_data = subs[sub]

# Generate primary distribution network by partitions
start_time = time.time()
P = Primary(sub_data,tmpPath+dirname)


dist_net = P.get_sub_network(tmpPath+'osm-sec-network/',inpPath,tmpPath)
end_time = time.time()
time_taken = end_time - start_time


# Run power flow and store the flows and voltages
powerflow(dist_net)
    
# Assign line types
assign_linetype(dist_net)

with open(tmpPath+'osm-prim-time.txt','a') as f:
    f.write(sys.argv[1]+'\t'+str(time_taken)+'\n')



nx.write_gpickle(dist_net,tmpPath+'osm-prim-network/'+str(sub)+'-dist-net.gpickle')