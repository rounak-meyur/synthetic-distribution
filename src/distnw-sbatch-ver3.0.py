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

workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Display,read_primary,combine_primary_secondary

# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

with open(inpPath+'arealist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')
q_object = Query(csvPath,inpPath)
secnet = q_object.GetAllSecondary(areas)

#%% Get primary network
sub=sys.argv[1]
prim_data = tmpPath+"prim-network/"+sub+"-primary.txt"
prim_net = read_primary(prim_data)

try:
    c = nx.find_cycle(prim_net)
    print("Cycles found:",c)
except:
    print("No cycles found!!!")

dist_net = combine_primary_secondary(prim_net,secnet)

D = Display(dist_net)
D.plot_network(figPath,sub+'-dist-network')
D.save_network(tmpPath,sub+'-dist-network')