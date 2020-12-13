# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates the pickle files for the road+transformer
network. The pickle files are stored in temp directory 

System inputs: fiscode of county
"""

import sys,os
import networkx as nx



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import GetRoads,GetTransformers,GetMappings
from pyVoronoilib import create_master_graph

# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

#%% Get transformers and store them in csv
with open(inpPath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')


for a in areas:
    roads = GetRoads(inpPath,fis=a)
    tsfr = GetTransformers(tmpPath,a)
    links = GetMappings(tmpPath,a)
    g = create_master_graph(roads,tsfr,links)
    nx.write_gpickle(g, tmpPath+'master-graph/'+a+'-graph.gpickle')

