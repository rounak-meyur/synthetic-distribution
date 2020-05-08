# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:12:30 2020

Author: Rounak Meyur
Description: This program reads the road link geometry and the link files to get the
total structure of the road network.
    a. The total road network structure would be read as a geopandas dataframe.
    b. The geopandas dataframe has a column for the geometry.
    c. Final objective is to place transformers at regular intervals on the geopandas
    data object.
"""

import sys,os
import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

workPath = os.getcwd()
inpPath = workPath + "/input/nrv/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/nrv/"
tmpPath = workPath + "/temp/nrv/"

sys.path.append(libPath)

corefile = "core-link-file-Roanoke-VA.txt"
linkgeom = "link-file-Roanoke-VA.txt"
nodegeom = "node-geometry-Roanoke-VA.txt"



datalink = {}
edgelist = []

with open(inpPath+corefile) as file:
    for temp in file.readlines()[1:]:
        edge = tuple([int(x) for x in temp.strip("\n").split("\t")[0:2]])
        lvl = int(temp.strip("\n").split("\t")[-1])
        if (edge not in edgelist) and ((edge[1],edge[0]) not in edgelist):
            edgelist.append(edge)
            datalink[edge] = {'level':lvl,'geometry':None}
            
roadcord = {}
with open(inpPath+nodegeom) as file:
    for temp in file.readlines()[1:]:
        data = temp.strip('\n').split('\t')
        roadcord[int(data[0])]=[float(data[1]),float(data[2])]




with open(inpPath+linkgeom) as file:
    for temp in file.readlines()[1:]:
        data = temp.strip("\n").split("\t")
        edge = tuple([int(x) for x in data[3:5]])
        pts = [tuple([float(x) for x in pt.split(' ')]) \
                for pt in data[10].lstrip('MULTILINESTRING((').rstrip('))').split(',')]
        geom = LineString(pts)
        if (edge in edgelist):
            datalink[edge]['geometry']=geom
        elif ((edge[1],edge[0]) in edgelist):
            datalink[(edge[1],edge[0])]['geometry']=geom
        else:
            print(','.join([str(x) for x in list(edge)])+": not in edgelist")
    
for edge in datalink:
    if datalink[edge]['geometry']==None:
        pts = [tuple(roadcord[r]) for r in list(edge)]
        geom = LineString(pts)
        datalink[edge]['geometry'] = geom