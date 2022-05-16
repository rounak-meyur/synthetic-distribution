# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import gurobipy as grb
import numpy as np


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
grbpath = workpath + "/out/gurobi/"
shappath = rootpath + "/output/optimal/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet

print("Imported modules")

#%% Functions to create 3 phase network

    

#%% Load a network and save as shape file
sub = 121144
# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

with open(workpath+'/out/phase.txt') as f:
    lines = f.readlines()

phase = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    phase[int(temp[0])] = temp[1]


dist = GetDistNet(distpath,sub)



























