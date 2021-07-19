# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
shappath = workpath + "/out/prim-geom/"


sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,assign_linetype
# from pyBuildPrimNetlib import powerflow

print("Imported modules")

#%% Load a network and save as shape file
sub  =121144
net = GetDistNet(distpath,sub)
assign_linetype(net)


#%%
d = {'label':[net.nodes[n]['label'] for n in net],
     'load':[net.nodes[n]['load'] for n in net],
     'voltage':[net.nodes[n]['voltage'] for n in net],
     'geometry':[Point(net.nodes[n]['cord']) for n in net]}
gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
gdf.to_file(shappath+sub+"-nodelist.shp")


































