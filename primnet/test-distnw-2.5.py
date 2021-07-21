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
def create_shapefile(sub,path):
    net = GetDistNet(distpath,sub)
    assign_linetype(net)
    nodelist = net.nodes
    d = {'label':[net.nodes[n]['label'] for n in nodelist],
         'load':[net.nodes[n]['load'] for n in nodelist],
         'voltage':[net.nodes[n]['voltage'] for n in nodelist],
         'geometry':[Point(net.nodes[n]['cord']) for n in nodelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    gdf.to_file(path+str(sub)+"-nodelist.shp")
    
    
    edgelist = net.edges
    d = {'label':[net.edges[e]['label'] for e in edgelist],
         'line_type':[net.edges[e]['type'] for e in edgelist],
         'r':[net.edges[e]['r'] for e in edgelist],
         'x':[net.edges[e]['x'] for e in edgelist],
         'length':[net.edges[e]['geo_length'] for e in edgelist],
         'flow':[net.edges[e]['flow'] for e in edgelist],
         'geometry':[net.edges[e]['geometry'] for e in edgelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    gdf.to_file(path+str(sub)+"-edgelist.shp")
    return

#%%
f_done = [int(f.strip('-prim-dist.gpickle')) for f in os.listdir(distpath)]
for s in f_done:
    create_shapefile(s,shappath)
    print("Network created for",s)




