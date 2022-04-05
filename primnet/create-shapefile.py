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

import shutil
import tempfile
from pathlib import Path

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
shappath = rootpath + "/output/optimal/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet

print("Imported modules")

#%% Load a network and save as shape file
def get_zipped(gdf,filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        localFile = filename
        
        gdf.to_file(filename=temp_dir, driver='ESRI Shapefile')
        
        archiveFile = shutil.make_archive(localFile, 'zip', temp_dir)
        shutil.rmtree(temp_dir)
    return

def create_shapefile(sub,path):
    net = GetDistNet(distpath,sub)
    nodelist = net.nodes
    d = {'node':[n for n in nodelist],
        'label':[net.nodes[n]['label'] for n in nodelist],
         'load':[net.nodes[n]['load'] for n in nodelist],
         'geometry':[Point(net.nodes[n]['cord']) for n in nodelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,path+str(sub)+"-nodelist")
    
    edgelist = net.edges
    d = {'label':[net.edges[e]['label'] for e in edgelist],
         'nodeA':[e[0] for e in edgelist],
         'nodeB':[e[1] for e in edgelist],
         'line_type':[net.edges[e]['type'] for e in edgelist],
         'r':[net.edges[e]['r'] for e in edgelist],
         'x':[net.edges[e]['x'] for e in edgelist],
         'length':[net.edges[e]['length'] for e in edgelist],
         'geometry':[net.edges[e]['geometry'] for e in edgelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,path+str(sub)+"-edgelist")
    return

#%%
# f_done = [int(f.strip('-prim-dist.gpickle')) for f in os.listdir(distpath)]

sublist = [121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

for s in sublist:
    create_shapefile(s,shappath)
    print("Network created for",s)




