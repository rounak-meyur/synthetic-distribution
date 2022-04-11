# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:19:04 2022

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist. This is done for 
all the ensemble of networks
"""

import sys,os
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

import shutil
import tempfile
from pathlib import Path

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
enspath = workpath + "/out/osm-ensemble/"
shappath = rootpath + "/output/ensemble/"

sys.path.append(libpath)
from pyMiscUtilslib import assign_linetype

print("Imported modules")

#%% Load a network and save as shape file
def GetEnsNet(path,sub,i):
    return nx.read_gpickle(path+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')

def get_zipped(gdf,filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        localFile = filename
        
        gdf.to_file(filename=temp_dir, driver='ESRI Shapefile')
        
        archiveFile = shutil.make_archive(localFile, 'zip', temp_dir)
        shutil.rmtree(temp_dir)
    return

def create_ens_shapefile(sub,src,i,dest):
    net = GetEnsNet(src,sub,i)
    assign_linetype(net)
    nodelist = net.nodes
    d = {'node':[n for n in nodelist],
        'label':[net.nodes[n]['label'] for n in nodelist],
         'load':[net.nodes[n]['load'] for n in nodelist],
         'geometry':[Point(net.nodes[n]['cord']) for n in nodelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,dest+str(sub)+"-nodelist-"+str(i+1))
    
    edgelist = net.edges
    d = {'label':[net.edges[e]['label'] for e in edgelist],
         'nodeA':[e[0] for e in edgelist],
         'nodeB':[e[1] for e in edgelist],
         'line_type':[net.edges[e]['type'] for e in edgelist],
         'r':[net.edges[e]['r'] for e in edgelist],
         'x':[net.edges[e]['x'] for e in edgelist],
         'length':[net.edges[e]['geo_length'] for e in edgelist],
         'geometry':[net.edges[e]['geometry'] for e in edgelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,dest+str(sub)+"-edgelist-"+str(i+1))
    return

#%%
# f_done = [int(f.strip('-prim-dist.gpickle')) for f in os.listdir(distpath)]

sublist = [121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]


for s in sublist:
    for i in range(20):
        create_ens_shapefile(s,enspath,i,shappath)
        print("Network "+str(i+1)+" created for "+str(s))

