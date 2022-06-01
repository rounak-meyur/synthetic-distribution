# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:34:58 2022

Author: Rounak Meyur

Description: Small example of creating synthetic distribution network
"""

import sys
from pyBuildNetlib import GetOSMRoads,GetHomes,GetSubstations,MapOSM,groups

homes = GetHomes('homes-interest.csv')
roads = GetOSMRoads(homes)
subs = GetSubstations('Electric_Substations.csv',homes)
sys.exit(0)

#%% Step 1a: Mapping

# Map the residence to nearest road network link
H2Link = MapOSM(roads).map_point(homes)

# Reverse mapping
L2Home = groups(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]


#%% Step 1b: Secondary network creation


#%% Check
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
d = {'nodes':[n for n in homes.cord],
     'geometry':[Point(homes.cord[n]) for n in homes.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='crimson',markersize=1,alpha=0.8)

d = {'nodes':[n for n in subs.cord],
     'geometry':[Point(subs.cord[n]) for n in subs.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='blue',markersize=50,alpha=0.8)


edgelist = list(roads.edges(keys=True))
d = {'edges':edgelist,
     'geometry':[roads.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor='black',linewidth=1.0,linestyle='dashed',alpha=0.8)

ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)