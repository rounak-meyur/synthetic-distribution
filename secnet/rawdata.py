# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the secondary distribution network in a given county.
"""

import sys,os
from shapely.geometry import Point
workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"


sys.path.append(libpath)
from pyExtractDatalib import GetRoads,GetHomes

#%% Data Extraction
# Extract residence and road network information for the fiscode
fiscode = '121'
roads = GetRoads(inppath,fis=fiscode)
homes = GetHomes(inppath,fis=fiscode)

#%% Plot the data
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D

rescolor = 'crimson'
roadcolor = 'black'

fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)

d = {'edges':[k for k in roads.links],
     'geometry':[roads.links[k]['geometry'] for k in roads.links]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor=roadcolor,linewidth=0.8)

d = {'nodes':[k for k in homes.cord],
     'geometry':[Point(homes.cord[h]) for h in homes.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color=rescolor,markersize=1.0)

ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)


leghands = [Line2D([0], [0], color=roadcolor, markerfacecolor=roadcolor, 
                   marker='o',markersize=0,label='road network'),
            Line2D([0], [0], color='white', markerfacecolor=rescolor, 
                   marker='o',markersize=10,label='residences')]
ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
fig.savefig("{}{}.png".format(figpath,'county-data'),bbox_inches='tight')

#%% Plot load profile
h = range(1,25)
colorlist=['red','green','blue']
fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
for i,home in enumerate(list(homes.profile.keys())[:20]):
    load = homes.profile[home]
    ax.step(h,load,colorlist[i%3])

