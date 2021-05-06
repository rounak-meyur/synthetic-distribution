# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:25:52 2021

Author: Rounak Meyur

Description: Extract the road network for a given county/independent city of
Virginia. Explore the dataset and other stuff related to it.
"""

import sys,os
import geopandas as gpd

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
# sys.exit(0)
sys.path.append(libpath)
from pyExtractDatalib import GetOSMRoads,GetHomes

fiscode = '161'
roads = GetOSMRoads(inppath,fis=fiscode)
roads1 = GetOSMRoads(inppath,fis='121')
sys.exit(0)
homes = GetHomes(inppath,fis=fiscode)

#%% Check by plotting data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point

rescolor = 'crimson'
roadcolor = 'black'

fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)

edgelist = list(roads.edges(keys=True))
d = {'edges':edgelist,
     'geometry':[roads.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor=roadcolor,linewidth=0.8)

# d = {'nodes':[k for k in homes.cord],
#       'geometry':[Point(homes.cord[h]) for h in homes.cord]}
# df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
# df_nodes.plot(ax=ax,color=rescolor,markersize=1.0)

edgelist = list(roads1.edges(keys=True))
d = {'edges':edgelist,
     'geometry':[roads1.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor=roadcolor,linewidth=0.8)

# d = {'nodes':[k for k in homes.cord],
#       'geometry':[Point(homes.cord[h]) for h in homes.cord]}
# df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
# df_nodes.plot(ax=ax,color=rescolor,markersize=1.0)

nodes1 = list(roads.nodes())
nodes2 = list(roads1.nodes())
nodelist = list(set(nodes1).intersection(set(nodes2)))

ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)


# leghands = [Line2D([0], [0], color=roadcolor, markerfacecolor=roadcolor, 
#                    marker='o',markersize=0,label='road network'),
#             Line2D([0], [0], color='white', markerfacecolor=rescolor, 
#                    marker='o',markersize=10,label='residences')]
# ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
# fig.savefig("{}{}.png".format(figpath,fiscode+'-OSM-county-data'),
#             bbox_inches='tight')