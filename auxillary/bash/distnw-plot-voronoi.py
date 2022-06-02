# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:50:33 2020

@author: rounak
"""

# Get list of substations
import os,sys
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

workpath = os.getcwd()
libpath = workpath + "/Libraries/"
sys.path.append(libpath)
from pyExtractDatalib import GetMaster


# Load scratchpath
scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
tmppath = scratchpath + "/temp/"
figpath = scratchpath + "/figs/"
mastpath = tmppath + "osm-prim-master/"


sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]



 

with open(tmppath+"subdata.txt") as f:
    lines = f.readlines()
data = [temp.strip('\n').split('\t') for temp in lines]
subs = {int(d[0]):{"id":int(d[0]),"near":int(d[1]),
                    "cord":[float(d[2]),float(d[3])]} for d in data}
subdata = {s:subs[s] for s in sublist}


colors = sns.color_palette()


fig = plt.figure(figsize=(40,40), dpi=72)
ax = fig.add_subplot(111)

for i,sub in enumerate(sublist):
    mastnet = GetMaster(mastpath,sub)
    
    d = {'nodes':[sub],
         'geometry':[Point(subs[sub]['cord'])]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=colors[i%len(colors)],markersize=2000,alpha=1.0)
    
    d = {'nodes':[n for n in mastnet],
         'geometry':[Point(mastnet.nodes[n]['cord']) for n in mastnet]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=colors[i%len(colors)],markersize=25,alpha=1.0)


ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
fig.savefig("{}{}.png".format(figpath,'51121-voronoi'),bbox_inches='tight')





