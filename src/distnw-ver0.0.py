# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program computes the mapping between homes and road network by
finding the nearest road link to a residential building.
"""

import sys,os
import pandas as pd



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyMapElementslib import MapLink
from pyBuildNetworklib import InvertMap as imap

#%% Initialization of data sets and mappings
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads(level=[1,2,3,4,5])
subs = q_object.GetSubstations()
# MapLink(roads).map_point(homes,path=csvPath,name='home')

print("DONE")

#%% Check the output
df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])

L2Home = imap(H2Link)
# links = [l for l in L2Home if 0<len(L2Home[l])<=70]


#%% Code for explaining the mapping algorithm
def get_neighbors(graph,u,v,hops=2):
    """
    """
    nlist = [u,v]
    for i in range(hops):
        temp = []
        for n in nlist:
            temp.extend(list(graph.neighbors(n)))
        nlist = list(set(temp))
    return nlist


#%% Get data for a single example
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple as nt

h = 511210211001462
link = H2Link[h]
nlist = get_neighbors(roads.graph,link[0],link[1])
interest_graph = nx.subgraph(roads.graph,nlist).copy()
interest_cords = {n:roads.cord[n] for n in nlist}
interest_obj = nt("network",field_names=["graph","cord"])
interest = interest_obj(graph=interest_graph,cord=interest_cords)

color = ['magenta' if e==link or (e[1],e[0])==link else 'black' \
         for e in list(interest.graph.edges())]

xmin = min([c[0] for c in list(interest_cords.values())])-0.001
xmax = max([c[0] for c in list(interest_cords.values())])+0.001
ymin = min([c[1] for c in list(interest_cords.values())])-0.001
ymax = max([c[1] for c in list(interest_cords.values())])+0.001


#%% Display bounding boxes around each edge in the network
from shapely.geometry import box,Point
M = MapLink(interest,radius=0.000005)
lines = M.lines

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
nx.draw_networkx(interest.graph,pos=interest.cord,
                 with_labels=False,node_size=5,edge_color='black')
ax.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=50,c='magenta')
ax.grid(b=True)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.tick_params(axis='both',left=True,bottom=True,labelleft=True,labelbottom=True)


for lobj in lines:
    x1,y1,x2,y2 = lobj[1]
    b = box(x1,y1,x2,y2)
    x,y = list(b.exterior.xy)
    ax.plot(x,y,color='blue',alpha=0.5,linewidth=2)


#%% Display bounding box for the point of interest
pt = Point(homes.cord[h])
radius = 0.0003
b = box(pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)
x,y = list(b.exterior.xy)
ax.plot(x,y,color='red',alpha=0.7,linewidth=2)