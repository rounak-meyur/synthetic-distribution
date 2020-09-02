# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program computes the mapping between homes and road network by
finding the nearest road link to a residential building.
"""

import sys,os
import pandas as pd
import networkx as nx
from collections import namedtuple as nt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyMapElementslib import MapLink
# from pyBuildNetworklib import InvertMap as imap

def display_data(ax,roads,homes,colors=['blue','green'],showhome=False):
    """
    Displays the road network and residences in the given region.
    Parameters
    ----------
    fig   : TYPE: matplotlib axis object
        DESCRIPTION.
    roads : TYPE: named tuple with road network data
        DESCRIPTION.
    homes : TYPE: named tuple with residential data
        DESCRIPTION.

    Returns
    -------
    None.
    """
    nx.draw_networkx(roads.graph,node_size=0.1,color=colors[0],with_labels=False,
                     ax=ax,pos=roads.cord,edge_color=colors[0],style="dashed",width=0.5)
    hpts = list(homes.cord.values())
    xpts = [h[0] for h in hpts]
    ypts = [h[1] for h in hpts]
    if showhome: ax.scatter(xpts,ypts,c=colors[1],s=0.5)
    return ax


#%% Initialization of data sets and mappings
fis = '121'
q_object = Query(csvPath,inpPath)
homes,roads = q_object.GetDataset(fis=fis)
print("Total number of links:",roads.graph.number_of_edges(),
      "\nTotal number of homes:",len(homes.cord))

sys.exit(0)

MapLink(roads).map_point(homes,path=csvPath,fiscode=fis)

print("DONE")
sys.exit(0)

#%% Plot of road network and residences
# Plot the road network and residences for verification
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)

cols = ['royalblue','lightgreen']
homes,roads = q_object.GetDataset(fislist=['161'])
ax = display_data(ax,roads,homes,colors=cols)
leglines = [Line2D([0], [0], color=cols[0], markerfacecolor=cols[0], marker='o',
                   markersize=15,linestyle='solid'),
                Line2D([0], [0], color='white', markerfacecolor=cols[1], marker='o',
                       markersize=15)]
leglabels = ['road links: Roanoke county','residences: Roanoke county']

cols = ['black','yellow']
homes,roads = q_object.GetDataset(fislist=['770'])
ax = display_data(ax,roads,homes,colors=cols)
leglines = leglines + [Line2D([0], [0], color=cols[0], markerfacecolor=cols[0], marker='o',
                    markersize=15,linestyle='solid'),
                Line2D([0], [0], color='white', markerfacecolor=cols[1], marker='o',
                        markersize=15)]
leglabels = leglabels + ['road links: Roanoke city','residences: Roanoke city']

cols = ['crimson','cyan']
homes,roads = q_object.GetDataset(fislist=['775'])
ax = display_data(ax,roads,homes,colors=cols)
leglines = leglines + [Line2D([0], [0], color=cols[0], markerfacecolor=cols[0], marker='o',
                    markersize=15,linestyle='solid'),
                Line2D([0], [0], color='white', markerfacecolor=cols[1], marker='o',
                        markersize=15)]
leglabels = leglabels + ['road links: Salem city','residences: Salem city']

ax.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)
ax.legend(leglines,leglabels,loc='best',ncol=1,prop={'size': 15})


#%% Check the output
# Import further necessary modules
from shapely.geometry import box,Point,LineString
h = 511210211001462
nlist = [889535586, 889535587, 335455019, 171517423, 171517458, 
	171517427, 171517461, 24866230, 922394295, 24866232,171517462]

# Define a named tuple interest to store a road network subgraph of interest
interest_graph = nx.subgraph(roads.graph,nlist).copy()
interest_graph.add_edges_from([(889535586,171517461),(24866230,171517458),
                               (171517461,24866230),(24866230,889535587),
                               (24866232,335455019),(24866232,171517458)])
elist = list(interest_graph.edges())
interest_cords = {n:roads.cord[n] for n in nlist}
interest_links = {e:{'geometry':LineString((interest_cords[e[0]],
                                            interest_cords[e[1]]))} \
                  for e in elist}
interest_obj = nt("network",field_names=["graph","cord","links"])
interest = interest_obj(graph=interest_graph,cord=interest_cords,
                        links=interest_links)
xmin = min([c[0] for c in list(interest_cords.values())])-0.001
xmax = max([c[0] for c in list(interest_cords.values())])+0.001
ymin = min([c[1] for c in list(interest_cords.values())])-0.001
ymax = max([c[1] for c in list(interest_cords.values())])+0.001



# Define an object for the class MapLink with radius of 5e-4
M = MapLink(interest,radius=0.0005)
lines = M.lines
pt = Point(homes.cord[h])
radius = 0.0003
font = 25
mark = 15
#%% Step 1 figure
# Display the residential node along with the links
# fig1 = plt.figure(figsize=(14,12))

# ax1 = fig1.add_subplot(111)
# nx.draw_networkx(interest.graph,pos=interest.cord,ax=ax1,node_color='blue',
#                  with_labels=False,node_size=100,edge_color='black',width=3)
# ax1.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=200,c='magenta')

# ax1.set_xlim(xmin,xmax)
# ax1.set_ylim(ymin,ymax)
# ax1.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)
# ax1.set_title("Get all road links and point of interest",fontsize=font)
# # ax1.set_xlabel("Longitude",fontsize=font)
# # ax1.set_ylabel("Latitude",fontsize=font)
# leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='o',
#                    markersize=mark,linestyle='dashed'),
#                 Line2D([0], [0], color='white', markerfacecolor='magenta', marker='D',
#                        markersize=mark)]
# ax1.legend(leglines,['road links','residential building'],
#           loc='best',ncol=1,prop={'size': mark})
# fig1.savefig("{}{}.png".format(figPath,'step0'),bbox_inches='tight')

#%% Figure 2
# Display the original network
# fig2 = plt.figure(figsize=(14,12))
# ax2 = fig2.add_subplot(111)
# nx.draw_networkx(interest.graph,pos=interest.cord,ax=ax2,node_color='blue',
#                  with_labels=False,node_size=100,edge_color='black',width=3)
# ax2.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=200,c='magenta')

# ax2.set_xlim(xmin,xmax)
# ax2.set_ylim(ymin,ymax)
# ax2.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)


# # Display the bounding boxes around each road network link
# for lobj in lines:
#     x1,y1,x2,y2 = lobj[1]
#     b = box(x1,y1,x2,y2)
#     x,y = list(b.exterior.xy)
#     ax2.plot(x,y,color='blue',alpha=1,linewidth=3)

# # Display the bounding box for the residential coordinate
# b = box(pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)
# x,y = list(b.exterior.xy)
# ax2.plot(x,y,color='red',alpha=1,linewidth=2)

# # ax2.set_title("Step 1: Draw bounding boxes around each link and the point",fontsize=font)
# # ax2.set_xlabel("Longitude",fontsize=font)
# # ax2.set_ylabel("Latitude",fontsize=font)

# leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='o',
#                    markersize=mark,linestyle='dashed'),
#                 Line2D([0], [0], color='white', markerfacecolor='magenta', marker='D',
#                        markersize=mark),
#            Patch(edgecolor='blue',fill=False),Patch(edgecolor='red',fill=False)]
# ax2.legend(leglines,['road links','residential building','bounding box for links', 
#                      'bounding box for residence'],
#           loc='best',ncol=2,prop={'size': mark})
# fig2.savefig("{}{}.png".format(figPath,'step1'),bbox_inches='tight')

#%% Figure 3
# Find intersections of bounding boxes
pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
matches = M.idx.intersect(pt_bounds)

match_link = [l[2] for i,l in enumerate(lines) if i in matches]
color = ['seagreen' if e in match_link or (e[1],e[0]) in match_link else 'black' \
         for e in list(interest.graph.edges())]

# Display the residential node along with the links
fig3 = plt.figure(figsize=(14,12))
ax3 = fig3.add_subplot(111)
nx.draw_networkx(interest.graph,pos=interest.cord,ax=ax3,nodelist = nlist,
                 node_color='blue',with_labels=False,node_size=100,
                 edge_color=color,width=3)
ax3.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=200,c='magenta')

ax3.set_xlim(xmin,xmax)
ax3.set_ylim(ymin,ymax)
ax3.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)

# Display the bounding box for the residential coordinate
pt = Point(homes.cord[h])
b = box(pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)
x,y = list(b.exterior.xy)
ax3.plot(x,y,color='red',alpha=0.7,linewidth=3)

# Draw the links which have bounding box intersections
for i,lobj in enumerate(lines):
    x1,y1,x2,y2 = lobj[1]
    b = box(x1,y1,x2,y2)
    x,y = list(b.exterior.xy)
    if i in matches:
        ax3.plot(x,y,color='green',alpha=1,linewidth=3)
    else:
        ax3.plot(x,y,color='blue',alpha=0.6,linewidth=3)

# ax3.set_title("Step 2: Short-list links with intersecting bounding box",fontsize=font)
# ax3.set_xlabel("Longitude",fontsize=font)
# ax3.set_ylabel("Latitude",fontsize=font)

leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='o',
                   markersize=mark,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='magenta', marker='D',
                       markersize=mark),
           Patch(edgecolor='green',fill=False),Patch(edgecolor='blue',fill=False),
           Patch(edgecolor='red',fill=False)]
ax3.legend(leglines,['road links','residential building',
                     'bounding box for nearby links',
                     'bounding box for further links',
                     'bounding box for residence'],
          loc='best',ncol=1,prop={'size': mark})
fig3.savefig("{}{}.png".format(figPath,'stepA'),bbox_inches='tight')

#%% Figure 4
# Find the nearest link among the short-listed links whose bounding boxes have intersected.
closest_path = min(matches, key=lambda i: M.lines[i][0].distance(pt))
nearest = M.lines[closest_path][-1]
print("The nearest link is :",nearest)

# Display the residential node along with the links
color = ['magenta' if e ==  nearest or (e[1],e[0]) == nearest else 'black' \
         for e in list(interest.graph.edges())]

fig4 = plt.figure(figsize=(14,12))
ax4 = fig4.add_subplot(111)
nx.draw_networkx(interest.graph,pos=interest.cord,ax=ax4,node_color='blue',
                 with_labels=False,node_size=100,edge_color=color,width=3)
ax4.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=200,c='magenta')

ax4.set_xlim(xmin,xmax)
ax4.set_ylim(ymin,ymax)
ax4.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)

# ax4.set_title("Step 3: Find the nearest among short-listed links",fontsize=font)
# ax4.set_xlabel("Longitude",fontsize=font)
# ax4.set_ylabel("Latitude",fontsize=font)

leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='o',
                   markersize=mark,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='magenta', marker='D',
                       markersize=mark),
           Line2D([0], [0], color='magenta', markerfacecolor='blue', marker='o',
                  markersize=0)]
ax4.legend(leglines,['road links','residential building','nearest road links'],
          loc='best',ncol=1,prop={'size': mark})
fig4.savefig("{}{}.png".format(figPath,'stepB'),bbox_inches='tight')
