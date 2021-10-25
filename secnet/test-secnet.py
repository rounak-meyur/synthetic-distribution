# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:19:25 2021

@author: Rounak Meyur
Description: Test figures for secondary network
"""

# Import modules
import sys,os
from shapely.geometry import box,Point
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
print("Modules imported")

#%% Directories

# Define directories and libraries
workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
outpath = workpath + "/out/"
figpath = workpath + "/figs/"
sys.path.append(libpath)

# Import further necessary modules from libraries
from pyExtractDatalib import GetHomes
from pyBuildSecNetlib import MapOSM

# Get data for the test
roads = nx.read_gpickle(outpath+'cache/121-road.gpickle')
homes = GetHomes(inppath,'121')


#%% Data for interest 

# Homes and road nodes
h = 511210211001462
residence = Point(homes.cord[h])
buffer = residence.buffer(0.0015, cap_style = 3)

link_geom = nx.get_edge_attributes(roads,'geometry')
elist = [e for e in link_geom if link_geom[e].intersects(buffer)]

# Define a named tuple interest to store a road network subgraph of interest
interest = roads.edge_subgraph(elist).copy()

M = MapOSM(interest,radius=0.0005)
links = M.links
radius = 0.00025
matches = M.idx.intersect((residence.x-radius, residence.y-radius, 
                          residence.x+radius, residence.y+radius))
match_link = [l[2] for i,l in enumerate(links) if i in matches]
closest_path = min(matches, key=lambda i: M.links[i][0].distance(residence))
nearest = M.links[closest_path][-1]

#%% Functions to plot mapping explanation
font = 15
mark = 10

def draw_base(ax,interest,color='black'):
    xcord = [interest.nodes[n]['x'] for n in interest]
    ycord = [interest.nodes[n]['y'] for n in interest]

    xmin = min(xcord)-0.001
    xmax = max(xcord)+0.001
    ymin = min(ycord)-0.001
    ymax = max(ycord)+0.001
    
    
    # Get the dataframe for node geometries
    d = {'nodes':[n for n in interest],
         'geometry':[Point(interest.nodes[n]['x'],interest.nodes[n]['y']) \
                     for n in interest]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color="blue",markersize=50)
    
    # Get the dataframe for edge geometries
    d = {'edges':[e for e in interest.edges(keys=True)],
         'geometry':[interest.edges[e]['geometry'] \
                     for e in interest.edges(keys=True)]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=3.0)
    
    ax.scatter(homes.cord[h][0],homes.cord[h][1],marker='D',s=50,c='magenta')

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.tick_params(axis='both',left=False,bottom=False,labelleft=False,labelbottom=False)

    # Update legends
    leglines = [Line2D([0], [0], color='black', markerfacecolor='blue', marker='o',
                markersize=mark,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='magenta', marker='D',
                markersize=mark)]
    labels = ['road links','residential building']
    return leglines,labels

def link_box(ax,lines,leglines,labels):
    """
    Display the bounding boxes around each road network link.
    """
    # Display bounding box for road links
    for lobj in lines:
        x1,y1,x2,y2 = lobj[1]
        b = box(x1,y1,x2,y2)
        x,y = list(b.exterior.xy)
        ax.plot(x,y,color='blue',alpha=1,linewidth=2)
    
    # Update legends and labels
    leglines += [Patch(edgecolor='blue',fill=False)]
    labels += ['bounding box for links']
    return leglines,labels

def pt_box(ax,pt,leglines,labels):
    """
    Display the bounding boxes around the residence node.
    """
    # Display the bounding box for the residential coordinate
    b = box(pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)
    x,y = list(b.exterior.xy)
    ax.plot(x,y,color='red',alpha=1,linewidth=2)
    
    # Update legends and labels
    leglines += [Patch(edgecolor='red',fill=False)]
    labels += ['bounding box for residence']
    return leglines,labels

def match_box(ax,lines,matches,leglines,labels):
    # Display bounding box for road links
    for i,lobj in enumerate(lines):
        x1,y1,x2,y2 = lobj[1]
        b = box(x1,y1,x2,y2)
        x,y = list(b.exterior.xy)
        if i in matches:
            ax.plot(x,y,color='green',alpha=1,linewidth=2)
        else:
            ax.plot(x,y,color='blue',alpha=0.3,linewidth=2)
    
    # Update legends and labels
    leglines += [Patch(edgecolor='green',fill=False),Patch(edgecolor='blue',fill=False)]
    labels += ['bounding box for nearby links','bounding box for further links']
    return leglines,labels


#%% Plot the steps of mapping residence to nearest link
fig = plt.figure(figsize=(16,10))

ax1 = fig.add_subplot(221)
leglines,labels = draw_base(ax1,interest)
ax1.set_title("Get all road links and point of interest",fontsize=font)
ax1.legend(leglines,labels,loc='best',ncol=1,prop={'size': mark})

ax2 = fig.add_subplot(222)
leglines,labels = draw_base(ax2,interest)
leglines,labels = link_box(ax2,links,leglines,labels)
leglines,labels = pt_box(ax2,residence,leglines,labels)
ax2.set_title("Draw bounding boxes around each link and the point",fontsize=font)
ax2.legend(leglines,labels,loc='best',ncol=2,prop={'size': mark})

ax3 = fig.add_subplot(223)
color = ['seagreen' if e in match_link or (e[1],e[0],e[2]) in match_link else 'black' \
         for e in list(interest.edges(keys=True))]
leglines,labels = draw_base(ax3,interest,color)
leglines,labels = pt_box(ax3,residence,leglines,labels)
leglines,labels = match_box(ax3,links,matches,leglines,labels)
ax3.set_title("Short-list links with intersecting bounding box",fontsize=font)
ax3.legend(leglines,labels,loc='best',ncol=1,prop={'size': mark})

ax4 = fig.add_subplot(224)
color = ['magenta' if e == nearest or (e[1],e[0],e[2]) == nearest else 'black' \
         for e in list(interest.edges(keys=True))]
leglines,labels = draw_base(ax4,interest,color)
leglines += [Line2D([0], [0], color='magenta', markerfacecolor='blue', marker='o',
                  markersize=0)]
labels += ["nearest road link"]
ax4.set_title("Find the nearest among short-listed links",fontsize=font)
ax4.legend(leglines,labels,loc='best',ncol=1,prop={'size': mark})

fig.savefig("{}{}.png".format(figpath,'mapping-steps'),bbox_inches='tight')


#%% Secondary Network Creation
from pyBuildSecNetlib import generate_optimal_topology as generate

fiscode = '121'
df_hmap = pd.read_csv(outpath+'osm-sec-network/'+fiscode+'-home2OSM.csv')
H2Link = dict([(t.hid, (t.source, t.target,t.key)) for t in df_hmap.itertuples()])
with open(outpath+'osm-sec-network/'+fiscode+'-link2home.txt','r') as f:
    linkdata = f.readlines()
dictlink = {}
for temp in linkdata:
    key = tuple([int(t) for t in temp.strip('\n').split('\t')[0].split(',')])
    value = [int(x) for x in temp.strip('\n').split('\t')[1].split(',')]
    dictlink[key]=value


#%% Create secondary network
link = list(dictlink.keys())[3]
homelist = dictlink[link]

linkgeom = roads.edges[link]['geometry']


dict_homes = {h:{'cord':homes.cord[h],'load':homes.average[h]} for h in homelist}
forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=10,
                        tsfr_max=25,heuristic=None,path=outpath)


#%% Display the example
from pyGeometrylib import Link
def display_link(link,homelist,roads,homes,forest,roots,path,name):
    """
    Displays the road link with the residences mapped to it. Also provides the option
    to display the probable locations of transformers along the link.

    Parameters
    ----------
    link : tuple of the terminal node IDs
        the road network link of interest.
    homelist : list of residence IDs
        list of homes mapped to the road network link.
    roads : named tuple of type road
        information related to the road network.
    homes : named tuple of type home
        information related to residences

    Returns
    -------
    None.

    """
    # Figure initialization
    fig = plt.figure(figsize=(28,8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    linegeom = roads.edges[link]['geometry']
    line = linegeom.xy
    link_line = Link(linegeom)
    tsfr = link_line.InterpolatePoints(50)

    # Plot the road network link with no interpolated transformer
    ax1.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    ax1.scatter([homes.cord[h][0] for i,h in enumerate(homelist)],
               [homes.cord[h][1] for i,h in enumerate(homelist)],
               c='red',s=25.0,marker='*')
    ax1.set_title("Residences mapped to a road link",fontsize=20)
    leglines = [Line2D([0], [0], color='black', markerfacecolor='c', marker='*',
                       markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',
                       markersize=15)]
    labels = ['road link','residences mapped']
    ax1.legend(leglines,labels,loc='best',ncol=1,prop={'size': 15})
    
    # Plot the road network with interpolated transformers
    ax2.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    ax2.scatter([homes.cord[h][0] for i,h in enumerate(homelist)],
               [homes.cord[h][1] for i,h in enumerate(homelist)],
               c='red',s=25.0,marker='*')
    ax2.scatter([t.x for t in tsfr],[t.y for t in tsfr],
               c='green',s=60.0,marker='*')
    leglines += [Line2D([0], [0], color='white', markerfacecolor='green', 
                        marker='*',markersize=15)]
    labels += ['possible transformers']
    ax2.set_title("Probable transformers along road link",fontsize=20)
    ax2.legend(leglines,labels,loc='best',ncol=1,prop={'size': 15})
    
    # Plot the road network with secondary network
    ax3.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    nodelist = list(forest.nodes())
    colors = ['red' if n not in roots else 'green' for n in nodelist]
    pos_nodes = nx.get_node_attributes(forest,'cord')
    
    # Draw network
    nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                     ax=ax3,edge_color='crimson',width=1,with_labels=False,
                     node_size=20.0,node_shape='*',node_color=colors)
    
    # Other updates
    leglines = [Line2D([0], [0], color='black', markerfacecolor='green', marker='*',
                       markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='*',
                       markersize=0),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='*',
                       markersize=15),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',
                       markersize=15)]
    labels = ['road link','secondary network','local transformers','residences']
    ax3.legend(leglines,labels,loc='best',ncol=1,prop={'size': 15})
    ax3.set_title("Secondary network creation for road link",fontsize=20)
    
    # Final adjustments
    ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax3.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Save the figure
    fig.savefig("{}{}.png".format(path,name),bbox_inches='tight')
    return

display_link(link,homelist,roads,homes,forest,roots,figpath,'secnet-link-home-tsfr')

#%% Computational Time Stats

import numpy as np
with open(inppath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

time_stat = []
for area in areas:
    with open(outpath+'sec-network/'+area+'-time-stat.txt','r') as f:
        times = [[float(x) for x in lines.strip('\n').split('\t')] \
                 for lines in f.readlines()]
    time_stat.extend(times)


xpts = [time[0] for time in time_stat]
ypts = [time[1] for time in time_stat]
_, xbins = np.histogram(np.log10(np.array(xpts)),bins=8)
ypts_dict = {'grp'+str(i+1):[np.log10(time[1]) for time in time_stat \
                             if 10**xbins[i]<=time[0]<=10**xbins[i+1]] \
             for i in range(len(xbins)-1)}

xmean = [int(round(10**((xbins[i]+xbins[i+1])/2))) for i in range(len(xbins)-1)]
xmean = [1,3,7,15,31,63,127,255]
ytick_old = np.linspace(-3,4,num=8)
ytick_new = 10**ytick_old

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax.boxplot(ypts_dict.values())
ax.set_xticklabels(xmean)
ax.set_yticklabels(ytick_new)
ax.set_xlabel('Number of residences to be connected',fontsize=15)
ax.set_ylabel('Time (in seconds) to create secondary network',
              fontsize=15)
fig.savefig("{}{}.png".format(figpath,'secnet-time'),bbox_inches='tight')