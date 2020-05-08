# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program approaches the set cover problem to find optimal/sub-
optimal placement of transformers along the road network graph. Thereafter it 
creates a spider network to cover all the residential buildings. The spider net
is a forest of trees rooted at the transformer nodes.

This program creates the spider network using heuristic based as well as power
flow based optimization setup.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import random
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider

def display_data(ax,roads,homes,showhome=True,colors=['royalblue','seagreen']):
    """
    Displays the road network and residences in the given region.
    Parameters
    ----------
    ax    : TYPE: matplotlib axis object
        DESCRIPTION.
    roads : TYPE: named tuple with road network data
        DESCRIPTION.
    homes : TYPE: named tuple with residential data
        DESCRIPTION.
    colors: TYPE: list, optional
        DESCRIPTION. The default is ['royalblue','seagreen']. 
        Color specifications for roads and residences

    Returns
    -------
    None.
    """
    nx.draw_networkx(roads.graph,node_size=0.1,color=colors[0],with_labels=False,
                     ax=ax,pos=roads.cord,edge_color=colors[0],style="dashed",width=0.5)
    if showhome:
        hpts = list(homes.cord.values())
        xpts = [h[0] for h in hpts]
        ypts = [h[1] for h in hpts]
        ax.scatter(xpts,ypts,c=colors[1],s=0.1)
    return ax

def inset_figure(link,ax,roads,homes,obj,loc=1):
    """
    Generates an inset figure for a road link with large number of residences.
    Parameters
    ----------
    ax    : TYPE: matplotlib axis object
        DESCRIPTION.
    roads : TYPE: named tuple with road network data
        DESCRIPTION.
    homes : TYPE: named tuple with residential data
        DESCRIPTION.
    link : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    axins = zoomed_inset_axes(ax, 4.0, loc=loc)
    axins.set_aspect(0.8)
    H,T = obj.get_nodes(link,minsep=50,followroad=True)
    # Get exact structure of road link
    line = roads.links[link]['geometry'].xy if link in roads.links \
        else roads.links[(link[1],link[0])]['geometry'].xy
    axins.plot(line[0],line[1],color='black',linewidth=1.2,linestyle='dashed')
    # Plot the homes and transformers
    axins.scatter([homes.cord[h][0] for h in H],[homes.cord[h][1] for h in H],c='red',
                s=80.0,marker='*')
    axins.scatter([t[0] for t in list(T.values())],[t[1] for t in list(T.values())],
                c='green',s=120.0,marker='*')
    axins.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    return ax

def display_secnet(roads,link,forest,roots):
    """
    Displays the forest network for a road network link.

    Parameters
    ----------
    forest : TYPE
        DESCRIPTION.
    roots : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pos_nodes = nx.get_node_attributes(forest,'cord')
    
    # Display the secondary network
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=[link],ax=ax,
                           edge_color='k',width=1.0,style='dashed')
    nodelist = list(forest.nodes())
    colors = ['red' if n not in roots else 'green' for n in nodelist]
    # shapes = ['*' if n not in roots else 's' for n in nodelist]
    nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                     ax=ax,edge_color='crimson',width=1,with_labels=False,
                     node_size=10.0,node_shape='*',node_color=colors)
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_title("Secondary network creation for road link",fontsize=20)
    
    
    leglines = [Line2D([0], [0], color='black', markerfacecolor='green', marker='*',
                       markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='crimson', markerfacecolor='crimson', 
                       marker='*',markersize=0),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='*',
                       markersize=15),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',
                       markersize=15)]
    ax.legend(leglines,['road link','secondary network','local transformers',
                         'residences'],
              loc='best',ncol=2,prop={'size': 15})
    return

#%% Initialization of data sets and mappings
q_object = Query(csvPath,inpPath)
homes,roads = q_object.GetDataset(fislist=[161,770,775])

fiscode = '%03.f'%(161)
df_hmap = pd.read_csv(csvPath+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home

# sys.exit(0)

#%% Display summary of progress for secondary distribution network creation
showlinks = [l for l in L2Home if len(L2Home[l])>200]
homelist = []
for link in showlinks: homelist.extend(L2Home[link])


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
ax = display_data(ax,roads,homes,showhome=True)

# Main figure
nx.draw_networkx_edges(roads.graph,pos=roads.cord,edgelist=showlinks,ax=ax,
                        width=0.8,edge_color='k',style='dashed')
ax.scatter([homes.cord[h][0] for h in homelist],[homes.cord[h][1] for h in homelist],
           c='red',s=10.0,marker='*')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax.set_title("Links with large number of residences mapped",fontsize=20)

# Inset figure
link = showlinks[0]
ax = inset_figure(link,ax,roads,homes,spider_obj,loc=1)

# sys.exit(0)

#%% Plot the base network / Delaunay Triangulation
def display_linkhome(link,homelist,roads,homes,tsfr=None):
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
    tsfr : dictionary of transformer IDs and coordinaes, optional
        list of transformer locations along the road network link. 
        The default is None so that the transformers are not displayed.

    Returns
    -------
    None.

    """
    leglines = [Line2D([0], [0], color='black', markerfacecolor='c', marker='*',
                       markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',
                       markersize=15)]
    labels = ['road link','residences mapped']
    
    # Figure initialization
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    line = roads.links[link]['geometry'].xy if link in roads.links \
        else roads.links[(link[1],link[0])]['geometry'].xy

    # Plot the road network link
    ax.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    # Plot the residences
    ax.scatter([homes.cord[h][0] for i,h in enumerate(homelist)],
               [homes.cord[h][1] for i,h in enumerate(homelist)],
               c='red',s=25.0,marker='*')
    
    if tsfr!=None:
        ax.scatter([t[0] for t in list(tsfr.values())],
                   [t[1] for t in list(tsfr.values())],
                   c='green',s=60.0,marker='*')
        leglines += [Line2D([0], [0], color='white', markerfacecolor='green', 
                            marker='*',markersize=15)]
        labels += ['possible transformers']
        ax.set_title("Probable transformers along road link",fontsize=20)
        figname = 'secnet-tsfr'
    else:
        ax.set_title("Residences mapped to a road link",fontsize=20)
        figname = 'secnet-home'
    
    # Final adjustments
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.legend(leglines,labels,loc='best',ncol=1,prop={'size': 15})
    
    # Save the figure
    fig.savefig("{}{}.png".format(figPath,figname),bbox_inches='tight')
    return

def display_sidehome(link,side,roads,homes):
    """
    Displays the road link with the residences mapped to it. Also provides displays
    which side of the road link each residence is located.

    Parameters
    ----------
    link : tuple of the terminal node IDs
        the road network link of interest.
    side : dictionary
        dictionary of residence IDs and values as which side of road link.
    roads : named tuple of type road
        information related to the road network.
    homes : named tuple of type home
        information related to residences

    Returns
    -------
    None.

    """
    homelist = list(side.keys())
    leglines = [Line2D([0], [0], color='black', markerfacecolor='c', marker='*',
                       markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='*',
                       markersize=15),
                Line2D([0], [0], color='white', markerfacecolor='blue', marker='*',
                       markersize=15)]
    labels = ['road link','residences on side A','residences on side B']
    
    # Figure initializations
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    line = roads.links[link]['geometry'].xy if link in roads.links \
        else roads.links[(link[1],link[0])]['geometry'].xy

    # Plot the road network link
    ax.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    # Plot the residences
    ax.scatter([homes.cord[h][0] for i,h in enumerate(homelist) if side[h]==1],
               [homes.cord[h][1] for i,h in enumerate(homelist) if side[h]==1],
               c='red',s=40.0,marker='*')
    ax.scatter([homes.cord[h][0] for i,h in enumerate(homelist) if side[h]==-1],
               [homes.cord[h][1] for i,h in enumerate(homelist) if side[h]==-1],
               c='blue',s=40.0,marker='*')
    
    ax.set_title("Separating residences on either side of road link",fontsize=20)
    
    # Final adjustments
    figname = 'secnet-side'
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.legend(leglines,labels,loc='best',ncol=1,prop={'size': 15})
    fig.savefig("{}{}.png".format(figPath,figname),bbox_inches='tight')
    return

link = (171537011, 171537012)
H,T = spider_obj.get_nodes(link,minsep=50,followroad=True)
side = spider_obj.separate_side(link)
display_linkhome(link,H,roads,homes,tsfr=None)
display_linkhome(link,H,roads,homes,tsfr=T)
display_sidehome(link,side,roads,homes)
sys.exit(0)
#%% Create secondary distribution network as a forest of disconnected trees
def display_secondary(forest,roots,link,roads):
    """
    Displays the output secondary network obtained from solving the optimization
    problem. 

    Parameters
    ----------
    forest : Networkx graph with coordinates as node attributes
        A forest of trees representing the secondary network rooted at transformers.
    roots : dictionary of transformer locations
        A dictionary with keys as transformer IDs and value as coordinates.
    link : tuple of the terminal node IDs
        the road network link of interest.
    roads : named tuple of type road
        information related to the road network.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    
    # Plot the road network link
    line = roads.links[link]['geometry'].xy if link in roads.links \
        else roads.links[(link[1],link[0])]['geometry'].xy
    ax.plot(line[0],line[1],color='black',linewidth=1,linestyle='dashed')
    
    # Get network data
    nodelist = list(forest.nodes())
    colors = ['red' if n not in roots else 'green' for n in nodelist]
    pos_nodes = nx.get_node_attributes(forest,'cord')
    
    # Draw network
    nx.draw_networkx(forest,pos=pos_nodes,edgelist=list(forest.edges()),
                     ax=ax,edge_color='crimson',width=1,with_labels=False,
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
    ax.legend(leglines,labels,loc='best',ncol=1,prop={'size': 10})
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_title("Secondary network creation for road link",fontsize=20)
    
    # Save the figure
    fig.savefig("{}{}.png".format(figPath,'secnet-result'),bbox_inches='tight')
    return

forest,roots = spider_obj.generate_optimal_topology(link,minsep=50,followroad=True,
                                                    heuristic=None)
display_secondary(forest,roots,link,roads)
sys.exit(0)
#%% Compare voltages at different nodes when heuristic choices are varied
dict_vol = {h:[] for h in H}
for hop in range(4,12):
    forest,tsfr = spider_obj.generate_optimal_topology(link,minsep=50,hops=hop)
    volts = spider_obj.checkpf(forest,tsfr)
    for h in H: dict_vol[h].append(volts[h])

# Plot variation in voltages at nodes
data = np.array(list(dict_vol.values()))
homeID = [str(h) for h in list(dict_vol.keys())]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.plot(data,'o-')
ax.set_xticks(range(len(H)))
ax.set_xticklabels(homeID)
ax.tick_params(axis='x',rotation=90)
ax.set_xlabel("Residential Building IDs",fontsize=15)
ax.set_ylabel("Voltage level in pu",fontsize=15)
ax.legend(labels=['max depth='+str(i) for i in range(4,12)])
ax.set_title("Voltage profile at residential nodes in the generated forest",
             fontsize=15)

print("DONE")