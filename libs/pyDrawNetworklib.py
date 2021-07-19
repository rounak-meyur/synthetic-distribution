# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations
"""

import sys
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D


#%% Network Geometries
def DrawNodes(synth_graph,ax,label='T',color='green',size=25):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    nodelist = [n for n in synth_graph.nodes() \
                if synth_graph.nodes[n]['label']==label \
                    or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size)
    return

def DrawEdges(synth_graph,ax,label='P',color='black',width=2.0):
    """
    """
    # Get the nodes for the specified label
    edgelist = [e for e in synth_graph.edges() \
                if synth_graph[e[0]][e[1]]['label']==label\
                    or synth_graph[e[0]][e[1]]['label'] in label]
    d = {'edges':edgelist,
         'geometry':[synth_graph[e[0]][e[1]]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width)
    return


def plot_network(net,inset={},path=None,with_secnet=False):
    """
    """
    fig = plt.figure(figsize=(30,30), dpi=72)
    ax = fig.add_subplot(111)
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax,inset[sub]['zoom'],loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes
        DrawNodes(inset[sub]['graph'],axins,label='S',color='dodgerblue',
                  size=2000)
        DrawNodes(inset[sub]['graph'],axins,label='T',color='green',size=25)
        DrawNodes(inset[sub]['graph'],axins,label='R',color='black',size=2.0)
        if with_secnet: DrawNodes(inset[sub]['graph'],axins,label='H',
                                  color='crimson',size=2.0)
        # Draw edges
        DrawEdges(inset[sub]['graph'],axins,label='P',color='black',width=2.0)
        DrawEdges(inset[sub]['graph'],axins,label='E',color='dodgerblue',width=2.0)
        if with_secnet: DrawEdges(inset[sub]['graph'],axins,label='S',
                                  color='crimson',width=1.0)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], 
                   loc2=inset[sub]['loc2'], fc="none", ec="0.5")
    
    # Legend for the plot
    leghands = [Line2D([0], [0], color='black', markerfacecolor='black', 
                   marker='o',markersize=0,label='primary network'),
            Line2D([0], [0], color='crimson', markerfacecolor='crimson', 
                   marker='o',markersize=0,label='secondary network'),
            Line2D([0], [0], color='dodgerblue', 
                   markerfacecolor='dodgerblue', marker='o',
                   markersize=0,label='high voltage feeder'),
            Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=20,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='red', 
                   marker='o',markersize=20,label='residence'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=20,label='substation')]
    ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
    if path != None: 
        fig.savefig("{}{}.png".format(path,'-51121-dist'),bbox_inches='tight')
    return


def color_nodes(net,inset={},path=None,vmax=1.05):
    fig = plt.figure(figsize=(35,30),dpi=72)
    ax = fig.add_subplot(111)
    
    # Draw edges
    DrawEdges(net,ax,label=['P','E','S'],color='black',width=1.0)
    
    # Draw nodes
    d = {'nodes':net.nodes(),
         'geometry':[Point(net.nodes[n]['cord']) for n in net.nodes()],
         'voltage':[net.nodes[n]['voltage'] for n in net.nodes()]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.001)
    df_nodes.plot(ax=ax,column='voltage',markersize=40.0,cmap=cm.plasma,
                  vmin=0.80,vmax=vmax,cax=cax,legend=True)
    cax.set_ylabel("Voltage(in pu)",fontsize=30)
    cax.tick_params(labelsize=30)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax, inset[sub]['zoom'], 
                                  loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes and edges
        DrawEdges(inset[sub]['graph'],axins,label=['P','E','S'],
                  color='black',width=1.0)
        d = {'nodes':inset[sub]['graph'].nodes(),
             'geometry':[Point(inset[sub]['graph'].nodes[n]['cord']) \
                         for n in inset[sub]['graph'].nodes()],
             'voltage':[inset[sub]['graph'].nodes[n]['voltage'] \
                        for n in inset[sub]['graph'].nodes()]}
        df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_nodes.plot(ax=axins,column='voltage',markersize=30.0,cmap=cm.plasma,
                      vmin=0.80,vmax=vmax)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], loc2=inset[sub]['loc2'], 
               fc="none", ec="0.5")
    if path!=None:
        fig.savefig("{}{}.png".format(path,'-dist-voltage'),bbox_inches='tight')
    return


def color_edges(net,inset={},path=None):
    fig = plt.figure(figsize=(35,30),dpi=72)
    ax = fig.add_subplot(111)
    
    # Draw nodes
    DrawNodes(net,ax,label=['S','T','R','H'],color='black',size=2.0)
    
    # Draw edges
    d = {'edges':net.edges(),
         'geometry':[net[e[0]][e[1]]['geometry'] for e in net.edges()],
         'flows':[net[e[0]][e[1]]['flow'] for e in net.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    fmin = np.log(0.2); fmax = np.log(800.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.001)
    df_edges.plot(column='flows',ax=ax,cmap=cm.plasma,vmin=fmin,vmax=fmax,
                  cax=cax,legend=True)
    cax.set_ylabel('Flow along edge in kVA',size=30)
    labels = [100,200,300,400,500,600,700,800]
    cax.set_yticklabels(labels)
    cax.tick_params(labelsize=20)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax, inset[sub]['zoom'], 
                                  loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes and edges
        DrawNodes(inset[sub]['graph'],axins,label=['S','T','R','H'],
                  color='black',size=2.0)
        d = {'edges':inset[sub]['graph'].edges(),
             'geometry':[inset[sub]['graph'][e[0]][e[1]]['geometry'] \
                         for e in inset[sub]['graph'].edges()],
             'flows':[np.exp(inset[sub]['graph'][e[0]][e[1]]['flow']) \
                      for e in inset[sub]['graph'].edges()]}
        df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_edges.plot(column='flows',ax=axins,cmap=cm.plasma)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], loc2=inset[sub]['loc2'], 
               fc="none", ec="0.5")
    if path!=None:
        fig.savefig("{}{}.png".format(path,'-dist-flows'),bbox_inches='tight')
    return
