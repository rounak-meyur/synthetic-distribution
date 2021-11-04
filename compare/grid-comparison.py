# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:16:52 2021

Author: Rounak
Description: Includes methods to compare pair of networks on the basis of 
partitioning into multiple grids and comparing each grid. Includes methods to
compare actual and synthetic networks of Blacksburg, VA.
"""

import os,sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
actpath = rootpath + "/input/actual/"
synpath = rootpath + "/primnet/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet,get_areadata
from pyGeometrylib import partitions
from pyDrawNetworklib import plot_gdf, plot_deviation, add_colorbar
from pyComparisonlib import compute_hausdorff
print("Imported modules")



#%% Data Extraction
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

#areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}
print("Area Data extracted and stored")

#%% Functions for comparing actual and synthetic networks
# Exract area data
def get_limits(area_data):
    # Get limits for the geographical region
    lims = np.empty(shape=(len(area_data),4))
    for i,area in enumerate(area_data):
        lims[i,:] = np.array(area_data[area]['limits'])
    LEFT = np.min(lims[:,0]); RIGHT = np.max(lims[:,1])
    BOTTOM = np.min(lims[:,2]); TOP = np.max(lims[:,3])
    return LEFT,RIGHT,BOTTOM,TOP

def get_edges(area_data):
    act_edges = [g for area in area_data \
                 for g in area_data[area]['df_lines']['geometry'].tolist()]
    syn_edges = [g for area in area_data \
                 for g in area_data[area]['df_synth']['geometry'].tolist()]
    return act_edges,syn_edges

def get_nodes(area_data):
    act_nodes_geom = [g for area in area_data \
                 for g in area_data[area]['df_buses']['geometry'].tolist()]
    synth_nodes_geom = [g for area in area_data \
                 for g in area_data[area]['df_cords']['geometry'].tolist()]
    return act_nodes_geom, synth_nodes_geom

# Plot the networks
def plot_network_pair(ax,area_data):
    for area in area_data:
        plot_gdf(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],
                      'orangered')
        plot_gdf(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],
                      'blue')
    leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                       marker='o',markersize=10, label='Actual distribution network'),
                Line2D([0], [0], color='blue', markerfacecolor='blue',
                       marker='o',markersize=10, label='Synthetic distribution network'),
                Patch(facecolor='white', edgecolor='black', hatch="./",
                             label='Grids with no actual network data')]
    
    ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})
    return

#%% Spatial Node Distribution
def get_suffix(delta):
    if delta>0:
        suf=str(int(abs(100*delta))) + 'pos'
    elif delta<0:
        suf=str(int(abs(100*delta))) + 'neg'
    else:
        suf='00'
    return suf

def node_stats(area_data,kx,ky,x0=0,y0=0,path=None):
    # Get the limits and data
    LEFT,RIGHT,BOTTOM,TOP = get_limits(area_data)
    gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky,x0=x0,y0=y0)
    act_nodes_geom, syn_nodes_geom = get_nodes(area_data)
    
    # Compute the deviation
    A = {g:sum([geom.within(g) for geom in act_nodes_geom]) for g in gridlist}
    B = {g:sum([geom.within(g) for geom in syn_nodes_geom]) for g in gridlist}
    A_nodes = sum(list(A.values()))
    B_nodes = sum(list(B.values()))
    
    C = {bound:100.0*(1-((B[bound]/B_nodes)/(A[bound]/A_nodes))) \
         if A[bound]!=0 else np.nan for bound in gridlist}
    
    C_vals = np.array([C[bound] for bound in C])
    C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))
    
    # Generate the plot
    DPI = 72    
    fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
    ax = plt.subplot()
    plot_network_pair(ax, area_data)
    plot_deviation(ax,gridlist,C_masked)
    add_colorbar(fig,ax)
    suffix = "-"+get_suffix(x0)+"-"+get_suffix(y0)
    if path != None:
        fig.savefig("{}{}.png".format(figpath,
            'spatial-comparison-'+str(kx)+str(ky)+suffix),bbox_inches='tight')
    return C

#%% Hausdorff distance between networks
def haus_dist(area_data,kx,ky,x0=0,y0=0):
    # Get the limits and data
    LEFT,RIGHT,BOTTOM,TOP = get_limits(area_data)
    gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky,x0=x0,y0=y0)
    act_edges,syn_edges = get_edges(area_data)
    
    # Compute Hausdorff distance
    C = compute_hausdorff(gridlist,act_edges,syn_edges)
    C_vals = np.array([C[bound] for bound in gridlist])
    C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))
    
    # Plot the deviation
    DPI = 72    
    fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
    ax = plt.subplot()
    plot_network_pair(ax, area_data)
    plot_deviation(ax,gridlist,C_masked,colormap=cm.Greens)
    add_colorbar(fig,ax,colormap=cm.Greens,
                 vmin=0.0,vmax=np.nanmax(C_vals),devname="Hausdorff distance (meters)")
    fig.savefig("{}{}.png".format(figpath,'hauss-comparison-'+str(kx)+'-'+str(ky)),
            bbox_inches='tight')
    return C


#%% Results
C = haus_dist(area_data,7,7)
# for s in [-0.1,-0.05,0,0.05,0.1]:
#     C = node_stats(area_data,7,7,x0=s)

# C = node_stats(area_data,10,10)
