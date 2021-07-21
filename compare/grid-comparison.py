# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:16:52 2021

@author: rouna
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.collections import PolyCollection
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
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
from pyGeometrylib import Grid,partitions,MeasureDistance
print("Imported modules")




sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

#%% Area specifications
#areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}

# Get limits for the geographical region
lims = np.empty(shape=(len(area_data),4))
for i,area in enumerate(area_data):
    lims[i,:] = np.array(area_data[area]['limits'])
LEFT = np.min(lims[:,0]); RIGHT = np.max(lims[:,1])
BOTTOM = np.min(lims[:,2]); TOP = np.max(lims[:,3])
width = RIGHT-LEFT
height = TOP-BOTTOM

act_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_buses']['geometry'].tolist()]
synth_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_cords']['geometry'].tolist()]

#%% Counter functions
# Distribution of edges
def edge_dist(grid,edge_geom):    
    m_within = {}
    m_cross = {}    
    for g in grid:
        m_within[g] = sum([geom.within(g) for geom in edge_geom])
        m_cross[g] = sum([geom.intersects(g.exterior) for geom in edge_geom])
    return m_within,m_cross

# Distribution of nodes
def node_dist(grid,node_geom):    
    return {g:sum([geom.within(g) for geom in node_geom]) for g in grid}


#%% Plot the spatial distribution
def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge,boundary.north_edge]
    vert2 = [boundary.east_edge,boundary.north_edge]
    vert3 = [boundary.east_edge,boundary.south_edge]
    vert4 = [boundary.west_edge,boundary.south_edge]
    return np.array([vert1,vert2,vert3,vert4])


def plot_deviation(gridlist,C_masked):
    x_array = np.array(sorted(list(set([g.west_edge for g in gridlist]\
                                       +[g.east_edge for g in gridlist]))))
    y_array = np.array(sorted(list(set([g.south_edge for g in gridlist]\
                                       +[g.north_edge for g in gridlist]))))
    # Initialize figure
    DPI = 72    
    fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
    ax = plt.subplot()
    ax.set_xlim(LEFT,RIGHT)
    ax.set_ylim(BOTTOM,TOP)
    
    # Plot the grid colors
    # colormap = cm.Greens
    colormap = cm.RdBu
    ky = len(x_array) - 1
    kx = len(y_array) - 1
    
    ax.pcolor(x_array,y_array,C_masked.reshape(kx,ky),cmap=colormap,
              edgecolor='black')
    
    # Get the boxes for absent actual data
    verts_invalid = [get_polygon(bound) for i,bound in enumerate(gridlist) \
                   if C_masked.mask[i]]
    c = PolyCollection(verts_invalid,hatch=r"./",facecolor='white',
                       edgecolor='black')
    ax.add_collection(c)
    
    # Plot the networks
    for area in area_data:
        plot_network(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],
                     'orangered')
        plot_network(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],
                     'blue')
    
    # Plot the accessory stuff
    ax.set_xticks([])
    ax.set_yticks([])
    
    cobj = cm.ScalarMappable(cmap=colormap)
    cobj.set_clim(vmin=-100.0,vmax=100.0)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label('Percentage Deviation',size=20)
    cbar.ax.tick_params(labelsize=20)
    
    leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                       marker='o',markersize=10, label='Actual distribution network'),
                Line2D([0], [0], color='blue', markerfacecolor='blue',
                       marker='o',markersize=10, label='Synthetic distribution network'),
                Patch(facecolor='white', edgecolor='black', hatch="./",
                             label='Grids with no actual network data')]
    
    ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})
    return fig

def node_stats(kx,ky,x0=0,y0=0):
    gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky,x0=x0,y0=y0)
    
    # Compute the deviation
    A = node_dist(gridlist,act_nodes_geom)
    B = node_dist(gridlist,synth_nodes_geom)
    A_nodes = sum(list(A.values()))
    B_nodes = sum(list(B.values()))
    
    C = {bound:100.0*(1-((B[bound]/B_nodes)/(A[bound]/A_nodes))) \
         if A[bound]!=0 else np.nan for bound in gridlist}
    
    C_vals = np.array([C[bound] for bound in C])
    C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))
    
    # Generate the plot
    fig = plot_deviation(gridlist,C_masked)
    fig.savefig("{}{}.png".format(figpath,
        'spatial-comparison-'+str(kx)+'-'+str(ky)+'-xs-'+str(int(x0*100))+'-ys-'+str(int(y0*100))),
        bbox_inches='tight')
    return C


#%% Generate the plots
for s in [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2]:
    C = node_stats(7,7,x0=s)

