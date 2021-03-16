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
synpath = rootpath + "/primnet/out/"

sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
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


#%% Grid comaprison
from shapely.geometry import Polygon, LineString, Point

class Grid(Polygon):
    """A rectangular grid with limits."""

    def __init__(self, poly_geom):
        super().__init__(poly_geom)
        self.west_edge, self.east_edge = poly_geom[0][0],poly_geom[1][0]
        self.north_edge, self.south_edge = poly_geom[0][1], poly_geom[2][1]
        return

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)
    
    def __hash__(self):
        return hash((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

# Partition into grids
def partitions(limits,kx,ky,x0=0,y0=0):
    """
    kx,ky: number of demarcations along x and y axes.
    x0,y0: percentage x and y shifts for the grid demarcations.
    """
    LEFT,RIGHT,BOTTOM,TOP = limits
    xmin = []; xmax = []; ymin = []; ymax = []
    x_shift = x0*width
    y_shift = y0*height
    for t in range(kx):
        xmin.append(LEFT+(t/kx)*(RIGHT-LEFT)+x_shift)
        xmax.append(LEFT+((1+t)/kx)*(RIGHT-LEFT)+x_shift)
    for t in range(ky):
        ymin.append(BOTTOM+(t/ky)*(TOP-BOTTOM)+y_shift)
        ymax.append(BOTTOM+((1+t)/ky)*(TOP-BOTTOM)+y_shift)
    # For shifted origins
    if x0>0:
        xmax = [xmin[0]]+xmax
        xmin = [LEFT]+xmin
    if x0<0:
        xmin = xmin+[xmax[-1]]
        xmax = xmax+[RIGHT]
    if y0>0:
        ymax = [ymin[0]]+ymax
        ymin = [BOTTOM]+ymin
    if y0<0:
        ymin = ymin+[ymax[-1]]
        ymax = ymax+[TOP]
    # get the grid polygons
    grid = []
    for i in range(len(xmin)):
        for j in range(len(ymin)):
            vertices = [(xmin[i],ymin[j]),(xmax[i],ymin[j]),
                        (xmax[i],ymax[j]),(xmin[i],ymax[j])]
            grid.append(Grid(vertices))
    return grid

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





#%% Count the number of node and edges in actual and synthetic networks
act_geom = [g for area in area_data \
             for g in area_data[area]['df_lines']['geometry'].tolist()]
synth_geom = [g for area in area_data \
             for g in area_data[area]['df_synth']['geometry'].tolist()]


act_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_buses']['geometry'].tolist()]
synth_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_cords']['geometry'].tolist()]


grid = partitions((LEFT,RIGHT,BOTTOM,TOP),7,7,x0=-0.15)
A = node_dist(grid,act_nodes_geom)
B = node_dist(grid,synth_nodes_geom)
A_nodes = sum(list(A.values()))
B_nodes = sum(list(B.values()))

C = {bound:100.0*(1-((B[bound]/B_nodes)/(A[bound]/A_nodes))) \
     if A[bound]!=0 else np.nan for bound in grid}

C_vals = np.array([C[bound] for bound in C])
C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))
##%% Plot the spatial distribution
colormap = cm.BrBG
def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge,boundary.north_edge]
    vert2 = [boundary.east_edge,boundary.north_edge]
    vert3 = [boundary.east_edge,boundary.south_edge]
    vert4 = [boundary.west_edge,boundary.south_edge]
    return np.array([vert1,vert2,vert3,vert4])

DPI = 72    
fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
ax = plt.subplot()
ax.set_xlim(LEFT,RIGHT)
ax.set_ylim(BOTTOM,TOP)

# Get the boxes for the valid comparisons
verts_valid = [get_polygon(bound) for i,bound in enumerate(C) \
               if not C_masked.mask[i]]
C_valid = [C_masked.data[i] for i in range(len(C)) if not C_masked.mask[i]]
c = PolyCollection(verts_valid,edgecolor='black')
c.set_array(C_masked)
c.set_cmap(colormap)
ax.add_collection(c)

# Get the boxes for absent actual data
verts_invalid = [get_polygon(bound) for i,bound in enumerate(C) \
               if C_masked.mask[i]]
c = PolyCollection(verts_invalid,hatch=r"./",facecolor='white',edgecolor='black')
ax.add_collection(c)

for area in area_data:
    plot_network(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],'orangered')
    plot_network(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],'blue')


ax.set_xticks([])
ax.set_yticks([])

cobj = cm.ScalarMappable(cmap=colormap)
cobj.set_clim(vmin=-100,vmax=100)
cbar = fig.colorbar(cobj,ax=ax)
cbar.set_label('Percentage Deviation',size=20)
cbar.ax.tick_params(labelsize=20)

leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                   marker='o',markersize=10, label='Actual distribution network'),
            Line2D([0], [0], color='blue', markerfacecolor='blue',
                   marker='o',markersize=10, label='Synthetic distribution network'),
            Patch(facecolor='white', edgecolor='black', hatch="./",
                         label='Grids with no actual network data')]

#ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})





















