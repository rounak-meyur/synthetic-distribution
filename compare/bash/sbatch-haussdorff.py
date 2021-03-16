# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:38:51 2021

@author: rouna
"""

import os,sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pygeodesy import hausdorff_

workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/graph-comp"
actpath = scratchPath + "/actual/"



workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
actpath = workpath + "/actual/"
synpath = workpath + "/synthetic/"

sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
from pyGeometrylib import Link,partitions,geodist
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

#%% Get edges within grid
act_edges = [g for area in area_data \
             for g in area_data[area]['df_lines']['geometry'].tolist()]
syn_edges = [g for area in area_data \
             for g in area_data[area]['df_synth']['geometry'].tolist()]


gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),7,7)
grid = gridlist[8]



#%% Plot the grid
C_vals = []
for grid in gridlist:
    print(grid)
    act_edges_pts = [p for geom in act_edges \
                     for p in Link(geom).InterpolatePoints() \
                         if geom.within(grid) or geom.intersects(grid.exterior)]
    syn_edges_pts = [p for geom in syn_edges \
                     for p in Link(geom).InterpolatePoints() \
                         if geom.within(grid) or geom.intersects(grid.exterior)]
    
    if len(act_edges_pts) != 0 and len(syn_edges_pts) != 0:
        C_vals.append(hausdorff(syn_edges_pts,act_edges_pts,radius=0.001))
    else:
        C_vals.append(np.nan)

C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))

#%% Plot the spatial distribution
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
verts_valid = [get_polygon(bound) for i,bound in enumerate(gridlist) \
               if not C_masked.mask[i]]
C_valid = [C_masked.data[i] for i in range(len(gridlist)) if not C_masked.mask[i]]
c = PolyCollection(verts_valid,edgecolor='black')
c.set_array(C_masked)
c.set_cmap(colormap)
ax.add_collection(c)

# Get the boxes for absent actual data
verts_invalid = [get_polygon(bound) for i,bound in enumerate(gridlist) \
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















