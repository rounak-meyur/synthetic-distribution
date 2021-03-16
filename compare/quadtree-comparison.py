# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:25:08 2021

@author: Rounak
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
from pyQuadTreelib import Point, Rect, QuadTree
from pyGeometrylib import Link
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
print("Imported modules")




sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
sgeom = nx.get_edge_attributes(synth_net,'geometry')
synthgeom = {e:Link(sgeom[e]) for e in sgeom}
glength = {e:synthgeom[e].geod_length for e in sgeom}
nx.set_edge_attributes(synth_net,glength,'geo_length')


# Area specifications
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
domain = Rect(LEFT+width/2, BOTTOM+height/2, width, height)

# Get synthetic network within the domain
sgraph_nodes = [n for n in synth_net \
                if LEFT<=synth_net.nodes[n]['cord'][0]<=RIGHT and \
                    BOTTOM<=synth_net.nodes[n]['cord'][1]<=TOP and\
                        synth_net.nodes[n]['label']=='T']
synth_coords = [synth_net.nodes[n]['cord'] for n in sgraph_nodes]
synth_points = [Point(*coord) for coord in synth_coords]

print("Synthetic network extracted")
#%% Quad tree comparisons

points = []
for area in area_data:
    bus_geom = area_data[area]['df_buses']['geometry'].tolist()
    points.extend(bus_geom)


qtree = QuadTree(domain, 30)
for point in points:
    qtree.insert(point)

A = {}
A = qtree.point_stat(A)
A_nodes = sum(list(A.values()))

B = {}
for bound in A:
    B[bound] = sum([bound.contains(p) for p in synth_points])
B_nodes = sum(list(B.values()))

C = {k:(A[k]/A_nodes)-(B[k]/B_nodes) for k in A}
# colormap = cm.coolwarm
colormap = cm.BrBG
# nan_color = 'white'
# colormap.set_bad(nan_color,1.)

for k in A:
    if A[k]!=0:
        C[k]=100.0*C[k]/(A[k]/A_nodes)
    else:
        C[k]=np.nan
C_vals = np.array([C[bound] for bound in C])
C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))


#%% Plot rectangles of the quad tree
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
c = PolyCollection(verts_valid)
c.set_array(C_masked)
c.set_cmap(colormap)
ax.add_collection(c)

# Get the boxes for absent actual data
verts_invalid = [get_polygon(bound) for i,bound in enumerate(C) \
               if C_masked.mask[i]]
c = PolyCollection(verts_invalid,hatch=r"./",facecolor='white')
ax.add_collection(c)

# Draw the box partitions
qtree.draw(ax)

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

ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})




#%% Search Quadtree problem
# region = Rect(140, 190, 150, 150)
# found_points = []
# qtree.query(region, found_points)
# print('Number of found points =', len(found_points))

# ax.scatter([p.x for p in found_points], [p.y for p in found_points],
#            facecolors='none', edgecolors='r', s=32)

# region.draw(ax, c='r')

# ax.invert_yaxis()
# plt.tight_layout()
# plt.savefig('search-quadtree.png', DPI=72)
# plt.show()


