# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:38:51 2021

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
from pygeodesy import hausdorff_
import threading


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
synpath = rootpath + "/primnet/out/"
enspath = rootpath + "/ensemble/out/"

sys.path.append(libpath)
from pyGeometrylib import Link,partitions,geodist
from pyPowerNetworklib import GetDistNet
print("Imported modules")


# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]
sub = 121144
net = 8
synth_net = GetDistNet(synpath,sub)
ensem_net = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net)+'.gpickle')
print("Synthetic network extracted")

#%% Network data
cords = np.array([list(synth_net.nodes[n]['cord']) for n in synth_net])
LEFT,BOTTOM = np.min(cords,0)
RIGHT,TOP = np.max(cords,0)

syn_edges = [synth_net.edges[e]['geometry'] for e in synth_net.edges()]
ens_edges = [ensem_net.edges[e]['geometry'] for e in ensem_net.edges()]


#%% Plot the grid
s = 20
C_vals = {}

# process: 
def process(items, start, end):
    for grid in items[start:end]:
        ens_edges_pts = [p for geom in ens_edges \
                         for p in Link(geom).InterpolatePoints(sep=s) \
                             if geom.within(grid) or \
                                 geom.intersects(grid.exterior)]
        syn_edges_pts = [p for geom in syn_edges \
                         for p in Link(geom).InterpolatePoints(sep=s) \
                             if geom.within(grid) or \
                                 geom.intersects(grid.exterior)]
        print(grid)
        if len(ens_edges_pts) != 0 and len(syn_edges_pts) != 0:
            C_vals[grid] = hausdorff_(syn_edges_pts,ens_edges_pts,
                                      distance=geodist).hd
        else:
            C_vals[grid] = np.nan
        

def split_processing(items, num_splits=5):
    split_size = len(items) // num_splits
    threads = []
    for i in range(num_splits):
        # determine the indices of the list this thread will handle
        start = i * split_size
        # special case on the last chunk to account for uneven splits
        end = None if i+1 == num_splits else (i+1) * split_size
        # create the thread
        threads.append(
            threading.Thread(target=process, args=(items, start, end)))
        threads[-1].start() # start the thread we just created

    # wait for all threads to finish
    for t in threads:
        t.join()
    return

# Make partition and compute hausdorff distance
gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),5,5)
split_processing(gridlist)
C = []
for grid in gridlist: C.append(C_vals[grid])
C_masked = np.ma.array(C, mask=np.isnan(C))

#%% Plot the spatial distribution
from shapely.geometry import Point
def DrawNodes(graph,ax,color='orangered',size=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    d = {'nodes':graph.nodes(),
         'geometry':[Point(graph.nodes[n]['cord']) for n in graph.nodes()]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size)
    return

def DrawEdges(graph,ax,color='orangered',width=1.0):
    """
    """
    d = {'edges':graph.edges(),
         'geometry':[graph[e[0]][e[1]]['geometry'] for e in graph.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width)
    return


colormap = cm.Greens
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


DrawNodes(ensem_net,ax,color='orangered')
DrawNodes(synth_net,ax,color='blue')
DrawEdges(ensem_net,ax,color='orangered')
DrawEdges(synth_net,ax,color='blue')



ax.set_xticks([])
ax.set_yticks([])

cobj = cm.ScalarMappable(cmap=colormap)
cobj.set_clim(vmin=0.0,vmax=max(C_valid))
cbar = fig.colorbar(cobj,ax=ax)
cbar.set_label('Hausdorff distance(in meters)',size=20)
cbar.ax.tick_params(labelsize=20)

leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                   marker='o',markersize=10, label='Variant distribution network'),
            Line2D([0], [0], color='blue', markerfacecolor='blue',
                   marker='o',markersize=10, label='Optimal distribution network'),
            Patch(facecolor='white', edgecolor='black', hatch="./",
                         label='Grids with no actual network data')]

ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})
fig.savefig("{}{}.png".format(figpath,'sep'+str(s)+'-hausdorff-5by5-comp-ens-'+str(net)),
            bbox_inches='tight')














