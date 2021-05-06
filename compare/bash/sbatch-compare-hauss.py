# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:38:51 2021

@author: rouna
"""

import os,sys
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import threading
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pygeodesy import hausdorff_

workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"

figpath = scratchPath + "/figs/"
synpath = scratchPath + "/temp/prim-network/"
enspath = scratchPath + "/temp/ensemble/"


from pyGeometrylib import Link,partitions,geodist
print("Imported modules")

sub = sys.argv[1]
net1 = sys.argv[2]
net2 = sys.argv[3]

ensem_net1 = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net1)+'.gpickle')
ensem_net2 = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net2)+'.gpickle')
print("Synthetic network extracted")

#%% Network data
cords = np.array([list(ensem_net1.nodes[n]['cord']) for n in ensem_net1])
LEFT,BOTTOM = np.min(cords,0)
RIGHT,TOP = np.max(cords,0)

ens_edges1 = [ensem_net1.edges[e]['geometry'] for e in ensem_net1.edges()]
ens_edges2 = [ensem_net2.edges[e]['geometry'] for e in ensem_net2.edges()]


#%% Plot the grid
s = 10
C_vals = {}

# process: 
def process(items, start, end):
    for grid in items[start:end]:
        ens_edges_pts1 = [p for geom in ens_edges1 \
                         for p in Link(geom).InterpolatePoints(sep=s) \
                             if geom.within(grid) or \
                                 geom.intersects(grid.exterior)]
        ens_edges_pts2 = [p for geom in ens_edges2 \
                         for p in Link(geom).InterpolatePoints(sep=s) \
                             if geom.within(grid) or \
                                 geom.intersects(grid.exterior)]
        print(grid)
        if len(ens_edges_pts1) != 0 and len(ens_edges_pts2) != 0:
            C_vals[grid] = hausdorff_(ens_edges_pts1,ens_edges_pts2,
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
kx = 5; ky = 5
gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky)
x_array = np.array([LEFT+(t/kx)*(RIGHT-LEFT) for t in range(kx)]+[RIGHT])
y_array = np.array([BOTTOM+(t/ky)*(TOP-BOTTOM) for t in range(ky)]+[TOP])


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


DPI = 72    
fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
ax = plt.subplot()
ax.set_xlim(LEFT,RIGHT)
ax.set_ylim(BOTTOM,TOP)
ax.pcolor(x_array,y_array,C_masked.reshape(kx,ky).T,cmap=colormap,
          edgecolor='black')


DrawNodes(ensem_net1,ax,color='orangered')
DrawNodes(ensem_net2,ax,color='blue')
DrawEdges(ensem_net1,ax,color='orangered')
DrawEdges(ensem_net2,ax,color='blue')



ax.set_xticks([])
ax.set_yticks([])

cobj = cm.ScalarMappable(cmap=colormap)
cobj.set_clim(vmin=0.0,vmax=max(C_masked))
cbar = fig.colorbar(cobj,ax=ax)
cbar.set_label('Hausdorff distance(in meters)',size=20)
cbar.ax.tick_params(labelsize=20)

leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                   marker='o',markersize=10, label='Variant distribution network '+str(net1)),
            Line2D([0], [0], color='blue', markerfacecolor='blue',
                   marker='o',markersize=10, label='Variant distribution network '+str(net2))]

ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})
fig.savefig("{}{}.png".format(figpath,str(sub)+'-sep'+str(s)+'-hausdorff-5by5-comp-ens-'\
                              +str(net1)+str(net2)),bbox_inches='tight')














