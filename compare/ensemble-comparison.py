# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:38:51 2021

Author: Rounak
Description: Compares the Hausdorff distance between pair of networks in the
ensemble of networks. Plots the color plot of the distance metric as an output.
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
synpath = rootpath + "/primnet/out/osm-primnet/"
enspath = rootpath + "/ensemble/out/osm-ensemble/"

sys.path.append(libpath)
from pyGeometrylib import partitions
from pyExtractDatalib import GetDistNet
from pyDrawNetworklib import plot_deviation, add_colorbar, DrawEdges, DrawNodes
from pyComparisonlib import compute_hausdorff
print("Imported modules")


#%% Network data
def get_limits(synth_net):
    # Get limits for the geographical region
    cords = np.array([list(synth_net.nodes[n]['cord']) for n in synth_net])
    LEFT,BOTTOM = np.min(cords,0)
    RIGHT,TOP = np.max(cords,0)
    return LEFT,RIGHT,BOTTOM,TOP

def plot_network_pair(ax,net1,net2,
                      label1 = "Optimal synthetic network",
                      label2 = "Variant synthetic network"):
    DrawNodes(net1,ax,color="orangered",size=1.0)
    DrawNodes(net2,ax,color="blue",size=1.0)
    DrawEdges(net1,ax,color="orangered",width=1.0)
    DrawEdges(net2,ax,color="blue",width=1.0)
    leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                       marker='o',markersize=10, label=label1),
                Line2D([0], [0], color='blue', markerfacecolor='blue',
                       marker='o',markersize=10, label=label2),
                Patch(facecolor='white', edgecolor='black', hatch="./",
                             label='Grids with no network data')]
    ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})
    return

#%% Plot the grid
def haus_dist(net1,net2,kx,ky,x0=0,y0=0):
    # Get the limits and data
    LEFT,RIGHT,BOTTOM,TOP = get_limits(net1)
    gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky,x0=x0,y0=y0)
    edges1 = [net1.edges[e]['geometry'] for e in net1.edges()]
    edges2 = [net2.edges[e]['geometry'] for e in net2.edges()]
    
    # Compute Hausdorff distance
    C = compute_hausdorff(gridlist,edges1,edges2)
    C_vals = np.array([C[bound] for bound in gridlist])
    C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))
    
    # Plot the deviation
    DPI = 72    
    fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
    ax = plt.subplot()
    plot_network_pair(ax,net1,net2)
    plot_deviation(ax,gridlist,C_masked,colormap=cm.Greens)
    add_colorbar(fig,ax,colormap=cm.Greens,
                 vmin=0.0,vmax=max(C_masked),devname="Hausdorff distance")
    fig.savefig("{}{}.png".format(figpath,
                str(sub)+'-haus-comp-'+str(kx)+'-'+str(ky)+suffix),
            bbox_inches='tight')
    return C

#%% Result of comparison Example 1
sub = 121144
net = 8
synth_net = GetDistNet(synpath,sub)
ensem_net = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net)+'.gpickle')
suffix = '-0-'+str(net)
print("Synthetic network extracted")
C = haus_dist(synth_net, ensem_net, 5, 5)

#%% Result of comparison Example 2
net1 = 1
net2 = 2
ensem_net1 = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net1)+'.gpickle')
ensem_net2 = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(net2)+'.gpickle')
suffix = '-'+str(net1)+'-'+str(net2)

print("Synthetic network extracted")
C = haus_dist(ensem_net1, ensem_net2, 5, 5)