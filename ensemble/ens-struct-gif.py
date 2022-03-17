# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:43:38 2021

@author: rounak
Description: Compares between networks in an ensemble. Creates a gif out of the
network structure variation
"""

import sys,os
import networkx as nx
import geopandas as gpd
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/osm-primnet/"
outpath = workpath + "/out/osm-ensemble/"
sys.path.append(libpath)


from pyDrawNetworklib import DrawNodes, DrawEdges
   
def makegif(src,dest):
    '''
    Input:  src : Source directory of images
            dest: Destination path of gif
    '''
    fnames = [f for f in os.listdir(src) if ".png" in f]
    fnames_sorted = [str(m)+'.png'for m in 
                     sorted([int(s.strip('.png')) for s in fnames])]

    with imageio.get_writer(dest+'.gif', mode='I') as writer:
        for f in fnames_sorted:
            image = imageio.imread(src+f)
            writer.append_data(image)
    
    for f in fnames:
        os.remove(src+f)
    return

def plot_network(net,inset={},path=None,with_secnet=False,alpha=1.0):
    """
    """
    plt.ioff()
    fig = plt.figure(figsize=(40,40), dpi=72)
    ax = fig.add_subplot(111)
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000,alpha=alpha)
    DrawNodes(net,ax,label='T',color='green',size=100,alpha=alpha)
    DrawNodes(net,ax,label='R',color='black',size=1.0,alpha=alpha)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=20.0,alpha=alpha)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=1.0,alpha=alpha)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=1.0,alpha=alpha)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=0.5,alpha=alpha)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    fig.savefig(path+".png",bbox_inches='tight')
    plt.close()
    return

# def color_edges(net,inset={},path=None,alpha=1.0):
#     plt.ioff()
#     fig = plt.figure(figsize=(35,30),dpi=72)
#     ax = fig.add_subplot(111)
    
#     # Draw nodes
#     DrawNodes(net,ax,label=['S','T','R','H'],color='black',size=2.0,alpha=alpha)
    
#     # Draw edges
#     d = {'edges':net.edges(),
#          'geometry':[net[e[0]][e[1]]['geometry'] for e in net.edges()],
#          'flows':[net[e[0]][e[1]]['flow'] for e in net.edges()]}
#     df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
#     fmin = np.log(0.2); fmax = np.log(800.0)
#     df_edges.plot(column='flows',ax=ax,cmap=cm.plasma,vmin=fmin,vmax=fmax,
#                   alpha=alpha)
#     ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
#     fig.savefig(path+".png",bbox_inches='tight')
#     plt.close()

sub = 121144
tmppath = figpath+"temp/"


for i in range(1,21):
    tree = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i)+'.gpickle')
    plot_network(tree,with_secnet=True,path=tmppath+str(i),alpha=0.3)
makegif(tmppath,figpath+str(sub)+"-ensemble")


