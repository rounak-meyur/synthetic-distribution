# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:29:49 2020

Author: Rounak Meyur
Description: This program creates a visualization of how the local transformers 
are distributed among the substations.
"""

import sys,os
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset



workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyBuildNetworklib import read_master_graph
from pyExtractDatalib import Query


# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
# scratchPath = workPath
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"

q_object = Query(csvPath,inpPath)
subs = q_object.GetAllSubstations()


with open(inpPath+'sublist.txt') as f:
    sublist = [int(x) for x in f.readlines()[0].split(' ')]

G = nx.Graph()
color_list = ['crimson','royalblue','seagreen','magenta','gold',
              'cyan','olive','maroon']*132
for i,s in enumerate(sublist):
    new_g = read_master_graph(tmpPath+'prim-master/',str(s))
    ncol = {n:color_list[i] for n in list(new_g.nodes())}
    nx.set_node_attributes(new_g,ncol,'color')
    G = nx.union(G,new_g)


nodepos = nx.get_node_attributes(G,'cord')
nodecol = nx.get_node_attributes(G,'color')
nodelist = list(G.nodes())
nodecollist = [nodecol[n] for n in nodelist]
fig = plt.figure(figsize=(100,50))
ax = fig.add_subplot(111)
nx.draw_networkx(G,pos=nodepos,ax=ax,nodelist=nodelist,node_color=nodecollist,
                 node_size=20.0,with_labels=False,edge_color='black',width=1)


axins = zoomed_inset_axes(ax, 20, loc=2)
subind = 1041#int(sys.argv[1])
subind = sublist.index(150692)
sub = sublist[subind]

axins.scatter(subs.cord[sub][0],subs.cord[sub][1],s=5000.0,c='black',
              marker='*')
sgraph = read_master_graph(tmpPath+'prim-master/',str(sub))
snodes = list(sgraph.nodes())
axins.set_aspect(1.2)
xpts = [nodepos[r][0] for r in snodes]
ypts = [nodepos[r][1] for r in snodes]
axins.scatter(xpts,ypts,s=100.0,c='royalblue')

axins.set_xlim(min(xpts),max(xpts))
axins.set_ylim(min(ypts),max(ypts))
axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
fig.savefig("{}{}.png".format(figPath,'partition-sub'),bbox_inches='tight')
