# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:30:14 2020

Author: Rounak
Description: Plots the histogram of time statistics for the primary network 
generation algorithm.
"""

import sys,os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


workPath = os.getcwd()
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"
inpPath = workPath + "/input/"

def get_binvar(path,sub):
    with open(path+str(sub)+'-master.txt') as f:
        lines = f.readlines()
    edgelist = []
    nodelabel = {}
    for line in lines:
        l = line.strip('\n').split('\t')
        edgelist.append((int(l[0]),int(l[6])))
        nodelabel[int(l[0])] = l[1]
        nodelabel[int(l[6])] = l[7]
    
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    nx.set_node_attributes(graph,nodelabel,'label')
    
    tsfr = len([n for n in list(graph.nodes()) if nodelabel[n]=='T'])
    return tsfr


with open(tmpPath+'prim-time.txt') as f:
    lines = [temp.strip('\n') for temp in f.readlines()]
sub_time = {int(line.split('\t')[0]): float(line.split('\t')[1]) \
            for line in lines}
sub_binvar = {k:get_binvar(tmpPath+'prim-master/',k) for k in sub_time}
sublist = list(sub_time.keys())

#%% Plot points
from matplotlib.colors import LogNorm
xpts = [sub_binvar[s] for s in sublist]
ypts = [sub_time[s] for s in sublist]

_, xbins = np.histogram(np.log10(np.array(xpts)),bins=5)
ypts_dict = {'grp'+str(i+1):[np.log10(sub_time[s]) for s in sublist \
                              if 10**xbins[i]<=sub_binvar[s]<=10**xbins[i+1]] \
              for i in range(len(xbins)-1)}

xmean = [int(round(10**((xbins[i]+xbins[i+1])/2))) for i in range(len(xbins)-1)]
# xmean = [200,400,800,1600,3200,6400,12800,25600]
xmean = [int(round(10**(xbins[i]))) for i in range(len(xbins)-1)]
ytick_old = np.linspace(-1,5,num=7)
ytick_new = 10**ytick_old

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
# ax.scatter(xpts,ypts,c='r',s=20.0,marker='D')
ax.boxplot(ypts_dict.values())
ax.set_xticklabels(xmean)
ax.set_yticklabels(ytick_new)
ax.set_xlabel('Number of transformers to be connected',fontsize=15)
ax.set_ylabel('Time (in seconds) to create primary network',
              fontsize=15)
fig.savefig("{}{}.png".format(figPath,'primnet-time'),bbox_inches='tight')
