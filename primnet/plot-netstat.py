# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak

Description: This program plots the results of network statistics in 
distribution network of Virginia. These include the degree, hop and reach
distributions
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
distpath = workpath + "/out/osm-primnet/"

sys.path.append(libpath)
print("Imported modules")



def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{0:.1f}".format(100*y)
    return s


#%% Get substations in rural and urban regions
sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
with open(inppath+"urban-sublist.txt") as f:
    urban_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]
with open(inppath+"rural-sublist.txt") as f:
    rural_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]


rural_sub = [s for s in sublist if s in rural_sublist]
urban_sub = [s for s in sublist if s in urban_sublist]

colors = ['slateblue','crimson']
labels = ['Rural Areas','Urban Areas']

#%% Get the data

with open(workpath + "/out/network-stats.txt") as f:
    lines = f.readlines()

deg1 = [int(x) for x in lines[0].strip('\n').split(' ')]
deg2 = [int(x) for x in lines[1].strip('\n').split(' ')]
hops1 = [int(x) for x in lines[2].strip('\n').split(' ')]
hops2 = [int(x) for x in lines[3].strip('\n').split(' ')]
dist1 = [float(x) for x in lines[4].strip('\n').split(' ')]
dist2 = [float(x) for x in lines[5].strip('\n').split(' ')]

hops = [hops1,hops2]
dist = [[d/1.6e3 for d in dist1],[d/1.6e3 for d in dist2]]
deg = [deg1,deg2]


w1_hops = np.ones_like(hops1)/float(len(hops1))
w1_dist = np.ones_like(dist1)/float(len(dist1))
w1_deg = np.ones_like(deg1)/float(len(deg1))
w2_hops = np.ones_like(hops2)/float(len(hops2))
w2_dist = np.ones_like(dist2)/float(len(dist2))
w2_deg = np.ones_like(deg2)/float(len(deg2))
w_hops = [w1_hops,w2_hops]
w_dist = [w1_dist,w2_dist]
w_deg = [w1_deg,w2_deg]

#%% Degree distribution
max_deg = 6
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
ax.hist(deg,bins=[x-0.5 for x in range(1,max_deg)],
        weights=w_deg,label=labels,color=colors)
ax.set_xticks(range(1,max_deg))
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Degree of node",fontsize=60)
ax.legend(prop={'size': 60})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,'deg-comp'),bbox_inches='tight')

#%% Hop distribution

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
ax.hist(hops,weights=w_hops,label=labels,color=colors)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Hops from root node",fontsize=60)
ax.legend(prop={'size': 60})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,'hop-comp'),bbox_inches='tight')

#%% Distance

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
ax.hist(dist,weights=w_dist,label=labels,color=colors)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Distance (in miles) from root node",fontsize=60)
ax.legend(prop={'size': 60})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,'dist-comp'),bbox_inches='tight')