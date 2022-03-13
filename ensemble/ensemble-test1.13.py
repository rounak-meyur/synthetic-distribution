# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program analyses the impact of PV penetration in the synthetic
distribution networks of Virginia.

Expectations: Long feeders should have overvoltage based on literature (??)
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
enspath = workpath + "/out/osm-ensemble/"


sys.path.append(libpath)
def GetDistNet(path,sub,i):
    return nx.read_gpickle(path+str(sub)+'-ensemble-'+str(i)+'.gpickle')

print("Imported modules")


sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

num_ens = 20


#%% Degree distribution

deg_data = {'count':[],'stack':[]}
hops_data = {'count':[],'stack':[]}
dist_data = {'count':[],'stack':[]}

deg_bins = [0,1,2,3,4,5]
hops_bins = np.arange(0,130,10)
dist_bins = np.arange(0,21,2)

for i in range(num_ens):
    deg = []
    hops = []
    dist = []
    size = 0
    for sub in sublist[1:]:
        net = GetDistNet(enspath,sub,i+1)
        size += len(net)
        deg.extend([nx.degree(net,n) for n in net])
        hops.extend([nx.shortest_path_length(net,n,sub) for n in net])
        dist.extend([nx.shortest_path_length(net,n,sub,'geo_length')/(1.6e3) \
                     for n in net])
    
    # Count for bars
    for i in range(len(deg_bins)-1):
        count = len([d for d in deg \
                     if deg_bins[i]<d<=deg_bins[i+1]])*(100/size)
        deg_data['count'].append(count)
        deg_data['stack'].append(deg_bins[i+1])
    
    for i in range(len(hops_bins)-1):
        count = len([d for d in hops \
                     if hops_bins[i]<d<=hops_bins[i+1]])*(100/size)
        hops_data['count'].append(count)
        hops_data['stack'].append(int((hops_bins[i]+hops_bins[i+1])/2))
    
    for i in range(len(dist_bins)-1):
        count = len([d for d in dist \
                     if dist_bins[i]<d<=dist_bins[i+1]])*(100/size)
        dist_data['count'].append(count)
        dist_data['stack'].append((dist_bins[i]+dist_bins[i+1])/2)


deg_df = pd.DataFrame(deg_data)
hops_df = pd.DataFrame(hops_data)
dist_df = pd.DataFrame(dist_data)



#%% Degree Distribution

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

# Draw the bar plot
ax = sns.barplot(data=deg_df, x="stack",y="count",ax=ax,color='salmon',
                 edgecolor="k",errwidth=20,capsize=0.1)

# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50)
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Degree of node",fontsize=60)
ax.set_title("Degree Distribution",fontsize=60)
ax.set_ylim(bottom=0,top=50)
fig.savefig("{}{}.png".format(figpath,'degree-ens20'),bbox_inches='tight')


#%% Hop Distribution
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

# Draw the bar plot
ax = sns.barplot(data=hops_df, x="stack",y="count",ax=ax,color='salmon',
                 edgecolor="k",errwidth=15,capsize=0.1)


xtix = np.arange(20,120,20)
xtix_loc = (xtix-5)/10
ax.set_xticks(xtix_loc)
ax.set_xticklabels([str(x) for x in xtix])

# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50)
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Hops from root node",fontsize=60)
ax.set_title("Hop Distribution",fontsize=60)
ax.set_ylim(bottom=0,top=30)
fig.savefig("{}{}.png".format(figpath,'hop-ens20'),bbox_inches='tight')

#%% Reach Distribution
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

# Draw the bar plot
ax = sns.barplot(data=dist_df, x="stack",y="count",ax=ax,color='salmon',
                 edgecolor="k",errwidth=25)

xtix = np.arange(4,21,4)
xtix_loc = (xtix-1)/2
ax.set_xticks(xtix_loc)
ax.set_xticklabels([str(x) for x in xtix])

# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50)
ax.set_ylabel("Percentage of nodes",fontsize=60)
ax.set_xlabel("Distance from substation (in miles)",fontsize=60)
ax.set_title("Reach Distribution",fontsize=60)
ax.set_ylim(bottom=0,top=60)
fig.savefig("{}{}.png".format(figpath,'dist-ens20'),bbox_inches='tight')


