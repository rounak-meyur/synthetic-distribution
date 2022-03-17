# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program plots the variation in number of path and star motifs
in the ensemble of networks. Networks belong to Montgomery county.
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
from pyResiliencelib import path,star
def GetDistNet(path,sub,i):
    return nx.read_gpickle(path+str(sub)+'-ensemble-'+str(i)+'.gpickle')

print("Imported modules")


sublist = [121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

num_ens = 20
k=4


#%% Count motifs

data_path = {'count':[],'size':[]}
data_star = {'count':[],'size':[]}

for i in range(num_ens):
    for sub in sublist:
        dist = GetDistNet(enspath,sub,i+1)
        data_star['count'].append(star(dist,k))
        data_path['count'].append(path(dist,k))
        data_star['size'].append(len(dist))
        data_path['size'].append(len(dist))
        


df_star = pd.DataFrame(data_star)
df_path = pd.DataFrame(data_path)

order = sorted(df_star["size"].unique())



#%% Star Motif Bar Plot

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

# Draw the bar plot
ax = sns.barplot(data=df_star, x="size",y="count",ax=ax,color='salmon',
                 edgecolor='k',errwidth=10.0)

xtix = np.arange(1,len(sublist),4)
ax.set_xticks(xtix)
ax.set_xticklabels([str(x+1) for x in xtix])

# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50)
ax.set_ylabel("Number of motifs",fontsize=60)
ax.set_xlabel("Network ID",fontsize=60)
ax.set_title(str(k)+"-node Star Motifs",fontsize=60)
fig.savefig("{}{}.png".format(figpath,'star-mot-ens20'),bbox_inches='tight')


#%% Path Motif Bar Plot
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

# Draw the bar plot
ax = sns.barplot(data=df_path, x="size",y="count",ax=ax,color='salmon',
                 edgecolor="k",errwidth=10.0)


xtix = np.arange(1,len(sublist),4)
ax.set_xticks(xtix)
ax.set_xticklabels([str(x+1) for x in xtix])

# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50)
ax.set_ylabel("Number of motifs",fontsize=60)
ax.set_xlabel("Network ID",fontsize=60)
ax.set_title(str(k)+"-node Path Motifs",fontsize=60)
fig.savefig("{}{}.png".format(figpath,'path-mot-ens20'),bbox_inches='tight')

