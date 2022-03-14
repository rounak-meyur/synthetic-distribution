# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak

Description: This program plots the results of motifs in distribution network 
of Virginia.
"""

import sys,os
import matplotlib.pyplot as plt



workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
distpath = workpath + "/out/osm-primnet/"

sys.path.append(libpath)
print("Imported modules")


#%% Get substations in rural and urban regions
sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
with open(inppath+"urban-sublist.txt") as f:
    urban_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]
with open(inppath+"rural-sublist.txt") as f:
    rural_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]

rural_sub = [s for s in sublist if s in rural_sublist]
urban_sub = [s for s in sublist if s in urban_sublist]


#%% Get motif data in rural and urban areas

k = 5

with open(workpath+"/out/"+str(k)+"star-motif.txt") as f:
    lines = f.readlines()
kstar_motif = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    kstar_motif[int(temp[0])] = (int(temp[1]),int(temp[2]))
    
with open(workpath+"/out/"+str(k)+"path-motif.txt") as f:
    lines = f.readlines()
kpath_motif = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    kpath_motif[int(temp[0])] = (int(temp[1]),int(temp[2]))


def scatter_plot(subs,ax,motifs,color='r',size=10,label='all areas'):
    all_nodes = [motifs[s][0] for s in subs]
    star_motifs = [motifs[s][1] for s in subs]
    ax.scatter(all_nodes,star_motifs,c=color,s=size,label=label)
    return


#%% 
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111)

scatter_plot(urban_sub,ax,kstar_motif,color='crimson',size=300,label='urban areas')
scatter_plot(rural_sub,ax,kstar_motif,color='slateblue',size=300,label='rural areas')

ax.set_xlabel("Size of network",fontsize=70)
ax.set_ylabel("Number of motifs",fontsize=70)
ax.set_title(str(k)+"-node star motifs",fontsize=70)

ax.legend(prop={'size': 70})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,str(k)+'star-motif-comp'),bbox_inches='tight')





fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111)

scatter_plot(urban_sub,ax,kpath_motif,color='crimson',size=300,label='urban areas')
scatter_plot(rural_sub,ax,kpath_motif,color='slateblue',size=300,label='rural areas')

ax.set_xlabel("Size of network",fontsize=70)
ax.set_ylabel("Number of motifs",fontsize=70)
ax.set_title(str(k)+"-node path motifs",fontsize=70)

ax.legend(prop={'size': 70})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,str(k)+'path-motif-comp'),bbox_inches='tight')
