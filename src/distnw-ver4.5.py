# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx
import numpy as np
# from scipy import stats

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/prim-ensemble/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network


#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble
import matplotlib.pyplot as plt


sub = 24665
theta_range = range(300,401,5)
phi_range = range(3,11)


Fdist = {}
Hops = {}
for theta in theta_range:
    for phi in phi_range:
        # Create the cumulative distribution for a given (theta,phi) pair
        fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
        graph = read_network(tmpPath+fname+'.txt',homes)
        
        
        N = graph.number_of_nodes()
        Hops[(theta,phi)] = np.array([nx.shortest_path_length(graph,n,sub) \
                                      for n in list(graph.nodes())])
        Fdist[(theta,phi)] = np.array([np.sum(Hops[(theta,phi)]<=k)/N \
                                       for k in range(100)])
    
#%%Plot the distributions
fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(1,1,1)

for theta in theta_range:
    for phi in phi_range:
        ax.plot(Fdist[(theta,phi)],color=((phi-3)/7.0,(theta-300)/100.0,0.0))

ax.set_xlabel("Number of hops from the root",fontsize=15)
ax.set_ylabel("Cumulative distribution of nodes",fontsize=15)


#%% Cluster the cpdfs
from scipy import stats

thresh = 0.01
kmax = 10
index_list =list(Hops.keys()) 
centroid = [index_list[0]]
Dclus = 10

cluster = {index_list[0]:index_list}

while (Dclus>thresh) and (len(centroid)<kmax):
    # intra-cluster distance computation
    for i in range(len(centroid)):
        D = [stats.ks_2samp(Hops[l],Hops[centroid[i]])[0] for l in cluster[centroid[i]]]
        Dclus = max(D)
        if (Dclus>thresh) and (len(centroid)<kmax):
            centroid.append(cluster[centroid[i]][np.argmax(np.array(D))])
    
    # centroid to cpdf distance computation
    cluster = {k:[] for k in centroid}
    for j in range(len(Hops)):
        # iterate over each cpdf
        Dstat = []
        for i in range(len(centroid)):
            # find distance of cpdf from each cluster centroid
            Dstat.append(stats.ks_2samp(Hops[index_list[j]],Hops[centroid[i]])[0])
        # Find the cluster to which the cpdf belongs
        cluster[centroid[np.argmin(np.array(Dstat))]].append(index_list[j]) 


#%%Plot the clustered distributions
import seaborn as sns
colors = sns.color_palette(n_colors=len(cluster))
fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(1,1,1)

for i,cpdf_list in enumerate(list(cluster.values())):
    for c in cpdf_list:
        ax.plot(Fdist[c],color=colors[i])

ax.set_xlabel("Number of hops from the root",fontsize=15)
ax.set_ylabel("Cumulative distribution of nodes",fontsize=15)
fig.savefig("{}{}.png".format(figPath,str(sub)+'-hopdist-cluster'),bbox_inches='tight')


#%% Plot the KDE
fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(1,1,1)

for i,cpdf_list in enumerate(list(cluster.values())):
    for c in cpdf_list:
        sns.kdeplot(Hops[c],shade=False,color=colors[i])

ax.set_xlabel("Number of hops from the root",fontsize=15)
ax.set_ylabel("Kernel density of hop distribution",fontsize=15)
fig.savefig("{}{}.png".format(figPath,str(sub)+'-hopdist-kde-cluster'),bbox_inches='tight')











































