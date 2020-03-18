# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx
import numpy as np
from scipy import stats

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/prim-ensemble/"
tmpPath = workPath + "/temp/prim-ensemble/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network

#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble
# sub = 24664
# param_range = range(100,601,5)

# dict_hops = {}
# for i,theta in enumerate(param_range):
#     print("Extracting:",i)
#     fname = str(sub)+'-network-f-'+str(theta)+'-s-8'
#     dist_net = read_network(tmpPath+fname+'.txt',homes)
#     dict_hops[theta] = [nx.shortest_path_length(dist_net,n,sub) for n in list(dist_net.nodes())]

# print("Extracted Network Data")


#%% Heat map of Statistic
# import pandas as pd

# K = np.zeros(shape=(len(param_range),len(param_range)))
# for i,lambda1 in enumerate(param_range):
#     for j,lambda2 in enumerate(param_range):
#         print(i,j)
#         K[i,j] = stats.ks_2samp(dict_hops[lambda1],dict_hops[lambda2])[0]

# K_df = pd.DataFrame(data=K)
# K_df.to_csv(csvPath+'K-stat.csv')


#%% Read from csv
from numpy import genfromtxt
K = genfromtxt(csvPath+'K-stat.csv', delimiter=',')

#%% Heat map generation
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(20,20))
# ax = fig.add_subplot(111)

# # We want to show all ticks...
# ticks = range(0,101,10)
# ticklabel = [str(5*x+100) for x in ticks]

# cbarlabel="Kolmogorov Smirnov Statistic"
# im = ax.imshow(K, cmap="magma_r")

# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=30)
# cbar.ax.tick_params(labelsize=30)


# # Axis tick labels
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(ticklabel)
# ax.set_yticklabels(ticklabel)
# ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,labelsize=20)

# # Axis labels
# ax.set_xlabel("Maximum allowable flow in kVA",fontsize=30)
# ax.xaxis.set_label_position('top')
# ax.set_ylabel("Maximum allowable flow in kVA",fontsize=30)

# # Turn spines off and create white grid.
# for edge, spine in ax.spines.items():
#     spine.set_visible(False)

# ax.set_xticks(np.arange(K.shape[1]+1)-.5, minor=True)
# ax.set_yticks(np.arange(K.shape[0]+1)-.5, minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
# ax.tick_params(which="minor", bottom=False, left=False)

# ax.set_title("Heatmap of Kolmogorov Smirnov Statistic for comparing hop distribution of synthetic networks",
#              fontsize=20)


#%% K statistics
diff_theta = []
K_stat = []
mean_K_stat = []
median_K_stat = []
for i in range(np.shape(K)[0]):
    diff_theta.append(i)
    K_stat.append(np.diagonal(K,offset=i))
    mean_K_stat.append(np.mean(np.diagonal(K,offset=i)))
    median_K_stat.append(np.median(np.diagonal(K,offset=i)))
xticks = [diff_theta[i] for i in range(0,101,10)]
xticklabels = [str(x) for x in xticks]


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.set_xlabel('Difference between parameters',fontsize=15)
ax.set_ylabel('Kolmogorov Smirnov Statistic',fontsize=15)
ax.set_title('Kolmogorov Smirnov Statistic dependence on difference between generation parameters',
             fontsize=20)
ax.boxplot(K_stat)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,labelsize=15)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

fig2 = plt.figure(figsize=(15,6))
ax2 = fig.add_subplot(212)
ax2.set_xlabel('Difference between parameters',fontsize=15)
ax2.set_ylabel('Kolmogorov Smirnov Statistic',fontsize=15)
ax2.set_title('Kolmogorov Smirnov Statistic dependence on difference between generation parameters',
             fontsize=20)
ax2.plot(diff_theta,median_K_stat,marker='*',markersize=5,color='b',label='median of statistic')
ax2.plot(diff_theta,mean_K_stat,marker='*',markersize=5,color='r', label='mean of statistic')
ax2.legend(loc='best',prop={'size': 20})
ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,labelsize=15)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)
ax2.grid(b=True)