# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:12:36 2021

Author: Rounak
Description: Plots three statistical distributions to compare the network
statistics of the synthetic and actual networks.
"""
import os,sys
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import networkx as nx
import collections


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
actpath = rootpath + "/input/actual/"
synpath = rootpath + "/primnet/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet,get_areadata
print("Imported modules")

colors = ['blue','orangered']
labels = ['Synthetic Network','Actual Network']


#%% Functions to compare network statistics

from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    s = "{0:.1f}".format(100*y)
    return s




#%% Results of statistical comparisons
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
# areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}
print("Area Data extracted and stored")


#%% Degree Distribution Comparison
for area in area_data:
    
    sub = area_data[area]['root']
    synth = area_data[area]['synthetic']
    act = area_data[area]['actual']
    
    deg_act = [nx.degree(act,n) for n in act]
    hop_act = [nx.shortest_path_length(act,n,sub) for n in act]
    dist_act = [nx.shortest_path_length(act,n,sub,weight='geo_length')/(1.6e3) \
                for n in act]
    
    deg_synth = [nx.degree(synth,n) for n in synth]
    hop_synth = []
    dist_synth = []
    for n in synth:
        reach = [nx.shortest_path_length(synth_net,n,s,weight='length')/(1.6e3) \
              if nx.has_path(synth_net,n,s) else 1e12 for s in sublist]
        hops = [nx.shortest_path_length(synth_net,n,s) \
              if nx.has_path(synth_net,n,s) else 1e12 for s in sublist]
        hop_synth.append(min(hops))
        dist_synth.append(min(reach))
    
    w_deg_act = np.ones_like(deg_act)/float(len(deg_act))
    w_deg_synth = np.ones_like(deg_synth)/float(len(deg_synth))
    w_hop_act = np.ones_like(hop_act)/float(len(hop_act))
    w_hop_synth = np.ones_like(hop_synth)/float(len(hop_synth))
    w_dist_act = np.ones_like(dist_act)/float(len(dist_act))
    w_dist_synth = np.ones_like(dist_synth)/float(len(dist_synth))
    
    deg = [deg_act,deg_synth]
    hop = [hop_act,hop_synth]
    dist = [dist_act,dist_synth]
    
    w_deg = [w_deg_act, w_deg_synth]
    w_hop = [w_hop_act, w_hop_synth]
    w_dist = [w_dist_act, w_dist_synth]
    
    
    # Create the degree distribution comparison
    # n_synth = synth.number_of_nodes()
    # n_act = act.number_of_nodes()
    # max_deg = 5
    # p_deg_act = [(len([d for d in deg_act if d==i])+1)/n_act \
    #       for i in range(1,max_deg+1)]
    # p_deg_synth = [(len([d for d in deg_synth if d==i])+1)/n_synth \
    #       for i in range(1,max_deg+1)]
    # hop_min = range(0,120,10)
    # hop_max = range(10,130,10)
    # p_hop_act = [(len([h for h in hop_act \
    #                    if hop_min[i]<=h<=hop_max[i]])+1)/n_act \
    #              for i in range(len(hop_max))]
    # p_hop_synth = [(len([h for h in hop_synth \
    #                    if hop_min[i]<=h<=hop_max[i]])+1)/n_synth \
    #              for i in range(len(hop_max))]
    # dist_min = range(0,120,10)
    # dist_max = range(10,130,10)
    # p_dist_act = [(len([h for h in dist_act \
    #                    if dist_min[i]<=h<=dist_max[i]])+1)/n_act \
    #              for i in range(len(dist_max))]
    # p_dist_synth = [(len([h for h in dist_synth \
    #                    if dist_min[i]<=h<=dist_max[i]])+1)/n_synth \
    #              for i in range(len(dist_max))]
    
    # Create the degree distribution comparison
    max_deg = 5
    fig = plt.figure(figsize=(30,18))
    ax = fig.add_subplot(111)
    ax.hist(deg,bins=[x-0.5 for x in range(1,max_deg+1)],
            weights=w_deg,label=labels,color=colors)
    ax.set_xticks(range(1,max_deg+1))
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_ylabel("Percentage of nodes",fontsize=70)
    ax.set_xlabel("Degree of node",fontsize=70)
    ax.legend(fontsize=70,markerscale=3)
    ax.tick_params(axis='both', labelsize=60)
    
    # KL-divergence
    # E_deg = entropy(p_deg_act,p_deg_synth)
    # ax.text(3,0.4,"KL-divergence: %.2f"%E_deg,fontsize=60)
    
    # Save the figure
    filename = "degree-distribution-"+str(sub)
    fig.savefig("{}{}.png".format(figpath,filename),bbox_inches='tight')
    
    
    # Create the hop distribution comparison
    fig = plt.figure(figsize=(30,18))
    ax = fig.add_subplot(111)
    ax.hist(hop,weights=w_hop,label=labels,color=colors)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_ylabel("Percentage of nodes",fontsize=70)
    ax.set_xlabel("Hops from root node",fontsize=70)
    ax.legend(fontsize=70,markerscale=3)
    ax.tick_params(axis='both', labelsize=60)
    
    # KL-divergence
    # E_hop = entropy(p_hop_act,p_hop_synth)
    # ax.text(3,0.4,"KL-divergence: %.2f"%E_hop,fontsize=60)
    
    # Save the figure
    filename = "hop-distribution-"+str(sub)
    fig.savefig("{}{}.png".format(figpath,filename),bbox_inches='tight')
    
    
    # Create the hop distribution comparison
    fig = plt.figure(figsize=(30,18))
    ax = fig.add_subplot(111)
    ax.hist(dist,weights=w_hop,label=labels,color=colors)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_ylabel("Percentage of nodes",fontsize=70)
    ax.set_xlabel("Distance from root node (in miles)",fontsize=70)
    ax.legend(fontsize=70,markerscale=3)
    ax.tick_params(axis='both', labelsize=60)
    
    # KL-divergence
    # E_dist = entropy(p_dist_act,p_dist_synth)
    # ax.text(3,0.4,"KL-divergence: %.2f"%E_dist,fontsize=60)
    
    # Save the figure
    filename = "dist-distribution-"+str(sub)
    fig.savefig("{}{}.png".format(figpath,filename),bbox_inches='tight')