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

#%% Functions to compare network statistics
def degree_dist(area_data,path):
    for area in area_data:
        synth = area_data[area]['synthetic']
        act = area_data[area]['actual']
        degree_sequence_a = sorted([d for n, d in synth.degree()],
                                   reverse=True)
        degree_sequence_b = sorted([d for n, d in act.degree()],
                                   reverse=True)
        na = synth.number_of_nodes()
        nb = act.number_of_nodes()
        sub = area_data[area]['root']
    
        degreeCount_a = collections.Counter(degree_sequence_a)
        degreeCount_b = collections.Counter(degree_sequence_b)
        deg_a = degreeCount_a.keys()
        deg_b = degreeCount_b.keys()
        
        max_deg = min(max(list(deg_a)),max(list(deg_b)))
        cnt_a = []
        cnt_b = []
        pa = []
        pb = []
        for i in range(1,max_deg+1):
            if i in degreeCount_a:
                cnt_a.append(100.0*degreeCount_a[i]/na)
                pa.append(degreeCount_a[i]/na)
            else:
                cnt_a.append(0)
                pa.append(0)
            if i in degreeCount_b:
                cnt_b.append(100.0*degreeCount_b[i]/nb)
                pb.append(degreeCount_b[i]/nb)
            else:
                cnt_b.append(0)
                pb.append(0)
        
        cnt_a = tuple(cnt_a)
        cnt_b = tuple(cnt_b)
        deg = np.arange(max_deg)+1
        width = 0.35
        
        # Create the degree distribution comparison
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        rects1 = ax.bar(deg, cnt_a, width, color='blue')
        rects2 = ax.bar(deg+width, cnt_b, width, color='orangered')
        ax.set_xticks(deg + width / 2)
        ax.set_xticklabels([str(x) for x in deg])
        ax.legend((rects1[0],rects2[0]),('Synthetic Network', 'Original Network'),
                  prop={'size': 20})
        ax.set_ylabel("Percentage of nodes",fontsize=20)
        ax.set_xlabel("Degree of nodes",fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        
        E = entropy(pa,pb)
        ax.text(max_deg-1,40,"KL-divergence: %.2f"%E,fontsize=20)
        
        # Save the figure
        filename = "degree-distribution-"+str(sub)
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return


from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    s = "{0:.1f}".format(100*y)
    return s

def hop_dist(area_data,path):
    """
    Creates the hop distribution of the networks. The synthetic network is compared
    with a base network. The hop distribution of both the networks is plotted 
    together in a stacked plot/histogram.

    Inputs: path: path to save the plot
    """
    for area in area_data:
        sub = area_data[area]['root']
        synth = area_data[area]['synthetic']
        act = area_data[area]['actual']
        h1 = []
        
        for n in list(synth.nodes()):
            hops = [nx.shortest_path_length(synth_net,n,s) if nx.has_path(synth_net,n,s) \
                    else 1e9 for s in sublist]
            h1.append(min(hops))
        w1 = np.ones_like(h1)/float(len(h1))
        h2 = [nx.shortest_path_length(act,n,sub) for n in list(act.nodes())]
        w2 = np.ones_like(h2)/float(len(h2))

        # Plot the hop distribution
        hops = [h1,h2]
        w = [w1,w2]
        
        colors = ['blue','orangered']
        labels = ['synthetic networ k','actual network']
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.hist(hops,weights=w,label=labels,color=colors)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.set_ylabel("Percentage of nodes",fontsize=20)
        ax.set_xlabel("Hops from root node",fontsize=20)
        ax.legend(prop={'size': 20})
        ax.tick_params(axis='both', labelsize=20)

        # Save the figure
        filename = "hop-dist-"+str(sub)
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return

def reach_dist(area_data,path):
    """
    Creates the hop distribution of the networks. The synthetic network is compared
    with a base network. The hop distribution of both the networks is plotted 
    together in a stacked plot/histogram.

    Inputs: path: path to save the plot
    """
    for area in area_data:
        sub = area_data[area]['root']
        synth = area_data[area]['synthetic']
        act = area_data[area]['actual']
        h1 = []
        
        for n in list(synth.nodes()):
            hops = [nx.shortest_path_length(synth_net,n,s,weight='length') if nx.has_path(synth_net,n,s) \
                    else 1e12 for s in sublist]
            h1.append(min(hops))
        w1 = np.ones_like(h1)/float(len(h1))
        h2 = [nx.shortest_path_length(act,n,sub,weight='geo_length') for n in list(act.nodes())]
        w2 = np.ones_like(h2)/float(len(h2))

        # Plot the hop distribution
        hops = [h1,h2]
        w = [w1,w2]
        colors = ['blue','orangered']
        labels = ['synthetic network','actual network']
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.hist(hops,weights=w,label=labels,color=colors)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.set_ylabel("Percentage of nodes",fontsize=20)
        ax.set_xlabel("Distance from root node (in meters)",fontsize=20)
        ax.legend(prop={'size': 20})
        ax.tick_params(axis='both', labelsize=20)

        # Save the figure
        filename = "reach-dist-"+str(sub)
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return

#%% Results of statistical comparisons
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

#areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}
print("Area Data extracted and stored")

#%% Distributions
degree_dist(area_data, figpath)
hop_dist(area_data, figpath)
reach_dist(area_data, figpath)
