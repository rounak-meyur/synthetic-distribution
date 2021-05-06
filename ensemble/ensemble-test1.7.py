# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:16:52 2021

@author: rouna
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
actpath = rootpath + "/input/actual/"
synpath = rootpath + "/primnet/out/"
enspath = workpath + "/out/"

sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
from pyGeometrylib import Grid,partitions,MeasureDistance
print("Imported modules")




sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)

sub = 121144
opt_net = GetDistNet(synpath,sub)

#%% Efficiency computation
from itertools import combinations

def compute_efficiency(net):
    nodes = [n for n in net.nodes if net.nodes[n]['label']=='T']
    node_pairs = list(combinations(nodes, 2))
    eff = 0.0
    for pair in node_pairs:
        length = nx.shortest_path_length(net,pair[0],
                                    pair[1],'geo_length')
        distance = MeasureDistance(net.nodes[pair[0]]['cord'],
                               net.nodes[pair[1]]['cord'])
        eff += distance/length if length != 0 else 0.0
    return eff/len(node_pairs)

all_eff = [compute_efficiency(opt_net)]
for i in range(20):
    print(i+1)
    graph = nx.read_gpickle(enspath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
    all_eff.append(compute_efficiency(graph))



#%% Actual network
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}

    
def compute_act_efficiency(net,df_buses):
    nodes = [n for n in net.nodes]
    node_pairs = list(combinations(nodes, 2))
    eff = 0.0
    for pair in node_pairs:
        length = nx.shortest_path_length(net,pair[0],
                                    pair[1],'geo_length')
        pt1 = df_buses.loc[df_buses.id == str(pair[0])]['geometry'].values[0]
        pt2 = df_buses.loc[df_buses.id == str(pair[1])]['geometry'].values[0]
        distance = MeasureDistance([pt1.x,pt1.y],[pt2.x,pt2.y])
        eff += distance/length if length != 0 else 0.0
    return eff,len(node_pairs)


num = 0.0
den = 0.0
for area in area_data:
    act_net = area_data[area]['actual']
    df_buses = area_data[area]['df_buses']
    act_eff,num_pairs = compute_act_efficiency(act_net,df_buses)
    num += act_eff
    den += num_pairs

act_eff = num/den

all_eff = [act_eff]+all_eff
#%% Plot efficiency

DPI = 72    
fig = plt.figure(figsize=(1000/DPI, 600/DPI), dpi=DPI)
ax = fig.add_subplot(111)
ax.bar(range(1,22),all_eff,color='blue')
ax.set_xticks(range(1,23))
ax.set_xticklabels(["Act. net","Optm. net"]+["Ensm. net "+str(x) for x in range(1,21)],
                   rotation=90)
ax.set_ylabel("Efficiency")









