# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program uses a Markov chain to create synthetic networks which
are solutions of the optimization program. Reformulates the original MILP with
altered constraints.
"""

import sys,os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
tsfrpath = rootpath + "/secnet/out/osm-prim-road/"
distpath = rootpath + "/primnet/out/osm-primnet/"
outpath = workpath + "/out/osm-ensemble-mc/"
sys.path.append(libpath)

from pyExtractDatalib import GetDistNet,GetPrimRoad
from pyMiscUtilslib import create_variant_network, powerflow
from pyEnsemblelib import MILP_primstruct, create_dummy


#%% Load the network data
sub = 121155
opt_net = GetDistNet(distpath,sub)
road_net = GetPrimRoad(tsfrpath,sub)


# Create new graph representing a realization of the network
# this graph changes for each state in the Markov chain
synth = nx.Graph()
synth.add_edges_from(opt_net.edges)
for n in synth:
    synth.nodes[n]['label'] = opt_net.nodes[n]['label']
    synth.nodes[n]['load'] = opt_net.nodes[n]['load']
    synth.nodes[n]['cord'] = opt_net.nodes[n]['cord']
for e in synth.edges:
    synth.edges[e]['label'] = opt_net.edges[e]['label']
    synth.edges[e]['geometry'] = opt_net.edges[e]['geometry']
    synth.edges[e]['length'] = opt_net.edges[e]['length']




#%% Create ensemble of networks
# Initial edgelist
np.random.seed(1234)

dict_var = {'home':[],'variant':[],'voltage':[]}
res_nodes = [n for n in opt_net if opt_net.nodes[n]['label']=='H']
h_idx = {h:k+1 for k,h in enumerate(res_nodes)}





i = 0
# Run loop to create networks
while(i<5):
    # Run powerflow and store voltage data
    powerflow(synth)
    for n in res_nodes:
        dict_var['home'].append(h_idx[n])
        dict_var['variant'].append(i)
        dict_var['voltage'].append(synth.nodes[n]['voltage'])
    
    # Create dummy road graph for next variant generation
    dummy = create_dummy(road_net,synth,sub)
    
    # Solve the restricted MILP
    M = MILP_primstruct(dummy,grbpath=workpath+"/out/")
    new_prim = M.solve()
    print("Edge(s) to be added:",[e for e in new_prim if e not in synth.edges])
    
    # Finalize the network
    if new_prim != []:
        synth = create_variant_network(synth,dummy,new_prim)
        print("Variant Network",i+1,"constructed\n\n")
        nx.write_gpickle(synth,outpath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
        i += 1

df = pd.DataFrame(dict_var)
df = df.pivot("variant","home","voltage")

#%% Compare the networks through voltage at residence

fig = plt.figure(figsize=(50,20))
ax = fig.add_subplot(111)

# Colorbar axes
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=1)
cax.set_ylabel("Number of times switched",fontsize=40)
cax.tick_params(labelsize=40)

# Format other stuff
num_col = df.shape[1]
ax = sns.heatmap(df,ax=ax,vmin=0.97,vmax=1,cmap="viridis",
                 cbar_ax=cax,xticklabels=int(num_col/8))
ax.tick_params(axis='y',labelsize=40)
ax.tick_params(axis='x',labelsize=40)
ax.set_ylabel("Variant network",fontsize=40)
ax.set_xlabel("Residences in network",fontsize=40)
# ax.set_title("Sensistivity Analysis",fontsize=50)

#%% Line plot

df = pd.DataFrame(dict_var)
fig = plt.figure(figsize=(50,20))
ax = fig.add_subplot(111)

# Format other stuff
num_group = len(df.variant.unique())
colors = sns.color_palette("Set2")[:num_group]
ax = sns.lineplot(df,x="variant",y="voltage",palette='red',
                 ax=ax,lw=3.0,linestyle='dashed',marker='o',markersize=20,
                 err_style='bars',err_kws={'capsize':10.0})
ax.tick_params(axis='y',labelsize=40)
ax.tick_params(axis='x',labelsize=40)
ax.set_ylabel("Variant network",fontsize=40)
ax.set_xlabel("Residences in network",fontsize=40)
# ax.set_title("Sensistivity Analysis",fontsize=50)


