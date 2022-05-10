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
sub = 147793
opt_net = GetDistNet(distpath,sub)
road_net = GetPrimRoad(tsfrpath,sub)


# Create new graph representing a realization of the network
# this graph changes for each state in the Markov chain
def create_tempGraph(parent):
    g = nx.Graph()
    g.add_edges_from(parent.edges)
    for n in g:
        g.nodes[n]['label'] = parent.nodes[n]['label']
        g.nodes[n]['load'] = parent.nodes[n]['load']
        g.nodes[n]['cord'] = parent.nodes[n]['cord']
    for e in g.edges:
        g.edges[e]['label'] = parent.edges[e]['label']
        g.edges[e]['geometry'] = parent.edges[e]['geometry']
        g.edges[e]['length'] = parent.edges[e]['length']
        g.edges[e]['r'] = parent.edges[e]['r']
        g.edges[e]['x'] = parent.edges[e]['x']
        g.edges[e]['type'] = parent.edges[e]['type']
    return g




#%% Create ensemble of networks

ens_size = 20

np.random.seed(1234)
i = 0
temp = create_tempGraph(opt_net)
# Run loop to create networks
while(i<ens_size):
    # Create dummy road graph for next variant generation
    dummy = create_dummy(road_net,temp,sub)
    
    # Solve the restricted MILP
    M = MILP_primstruct(dummy,grbpath=workpath+"/out/")
    new_prim = M.solve()
    print("Edge(s) to be added:",[e for e in new_prim if e not in temp.edges])
    
    # Finalize the network
    if new_prim != []:
        synth = create_variant_network(temp,dummy,new_prim)
        print("Variant Network",i+1,"constructed\n\n")
        nx.write_gpickle(synth,outpath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
        i += 1
        temp = create_tempGraph(synth)



#%% Compare the networks through voltage at residence

# df = pd.DataFrame(dict_var)
# df = df.pivot("variant","home","voltage")

# fig = plt.figure(figsize=(50,20))
# ax = fig.add_subplot(111)

# # Colorbar axes
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=1)
# cax.set_ylabel("Number of times switched",fontsize=40)
# cax.tick_params(labelsize=40)

# # Format other stuff
# num_col = df.shape[1]
# ax = sns.heatmap(df,ax=ax,vmin=0.97,vmax=1,cmap="viridis",
#                  cbar_ax=cax,xticklabels=int(num_col/8))
# ax.tick_params(axis='y',labelsize=40)
# ax.tick_params(axis='x',labelsize=40)
# ax.set_ylabel("Variant network",fontsize=40)
# ax.set_xlabel("Residences in network",fontsize=40)
# ax.set_title("Sensistivity Analysis",fontsize=50)

#%% Run power flow on ensemble of networks

dict_var = {'variant':[],'voltage':[], 'sub':[]}
sublist = [121155,147793]
for sub in sublist:
    opt_net = GetDistNet(distpath,sub)
    powerflow(opt_net)
    res_nodes = [n for n in opt_net if opt_net.nodes[n]['label']=='H']
    for n in res_nodes:
        dict_var['sub'].append(sub)
        dict_var['variant'].append(0)
        dict_var['voltage'].append(opt_net.nodes[n]['voltage'])
    for i in range(ens_size):
        synth = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
        powerflow(synth)
        for n in res_nodes:
            dict_var['sub'].append(sub)
            dict_var['variant'].append(i+1)
            dict_var['voltage'].append(synth.nodes[n]['voltage'])


#%% Bar Plot
df = pd.DataFrame(dict_var)
fig = plt.figure(figsize=(35,20))

df1 = df[df['sub']==121155]
ax1 = fig.add_subplot(1,1,1)

# Format other stuff
ax1 = sns.lineplot(data=df1,x="variant",y="voltage",palette = 'red',
                  ax=ax1,lw=5.0,linestyle='solid',marker='o',markersize=20,
                  err_style='bars',err_kws={'capsize':10.0})
ax1.tick_params(axis='y',labelsize=40)
ax1.tick_params(axis='x',labelsize=40)
ax1.set_ylabel("Residence voltage (in pu)",fontsize=40)
ax1.set_xlabel("Network variants",fontsize=40)

ax1.set_xticks(range(0,21,4))
ax1.set_xticklabels(['Opt. nw.']+[str(x) for x in range(4,21,4)])

fig.savefig("{}{}.png".format(figpath,'ens-voltcomp'),bbox_inches='tight')

# df2 = df[df['sub']==147793]
# ax2 = fig.add_subplot(1,2,2)

# # Format other stuff
# ax2 = sns.lineplot(data=df2,x="variant",y="voltage",palette='blue',
#                   ax=ax2,lw=3.0,linestyle='dashed',marker='o',markersize=20,
#                   err_style='bars',err_kws={'capsize':10.0})
# ax2.tick_params(axis='y',labelsize=40)
# ax2.tick_params(axis='x',labelsize=40)
# ax2.set_ylabel("Residence voltage (in pu)",fontsize=40)
# ax2.set_xlabel("Network variants",fontsize=40)

# ax2.set_xticks(range(0,21,4))
# ax2.set_xticklabels(['Opt. nw.']+[str(x) for x in range(4,21,4)])




#%%


