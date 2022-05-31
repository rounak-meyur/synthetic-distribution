# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:14:58 2021

Author: Rounak
"""

import sys,os
import numpy as np
import networkx as nx
workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
secpath = rootpath + "/secnet/out/osm-sec-network/"


sys.path.append(libpath)
print("Imported modules")
from pyExtractDatalib import GetDistNet
from pyUtilsSCAlib import make_Zbus
from pyMiscUtilslib import powerflow


def get_nodes(dist,s,com=0,label='H'):
    tree = nx.dfs_tree(dist,s)
    reg_nodes = list(nx.neighbors(tree, s))
    return [n for n in nx.descendants(tree, reg_nodes[com]) \
                 if dist.nodes[n]['label']==label]

def get_edges(dist,s,com=0):
    tree = nx.dfs_tree(dist,s)
    reg_nodes = list(nx.neighbors(tree, s))
    des = [n for n in nx.descendants(tree, reg_nodes[com])] + [reg_nodes[com]]
    return tree.edges(des)


#%% Display example network
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
#%% Network topology and pre fault voltages

# Load network
distpath = outpath + "osm-primnet/"
sub = 121144
synth_net = GetDistNet(distpath,sub)




# Get Zbus matrix
Zbus,Nodes = make_Zbus(synth_net,sub)

# Get prefault voltages
powerflow(synth_net)
V_prefault = np.array([synth_net.nodes[n]['voltage'] for n in Nodes])

#%% Short Circuit Analysis

Delta_V = Zbus @ np.diag(V_prefault/np.diagonal(Zbus.todense()))
V_fault = V_prefault.reshape(-1,1) - Delta_V

dict_V_fault = {n:{m:abs(V_fault[i,j]) for j,m in enumerate(Nodes)} \
                for i,n in enumerate(Nodes)}


conv = 0.000621371 # meter to mile conversion
node_loc = {n: nx.shortest_path_length(synth_net,sub,n,'length')*conv \
            for n in Nodes+[sub]}

#%% Plot 3D
import matplotlib.pyplot as plt

nodelist = get_nodes(synth_net,sub)
edgelist = get_edges(synth_net,sub)

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111,projection="3d")

for n in nodelist[:5]:
    xline = [node_loc[n],node_loc[n]]
    for e in edgelist:
        yline = [node_loc[e[0]],node_loc[e[1]]]
        zline = [dict_V_fault[n][e[0]],dict_V_fault[n][e[1]]]
        if synth_net.edges[e]['label'] == 'P':
            ax.plot3D(xline, yline, zline,'b--')
        elif synth_net.edges[e]['label'] == 'S':
            ax.plot3D(xline, yline, zline,'r--')
        else:
            print("Something is wrong!!!")
    
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='z', labelsize=20)
ax.set_xlabel("Distance of fault node from substation (miles)",fontsize=40)
ax.set_ylabel("Distance of node from substation (miles)",fontsize=40)
ax.set_zlabel("Post fault voltage at node (pu)",fontsize=40)

#%% Plot 2D
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)

for n in Nodes[2:3]:
    for e in edgelist:
        xline = [node_loc[e[0]],node_loc[e[1]]]
        yline = [dict_V_fault[n][e[0]],dict_V_fault[n][e[1]]]
        if synth_net.edges[e]['label'] != 'P':
            ax.plot(xline, yline,'b')
        elif synth_net.edges[e]['label'] != 'S':
            ax.plot(xline, yline,'r')
    
ax.tick_params(axis='x', labelsize=50)
ax.tick_params(axis='y', labelsize=50)
ax.set_xlabel("Distance of node from substation (miles)",fontsize=60)
ax.set_ylabel("Post fault voltage at node (pu)",fontsize=60)











