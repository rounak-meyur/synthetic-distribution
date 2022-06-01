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


def get_nodes(dist,s,com=0,labels=['H']):
    tree = nx.dfs_tree(dist,s)
    reg_nodes = list(nx.neighbors(tree, s))
    return [n for n in nx.descendants(tree, reg_nodes[com]) \
                 if dist.nodes[n]['label'] in labels]



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



conv = 0.000621371 # meter to mile conversion
node_loc = {n: nx.shortest_path_length(synth_net,sub,n,'length')*conv \
            for n in Nodes+[sub]}

#%% Plot bar plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch




# Get residential nodes of interest
com=4
res_interest = get_nodes(synth_net,sub,com=com)
node_interest = get_nodes(synth_net,sub,com=com,labels=['H','T','R'])
N = len(res_interest)

locs = [node_loc[n] for n in node_interest]
l_min = min(locs)
l_max = max(locs)
l_dif = l_max-l_min
num_range = 5
l_range = [l_min+((x/num_range)*l_dif) for x in range(num_range+1)]

# Initialize data for pandas dataframe
data = {'count':[],'stack':[],'loc':[]}

# Range of voltages in pu
v_range = [0.3,0.6,0.9]
v_str = ["< "+str(v_range[0])+" p.u."] \
    + [str(v_range[i])+"-"+str(v_range[i+1])+" p.u." \
              for i in range(len(v_range)-1)]

# Range of location in miles
l_nodes = {"{:.1f}".format(l_range[i])+"-"+"{:.1f}".format(l_range[i+1]): \
           [n for n in node_interest if l_range[i]<=node_loc[n]<=l_range[i+1]]\
              for i in range(len(l_range)-1)}

# Fill in the dictionary for plot data
for loc in l_nodes:
    for n in l_nodes[loc]:
        j = Nodes.index(n)  # get index of fault node in the Zbus matrix
        v_fault = abs(V_fault[:,j])
        for i in range(len(v_range)):
            num_res = len([n for n in res_interest \
                           if v_fault[Nodes.index(n)]<=v_range[i]])
            
            data['loc'].append(loc)
            data['count'].append(100.0*num_res/N)
            data['stack'].append(v_str[i])

        
df = pd.DataFrame(data)

# Draw the bar plot
num_stack = len(v_str)
colors = sns.color_palette("Spectral")[:num_stack]


fig = plt.figure(figsize=(30,25))
ax = fig.add_subplot(111)
for i,g in enumerate(df.groupby("stack",sort=False)):
    ax = sns.barplot(data=g[1], x="loc",y="count",hue="stack",
                          palette=[colors[i]],ax=ax,
                          zorder=-i, edgecolor="k",errwidth=5,capsize=0.1)


# Format other stuff
ax.tick_params(axis='y',labelsize=50)
ax.tick_params(axis='x',labelsize=50,rotation=0)
ax.set_ylabel("Percentage of residences",fontsize=60)
ax.set_xlabel("Location of three phase fault from substation (miles)",fontsize=60)
ax.set_ylim(bottom=0,top=80)

han = [Patch(facecolor=color, edgecolor='black', label=label) \
              for label, color in zip(v_str, colors)]
ax.legend(handles=han,ncol=1,prop={'size': 50},loc='upper right')

fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-sca.png",
            bbox_inches='tight')











