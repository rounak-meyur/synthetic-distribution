# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program analyses the impact of PV penetration in the synthetic
distribution networks of Virginia.

Expectations: Long feeders should have overvoltage based on literature (??)
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
enspath = workpath + "/out/osm-ensemble/"


sys.path.append(libpath)
from pyMiscUtilslib import powerflow,assign_linetype

print("Imported modules")




#%% PV Hosting capacity for random residences
def GetDistNet(path,sub,i):
    return nx.read_gpickle(path+str(sub)+'-ensemble-'+str(i)+'.gpickle')


def run_powerflow(sub,i,homes,rating):
    """
    Runs the power flow on the network with the installed rooftop solar panels.

    Parameters
    ----------
    sub : integer type
        substation ID.
    homes : list type
        list of node IDs where solar panels are installed.
    rating : float type
        power rating of solar panel in kW.

    Returns
    -------
    voltages : TYPE
        DESCRIPTION.
    flows : TYPE
        DESCRIPTION.

    """
    graph = GetDistNet(enspath,sub,i)
    assign_linetype(graph)
    for h in homes:
        graph.nodes[h]['load'] = graph.nodes[h]['load'] - (rating*1e3)
    powerflow(graph)
    voltages = [graph.nodes[n]['voltage'] for n in graph]
    return voltages



#%%

def draw_barplot(df,groups,ax):
    
    # Draw the bar plot
    num_stack = len(groups)
    colors = sns.color_palette("Set3")[:num_stack]
    pen = df.pen.unique()
    for i,g in enumerate(list(df.groupby("stack",sort=False))[::-1]):
        ax = sns.barplot(data=g[1], x="pen",y="count",hue="group",
                              palette=[colors[i]],ax=ax,
                              zorder=-i, edgecolor="k",errwidth=5)
    
    
    # Format other stuff
    ax.set_xticklabels([100.0*p for p in pen])
    ax.tick_params(axis='y',labelsize=40)
    ax.tick_params(axis='x',labelsize=40)
    ax.set_ylabel("Percentage of nodes (%)",fontsize=50)
    ax.set_xlabel("Percentage Penetration (%)",fontsize=50)
    ax.set_title("PV penetration in urban feeder",fontsize=50)
    # ax.set_ylim(bottom=0,top=1000)


    hatches = itertools.cycle(['/', ''])
    for i, bar in enumerate(ax.patches):
        if i%(len(pen)) == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)


    han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                  for label, color in zip(groups, colors)]
    han2 = [Patch(facecolor="white",edgecolor='black',
                  label="MV level penetration",hatch='/'),
                   Patch(facecolor="white",edgecolor='black',
                         label="LV level penetration",hatch='')]
    leg1 = ax.legend(handles=han1,ncol=1,prop={'size': 30},loc='upper left')
    ax.legend(handles=han2,ncol=1,prop={'size': 30},loc='upper center')
    ax.add_artist(leg1)
    return ax



#%% Input
sub = 147793
# sub = 113088
# sub = 121248
# sub = 121144

m_host = 3
l_host = 0.5 # fraction of residences hosting PV generators
synth_net = GetDistNet(enspath,sub,1)
N = synth_net.number_of_nodes()
total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])
prim = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
res = [n for n in synth_net if synth_net.nodes[n]['label']=='H']

# MV level penetration with number of PV hosted
prim_solar = np.random.choice(prim,m_host,replace=False)
# Random LV hosting
n_solar = int(l_host*len(res))
res_solar = np.random.choice(res,n_solar,replace=False)

#%% Construct the Dataframe

# Initialize data for pandas dataframe
data = {'count':[],'stack':[],'pen':[],'group':[]}

# Fill in the dictionary for plot data
v_range = [1.00,1.03,1.05]
v_str = [str(v_range[i])+"-"+str(v_range[i+1])+" p.u." \
              for i in range(len(v_range)-1)] + ["> "+str(v_range[-1])+" p.u."]
v_str = v_str[::-1]
pen_range = [0.3,0.5,0.8]
    
for k in range(20):
    # Get voltage for MV penetration
    for pen in pen_range:
        print("MV Penetration:",pen,"Network",k+1)
        m_rating = 1e-3*pen*total_load/m_host
        m_voltages = run_powerflow(sub,k+1,prim_solar,m_rating)
        for i in range(len(v_range)):
            num_mvolt = len([v for v in m_voltages if v>=v_range[i]])*(100.0/N)
            data['count'].append(num_mvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen)
            data['group'].append("MV level penetration")
        
    
    # Get voltage for LV penetration
    for pen in pen_range:
        print("LV Penetration:",pen,"Network",k+1)
        l_rating = 1e-3*pen*total_load/n_solar
        l_voltages = run_powerflow(sub,k+1,res_solar,l_rating)
        for i in range(len(v_range)):
            num_lvolt = len([v for v in l_voltages if v>=v_range[i]])*(100.0/N)
            data['count'].append(num_lvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen)
            data['group'].append("LV level penetration")
    


df = pd.DataFrame(data)

#%% Plot and save figure
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)
ax = draw_barplot(df,v_str,ax)
fig.savefig(figpath+str(sub)+"-out-limit.png",bbox_inches='tight')
