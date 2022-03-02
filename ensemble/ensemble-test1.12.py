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
import geopandas as gpd
from shapely.geometry import Point

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

def generate_bars(ax,ax0,data1,data2):
    width = 0.003
    bins = np.linspace(0.95,1.05,15)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y1,_,_ = ax0.hist(data1,bins=bins)
    y2,_,_ = ax0.hist(data2,bins=bins)
    
    # Data 1
    y1_mean = np.mean(y1,axis=0)
    y1_var = np.var(y1,axis=0)
    y1_err = np.sqrt(y1_var)
    x1 = bincenters - width/2
    
    # Data 2
    y2_mean = np.mean(y2,axis=0)
    y2_var = np.var(y2,axis=0)
    y2_err = np.sqrt(y2_var)
    x2 = bincenters + width/2
    
    # Add bar plots to the axis object
    ax.bar(x1, y1_mean, width=width, color='royalblue', yerr=y1_err, capsize=5.0, 
           error_kw = {'mew':1.5, 'elinewidth':2}, label='LV level penetration')
    ax.bar(x2, y2_mean, width=width, color='crimson', yerr=y2_err, capsize=5.0, 
           error_kw = {'mew':1.5, 'elinewidth':2}, label='MV level penetration')
    ax.legend(loc='best',prop={'size': 14})
    return

def draw_barplot(df,groups,ax=None,adopt=90,rate=4800):
    if ax == None:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
    
    # Draw the bar plot
    num_stack = len(groups)
    colors = sns.color_palette("Set3")[:num_stack]
    hours = df.hour.unique()
    for i,g in enumerate(df.groupby("stack",sort=False)):
        ax = sns.barplot(data=g[1], x="hour",y="count",hue="group",
                              palette=[colors[i]],ax=ax,
                              zorder=-i, edgecolor="k",errwidth=5)
    
    
    # Format other stuff
    ax.tick_params(axis='y',labelsize=40)
    ax.tick_params(axis='x',labelsize=40,rotation=90)
    ax.set_ylabel("Number of residences",fontsize=50)
    ax.set_xlabel("Hours",fontsize=50)
    ax.set_title("Adoption percentage: "+str(adopt)+"%",fontsize=50)
    ax.set_ylim(bottom=0,top=80)


    hatches = itertools.cycle(['/', ''])
    for i, bar in enumerate(ax.patches):
        if i%(len(hours)) == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)


    han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                  for label, color in zip(groups, colors)]
    han2 = [Patch(facecolor="white",edgecolor='black',
                  label="Distributed optimization",hatch='/'),
                   Patch(facecolor="white",edgecolor='black',
                         label="Individual optimization",hatch='')]
    leg1 = ax.legend(handles=han1,ncol=1,prop={'size': 30},loc='upper right')
    ax.legend(handles=han2,ncol=1,prop={'size': 30},loc='upper left')
    ax.add_artist(leg1)
    return ax


#%% Plot the histogram
# fig0 = plt.figure()
# ax0 = fig0.add_subplot(111)

# fig = plt.figure(figsize=(24,6))
# ax1 = fig.add_subplot(131)
# ax2 = fig.add_subplot(132)
# ax3 = fig.add_subplot(133)

# generate_bars(ax1,ax0,v_random1_lv,v_random1_mv)
# ax1.set_xlabel("Voltage in pu",fontsize=15)
# ax1.set_ylabel("Percentage of nodes",fontsize=15)
# ax1.yaxis.set_major_formatter(PercentFormatter(N))
# ax1.set_title("PV penetration of 30% in LV and MV networks",fontsize=15)

# generate_bars(ax2,ax0,v_random2_lv,v_random2_mv)
# ax2.set_xlabel("Voltage in pu",fontsize=15)
# ax2.set_ylabel("Percentage of nodes",fontsize=15)
# ax2.yaxis.set_major_formatter(PercentFormatter(N))
# ax2.set_title("PV penetration of 50% in LV and MV networks",fontsize=15)

# generate_bars(ax3,ax0,v_random3_lv,v_random3_mv)
# ax3.set_xlabel("Voltage in pu",fontsize=15)
# ax3.set_ylabel("Percentage of nodes",fontsize=15)
# ax3.yaxis.set_major_formatter(PercentFormatter(N))
# ax3.set_title("PV penetration of 80% in LV and MV networks",fontsize=15)

# fig.savefig("{}{}.png".format(figpath+str(sub),
#                               '-volt-penetration-ens-comp'),
#             bbox_inches='tight')



#%% Plot outliers
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.patches import Patch

# Input
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

pen_range = [0.3,0.5,0.8]
    
for k in range(20):
    # Get voltage for MV penetration
    for pen in pen_range:
        m_rating = 1e-3*pen*total_load/m_host
        m_voltages = run_powerflow(sub,k+1,prim_solar,m_rating)
        for i in range(len(v_range)):
            num_mvolt = len([v for v in m_voltages if v>=v_range[i]])
            data['count'].append(num_mvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen)
            data['group'].append("MV level penetration")
        
    
    # Get voltage for LV penetration
    for pen in pen_range:
        l_rating = 1e-3*pen*total_load/n_solar
        l_voltages = run_powerflow(sub,k+1,res_solar,l_rating)
        for i in range(len(v_range)):
            num_mvolt = len([v for v in m_voltages if v>=v_range[i]])
            data['count'].append(num_mvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen)
            data['group'].append("MV level penetration")
    


df = pd.DataFrame(data)
ax = draw_barplot(df,v_str,ax,adopt=adopt,rate=rate)
