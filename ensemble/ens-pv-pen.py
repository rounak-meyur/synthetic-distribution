# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program analyses the impact of PV penetration in the synthetic
distribution networks of Virginia.

"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
    

#%% Construct dataframe of results
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.patches import Patch

# Input
# sub = 147793
# sub = 113088
sub = 121248
# sub = 121144
nettype = {147793:'urban',121248:'rural'}

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


# Initialize data for pandas dataframe
data = {'count':[],'stack':[],'pen':[],'group':[]}

# Fill in the dictionary for plot data
v_range = [1.00,1.03,1.05]
v_str = ["%.2f"%(v_range[i])+" - "+"%.2f"%(v_range[i+1])+" p.u." \
              for i in range(len(v_range)-1)] + ["> "+"%.2f"%(v_range[-1])+" p.u."]

pen_range = [0.3,0.5,0.8]
    
for k in range(20):
    # Get voltage for MV penetration
    for pen in pen_range:
        m_rating = 1e-3*pen*total_load/m_host
        m_voltages = run_powerflow(sub,k+1,prim_solar,m_rating)
        for i in range(len(v_range)):
            num_mvolt = len([v for v in m_voltages if v>=v_range[i]])*(100/len(m_voltages))
            data['count'].append(num_mvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen*100.0)
            data['group'].append("MV level penetration")
        
    
    # Get voltage for LV penetration
    for pen in pen_range:
        l_rating = 1e-3*pen*total_load/n_solar
        l_voltages = run_powerflow(sub,k+1,res_solar,l_rating)
        for i in range(len(v_range)):
            num_lvolt = len([v for v in l_voltages if v>=v_range[i]])*(100/len(l_voltages))
            data['count'].append(num_lvolt)
            data['stack'].append(v_str[i])
            data['pen'].append(pen*100.0)
            data['group'].append("LV level penetration")
    
    print("Done with network ",str(k+1))



#%% Plot outliers
def outlier_barplot(df,groups,ax,net_type="urban"):
    
    # Draw the bar plot
    num_stack = len(groups)
    colors = sns.color_palette("rocket")[::-1][:num_stack]
    hours = df.pen.unique()
    for i,g in enumerate(df.groupby("stack")):
        ax = sns.barplot(data=g[1], x="pen",y="count",hue="group",
                              palette=[colors[i]],ax=ax,
                              zorder=i, edgecolor="k",errwidth=10,capsize=0.1)
    
    
    # Format other stuff
    ax.tick_params(axis='y',labelsize=60)
    ax.tick_params(axis='x',labelsize=60)
    ax.set_ylabel("Percentage of residences",fontsize=70)
    ax.set_xlabel("Percentage of PV penetration",fontsize=70)
    ax.set_title("PV penetration in " + str(net_type) + " network",fontsize=70)
    ax.set_ylim(bottom=0,top=30)


    hatches = itertools.cycle(['/', ''])
    for i, bar in enumerate(ax.patches):
        if i%(len(hours)) == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)


    han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                  for label, color in zip(groups, colors)]
    han2 = [Patch(facecolor="white",edgecolor='black',
                  label="MV level penetration",hatch='/'),
            Patch(facecolor="white",edgecolor='black',
                  label="LV level penetration",hatch='')]
    leg1 = ax.legend(handles=han1,ncol=1,fontsize=60,bbox_to_anchor=(0.0, 0.8),
                      loc='upper left')
    ax.legend(handles=han2,ncol=1,fontsize=60,loc='upper left')
    ax.add_artist(leg1)
    return ax

    

# Plot the Dataframe results
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(1,1,1)
df = pd.DataFrame(data)
ax = outlier_barplot(df,v_str,ax,net_type=nettype[sub])


fig.savefig("{}{}.png".format(figpath+str(sub),'-out-limit'),bbox_inches='tight')

sys.exit(0)



































#%% Voltage bar plots

def voltage_barplot(ax,ax0,data1,data2):
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

#%% Plot histogram of voltages
from matplotlib.ticker import PercentFormatter,ScalarFormatter
sub = 147793
# sub = 113088
# sub = 121248
# sub = 121144

host = 3
v_random1_mv = []; v_random2_mv = []; v_random3_mv = []
synth_net = GetDistNet(enspath,sub,1)
N = synth_net.number_of_nodes()
total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])
# MV level penetration with number of PV hosted
prim = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
prim_solar = np.random.choice(prim,host,replace=False)

for i in range(20):
    print("Loading network",i+1)
    
    # Multiple penetration percentage
    pen1 = 0.3 
    rating1 = 1e-3*pen1*total_load/host
    v_random1_mv.append(run_powerflow(sub,i+1,prim_solar,rating1))
    pen2 = 0.5 
    rating2 = 1e-3*pen2*total_load/host
    v_random2_mv.append(run_powerflow(sub,i+1,prim_solar,rating2))
    pen3 = 0.8 
    rating3 = 1e-3*pen3*total_load/host
    v_random3_mv.append(run_powerflow(sub,i+1,prim_solar,rating3))


#%% Compare LV level penetration impact
v_random1_lv = []; v_random2_lv = []; v_random3_lv = []
host = 0.5 # fraction of residences hosting PV generators
# Random hosting
synth_net = GetDistNet(enspath,sub,1)
res = [n for n in synth_net if synth_net.nodes[n]['label']=='H']
n_solar = int(host*len(res))
res_solar = np.random.choice(res,n_solar,replace=False)

for i in range(20):
    print("Loading network",i+1)
    
    # Multiple penetration percentage
    pen1 = 0.3 
    rating1 = 1e-3*pen1*total_load/n_solar
    v_random1_lv.append(run_powerflow(sub,i+1,res_solar,rating1))
    pen2 = 0.5 
    rating2 = 1e-3*pen2*total_load/n_solar
    v_random2_lv.append(run_powerflow(sub,i+1,res_solar,rating2))
    pen3 = 0.8 
    rating3 = 1e-3*pen3*total_load/n_solar
    v_random3_lv.append(run_powerflow(sub,i+1,res_solar,rating3))


#%% Plot the histogram
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

voltage_barplot(ax1,ax0,v_random1_lv,v_random1_mv)
ax1.set_xlabel("Voltage in pu",fontsize=15)
ax1.set_ylabel("Percentage of nodes",fontsize=15)
ax1.yaxis.set_major_formatter(PercentFormatter(N))
ax1.set_title("PV penetration of 30% in LV and MV networks",fontsize=15)

voltage_barplot(ax2,ax0,v_random2_lv,v_random2_mv)
ax2.set_xlabel("Voltage in pu",fontsize=15)
ax2.set_ylabel("Percentage of nodes",fontsize=15)
ax2.yaxis.set_major_formatter(PercentFormatter(N))
ax2.set_title("PV penetration of 50% in LV and MV networks",fontsize=15)

voltage_barplot(ax3,ax0,v_random3_lv,v_random3_mv)
ax3.set_xlabel("Voltage in pu",fontsize=15)
ax3.set_ylabel("Percentage of nodes",fontsize=15)
ax3.yaxis.set_major_formatter(PercentFormatter(N))
ax3.set_title("PV penetration of 80% in LV and MV networks",fontsize=15)

fig.savefig("{}{}.png".format(figpath+str(sub),
                              '-volt-penetration-ens-comp'),
            bbox_inches='tight')






