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
from pyBuildPrimNetlib import powerflow,assign_linetype

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




#%% Plot the histogram
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

generate_bars(ax1,ax0,v_random1_lv,v_random1_mv)
ax1.set_xlabel("Voltage in pu",fontsize=15)
ax1.set_ylabel("Percentage of nodes",fontsize=15)
ax1.yaxis.set_major_formatter(PercentFormatter(N))
ax1.set_title("PV penetration of 30% in LV and MV networks",fontsize=15)

generate_bars(ax2,ax0,v_random2_lv,v_random2_mv)
ax2.set_xlabel("Voltage in pu",fontsize=15)
ax2.set_ylabel("Percentage of nodes",fontsize=15)
ax2.yaxis.set_major_formatter(PercentFormatter(N))
ax2.set_title("PV penetration of 50% in LV and MV networks",fontsize=15)

generate_bars(ax3,ax0,v_random3_lv,v_random3_mv)
ax3.set_xlabel("Voltage in pu",fontsize=15)
ax3.set_ylabel("Percentage of nodes",fontsize=15)
ax3.yaxis.set_major_formatter(PercentFormatter(N))
ax3.set_title("PV penetration of 80% in LV and MV networks",fontsize=15)

fig.savefig("{}{}.png".format(figpath+str(sub),
                              '-volt-penetration-ens-comp'),
            bbox_inches='tight')





