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
distpath = workpath + "/out/osm-primnet/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyBuildPrimNetlib import powerflow,assign_linetype
print("Imported modules")



#%% PV Hosting capacity for random residences

def run_powerflow(sub,homes,rating):
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
    graph = GetDistNet(distpath,sub)
    assign_linetype(graph)
    for h in homes:
        graph.nodes[h]['load'] = graph.nodes[h]['load'] - (rating*1e3)
    powerflow(graph)
    voltages = [graph.nodes[n]['voltage'] for n in graph]
    return voltages



# color_nodes(synth_net,path=figpath+str(sub)+'-'+str(100*hosting))
# color_edges(synth_net,path=figpath+str(sub)+'-'+str(100*hosting))


def plot_hosting(sub,solar,path,name):
    synth_graph = GetDistNet(distpath,sub)
    fig = plt.figure(figsize=(30,30), dpi=72)
    ax = fig.add_subplot(111)
    
    # Get the nodes
    nodelist = [n for n in synth_graph.nodes() if n not in solar]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color='black',markersize=20)
    
    # Get the edges
    d = {'edges':synth_graph.edges(),
         'geometry':[synth_graph[e[0]][e[1]]['geometry'] \
                     for e in synth_graph.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor='black',linewidth=2.0)
    
    # Get the solar nodes
    nodelist = [n for n in synth_graph.nodes() if n in solar]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color='red',markersize=60,marker = '*')
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_title("PV hosting at "+name+" residences",fontsize=30)
    fig.savefig("{}{}.png".format(path,'-51121-dist-host'),bbox_inches='tight')
    return


#%% Get some substation IDs
with open(inppath+"rural-sublist.txt") as f:
    rural = f.readlines()
rural = [int(r) for r in rural[0].strip('\n').split(' ')]
f_done = [int(f.strip('-prim-dist.gpickle')) for f in os.listdir(distpath)]
rural_done = [r for r in f_done if r in rural]

#%% Plot histogram of voltages
from matplotlib.ticker import PercentFormatter,ScalarFormatter
sub = rural_done[10]
# sub = 121144
# sub = 147793
synth_net = GetDistNet(distpath,sub)
N = synth_net.number_of_nodes()
total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])


#%% Plot histogram of voltages

sub = 147793
# sub = 113088
# sub = 121248
# sub = 121144


synth_net = GetDistNet(distpath,sub)
N = synth_net.number_of_nodes()
total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])

# MV level penetration with number of PV hosted
host = 3
prim = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
prim_solar = np.random.choice(prim,host,replace=False)



# Multiple penetration percentage
pen1 = 0.3 
rating1 = 1e-3*pen1*total_load/host
v_random1_mv = run_powerflow(sub,prim_solar,rating1)
pen2 = 0.5 
rating2 = 1e-3*pen2*total_load/host
v_random2_mv = run_powerflow(sub,prim_solar,rating2)
pen3 = 0.8 
rating3 = 1e-3*pen3*total_load/host
v_random3_mv = run_powerflow(sub,prim_solar,rating3)


#%% Compare LV level penetration impact


# Random hosting
host = 0.5 # fraction of residences hosting PV generators
synth_net = GetDistNet(distpath,sub)
res = [n for n in synth_net if synth_net.nodes[n]['label']=='H']
n_solar = int(host*len(res))
res_solar = np.random.choice(res,n_solar,replace=False)


    
# Multiple penetration percentage
pen1 = 0.3 
rating1 = 1e-3*pen1*total_load/n_solar
v_random1_lv = run_powerflow(sub,res_solar,rating1)
pen2 = 0.5 
rating2 = 1e-3*pen2*total_load/n_solar
v_random2_lv = run_powerflow(sub,res_solar,rating2)
pen3 = 0.8 
rating3 = 1e-3*pen3*total_load/n_solar
v_random3_lv = run_powerflow(sub,res_solar,rating3)
    

#%%

def generate_bars(ax,data1,data2):
    width = 0.003
    bins = np.linspace(0.95,1.05,15)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    
    # Data
    y1,_ = np.histogram(data1,bins=bins)
    x1 = bincenters - width/2
    y2,_ = np.histogram(data2,bins=bins)
    x2 = bincenters + width/2
    
    # Add bar plots to the axis object
    ax.bar(x1, y1, width=width, color='royalblue', 
           label='LV level penetration')
    ax.bar(x2, y2, width=width, color='crimson', 
           label='MV level penetration')
    ax.legend(loc='best',prop={'size': 14})
    return




#%% Plot the histogram
fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

generate_bars(ax1,v_random1_lv,v_random1_mv)
ax1.set_xlabel("Voltage in pu",fontsize=15)
ax1.set_ylabel("Percentage of nodes",fontsize=15)
ax1.yaxis.set_major_formatter(PercentFormatter(N))
ax1.set_title("PV penetration of 30% in LV and MV networks",fontsize=15)

generate_bars(ax2,v_random2_lv,v_random2_mv)
ax2.set_xlabel("Voltage in pu",fontsize=15)
ax2.set_ylabel("Percentage of nodes",fontsize=15)
ax2.yaxis.set_major_formatter(PercentFormatter(N))
ax2.set_title("PV penetration of 50% in LV and MV networks",fontsize=15)

generate_bars(ax3,v_random3_lv,v_random3_mv)
ax3.set_xlabel("Voltage in pu",fontsize=15)
ax3.set_ylabel("Percentage of nodes",fontsize=15)
ax3.yaxis.set_major_formatter(PercentFormatter(N))
ax3.set_title("PV penetration of 80% in LV and MV networks",fontsize=15)

fig.savefig("{}{}.png".format(figpath+str(sub),'-volt-penetration-comp'),
            bbox_inches='tight')
