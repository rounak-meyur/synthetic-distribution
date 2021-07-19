# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program plots the distribution network of Virginia state.
"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"


sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,assign_linetype
from pyBuildPrimNetlib import powerflow
from pyDrawNetworklib import plot_network, color_nodes, color_edges

print("Imported modules")


# plot_network(synth_net,path=figpath+str(sub),with_secnet=True)
# color_nodes(synth_net,path=figpath+str(sub))
# color_edges(synth_net,path=figpath+str(sub))


#%% PV Hosting capacity for random residences

def run_powerflow(sub,homes,rating,savefig=None):
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
    if savefig!=None:
        plot_hosting(graph,homes,savefig)
        
    voltages = [graph.nodes[n]['voltage'] for n in graph]
    flows = [np.exp(graph.edges[e]['flow']) for e in graph.edges]
    return voltages,flows



def plot_hosting(graph,homes,path):
    fig = plt.figure(figsize=(35,30),dpi=72)
    ax = fig.add_subplot(111)

    # Draw edges
    d = {'edges':graph.edges(),
         'geometry':[graph[e[0]][e[1]]['geometry'] for e in graph.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor='black',linewidth=1.0)
    
    # Get the solar nodes
    nodelist = [n for n in graph.nodes() if n in homes]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color='red',marker='*')
    
    # Draw nodes
    d = {'nodes':graph.nodes(),
         'geometry':[Point(graph.nodes[n]['cord']) for n in graph.nodes()],
         'voltage':[graph.nodes[n]['voltage'] for n in graph.nodes()]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    
    # Color axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.001)
    df_nodes.plot(ax=ax,column='voltage',markersize=40.0,cmap=cm.plasma,
                  vmin=0.85,vmax=1.05,cax=cax,legend=True)
    cax.set_ylabel("Voltage(in pu)",fontsize=30)
    cax.tick_params(labelsize=30)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    fig.savefig("{}{}.png".format(path,'-dist-host'),bbox_inches='tight')
    return


#%% Plot histogram of voltages
from matplotlib.ticker import PercentFormatter,ScalarFormatter
sub = 121248
synth_net = GetDistNet(distpath,sub)
N = synth_net.number_of_nodes()
M = synth_net.number_of_edges()
total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])


#%% Node centrality versus random nodes

synth_net = GetDistNet(distpath,sub)
color_nodes(synth_net,path=figpath+str(sub)+'-nohosting')

# Get the individual components and find central nodes for each component
# Compare it with random nodes

# No hosting
v_nohost, f_nohost = run_powerflow(sub,[],rating=0.5)

# Central node hosting
# synth_net.remove_node(sub)
# host = 0.2
# res_solar = []
# for c in nx.connected_components(synth_net):
#     g = synth_net.subgraph(c).copy()
#     res = [n for n in g if g.nodes[n]['label']=='H']
#     n_solar = int(host*len(res))
#     n_central = nx.betweenness_centrality(g,weight='load')
#     n_central_sorted = [k for k,v in sorted(n_central.items(),
#                                             key=lambda item:item[1])[::-1]]
#     res_central = [n for n in n_central_sorted if n in res]
#     res_solar.extend(res_central[:n_solar])
    
# v_central,f_central = run_powerflow(sub,res_solar,rating=0.5)
# plot_hosting(sub,res_solar,figpath+str(sub)+"-"+str(int(host*100))+"-central",
#              "central")


# Random hosting
host = 0.5
synth_net = GetDistNet(distpath,sub)
res = [n for n in synth_net if synth_net.nodes[n]['label']=='H']
n_solar = int(host*len(res))
res_solar = np.random.choice(res,n_solar,replace=False)

pen = 0.3 
rating = 1e-3*pen*total_load/n_solar
v_random,f_random = run_powerflow(sub,res_solar,rating,
        savefig=figpath+str(sub)+'-'+str(int(100*pen))+'-penetration')





sys.exit(0)

# Histograms
bins = np.linspace(0.95,1.0,15)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.hist([v_nohost,v_random], bins, alpha=0.8, 
        label=['No PV hosting',
               str(int(host*100))+'% PV hosting at random locations',
               str(int(host*100))+'% PV hosting at central locations'])

ax.legend(loc='upper left')
ax.set_xlabel("Voltage in pu",fontsize=15)
ax.set_ylabel("Percentage of nodes",fontsize=15)
fig.gca().yaxis.set_major_formatter(PercentFormatter(N))
fig.savefig("{}{}.png".format(figpath+str(sub)+"-"+str(int(host*100)),
                              '-51121-volt-hosting-comparison'),
            bbox_inches='tight')



sys.exit(0)

#%% Plot the histograms
volt_hostvar,flow_hostvar = get_powerflow_hostvar(sub,hosting=[0,0.1,0.5,0.9])


bins = np.linspace(0.95,1.0,15)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.hist(volt_hostvar, bins, alpha=0.8, 
        label=['No PV hosting',
               '10% PV hosting rated at 50% of load',
               '50% PV hosting rated at 50% of load',
               '80% PV hosting rated at 50% of load'])

ax.legend(loc='upper left')
ax.set_xlabel("Voltage in pu",fontsize=15)
ax.set_ylabel("Percentage of nodes",fontsize=15)
fig.gca().yaxis.set_major_formatter(PercentFormatter(N))
fig.savefig("{}{}.png".format(figpath+str(sub),'-51121-volt-hosting'),
            bbox_inches='tight')


bins = np.linspace(-2,6,16)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.hist(flow_hostvar, bins, alpha=0.8, 
        label=['No PV hosting',
               '10% PV hosting rated at 50% of load',
               '50% PV hosting rated at 50% of load',
               '80% PV hosting rated at 50% of load'])

ax.legend(loc='upper right')
ax.set_xlabel("Flow in kVA",fontsize=15)
ax.set_ylabel("Percentage of edges",fontsize=15)
fig.gca().yaxis.set_major_formatter(PercentFormatter(M))
fig.savefig("{}{}.png".format(figpath+str(sub),'-51121-flow-hosting'),
            bbox_inches='tight')

#%% Plot histogram of voltages and flows for different rooftop rating

volt_ratevar,flow_ratevar = get_powerflow_ratevar(sub,rating=[0,0.1,0.3,0.5])

bins = np.linspace(0.95,1.0,15)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.hist(volt_ratevar, bins, alpha=0.8, 
        label=['No PV hosting',
               '10% PV hosting rated at 10% of load',
               '10% PV hosting rated at 30% of load',
               '10% PV hosting rated at 50% of load'])

ax.legend(loc='upper left')
ax.set_xlabel("Voltage in pu",fontsize=15)
ax.set_ylabel("Percentage of nodes",fontsize=15)
fig.gca().yaxis.set_major_formatter(PercentFormatter(N))
fig.savefig("{}{}.png".format(figpath+str(sub),'-51121-volt-rating'),
            bbox_inches='tight')



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.hist(flow_ratevar, alpha=0.8,
        label=['No PV hosting',
               '10% PV hosting rated at 10% of load',
               '10% PV hosting rated at 30% of load',
               '10% PV hosting rated at 50% of load'])

ax.legend(loc='upper right')
ax.set_xlabel("Flow in kVA",fontsize=15)
ax.set_ylabel("Percentage of edges",fontsize=15)
ax.set_xscale("log")
# ax.set_xticks([0.1, 1.0, 10, 100, 500])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
fig.gca().yaxis.set_major_formatter(PercentFormatter(M))
fig.savefig("{}{}.png".format(figpath+str(sub),'-51121-flow-rating'),
            bbox_inches='tight')

