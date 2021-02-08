# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:10:18 2021

@author: rouna
"""

import sys,os
import networkx as nx
import matplotlib.pyplot as plt
from pyqtree import Index
import numpy as np
from shapely.geometry import Point,LineString
from geographiclib.geodesic import Geodesic
from math import log
import seaborn as sns


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/prim-network/"
outpath = workpath + "/out/"
sys.path.append(libpath)


from pyDrawNetworklib import plot_network, color_nodes, color_edges

def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-prim-dist.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-prim-dist.gpickle')
    return graph

# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]
sub = 121144
synth_net = GetDistNet(distpath,sub)
org_length = sum([synth_net[e[0]][e[1]]['geo_length'] for e in synth_net.edges])/1000
org_flows = {e:synth_net[e[0]][e[1]]['flow'] for e in synth_net.edges}
org_voltage = {n:synth_net.nodes[n]['voltage'] for n in synth_net.nodes}
# plot_network(synth_net,with_secnet=True)



#%% Get ensemble of networks
net_length = []
voltage = []
flows = []
Hops = []
for i in range(100):
    graph = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i)+'.gpickle')
    net_length.append(sum([graph[e[0]][e[1]]['geo_length'] \
                           for e in graph.edges])/1000.0)
    voltage.append([graph.nodes[n]['voltage'] for n in graph.nodes \
                    if graph.nodes[n]['label']!='H'])
    flows.append([graph[e[0]][e[1]]['flow'] for e in graph.edges])
    Hops.append(np.array([nx.shortest_path_length(graph,n,sub) \
                                  for n in list(graph.nodes())]))

#%% Plot the comparisons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.scatter(range(1,101),net_length,c='crimson',marker='^')
ax.hlines(org_length,xmin=1,xmax=100,linestyle='dashed',color='blue')
ax.tick_params(bottom=False,labelbottom=False)
ax.set_ylabel('Length of network (in km)',fontsize=14)
ax.set_title("Comparison of network length",fontsize=14)

#%% Voltages
vmin = 0.9
vmax = 1.0
volt_count = {'valid':[],'invalid':[]}
for volt in voltage:
    valid = 100.0*sum([vmin<=v<=vmax for v in volt])/len(volt)
    invalid = 100.0 - valid
    volt_count['valid'].append(valid)
    volt_count['invalid'].append(invalid)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

ax.bar(range(1,101),volt_count['valid'],
       label='Feasible voltages',color='limegreen')
ax.bar(range(1,101),volt_count['invalid'],
       label='Infeasible voltages',color='orangered',
       bottom=volt_count['valid'])

ax.tick_params(bottom=False,labelbottom=False)
ax.set_ylabel('Percentage of nodes',fontsize=14)
ax.legend()
ax.set_title("Comparison of node voltages",fontsize=14)

#%% Flows
fmax = log(1000.0)
flow_count = {'valid':[],'invalid':[]}
for flow in flows:
    valid = 100.0*sum([f<=fmax for f in flow])/len(flow)
    invalid = 100.0 - valid
    flow_count['valid'].append(valid)
    flow_count['invalid'].append(invalid)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

ax.bar(range(1,101),flow_count['valid'],
       label='Feasible flows',color='limegreen')
ax.bar(range(1,101),flow_count['invalid'],
       label='Infeasible flows',color='orangered',
       bottom=flow_count['valid'])

ax.tick_params(bottom=False,labelbottom=False)
ax.set_ylabel('Percentage of edges',fontsize=14)
ax.legend()
ax.set_title("Comparison of edge flows",fontsize=14)

#%% Hop distribution
col = ['r','g','b']
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
for i in range(100):
    sns.kdeplot(Hops[i],shade=False,color=col[i%3])
ax.set_ylabel('Density',fontsize=20)
ax.set_xlabel('Hops from root node',fontsize=20)
ax.set_title("Hop distribution",fontsize=20)




























