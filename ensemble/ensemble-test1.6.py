# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:43:38 2021

@author: rounak
Description: Compares between networks in an ensemble.
"""

import sys,os
import networkx as nx
from pyqtree import Index
import numpy as np
from shapely.geometry import Point,LineString
from geographiclib.geodesic import Geodesic
from math import log
import matplotlib.pyplot as plt
import seaborn as sns


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/osm-primnet/"
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
# sub = 147793
synth_net = GetDistNet(distpath,sub)

org_length = sum([synth_net[e[0]][e[1]]['geo_length'] for e in synth_net.edges])/1000
org_flows = {e:synth_net[e[0]][e[1]]['flow'] for e in synth_net.edges}
org_voltage = {n:synth_net.nodes[n]['voltage'] for n in synth_net.nodes}

plot_network(synth_net,with_secnet=True,path=figpath+str(sub)+'-org-net-')
color_nodes(synth_net,path=figpath+str(sub)+'-orgvolt-')
color_edges(synth_net,path=figpath+str(sub)+'-orgflow-')
sys.exit(0)

#%% Hop distribution
num_nets = 4
net_length = []
voltage = []
flows = []
Hops = []
Dist = []
for i in range(1,num_nets+1):
    graph = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i)+'.gpickle')
    net_length.append(sum([graph[e[0]][e[1]]['geo_length'] \
                           for e in graph.edges])/1000.0)
    voltage.append([graph.nodes[n]['voltage'] for n in graph.nodes \
                    if graph.nodes[n]['label']!='H'])
    flows.append([graph[e[0]][e[1]]['flow'] for e in graph.edges])
    Hops.append(np.array([nx.shortest_path_length(graph,n,sub) \
                                  for n in list(graph.nodes())]))
    Dist.append(np.array([nx.shortest_path_length(graph,n,sub,weight='geo_length') \
                                  for n in list(graph.nodes())])*1e-3)


#%% Plot the comparisons
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.scatter(range(1,num_nets+1),net_length,c='crimson',marker='^')
ax.hlines(org_length,xmin=1,xmax=num_nets,linestyle='dashed',color='blue')
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

ax.bar(range(1,num_nets+1),volt_count['valid'],
       label='Feasible voltages',color='limegreen')
ax.bar(range(1,num_nets+1),volt_count['invalid'],
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

ax.bar(range(1,num_nets+1),flow_count['valid'],
       label='Feasible flows',color='limegreen')
ax.bar(range(1,num_nets+1),flow_count['invalid'],
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
for i in range(num_nets):
    sns.kdeplot(Hops[i],shade=False,color=col[i%3])
ax.set_ylabel('Percentage of nodes',fontsize=20)
ax.set_xlabel('Hops from root node',fontsize=20)
ax.set_title("Hop distribution",fontsize=20)
labels = ax.get_yticks()
ax.set_yticklabels(["{:.1f}".format(100.0*i) for i in labels])

#%% Reach distribution
col = ['r','g','b']
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
for i in range(num_nets):
    sns.kdeplot(Dist[i],shade=False,color=col[i%3])
ax.set_ylabel('Percentage of nodes',fontsize=20)
ax.set_xlabel('Distance (in km) from root node',fontsize=20)
ax.set_title("Reach distribution",fontsize=20)
labels = ax.get_yticks()
ax.set_yticklabels(["{:.1f}".format(100.0*i) for i in labels])


#%% Plot new network
i = 1
suffix = str(sub)+'-'+str(i)
tree = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i)+'.gpickle')
plot_network(tree,with_secnet=True,path=figpath+'mc-net-'+suffix)
color_nodes(tree,path=figpath+'mc-volt-'+suffix)
color_edges(tree,path=figpath+'mc-flow-'+suffix)

#%%
sns.kdeplot(Hops[0],shade=False,color='red')
sns.kdeplot(Hops[1],shade=False,color='blue')
sns.kdeplot(Hops[2],shade=False,color='green')
sns.kdeplot(Hops[3],shade=False,color='magenta')
sns.kdeplot(Hops[4],shade=False,color='black')

#%% Plot network
from matplotlib.collections import PolyCollection
from pyGeometrylib import partitions
import geopandas as gpd

def DrawNodes(graph,ax,color='orangered',size=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    d = {'nodes':graph.nodes(),
         'geometry':[Point(graph.nodes[n]['cord']) for n in graph.nodes()]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size)
    return

def DrawEdges(graph,ax,color='orangered',width=1.0):
    """
    """
    d = {'edges':graph.edges(),
         'geometry':[graph[e[0]][e[1]]['geometry'] for e in graph.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width)
    return

def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge,boundary.north_edge]
    vert2 = [boundary.east_edge,boundary.north_edge]
    vert3 = [boundary.east_edge,boundary.south_edge]
    vert4 = [boundary.west_edge,boundary.south_edge]
    return np.array([vert1,vert2,vert3,vert4])

cords = np.array([list(synth_net.nodes[n]['cord']) for n in synth_net])
LEFT,BOTTOM = np.min(cords,0)
RIGHT,TOP = np.max(cords,0)
gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),5,5)


DPI = 72    
fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
ax = plt.subplot()
ax.set_xlim(LEFT,RIGHT)
ax.set_ylim(BOTTOM,TOP)

# Get the boxes for the valid comparisons
verts_valid = [get_polygon(bound) for i,bound in enumerate(gridlist)]
c = PolyCollection(verts_valid,edgecolor='black',facecolor='white')
ax.add_collection(c)

i = 4
suffix = str(sub)+'-'+str(i)
tree = nx.read_gpickle(outpath+str(sub)+'-ensemble-'+str(i)+'.gpickle')
DrawNodes(tree,ax,color='black')
DrawEdges(tree,ax,color='black')

ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
fig.savefig("{}{}.png".format(figpath,str(sub)+'-ens-'+str(i)),bbox_inches='tight')