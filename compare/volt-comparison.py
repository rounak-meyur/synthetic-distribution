# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:36:57 2022

@author: Rounak

Description: Perform operational validation by comparing the node voltages and
edge flows between synthetic and actual networks. 
"""

import os,sys
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import networkx as nx
import geopandas as gpd
from pyqtree import Index
from shapely.geometry import Point, LineString
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
actpath = rootpath + "/input/actual/"
synpath = rootpath + "/primnet/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyMiscUtilslib import powerflow
from pyGeometrylib import Link
print("Imported modules")

colors = ['blue','orangered']
labels = ['Synthetic Network','Actual Network']


#%% Functions to compare network statistics

from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    s = "{0:.1f}".format(100*y)
    return s




#%% Results of statistical comparisons
sublist = [150724]
synth_net = GetDistNet(synpath,sublist)
powerflow(synth_net)
print("Synthetic network extracted")

# areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
# areas = {'patrick_henry':194,'mcbryde':9001}
areas = {'hethwood':7001}

area = 'hethwood'

# Get dataframe for buses and lines
df_lines = gpd.read_file(actpath+area+'/'+area+'_edges.shp')
df_buses = gpd.read_file(actpath+area+'/'+area+'_nodes.shp')

# Get dictionary of node geometries and edgelist
edgegeom = {}
edgelist = []

for i in range(len(df_lines)):
    edge = tuple([int(x) for x in df_lines['ID'][i].split('_')])
    edgelist.append(edge)
    edgegeom[edge] = df_lines['geometry'][i]

# Create actual graph network
act_graph = nx.Graph()
act_graph.add_edges_from(edgelist)
nx.set_edge_attributes(act_graph,edgegeom,'geometry')

nodelabel = {}
for n in act_graph.nodes:
    if n==9001:
        nodelabel[n] = 'S'
    elif 9001 in nx.neighbors(act_graph,n):
        nodelabel[n] = 'R'
    else:
        nodelabel[n] = 'T'

nx.set_node_attributes(act_graph,nodelabel,'label')

nodegeom = {df_buses['id'][i]:df_buses['geometry'][i] \
            for i in range(len(df_buses))}

busgeom = df_buses['geometry']
x_bus = [geom.x for geom in busgeom]
y_bus = [geom.y for geom in busgeom]

buffer = 1e-4
left = min(x_bus)-buffer
right = max(x_bus)+buffer
bottom = min(y_bus)-buffer
top = max(y_bus)+buffer
print("Area Data extracted and stored")



homelist = [n for n in synth_net if synth_net.nodes[n]['label']=='H' \
        and left<=synth_net.nodes[n]['cord'][0]<=right \
        and bottom<=synth_net.nodes[n]['cord'][1]<=top ]
    

#%% Assign residence to nearest transformer
bbox = (left,bottom,right,top)

# keep track of lines so we can recover them later
all_tsfr = [n for n in act_graph if act_graph.nodes[n]['label']=='T']
points = []

# initialize the quadtree index
idx = Index(bbox)
radius = 0.01

# add edge bounding boxes to the index
for i, t in enumerate(all_tsfr):
    # create tsfr geometry
    tsfr_geom = nodegeom[t]

    # bounding boxes, with padding
    x1, y1, x2, y2 = tsfr_geom.bounds
    bounds = x1-radius, y1-radius, x2+radius, y2+radius

    # add to quadtree
    idx.insert(i, bounds)

    # save the line for later use
    points.append((tsfr_geom, bounds, t))

sec_edges = []
for h in homelist:
    pt = Point(synth_net.nodes[h]['cord'])
    pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
    matches = idx.intersect(pt_bounds)
    
    # find closest path
    closest_path = min(matches, key=lambda i: points[i][0].distance(pt))
    t_map = points[closest_path][-1]
    sec_edges.append((h,t_map))


# Add secondary edges to the actual graph
act_graph.add_edges_from(sec_edges)

# Add node attributes to the actual graph
nodelabel = {}
nodecord = {}
nodeload = {}
for n in act_graph.nodes:
    if n==9001:
        nodelabel[n] = 'S'
        nodecord[n] = nodegeom[n].coords[0]
        nodeload[n] = 0.0
    elif 9001 in nx.neighbors(act_graph,n):
        nodelabel[n] = 'R'
        nodecord[n] = nodegeom[n].coords[0]
        nodeload[n] = 0.0
    elif n in homelist:
        nodelabel[n] = 'H'
        nodecord[n] = synth_net.nodes[n]['cord']
        nodeload[n] = synth_net.nodes[n]['load']
    else:
        nodelabel[n] = 'T'
        nodecord[n] = nodegeom[n].coords[0]
        nodeload[n] = 0.0

nx.set_node_attributes(act_graph,nodelabel,'label')
nx.set_node_attributes(act_graph,nodecord,'cord')
nx.set_node_attributes(act_graph,nodeload,'load')        

# Add all edge attributes to the graph
edgelabel = {}
edgelength = {}
edge_r = {}
edge_x = {}
for e in act_graph.edges:
    if (e in sec_edges) or ((e[1],e[0]) in sec_edges):
        edgelabel[e] = 'S'
        geom = LineString((Point(nodecord[e[0]]),Point(nodecord[e[1]])))
        act_graph.edges[e]['geometry'] = geom
        edgelength[e] = Link(geom).geod_length
        edge_r[e] = 0.082/57.6 * edgelength[e] * 1e-3
        edge_x[e] = 0.027/57.6 * edgelength[e] * 1e-3
    elif (e[0]==9001) or (e[1]==9001):
        edgelabel[e] = 'E'
        if e in edgegeom:
            edgelength[e] = Link(edgegeom[e]).geod_length
        else:
            edgelength[e] = Link(edgegeom[(e[1],e[0])]).geod_length
        edge_r[e] = (0.0822/363000) * edgelength[e] * 1e-3
        edge_x[e] = (0.0964/363000) * edgelength[e] * 1e-3
    else:
        edgelabel[e] = 'P'
        if e in edgegeom:
            edgelength[e] = Link(edgegeom[e]).geod_length
        else:
            edgelength[e] = Link(edgegeom[(e[1],e[0])]).geod_length
        edge_r[e] = (0.0822/39690) * edgelength[e] * 1e-3
        edge_x[e] = (0.0964/39690) * edgelength[e] * 1e-3

nx.set_edge_attributes(act_graph,edgelabel,'label')
nx.set_edge_attributes(act_graph,edgelength,'length')
nx.set_edge_attributes(act_graph,edge_r,'r')
nx.set_edge_attributes(act_graph,edge_x,'x')


powerflow(act_graph)


#%% Compare voltages

act_volt = [act_graph.nodes[h]['voltage']*120.0 for h in homelist]
syn_volt = [synth_net.nodes[h]['voltage']*120.0 for h in homelist]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.scatter(act_volt,syn_volt,marker='+',c='red',s=100.0)
ax.plot([119,120],[119,120],'k--',linewidth=2.0)
ax.plot([119.5,120.5],[119,120],'g--',linewidth=2.0)
ax.plot([119,120],[119.5,120.5],'g--',linewidth=2.0)
ax.set_xlim(119.0,120.0)
ax.set_ylim(119.0,120.0)

ax.set_xlabel("Voltage in actual network (Volts)",fontsize=30)
ax.set_ylabel("Voltage in synthetic network (Volts)",fontsize=30)
ax.tick_params(axis='both', labelsize=20)

filename = "voltage-comparison"
fig.savefig("{}{}.png".format(figpath,filename),bbox_inches='tight')


#%% Compare flows
node_interest = [n for n in synth_net \
                 if left<=synth_net.nodes[n]['cord'][0]<=right \
                     and bottom<=synth_net.nodes[n]['cord'][1]<=top]
syn_edges = synth_net.edges(node_interest)

act_flows = [np.exp(act_graph.edges[e]['flow']) for e in act_graph.edges]
syn_flows = [np.exp(synth_net.edges[e]['flow']) for e in syn_edges]

w_flow_act = np.ones_like(act_flows)/float(len(act_flows))
w_flow_syn = np.ones_like(syn_flows)/float(len(syn_flows))

flow = [act_flows,syn_flows]
w_flow = [w_flow_act, w_flow_syn]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.hist(flow,bins=np.logspace(np.log(1),np.log(10.0), 10),
        weights=w_flow,label=labels,color=colors)
ax.set_xscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
ax.set_ylabel("Percentage of lines",fontsize=45)
ax.set_xlabel("Flow in kVA",fontsize=45)
ax.legend(fontsize=30,markerscale=2)
ax.set_xticks([1,2.5,5,10,25,50,100], 
              labels=['1','2.5','5','10','25','50','100'])
ax.tick_params(axis='both', labelsize=30)


n_act = len(act_flows)
n_syn = len(syn_flows)
flow_min = np.logspace(np.log(1),np.log(10.0), 10)[:-1]
flow_max = np.logspace(np.log(1),np.log(10.0), 10)[1:]
p_flow_act = [(len([f for f in act_flows if flow_min[i]<=f<=flow_max[i]]))/n_act \
      for i in range(len(flow_max))]
p_flow_syn = [(len([f for f in syn_flows if flow_min[i]<=f<=flow_max[i]]))/n_syn \
      for i in range(len(flow_max))]
E_flow = entropy(p_flow_act,p_flow_syn)

filename = "flow-comparison"
fig.savefig("{}{}.png".format(figpath,filename),bbox_inches='tight')

    














