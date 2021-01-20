# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations
"""

import sys
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString, Point
import networkx as nx
import numpy as np
from math import log
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import namedtuple as nt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D

#%% Geometry of Primary Network
def MeasureDistance(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates. The format of each point 
    is (longitude,latitude).
    pt1: (longitude,latitude) of point 1
    pt2: (longitude,latitude) of point 2
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

class Link(LineString):
    """
    Derived class from Shapely LineString to compute metric distance based on 
    geographical coordinates over geometric coordinates.
    """
    def __init__(self,line_geom):
        """
        """
        super().__init__(line_geom)
        self.geod_length = self.__length()
        return
    
    
    def __length(self):
        '''
        Computes the geographical length in meters between the ends of the link.
        '''
        if self.geom_type != 'LineString':
            print("Cannot compute length!!!")
            return None
        # Compute great circle distance
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length
    
#%% Primary Network Data
def GetSynthNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            load: load for each node for consumers, otherwise it is 0.0
            label: 'H' for home, 'T' for transformer, 'R' for road node, 'S' for subs
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
    """
    graph = nx.Graph()
    for c in code:
        g = nx.read_gpickle(path+str(c)+'-prim-dist.gpickle')
        graph = nx.compose(graph,g)
    return graph

#%% Combine primary and secondary networks
def GetSecnet(path,areas):
    # Extract the secondary network data from all areas
    nodelabel = {}
    nodepos = {}
    edgelist = []
    edger = {}
    edgex = {}
    edgelabel = {}
    edgegeom = {}
    glength = {}
    for area in areas:
        with open(path+str(area)+'-sec-dist.txt','r') as f:
            lines = f.readlines()
        for line in lines:
            data = line.strip('\n').split('\t')
            n1 = int(data[0]); n2 = int(data[4])
            edgelist.append((n1,n2))
            # nodedata
            nodepos[n1]=(float(data[2]),float(data[3]))
            nodepos[n2]=(float(data[6]),float(data[7]))
            nodelabel[n1] = data[1] 
            nodelabel[n2] = data[5]
            # edgedata
            length = 1e-3 * MeasureDistance(nodepos[n1],nodepos[n2])
            length = length if length != 0.0 else 1e-12
            edger[(n1,n2)] = 0.81508/57.6 * length
            edgex[(n1,n2)] = 0.3496/57.6 * length
            edgegeom[(n1,n2)] = Link((nodepos[n1],nodepos[n2]))
            edgelabel[(n1,n2)] = 'S'
            glength[(n1,n2)] = edgegeom[(n1,n2)].geod_length
    # Construct the graph
    secnet = nx.Graph()
    secnet.add_edges_from(edgelist)
    nx.set_node_attributes(secnet,nodepos,'cord')
    nx.set_node_attributes(secnet,nodelabel,'label')
    nx.set_edge_attributes(secnet, edgegeom, 'geometry')
    nx.set_edge_attributes(secnet, edgelabel, 'label')
    nx.set_edge_attributes(secnet, edger, 'r')
    nx.set_edge_attributes(secnet, edgex, 'x')
    nx.set_edge_attributes(secnet, glength, 'geo_length')
    return secnet

def combine_networks(primary,secondary):
    roots = [n for n in list(primary.nodes()) if primary.nodes[n]['label']=='T']
    secnodes = [n for t in roots for n in list(nx.descendants(secondary,t))]   
    secnet = secondary.subgraph(secnodes+roots)
    net = nx.compose(primary,secnet)
    return net

#%% Power flow visualizations
def GetHomes(path,fislist):
    '''
    Returns the dictionary of home data
    '''
    dict_load = {}
    dict_avg = {}
    dict_cord = {}
    for fis in fislist:
        df_home = pd.read_csv(path+'load/'+fis+'-home-load.csv')
        df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
        df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
        
        dict_load.update(df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list'))
        dict_cord.update(df_home.iloc[:,0:3].set_index('hid').T.to_dict('list'))
        dict_avg.update(dict(zip(df_home.hid,df_home.average)))
    home = nt("home",field_names=["cord","profile","average"])
    homes = home(cord=dict_cord,profile=dict_load,average=dict_avg)
    return homes

def check_voltage(dist_net,homes):
    """
    Checks power flow solution and plots the voltage at different nodes in the 
    network through colorbars.
    """
    A = nx.incidence_matrix(dist_net,nodelist=list(dist_net.nodes()),
                            edgelist=list(dist_net.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(dist_net.nodes()) \
                if dist_net.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(dist_net.nodes()) \
                if dist_net.nodes[node]['label'] != 'S']
    
    # Resistance data
    edge_r = nx.get_edge_attributes(dist_net,'r')
    R = np.diag([1.0/edge_r[e] if e in edge_r else 1.0/edge_r[(e[1],e[0])] \
         for e in list(dist_net.edges())])
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*homes.average[h] \
                  if dist_net.nodes[h]['label']=='H' else 0.0 \
                  for h in nodelist])
    
    # Voltages
    v = np.matmul(np.linalg.inv(G),p)
    voltage = {h:1.0-v[i] for i,h in enumerate(nodelist)}
    subnodes = [node for node in list(dist_net.nodes()) \
                if dist_net.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = 1.0
    return voltage

def check_flows(dist_net,homes):
    """
    Checks power flow solution and plots the flows at different edges in the 
    network through colorbars.
    """
    A = nx.incidence_matrix(dist_net,nodelist=list(dist_net.nodes()),
                            edgelist=list(dist_net.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(dist_net.nodes()) \
                if dist_net.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(dist_net.nodes()) \
                if dist_net.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(dist_net.edges())]
    
    # Resistance data
    p = np.array([1e-3*homes.average[h] \
                  if dist_net.nodes[h]['label']=='H' else 0.0 \
                  for h in nodelist])
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    
    flows = {e:log(abs(f[i])) for i,e in enumerate(edgelist)}
    return flows
#%% Network Geometries
def DrawNodes(synth_graph,ax,label='T',color='green',size=25):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    nodelist = [n for n in synth_graph.nodes() \
                if synth_graph.nodes[n]['label']==label \
                    or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size)
    return

def DrawEdges(synth_graph,ax,label='P',color='black',width=2.0):
    """
    """
    # Get the nodes for the specified label
    edgelist = [e for e in synth_graph.edges() \
                if synth_graph[e[0]][e[1]]['label']==label\
                    or synth_graph[e[0]][e[1]]['label'] in label]
    d = {'edges':edgelist,
         'geometry':[synth_graph[e[0]][e[1]]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width)
    return


def plot_network(net,inset,path,with_secnet=False):
    """
    """
    fig = plt.figure(figsize=(30,30), dpi=72)
    ax = fig.add_subplot(111)
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax,inset[sub]['zoom'],loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes
        DrawNodes(inset[sub]['graph'],axins,label='S',color='dodgerblue',
                  size=2000)
        DrawNodes(inset[sub]['graph'],axins,label='T',color='green',size=25)
        DrawNodes(inset[sub]['graph'],axins,label='R',color='black',size=2.0)
        if with_secnet: DrawNodes(inset[sub]['graph'],axins,label='H',
                                  color='crimson',size=2.0)
        # Draw edges
        DrawEdges(inset[sub]['graph'],axins,label='P',color='black',width=2.0)
        DrawEdges(inset[sub]['graph'],axins,label='E',color='dodgerblue',width=2.0)
        if with_secnet: DrawEdges(inset[sub]['graph'],axins,label='S',
                                  color='crimson',width=1.0)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], 
                   loc2=inset[sub]['loc2'], fc="none", ec="0.5")
    
    # Legend for the plot
    leghands = [Line2D([0], [0], color='black', markerfacecolor='black', 
                   marker='o',markersize=0,label='primary network'),
            Line2D([0], [0], color='crimson', markerfacecolor='crimson', 
                   marker='o',markersize=0,label='secondary network'),
            Line2D([0], [0], color='dodgerblue', 
                   markerfacecolor='dodgerblue', marker='o',
                   markersize=0,label='high voltage feeder'),
            Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=20,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='red', 
                   marker='o',markersize=20,label='residence'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=20,label='substation')]
    ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
    fig.savefig("{}{}.png".format(path,'51121-dist'),bbox_inches='tight')
    return


def color_nodes(net,voltage,inset,path):
    fig = plt.figure(figsize=(30,30),dpi=72)
    ax = fig.add_subplot(111)
    
    # Draw edges
    DrawEdges(net,ax,label=['P','E','S'],color='black',width=1.0)
    
    # Draw nodes
    d = {'nodes':net.nodes(),
         'geometry':[Point(net.nodes[n]['cord']) for n in net.nodes()],
         'voltage':[voltage[e] for e in net.nodes()]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,column='voltage',markersize=2.0,cmap=cm.plasma)
    
    # Colorbar
    cobj = cm.ScalarMappable(cmap='plasma')
    cobj.set_clim(vmin=0.80,vmax=1.05)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label('Voltage(pu)',size=30)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax, inset[sub]['zoom'], 
                                  loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes and edges
        DrawEdges(inset[sub]['graph'],axins,label=['P','E','S'],
                  color='black',width=1.0)
        d = {'nodes':inset[sub]['graph'].nodes(),
             'geometry':[Point(inset[sub]['graph'].nodes[n]['cord']) \
                         for n in inset[sub]['graph'].nodes()],
             'voltage':[voltage[n] for n in inset[sub]['graph'].nodes()]}
        df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_nodes.plot(ax=axins,column='voltage',markersize=2.0,cmap=cm.plasma)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], loc2=inset[sub]['loc2'], 
               fc="none", ec="0.5")
    fig.savefig("{}{}.png".format(path,'51121-dist-voltage'),bbox_inches='tight')
    return


def color_edges(net,flows,inset,path):
    fig = plt.figure(figsize=(30,30),dpi=72)
    ax = fig.add_subplot(111)
    
    # Draw nodes
    DrawNodes(net,ax,label=['S','T','R','H'],color='black',size=2.0)
    
    # Draw edges
    d = {'edges':net.edges(),
         'geometry':[net[e[0]][e[1]]['geometry'] for e in net.edges()],
         'flows':[flows[e] for e in net.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(column='flows',ax=ax,cmap=cm.plasma)
    
    # Colorbar
    fmin = 0.2; fmax = 800.0
    cobj = cm.ScalarMappable(cmap='plasma')
    cobj.set_clim(vmin=fmin,vmax=fmax)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label('Flow along edge in kVA',size=30)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax, inset[sub]['zoom'], 
                                  loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes and edges
        DrawNodes(inset[sub]['graph'],axins,label=['S','T','R','H'],
                  color='black',size=2.0)
        d = {'edges':inset[sub]['graph'].edges(),
             'geometry':[inset[sub]['graph'][e[0]][e[1]]['geometry'] for e in net.edges()],
             'flows':[flows[e] for e in net.edges()]}
        df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_edges.plot(column='flows',ax=axins,cmap=cm.plasma)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], loc2=inset[sub]['loc2'], 
               fc="none", ec="0.5")
    fig.savefig("{}{}.png".format(path,'51121-dist-flows'),bbox_inches='tight')
    return
























