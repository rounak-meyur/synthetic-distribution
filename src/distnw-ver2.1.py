# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network.
"""

import sys,os
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple as nt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D


workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Primary
from pyBuildNetworklib import read_primary,combine_primary_secondary


scratchPath = workPath
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"


#%% Get transformers and store them in csv
q_object = Query(csvPath,inpPath)
subs = q_object.GetAllSubstations()
homes = q_object.GetHomes()

#%% Get primary distribution for the substation
sub=150692
substation = nt("local_substation",field_names=["id","cord"])
sub_data = substation(id=sub,cord=subs.cord[sub])

# Generate partitions
master_graph_path = tmpPath+'prim-master/'
P = Primary(sub_data,master_graph_path,max_node=700)

#%% Partition figures
color_list = ['crimson','royalblue','magenta','gold','seagreen','olive']
node_comp = [list(c) for c in list(nx.connected_components(P.graph))]
nodepos = nx.get_node_attributes(P.graph,'cord')

nodelist = []
nodecol = []
for i in range(nx.number_connected_components(P.graph)):
    nodelist += node_comp[i]
    nodecol += [color_list[i]]*len(node_comp[i])
fig = plt.figure(figsize=(30,18))
ax = fig.add_subplot(111)
ax.scatter([subs.cord[sub][0]],[subs.cord[sub][1]],c='black',marker='*',s=5000)
nx.draw_networkx(P.graph,pos=nodepos,ax=ax,nodelist=nodelist,node_color=nodecol,
                 node_size=25.0,with_labels=False,edge_color='black',width=1)


axins = zoomed_inset_axes(ax, 2.2, loc=9)
# axins.scatter(subs.cord[sub][0],subs.cord[sub][1],s=500,c='magenta',
#               marker='*')
sgraph = nx.subgraph(P.graph,node_comp[0])
snodes = list(sgraph.nodes())
axins.set_aspect(1.3)
xpts = [nodepos[r][0] for r in snodes]
ypts = [nodepos[r][1] for r in snodes]
nx.draw_networkx(sgraph,pos=nodepos,ax=axins,node_color=color_list[0],
                 node_size=50.0,with_labels=False,edge_color='black',width=1)

axins.set_xlim(min(xpts),max(xpts))
axins.set_ylim(min(ypts),max(ypts))
axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

leglines = [Line2D([0], [0], color='white', markerfacecolor=color_list[i], 
                   marker='o',markersize=20) \
            for i in range(nx.number_connected_components(P.graph))]+\
    [Line2D([0], [0], color='white', markerfacecolor='black', marker='*',
            markersize=30),
     Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0)]
ax.legend(leglines,['Partition '+str(i+1) \
                    for i in range(nx.number_connected_components(P.graph))]+\
          ['substation','road links'],
          loc='best',ncol=1,prop={'size': 30})
fig.savefig("{}{}.png".format(figPath,'girvan-newman'),bbox_inches='tight')
sys.exit(0)

#%% Remaining stuff
# Generate the primary distribution by partitions
prim_net = P.get_sub_network(flowmax=1000,feedermax=5,grbpath=tmpPath)



#%% Output stuff
with open(inpPath+'arealist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')
secnet = q_object.GetAllSecondary(areas)

prim_data = tmpPath+"prim-network/"+str(sub)+"-primary.txt"
prim_net = read_primary(prim_data)
dist_net = combine_primary_secondary(prim_net,secnet)

try:
    print("Number of cycles:",len(nx.find_cycle(dist_net)))
except:
    print("No cycles found!!!")
    pass

#%% Plot network
xmin,ymin = (-80.4714,37.20369)
xmax,ymax = (-80.4442,37.22839)

def plot_network(dist_net,path,filename):
    """
    Plots the generated synthetic distribution network with specific colors for
    primary and secondary networks and separate color for different nodes in the
    network.
    """
    nodelist = list(dist_net.nodes())
    edgelist = list(dist_net.edges())
    nodepos = nx.get_node_attributes(dist_net,'cord')
    node_label = nx.get_node_attributes(dist_net,'label')
    edge_label = nx.get_edge_attributes(dist_net,'label')
    
    # Format the nodes in the network
    colors = []
    size = []
    for n in nodelist:
        if node_label[n] == 'T':
            colors.append('green')
            size.append(25.0)
        elif node_label[n] == 'H':
            colors.append('red')
            size.append(5.0)
        elif node_label[n] == 'R':
            colors.append('black')
            size.append(5.0)
        elif node_label[n] == 'S':
            colors.append('dodgerblue')
            size.append(2000.0)
    
    # Format the edges in the network
    edge_color = []
    edge_width = []
    for e in edgelist:
        if e in edge_label: 
            edge = e
        else:
            edge = (e[1],e[0])
        if edge_label[edge] == 'P':
            edge_color.append('black')
            edge_width.append(2.0)
        elif edge_label[edge] == 'S':
            edge_color.append('crimson')
            edge_width.append(1.0)
        else:
            edge_color.append('dodgerblue')
            edge_width.append(2.0)
    
    fig = plt.figure(figsize=(30,18))
    ax = fig.add_subplot(111)
    nx.draw_networkx(dist_net,pos=nodepos,with_labels=False,
                     ax=ax,node_size=size,node_color=colors,
                     edgelist=edgelist,edge_color=edge_color,width=edge_width)
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Define legends for the plot
    leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0),
                Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='o',markersize=0),
                Line2D([0], [0], color='dodgerblue', markerfacecolor='dodgerblue', marker='o',markersize=0),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='o',markersize=20),
                Line2D([0], [0], color='white', markerfacecolor='red', marker='o',markersize=20),
                Line2D([0], [0], color='white', markerfacecolor='dodgerblue', marker='o',markersize=20)]
    ax.legend(leglines,['primary network','secondary network','high voltage feeders',
                        'transformers','residences','substation'],
              loc='best',ncol=1,prop={'size': 30})
    
    axins = zoomed_inset_axes(ax, 2.2, loc=9)
    axins.set_aspect(1.3)
    nx.draw_networkx(dist_net,pos=nodepos,with_labels=False,
                     ax=axins,node_size=size,node_color=colors,
                     edgelist=edgelist,edge_color=edge_color,width=edge_width)
    
    axins.set_xlim(xmin,xmax)
    axins.set_ylim(ymin,ymax)
    axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
    
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return

def plot_primary(dist_net,path,filename):
    """
    Plots the generated synthetic distribution network with specific colors for
    primary and secondary networks and separate color for different nodes in the
    network.
    """
    # Delete the homes from graph
    graph = dist_net.copy()
    
    nodelist = list(graph.nodes())
    edgelist = list(graph.edges())
    nodepos = nx.get_node_attributes(graph,'cord')
    node_label = nx.get_node_attributes(graph,'label')
    edge_label = nx.get_edge_attributes(graph,'label')
    
    # Format the nodes in the network
    colors = []
    size = []
    for n in nodelist:
        if node_label[n] == 'T':
            colors.append('green')
            size.append(25.0)
        elif node_label[n] == 'H':
            colors.append('red')
            size.append(5.0)
        elif node_label[n] == 'R':
            colors.append('black')
            size.append(5.0)
        elif node_label[n] == 'S':
            colors.append('dodgerblue')
            size.append(2000.0)
    
    # Format the edges in the network
    edge_color = []
    edge_width = []
    for e in edgelist:
        if e in edge_label: 
            edge = e
        else:
            edge = (e[1],e[0])
        if edge_label[edge] == 'P':
            edge_color.append('black')
            edge_width.append(2.0)
        elif edge_label[edge] == 'S':
            edge_color.append('crimson')
            edge_width.append(1.0)
        else:
            edge_color.append('dodgerblue')
            edge_width.append(2.0)
    
    fig = plt.figure(figsize=(30,18))
    ax = fig.add_subplot(111)
    nx.draw_networkx(graph,pos=nodepos,with_labels=False,
                     ax=ax,node_size=size,node_color=colors,
                     edgelist=edgelist,edge_color=edge_color,width=edge_width)
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Define legends for the plot
    leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0),
                Line2D([0], [0], color='dodgerblue', markerfacecolor='dodgerblue', marker='o',markersize=0),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='o',markersize=20),
                Line2D([0], [0], color='white', markerfacecolor='dodgerblue', marker='o',markersize=20)]
    ax.legend(leglines,['primary network','high voltage feeders',
                        'transformers','substation'],
              loc='best',ncol=1,prop={'size': 30})
    
    axins = zoomed_inset_axes(ax, 2.2, loc=9)
    axins.set_aspect(1.3)
    nx.draw_networkx(graph,pos=nodepos,with_labels=False,
                     ax=axins,node_size=size,node_color=colors,
                     edgelist=edgelist,edge_color=edge_color,width=edge_width)
    
    axins.set_xlim(xmin,xmax)
    axins.set_ylim(ymin,ymax)
    axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
    
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return

plot_primary(prim_net,figPath,'150692-primary')
plot_network(dist_net,figPath,'150692-network')


#%% Voltage and power flow
import numpy as np
from matplotlib import cm

def check_pf(dist_net,path,filename):
    """
    Checks power flow solution and plots the voltage at different nodes in the 
    network through colorbars.
    """
    nodepos = nx.get_node_attributes(dist_net,'cord')
    A = nx.incidence_matrix(dist_net,nodelist=list(dist_net.nodes()),
                            edgelist=list(dist_net.edges()),oriented=True).toarray()
    
    nodelabel = nx.get_node_attributes(dist_net,'label')
    
    node_ind = [i for i,node in enumerate(dist_net.nodes()) \
                if nodelabel[node] != 'S']
    nodelist = [node for node in list(dist_net.nodes()) if nodelabel[node] != 'S']
    
    # Resistance data
    edge_r = nx.get_edge_attributes(dist_net,'r')
    R = np.diag([1.0/edge_r[e] if e in edge_r else 1.0/edge_r[(e[1],e[0])] \
         for e in list(dist_net.edges())])
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*homes.average[h] if nodelabel[h]=='H' else 0.0 \
                  for h in nodelist])
    
    
    
    v = np.matmul(np.linalg.inv(G),p)
    voltage = {h:1.0-v[i] for i,h in enumerate(nodelist)}
    subnodes = [node for node in list(dist_net.nodes()) if nodelabel[node] == 'S']
    for s in subnodes: voltage[s] = 1.0
    nodes = list(dist_net.nodes())
    colors = [voltage[n] for n in nodes]
    
    # Generate visual representation
    fig = plt.figure(figsize=(32,18))
    ax = fig.add_subplot(111)
    nx.draw_networkx(dist_net, nodepos, ax=ax, node_color=colors,
        node_size=25, cmap=plt.cm.plasma, with_labels=False, 
        vmin=0.85, vmax=1.05)
    cobj = cm.ScalarMappable(cmap='plasma')
    cobj.set_clim(vmin=0.80,vmax=1.05)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label('Voltage(pu)',size=30)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    # ax.set_xlabel('Longitude',fontsize=20)
    # ax.set_ylabel('Latitude',fontsize=20)
    
    
    axins = zoomed_inset_axes(ax, 2.2, loc=9)
    axins.set_aspect(1.3)
    nx.draw_networkx(dist_net, nodepos, ax=axins, node_color=colors,
        node_size=25, cmap=plt.cm.plasma, with_labels=False, 
        vmin=0.85, vmax=1.05)
    
    axins.set_xlim(xmin,xmax)
    axins.set_ylim(ymin,ymax)
    axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
    
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return colors
    
def check_flows(dist_net,path,filename):
    """
    Checks power flow solution and plots the flows at different edges in the 
    network through colorbars.
    """
    nodepos = nx.get_node_attributes(dist_net,'cord')
    A = nx.incidence_matrix(dist_net,nodelist=list(dist_net.nodes()),
                            edgelist=list(dist_net.edges()),oriented=True).toarray()
    
    nodelabel = nx.get_node_attributes(dist_net,'label')
    
    node_ind = [i for i,node in enumerate(dist_net.nodes()) \
                if nodelabel[node] != 'S']
    nodelist = [node for node in list(dist_net.nodes()) if nodelabel[node] != 'S']
    edgelist = [edge for edge in list(dist_net.edges())]
    
    # Resistance data
    p = np.array([1e-3*homes.average[h] if nodelabel[h]=='H' else 0.0 \
                  for h in nodelist])
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    
    from math import log
    flows = {e:log(abs(f[i])) for i,e in enumerate(edgelist)}
    # edgelabel = nx.get_edge_attributes(dist_net,'label')
    colors = [flows[e] for e in edgelist]
    fmin = 0.2
    fmax = 800.0
    
    # Generate visual representation
    fig = plt.figure(figsize=(32,18))
    ax = fig.add_subplot(111)
    nx.draw_networkx(dist_net, nodepos, ax=ax, edge_color=colors,node_color='black',
        node_size=1.0, edge_cmap=plt.cm.plasma, with_labels=False, 
        vmin=log(fmin), vmax=log(fmax), width=3)
    cobj = cm.ScalarMappable(cmap='plasma')
    cobj.set_clim(vmin=fmin,vmax=fmax)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label('Flow along edge in kVA',size=30)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    # ax.set_xlabel('Longitude',fontsize=20)
    # ax.set_ylabel('Latitude',fontsize=20)
    
    
    axins = zoomed_inset_axes(ax, 2.2, loc=9)
    axins.set_aspect(1.3)
    nx.draw_networkx(dist_net, nodepos, ax=axins, edge_color=colors,node_color='black',
        node_size=1.0, edge_cmap=plt.cm.plasma, with_labels=False, 
        vmin=log(fmin), vmax=log(fmax), width=3)
    
    axins.set_xlim(xmin,xmax)
    axins.set_ylim(ymin,ymax)
    axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
    
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return colors

#run power flow
voltage = check_pf(dist_net,figPath,'150692-voltage')
flows = check_flows(dist_net,figPath,'150692-flows')

