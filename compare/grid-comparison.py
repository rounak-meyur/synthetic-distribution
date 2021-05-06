# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:16:52 2021

@author: rouna
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
actpath = rootpath + "/input/actual/"
synpath = rootpath + "/primnet/out/"

sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet,get_areadata,plot_network
from pyGeometrylib import Grid,partitions,MeasureDistance
print("Imported modules")




sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

#%% Area specifications
#areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}

# Get limits for the geographical region
lims = np.empty(shape=(len(area_data),4))
for i,area in enumerate(area_data):
    lims[i,:] = np.array(area_data[area]['limits'])
LEFT = np.min(lims[:,0]); RIGHT = np.max(lims[:,1])
BOTTOM = np.min(lims[:,2]); TOP = np.max(lims[:,3])
width = RIGHT-LEFT
height = TOP-BOTTOM


#%% Grid comaprison
from shapely.geometry import Polygon, LineString, Point



# Distribution of edges
def edge_dist(grid,edge_geom):    
    m_within = {}
    m_cross = {}    
    for g in grid:
        m_within[g] = sum([geom.within(g) for geom in edge_geom])
        m_cross[g] = sum([geom.intersects(g.exterior) for geom in edge_geom])
    return m_within,m_cross

# Distribution of nodes
def node_dist(grid,node_geom):    
    return {g:sum([geom.within(g) for geom in node_geom]) for g in grid}





#%% Count the number of node and edges in actual and synthetic networks
act_geom = [g for area in area_data \
             for g in area_data[area]['df_lines']['geometry'].tolist()]
synth_geom = [g for area in area_data \
             for g in area_data[area]['df_synth']['geometry'].tolist()]


act_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_buses']['geometry'].tolist()]
synth_nodes_geom = [g for area in area_data \
             for g in area_data[area]['df_cords']['geometry'].tolist()]

kx = 5; ky = 5
gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky)
x_array = np.array([LEFT+(t/kx)*(RIGHT-LEFT) for t in range(kx)]+[RIGHT])
y_array = np.array([BOTTOM+(t/ky)*(TOP-BOTTOM) for t in range(ky)]+[TOP])



A = node_dist(gridlist,act_nodes_geom)
B = node_dist(gridlist,synth_nodes_geom)
A_nodes = sum(list(A.values()))
B_nodes = sum(list(B.values()))

C = {bound:100.0*(1-((B[bound]/B_nodes)/(A[bound]/A_nodes))) \
     if A[bound]!=0 else np.nan for bound in gridlist}

C_vals = np.array([C[bound] for bound in C])
C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))

#%% Plot the spatial distribution
colormap = cm.BrBG

def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge,boundary.north_edge]
    vert2 = [boundary.east_edge,boundary.north_edge]
    vert3 = [boundary.east_edge,boundary.south_edge]
    vert4 = [boundary.west_edge,boundary.south_edge]
    return np.array([vert1,vert2,vert3,vert4])

# colormap = cm.Greens

DPI = 72    
fig = plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
ax = plt.subplot()
ax.set_xlim(LEFT,RIGHT)
ax.set_ylim(BOTTOM,TOP)
ax.pcolor(x_array,y_array,C_masked.reshape(kx,ky).T,cmap=colormap,
          edgecolor='black')

# Get the boxes for absent actual data
verts_invalid = [get_polygon(bound) for i,bound in enumerate(C) \
               if C_masked.mask[i]]
c = PolyCollection(verts_invalid,hatch=r"./",facecolor='white',
                   edgecolor='black')
ax.add_collection(c)


for area in area_data:
    plot_network(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],
                 'orangered')
    plot_network(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],
                 'blue')



ax.set_xticks([])
ax.set_yticks([])

cobj = cm.ScalarMappable(cmap=colormap)
cobj.set_clim(vmin=-100.0,vmax=100.0)
cbar = fig.colorbar(cobj,ax=ax)
cbar.set_label('Percentage Deviation',size=20)
cbar.ax.tick_params(labelsize=20)

leg_data = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                   marker='o',markersize=10, label='Actual distribution network'),
            Line2D([0], [0], color='blue', markerfacecolor='blue',
                   marker='o',markersize=10, label='Synthetic distribution network'),
            Patch(facecolor='white', edgecolor='black', hatch="./",
                         label='Grids with no actual network data')]

ax.legend(handles=leg_data,loc='best',ncol=1,prop={'size': 8})



#%% Efficiency computation
from itertools import combinations

act_nodes = [n for area in area_data \
             for n in area_data[area]['df_buses']['id'].tolist()]
synth_nodes = [n for area in area_data \
             for n in area_data[area]['df_cords']['nodes'].tolist()]

eff = []
ncount = []
for grid in gridlist:

    # Actual network
    nodes = [n for i,n in enumerate(act_nodes) if act_nodes_geom[i].within(grid)]
    node_pairs = list(combinations(nodes, 2))
    frac = 0
    for pair in node_pairs:
        for area in area_data:
            net = area_data[area]['actual']
            df_buses = area_data[area]["df_buses"]
            if int(pair[0]) in net and int(pair[1]) in net:
                length = nx.shortest_path_length(net,int(pair[0]),
                                                int(pair[1]),'geo_length')
                pt1 = df_buses.loc[df_buses.id == pair[0]]['geometry'].values[0]
                pt2 = df_buses.loc[df_buses.id == pair[1]]['geometry'].values[0]
                distance = MeasureDistance([pt1.x,pt1.y],[pt2.x,pt2.y])
                break
        frac += distance/length
    try:
        act_eff = frac/len(node_pairs)
    except:
        act_eff = float('nan')
    
    # Synthetic network
    s_nodes = [n for i,n in enumerate(synth_nodes) if synth_nodes_geom[i].within(grid)\
               and synth_net.nodes[n]['label']=='T']
    s_node_pairs = list(combinations(s_nodes, 2))
    s_frac = 0
    for s_pair in s_node_pairs:
        try:
            length = nx.shortest_path_length(synth_net,int(s_pair[0]),
                                        int(s_pair[1]),'geo_length')
        except:
            length = float('inf')
        distance = MeasureDistance(synth_net.nodes[s_pair[0]]['cord'],
                                   synth_net.nodes[s_pair[1]]['cord'])
        s_frac += distance/length if length != 0 else 0.0
    try:
        synth_eff = s_frac/len(s_node_pairs)
    except:
        synth_eff = float('nan')
    
    eff.append((act_eff,synth_eff))
    ncount.append((len(nodes),len(s_nodes)))


#%% Plot efficiency
import math
act_node_count = [ncount[i][0] for i in range(len(gridlist)) if ncount[i][0]!=0]
syn_node_count = [ncount[i][1] for i in range(len(gridlist))]
act_efficiency = [eff[i][0] for i in range(len(gridlist)) if not math.isnan(eff[i][0])]
syn_efficiency = [eff[i][1] for i in range(len(gridlist))]


DPI = 72    
fig = plt.figure(figsize=(1000/DPI, 1000/DPI), dpi=DPI)
ax = fig.add_subplot(111)
ax.scatter(syn_node_count,syn_efficiency,marker='*',c='blue',s=500,label="Synthetic Network")
ax.scatter(act_node_count,act_efficiency,marker='*',c='orangered',s=500,label="Actual Network")
ax.legend(loc='best',ncol=1,prop={'size': 15})



K_act = np.polyfit(np.log(act_node_count),act_efficiency, 1)
K_syn = np.polyfit(np.log(syn_node_count),syn_efficiency, 1)
x = sorted(act_node_count)
y = K_act[0]*np.log(x)+K_act[1]
ax.plot(x,y,color='orangered')
x = sorted(syn_node_count)
y = K_syn[0]*np.log(x)+K_syn[1]
ax.plot(x,y,color='blue')


#%% Local Efficiency











