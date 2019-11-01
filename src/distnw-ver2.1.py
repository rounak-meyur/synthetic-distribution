# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider
from pyBuildNetworklib import MeasureDistance as Dist
from pyMILPlib import MILP_primary

#%% Functions
f = open(inpPath+'secondary-network.txt')
lines = f.readlines()
f.close()

sec_edgelist = [(int(temp.strip('\n').split(' ')[0]),
                 int(temp.strip('\n').split(' ')[4])) \
                for temp in lines]
secondary_net = nx.Graph()
secondary_net.add_edges_from(sec_edgelist)

#%% Classes
from scipy.spatial import Voronoi,cKDTree,voronoi_plot_2d
class Primary:
    """
    """
    def __init__(self,dictsubs,dictnode):
        """
        """
        self.subs = dictsubs
        self.nodes = dictnode
        self.node_pts = [dictnode[n] for n in dictnode]
        self.sub_pts = [dictsubs[s] for s in dictsubs]
        self.S2Node = {}
        self.S2Near = {}
    
    def GetGroups(self):
        """
        """
        # Find number of road nodes mapped to each substation
        voronoi_kdtree = cKDTree(self.sub_pts)
        _, node_regions = voronoi_kdtree.query(self.node_pts, k=1)
        sub_map = {s:node_regions.tolist().count(s)\
                   for s in range(len(self.subs))}
        
        # Get substations with minimum number of road nodes
        sub_region = [list(self.subs.keys())[s] \
                      for s in sub_map if sub_map[s]>50]
        self.sub_pts = [self.subs[s] for s in sub_region]
        
        # Recompute the Voronoi regions and generate the final map
        voronoi_kdtree = cKDTree(self.sub_pts)
        _, node_regions = voronoi_kdtree.query(self.node_pts, k=1)
        
        # Index the region and assign the nodes to the substation
        indS2N = [np.argwhere(i==node_regions)[:,0]\
                  for i in np.unique(node_regions)]
        self.S2Node = {sub_region[i]:[list(self.nodes.keys())[j]\
                       for j in indS2N[i]] for i in range(len(indS2N))}
        
        self.S2Near = {}
        for s in self.S2Node:
            nodes = self.S2Node[s]
            dis = [Dist(subs.cord[s],self.nodes[n]) for n in nodes]
            self.S2Near[s] = nodes[dis.index(min(dis))]
        return
    
    
    def plot_voronoi(self,S,V,title):
        """
        """
        # Plot Voronoi regions with mapped road network nodes
        vor = Voronoi([self.subs[s] for s in S])
        RW_xval = [self.nodes[n][0] for n in V]
        RW_yval = [self.nodes[n][1] for n in V]
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.scatter(RW_xval,RW_yval,c='r',s=0.5)
        voronoi_plot_2d(vor,ax=ax,show_vertices=False,point_size=20,
                        line_width=1.0,line_colors='g')
        ax.set_xlabel("Longitude",fontsize=20)
        ax.set_ylabel("Latitude",fontsize=20)
        ax.set_title(title,fontsize=20)
        plt.show()
        return
#%% Get transformers and store them in csv
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
#roads = q_object.GetRoads(level=[1,2,3,4,5])
roads = q_object.GetRoads()
subs = q_object.GetSubstations()
tsfr = q_object.GetTransformers()

df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
spider_obj = Spider(homes,roads,H2Link)
L2Home = spider_obj.link_to_home
links = [l for l in L2Home if 0<len(L2Home[l])<=70]

#%% Create the dummy graph with all possible edges between nodes
road_edges = list(roads.graph.edges())
tsfr_edges = list(tsfr.graph.edges())

for edge in links:
    try:
        road_edges.remove(edge)
    except:
        road_edges.remove((edge[1],edge[0]))

edgelist = road_edges + tsfr_edges
graph = nx.Graph()
graph.add_edges_from(edgelist)
nodelist = list(graph.nodes())
dict_node = {n:roads.cord[n] if n in roads.cord else tsfr.cord[n] \
             for n in nodelist}


#%% Primary Network Generation
P = Primary(subs.cord,dict_node)
#P.plot_voronoi(subs.cord,dict_node,
#               title="Voronoi regions formed by all substations")
P.GetGroups()
S2Node = P.S2Node
S2Near = P.S2Near
#P.plot_voronoi(S2Node,dict_node,
#               title="Voronoi regions formed by substations in the region")


#%% Join disconnected components of the subgraph
from scipy.spatial import KDTree
s = 34816
G = nx.subgraph(graph,S2Node[s])
C = list(nx.connected_components(G))
comp_size = [len(list(c)) for c in C]
largest = list(C[comp_size.index(max(comp_size))])
others = [i for i in range(len(C)) if i!=comp_size.index(max(comp_size))]
tree = KDTree(np.array([dict_node[n] for n in largest]))
new_edges = []
for j in others:
    pts = list(C[j])
    pt_cords = np.array([dict_node[n] for n in pts])
    dis,index = tree.query(pt_cords)
    node1 = pts[np.argmin(dis)]
    node2 = largest[index[np.argmin(dis)]]
    new_edges.append((node1,node2))

#%%
graph.add_edges_from(new_edges)
length = {e:Dist(dict_node[e[0]],dict_node[e[1]]) for e in list(graph.edges())}
nx.set_edge_attributes(graph,length,'length')
G = nx.subgraph(graph,S2Node[s])
tnodes = [n for n in S2Node[s] if n in tsfr.cord]
snodes = [S2Near[s]]

homelist = []
for t in tnodes:
    homelist.extend(list(nx.descendants(secondary_net,t)))
secondary = list(secondary_net.edges(homelist+tnodes))

#%%
M = MILP_primary(G,tnodes,snodes)

#%% Plot the result
primary = M.optimal_edges
F = nx.Graph()
F.add_edges_from(primary+secondary)
nodelist = list(F.nodes())
dict_node.update(homes.cord)
colors = []
size = []
for n in nodelist:
    if n in tsfr.cord:
        colors.append('red')
        size.append(5.0)
    elif n in homes.cord:
        colors.append('blue')
        size.append(2.0)
    else:
        colors.append('black')
        size.append(1.0)

edge_color = []
for e in list(F.edges()):
    if e in primary or (e[1],e[0]) in primary:
        edge_color.append('black')
    else:
        edge_color.append('orchid')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
nx.draw_networkx(F,pos=dict_node,with_labels=False,
                 ax=ax,node_size=size,node_color=colors,
                 edgelist=list(F.edges()),edge_color=edge_color)
ax.scatter(subs.cord[s][0],subs.cord[s][1],s=100,c='seagreen')
ax.set_title("Distribution Network from Ellet Road Substation",fontsize=20)



#%% Plot check
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
roads = q_object.GetRoads(level=[1,2,3,4,5])
labels = nx.get_edge_attributes(roads.graph,'level')
col = ['red','yellow','black','green','blue']
edgelist = list(labels.keys())
edgecolor = [col[labels[e]-1] for e in edgelist]
nx.draw_networkx(roads.graph,pos=roads.cord,ax=ax,edge_color=edgecolor,
                 node_size=0.5,with_labels=False,line_width=0.5)
#S=[28236]
#subx = [subs.cord[s][0] for s in S]
#suby = [subs.cord[s][1] for s in S]
#ax.scatter(subx,suby,s=100,c='red')

subx = [homes.cord[h][0] for h in homes.cord]
suby = [homes.cord[h][1] for h in homes.cord]
ax.scatter(subx,suby,s=0.5,c='red')
plt.show()
