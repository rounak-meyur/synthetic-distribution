# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:09:58 2018

Author: Dr Anil Vullikanti
        Rounak Meyur
"""


import sys,os
import networkx as nx
import numpy as np
import time
import cPickle as pkl
import matplotlib.pyplot as plt
workPath = os.getcwd()

LibPath = workPath + "/Libraries/"
BinPath = workPath + "/tmp/"
InpPath = workPath + "/input/"
FigPath = workPath + "/figs/"

sys.path.append(LibPath)
import pyGenerateLinklib
reload(pyGenerateLinklib)

sys.exit(0)
#%%
if __name__ == "__main__":
    
    start_time = time.time()
    
    road_network_fname = "core-link-file-Montgomery-VA.txt"
    road_coord_fname = "node-geometry-Montgomery-VA.txt"
    
    homeact_out_fname = "homeact_pkl_file.p"
    subpos_out_fname = "subpos_pkl_file.p"
    roadpos_out_fname = "roadnet_pkl_file.p"
    road_graph_fname = "roadgraph_pkl_file.p"
    home2link_map_fname = "home2link_pkl_file.p"
    Sub2RoadNear_map_fname = "mapsub2roadnear_pkl_file.p"
    network_dist_fname = "networkdistance_pkl_file.p"
    Sub2Road_map_fname = "mapsub2road_pkl_file.p"
    Road2Sub_map_fname = "maproad2sub_pkl_file.p"
    road2home_map_fname = "maproad2home_pkl_file.p"
    
    # input coordinates
    Home_Coord = pkl.load(open(BinPath+homeact_out_fname,'rb'))
    Sub_Coord = pkl.load(open(BinPath+subpos_out_fname,'rb'))
    Sub_Coord = {k:Sub_Coord[k] for k in Sub_Coord if (-80.575<=Sub_Coord[k][0]<-80.2) and (Sub_Coord[k][1]>=37.0)}
    RoadNW_Coord = pkl.load(open(BinPath+roadpos_out_fname,'rb'))
    
    # graph of road network
    G = pkl.load(open(BinPath+road_graph_fname,'rb'))
    
    # map between road and homes
    Home2Link = pkl.load(open(BinPath+home2link_map_fname,'rb'))
    HomeSideA,HomeSideB = pyGenerateLinklib.SeparateHomeLink(Home2Link,Home_Coord,RoadNW_Coord)
    
    #RW2Home = {r:[] for r in RoadNW_Coord}
    #RW2Home = pyGenerateLinklib.MapHomeRoad(Home_Coord,RoadNW_Coord,HomeSideA,RW2Home)
    #RW2Home = pyGenerateLinklib.MapHomeRoad(Home_Coord,RoadNW_Coord,HomeSideB,RW2Home)
    
    # map between road and substation
    Sub2RoadNear = pkl.load(open(BinPath+Sub2RoadNear_map_fname,'rb'))
    Sub2Road = pkl.load(open(BinPath+Sub2Road_map_fname,'rb'))
    Road2Sub = pkl.load(open(BinPath+Road2Sub_map_fname,'rb'))
    
    
    
    RW2Home = pkl.load(open(road2home_map_fname,'rb'))
    
#%%
#    sub_list = [28228,28234,28235,28236,34722,34780,34811,34812,34813]
    sub_list = [28228]
#    sub_list = [34722,34780,34811,34812,34813]
    
    # Build the edges for the primary distribution network
    # Update each edge with the feeder substation ID and type as 'prim'
    # Update each node with the coordinate information
    P = nx.DiGraph()
    dict_sub = {}
    for sub in sub_list:
        H = nx.subgraph(G,Sub2Road[sub])
        P = nx.compose(P,nx.dfs_tree(H,Sub2RoadNear[sub]))
        dict_sub.update({e:sub for e in list(P.edges())})
    # set edge attribute as primary distribution network
    nx.set_edge_attributes(P,dict_sub,'sub')
    nx.set_edge_attributes(P, 'P', 'type')
    # Update 
    dict_coord = {r:RoadNW_Coord[r] for r in list(P.nodes())}
    
    
    #
    T = nx.DiGraph()
    for r in list(P.nodes()):
        S = nx.DiGraph()
        for path in RW2Home[r]:
            dict_coord.update({h:Home_Coord[h] for h in path})
            new_path = [r]+path
            S.add_path(new_path)
        dict_sub = {e:Road2Sub[r] for e in list(S.edges())}
        nx.set_edge_attributes(S,dict_sub,'sub')
        nx.set_edge_attributes(S, 'S', 'type')
        T = nx.compose(T,S)
    T = nx.compose(T,P)
    nx.set_node_attributes(T,dict_coord,'Coordinates')
    
    f = plt.figure(figsize=(10,10))
    for sub in sub_list:
        plt.plot(Sub_Coord[sub][0],Sub_Coord[sub][1],'go')
    for e in T.edges():
        x=[T.nodes[e[0]]['Coordinates'][0],T.nodes[e[1]]['Coordinates'][0]]
        y=[T.nodes[e[0]]['Coordinates'][1],T.nodes[e[1]]['Coordinates'][1]]
        color = 'r' if T.edges[e[0],e[1]]['type']=='P' else 'b'
        size = 1.0 if T.edges[e[0],e[1]]['type']=='P' else 0.5
        plt.plot(x,y,color=color,linewidth=size)

    f.savefig(FigPath+"dist-blacksburg28228-prim-sec.pdf",bbox_inches='tight')
    
#%%
#    Wt_Home = {k:rnd.random() for k in Home_Coord}
#    Wt_Road = {m:sum([Wt_Home[h] for h in RW2Home[m]]) if m in RW2Home else 0 for m in list(G.nodes())}    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    