# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:20:45 2019

Author: Dr Anil Vullikanti
        Rounak Meyur
        
Description: Generates primary distribution network for Montgomery county
"""


import sys,os
import numpy as np
import networkx as nx
import cPickle as pkl
import matplotlib.pyplot as plt


workPath = os.getcwd()
LibPath = workPath + "\\Libraries\\"
BinPath = workPath + "\\tmp\\"
InpPath = workPath + "\\input\\"
FigPath = workPath + "\\figs\\"

sys.path.append(LibPath)
import pyGenerateDistlib
reload(pyGenerateDistlib)


#%%
road_network_fname = "core-link-file-Montgomery-VA.txt"
road_coord_fname = "node-geometry-Montgomery-VA.txt"

road_graph_pkl_file = "road-network-graph.p"
road_coord_pkl_file = "road-coordinates.p"
home_coord_pkl_file = "home-coordinates.p"
home2road_pkl_file = "home2road-mapping.p"
road2home_pkl_file = "road2home-mapping.p"

RoadNW_coord = pkl.load(open(BinPath+road_coord_pkl_file,'rb'))
Home_coord = pkl.load(open(BinPath+home_coord_pkl_file,'rb'))
G = pkl.load(open(BinPath+road_graph_pkl_file,'rb'))
H2R = pkl.load(open(BinPath+home2road_pkl_file,'rb'))
R2H = pkl.load(open(BinPath+road2home_pkl_file,'rb'))

#%% COMMENT OUT THIS SECTION AFTER FIRST RUN
#RoadNW_coord, G = pyGenerateDistlib.ReadRoadNetwork(InpPath,road_network_fname,road_coord_fname)
#Home_coord = pkl.load(open(BinPath+home_coord_pkl_file,'rb'))
#H2R,R2H = pyGenerateDistlib.map_Homes_to_RoadNW(Home_coord, RoadNW_coord)
#
#pkl.dump(RoadNW_coord,open(BinPath+road_coord_pkl_file,'wb'))
#pkl.dump(G,open(BinPath+road_graph_pkl_file,'wb'))
#pkl.dump(H2R,open(BinPath+home2road_pkl_file,'wb'))
#pkl.dump(R2H,open(BinPath+road2home_pkl_file,'wb'))

#%%
#input_csv_file = InpPath + "121RounakEPData\\121VAMay2015_05Active.csv"
#f = open(input_csv_file,'r')
#lines = f.readlines()[1:]
#f.close()
#L = [l.strip('\n').split(',') for l in lines]
#HomeAct = {int(L[k][0]):[float(x) for x in L[k][147:171]] for k in range(len(L))}


#%%
f = plt.figure(figsize=(10,10))
#for sub in sub_list:
#    plt.plot(Sub_Coord[sub][0],Sub_Coord[sub][1],'go')
for e in G.edges():
    x=[RoadNW_coord[e[0]][0],RoadNW_coord[e[1]][0]]
    y=[RoadNW_coord[e[0]][1],RoadNW_coord[e[1]][1]]
    plt.plot(x,y,linewidth=0.1,color = 'b')
    
for r in R2H:
    x = RoadNW_coord[r][0]; y = RoadNW_coord[r][1]
    plt.plot(x,y,'r.')











