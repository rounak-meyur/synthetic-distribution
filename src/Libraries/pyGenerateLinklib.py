# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:42:21 2018

Authors: Dr Anil Vullikanti
         Dr Henning Mortveit
         Rounak Meyur
"""


import networkx as nx
import csv
#import copy
import numpy as np
#from scipy.spatial import cKDTree
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
#from joblib import Parallel, delayed
#import multiprocessing as mp


def MeasureDistance(Point1,Point2):
    '''
    '''
    # Approximate radius of earth in km
    R = 6373.0
    
    # Get the longitude and latitudes of the two points
    lat1 = radians(Point1[1])
    lon1 = radians(Point1[0])
    lat2 = radians(Point2[1])
    lon2 = radians(Point2[0])
    
    # Measure the long-lat difference
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Calculate distance between points in km
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def DistanceBetweenSubstation(Sub_Coord):
    '''
    '''
    k = Sub_Coord.keys()
    A=np.transpose([np.tile(k, len(k)), np.repeat(k, len(k))])
    distance = []
    for i in range(len(A)):
        distance.append(MeasureDistance(Sub_Coord[A[i][0]],Sub_Coord[A[i][1]]))
    return np.array(distance).reshape((len(k),len(k)))


def ReadRoadNetwork(pathname,road_network_fname,road_coord_fname):
    '''
    Creates a networkx graph for the road network with only the edges between
    Level 4 and Level 5 nodes. The isolated nodes are removed from the graph.
    
    Inputs: road_network_fname: file name for the road network edgelist data
            road_coord_fname: file name for the road network coordinates data
    
    Outputs: RoadNW_coord: dictionary of road network nodes and their coordinates
             G: Networkx graph representation of the road network
    '''
    # Create the road network graph
    road_f_reader = csv.reader(open(pathname+road_network_fname), delimiter = '\t')
    G=nx.Graph()
    V={}
    for line in road_f_reader:
        if ('#' in line[0]): continue
        v = int(line[0]); w=int(line[1])
        if (v not in V): 
            G.add_node(v)
            V[v]=1
        if (w not in V): 
            G.add_node(w)
            V[w]=1
        if (line[2] == '3' or line[2] == '4' or line[2] == '5'): G.add_edge(v, w, weight=line[2])
        
    
    # Remove isolated nodes
    isolated_nodes = [nd for nd in G.nodes() if G.degree(nd)==0]
    for v in isolated_nodes:
        G.remove_node(v)
    
    # Create the road network coordinate dictionary
    RoadNW_coord={}
    coord_f_reader = csv.reader(open(pathname+road_coord_fname), delimiter = '\t')
    for line in coord_f_reader:
        if ('#' in line[0]): continue
        RoadNW_coord[int(line[0])] = float(line[1]), float(line[2])
  
    return RoadNW_coord, G


###############################################################################
# Functions to map road network with substations
def MapSub2RoadNear(Sub_Coord,RoadNW_Coord,G):
    '''
    '''
    sub = Sub_Coord.keys()
    k = list(G.nodes())
    
    A = np.transpose([np.tile(sub, len(k)), np.repeat(k, len(sub))])
    distance = []
    for i in range(len(A)):
        distance.append(MeasureDistance(Sub_Coord[A[i][0]],RoadNW_Coord[A[i][1]]))
    B = np.array(distance).reshape((len(k),len(sub)))
    
    Sub2NearRoad = {sub[s]:k[np.argmin(B[:,s])] for s in range(len(sub))}
    return Sub2NearRoad


def MapRoadSub(G,Sub2RoadNear,network_distance):
    '''
    '''
    # Get index of the substation nearby road nodes in the nodelist of graph 
    index = [list(G.nodes()).index(i) for i in Sub2RoadNear.values()]
    
    # Select only the columns of network distance corresponding to these nodes
    net_dist = network_distance[:,index].tolist()
    
    # Map each road node to a substation nearby road node
    Road2RoadNear = {list(G.nodes())[k]:list(G.nodes())[index[np.argmin(net_dist[k])]] for k in range(G.number_of_nodes())}
    
    # Map each road node to a substation
    RoadNear2Sub = {Sub2RoadNear[sub]:sub for sub in Sub2RoadNear}
    Road2Sub = {k:RoadNear2Sub[Road2RoadNear[k]] for k in Road2RoadNear}
    
    # Mapping between roads and substations
    Sub2Road = {sub:[] for sub in Sub2RoadNear}
    for r in Road2Sub:
        Sub2Road[Road2Sub[r]].append(r)
    return Road2Sub,Sub2Road


###############################################################################
# Functions to map road network with houses
def GetNearLink(homeID,HomeCoord,RWCoord,Edges):
    '''
    '''
    dist = []
    point = Point(HomeCoord[homeID])
    for e in Edges:
        line = LineString([(RWCoord[e[0]][0],RWCoord[e[0]][1]),(RWCoord[e[1]][0],RWCoord[e[1]][1])])
        dist.append(point.distance(line))
    
    return Edges[np.argmin(dist)]
        

def MapHomeLink(G,HomeCoord,RWCoord):
    '''
    '''
    Homes = HomeCoord.keys()
    Edges = list(G.edges())
    
    Home2Link=[]
    for h in Homes:
        Home2Link.append(GetNearLink(h,HomeCoord,RWCoord,Edges))
    return Home2Link


def SeparateHomeLink(Home2Link,HomeCoord,RWCoord):
    '''
    '''
    Link2Home = {}
    # Generate the opposite mapping
    for h in Home2Link:
        if Home2Link[h] in Link2Home:
            Link2Home[Home2Link[h]].append(h)
        else:
            Link2Home[Home2Link[h]]=[h]
    # for each link, separate the homes
    HomeSideA = {l:[] for l in Link2Home}
    HomeSideB = {l:[] for l in Link2Home}
    for l in Link2Home:
        d = [(HomeCoord[h][0]-RWCoord[l[0]][0])*(RWCoord[l[1]][1]-RWCoord[l[0]][1]) \
             -(HomeCoord[h][1]-RWCoord[l[0]][1])*(RWCoord[l[1]][0]-RWCoord[l[0]][0]) \
             for h in Link2Home[l]]
        for i in range(len(Link2Home[l])):
            if d[i]>=0: HomeSideA[l].append(Link2Home[l][i])
            else: HomeSideB[l].append(Link2Home[l][i])
    return HomeSideA,HomeSideB


def MapHomeRoad(HomeCoord,RWCoord,HomeSide,RW2Home):
    '''
    '''
    for l in HomeSide:
        dist1 = [MeasureDistance(HomeCoord[h],RWCoord[l[0]]) for h in HomeSide[l]]
        dist2 = [MeasureDistance(HomeCoord[h],RWCoord[l[1]]) for h in HomeSide[l]]
        home_list_p1 = []; home_dist_p1 = []; home_list_p2 = []; home_dist_p2 = []
        for i in range(len(HomeSide[l])):
            if dist1[i]<=dist2[i]:
                home_list_p1.append(HomeSide[l][i])
                home_dist_p1.append(dist1[i])
            else:
                home_list_p2.append(HomeSide[l][i])
                home_dist_p2.append(dist2[i]) 
        RW2Home[l[0]].append([x for _,x in sorted(zip(home_dist_p1,home_list_p1))])
        RW2Home[l[1]].append([x for _,x in sorted(zip(home_dist_p2,home_list_p2))])
    
    return RW2Home


###############################################################################
#Functions to draw graph
def draw_graph(G, Coords, fname, justplot, shape=0, color=0):
    '''
    '''
    if (justplot==0): f = plt.figure(figsize=(10,10))
    for e in G.edges():
        x=[Coords[e[0]][0], Coords[e[1]][0]]
        y=[Coords[e[0]][1], Coords[e[1]][1]]
        if (shape==0 and color==0): plt.plot(x, y, 'r-')
        if (shape==1 and color==0): plt.plot(x, y, 'ro-')
        if (shape==1 and color==1): plt.plot(x, y, color = 'grey', linestyle='-', marker='o', alpha=0.3)
    if (justplot==0): f.savefig(fname, bbox_inches='tight')



def draw_paths(Paths, RoadNW_coord, Sub_coord):
    '''
    '''
    fname = "tmp/dist-minus-substation.pdf"
  #fname = "tmp/dist-with-substation.pdf"
    f = plt.figure(figsize=(10,10))
    outf = open("tmp/dist.txt", 'w')
    #map RW coords to ids
    RW_ids={}; ID=1
    for i in RoadNW_coord:
        RW_ids[i] = ID; ID += 1
    #map substation ids
    Sub_ids={}
    for i in Sub_coord:
        Sub_ids[i] = ID; ID +=1

    for s in Paths:
        for n in Paths[s]:
            L=Paths[s][n]
      #draw line from substation to transformer
            H=nx.Graph(); H.add_edge(s, L[0])
            coord={}; coord[s] = Sub_coord[s]; coord[L[0]] = RoadNW_coord[L[0]]
            outf.write(str(Sub_ids[s]) + ' S ' + str(coord[s][0])+ ' ' + str(coord[s][1])+' '+ str(RW_ids[L[0]]) + ' T ' + str(coord[L[0]][0])+' '+str(coord[L[0]][1])+'\n')
            #draw_graph(H, coord, fname, justplot=1, shape=1, color=1)
      #draw remaining segments
            H=nx.Graph()
            for i in range(1, len(L)):
                H.add_edge(L[i-1], L[i])
                outf.write(str(RW_ids[L[i]]) + ' L ' + str(RoadNW_coord[L[i-1]][0])+ ' '+str(RoadNW_coord[L[i-1]][1])+' '+ str(RW_ids[L[i]]) + ' L ' + str(RoadNW_coord[L[i]][0])+' '+str(RoadNW_coord[L[i]][1])+'\n')
            draw_graph(H, RoadNW_coord, fname, justplot=1)
#      x = [Sub_coord[s][0], RoadNW_coord[L[0]][0]]
#      y = [Sub_coord[s][1], RoadNW_coord[L[0]][1]]
#      #plt.plot(x, y, 'ro-')
#      for i in range(1, len(L)):
#        x = [RoadNW_coord[L[i-1]][0], RoadNW_coord[L[i]][0]]
#        y = [RoadNW_coord[L[i-1]][1], RoadNW_coord[L[i]][1]]
#        #print "s=", s, "n=", n, "i=", i, "x=", x, "y=", y
#        print RoadNW_coord[L[i]]
#        plt.plot(x, y, 'b-')
    f.savefig(fname, bbox_inches='tight')
  #f.savefig("tmp/foo.pdf", bbox_inches='tight')


