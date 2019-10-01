# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""

from math import sin, cos, sqrt, atan2, radians
from collections import namedtuple as nt
import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree as st_tree
from itertools import combinations

#%% Functions
def MeasureDistance(Point1,Point2):
    '''
    The format of each point is (longitude,latitude).
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
    return distance*1000


#%% Classes
class Steiner:
    """
    """
    def __init__(self,homes,roads,home_to_link):
        """
        """
        self.home_cord = homes.cord
        self.road_cord = roads.cord
        self.link_to_home = {}
        self.home_load = homes.average
        self.road_to_home = {k:0.0 for k in list(roads.graph.nodes())}
        for h in home_to_link:
            if home_to_link[h] in self.link_to_home.keys():
                self.link_to_home[home_to_link[h]].append(h)
            else:
                self.link_to_home[home_to_link[h]]=[h]
        return
    
    
    def separate_side(self,link):
        """
        Evaluates the groups of homes on either side of the link
        """
        homelist = self.link_to_home[link] if link in self.link_to_home\
            else self.link_to_home[(link[1],link[0])]
        points = [self.home_cord[h] for h in homelist]
        (x1,y1) = self.road_cord[link[0]]
        (x2,y2) = self.road_cord[link[1]]
        eqn = [((x-x1)*(y2-y1))-((y-y1)*(x2-x1)) for (x,y) in points]
        side = {}
        for index,home in enumerate(homelist):
            if eqn[index]>=0: side[home]=1
            else: side[home]=-1
        return side
    
    
    def separate_node(self,link):
        """
        Evaluates the groups of homes based on nearness to link ends 
        """
        (x1,y1) = self.road_cord[link[0]]
        (x2,y2) = self.road_cord[link[1]]
        node1 = []; node2 = []
        homelist = self.link_to_home[link] if link in self.link_to_home\
            else self.link_to_home[(link[1],link[0])]
        for home in homelist:
            if MeasureDistance(self.road_cord[link[0]],self.home_cord[home])<=\
                               MeasureDistance(self.road_cord[link[1]],self.home_cord[home]):
                node1.append(home)
            else:
                node2.append(home)
        return [node1,node2]
    
    
    def complete_graph_from_list(self,L,create_using=None):
        """
        """
        G = nx.Graph()
        edges = combinations(L,2)
        G.add_edges_from(edges)
        return G
    
    
    def create_dummy_graph(self,link,group,penalty=0.5):
        """
        """
        node_maps = self.separate_node(link)
        sides = self.separate_side(link)
        root = link[group]
        home_pts = node_maps[group]
        if home_pts == []:
            g = nx.Graph()
            g.add_node(root)
            node_pos = {root:self.road_cord[root]}
            nx.set_node_attributes(g,node_pos,'cord')
            return g
        else:
            node_pos = {h:self.home_cord[h] for h in home_pts}
            self.road_to_home[root] += sum([self.home_load[h] \
                                           for h in home_pts])
            if root not in home_pts:
                node_pos[root] = self.road_cord[root]
                sides[root] = 0
            else:
                print("Road node ID and Home ID matches!!! PROBLEM")
            graph = self.complete_graph_from_list(home_pts)
            new_edges = [(root,n) for n in home_pts]
            graph.add_edges_from(new_edges)
            nx.set_node_attributes(graph,node_pos,'cord')
            edge_dist = {e:MeasureDistance(node_pos[e[0]],node_pos[e[1]])*\
                         (1+penalty*abs(sides[e[0]]*sides[e[1]])*\
                          abs(sides[e[0]]-sides[e[1]])) \
                          for e in list(graph.edges())}
            nx.set_edge_attributes(graph,edge_dist,'dist')
            steiner = st_tree(graph,list(graph.nodes()),weight='dist')
            return steiner



class Spider:
    """
    """
    def __init__(self,homes,tsfrs,roads,home_to_tsfr):
        """
        """
        self.home_cord = homes.cord
        self.home_load = homes.average
        self.tsfr_cord = tsfrs.cord
        self.road_cord = roads.cord
        self.tsfr_link = tsfrs.link
        self.tsfr_to_home = {}
        self.tsfr_rating = {k:0.0 for k in tsfrs.cord}
        for h in home_to_tsfr:
            if home_to_tsfr[h] in self.tsfr_to_home.keys():
                self.tsfr_to_home[home_to_tsfr[h]].append(h)
            else:
                self.tsfr_to_home[home_to_tsfr[h]]=[h]
        return
    
    
    def separate_side(self,link,homelist):
        """
        Evaluates the groups of homes on either side of the link
        """
        points = [self.home_cord[h] for h in homelist]
        (x1,y1) = self.road_cord[link[0]]
        (x2,y2) = self.road_cord[link[1]]
        eqn = [((x-x1)*(y2-y1))-((y-y1)*(x2-x1)) for (x,y) in points]
        side = {}
        for index,home in enumerate(homelist):
            if eqn[index]>=0: side[home]=1
            else: side[home]=-1
        return side
    
    
    def complete_graph_from_list(self,L,create_using=None):
        """
        """
        G = nx.Graph()
        edges = combinations(L,2)
        G.add_edges_from(edges)
        return G
    
    
    def generate_spider(self,tsfr,penalty=0.5):
        """
        """
        homes_mapped = self.tsfr_to_home[tsfr]
        link = self.tsfr_link[tsfr]
        sides = self.separate_side(link,homes_mapped)
        node_pos = {h:self.home_cord[h] for h in homes_mapped}
        self.tsfr_rating[tsfr] = sum([self.home_load[h] for h in homes_mapped])
        if tsfr not in homes_mapped:
            node_pos[tsfr] = self.tsfr_cord[tsfr]
            sides[tsfr] = 0
        else:
            print("Tsfr node ID and Home ID matches!!! PROBLEM")
        graph = self.complete_graph_from_list(homes_mapped)
        new_edges = [(tsfr,n) for n in homes_mapped]
        graph.add_edges_from(new_edges)
        nx.set_node_attributes(graph,node_pos,'cord')
        edge_dist = {e:MeasureDistance(node_pos[e[0]],node_pos[e[1]])*\
                     (1+penalty*abs(sides[e[0]]*sides[e[1]])*\
                      abs(sides[e[0]]-sides[e[1]])) \
                      for e in list(graph.edges())}
        nx.set_edge_attributes(graph,edge_dist,'dist')
        spider = st_tree(graph,list(graph.nodes()),weight='dist')
        return spider
    
