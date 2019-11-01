# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
from geographiclib.geodesic import Geodesic
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
from shapely.geometry import LineString,MultiPoint
from pyMILPlib import MILP_noPF,MILP_withPF

#%% Functions

def MeasureDistance(pt1,pt2):
    '''
    The format of each point is (longitude,latitude).
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']




    



#%% Classes
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
        # Copute great circle distance
        lon1,lon2 = self.xy[0]
        lat1,lat2 = self.xy[1]
        geod = Geodesic.WGS84
        return geod.Inverse(lat1, lon1, lat2, lon2)['s12']
    
    
    def InterpolatePoints(self,min_sep=50):
        """
        """
        points = []
        length = self.geod_length
        sep = max(min_sep,(length/15))
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(xy)
        return {i:[pt.x,pt.y] for i,pt in enumerate(MultiPoint(points))}



class Spider:
    """
    """
    def __init__(self,homes,roads,home_to_link):
        """
        """
        self.home_load = homes.average
        self.home_cord = homes.cord
        self.road_cord = roads.cord
        self.link_to_home = {}
        for h in home_to_link:
            if home_to_link[h] in self.link_to_home.keys():
                self.link_to_home[home_to_link[h]].append(h)
            else:
                self.link_to_home[home_to_link[h]]=[h]
        return
    
    
    def __separate_side(self,link):
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
    
    
    def __complete_graph_from_list(self,L):
        """
        """
        G = nx.Graph()
        edges = combinations(L,2)
        G.add_edges_from(edges)
        return G
    
    def __delaunay_graph_from_list(self,L):
        """
        """
        points = np.array([[self.home_cord[h][0],
                            self.home_cord[h][1]] for h in L])
        triangles = Delaunay(points).simplices
        edgelist = []
        for t in triangles:
            edges = [(L[t[0]],L[t[1]]),(L[t[1]],L[t[2]]),(L[t[2]],L[t[0]])]
            edgelist.extend(edges)
        G = nx.Graph()
        G.add_edges_from(edgelist)
        return G
    
    def get_nodes(self,link,minsep):
        """
        """
        home_pts = self.link_to_home[link]
        link_line = Link(LineString([tuple(self.road_cord[n]) for n in link]))
        transformers = link_line.InterpolatePoints(minsep)
        return home_pts,transformers
    
    
    def create_dummy_graph(self,link,minsep,penalty):
        """
        """
        sides = self.__separate_side(link)
        home_pts = self.link_to_home[link]
        node_pos = {h:self.home_cord[h] for h in home_pts}
        load = {h:self.home_load[h]/1000.0 for h in home_pts}
                
        # Update the attributes of nodes
        link_line = Link(LineString([tuple(self.road_cord[n]) for n in link]))
        transformers = link_line.InterpolatePoints(minsep)
        node_pos.update(transformers)
        sides.update({t:0 for t in transformers})
        load.update({t:1.0 for t in transformers})
        
        if len(home_pts)>10:
            graph = self.__delaunay_graph_from_list(home_pts)
        else:
            graph = self.__complete_graph_from_list(home_pts)
        new_edges = [(t,n) for t in transformers for n in home_pts]
        graph.add_edges_from(new_edges)
        nx.set_node_attributes(graph,node_pos,'cord')
        nx.set_node_attributes(graph,load,'load')
        edge_cost = {e:MeasureDistance(node_pos[e[0]],node_pos[e[1]])*\
                     (1+penalty*abs(sides[e[0]]-sides[e[1]])) \
                      for e in list(graph.edges())}
        edge_length = {e:MeasureDistance(node_pos[e[0]],node_pos[e[1]])\
                      for e in list(graph.edges())}
        nx.set_edge_attributes(graph,edge_length,'length')
        nx.set_edge_attributes(graph,edge_cost,'cost')
        return graph,transformers
    
    def generate_optimal_topology(self,link,minsep=50,penalty=0.5,k=2,hops=4):
        """
        """
        graph,roots = self.create_dummy_graph(link,minsep,penalty)
        edgelist = MILP_noPF(graph,roots,k,hops).optimal_edges
        forest = nx.Graph()
        forest.add_edges_from(edgelist)
        node_cord = {node: roots[node] if node in roots\
                     else self.home_cord[node]\
                     for node in list(forest.nodes())}
        nx.set_node_attributes(forest,node_cord,'cord')
        node_load = {node:sum([self.home_load[h] for h in list(nx.descendants(forest,node))]) \
                     if node in roots else self.home_load[node] for node in list(forest.nodes())}
        nx.set_node_attributes(forest,node_load,'load')
        return forest,roots
    
    def generate_optimalpf_topology(self,link,minsep=50,penalty=0.5):
        """
        """
        graph,roots = self.create_dummy_graph(link,minsep,penalty)
        edgelist = MILP_withPF(graph,roots).optimal_edges
        forest = nx.Graph()
        forest.add_edges_from(edgelist)
        node_cord = {node: roots[node] if node in roots\
                     else self.home_cord[node]\
                     for node in list(forest.nodes())}
        nx.set_node_attributes(forest,node_cord,'cord')
        return forest,roots
    
    def checkpf(self,forest,roots,r=0.81508/57.6):
        """
        """
        A = nx.incidence_matrix(forest,nodelist=list(forest.nodes()),
                                edgelist=list(forest.edges()),oriented=True).toarray()
        node_pos = nx.get_node_attributes(forest,'cords')
        R = [1.0/(MeasureDistance(node_pos[e[0]],node_pos[e[1]])*0.001*r) \
             for e in list(forest.edges())]
        D = np.diag(R)
        home_ind = [i for i,node in enumerate(forest.nodes()) \
                    if node not in roots]
        homelist = [node for node in list(forest.nodes()) if node not in roots]
        G = np.matmul(np.matmul(A,D),A.T)[home_ind,:][:,home_ind]
        p = np.array([self.home_load[h]*0.001 for h in homelist])
        v = np.matmul(np.linalg.inv(G),p)
        voltage = {h:1.0-v[i] for i,h in enumerate(homelist)}
        return voltage
    
    
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
