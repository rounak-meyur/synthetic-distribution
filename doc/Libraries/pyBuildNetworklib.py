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
from pyMILPlib import MILP_secondary,MILP_secondary_pf,MILP_primary,MILP_primary_modified
from scipy.spatial import Voronoi,cKDTree,voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#%% Functions

def MeasureDistance(pt1,pt2):
    '''
    The format of each point is (longitude,latitude).
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']


def get_secondary_graph_edges(sec_net_file,tnodes):
    """
    """
    # Get the secondary network
    f = open(sec_net_file,'r')
    lines = f.readlines()
    f.close()
    
    # create the entire secondary network
    secondary = [(int(temp.strip('\n').split(' ')[0]),
                     int(temp.strip('\n').split(' ')[4])) \
                    for temp in lines]
    sec_net = nx.Graph()
    sec_net.add_edges_from(secondary)
    
    # get subgraph for required primary network
    nodelist = []
    for t in tnodes: nodelist.extend(list(nx.descendants(sec_net,t)))
    return list(sec_net.subgraph(nodelist+tnodes).edges())

    



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
        edgelist = MILP_secondary(graph,roots,k,hops).optimal_edges
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
        edgelist = MILP_secondary_pf(graph,roots).optimal_edges
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
    
    
        
class Primary:
    """
    Creates the primary distribuion network by solving an optimization problem.
    First the set of possible edges are identified from the links in road net-
    -work and transformer connections.
    """
    def __init__(self,subs,tsfr,roads,homes,links):
        """
        """
        self.subs = subs
        self.tsfr = tsfr
        self.roads = roads
        self.links = links
        self.homes = homes
        self.graph = self.__create_master_graph()
        self.S2Near = self._get_nearest()
        self.S2Node = self._get_nodes()
        return
    
    def __create_master_graph(self):
        """
        Creates the master graph consisting of all possible edges from the road
        network links. Each node in the graph has a spatial attribute called
        cord.
        """
        road_edges = list(self.roads.graph.edges())
        tsfr_edges = list(self.tsfr.graph.edges())
        for edge in self.links:
            try:
                road_edges.remove(edge)
            except:
                road_edges.remove((edge[1],edge[0]))
        edgelist = road_edges + tsfr_edges
        graph = nx.Graph()
        graph.add_edges_from(edgelist)
        nodelist = list(graph.nodes())
        nodepos = {n:self.roads.cord[n] if n in self.roads.cord \
                   else self.tsfr.cord[n] for n in nodelist}
        nx.set_node_attributes(graph,nodepos,name='cord')
        edge_length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                       for e in edgelist}
        nx.set_edge_attributes(graph,edge_length,name='hop')
        return graph
    
    def _get_nearest(self):
        """
        """
        # Get the Voronoi centers and data points
        centers = list(self.subs.cord.keys())
        center_pts = [self.subs.cord[s] for s in centers]
        nodes = list(self.graph.nodes())
        nodepos = nx.get_node_attributes(self.graph,'cord')
        node_pts = [nodepos[n] for n in nodes]
        
        # Find number of road nodes mapped to each substation
        voronoi_kdtree = cKDTree(center_pts)
        _, node_regions = voronoi_kdtree.query(node_pts, k=1)
        sub_map = {s:node_regions.tolist().count(s) for s in range(len(centers))}
        
        # Compute new centers of Voronoi regions
        centers = [centers[s] for s in sub_map if sub_map[s]>50]
        center_pts = [self.subs.cord[s] for s in centers]
        
        # Recompute the Voronoi regions and generate the final map
        voronoi_kdtree = cKDTree(center_pts)
        _, node_regions = voronoi_kdtree.query(node_pts, k=1)
        
        # Index the region and assign the nodes to the substation
        indS2N = {i:np.argwhere(i==node_regions)[:,0]\
                  for i in np.unique(node_regions)}
        S2Node = {centers[i]:[nodes[j] for j in indS2N[i]] for i in indS2N}
        
        # Compute nearest node to substation
        S2Near = {}
        for s in S2Node:
            nodes = [n for n in S2Node[s] if n in self.roads.cord]
            dis = [MeasureDistance(self.subs.cord[s],nodepos[n]) for n in nodes]
            S2Near[s] = nodes[dis.index(min(dis))]
        return S2Near
    
    def plot_voronoi(self,S,title="Voronoi clustering of transformer nodes"):
        """
        """
        # Plot Voronoi regions with mapped road network nodes
        vor = Voronoi([self.subs.cord[s] for s in S])
        nodelist = list(self.graph.nodes())
        nodepos = nx.get_node_attributes(self.graph,'cord')
        RW_xval = [nodepos[n][0] for n in nodelist]
        RW_yval = [nodepos[n][1] for n in nodelist]
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
    
    def plot_graph(self,graph,sublist,
                   title="Possible set of edges in primary network"):
        """
        """
        nodepos = nx.get_node_attributes(graph,'cord')
        col = ['black' if n in self.roads.cord else 'red' \
               for n in list(graph.nodes())]
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(graph,pos=nodepos,node_size=10.0,node_color=col,
                         edge_width=1.0, edge_color='black',ax=ax,
                         with_labels=False)
        subx = [self.subs.cord[s][0] for s in sublist]
        suby = [self.subs.cord[s][1] for s in sublist]
        ax.scatter(subx,suby,s=80.0,c='green')
        ax.set_xlabel("Longitude",fontsize=20)
        ax.set_ylabel("Latitude",fontsize=20)
        ax.set_title(title,fontsize=20)
        
        
        leglines = [Line2D([0], [0], color='white', markerfacecolor=c, 
                           marker='o',markersize=10) \
                    for c in ['black','red','green']]
        ax.legend(leglines,['road nodes','transformer nodes','substations'],
                  loc='lower right',ncol=1,prop={'size': 20})
        
        plt.show()
        return
    
    def _get_nodes(self):
        """
        """
        centers = list(self.S2Near.values())
        cells = nx.voronoi_cells(self.graph, centers, 'hop')
        S2Node = {s:list(cells[self.S2Near[s]]) for s in self.S2Near}
        return S2Node
    
    def get_sub_network(self,sublist,sec_file):
        """
        """
        # Get and set attributes
        nodepos = nx.get_node_attributes(self.graph,'cord')
        length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                  for e in list(self.graph.edges())}
        nx.set_edge_attributes(self.graph,length,'length')
        nodepos.update(self.homes.cord)
        nodepos.update(self.subs.cord)
        
        # Get list of nodes to be connected 
        nodelist = []
        for s in sublist: nodelist.extend(self.S2Node[s])
        G = self.graph.subgraph(nodelist)
        
        # Optimizaton problem to get the primary network
        tnodes = [n for n in nodelist if n in self.tsfr.cord]
        snodes = []
        for s in sublist: snodes.append(self.S2Near[s])
        M = MILP_primary(G,tnodes,snodes)
        primary = M.optimal_edges
        
        # Add the first edge between substation and nearest road node
        ehvlines = [(s,self.S2Near[s]) for s in sublist]
        
        # Get the edges for the secondary network
        secondary = get_secondary_graph_edges(sec_file,tnodes)
        
        # Combione both networks and update labels
        dist_net = nx.Graph()
        dist_net.add_edges_from(primary+secondary+ehvlines)
        nx.set_node_attributes(dist_net,nodepos,'cord')
        node_label = self.label_nodes(dist_net)
        nx.set_node_attributes(dist_net, node_label, 'label')
        edge_label = {}
        for e in list(dist_net.edges()):
            if e in primary or (e[1],e[0]) in primary:
                edge_label[e] = 'P'
            elif e in ehvlines or (e[1],e[0]) in ehvlines:
                edge_label[e] = 'E'
            else:
                edge_label[e] = 'S'
        nx.set_edge_attributes(dist_net, edge_label, 'label')
        return dist_net
    
    def get_total_network(self,sec_file):
        """
        """
        # Get and set attributes
        nodepos = nx.get_node_attributes(self.graph,'cord')
        length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                  for e in list(self.graph.edges())}
        nx.set_edge_attributes(self.graph,length,'length')
        nodepos.update(self.homes.cord)
        nodepos.update(self.subs.cord)
        
        # Optimizaton problem to get the primary network
        tnodes = list(self.tsfr.cord.keys())
        snodes = list(self.S2Near.values())
        M = MILP_primary(self.graph,tnodes,snodes)
        primary = M.optimal_edges
        
        # Add the first edge between substation and nearest road node
        ehvlines = [(s,self.S2Near[s]) for s in self.S2Near]
        
        # Get the edges for the secondary network
        secondary = get_secondary_graph_edges(sec_file,tnodes)
        
        # Combione both networks and update labels
        dist_net = nx.Graph()
        dist_net.add_edges_from(primary+secondary+ehvlines)
        nx.set_node_attributes(dist_net,nodepos,'cord')
        node_label = self.label_nodes(dist_net)
        nx.set_node_attributes(dist_net, node_label, 'label')
        edge_label = {}
        for e in list(dist_net.edges()):
            if e in primary or (e[1],e[0]) in primary:
                edge_label[e] = 'P'
            elif e in ehvlines or (e[1],e[0]) in ehvlines:
                edge_label[e] = 'E'
            else:
                edge_label[e] = 'S'
        nx.set_edge_attributes(dist_net, edge_label, 'label')
        return dist_net
    
    def label_nodes(self,graph):
        """
        """
        nodelist = list(graph.nodes())
        node_label = {}
        for n in nodelist:
            if n in self.roads.cord: node_label[n] = 'R'
            elif n in self.tsfr.cord: node_label[n] = 'T'
            elif n in self.homes.cord: node_label[n] = 'H'
            elif n in self.subs.cord: node_label[n] = 'S'
            else: node_label[n] = 'UNKNOWN'
        return node_label
    
    def get_sub_network_modified(self,sublist,sec_file):
        """
        """
        # Get and set attributes
        nodepos = nx.get_node_attributes(self.graph,'cord')
        length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                  for e in list(self.graph.edges())}
        nx.set_edge_attributes(self.graph,length,'length')
        nodepos.update(self.homes.cord)
        nodepos.update(self.subs.cord)
        
        # Get list of nodes to be connected 
        nodelist = []
        for s in sublist: nodelist.extend(self.S2Node[s])
        
        # Get the distance from the nearest substation
        hvdist = {r:min([MeasureDistance(self.subs.cord[s],self.roads.cord[r]) \
                for s in sublist]) if r in self.roads.cord else 0.0 for r in nodelist}
        
        G = self.graph.subgraph(nodelist)
        nx.set_node_attributes(G,hvdist,'distance')
        
        # Optimizaton problem to get the primary network
        tnodes = {n:self.tsfr.load[n] for n in nodelist if n in self.tsfr.cord}
        M = MILP_primary_modified(G,tnodes)
        primary = M.optimal_edges
        roots = M.roots
        
        # Add the first edge between substation and nearest road node
        ehvlines = []
        for r in roots:
            distance = [MeasureDistance(self.subs.cord[s],self.roads.cord[r]) \
                        for s in sublist]
            ind_dist = distance.index(min(distance))
            ehvlines.append((sublist[ind_dist],r))
        
        # Get the edges for the secondary network
        secondary = get_secondary_graph_edges(sec_file,list(tnodes.keys()))
        
        # Combione both networks and update labels
        dist_net = nx.Graph()
        dist_net.add_edges_from(ehvlines+primary+secondary)
        nx.set_node_attributes(dist_net,nodepos,'cord')
        node_label = self.label_nodes(dist_net)
        nx.set_node_attributes(dist_net, node_label, 'label')
        edge_label = {}
        for e in list(dist_net.edges()):
            if e in primary or (e[1],e[0]) in primary:
                edge_label[e] = 'P'
            elif e in ehvlines or (e[1],e[0]) in ehvlines:
                edge_label[e] = 'E'
            else:
                edge_label[e] = 'S'
        nx.set_edge_attributes(dist_net, edge_label, 'label')
        return dist_net

    



        
        