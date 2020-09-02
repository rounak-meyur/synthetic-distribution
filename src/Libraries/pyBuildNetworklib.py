# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
import sys
from geographiclib.geodesic import Geodesic
import networkx as nx
from matplotlib import cm
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
from shapely.geometry import LineString,MultiPoint,LinearRing,Point
from pyMILPlib import MILP_secondary,MILP_primary
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyqtree import Index

#%% Functions

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

def InvertMap(input_dict):
    """
    Inverts a given mapping. The input/output mapping is provided through a dictionary.
    input_dict: input mapping
    output_dict: output mapping
    """
    output_dict = {}
    for key in input_dict:
        if input_dict[key] in list(output_dict.keys()):
            output_dict[input_dict[key]].append(key)
        else:
            output_dict[input_dict[key]]=[key]
    return output_dict

def read_primary(filename):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        filename: name of the .txt file
        
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
    # Open file and readlines
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    
    # Create the list/dictionary of node/edge labels and attributes
    edges = []
    edgelabel = {}
    edge_r = {}
    edge_x = {}
    nodelabel = {}
    nodepos = {}
    for line in lines:
        data = line.strip('\n').split(' ')
        edges.append((int(data[0]),int(data[4])))
        nodepos[int(data[0])] = [float(data[2]),float(data[3])]
        nodepos[int(data[4])] = [float(data[6]),float(data[7])]
        nodelabel[int(data[0])] = data[1]
        nodelabel[int(data[4])] = data[5]
        edgelabel[(int(data[0]),int(data[4]))] = data[-1]
        edge_r[(int(data[0]),int(data[4]))] = float(data[-3])
        edge_x[(int(data[0]),int(data[4]))] = float(data[-2])
    
    # Create the networkx graph
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph,edgelabel,'label')
    nx.set_edge_attributes(graph,edge_r,'r')
    nx.set_edge_attributes(graph,edge_x,'x')
    nx.set_node_attributes(graph,nodelabel,'label')
    nx.set_node_attributes(graph,nodepos,'cord')
    return graph

def bounds(pt,radius):
    return (pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)

def find_nearest_node(center_cord,node_cord):
    xmin,ymin = np.min(np.array(list(node_cord.values())),axis=0)
    xmax,ymax = np.max(np.array(list(node_cord.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    nodes = []
    for i,n in enumerate(list(node_cord.keys())):
        node_geom = Point(node_cord[n])
        node_bound = bounds(node_geom,0.0)
        idx.insert(i,node_bound)
        nodes.append((node_geom, node_bound, n))
    
    pt_center = Point(center_cord)
    center_bd = bounds(pt_center,0.1)
    matches = idx.intersect(center_bd)
    closest_node = min(matches,key=lambda i: nodes[i][0].distance(pt_center))
    return nodes[closest_node][-1]
    

def create_master_graph(roads,tsfr,links):
    """
    Creates the master graph consisting of all possible edges from the road
    network links. Each node in the graph has a spatial attribute called
    cord.
    """
    road_edges = list(roads.graph.edges())
    tsfr_edges = list(tsfr.graph.edges())
    for edge in links:
        if edge in road_edges:
            road_edges.remove(edge)
        elif (edge[1],edge[0]) in road_edges:
            road_edges.remove((edge[1],edge[0]))
    edgelist = road_edges + tsfr_edges
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    nodelist = list(graph.nodes())
    
    # Coordinates of nodes in network
    nodepos = {n:roads.cord[n] if n in roads.cord else tsfr.cord[n] for n in nodelist}
    nx.set_node_attributes(graph,nodepos,name='cord')
    
    # Length of edges
    edge_length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                    for e in edgelist}
    nx.set_edge_attributes(graph,edge_length,name='length')
    
    # Label the nodes in network
    node_label = {n:'T' if n in tsfr.cord else 'R' for n in list(graph.nodes())}
    nx.set_node_attributes(graph,node_label,'label')
    
    # Add load at local transformer nodes
    node_load = {n:tsfr.load[n] if node_label[n]=='T' else 0.0 \
                 for n in list(graph.nodes())}
    nx.set_node_attributes(graph,node_load,'load')
    return graph

def get_substation_areas(subs,roads,graph):
    """
    Get list of nodes mapped in the Voronoi cell of the substation. The Voronoi 
    cell is determined on the basis of shortest path distance from each node to
    the nearest node to the substation.
    Returns: dictionary of substations with list of nodes mapped to it as values
    """
    # Get the Voronoi centers and data points
    centers = list(subs.cord.keys())
    center_pts = [subs.cord[s] for s in centers]
    nodes = list(graph.nodes())
    nodepos = nx.get_node_attributes(graph,'cord')
    node_pts = [nodepos[n] for n in nodes]
    
    # Find number of road nodes mapped to each substation
    voronoi_kdtree = cKDTree(center_pts)
    _, node_regions = voronoi_kdtree.query(node_pts, k=1)
    sub_map = {s:node_regions.tolist().count(s) for s in range(len(centers))}
    
    # Compute new centers of Voronoi regions
    centers = [centers[s] for s in sub_map if sub_map[s]>50]
    center_pts = [subs.cord[s] for s in centers]
    
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
        nodes = [n for n in S2Node[s] if n in roads.cord]
        nodecord = {n: nodepos[n] for n in nodes}
        S2Near[s] = find_nearest_node(subs.cord[s],nodecord)
    
    # Compute Voronoi cells with network distance 
    centers = list(S2Near.values())
    cells = nx.voronoi_cells(graph, centers, 'length')
    
    # Recompute Voronoi cells for larger primary networks
    centers = [c for c in centers if len(cells[c])>100]
    cells = nx.voronoi_cells(graph, centers, 'length')
    
    # Recompute S2Near and S2Node
    S2Near = {s:S2Near[s] for s in S2Near if S2Near[s] in centers}
    S2Node = {s:list(cells[S2Near[s]]) for s in S2Near}
    return S2Node

def Initialize_Primary(subs,roads,tsfr,links):
    """
    Initialization function to generate master graph for primary network generation
    and node to substation mapping. The function is called before calling the class
    object to optimize the primary network.
    
    Inputs: 
        subs: named tuple for substations
        roads: named tuple for road network
        tsfr: named tuple for local transformers
        links: list of road links along which transformers are placed.
    Returns:
        graph: master graph from which the optimal network would be generated.
        S2Node: mapping between substation and road/transformer nodes based on shortest
        path distance in the master network.
    """
    graph = create_master_graph(roads, tsfr, links)
    S2Node = get_substation_areas(subs,roads,graph)
    return graph,S2Node

def read_master_graph(path,sub):
    with open(path+str(sub)+'-master.txt') as f:
        lines = f.readlines()
    edgelist = []
    nodepos = {}
    nodeload = {}
    nodedist = {}
    nodelabel = {}
    for line in lines:
        l = line.strip('\n').split('\t')
        edgelist.append((int(l[0]),int(l[6])))
        nodelabel[int(l[0])] = l[1]
        nodeload[int(l[0])] = float(l[4])
        nodedist[int(l[0])] = float(l[5])
        nodepos[int(l[0])] = [float(l[2]),float(l[3])]
        nodepos[int(l[6])] = [float(l[8]),float(l[9])]
        nodelabel[int(l[6])] = l[7]
        nodeload[int(l[6])] = float(l[10])
        nodedist[int(l[6])] = float(l[11])
    
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    nx.set_node_attributes(graph,nodelabel,'label')
    nx.set_node_attributes(graph,nodeload,'load')
    nx.set_node_attributes(graph,nodedist,'distance')
    nx.set_node_attributes(graph,nodepos,'cord')
    return graph

def plot_graph(graph,subdata=None,path=None,filename=None,rcol = ['green']):
    """
    Plots the graph for representing the possible set of edges for generating the
    primary network.
    Inputs: 
        graph: graph of road network edges with transformers.
        subdata: color of road in each graph
    Returns: displays a figure with all the possible edges.
    """
    nodepos = nx.get_node_attributes(graph,'cord')
    nodelabel = nx.get_node_attributes(graph,'label')
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    
    # Plot individual subgraphs
    for i,nlist in enumerate(list(nx.connected_components(graph))):
        sub_graph = nx.subgraph(graph,list(nlist))
        col = ['black' if nodelabel[n]=='R' else rcol[i] for n in list(nlist)]
        nx.draw_networkx(sub_graph,pos=nodepos,nodelist = list(nlist),node_size=10.0,
                         node_color=col,width=0.5, edge_color='black',ax=ax,style='dashed',
                         with_labels=False)
    
    # Scatter plot the substations
    if subdata != None:
        subx = subdata.cord[0]
        suby = subdata.cord[1]
        ax.scatter(subx,suby,s=500.0,marker='*',c='green')
    
    # Titles and labels for plot
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    # ax.set_xlabel("Longitude",fontsize=20)
    # ax.set_ylabel("Latitude",fontsize=20)
    ax.set_title("Possible set of edges in primary network",fontsize=30)
    
    # Define legends for the plot
    leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0,linestyle='dashed'),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='*',markersize=10)]+\
                [Line2D([0], [0], color='white', markerfacecolor=c, marker='o',markersize=10) for c in rcol]
    ax.legend(leglines,['road network','substations']+\
              ['Partition '+str(i+1) for i in range(len(rcol))],
              loc='best',ncol=5,prop={'size': 10})
    plt.show()
    if path != None:
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return


def combine_primary_secondary(primnet,master_secnet):
    """
    Create a network with the primary network and part of the associated secondary
    network. The roots of the primary network are connected to the substation 
    through high voltage lines.
    """
    
    # Get associated secondary network
    nodelabel = nx.get_node_attributes(primnet,'label')
    edge_label = nx.get_edge_attributes(primnet,'label')
    edge_r = nx.get_edge_attributes(primnet,'r')
    edge_x = nx.get_edge_attributes(primnet,'x')
    roots = [n for n in list(primnet.nodes()) if nodelabel[n]=='T']
    secnodes = []
    for t in roots: secnodes.extend(list(nx.descendants(master_secnet,t)))
    
    # Update secondary network data
    secnet = master_secnet.subgraph(secnodes+roots)
    dist_net = nx.Graph()
    dist_net = nx.compose(dist_net,primnet)
    dist_net = nx.compose(dist_net,secnet)
    
    # Label edges of the created network
    nodepos = nx.get_node_attributes(dist_net,'cord')
    sec_edge_label = {}
    sec_edge_r = {}
    sec_edge_x = {}
    sec_edges = [e for e in list(secnet.edges())]
    for e in sec_edges:
        length = 1e-3 * MeasureDistance(nodepos[e[0]],nodepos[e[1]])
        length = length if length != 0.0 else 1e-12
        sec_edge_label[e] = 'S'
        sec_edge_r[e] = 0.81508/57.6 * length
        sec_edge_x[e] = 0.3496/57.6 * length
    
    # Update network attributes
    edge_label.update(sec_edge_label)
    edge_r.update(sec_edge_r)
    edge_x.update(sec_edge_x)
    nx.set_edge_attributes(dist_net, edge_label, 'label')
    nx.set_edge_attributes(dist_net, edge_r, 'r')
    nx.set_edge_attributes(dist_net, edge_x, 'x')
    return dist_net


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
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length
    
    
    def InterpolatePoints(self,min_sep=50):
        """
        """
        points = []
        length = self.geod_length
        sep = max(min_sep,(length/25))
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(xy)
        if len(points)==0: 
            points.append(Point((self.xy[0][0],self.xy[1][0])))
        return {i:[pt.x,pt.y] for i,pt in enumerate(MultiPoint(points))}



class Spider:
    """
    Contains methods and attributes to generate the secondary distribution network
    originating from a link. The link consists of multiple transformers and uses 
    multiple engineering and economic heuristics to generate the network.
    """
    def __init__(self,homes,roads,home_to_link):
        """
        Initializes the class object with all home nodes, road network and the mapping
        between them.
        
        Input:  homes: named tuple with all residential building data
                roads: named tuple with all road network information
                home_to_link: mapping between homes and road links
        """
        self.home_load = homes.average
        self.home_cord = homes.cord
        self.road_cord = roads.cord
        self.links = roads.links
        self.link_to_home = InvertMap(home_to_link)
        return
    
    
    def separate_side(self,link):
        """
        Evaluates the groups of homes on either side of the link. This would help in 
        creating network with minimum crossover of distribution lines over the road
        network.
        
        Input: link: the road link of interest
        Output: side: dictionary of homes as keys and value as 1 or -1 depending on
                which side of the link it is present.
        """
        homelist = self.link_to_home[link] if link in self.link_to_home\
            else self.link_to_home[(link[1],link[0])]
        line = list(self.links[link]['geometry'].coords) if link in self.links \
            else list(self.links[(link[1],link[0])]['geometry'].coords)
        side = {h: LinearRing(line+[tuple(self.home_cord[h]),line[0]]).is_ccw \
                for h in homelist}
        return {h:1 if side[h]==True else -1 for h in homelist}
    
    
    def __complete_graph_from_list(self,L):
        """
        Computes the full graph of the nodes in list L. There would be L(L-1)/2 edges 
        in the network. This is used as base network when the number of nodes mapped 
        to the link is small.
        
        Input: L: list of nodes mapped to the link of interest
        Output: graph: the full graph which would be used as base network for the 
                optimization problem.
        """
        G = nx.Graph()
        edges = combinations(L,2)
        G.add_edges_from(edges)
        return G
    
    def __delaunay_graph_from_list(self,L):
        """
        Computes the Delaunay graph of the nodes in list L. L edges in the network 
        based on the definition of Delaunay triangulation. This is used as base 
        network when the number of nodes mapped to the link is small.
        
        Input: L: list of nodes mapped to the link of interest
        Output: graph: the Delaunay graph which would be used as base network for the 
                optimization problem.
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
    
    def get_nodes(self,link,minsep=50,followroad=False):
        """
        Gets all the nodes from the dummy graph created before solving the optimization 
        problem.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                followroad: default is False. follows only road terminals
                            if True it follows the exact road.
        Outputs:home_pts: list of residential points
                transformers: list of points along link which are probable locations 
                of transformers.
        """
        home_pts = self.link_to_home[link]
        if followroad:
            link_line = Link(self.links[link]['geometry']) if link in self.links \
                else Link(self.links[(link[1],link[0])]['geometry'])
        else:
            link_line = Link(LineString([tuple(self.road_cord[n]) for n in link]))
        transformers = link_line.InterpolatePoints(minsep)
        return home_pts,transformers
    
    def get_nearpts_tsfr(self,transformers,homelist,heuristic):
        """
        """
        edgelist = []
        for t in transformers:
            distlist = [MeasureDistance(transformers[t],self.home_cord[h]) \
                        for h in homelist]
            imphomes = np.array(homelist)[np.argsort(distlist)[:heuristic]]
            edgelist.extend([(t,n) for n in imphomes])
        return edgelist
    
    
    def create_dummy_graph(self,link,minsep,penalty,followroad=False,heuristic=None):
        """
        Creates the base network to carry out the optimization problem. The base graph
        may be a Delaunay graph or a full graph depending on the size of the problem.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
                followroad: default is False. follows only road terminals
                            if True it follows the exact road.
                heuristic: join transformers to nearest few nodes given by heuristic.
                        Used to create the dummy graph
        Outputs:graph: the generated base graph also called the dummy graph
                transformers: list of points along link which are probable locations 
                of transformers.
        """
        sides = self.separate_side(link)
        home_pts,transformers = self.get_nodes(link,minsep=minsep,
                                               followroad=followroad)
        node_pos = {h:self.home_cord[h] for h in home_pts}
        load = {h:self.home_load[h]/1000.0 for h in home_pts}
                
        # Update the attributes of nodes
        node_pos.update(transformers)
        sides.update({t:0 for t in transformers})
        load.update({t:1.0 for t in transformers})
        
        # Create the base graph
        if len(home_pts)>10:
            graph = self.__delaunay_graph_from_list(home_pts)
        else:
            graph = self.__complete_graph_from_list(home_pts)
        
        if heuristic != None:
            new_edges = self.get_nearpts_tsfr(transformers,home_pts,heuristic)
        else:
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
    
    def generate_optimal_topology(self,link,minsep=50,penalty=0.5,followroad=False,
                                  heuristic=None,hops=4,tsfr_max=25,path=None):
        """
        Calls the MILP problem and solves it using gurobi solver.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
                followroad: default is False. follows only road terminals
                            if True it follows the exact road.
                heuristic: join transformers to nearest few nodes given by heuristic.
                          Used to create the dummy graph
        Outputs:forest: the generated forest graph which is the secondary network
                roots: list of points along link which are actual locations of 
                transformers.
        """
        graph,roots = self.create_dummy_graph(link,minsep,penalty,
                                              followroad=followroad,heuristic=heuristic)
        edgelist = MILP_secondary(graph,roots,max_hop=hops,tsfr_max=tsfr_max,
                                  grbpath=path).optimal_edges
        forest = nx.Graph()
        forest.add_edges_from(edgelist)
        node_cord = {node: roots[node] if node in roots\
                     else self.home_cord[node]\
                     for node in list(forest.nodes())}
        nx.set_node_attributes(forest,node_cord,'cord')
        node_load = {node:sum([self.home_load[h] for h in list(nx.descendants(forest,node))]) \
                     if node in roots else self.home_load[node] \
                         for node in list(forest.nodes())}
        nx.set_node_attributes(forest,node_load,'load')
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
    def __init__(self,subdata,path,max_node=900):
        """
        """
        self.subdata = subdata
        self.graph = nx.Graph()
        self.__get_subgraph(path,max_node)
        return
    
    def __get_subgraph(self,path,max_node):
        """
        Get the subgraph of master graph from the nodes mapped to the Voronoi region
        of each substation. Thereafter, it calls the partioning function to create the
        partition of faster computation.
        Inputs:
            path: path where the master graph is stored.
        """
        master_graph = read_master_graph(path,self.subdata.id)
        # Get and set attributes
        nodepos = nx.get_node_attributes(master_graph,'cord')
        length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                  for e in list(master_graph.edges())}
        nx.set_edge_attributes(master_graph,length,'length')
        
        # Create partitions
        print("Partitioning...")
        self.__get_partitions(master_graph,max_node)
        return
    
    
    def __get_partitions(self,graph_list,max_node):
        """
        This function handles primary network creation for large number of nodes.
        It divides the network into multiple partitions of small networks and solves
        the optimization problem for each sub-network.
        """
        if type(graph_list) == nx.Graph: graph_list = [graph_list]
        for g in graph_list:
            if len(g) < max_node:
                self.graph = nx.compose(self.graph,g)
            else:
                comp = nx.algorithms.community.girvan_newman(g)
                nodelist = list(sorted(c) for c in next(comp))
                sglist = [nx.subgraph(g,nlist) for nlist in nodelist]
                print("Graph of ",len(g)," is partioned to",
                      [len(sg) for sg in sglist])
                self.__get_partitions(sglist,max_node)
        return
    
    
    def get_secondary(self,filename,roots):
        """
        Get the list of edges representing the secondary network rooted at the list 
        of nodes provided in the list.
        Inputs:
            filename: .txt file which has list of edges in secondary network.
            roots: list of root nodes for which the secondary network is required.
        Returns: list of edges for the required secondary network
        """
        # Get the secondary network
        f = open(filename,'r')
        lines = f.readlines()
        f.close()
        
        # create the entire secondary network
        secondary = [(int(temp.strip('\n').split('\t')[0]),
                         int(temp.strip('\n').split('\t')[4])) \
                        for temp in lines]
        sec_net = nx.Graph()
        sec_net.add_edges_from(secondary)
        
        # get subgraph for required primary network
        nodelist = []
        for t in roots: nodelist.extend(list(nx.descendants(sec_net,t)))
        return list(sec_net.subgraph(nodelist+roots).edges())
        
    
    def optimal_network(self,fmax=400,feedmax=10,grbpath=None):
        """
        Solve the MILP optimization problem to obtain the primary distribution with
        multiple feeders.
        """
        primary = []; roots = []; tnodes = []
        for nlist in list(nx.connected_components(self.graph)):
            nodelist = list(nlist)
            subgraph = nx.subgraph(self.graph,nodelist)
            agg_load = nx.get_node_attributes(subgraph,'load')
            nlabel = nx.get_node_attributes(subgraph,'label')
            dict_tsfr = {n:agg_load[n] for n in nodelist if nlabel[n]=='T'}
            # adjust feeder count limit to avoid infeasibility 
            net_load = sum(list(dict_tsfr.values()))/1000.0
            if feedmax < int(net_load/fmax)+2:
                feedmax = int(net_load/fmax)+2
                print("Maximum feeder limit changed to:",feedmax)
            M = MILP_primary(subgraph,dict_tsfr,
                             flow=fmax,feeder=feedmax,grbpath=grbpath)
            print("#####################Optimization Ended#####################")
            print("\n\n")
            primary += M.optimal_edges
            roots += M.roots
            tnodes += list(dict_tsfr.keys())
        return primary,roots,tnodes
        
    
    def get_sub_network(self,flowmax=400,feedermax=10,
                        grbpath=None):
        """
        """
        # Optimizaton problem to get the primary network
        primary,roots,tnodes = self.optimal_network(fmax=flowmax,
                                                    feedmax=feedermax,
                                                    grbpath=grbpath)            
        
        # Add the first edge between substation and nearest road node
        hvlines = [(self.subdata.id,r) for r in roots]
        
        # Create the network with data as attributes
        dist_net = self.create_network(primary,hvlines)
        return dist_net
    
    
    def get_partition_network(self,nodelist,flowmax=400,feedermax=10,
                              grbpath=None):
        """
        """
        subgraph = nx.subgraph(self.graph,nodelist)
        agg_load = nx.get_node_attributes(subgraph,'load')
        nlabel = nx.get_node_attributes(subgraph,'label')
        dict_tsfr = {n:agg_load[n] for n in nodelist if nlabel[n]=='T'}
        # adjust feeder count limit to avoid infeasibility 
        net_load = sum(list(dict_tsfr.values()))/1000.0
        if feedermax < int(net_load/flowmax)+2:
            feedermax = int(net_load/flowmax)+2
            print("Maximum feeder limit changed to:",feedermax)
        M = MILP_primary(subgraph,dict_tsfr,
                         flow=flowmax,feeder=feedermax,grbpath=grbpath)
        print("#####################Optimization Ended#####################")
        print("\n\n")
        
        # Get the primary, secondary and ehv line edges
        primary = M.optimal_edges
        hvlines = [(self.subdata.id,r) for r in M.roots]
        dist_net = self.create_network(primary,hvlines)
        return dist_net
    
    
    def create_network(self,primary,hvlines):
        """
        Create a network with the primary network and part of the associated secondary
        network. The roots of the primary network are connected to the substation 
        through high voltage lines.
        Input: 
            primary: list of edges forming the primary network
            hvlines: list of edges joining the roots of the primary network with the
            substation(s).
        """
        # Combine both networks and update labels
        dist_net = nx.Graph()
        dist_net.add_edges_from(hvlines+primary)
        
        # Add coordinate attributes to the nodes of the network.
        nodepos = nx.get_node_attributes(self.graph,'cord')
        nodepos[self.subdata.id] = self.subdata.cord
        nx.set_node_attributes(dist_net,nodepos,'cord')
        
        # Label nodes of the created network
        node_label = nx.get_node_attributes(self.graph,'label')
        node_label[self.subdata.id] = 'S'
        nx.set_node_attributes(dist_net, node_label, 'label')
        
        # Label edges of the created network
        edge_label = {}
        edge_r = {}
        edge_x = {}
        for e in list(dist_net.edges()):
            length = 1e-3 * MeasureDistance(nodepos[e[0]],nodepos[e[1]])
            length = length if length != 0.0 else 1e-12
            if e in primary or (e[1],e[0]) in primary:
                edge_label[e] = 'P'
                edge_r[e] = 0.8625/39690 * length
                edge_x[e] = 0.4154/39690 * length
            elif e in hvlines or (e[1],e[0]) in hvlines:
                edge_label[e] = 'E'
                edge_r[e] = 1e-12 * length
                edge_x[e] = 1e-12 * length
            else:
                edge_label[e] = 'S'
                edge_r[e] = 0.81508/57.6 * length
                edge_x[e] = 0.3496/57.6 * length
        nx.set_edge_attributes(dist_net, edge_label, 'label')
        nx.set_edge_attributes(dist_net, edge_r, 'r')
        nx.set_edge_attributes(dist_net, edge_x, 'x')
        return dist_net
    
    
    
class Display:
    """
    """
    def __init__(self,network):
        """
        """
        self.dist_net = network
        return
    
    def plot_network(self,path,filename):
        """
        Plots the generated synthetic distribution network with specific colors for
        primary and secondary networks and separate color for different nodes in the
        network.
        """
        nodelist = list(self.dist_net.nodes())
        edgelist = list(self.dist_net.edges())
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        node_label = nx.get_node_attributes(self.dist_net,'label')
        edge_label = nx.get_edge_attributes(self.dist_net,'label')
        
        # Format the nodes in the network
        colors = []
        size = []
        for n in nodelist:
            if node_label[n] == 'T':
                colors.append('green')
                size.append(20.0)
            elif node_label[n] == 'H':
                colors.append('red')
                size.append(5.0)
            elif node_label[n] == 'R':
                colors.append('black')
                size.append(5.0)
            elif node_label[n] == 'S':
                colors.append('dodgerblue')
                size.append(100.0)
        
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
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net,pos=nodepos,with_labels=False,
                         ax=ax,node_size=size,node_color=colors,
                         edgelist=edgelist,edge_color=edge_color,width=edge_width)
        
        ax.set_title("Distribution Network in the County",fontsize=30)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        # ax.set_xlabel("Longitude",fontsize=20)
        # ax.set_ylabel("Latitude",fontsize=20)
        
        # Define legends for the plot
        leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0),
                    Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='o',markersize=0),
                    Line2D([0], [0], color='dodgerblue', markerfacecolor='dodgerblue', marker='o',markersize=0),
                    Line2D([0], [0], color='white', markerfacecolor='green', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='red', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='dodgerblue', marker='o',markersize=10)]
        ax.legend(leglines,['primary network','secondary network','high voltage feeders',
                            'transformers','residences','substation'],
                  loc='best',ncol=2,prop={'size': 20})
        
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
        return
    
    def save_network(self,path,filename):
        """
        """
        edgelist = list(self.dist_net.edges())
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        node_label = nx.get_node_attributes(self.dist_net,'label')
        redge = nx.get_edge_attributes(self.dist_net,'r')
        xedge = nx.get_edge_attributes(self.dist_net,'x')
        edge_label = nx.get_edge_attributes(self.dist_net,'label')
        lines = ''
        for (n1,n2) in edgelist:
            r = redge[(n1,n2)] if (n1,n2) in redge else redge[(n2,n1)]
            x = xedge[(n1,n2)] if (n1,n2) in xedge else xedge[(n2,n1)]
            label = edge_label[(n1,n2)] if (n1,n2) in edge_label else edge_label[(n2,n1)]
            t1 = node_label[n1]; t2 = node_label[n2]
            long1 = nodepos[n1][0]; long2 = nodepos[n2][0]
            lat1 = nodepos[n1][1]; lat2 = nodepos[n2][1]
            lines += ' '.join([str(n1),t1,str(long1),str(lat1),
                               str(n2),t2,str(long2),str(lat2),
                               str(r),str(x),label])+'\n'
        
        # Write the edgelist in a .txt file
        f = open("{}{}.txt".format(path,filename),'w')
        f.write(lines)
        f.close()
        return
    
    def check_pf(self,path,filename):
        """
        Checks power flow solution and plots the voltage at different nodes in the 
        network through colorbars.
        """
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        A = nx.incidence_matrix(self.dist_net,nodelist=list(self.dist_net.nodes()),
                                edgelist=list(self.dist_net.edges()),oriented=True).toarray()
        
        nodelabel = nx.get_node_attributes(self.dist_net,'label')
        nodeload = nx.get_node_attributes(self.dist_net,'load')
        node_ind = [i for i,node in enumerate(self.dist_net.nodes()) \
                    if nodelabel[node] != 'S']
        nodelist = [node for node in list(self.dist_net.nodes()) if nodelabel[node] != 'S']
        
        # Resistance data
        edge_r = nx.get_edge_attributes(self.dist_net,'r')
        R = np.diag([1.0/edge_r[e] if e in edge_r else 1.0/edge_r[(e[1],e[0])] \
             for e in list(self.dist_net.edges())])
        G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
        p = np.array([nodeload[h] for h in nodelist])
        
        # Reactance data
    #    edge_x = nx.get_edge_attributes(graph,'x')
    #    X = np.diag([1.0/edge_x[e] if e in edge_x else 1.0/edge_r[(e[1],e[0])] \
    #         for e in list(graph.edges())])
    #    B = np.matmul(np.matmul(A,X),A.T)[node_ind,:][:,node_ind]
    #    q = np.array([nodeload[h]*0.328 for h in nodelist])
        
        v = np.matmul(np.linalg.inv(G),p) #+ np.matmul(np.linalg.inv(B),q)
        voltage = {h:1.0-v[i] for i,h in enumerate(nodelist)}
        subnodes = [node for node in list(self.dist_net.nodes()) if nodelabel[node] == 'S']
        for s in subnodes: voltage[s] = 1.0
        nodes = list(self.dist_net.nodes())
        colors = [voltage[n] for n in nodes]
        
        # Generate visual representation
        fig = plt.figure(figsize=(18,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net, nodepos, ax=ax, node_color=colors,
            node_size=15, cmap=plt.cm.plasma, with_labels=False, vmin=0.85, vmax=1.05)
        cobj = cm.ScalarMappable(cmap='plasma')
        cobj.set_clim(vmin=0.80,vmax=1.05)
        cbar = fig.colorbar(cobj,ax=ax)
        cbar.set_label('Voltage(pu)',size=30)
        cbar.ax.tick_params(labelsize=20)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        # ax.set_xlabel('Longitude',fontsize=20)
        # ax.set_ylabel('Latitude',fontsize=20)
        ax.set_title('Node voltages in the distribution network',
                     fontsize=30)
        
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
        return colors
    
    def check_flows(self,path,filename):
        """
        Checks power flow solution and plots the flows at different edges in the 
        network through colorbars.
        """
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        A = nx.incidence_matrix(self.dist_net,nodelist=list(self.dist_net.nodes()),
                                edgelist=list(self.dist_net.edges()),oriented=True).toarray()
        
        nodelabel = nx.get_node_attributes(self.dist_net,'label')
        nodeload = nx.get_node_attributes(self.dist_net,'load')
        node_ind = [i for i,node in enumerate(self.dist_net.nodes()) \
                    if nodelabel[node] != 'S']
        nodelist = [node for node in list(self.dist_net.nodes()) if nodelabel[node] != 'S']
        edgelist = [edge for edge in list(self.dist_net.edges())]
        
        # Resistance data
        p = np.array([nodeload[h] for h in nodelist])
        f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
        
        from math import log
        flows = {e:log(abs(f[i])) for i,e in enumerate(edgelist)}
        # edgelabel = nx.get_edge_attributes(self.dist_net,'label')
        colors = [flows[e] for e in edgelist]
        fmin = 0.2
        fmax = 800.0
        
        # Generate visual representation
        fig = plt.figure(figsize=(18,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net, nodepos, ax=ax, edge_color=colors,node_color='black',
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
        ax.set_title('Edge flows in the distribution network',
                     fontsize=30)
        
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
        return colors
    
    def plot_primary(self,path,filename):
        """
        Plots the generated synthetic distribution network with specific colors for
        primary and secondary networks and separate color for different nodes in the
        network.
        """
        # Delete the homes from graph
        graph = self.dist_net.copy()
        
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
                size.append(20.0)
            elif node_label[n] == 'H':
                colors.append('red')
                size.append(5.0)
            elif node_label[n] == 'R':
                colors.append('black')
                size.append(5.0)
            elif node_label[n] == 'S':
                colors.append('dodgerblue')
                size.append(100.0)
        
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
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(graph,pos=nodepos,with_labels=False,
                         ax=ax,node_size=size,node_color=colors,
                         edgelist=edgelist,edge_color=edge_color,width=edge_width)
        
        ax.set_title("Distribution Network in county",fontsize=30)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        
        # Define legends for the plot
        leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=0),
                    Line2D([0], [0], color='dodgerblue', markerfacecolor='dodgerblue', marker='o',markersize=0),
                    Line2D([0], [0], color='white', markerfacecolor='green', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='dodgerblue', marker='o',markersize=10)]
        ax.legend(leglines,['primary network','high voltage feeders',
                            'transformers','substation'],
                  loc='best',ncol=1,prop={'size': 20})
        
        fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
        return
