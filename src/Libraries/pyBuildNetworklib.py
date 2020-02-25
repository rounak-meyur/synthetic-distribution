# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
from geographiclib.geodesic import Geodesic
import networkx as nx
import seaborn as sns
from matplotlib import cm
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
from shapely.geometry import LineString,MultiPoint
from pyMILPlib import MILP_secondary,MILP_primary
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def read_network(filename,homes):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        filename: name of the .txt file
        homes: named tuple for residential consumer data
        
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
    nodeload = {n:homes.average[n]*0.001 if nodelabel[n]=='H' else 0.0 \
                for n in list(graph.nodes())}
    nx.set_edge_attributes(graph,edgelabel,'label')
    nx.set_edge_attributes(graph,edge_r,'r')
    nx.set_edge_attributes(graph,edge_x,'x')
    nx.set_node_attributes(graph,nodelabel,'label')
    nx.set_node_attributes(graph,nodepos,'cord')
    nx.set_node_attributes(graph,nodeload,'load')
    return graph

def create_master_graph(roads,tsfr,links):
    """
    Creates the master graph consisting of all possible edges from the road
    network links. Each node in the graph has a spatial attribute called
    cord.
    """
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
    
    # Coordinates of nodes in network
    nodepos = {n:roads.cord[n] if n in roads.cord else tsfr.cord[n] for n in nodelist}
    nx.set_node_attributes(graph,nodepos,name='cord')
    
    # Length of edges
    edge_length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                   for e in edgelist}
    nx.set_edge_attributes(graph,edge_length,name='hop')
    
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
        dis = [MeasureDistance(subs.cord[s],nodepos[n]) for n in nodes]
        S2Near[s] = nodes[dis.index(min(dis))]
    
    # Compute Voronoi cells with network distance 
    centers = list(S2Near.values())
    cells = nx.voronoi_cells(graph, centers, 'hop')
    
    # Recompute Voronoi cells for larger primary networks
    centers = [c for c in centers if len(cells[c])>100]
    cells = nx.voronoi_cells(graph, centers, 'hop')
    
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

def plot_graph(graph,subdata=None,path=None,filename=None):
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
    rcol = sns.color_palette('bright',nx.number_connected_components(graph))
    for i,nlist in enumerate(list(nx.connected_components(graph))):
        sub_graph = nx.subgraph(graph,list(nlist))
        col = ['black' if nodelabel[n]=='R' else rcol[i] for n in list(nlist)]
        nx.draw_networkx(sub_graph,pos=nodepos,nodelist = list(nlist),node_size=10.0,
                         node_color=col,edge_width=1.0, edge_color='black',ax=ax,
                         with_labels=False)
    
    # Scatter plot the substations
    if subdata != None:
        subx = subdata.cord[0]
        suby = subdata.cord[1]
        ax.scatter(subx,suby,s=80.0,c='green')
    
    # Titles and labels for plot
    ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
    ax.set_xlabel("Longitude",fontsize=20)
    ax.set_ylabel("Latitude",fontsize=20)
    ax.set_title("Possible set of edges in primary network",fontsize=20)
    
    # Define legends for the plot
    leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=10),
                Line2D([0], [0], color='white', markerfacecolor='green', marker='o',markersize=10)]+\
                [Line2D([0], [0], color='white', markerfacecolor=c, marker='o',markersize=10) for c in rcol]
    ax.legend(leglines,['road network','substations']+\
              ['transformer nodes' for i in range(len(rcol))],
              loc='best',ncol=1,prop={'size': 20})
    plt.show()
    if path != None:
        fig.savefig("{}{}.png".format(path,filename))
    return
    

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
        self.link_to_home = InvertMap(home_to_link)
        return
    
    
    def __separate_side(self,link):
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
        points = [self.home_cord[h] for h in homelist]
        (x1,y1) = self.road_cord[link[0]]
        (x2,y2) = self.road_cord[link[1]]
        eqn = [((x-x1)*(y2-y1))-((y-y1)*(x2-x1)) for (x,y) in points]
        side = {home: 1 if eqn[index]>=0 else -1 for index,home in enumerate(homelist)}
        return side
    
    
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
    
    def get_nodes(self,link,minsep):
        """
        Gets all the nodes from the dummy graph created before solving the optimization 
        problem.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
        Outputs:home_pts: list of residential points
                transformers: list of points along link which are probable locations 
                of transformers.
        """
        home_pts = self.link_to_home[link]
        if len(home_pts) > 100:
            link_line = Link(LineString([tuple(self.road_cord[n]) for n in link]))
            transformers = link_line.InterpolatePoints(minsep)
        else:
            link_line = Link(LineString([tuple(self.road_cord[n]) for n in link]))
            transformers = link_line.InterpolatePoints(minsep)
        return home_pts,transformers
    
    
    def create_dummy_graph(self,link,minsep,penalty):
        """
        Creates the base network to carry out the optimization problem. The base graph
        may be a Delaunay graph or a full graph depending on the size of the problem.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
        Outputs:graph: the generated base graph also called the dummy graph
                transformers: list of points along link which are probable locations 
                of transformers.
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
        
        # Create the base graph
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
    
    def generate_optimal_topology(self,link,minsep=50,penalty=0.5,hops=4):
        """
        Calls the MILP problem and solves it using gurobi solver.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
        Outputs:forest: the generated forest graph which is the secondary network
                roots: list of points along link which are actual locations of 
                transformers.
        """
        graph,roots = self.create_dummy_graph(link,minsep,penalty)
        edgelist = MILP_secondary(graph,roots,hops).optimal_edges
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
    def __init__(self,subdata,homes,master_graph,max_node=1200):
        """
        """
        self.max_node = max_node
        self.subdata = subdata
        self.homes = homes
        self.graph = nx.Graph()
        self.__get_subgraph(master_graph)
        self.dist_net = nx.Graph()
        return
    
    def __get_subgraph(self,master_graph):
        """
        Get the subgraph of master graph from the nodes mapped to the Voronoi region
        of each substation. Thereafter, it calls the partioning function to create the
        partition of faster computation.
        Inputs:
            master_graph: master graph from which the subgraph is to be extracted.
        """
        # Get and set attributes
        nodepos = nx.get_node_attributes(master_graph,'cord')
        length = {e:MeasureDistance(nodepos[e[0]],nodepos[e[1]]) \
                  for e in list(master_graph.edges())}
        nx.set_edge_attributes(master_graph,length,'length')
        
        # Get the distance from the nearest substation
        hvdist = {r:MeasureDistance(self.subdata.cord,nodepos[r]) \
                  for r in self.subdata.nodes}
        
        graph = master_graph.subgraph(self.subdata.nodes)
        nx.set_node_attributes(graph,hvdist,'distance')
        
        # Create partitions
        self.__get_partitions(graph)
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
        secondary = [(int(temp.strip('\n').split(' ')[0]),
                         int(temp.strip('\n').split(' ')[4])) \
                        for temp in lines]
        sec_net = nx.Graph()
        sec_net.add_edges_from(secondary)
        
        # get subgraph for required primary network
        nodelist = []
        for t in roots: nodelist.extend(list(nx.descendants(sec_net,t)))
        return list(sec_net.subgraph(nodelist+roots).edges())
    
    def __get_partitions(self,graph_list):
        """
        """
        if type(graph_list) == nx.Graph: graph_list = [graph_list]
        for g in graph_list:
            if len(g) < self.max_node:
                self.graph = nx.compose(self.graph,g)
            else:
                centers = nx.periphery(g)
                cells = nx.voronoi_cells(g, centers, 'length')
                nodelist = [cells[c] for c in centers]
                sg_list = [nx.subgraph(g,nlist) for nlist in nodelist]
                print("Graph of ",len(g)," is partioned to",[len(sg) for sg in sg_list])
                self.__get_partitions(sg_list)
        return
    
    def optimal_network(self,fmax=400,feedmax=10):
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
            M = MILP_primary(subgraph,dict_tsfr,flow=fmax,feeder=feedmax)
            print("#####################Optimization Ended#####################")
            print("\n\n")
            primary += M.optimal_edges
            roots += M.roots
            tnodes += list(dict_tsfr.keys())
        return primary,roots,tnodes
        
    
    def get_sub_network(self,sec_file,flowmax=400,feedermax=10):
        """
        """
        # Optimizaton problem to get the primary network
        primary,roots,tnodes = self.optimal_network(fmax=flowmax,feedmax=feedermax)            
        
        # Get the associated secondary distribution network
        secondary = self.get_secondary(sec_file,tnodes)
        
        # Add the first edge between substation and nearest road node
        hvlines = [(self.subdata.id,r) for r in roots]
        
        # Create the network with data as attributes
        self.create_network(primary,secondary,hvlines)
        return
    
    def create_network(self,primary,secondary,hvlines):
        """
        Create a network with the primary network and part of the associated secondary
        network. The roots of the primary network are connected to the substation 
        through high voltage lines.
        Input: 
            primary: list of edges forming the primary network
            secondary: list of edges forming the associated secondary network
            hvlines: list of edges joining the roots of the primary network with the
            substation(s).
        """
        # Combione both networks and update labels
        self.dist_net.add_edges_from(hvlines+primary+secondary)
        
        # Add coordinate attributes to the nodes of the network.
        nodepos = nx.get_node_attributes(self.graph,'cord')
        nodepos.update(self.homes.cord)
        nodepos[self.subdata.id] = self.subdata.cord
        nx.set_node_attributes(self.dist_net,nodepos,'cord')
        
        # Label nodes of the created network
        node_label = nx.get_node_attributes(self.graph,'label')
        node_label.update({h:'H' for h in self.homes.cord})
        node_label[self.subdata.id] = 'S'
        nx.set_node_attributes(self.dist_net, node_label, 'label')
        
        node_load = {n:self.homes.average[n]*0.001 if node_label[n]=='H' else 0.0 \
                for n in list(self.dist_net.nodes())}
        nx.set_node_attributes(self.dist_net, node_load, 'load')
        
        # Label edges of the created network
        edge_label = {}
        edge_r = {}
        edge_x = {}
        for e in list(self.dist_net.edges()):
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
        nx.set_edge_attributes(self.dist_net, edge_label, 'label')
        nx.set_edge_attributes(self.dist_net, edge_r, 'r')
        nx.set_edge_attributes(self.dist_net, edge_x, 'x')
        return
    
    
    
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
                colors.append('blue')
                size.append(4.0)
            elif node_label[n] == 'H':
                colors.append('red')
                size.append(2.0)
            elif node_label[n] == 'R':
                colors.append('black')
                size.append(1.0)
            elif node_label[n] == 'S':
                colors.append('darkgreen')
                size.append(100.0)
        
        # Format the edges in the network
        edge_color = []
        for e in edgelist:
            if e in edge_label: 
                edge = e
            else:
                edge = (e[1],e[0])
            if edge_label[edge] == 'P':
                edge_color.append('black')
            elif edge_label[edge] == 'S':
                edge_color.append('crimson')
            else:
                edge_color.append('darkgreen')
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net,pos=nodepos,with_labels=False,
                         ax=ax,node_size=size,node_color=colors,
                         edgelist=edgelist,edge_color=edge_color)
        
        ax.set_title("Distribution Network in Montgomery County",fontsize=20)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set_xlabel("Longitude",fontsize=20)
        ax.set_ylabel("Latitude",fontsize=20)
        
        # Define legends for the plot
        leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=10),
                    Line2D([0], [0], color='crimson', markerfacecolor='crimson', marker='o',markersize=10),
                    Line2D([0], [0], color='darkgreen', markerfacecolor='darkgreen', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='blue', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='red', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='darkgreen', marker='o',markersize=10)]
        ax.legend(leglines,['primary network','secondary network','high voltage feeders',
                            'transformers','residences','substation'],
                  loc='best',ncol=2,prop={'size': 20})
        
        fig.savefig("{}{}.png".format(path,filename))
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
            node_size=10, cmap='viridis', with_labels=False, vmin=0.7, vmax=1.0)
        cobj = cm.ScalarMappable(cmap='viridis')
        cobj.set_clim(vmin=0.7,vmax=1.0)
        cbar = fig.colorbar(cobj,ax=ax)
        cbar.set_label('Voltage(pu)',size=20)
        cbar.ax.tick_params(labelsize=20)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set_xlabel('Longitude',fontsize=20)
        ax.set_ylabel('Latitude',fontsize=20)
        ax.set_title('Operating voltage at the nodes in the distribution network',
                     fontsize=20)
        
        fig.savefig("{}{}.png".format(path,filename))
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
        
        flows = {e:f[i] for i,e in enumerate(edgelist)}
        # edgelabel = nx.get_edge_attributes(self.dist_net,'label')
        colors = [abs(flows[e]) for e in edgelist]
        fmin = 0
        fmax = 0.3
        
        # Generate visual representation
        fig = plt.figure(figsize=(18,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net, nodepos, ax=ax, edge_color=colors,node_color='black',
            node_size=0.1, edge_cmap=plt.cm.plasma, with_labels=False, 
            vmin=fmin, vmax=fmax,width=2)
        cobj = cm.ScalarMappable(cmap='plasma')
        cobj.set_clim(vmin=fmin,vmax=fmax)
        cbar = fig.colorbar(cobj,ax=ax)
        cbar.set_label('Loading level',size=20)
        cbar.ax.tick_params(labelsize=20)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set_xlabel('Longitude',fontsize=20)
        ax.set_ylabel('Latitude',fontsize=20)
        ax.set_title('Loading level of edges in the distribution network',
                     fontsize=20)
        
        fig.savefig("{}{}.png".format(path,filename))
        return colors
    
    def plot_primary(self,homes,path,filename):
        """
        Plots the generated synthetic distribution network with specific colors for
        primary and secondary networks and separate color for different nodes in the
        network.
        """
        # Delete the homes from graph
        graph = self.dist_net.copy()
        for n in list(graph.nodes()):
            if n in homes.cord: graph.remove_node(n)
        
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
                colors.append('blue')
                size.append(4.0)
            elif node_label[n] == 'H':
                colors.append('red')
                size.append(2.0)
            elif node_label[n] == 'R':
                colors.append('black')
                size.append(1.0)
            elif node_label[n] == 'S':
                colors.append('darkgreen')
                size.append(100.0)
        
        # Format the edges in the network
        edge_color = []
        for e in edgelist:
            if e in edge_label: 
                edge = e
            else:
                edge = (e[1],e[0])
            if edge_label[edge] == 'P':
                edge_color.append('black')
            elif edge_label[edge] == 'S':
                edge_color.append('crimson')
            else:
                edge_color.append('darkgreen')
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(graph,pos=nodepos,with_labels=False,
                         ax=ax,node_size=size,node_color=colors,
                         edgelist=edgelist,edge_color=edge_color)
        
        ax.set_title("Distribution Network in Montgomery County",fontsize=20)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set_xlabel("Longitude",fontsize=20)
        ax.set_ylabel("Latitude",fontsize=20)
        
        # Define legends for the plot
        leglines = [Line2D([0], [0], color='black', markerfacecolor='black', marker='o',markersize=10),
                    Line2D([0], [0], color='darkgreen', markerfacecolor='darkgreen', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='blue', marker='o',markersize=10),
                    Line2D([0], [0], color='white', markerfacecolor='darkgreen', marker='o',markersize=10)]
        ax.legend(leglines,['primary network','high voltage feeders',
                            'transformers','substation'],
                  loc='best',ncol=1,prop={'size': 20})
        
        fig.savefig("{}{}.png".format(path,filename))
        return
