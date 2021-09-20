# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
import sys,os
from pyGeometrylib import geodist, Link
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from shapely.geometry import LinearRing,Point,LineString
from pyqtree import Index
import gurobipy as grb
import datetime

#%% Function for callback
def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if(time>300 and abs(objbst - objbnd) < 0.005 * (1.0 + abs(objbst))):
            print('Stop early - 0.50% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.0025 * (1.0 + abs(objbst))):
            print('Stop early - 0.25% gap achieved time exceeds 1 minute')
            model.terminate()
        elif(time>300 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1.00% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>480 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5.00% gap achieved time exceeds 8 minutes')
            model.terminate()
        elif(time>600 and abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst))):
            print('Stop early - 10.0% gap achieved time exceeds 10 minutes')
            model.terminate()
        elif(time>1500 and abs(objbst - objbnd) < 0.15 * (1.0 + abs(objbst))):
            print('Stop early - 15.0% gap achieved time exceeds 25 minutes')
            model.terminate()
        elif(time>3000 and abs(objbst - objbnd) < 0.2 * (1.0 + abs(objbst))):
            print('Stop early - 20.0% gap achieved time exceeds 50 minutes')
            model.terminate()
        elif(time>6000 and abs(objbst - objbnd) < 0.3 * (1.0 + abs(objbst))):
            print('Stop early - 30.0% gap achieved time exceeds 100 minutes')
            model.terminate()
        elif(time>12000 and abs(objbst - objbnd) < 0.4 * (1.0 + abs(objbst))):
            print('Stop early - 40.0% gap achieved time exceeds 200 minutes')
            model.terminate()
    return

#%% Classes
class MapLink:
    """
    This class consists of attributes and methods to evaluate the nearest road
    network link to a point. The point may be a home location or a substation.
    The algorithm uses a QD-Tree approach to evaluate a bounding box for each
    point and link then finds the nearest link to each point.
    """
    def __init__(self,road,radius=0.01):
        '''
        '''
        longitudes = [road.nodes[n]['x'] for n in road.nodes()]
        latitudes = [road.nodes[n]['y'] for n in road.nodes()]
        xmin = min(longitudes); xmax = max(longitudes)
        ymin = min(latitudes); ymax = max(latitudes)
        bbox = (xmin,ymin,xmax,ymax)
    
        # keep track of lines so we can recover them later
        all_link = list(road.edges())
        self.lines = []
    
        # initialize the quadtree index
        self.idx = Index(bbox)
        
        # add edge bounding boxes to the index
        for i, link in enumerate(all_link):
            # create line geometry
            link_geom = road.edges[link]['geometry']
        
            # bounding boxes, with padding
            x1, y1, x2, y2 = link_geom.bounds
            bounds = x1-radius, y1-radius, x2+radius, y2+radius
        
            # add to quadtree
            self.idx.insert(i, bounds)
        
            # save the line for later use
            self.lines.append((link_geom, bounds, link))
        return
    
    
    def map_point(self,points,path=os.getcwd(),fiscode='121',
                  radius=0.01):
        '''
        Finds the nearest link to the point under consideration and saves the
        map as a csv file in the specified location.
        '''
        Map2Link = {}
        for h in points.cord:
            pt = Point(points.cord[h])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = self.idx.intersect(pt_bounds)
            
            # find closest path
            try:
                closest_path = min(matches, 
                                   key=lambda i: self.lines[i][0].distance(pt))
                Map2Link[h] = self.lines[closest_path][-1]
            except:
                Map2Link[h] = None
        
        # Delete unmapped points
        unmapped = [p for p in Map2Link if Map2Link[p]==None]
        for p in unmapped:
            del Map2Link[p]
        
        # Save as a csv file
        df_map = pd.DataFrame.from_dict(Map2Link,orient='index',
                                        columns=['source','target'])
        df_map.index.names = ['hid']
        df_map.to_csv(path+fiscode+'-home2link.csv')
        return Map2Link

#%% Mapping functions compatible with Open Street Maps    
class MapOSM:
    """
    Class consisting of attributes and methods to map OSM links to the residences
    """
    def __init__(self,road,radius=0.01):
        """
        Initializes the class object by creating a bounding box of known radius
        around each OSM road link.

        Parameters
        ----------
        road : networkx Multigraph
            The Open Street Map multigraph with node and edge attributes.
        radius : float, optional
            The radius of the bounding box around each road link. 
            The default is 0.01.

        Returns
        -------
        None.

        """
        longitudes = [road.nodes[n]['x'] for n in road.nodes()]
        latitudes = [road.nodes[n]['y'] for n in road.nodes()]
        xmin = min(longitudes); xmax = max(longitudes)
        ymin = min(latitudes); ymax = max(latitudes)
        bbox = (xmin,ymin,xmax,ymax)
        
        # keep track of edges so we can recover them later
        all_link = list(road.edges(keys=True))
        self.links = []
    
        # initialize the quadtree index
        self.idx = Index(bbox)
        
        # add edge bounding boxes to the index
        for i, link in enumerate(all_link):
            # create line geometry
            link_geom = road.edges[link]['geometry']
        
            # bounding boxes, with padding
            x1, y1, x2, y2 = link_geom.bounds
            bounds = x1-radius, y1-radius, x2+radius, y2+radius
        
            # add to quadtree
            self.idx.insert(i, bounds)
        
            # save the line for later use
            self.links.append((link_geom, bounds, link))
        return
    
    def map_point(self,points,path=os.getcwd(),fiscode='121',
                  radius=0.01):
        '''
        Finds the nearest link to the residence under consideration and saves the
        map as a csv file in the specified location.
        '''
        Map2Link = {}
        for h in points.cord:
            pt = Point(points.cord[h])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = self.idx.intersect(pt_bounds)
            
            # find closest path
            try:
                closest_path = min(matches, 
                                   key=lambda i: self.links[i][0].distance(pt))
                Map2Link[h] = self.links[closest_path][-1]
            except:
                Map2Link[h] = None
        
        # Delete unmapped points
        unmapped = [p for p in Map2Link if Map2Link[p]==None]
        for p in unmapped:
            del Map2Link[p]
        
        # Save as a csv file
        df_map = pd.DataFrame.from_dict(Map2Link,orient='index',
                                        columns=['source','target','key'])
        df_map.index.names = ['hid']
        df_map.to_csv(path+fiscode+'-home2OSM.csv')
        return Map2Link
    



#%% Mixed Integer Linear Program to create secondary network
class MILP_secondary:
    """
    """
    def __init__(self,graph,roots,max_hop=10,tsfr_max=25,grbpath=None):
        """
        """
        suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
        suffix = ''
        self.tmp = grbpath+"gurobi/"+suffix+"-"
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        print("Number of edges:",len(self.edges))
        self.hindex = [i for i,node in enumerate(self.nodes) if node not in roots]
        self.tindex = [i for i,node in enumerate(self.nodes) if node in roots]
        self.A = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        COST = nx.get_edge_attributes(graph,name='cost')
        LENGTH = nx.get_edge_attributes(graph,name='length')
        LOAD = nx.get_node_attributes(graph,name='load')
        self.c = np.array([1e-3*COST[e] for e in self.edges])
        self.l = np.array([1e-3*LENGTH[e] for e in self.edges])
        self.p = np.array([1e-3*LOAD[self.nodes[i]] for i in self.hindex])
        self.model = grb.Model(name="Get Spiders")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__radiality()
        self.__heuristic(M=max_hop)
        self.__powerflow(M=tsfr_max)
        self.__objective()
        self.model.write(self.tmp+"secondary.lp")
        self.optimal_edges = self.__solve()
        return
    
    def __variables(self):
        """
        Create variable initialization for MILP such that x is binary, f is a 
        continuous variable.
        """
        self.x = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.BINARY,name='x')
        self.f = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=-grb.GRB.INFINITY,name='f')
        self.z = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=-grb.GRB.INFINITY,name='z')
        self.model.update()
        return
    
    def __radiality(self):
        """
        Radiality constraints in the form of linear problem are defined:
            1. Number of edges of a forest is number of nodes except root
            2. Each connected component of forest should have a transformer
        """
        print("Setting up radiality constraints")
        self.model.addConstr(self.x.sum() == len(self.hindex), name="radiality")
        self.model.update()
        return
    
    def __heuristic(self,M=10):
        """
        """
        self.model.addConstr(
            self.A[self.hindex,:]@self.z == -np.ones(shape=(len(self.hindex),)),
            name = 'connectivity')
        self.model.addConstr(self.z - M*self.x <= 0,name="hop_a")
        self.model.addConstr(self.z + M*self.x >= 0,name="hop_b")
        self.model.addConstr(
            self.I[self.hindex,:]@self.x <= 2*np.ones(shape=(len(self.hindex),)),
            name = 'degree')
        self.model.update()
        return
    
    def __powerflow(self,r=0.81508/57.6,M=25):
        """
        """
        print("Setting up power flow constraints")
        self.model.addConstr(self.A[self.hindex,:]@self.f == -self.p,name='balance')
        self.model.addConstr(self.f - M*self.x <= 0,name="flow_a")
        self.model.addConstr(self.f + M*self.x >= 0,name="flow_b")
        self.model.update()
        return
    
    def __objective(self):
        """
        """
        self.model.setObjective(self.c @ self.x)
        self.model.update()
        return
    
    def __solve(self):
        """
        """
        # Turn off display and heuristics
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(self.tmp+'gurobi.log', 'w')
        
        # Pass data into my callback function
        self.model._lastiter = -grb.GRB.INFINITY
        self.model._lastnode = -grb.GRB.INFINITY
        self.model._logfile = logfile
        self.model._vars = self.model.getVars()
        
        # Solve model and capture solution information
        self.model.optimize(mycallback)
        
        # Close log file
        logfile.close()
        print('')
        print('Optimization complete')
        if self.model.SolCount == 0:
            print('No solution found, optimization status = %d' % self.model.Status)
            sys.exit(0)
        else:
            print('Solution found, objective = %g' % self.model.ObjVal)
            x_optimal = self.x.getAttr("x").tolist()
            return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]


#%% Secondary network generation based on Open Street Maps or HERE dataset
def generate_optimal_topology(linkgeom,homes,minsep=50,penalty=0.5,
                              heuristic=None,hops=4,tsfr_max=25,path=None):
    """
    Calls the MILP problem and solves it using gurobi solver.
    
    Inputs: linkgeom: road link geometry.
            minsep: minimum separation in meters between the transformers.
            penalty: penalty factor for crossing the link.
            heuristic: join transformers to nearest few nodes given by heuristic.
                      Used to create the dummy graph
    Outputs:forest: the generated forest graph which is the secondary network
            roots: list of points along link which are actual locations of 
            transformers.
    """
    graph,roots = create_dummy_graph(linkgeom,homes,minsep,penalty,
                                          heuristic=heuristic)
    edgelist = MILP_secondary(graph,roots,max_hop=hops,tsfr_max=tsfr_max,
                              grbpath=path).optimal_edges
    
    # Generate the forest of trees
    forest = nx.Graph()
    forest.add_edges_from(edgelist)
    node_cord = {node: (roots[node].x,roots[node].y) \
                 if node in roots else homes[node]['cord']\
                 for node in forest}
    nx.set_node_attributes(forest,node_cord,'cord')
    node_load = {node:sum([homes[h]['load'] \
                           for h in list(nx.descendants(forest,node))]) \
                 if node in roots else homes[node]['load'] \
                     for node in list(forest.nodes())}
    nx.set_node_attributes(forest,node_load,'load')
    return forest,roots


def create_dummy_graph(linkgeom,homes,minsep=50,penalty=0.5,heuristic=None):
    """
    Creates the base network to carry out the optimization problem. The base graph
    may be a Delaunay graph or a full graph depending on the size of the problem.
    
    Inputs: 
        linkgeom: shapely geometry of road link.
        homes: dictionary of residence data
        minsep: minimum separation in meters between the transformers.
        penalty: penalty factor for crossing the link.
        heuristic: join transformers to nearest few nodes given by heuristic.
        Used to create the dummy graph
    Outputs:graph: the generated base graph also called the dummy graph
            transformers: list of points along link which are probable locations 
            of transformers.
    """
    # Interpolate points along link for probable transformer locations
    tsfr_pts = Link(linkgeom).InterpolatePoints(minsep)
    transformers = {i:pt for i,pt in enumerate(tsfr_pts)}
    
    # Identify which side of road each home is located
    link_cords = list(linkgeom.coords)
    sides = {h:1 if LinearRing(link_cords+[tuple(homes[h]['cord']),
            link_cords[0]]).is_ccw else -1 for h in homes}
    
    # Node attributes
    node_pos = {h:homes[h]['cord'] for h in homes}
    load = {h:homes[h]['load']/1000.0 for h in homes}
    
    # Create the base graph
    if len(homes)>10:
        graph = delaunay_graph_from_list(node_pos)
    else:
        graph = complete_graph_from_list(node_pos)
    
    # Add the new edges
    if heuristic != None:
        new_edges = get_nearpts_tsfr(transformers,node_pos,heuristic)
    else:
        new_edges = [(t,n) for t in transformers for n in homes]
    graph.add_edges_from(new_edges)
    
    # Update the attributes of nodes with transformer attributes
    node_pos.update(transformers)
    sides.update({t:0 for t in transformers})
    load.update({t:1.0 for t in transformers})
    
    # Add node attributes to the graph
    nx.set_node_attributes(graph,node_pos,'cord')
    nx.set_node_attributes(graph,load,'load')
    edge_cost = {e:geodist(node_pos[e[0]],node_pos[e[1]])*\
                 (1+penalty*abs(sides[e[0]]-sides[e[1]])) \
                  for e in list(graph.edges())}
    edge_length = {e:geodist(node_pos[e[0]],node_pos[e[1]])\
                  for e in list(graph.edges())}
    nx.set_edge_attributes(graph,edge_length,'length')
    nx.set_edge_attributes(graph,edge_cost,'cost')
    return graph,transformers


def delaunay_graph_from_list(homes):
    """
    Computes the Delaunay graph of the nodes in list L. L edges in the network 
    based on the definition of Delaunay triangulation. This is used as base 
    network when the number of nodes mapped to the link is small.
    
    Input: homes: dictionary of nodes coordinates
    Output: graph: the Delaunay graph which would be used as base network for the 
            optimization problem.
    """
    points = np.array([[homes[h][0],homes[h][1]] for h in homes])
    homelist = [h for h in homes]
    triangles = Delaunay(points).simplices
    edgelist = []
    for t in triangles:
        edges = [(homelist[t[0]],homelist[t[1]]),
                 (homelist[t[1]],homelist[t[2]]),
                 (homelist[t[2]],homelist[t[0]])]
        edgelist.extend(edges)
    G = nx.Graph()
    G.add_edges_from(edgelist)
    return G

def complete_graph_from_list(homes):
    """
    Computes the full graph of the nodes in list L. There would be L(L-1)/2 edges 
    in the network. This is used as base network when the number of nodes mapped 
    to the link is small.
    
    Input: homes: list of home IDs
    Output: graph: the full graph which would be used as base network for the 
            optimization problem.
    """
    G = nx.Graph()
    homelist = [h for h in homes]
    edges = combinations(homelist,2)
    G.add_edges_from(edges)
    return G


def get_nearpts_tsfr(transformers,homes,heuristic):
    """
    Heuristic to add edges between transformers and residences. The heuristic 
    denotes the number of nearest residences to consider. Construct edges between
    the transformer and those nearby residences.
    """
    edgelist = []
    homelist = [h for h in homes]
    for t in transformers:
        distlist = [geodist(transformers[t],homes[h]) for h in homes]
        imphomes = np.array(homelist)[np.argsort(distlist)[:heuristic]]
        edgelist.extend([(t,n) for n in imphomes])
    return edgelist

def checkpf(forest,roots,r=0.81508/57.6):
    """
    """
    A = nx.incidence_matrix(forest,nodelist=list(forest.nodes()),
                            edgelist=list(forest.edges()),oriented=True).toarray()
    node_pos = nx.get_node_attributes(forest,'cord')
    node_load = nx.get_node_attributes(forest,'load')
    R = [1.0/(geodist(node_pos[e[0]],node_pos[e[1]])*0.001*r) \
         for e in list(forest.edges())]
    D = np.diag(R)
    home_ind = [i for i,node in enumerate(forest.nodes()) \
                if node not in roots]
    homelist = [node for node in list(forest.nodes()) if node not in roots]
    G = np.matmul(np.matmul(A,D),A.T)[home_ind,:][:,home_ind]
    p = np.array([node_load[h]*0.001 for h in homelist])
    v = np.matmul(np.linalg.inv(G),p)
    voltage = {h:1.0-v[i] for i,h in enumerate(homelist)}
    return voltage

#%% Combine networks
def create_master_graph(roads,tsfr,links):
    """
    Creates the master graph consisting of all possible edges from the road
    network links. Each node in the graph has a spatial attribute called
    cord.
    
    roads:
        TYPE: namedtuple
        DESCRIPTION: road network data
    tsfr:
        TYPE: namedtuple
        DESCRIPTION: local transformer data
    
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
    
    # Node attributes in network
    for n in graph.nodes:
        if n in roads:
            graph.nodes[n]['cord'] = (roads.nodes[n]['x'],roads.nodes[n]['y'])
            graph.nodes[n]['label'] = 'R'
            graph.nodes[n]['load'] = 0.0
        else:
            graph.nodes[n]['cord'] = tsfr.cord[n]
            graph.nodes[n]['label'] = 'T'
            graph.nodes[n]['load'] = tsfr.load[n]
    
    # Length of edges
    for e in graph.edges:
        edge_geom = LineString((graph.nodes[e[0]]['cord'],graph.nodes[e[1]]['cord']))
        graph.edges[e]['length'] = Link(edge_geom).geod_length
    return graph

def create_master_graph_OSM(roads,tsfr,links):
    """
    Creates the master graph consisting of all possible edges from the road
    network links. Each node in the graph has a spatial attribute called
    cord.
    
    roads:
        TYPE: namedtuple
        DESCRIPTION: road network data
    tsfr:
        TYPE: namedtuple
        DESCRIPTION: local transformer data
    
    """
    road_edges = list(roads.edges(keys=True))
    tsfr_edges = list(tsfr.graph.edges())
    for edge in links:
        if edge in road_edges:
            road_edges.remove(edge)
        elif (edge[1],edge[0],edge[2]) in road_edges:
            road_edges.remove((edge[1],edge[0],edge[2]))
    edgelist = [(e[0],e[1]) for e in road_edges] + tsfr_edges
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    
    # Node attributes in network
    for n in graph.nodes:
        if n in roads:
            graph.nodes[n]['cord'] = (roads.nodes[n]['x'],roads.nodes[n]['y'])
            graph.nodes[n]['label'] = 'R'
            graph.nodes[n]['load'] = 0.0
        else:
            graph.nodes[n]['cord'] = tsfr.cord[n]
            graph.nodes[n]['label'] = 'T'
            graph.nodes[n]['load'] = tsfr.load[n]
    
    # Length of edges
    for e in graph.edges:
        edge_geom = LineString((graph.nodes[e[0]]['cord'],graph.nodes[e[1]]['cord']))
        graph.edges[e]['length'] = Link(edge_geom).geod_length
    return graph