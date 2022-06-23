# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:35:55 2022

Author: Rounak Meyur

Description: Functions to extract data from open sources
"""

import sys
import pandas as pd
import numpy as np
import osmnx as ox
from collections import namedtuple as nt
from shapely.geometry import Point,LineString,MultiPoint,LinearRing
import geopandas as gpd
from geographiclib.geodesic import Geodesic
from pyqtree import Index
import networkx as nx
import datetime
import gurobipy as grb
from scipy.spatial import Delaunay
from itertools import combinations
from collections import defaultdict
from scipy.spatial import cKDTree
from math import log,exp
import shutil
import tempfile
from pathlib import Path


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
        # Compute great circle distance
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length
    
    def InterpolatePoints(self,sep=20):
        """
        """
        points = []
        length = self.geod_length
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(Point(xy))
        if len(points)==0: 
            points.append(Point((self.xy[0][0],self.xy[1][0])))
        return points

#%% Class to perform Step 1a: Mapping between homes and road links

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
    
    def map_point(self,points,radius=0.01):
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
        
        return Map2Link

#%% Mixed Integer Linear Program to perform Step 1b: create secondary network
class MILP_secondary:
    """
    """
    def __init__(self,graph,roots,max_hop=10,tsfr_max=25,grbpath=None):
        """
        """
        suffix = datetime.datetime.now().isoformat().replace(':','-').replace('.','-')
        self.tmp = grbpath+suffix
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

#%% Classes to construct primary network
class MILP_primary:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,graph,grbpath=None,flowcap=1000,feeder_buffer=1):
        """
        graph: the base graph which has the list of possible edges.
        tnodes: dictionary of transformer nodes with power consumption as value.
        """
        # Get tmp path for gurobi log files
        self.tmp = grbpath
        
        # Get data from graph
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        LABEL = nx.get_node_attributes(graph,name='label')
        LOAD = nx.get_node_attributes(graph,name='load')
        LENGTH = nx.get_edge_attributes(graph,name='length')
        DIST = nx.get_node_attributes(graph,name='distance')
        self.tindex = [i for i,n in enumerate(self.nodes) if LABEL[n]=='T']
        self.rindex = [i for i,n in enumerate(self.nodes) if LABEL[n]=='R']
        self.tnodes = [n for n in self.nodes if LABEL[n]=='T']
        
        # Vectorize the data for matrix computation
        self.d = [1e-3*DIST[self.nodes[i]] for i in self.rindex]
        self.A = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        self.c = [1e-3*LENGTH[self.edges[i]] for i in range(len(self.edges))]
        self.p = np.array([1e-3*LOAD[self.nodes[i]] for i in self.tindex])
        
        # Get feeder rating and number
        total_cap = sum(LOAD.values())*1e-3 # total kVA load to be served
        feeder_cap = int(total_cap/1000)+feeder_buffer # Maximum number of feeders
        
        # Create the optimization model
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.masterTree()
        self.powerflow()
        self.radiality()
        self.flowconstraint(M=flowcap)
        self.connectivity()
        self.limit_feeder(M=feeder_cap)
        self.objective()
        self.model.write(self.tmp+"primary.lp")
        self.solve()
        return
        
    
    def variables(self):
        """
        """
        print("Setting up variables")
        self.x = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.BINARY,name='x')
        self.y = self.model.addMVar(len(self.rindex),
                                    vtype=grb.GRB.BINARY,name='y')
        self.z = self.model.addMVar(len(self.rindex),
                                    vtype=grb.GRB.BINARY,name='z')
        self.f = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=-grb.GRB.INFINITY,name='f')
        self.v = self.model.addMVar(len(self.nodes),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=0.9,ub=1.0,name='v')
        self.t = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.BINARY,name='t')
        self.g = self.model.addMVar(len(self.edges),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=-grb.GRB.INFINITY,name='g')
        self.model.update()
        return
    
    def powerflow(self,r=0.8625/39690,M=0.15):
        """
        """
        print("Setting up power flow constraints")
        expr = r*np.diag(self.c)@self.f
        self.model.addConstr(self.A.T@self.v - expr - M*(1-self.x) <= 0)
        self.model.addConstr(self.A.T@self.v - expr + M*(1-self.x) >= 0)
        self.model.addConstr(1-self.v[self.rindex] <= self.z)
        self.model.update()
        return
    
    def radiality(self):
        """
        """
        print("Setting up radiality constraints")
        self.model.addConstr(
                self.x.sum() == len(self.nodes) - 2*len(self.rindex) + \
                self.y.sum()+self.z.sum())
        self.model.update()
        return
    
    def connectivity(self):
        """
        """
        print("Setting up connectivity constraints")
        self.model.addConstr(
                self.I[self.rindex,:]@self.x <= len(self.edges)*self.y)
        self.model.addConstr(
                self.I[self.rindex,:]@self.x >= 2*(self.y+self.z-1))
        self.model.addConstr(1-self.z <= self.y)
        self.model.update()
        return
    
    def flowconstraint(self,M=400):
        """
        """
        print("Setting up capacity constraints for road nodes")
        self.model.addConstr(self.A[self.rindex,:]@self.f - M*(1-self.z) <= 0)
        self.model.addConstr(self.A[self.rindex,:]@self.f + M*(1-self.z) >= 0)
        
        print("Setting up flow constraints for transformer nodes")
        self.model.addConstr(self.A[self.tindex,:]@self.f == -self.p)
        
        print("Setting up flow constraints")
        self.model.addConstr(self.f - M*self.x <= 0)
        self.model.addConstr(self.f + M*self.x >= 0)
        self.model.update()
        return
    
    def limit_feeder(self,M=10):
        """
        """
        print("Setting up constraint for number of feeders")
        self.model.addConstr(len(self.rindex)-self.z.sum() <= M)
        self.model.update()
        return
    
    def masterTree(self):
        """
        """
        print("Setting up imaginary constraints for master solution")
        M = len(self.nodes)
        self.model.addConstr(self.A[1:,:]@self.g == -np.ones(shape=(M-1,)))
        self.model.addConstr(self.g - M*self.t <= 0)
        self.model.addConstr(self.g + M*self.t >= 0)
        self.model.addConstr(self.x <= self.t)
        self.model.update()
        
    
    def objective(self):
        """
        """
        print("Setting up objective function")
        self.model.setObjective(
                np.array(self.c)@self.x - np.array(self.d)@self.z + sum(self.d))
        return
    
    def solve(self):
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
            z_optimal = self.z.getAttr("x").tolist()
            self.optimal_edges = [e for i,e in enumerate(self.edges) if x_optimal[i]>0.8]
            self.roots = [self.nodes[ind] for i,ind in enumerate(self.rindex) if z_optimal[i]<0.2]
            return    


class Primary:
    """
    Creates the primary distribuion network by solving an optimization problem.
    First the set of possible edges are identified from the links in road net-
    -work and transformer connections.
    """
    def __init__(self,subdata,master,feedcap=1000,div=10):
        """
        """
        self.subdata = subdata
        self.graph = nx.Graph()
        
        # Update master graph with substation distance data
        self.hvpath = update_data(master,subdata)
        print("Road distance updated")
        
        # Partition the master graph based on load
        M = sum(nx.get_node_attributes(master,'load').values())/1e3
        self.max_mva = max([feedcap,M/div])
        self.get_partitions(master)
        print("Master graph partitioned")
        return
    
    def get_partitions(self,graph_list):
        """
        This function handles primary network creation for large number of nodes.
        It divides the network into multiple partitions of small networks and solves
        the optimization problem for each sub-network.
        """
        if type(graph_list) == nx.Graph: graph_list = [graph_list]
        for g in graph_list:
            total_mva = sum(nx.get_node_attributes(g,'load').values())/1e3
            if total_mva < self.max_mva:
                self.graph = nx.compose(self.graph,g)
            else:
                comp = nx.algorithms.community.girvan_newman(g)
                nodelist = list(sorted(c) for c in next(comp))
                sglist = [nx.subgraph(g,nlist) for nlist in nodelist]
                print("Graph with load of ",total_mva," is partioned to",
                      [sum(nx.get_node_attributes(sg,'load').values())/1e3 \
                       for sg in sglist])
                self.get_partitions(sglist)
        return
        
    def get_sub_network(self,grbpath=None,fcap=1000,fbuf=1):
        """
        """
        # Optimizaton problem to get the primary network
        primary = []; roots = []; tnodes = []
        for nlist in list(nx.connected_components(self.graph)):
            subgraph = nx.subgraph(self.graph,list(nlist))
            M = MILP_primary(subgraph,grbpath=grbpath,flowcap=fcap,
                             feeder_buffer=fbuf)
            print("\n\n\n")
            primary += M.optimal_edges
            roots += M.roots
            tnodes += M.tnodes
        
        # Add the first edge between substation and nearest road node
        hvlines = [(self.subdata["id"],r) for r in roots]
        
        # Create the network with data as attributes
        net = self.create_network(primary,hvlines)
        return net
    
    
    def create_network(self,primary,hvlines):
        """
        Create a network with the primary network. The roots of the primary 
        network are connected to the substation through high voltage lines.
        Input: 
            primary: list of edges forming the primary network
            hvlines: list of edges joining the roots of the primary network with the
            substation(s).
        """
        
        # Combine both networks and update labels
        prim_net = nx.Graph()
        prim_net.add_edges_from(hvlines+primary)
        
        # Add node attributes to the primary network
        for n in prim_net:
            if n == self.subdata["id"]:
                prim_net.nodes[n]["cord"] = self.subdata["cord"]
                prim_net.nodes[n]["label"] = "S"
                prim_net.nodes[n]["load"] = 0.0
            else:
                prim_net.nodes[n]["cord"] = self.graph.nodes[n]["cord"]
                prim_net.nodes[n]["label"] = self.graph.nodes[n]["label"]
                prim_net.nodes[n]["load"] = 0.0
        
        # Remove cycles in the network formed of road network nodes
        remove_cycle(prim_net)
        
        # Label edges of the created network
        for e in prim_net.edges:
            if self.subdata["id"] in e:
                rnode = e[0] if e[1]==self.subdata["id"] else e[1]
                path_cords = [self.subdata["cord"]]+\
                                   [self.graph.nodes[n]["cord"] for n in self.hvpath[rnode]]
                prim_net.edges[e]['geometry'] = LineString(path_cords)
                prim_net.edges[e]['length'] = Link(prim_net.edges[e]['geometry']).geod_length
                prim_net.edges[e]['label'] = 'E'
                prim_net.edges[e]['r'] = 1e-12 * prim_net.edges[e]['length']
                prim_net.edges[e]['x'] = 1e-12 * prim_net.edges[e]['length']
            else:
                prim_net.edges[e]['geometry'] = LineString((prim_net.nodes[e[0]]["cord"],
                                                            prim_net.nodes[e[1]]["cord"]))
                prim_net.edges[e]['length'] = max(Link(prim_net.edges[e]['geometry']).geod_length,1e-12)
                prim_net.edges[e]['label'] = 'P'
                prim_net.edges[e]['r'] = 0.8625/39690 * prim_net.edges[e]['length']
                prim_net.edges[e]['x'] = 0.4154/39690 * prim_net.edges[e]['length']
        return prim_net
    

#%% Generic functions

def geodist(geomA,geomB):
    if type(geomA) != Point: geomA = Point(geomA)
    if type(geomB) != Point: geomB = Point(geomB)
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']

def groups(many_to_one):
    """Converts a many-to-one mapping into a one-to-many mapping.

    `many_to_one` must be a dictionary whose keys and values are all
    :term:`hashable`.

    The return value is a dictionary mapping values from `many_to_one`
    to sets of keys from `many_to_one` that have that value.

    """
    one_to_many = defaultdict(set)
    for v, k in many_to_one.items():
        one_to_many[k].add(v)
    D = dict(one_to_many)
    return {k:list(D[k]) for k in D}

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

#%% Functions to extract data

def GetHomes(filename):
    """
    Gets residence data from the file

    Parameters
    ----------
    filename : string
        Path to the residence file.

    Returns
    -------
    homes : named tuple
        consists coordinates, load demand profile, average and peak hourly load.

    """
    df_home = pd.read_csv(filename)
    df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","average"])
    dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
    dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
    dict_peak = dict(zip(df_home.hid,df_home.peak))
    dict_avg = dict(zip(df_home.hid,df_home.average))
    homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
    return homes


def combine_osm_components(road,radius = 0.01):
    """
    Combines network components by finding nearest nodes
    based on a QD Tree Approach.

    Parameters
    ----------
    graph : networkx MultiGraph
        graph representing the road network.

    Returns
    -------
    edgelist: list of tuples.
        list of edges for the combined road network graph

    """
    # Initialize QD Tree
    longitudes = [road.nodes[n]['x'] for n in road.nodes()]
    latitudes = [road.nodes[n]['y'] for n in road.nodes()]
    xmin = min(longitudes); xmax = max(longitudes)
    ymin = min(latitudes); ymax = max(latitudes)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    # Differentiate large and small components
    comps = [c for c in list(nx.connected_components(road))]
    lencomps = [len(c) for c in list(nx.connected_components(road))]
    indlarge = lencomps.index(max(lencomps))
    node_main = list(road.subgraph(comps[indlarge]).nodes())
    del(comps[indlarge])
    
    # keep track of nodes so we can recover them later
    nodes = []
    
    # create bounding box around each point in large component
    for i,node in enumerate(node_main):
        pt = Point([road.nodes[node]['x'],road.nodes[node]['y']])
        pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
        idx.insert(i, pt_bounds)
        nodes.append((pt, pt_bounds, node))
        
    # find intersection and add edges
    edgelist = []
    for c in comps:
        node_comp = list(road.subgraph(c).nodes())
        nodepairs = []
        for n in node_comp:
            pt = Point([road.nodes[n]['x'],road.nodes[n]['y']])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = idx.intersect(pt_bounds)
            closest_pt = min(matches,key=lambda i: nodes[i][0].distance(pt))
            nodepairs.append((n,nodes[closest_pt][-1]))
        
        # Get the geodesic distance
        dist = [geodist([road.nodes[p[0]]['x'],road.nodes[p[0]]['y']],
                        [road.nodes[p[1]]['x'],road.nodes[p[1]]['y']]) \
                for p in nodepairs]
        edgelist.append(nodepairs[np.argmin(dist)]+tuple([0]))
    return edgelist


def GetOSMRoads(homes):
    points = [Point(homes.cord[n]) for n in homes.cord]
    bound_polygon = MultiPoint(points).convex_hull
    
    # Get the OSM links within the county polygon
    osm_graph = ox.graph_from_polygon(bound_polygon, retain_all=True,
                                  truncate_by_edge=True)
    
    G = osm_graph.to_undirected()
    
    # Add geometries for links without it
    edge_nogeom = [e for e in G.edges(keys=True) if 'geometry' not in G.edges[e]]
    for e in edge_nogeom:
        pts = [(G.nodes[e[0]]['x'],G.nodes[e[0]]['y']),
               (G.nodes[e[1]]['x'],G.nodes[e[1]]['y'])]
        link_geom = LineString(pts)
        G.edges[e]['geometry'] = link_geom
    
    # Join disconnected components in the road network
    new_edges = combine_osm_components(G,radius=0.1)
    G.add_edges_from(new_edges)
    for i,e in enumerate(new_edges):
        pts = [(G.nodes[e[0]]['x'],G.nodes[e[0]]['y']),
               (G.nodes[e[1]]['x'],G.nodes[e[1]]['y'])]
        link_geom = LineString(pts)
        G.edges[e]['geometry'] = link_geom
        G.edges[e]['length'] = Link(link_geom).geod_length
        G.edges[e]['oneway'] = float('nan')
        G.edges[e]['highway'] = 'extra'
        G.edges[e]['name'] = 'extra'
        G.edges[e]['osmid'] = str(80000+i)
    return G

def GetSubstations(filename,homes):
    """
    Get the substations within the geographic region

    Parameters
    ----------
    filename : string
        path name to the shape file with substation data.
    homes : named tuple
        residence data.

    Returns
    -------
    subs: named tuple
        coordinates and ID of the substations.

    """
    subs = nt("substation",field_names=["cord"])
    
    points = [Point(homes.cord[n]) for n in homes.cord]
    bound_polygon = MultiPoint(points).convex_hull
    
    df_subs = pd.read_csv(filename)
    df_subs["geometry"]= [Point(df_subs["X"][i],df_subs["Y"][i]) \
                          for i in range(len(df_subs))]
    
    dict_cord = {}
    for i in range(len(df_subs)):
        if df_subs["geometry"][i].within(bound_polygon):
            dict_cord[int(df_subs["ID"][i])] = (df_subs["X"][i],df_subs["Y"][i])
    return subs(cord=dict_cord)



#%% Functions to perform Step 1b: create secondary network

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

#%% Functions for Voronoi partitioning
def bounds(pt,radius):
    """
    Returns the bounds for a point geometry. The bound is a square around the
    point with side of 2*radius units.
    
    pt:
        TYPE: shapely point geometry
        DESCRIPTION: the point for which the bound is to be returned
    
    radius:
        TYPE: floating type 
        DESCRIPTION: radius for the bounding box
    """
    return (pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)

def find_nearest_node(center_cord,node_cord):
    """
    Computes the nearest node in the dictionary 'node_cord' to the point denoted
    by the 'center_cord'
    
    center_cord: 
        TYPE: list of two entries
        DESCRIPTION: geographical coordinates of the center denoted by a list
                     of two entries
    
    node_cord: 
        TYPE: dictionary 
        DESCRIPTION: dictionary of nodelist with values as the geographical 
                     coordinate
    """
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

def get_nearest_road(subs,graph):
    """
    Get list of nodes mapped in the Voronoi cell of the substation. The Voronoi 
    cell is determined on the basis of geographical distance.
    Returns: dictionary of substations with list of nodes mapped to it as values
    """
    # Get the Voronoi centers and data points
    centers = list(subs.cord.keys())
    center_pts = [subs.cord[s] for s in centers]
    nodes = list(graph.nodes())
    nodepos = nx.get_node_attributes(graph,'cord')
    nodelabel = nx.get_node_attributes(graph,'label')
    node_pts = [nodepos[n] for n in nodes]
    
    # Find number of road nodes mapped to each substation
    voronoi_kdtree = cKDTree(center_pts)
    _, node_regions = voronoi_kdtree.query(node_pts, k=1, 
                                           distance_upper_bound=0.01)
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
        nodes_partition = [n for n in S2Node[s] if nodelabel[n]=='R']
        nodecord = {n: nodepos[n] for n in nodes_partition}
        S2Near[s] = find_nearest_node(subs.cord[s],nodecord)
    return S2Near

def get_partitions(S2Near,graph):
    """
    Get list of nodes mapped in the Voronoi cell of the substation. The Voronoi 
    cell is determined on the basis of shortest path distance from each node to
    the nearest node to the substation.
    Returns: dictionary of substations with list of nodes mapped to it as values
    """
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

def create_voronoi(subs,graph):
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
    S2Near = get_nearest_road(subs,graph)
    S2Node = get_partitions(S2Near,graph)
    return S2Near,S2Node

#%% Functions for primary network creation
def update_data(graph,subdata):
    # Get the distance from the nearest substation
    hvpath = {r:nx.shortest_path(graph,source=subdata['near'],
                                 target=r,weight='length') \
              if graph.nodes[r]['label']=='R' else [] for r in graph}
    hvdist = {r:sum([graph[hvpath[r][i]][hvpath[r][i+1]]['length']\
                     for i in range(len(hvpath[r])-1)]) for r in graph}
    nx.set_node_attributes(graph,hvdist,'distance')
    return hvpath

def remove_cycle(graph):
    try:
        cycle = nx.find_cycle(graph)
        print("Cycles found:",cycle)
        nodes = list(set([c[0] for c in cycle] + [c[1] for c in cycle]))
        nodes = [n for n in nodes if graph.nodes[n]['label']=='R']
        print("Number of nodes:",graph.number_of_nodes())
        print("Removing cycles...")
        for n in nodes:
            graph.remove_node(n)
        print("After removal...Number of nodes:",graph.number_of_nodes())
        print("Number of comps.",nx.number_connected_components(graph))
        remove_cycle(graph)
    except:
        print("No cycles found!!!")
        return

def add_secnet(graph,tsfrdat,homedat):
    # Add secondary network edges
    secnet = nx.Graph()
    for t in tsfrdat:
        graph.add_edges_from(tsfrdat[t]['secnet'])
        secnet.add_edges_from(tsfrdat[t]['secnet'])
    
    # Add new node/edge attributes
    hnodes = [n for n in secnet if n not in tsfrdat]
    for n in hnodes:
        graph.nodes[n]['cord'] = homedat.cord[n]
        graph.nodes[n]['label'] = 'H'
        if n in homedat.average:
            graph.nodes[n]['load'] = homedat.average[n]
        else:
            # use a sample house as the load
            graph.nodes[n]['load'] = homedat.average[[h for h in homedat.average][0]]
    
    for e in graph.edges:
        if e in secnet.edges:
            graph.edges[e]['geometry'] = LineString((graph.nodes[e[0]]["cord"],
                                                     graph.nodes[e[1]]["cord"]))
            graph.edges[e]['length'] = Link(graph.edges[e]['geometry']).geod_length
            graph.edges[e]['label'] = 'S'
            graph.edges[e]['r'] = 0.81508/57.6 * graph.edges[e]['length']
            graph.edges[e]['x'] = 0.34960/57.6 * graph.edges[e]['length']
    return graph

#%% Post processing Steps
def powerflow(graph,v0=1.0):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    # Pre-processing to rectify incorrect code
    hv_lines = [e for e in graph.edges if graph.edges[e]['label']=='E']
    for e in hv_lines:
        try:
            length = graph.edges[e]['length']
        except:
            length = graph.edges[e]['geo_length']
        graph.edges[e]['r'] = (0.0822/363000)*length*1e-3
        graph.edges[e]['x'] = (0.0964/363000)*length*1e-3
    
    # Main function begins here
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(graph.edges())]
    
    # Resistance data
    edge_r = []
    for e in graph.edges:
        try:
            edge_r.append(1.0/graph.edges[e]['r'])
        except:
            edge_r.append(1.0/1e-14)
    R = np.diag(edge_r)
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*graph.nodes[n]['load'] for n in nodelist])
    
    # Voltages and flows
    v = np.matmul(np.linalg.inv(G),p)
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    voltage = {n:v0-v[i] for i,n in enumerate(nodelist)}
    flows = {e:log(abs(f[i])+1e-10) for i,e in enumerate(edgelist)}
    subnodes = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = v0
    nx.set_node_attributes(graph,voltage,'voltage')
    nx.set_edge_attributes(graph,flows,'flow')
    return

def assign_linetype(graph):
    prim_amps = {e:2.2*exp(graph[e[0]][e[1]]['flow'])/6.3 \
                 for e in graph.edges if graph[e[0]][e[1]]['label']=='P'}
    sec_amps = {e:1.5*exp(graph[e[0]][e[1]]['flow'])/0.12 \
                for e in graph.edges if graph[e[0]][e[1]]['label']=='S'}
    
    
    edge_name = {}
    for e in graph.edges:
        # names of secondary lines
        if graph[e[0]][e[1]]['label']=='S':
            if sec_amps[e]<=95:
                edge_name[e] = 'OH_Voluta'
                r = 0.661/57.6; x = 0.033/57.6
            elif sec_amps[e]<=125:
                edge_name[e] = 'OH_Periwinkle'
                r = 0.416/57.6; x = 0.031/57.6
            elif sec_amps[e]<=165:
                edge_name[e] = 'OH_Conch'
                r = 0.261/57.6; x = 0.03/57.6
            elif sec_amps[e]<=220:
                edge_name[e] = 'OH_Neritina'
                r = 0.164/57.6; x = 0.03/57.6
            elif sec_amps[e]<=265:
                edge_name[e] = 'OH_Runcina'
                r = 0.130/57.6; x = 0.029/57.6
            else:
                edge_name[e] = 'OH_Zuzara'
                r = 0.082/57.6; x = 0.027/57.6
        
        # names of primary lines
        elif graph[e[0]][e[1]]['label']=='P':
            if prim_amps[e]<=140:
                edge_name[e] = 'OH_Swanate'
                r = 0.407/39690; x = 0.113/39690
            elif prim_amps[e]<=185:
                edge_name[e] = 'OH_Sparrow'
                r = 0.259/39690; x = 0.110/39690
            elif prim_amps[e]<=240:
                edge_name[e] = 'OH_Raven'
                r = 0.163/39690; x = 0.104/39690
            elif prim_amps[e]<=315:
                edge_name[e] = 'OH_Pegion'
                r = 0.103/39690; x = 0.0992/39690
            else:
                edge_name[e] = 'OH_Penguin'
                r = 0.0822/39690; x = 0.0964/39690
        else:
            edge_name[e] = 'OH_Penguin'
            r = 0.0822/363000; x = 0.0964/363000
        
        # Assign new resitance and reactance
        try:
            l = graph.edges[e]['length']
        except:
            l = graph.edges[e]['geo_length']
        graph.edges[e]['r'] = r * l * 1e-3
        graph.edges[e]['x'] = x * l * 1e-3
    
    # Add new edge attribute
    nx.set_edge_attributes(graph,edge_name,'type')
    return

#%% Shape file generation
def get_zipped(gdf,filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        localFile = filename
        
        gdf.to_file(filename=temp_dir, driver='ESRI Shapefile')
        
        archiveFile = shutil.make_archive(localFile, 'zip', temp_dir)
        shutil.rmtree(temp_dir)
    return

def create_shapefile(net,dest):
    nodelist = net.nodes
    d = {'node':[n for n in nodelist],
        'label':[net.nodes[n]['label'] for n in nodelist],
         'load':[net.nodes[n]['load'] for n in nodelist],
         'geometry':[Point(net.nodes[n]['cord']) for n in nodelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,dest+"nodelist")
    
    edgelist = net.edges
    d = {'label':[net.edges[e]['label'] for e in edgelist],
         'nodeA':[e[0] for e in edgelist],
         'nodeB':[e[1] for e in edgelist],
         'line_type':[net.edges[e]['type'] for e in edgelist],
         'r':[net.edges[e]['r'] for e in edgelist],
         'x':[net.edges[e]['x'] for e in edgelist],
         'length':[net.edges[e]['length'] for e in edgelist],
         'geometry':[net.edges[e]['geometry'] for e in edgelist]}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    get_zipped(gdf,dest+"edgelist")
    return