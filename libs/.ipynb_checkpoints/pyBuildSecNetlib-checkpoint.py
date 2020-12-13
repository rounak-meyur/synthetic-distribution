# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
import sys,os
from geographiclib.geodesic import Geodesic
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from shapely.geometry import LineString,MultiPoint,LinearRing,Point
from collections import defaultdict
from pyqtree import Index
import gurobipy as grb
import datetime


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
        xmin,ymin = np.min(np.array(list(road.cord.values())),axis=0)
        xmax,ymax = np.max(np.array(list(road.cord.values())),axis=0)
        bbox = (xmin,ymin,xmax,ymax)
    
        # keep track of lines so we can recover them later
        all_link = list(road.graph.edges())
        self.lines = []
    
        # initialize the quadtree index
        self.idx = Index(bbox)
        
        # add edge bounding boxes to the index
        for i, path in enumerate(all_link):
            # create line geometry
            line = road.links[path]['geometry'] if path in road.links \
                else road.links[(path[1],path[0])]['geometry']
        
            # bounding boxes, with padding
            x1, y1, x2, y2 = line.bounds
            bounds = x1-radius, y1-radius, x2+radius, y2+radius
        
            # add to quadtree
            self.idx.insert(i, bounds)
        
            # save the line for later use
            self.lines.append((line, bounds, path))
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
        return
    

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

        
class SecNet:
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
        self.link_to_home = groups(home_to_link)
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
    
    def get_nodes(self,link,minsep=50):
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
        link_line = Link(self.links[link]['geometry']) if link in self.links \
                    else Link(self.links[(link[1],link[0])]['geometry'])
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
    
    
    def create_dummy_graph(self,link,minsep,penalty,heuristic=None):
        """
        Creates the base network to carry out the optimization problem. The base graph
        may be a Delaunay graph or a full graph depending on the size of the problem.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
                heuristic: join transformers to nearest few nodes given by heuristic.
                        Used to create the dummy graph
        Outputs:graph: the generated base graph also called the dummy graph
                transformers: list of points along link which are probable locations 
                of transformers.
        """
        sides = self.separate_side(link)
        home_pts,transformers = self.get_nodes(link,minsep=minsep)
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
    
    def generate_optimal_topology(self,link,minsep=50,penalty=0.5,
                                  heuristic=None,hops=4,tsfr_max=25,path=None):
        """
        Calls the MILP problem and solves it using gurobi solver.
        
        Inputs: link: road link of interest for which the problem is solved.
                minsep: minimum separation in meters between the transformers.
                penalty: penalty factor for crossing the link.
                heuristic: join transformers to nearest few nodes given by heuristic.
                          Used to create the dummy graph
        Outputs:forest: the generated forest graph which is the secondary network
                roots: list of points along link which are actual locations of 
                transformers.
        """
        graph,roots = self.create_dummy_graph(link,minsep,penalty,
                                              heuristic=heuristic)
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