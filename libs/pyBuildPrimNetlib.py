# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
import sys
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString
import networkx as nx
import gurobipy as grb
import numpy as np


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


def read_master_graph(path,sub):
    """
    Reads the master graph from the binary/gpickle file.

    Parameters
    ----------
    path : string
        directory path for the master graph gpickle file.
    sub : string
        ID for the substation under consideration.

    Returns
    -------
    graph : networkx graph
        The master graph of road network edges with transformer nodes. The data
        associated with the nodes and edges are listed below:
            cord: node geographical coordinates
            distance: geodesic distance of nodes from the substation
            load: load at the transformer nodes
            length: geodesic length of the edges
    """
    graph = nx.read_gpickle(path+'prim-master/'+sub+'-master.gpickle')
    return graph

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


class MILP_primary:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,graph,grbpath=None):
        """
        graph: the base graph which has the list of possible edges.
        tnodes: dictionary of transformer nodes with power consumption as value.
        """
        # Get tmp path for gurobi log files
        self.tmp = grbpath+"gurobi/"
        
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
        flow_cap = 1000 # Maximum feeder capacity in kVA
        feeder_cap = int(total_cap/1000)+1 # Maximum number of feeders
        
        # Create the optimization model
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.masterTree()
        self.powerflow()
        self.radiality()
        self.flowconstraint(M=flow_cap)
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
            t_optimal = self.t.getAttr("x").tolist()
            self.optimal_edges = [e for i,e in enumerate(self.edges) if x_optimal[i]>0.8]
            self.roots = [self.nodes[ind] for i,ind in enumerate(self.rindex) if z_optimal[i]<0.2]
            self.dummy = [e for i,e in enumerate(self.edges) if t_optimal[i]>0.8]
            self.droots = [self.nodes[self.rindex[0]]]
            return    


class Primary:
    """
    Creates the primary distribuion network by solving an optimization problem.
    First the set of possible edges are identified from the links in road net-
    -work and transformer connections.
    """
    def __init__(self,subdata,path,feedcap=800,div=8):
        """
        """
        self.subdata = subdata
        self.graph = nx.Graph()
        master = read_master_graph(path,str(subdata.id))
        M = sum(nx.get_node_attributes(master,'load').values())/1e3
        self.max_mva = max([feedcap,M/div])
        self.get_partitions(master)
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
        
    def get_sub_network(self,grbpath=None):
        """
        """
        # Optimizaton problem to get the primary network
        primary = []; roots = []; tnodes = []; dummy = []; droots = []
        for nlist in list(nx.connected_components(self.graph)):
            subgraph = nx.subgraph(self.graph,list(nlist))
            M = MILP_primary(subgraph,grbpath=grbpath)
            print("\n\n\n")
            primary += M.optimal_edges
            roots += M.roots
            tnodes += M.tnodes
            dummy += M.dummy
            droots += M.droots
        
        # Add the first edge between substation and nearest road node
        hvlines = [(self.subdata.id,r) for r in roots]
        dhvlines = [(self.subdata.id,r) for r in droots]
        
        # Create the network with data as attributes
        dist_net = self.create_network(primary,hvlines)
        dummy_net = self.create_network(dummy,dhvlines)
        return dist_net,dummy_net
    
    
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
        feed_path = nx.get_node_attributes(self.graph,'feedpath')
        edge_geom = {}
        edge_label = {}
        edge_r = {}
        edge_x = {}
        glength = {}
        for e in list(dist_net.edges()):
            length = 1e-3 * MeasureDistance(nodepos[e[0]],nodepos[e[1]])
            length = length if length != 0.0 else 1e-12
            if e in primary or (e[1],e[0]) in primary:
                edge_geom[e] = Link((nodepos[e[0]],nodepos[e[1]]))
                edge_label[e] = 'P'
                edge_r[e] = 0.8625/39690 * length
                edge_x[e] = 0.4154/39690 * length
                glength[e] = edge_geom[e].geod_length
            elif e in hvlines or (e[1],e[0]) in hvlines:
                rnode = e[0] if e[1]==self.subdata.id else e[1]
                path_cords = [self.subdata.cord]+\
                                   [nodepos[nd] for nd in feed_path[rnode]]
                edge_geom[e] = Link(path_cords)
                edge_label[e] = 'E'
                edge_r[e] = 1e-12 * length
                edge_x[e] = 1e-12 * length
                glength[e] = edge_geom[e].geod_length
            else:
                edge_geom[e] = Link((nodepos[e[0]],nodepos[e[1]]))
                edge_label[e] = 'S'
                edge_r[e] = 0.81508/57.6 * length
                edge_x[e] = 0.3496/57.6 * length
                glength[e] = edge_geom[e].geod_length
        nx.set_edge_attributes(dist_net, edge_geom, 'geometry')
        nx.set_edge_attributes(dist_net, edge_label, 'label')
        nx.set_edge_attributes(dist_net, edge_r, 'r')
        nx.set_edge_attributes(dist_net, edge_x, 'x')
        nx.set_edge_attributes(dist_net,glength,'geo_length')
        return dist_net
    

