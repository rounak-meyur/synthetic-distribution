# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:01:21 2019

@author: rounak
"""
import sys
from shapely.geometry import LineString
import networkx as nx
import gurobipy as grb
import numpy as np
from pyGeometrylib import Link
from pyExtractDatalib import GetPrimRoad
from pyMiscUtilslib import get_secnet


#%% Functions
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
            self.optimal_edges = [e for i,e in enumerate(self.edges) if x_optimal[i]>0.8]
            self.roots = [self.nodes[ind] for i,ind in enumerate(self.rindex) if z_optimal[i]<0.2]
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
        
        # Update master graph with substation distance data
        master = GetPrimRoad(path,str(subdata["id"]))
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
        
    def get_sub_network(self,secpath=None,inppath=None,grbpath=None):
        """
        """
        # Optimizaton problem to get the primary network
        primary = []; roots = []; tnodes = []
        for nlist in list(nx.connected_components(self.graph)):
            subgraph = nx.subgraph(self.graph,list(nlist))
            M = MILP_primary(subgraph,grbpath=grbpath)
            print("\n\n\n")
            primary += M.optimal_edges
            roots += M.roots
            tnodes += M.tnodes
        
        # Add the first edge between substation and nearest road node
        hvlines = [(self.subdata["id"],r) for r in roots]
        
        # Create the network with data as attributes
        net = self.create_network(primary,hvlines)
        
        # Update secondary network
        net = get_secnet(net,secpath,inppath)
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
    


