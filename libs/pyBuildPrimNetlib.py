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
from math import log,exp
from pyGeometrylib import Link


#%% Functions
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
    graph = nx.read_gpickle(path+sub+'-master.gpickle')
    print("Master graph read from stored gpickle file")
    return graph

def powerflow(graph):
    """
    Checks power flow solution and save dictionary of voltages.
    """
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
    voltage = {h:1.0-v[i] for i,h in enumerate(nodelist)}
    flows = {e:log(abs(f[i])) for i,e in enumerate(edgelist)}
    subnodes = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = 1.0
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
            r = 1e-10; x = 1e-10
        
        # Assign new resitance and reactance
        graph.edges[e]['r'] = r * graph.edges[e]['geo_length'] * 1e-3
        graph.edges[e]['x'] = x * graph.edges[e]['geo_length'] * 1e-3
    
    # Add new edge attribute
    nx.set_edge_attributes(graph,edge_name,'type')
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
        master = read_master_graph(path,str(subdata.id))
        M = sum(nx.get_node_attributes(master,'load').values())/1e3
        self.secnet = nx.get_node_attributes(master,'secnet')
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
        
    def get_sub_network(self,grbpath=None):
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
        hvlines = [(self.subdata.id,r) for r in roots]
        
        # Create the network with data as attributes
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
        # Get the secondary network associated with transformers in partition
        sec_net = nx.Graph()
        for t in self.secnet:
            sec_net = nx.compose(sec_net,self.secnet[t])
        secondary = list(sec_net.edges())
        
        secpos = nx.get_node_attributes(sec_net,'cord')
        seclabel = nx.get_node_attributes(sec_net,'label')
        secload = nx.get_node_attributes(sec_net,'resload')
        
        # Combine both networks and update labels
        dist_net = nx.Graph()
        dist_net.add_edges_from(hvlines+primary+secondary)
        
        # Add coordinate attributes to the nodes of the network.
        nodepos = nx.get_node_attributes(self.graph,'cord')
        nodepos[self.subdata.id] = self.subdata.cord
        nodepos.update(secpos)
        nx.set_node_attributes(dist_net,nodepos,'cord')
        
        # Label nodes of the created network
        node_label = nx.get_node_attributes(self.graph,'label')
        node_label[self.subdata.id] = 'S'
        node_label.update(seclabel)
        nx.set_node_attributes(dist_net, node_label, 'label')
        
        # Load at nodes
        nodeload = {n:secload[n] if n in secload else 0.0 \
                    for n in dist_net.nodes()}
        nx.set_node_attributes(dist_net,nodeload,'load')
        
        # Label edges of the created network
        feed_path = nx.get_node_attributes(self.graph,'feedpath')
        for e in list(dist_net.edges()):
            if e in primary or (e[1],e[0]) in primary:
                dist_net.edges[e]['geometry'] = LineString((nodepos[e[0]],nodepos[e[1]]))
                dist_net.edges[e]['geo_length'] = Link(dist_net.edges[e]['geometry']).geod_length
                dist_net.edges[e]['label'] = 'P'
                dist_net.edges[e]['r'] = 0.8625/39690 * dist_net.edges[e]['geo_length']
                dist_net.edges[e]['x'] = 0.4154/39690 * dist_net.edges[e]['geo_length']
            elif e in hvlines or (e[1],e[0]) in hvlines:
                rnode = e[0] if e[1]==self.subdata.id else e[1]
                path_cords = [self.subdata.cord]+\
                                   [nodepos[nd] for nd in feed_path[rnode]]
                dist_net.edges[e]['geometry'] = LineString(path_cords)
                dist_net.edges[e]['geo_length'] = Link(dist_net.edges[e]['geometry']).geod_length
                dist_net.edges[e]['label'] = 'E'
                dist_net.edges[e]['r'] = 1e-12 * dist_net.edges[e]['geo_length']
                dist_net.edges[e]['x'] = 1e-12 * dist_net.edges[e]['geo_length']
            else:
                dist_net.edges[e]['geometry'] = LineString((nodepos[e[0]],nodepos[e[1]]))
                dist_net.edges[e]['geo_length'] = Link(dist_net.edges[e]['geometry']).geod_length
                dist_net.edges[e]['label'] = 'S'
                dist_net.edges[e]['r'] = 0.81508/57.6 * dist_net.edges[e]['geo_length']
                dist_net.edges[e]['x'] = 0.34960/57.6 * dist_net.edges[e]['geo_length']
        return dist_net
    


