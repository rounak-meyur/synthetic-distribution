# -*- coding: utf-8 -*-
"""
Created on Tue Sept 15 10:19:05 2021

@author: Rounak Meyur
Description: Library of functions for miscellaneous applications
"""


from collections import defaultdict
from shapely.geometry import LineString
import networkx as nx
import numpy as np
from math import log,exp
from pyGeometrylib import Link
import gurobipy as grb
from pyExtractDatalib import GetSecnet,GetHomes


#%% Functions on data structures
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


#%% Functions on total distribution network
def get_load(graph):
    """
    Get a dictionary of loads corresponding to each transformer node in the 
    graph. This data will be used as a proxy for the secondary network.

    Parameters
    ----------
    graph : networkx Graph
        The optimal distribution network comprising of primary and secondary.

    Returns
    -------
    LOAD : dictionary of loads
        The load supplied by each transformer node.
    """
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H']
    sub = [n for n in graph if graph.nodes[n]['label']=='S'][0]
    res2tsfr = {h:[n for n in nx.shortest_path(graph,h,sub) if n in tnodes][0] \
                for h in hnodes}
    tsfr2res = groups(res2tsfr)
    LOAD = {t:sum([graph.nodes[n]['load'] for n in tsfr2res[t]]) for t in tnodes}
    return LOAD

def get_secnet(graph,secpath,homepath):
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    fislist = list(set([str(x)[2:5] for x in tnodes]))
    
    # Get all secondary networks
    secnet = nx.Graph()
    hcord = {}; hload = {}
    for fis in fislist:
        g = GetSecnet(secpath, fis)
        secnet = nx.compose(secnet,g)
        h = GetHomes(homepath,fis)
        hcord.update(h.cord); hload.update(h.average)
    
    # Extract only associated secondaries
    sec_graph = nx.Graph()
    comps = list(nx.connected_components(secnet))
    for t in tnodes:
        comp = [c for c in comps if t in c][0]
        g = secnet.subgraph(comp)
        sec_graph = nx.compose(sec_graph,g)
    
    # Add the secondary network to the primary network
    graph = nx.compose(graph,sec_graph)
    
    # Add new node attributes
    hnodes = [n for n in sec_graph if n not in tnodes]
    for n in hnodes:
        graph.nodes[n]['cord'] = secnet.nodes[n]['cord']
        graph.nodes[n]['label'] = 'H'
        if n in hload:
            graph.nodes[n]['load'] = hload[n]
        else:
            # use a sample house as the load
            graph.nodes[n]['load'] = hload[[n for n in hload][0]]
    
    for e in graph.edges:
        if e in sec_graph.edges:
            graph.edges[e]['geometry'] = LineString((graph.nodes[e[0]]["cord"],
                                                     graph.nodes[e[1]]["cord"]))
            graph.edges[e]['length'] = Link(graph.edges[e]['geometry']).geod_length
            graph.edges[e]['label'] = 'S'
            graph.edges[e]['r'] = 0.81508/57.6 * graph.edges[e]['length']
            graph.edges[e]['x'] = 0.34960/57.6 * graph.edges[e]['length']
    return graph

def powerflow(graph,v0=1.0):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    # Pre-processing to rectify incorrect code
    hv_lines = [e for e in graph.edges if graph.edges[e]['label']=='E']
    for e in hv_lines:
        length = graph.edges[e]['length']
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
        graph.edges[e]['r'] = r * graph.edges[e]['length'] * 1e-3
        graph.edges[e]['x'] = x * graph.edges[e]['length'] * 1e-3
    
    # Add new edge attribute
    nx.set_edge_attributes(graph,edge_name,'type')
    return


#%% Creating variant networks
def create_variant_network(graph,road,new_prim,rem_edges):
    """
    Alters the input network to create a variant version of it. It is used to
    create variant networks in the ensemble of networks

    Parameters
    ----------
    graph : networkx graph
        input distribution network.
    road : networkx graph
        underlying road network.
    new_prim : list of edge tuples
        list of primary network edges in the new network.
    rem_edges : list of edge tuples
        list of primary network edges in the old network which are to be removed.

    Returns
    -------
    None.
    Essentially returns the altered graph.

    """
    new_edges = [e for e in new_prim if e not in graph.edges]
    nodelist = list(graph.nodes)
    for e in rem_edges:
        graph.remove_edge(*e)
    graph.add_edges_from(new_edges)
    
    # node attributes
    for n in graph:
        if n not in nodelist:
            graph.nodes[n]['cord'] = road.nodes[n]['cord']
            graph.nodes[n]['load'] = 0.0
            graph.nodes[n]['label'] = 'R'
    
    # edge attributes
    for e in new_edges:
        graph.edges[e]['geometry'] = LineString((graph.nodes[e[0]]['cord'],
                                                 graph.nodes[e[1]]['cord']))
        graph.edges[e]['geo_length'] = Link(graph.edges[e]['geometry']).geod_length
        graph.edges[e]['label'] = 'P'
        length = graph.edges[e]['geo_length'] if graph.edges[e]['geo_length'] != 0.0 else 1e-12
        graph.edges[e]['r'] = 0.8625/39690 * length
        graph.edges[e]['x'] = 0.4154/39690 * length
    
    # Run powerflow
    powerflow(graph)
    assign_linetype(graph)
    return


#%% Restricted MILP for variant networks
def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if(time>60 and abs(objbst - objbnd) < 0.0025 * (1.0 + abs(objbst))):
            print('Stop early - 0.25% gap achieved time exceeds 1 minute')
            model.terminate()
        elif(time>300 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1.00% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>480 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5.00% gap achieved time exceeds 8 minutes')
            model.terminate()
        elif(time>600 and objbst < grb.GRB.INFINITY):
            print('Stop early - feasible solution found time exceeds 10 minutes')
            model.terminate()
        elif(time>600 and objbst == grb.GRB.INFINITY):
            print('Stop early - no solution found in 10 minutes')
            model.terminate()
    return

class reduced_MILP_primary:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,road,grbpath=None):
        """
        """
        # Get tmp path for gurobi log files
        self.tmp = grbpath+"gurobi/"
        
        # Get data from graph
        self.edges = list(road.edges())
        self.nodes = list(road.nodes())
        
        self.tindex = [i for i,n in enumerate(self.nodes) if road.nodes[n]['label']=='T']
        self.rindex = [i for i,n in enumerate(self.nodes) if road.nodes[n]['label']=='R']
        
        # Vectorize the data for matrix computation
        self.A = nx.incidence_matrix(road,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(road,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        self.c = [1e-3*road.edges[self.edges[i]]['length'] \
                  for i in range(len(self.edges))]
        self.p = np.array([1e-3*road.nodes[self.nodes[i]]['load'] for i in self.tindex])
        
        
        # Create the optimization model
        self.model = grb.Model(name="Get Variant Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.masterTree()
        self.power_flow()
        self.radiality()
        self.flowconstraint(M=1000)
        self.connectivity()
        self.objective()
        self.model.write(self.tmp+"variant.lp")
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
    
    def power_flow(self,r=0.8625/39690,M=0.15):
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
        return
    
    def restrict(self,sub,net,rem_edges):
        reg_nodes = list(nx.neighbors(net,sub))
        for i,e in enumerate(self.edges):
            if e in rem_edges or (e[1],e[0]) in rem_edges:
                self.model.addConstr(self.x[i] == 0)
                print(i)
            elif e in net.edges: 
                self.model.addConstr(self.x[i] == 1)
            
        for i,n in enumerate(self.rindex):
            if self.nodes[n] in net: 
                self.model.addConstr(self.y[i] == 1)
            if self.nodes[n] in reg_nodes:
                self.model.addConstr(self.z[i] == 0)
            else:
                self.model.addConstr(self.z[i] == 1)
        return
        
    def objective(self):
        """
        """
        print("Setting up objective function")
        self.model.setObjective(np.array(self.c)@self.x)
        return
    
    def solve(self):
        """
        """
        # Turn off display and heuristics
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(self.tmp+'var_gurobi.log', 'w')
        
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
            return []
        else:
            print('Solution found, objective = %g' % self.model.ObjVal)
            x_optimal = self.x.getAttr("x").tolist()
            return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.8]