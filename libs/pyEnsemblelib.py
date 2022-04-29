# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:06:39 2022

Author: Rounak Meyur
Description: Methods to create an ensemble of networks and analyze them.
"""

import gurobipy as grb
import networkx as nx
import numpy as np

def create_dummy(road,prim,sub):
    # Get list of edges to select edge to be deleted
    edgelist = [e for e in prim.edges if prim.nodes[e[0]]['label']=='T' \
                and prim.nodes[e[1]]['label']=='T']
    # Create new dummy graph
    while(1):
        rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
        net = road.__class__()
        net.add_edges_from(road.edges)
        net.remove_edge(*rand_edge)
        # Check if connected components are formed
        if nx.number_connected_components(net)==1:
            print("Edge deleted:",rand_edge)
            break
        else:
            print("Retrying... to select a valid edge!!!")
    
    # Add the attributes required for optimization
    for n in net.nodes:
        net.nodes[n]['cord'] = road.nodes[n]['cord']
        net.nodes[n]['label'] = road.nodes[n]['label']
        net.nodes[n]['load'] = road.nodes[n]['load']
        
        # Store binary variables as attributes
        if road.nodes[n]['label'] == 'T':
            net.nodes[n]['y'] = 1
            net.nodes[n]['z'] = 1
        else:
            # define binary variable 'y' for each road node
            # y=1 for included nodes, y=0 for not included nodes
            if n in prim:
                net.nodes[n]['y'] = 1
            else:
                net.nodes[n]['y'] = 0
            # define binary variable 'z' for each road node
            # z=1 for non-root nodes, z=0 for root nodes
            if n in nx.neighbors(prim, sub):
                net.nodes[n]['z'] = 0
            else:
                net.nodes[n]['z'] = 1
    
    # define binary variable 'x' for each road edge
    # x=1 for included edges, x=0 for not included edges
    for e in net.edges:
        net.edges[e]['length'] = road.edges[e]['length']
        if e in prim.edges:
            net[e[0]][e[1]]['x'] = 1
        else:
            net[e[0]][e[1]]['x'] = 0
    return net


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


#%% Altered MILP
class MILP_primstruct:
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
        self.edges = list(road.edges)
        self.nodes = list(road.nodes)
        
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
        self.restrict(road)
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
    
    def restrict(self,dummy):
        for i,e in enumerate(self.edges):
            if dummy.edges[e]['x'] == 1: 
                self.model.addConstr(self.x[i] == 1)
            
        for i,n in enumerate(self.rindex):
            if dummy.nodes[self.nodes[n]]['label'] == 'T':
                print("WARNING!!!")
            if dummy.nodes[self.nodes[n]]['y'] == 1:
                self.model.addConstr(self.y[i] == 1)
            if dummy.nodes[self.nodes[n]]['z'] == 0:
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
        logfile = open(self.tmp+'ens_gurobi.log', 'w')
        
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