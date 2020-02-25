# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 07:40:36 2019

Author: Rounak Meyur
Description: This library contains functions and classes to formulate an MILP
required to solve the optimal network construction problem, scheduling electric
vehicles etc.
"""
import sys
import networkx as nx
import gurobipy as grb
import numpy as np

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
    return


#%% Secondary Network Creation
class MILP_secondary:
    """
    """
    def __init__(self,graph,roots):
        """
        """
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
        print(self.p)
        self.model = grb.Model(name="Get Spiders")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__radiality()
        self.__heuristic()
        self.__powerflow()
        self.__objective()
        self.model.write("secondary.lp")
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
        self.v = self.model.addMVar(len(self.nodes),
                                    vtype=grb.GRB.CONTINUOUS,
                                    lb=0.95,ub=1.00,name='v')
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
        expr = r*np.diag(self.l)@self.f
        self.model.addConstr(self.A.T@self.v - expr - 0.1*(1-self.x) <= 0,name='va')
        self.model.addConstr(self.A.T@self.v - expr + 0.1*(1-self.x) >= 0,name='vb')
        for i in self.tindex:
            self.model.addConstr(self.v[i]==1,name="voltage")
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
        grb.setParam('OutputFlag', 1)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open('gurobi.log', 'w')
        
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


#%% Primary Network Creation

class MILP_primary:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,graph,tnodes,flow=400,feeder=10):
        """
        graph: the base graph which has the list of possible edges.
        tnodes: dictionary of transformer nodes with power consumption as value.
        """
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        self.tindex = [i for i,n in enumerate(self.nodes) if n in tnodes]
        self.rindex = [i for i,n in enumerate(self.nodes) if n not in tnodes]
        LENGTH = nx.get_edge_attributes(graph,name='length')
        DIST = nx.get_node_attributes(graph,name='distance')
        
        self.d = [1e-3*DIST[self.nodes[i]] for i in self.rindex]
        self.A = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        self.c = [1e-3*LENGTH[self.edges[i]] for i in range(len(self.edges))]
        self.p = np.array([1e-3*tnodes[self.nodes[i]] for i in self.tindex])
        
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__powerflow()
        self.__radiality()
        self.__flowconstraint(M=flow)
        self.__connectivity()
        self.__limit_feeder(M=feeder)
        self.__objective()
        self.model.write("primary.lp")
        self.solve()
        return
        
    
    def __variables(self):
        """
        Create variable initialization for MILP such that x is binary, y is a 
        binary variable and z is a continuous variable. 
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
        self.model.update()
        return
    
    def __powerflow(self,r=0.8625/39690,M=0.15):
        """
        """
        print("Setting up power flow constraints")
        expr = r*np.diag(self.c)@self.f
        self.model.addConstr(self.A.T@self.v - expr - M*(1-self.x) <= 0)
        self.model.addConstr(self.A.T@self.v - expr + M*(1-self.x) >= 0)
        self.model.addConstr(1-self.v[self.rindex] <= self.z)
        self.model.update()
        return
    
    def __radiality(self):
        """
        """
        print("Setting up radiality constraints")
        self.model.addConstr(
                self.x.sum() == len(self.nodes) - 2*len(self.rindex) + \
                self.y.sum()+self.z.sum())
        self.model.update()
        return
    
    def __connectivity(self):
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
    
    def __flowconstraint(self,M=400):
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
    
    def __limit_feeder(self,M=10):
        """
        """
        print("Setting up constraint for number of feeders")
        self.model.addConstr(len(self.rindex)-self.z.sum() <= M)
        self.model.update()
        return
    
    def __objective(self):
        """
        """
        print("Setting up objective function")
        self.model.setObjective(
                np.array(self.c)@self.x - np.array(self.d)@self.z + sum(self.d))
                #np.array(self.c)@self.x + len(self.rindex) - self.z.sum())
        return
    
    def solve(self):
        """
        """
        # Turn off display and heuristics
        grb.setParam('OutputFlag', 1)
        grb.setParam('Heuristics', 1)
        
        # Open log file
        logfile = open('gurobi.log', 'w')
        
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
            self.optimal_edges = [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]
            self.roots = [self.nodes[ind] for i,ind in enumerate(self.rindex) if z_optimal[i]<0.5]
            return
        
        
