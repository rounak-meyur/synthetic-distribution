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
import numpy as np
import gurobipy as grb

#%% Function for callback
def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if time>300 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst)):
            print('Stop early - 5% gap achieved')
            model.terminate()
    return


#%% Synthetic secondary distribution network
class MILP_noPF:
    """
    """
    def __init__(self,graph,roots,max_deg,max_hop):
        """
        """
        NODES = [i for i,node in enumerate(graph.nodes()) \
                          if node not in roots]
        self.edges = list(graph.edges())
        EDGES = range(len(self.edges))
        self.A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                                     edgelist=list(graph.edges()),oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                                     edgelist=list(graph.edges()),oriented=False)
        COST = nx.get_edge_attributes(graph,name='cost')
        
        self.c = [COST[e] for e in self.edges]
        self.d = [max_deg]*graph.number_of_nodes()
        self.h = [max_hop]*graph.number_of_edges()
        
        
        self.model = grb.Model(name="Get Spiders")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables(EDGES)
        self.__radiality(EDGES,NODES)
        self.__connectivity(EDGES,NODES)
        self.__degree(EDGES,NODES)
        self.__hops(EDGES)
        self.__objective(EDGES)
        self.optimal_edges = self.__solve()
        return
    
    def __variables(self,iter_edge):
        """
        Create variable initialization for MILP such that x is binary, f is a 
        continuous variable and z is product of them. This type of initialization
        is required to apply the Big M trick
        """
        self.x = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="x_{0}".format(i)) for i in iter_edge}
        self.z = {i:self.model.addVar(vtype=grb.GRB.CONTINUOUS,lb=-grb.GRB.INFINITY,
                                      name="z_{0}".format(i)) for i in iter_edge}
        return
    
    def __radiality(self,iter_edge,iter_homes):
        """
        Radiality constraints in the form of linear problem are defined:
            1. Number of edges of a forest is number of nodes except root
            2. Each connected component of forest should have a transformer
        """
        nhomes = len(iter_homes)
        self.model.addLConstr(lhs=grb.quicksum(self.x[j] for j in iter_edge),
                sense=grb.GRB.EQUAL,rhs=nhomes,
                name="radiality constraint")
        return
    
    def __connectivity(self,iter_edge,iter_homes):
        """
        """
        for i in iter_homes:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.A[i,j]*self.z[j] for j in iter_edge),
                    sense=grb.GRB.EQUAL,rhs=1.0,
                    name="node{0} demand constraint".format(i))
    
    def __degree(self,iter_edge,iter_homes):
        """
        """
        for i in iter_homes:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.I[i,j]*self.x[j] for j in iter_edge),
                    sense=grb.GRB.LESS_EQUAL,rhs=self.d[i],
                    name="node{0} degree constraint".format(i))
        return
    
    def __hops(self,iter_edge):
        """
        """
        for j in iter_edge:
            self.model.addLConstr(
                    lhs = self.z[j]-self.x[j]*self.h[j],
                    sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="edge{0} hop constraint 1".format(j))
            self.model.addLConstr(
                    lhs = self.z[j]+self.x[j]*self.h[j],
                    sense=grb.GRB.GREATER_EQUAL,rhs=0,
                    name="edge{0} hop constraint 1".format(j))
        return
        
    
    def __objective(self,iter_edge):
        """
        """
        self.model.setObjective(grb.quicksum(self.x[j]*self.c[j]\
                                           for j in iter_edge))
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
            x_optimal = [self.x[i].getAttr("x") for i in range(len(self.x))]
            return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]
    
    




class MILP_withPF:
    """
    """
    def __init__(self,graph,roots):
        """
        """
        NODES = range(len(list(graph.nodes())))
        HOMES = [i for i,node in enumerate(graph.nodes()) \
                          if node not in roots]
        self.edges = list(graph.edges())
        EDGES = range(len(self.edges))
        self.A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                                     edgelist=list(graph.edges()),oriented=True)
        COST = nx.get_edge_attributes(graph,name='cost')
        LENGTH = nx.get_edge_attributes(graph,name='length')
        LOAD = nx.get_node_attributes(graph,name='load')
        self.c = [COST[e] for e in self.edges]
        self.l = [LENGTH[e]*0.001 for e in self.edges]
        self.p = [LOAD[n]*0.001 for n in list(graph.nodes())]
                
        self.model = grb.Model(name="Get Spiders")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables(EDGES,NODES)
        self.__VoltageConstraints(NODES,HOMES)
        self.__radiality(EDGES,HOMES)
        self.__connectivity(EDGES,HOMES)
        self.__McCormick(EDGES,NODES)
        self.__objective(EDGES)
        self.optimal_edges = self.__solve()
        return
    
    def __variables(self,iter_edge,iter_node):
        """
        Create variable initialization for MILP such that x is binary, f is a 
        continuous variable and z is product of them. This type of initialization
        is required to apply the Big M trick
        """
        self.x = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="x_{0}".format(i)) for i in iter_edge}
        self.z = {i:self.model.addVar(vtype=grb.GRB.CONTINUOUS,lb=-grb.GRB.INFINITY,
                                      name="z_{0}".format(i)) for i in iter_edge}
        self.v = {i:self.model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0.9,ub=1.1,
                                      name="z_{0}".format(i)) for i in iter_node}
        return
    
    def __radiality(self,iter_edge,iter_homes):
        """
        Radiality constraints in the form of linear problem are defined:
            1. Number of edges of a forest is number of nodes except root
            2. Each connected component of forest should have a transformer
        """
        nhomes = len(iter_homes)
        self.model.addLConstr(lhs=grb.quicksum(self.x[j] for j in iter_edge),
                sense=grb.GRB.EQUAL,rhs=nhomes,
                name="radiality constraint")
        return
    
    def __connectivity(self,iter_edge,iter_homes):
        """
        """
        for i in iter_homes:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.A[i,j]*self.z[j] for j in iter_edge),
                    sense=grb.GRB.EQUAL,rhs=-self.p[i],
                    name="node{0} demand constraint".format(i))
        
    
    def __McCormick(self,iter_edge,iter_nodes,M=50,r=0.81508/57.6):
        """
        """
        for i in iter_edge:
            self.model.addLConstr(
                    lhs=self.z[i]-(M*self.x[i]),sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="Big M-constraint11 var z{0}".format(i))
            self.model.addLConstr(
                    lhs=self.z[i]+(M*self.x[i]),sense=grb.GRB.GREATER_EQUAL,
                    rhs=0,name="Big M-constraint12 var z{0}".format(i))
            self.model.addLConstr(
                    lhs=self.z[i]-grb.quicksum(self.A[j,i]*self.v[j] \
                              for j in iter_nodes)/(r*self.l[i])\
                              +M*self.x[i],
                    sense=grb.GRB.LESS_EQUAL,rhs=M,
                    name="Voltage_drop_constraint_1{0}".format(i))
            self.model.addLConstr(
                    lhs=self.z[i]-grb.quicksum(self.A[j,i]*self.v[j] \
                              for j in iter_nodes)/(r*self.l[i])\
                              -M*self.x[i],
                    sense=grb.GRB.GREATER_EQUAL,rhs=-M,
                    name="Voltage_drop_constraint_2{0}".format(i))
        return
    
    
    def __VoltageConstraints(self,iter_nodes,iter_homes):
        """
        """
        roots = [i for i in iter_nodes if i not in iter_homes]
        for i in roots:
            self.model.addLConstr(lhs=self.v[i],
                sense=grb.GRB.EQUAL,rhs=1.0,
                name="radiality constraint")
        return
    
    def __objective(self,iter_edge):
        """
        """
        self.model.setObjective(grb.quicksum(self.x[j]*self.c[j]\
                                           for j in iter_edge))
        return
    
    def __solve(self):
        """
        """
        self.model.optimize()
        x_optimal = [self.x[i].getAttr("x") for i in range(len(self.x))]
        v_opt = [self.v[i].getAttr("x") for i in range(len(self.v))]
        print (v_opt)
        return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]


#%% Primary Network Creation
class MILP_primary:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,graph,tnodes,snodes):
        """
        graph: the base graph which has the list of possible edges.
        rnodes: the set of road network nodes which can only act as transfer nodes
        connecting other local transformer nodes.
        snodes: the set of road network nodes which are nearest to the substations
        these can be considered as roots of the trees in the generated forest.
        """
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        self.sindex = [i for i,n in enumerate(self.nodes) if n in snodes]
        self.tindex = [i for i,n in enumerate(self.nodes) if n in tnodes]
        self.rindex = [i for i,n in enumerate(self.nodes) if n not in tnodes+snodes]
        LENGTH = nx.get_edge_attributes(graph,name='length')
        
        self.A = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        self.c = [LENGTH[e] for e in self.edges]
        
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__connectivity()
        self.__radiality()
        self.__flowconstraint()
        self.__objective()
        self.optimal_edges = self.solve()
        return
    
    def __variables(self):
        """
        Create variable initialization for MILP such that x is binary, y is a 
        binary variable and z is a continuous variable. 
        """
        self.x = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="x_{0}".format(i)) \
                    for i in range(len(self.edges))}
        self.y = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="y_{0}".format(i)) \
                    for i in range(len(self.rindex))}
        self.z = {i:self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      lb=-grb.GRB.INFINITY,
                                      name="z_{0}".format(i)) \
                    for i in range(len(self.edges))}
        return
    
    def __connectivity(self):
        """
        """
        for j,ind in enumerate(self.rindex):
            x_index = np.where(self.I.toarray()[ind,:]==1)[0]
            for i in x_index:
                self.model.addLConstr(lhs=self.x[i],sense=grb.GRB.LESS_EQUAL,
                                      rhs=self.y[j],
                                      name="connectivity_{0}_{1}".format(i,j))
        
        for j,ind in enumerate(self.rindex):
            self.model.addLConstr(
                    lhs=grb.quicksum(self.I[ind,i]*self.x[i] \
                                     for i in range(len(self.edges))),
                    sense=grb.GRB.GREATER_EQUAL,rhs=2*self.y[j],
                    name="node{0} degree constraint".format(j))
        return
    
    def __flowconstraint(self,M=1000):
        """
        """
        for i in self.tindex:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.A[i,j]*self.z[j] \
                                     for j in range(len(self.edges))),
                    sense=grb.GRB.EQUAL,rhs=1,
                    name="node{0} flow constraint".format(i))
        for i in self.rindex:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.A[i,j]*self.z[j] \
                                     for j in range(len(self.edges))),
                    sense=grb.GRB.EQUAL,rhs=0,
                    name="node{0} zero injection".format(i))
        for j in range(len(self.edges)):
            self.model.addLConstr(
                    lhs = self.z[j]-self.x[j]*M,
                    sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 1".format(j))
            self.model.addLConstr(
                    lhs = self.z[j]+self.x[j]*M,
                    sense=grb.GRB.GREATER_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 1".format(j))
        return
    
    def __radiality(self):
        """
        """
        self.model.addLConstr(
                lhs=grb.quicksum(self.x[j] for j in range(len(self.edges))),
                sense=grb.GRB.EQUAL,
                rhs=-grb.quicksum((1-self.y[j]) for j in range(len(self.rindex)))\
                +len(self.nodes)-len(self.sindex),
                name="radiality constraint")
        return
    
    def __objective(self):
        """
        """
        self.model.setObjective(grb.quicksum(self.x[j]*self.c[j]\
                                           for j in range(len(self.edges))))
        return
    
    def solve(self):
        """
        """
        self.model.optimize()
        x_optimal = [self.x[i].getAttr("x") for i in range(len(self.x))]
        return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]
        
#%% Electric Vehicle Scheduling Problem
class MILP_EV:
    """
    """
    def __init__(self,s,a,c,nold,t,pmax=19.2,pmin=3.3,spot_max=30,
                 tsfr_max=600.0,T=10):
        """
        """
        # Input vectors,scalars
        self.T = T
        self.t = t
        self.s = s
        self.a = a
        self.c = c
        # Constraint scalars
        self.tsfr = tsfr_max
        self.spots = spot_max
        self.pmax = pmax
        self.pmin = pmin
        # Iterable variables
        self.iter_EV = range(len(s))
        self.iter_trem = range(T-t)
        self.iter_g3 = range(nold)
        self.iter_g12 = range(nold,len(s))
        # Optimization problem
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MAXIMIZE
        self.__variables()
        self.__spotconstraint()
        self.__tsfrconstraint()
        self.__chargingconstraint()
        self.__selectionconstraint()
        self.__selectg3()
        self.__availabilityconstraint()
        self.__objective()
        self.solve()
        return
    
    def __variables(self):
        """
        """
        self.W = {(i,j):self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="W_{0}_{1}".format(i,j)) \
                    for i in self.iter_EV for j in self.iter_trem}
        self.P = {(i,j):self.model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,
                                      name="P_{0}_{1}".format(i,j)) \
                    for i in self.iter_EV for j in self.iter_trem}
        self.u = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="u_{0}".format(i))\
                    for i in self.iter_EV}
        return
    
    def __spotconstraint(self):
        """
        """
        for j in self.iter_trem:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.W[i,j] for i in self.iter_EV),
                    sense=grb.GRB.LESS_EQUAL,rhs=self.spots,
                    name="Spot_constraint_time_{0}".format(j))
        return
    
    def __tsfrconstraint(self):
        """
        """
        for j in self.iter_trem:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.P[i,j] for i in self.iter_EV),
                    sense=grb.GRB.LESS_EQUAL,rhs=self.tsfr,
                    name="Tsfr_constraint_time_{0}".format(j))
        return
    
    def __chargingconstraint(self):
        """
        """
        for i in self.iter_EV:
            for j in self.iter_trem:
                self.model.addLConstr(lhs=self.P[i,j],
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=(self.W[i,j]*self.pmin),
                                      name="big-m1-trick_{0}_{1}".format(i,j))
                self.model.addLConstr(lhs=self.P[i,j],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=(self.W[i,j]*self.pmax),
                                      name="big-m2-trick_{0}_{1}".format(i,j))
        return
    
    def __selectionconstraint(self):
        """
        """
        for i in self.iter_EV:
            self.model.addLConstr(
                    lhs=grb.quicksum(self.P[i,j] for j in self.iter_trem),
                    sense=grb.GRB.EQUAL,rhs=self.s[i]*self.u[i],
                    name="Selection_constraint_{0}".format(i))
        return
    
    def __selectg3(self):
        """
        """
        for i in self.iter_g3:
            self.model.addLConstr(
                    lhs=self.u[i],sense=grb.GRB.EQUAL,rhs=1,
                    name="choose_EV_{0}".format(i))
        return
    
    def __availabilityconstraint(self):
        """
        """
        for i in self.iter_EV:
            for j in range(self.a[i],len(self.iter_trem)):
                self.model.addLConstr(
                        lhs=self.W[i,j],sense=grb.GRB.EQUAL,rhs=0,
                        name="choose_EVavail_{0}_{1}".format(i,j))
        return
    
    def __objective(self):
        """
        """
        self.model.setObjective(grb.quicksum(self.u[j]*self.c[j]*self.s[j]\
                                           for j in self.iter_g12))
        return
    
    def solve(self):
        """
        """
        self.model.optimize()
        self.u_opt = np.array([int(self.u[i].getAttr("x")>0.5) \
                               for i in self.iter_EV])
        self.W_opt = np.zeros(shape=(len(self.s),self.T),dtype=int)
        self.P_opt = np.zeros(shape=(len(self.s),self.T))
        for i in self.iter_EV:
            for j in self.iter_trem:
                self.W_opt[i,j] = int(self.W[(i,j)].getAttr("x")>0.5)
                self.P_opt[i,j] = self.P[(i,j)].getAttr("x")
        return