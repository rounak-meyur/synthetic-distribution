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

#%% Function for callback
def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
#        nodecnt = model.cbGet(grb.GRB.Callback.MIP_NODCNT)
#        solcnt = model.cbGet(grb.GRB.Callback.MIP_SOLCNT)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if (time>300 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5% gap achieved')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1% gap achieved')
            model.terminate()
#        if nodecnt >= 20000 and solcnt:
#            print('Stop early - 10000 nodes explored')
#            model.terminate()
    return


#%% Secondary Network Creation
class MILP_secondary:
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
    
    




class MILP_secondary_pf:
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
                    sense=grb.GRB.LESS_EQUAL,rhs=M*(1-self.x[i]),
                    name="Voltage_drop_constraint_1{0}".format(i))
            self.model.addLConstr(
                    lhs=self.z[i]-grb.quicksum(self.A[j,i]*self.v[j] \
                              for j in iter_nodes)/(r*self.l[i])\
                              -M*self.x[i],
                    sense=grb.GRB.GREATER_EQUAL,rhs=-M*(1-self.x[i]),
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
        choice_M = len(self.tindex)
        LENGTH = nx.get_edge_attributes(graph,name='length')
        print("substations",len(self.sindex),"transformers",len(self.tindex),
              "roads",len(self.rindex),"nodes",len(self.nodes))
        
        self.A = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=True)
        self.I = nx.incidence_matrix(graph,nodelist=self.nodes,
                                     edgelist=self.edges,oriented=False)
        self.c = [LENGTH[e] for e in self.edges]
        
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__radiality()
        self.__flowconstraint(M=choice_M)
        self.__connectivity()
        self.__objective()
        self.optimal_edges = self.solve()
        return
    
    def __variables(self):
        """
        Create variable initialization for MILP such that x is binary, y is a 
        binary variable and z is a continuous variable. 
        """
        print("Setting up variables")
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
        print(len(self.x)+len(self.y),"binary variables",
              len(self.z),"continuous variables")
        return
    
    def __connectivity(self):
        """
        """
        print("Setting up connectivity constraints")
        for j,ind in enumerate(self.rindex):
            expr = grb.LinExpr(self.I[ind,:].toarray()[0],
                               list(self.x.values()))
            self.model.addLConstr(lhs=expr,
                                  sense=grb.GRB.LESS_EQUAL,
                                  rhs=len(self.edges)*self.y[j])
            self.model.addLConstr(lhs=expr,
                                  sense=grb.GRB.GREATER_EQUAL,rhs=2*self.y[j])
        return
    
    def __flowconstraint(self,M=3000):
        """
        """
        print("Setting up flow constraints for road nodes")
        for i in self.rindex:
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[i,:].toarray()[0],
                                    list(self.z.values())),
                    sense=grb.GRB.EQUAL,rhs=0,
                    name="node{0} zero injection".format(i))
        print("Setting up flow constraints for transformer nodes")
        for i in self.tindex:
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[i,:].toarray()[0],
                                    list(self.z.values())),
                    sense=grb.GRB.EQUAL,rhs=1,
                    name="node{0} flow constraint".format(i))
        print("Setting up Big M constraints; M=",M)
        for j in range(len(self.edges)):
            self.model.addLConstr(
                    lhs = self.z[j]-self.x[j]*M,
                    sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 1".format(j))
            self.model.addLConstr(
                    lhs = self.z[j]+self.x[j]*M,
                    sense=grb.GRB.GREATER_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 2".format(j))
        return
    
    def __radiality(self):
        """
        """
        print("Setting up radiality constraints")
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
        print("Setting up objective function")
        self.model.setObjective(grb.quicksum(self.x[j]*self.c[j]\
                                           for j in range(len(self.edges))))
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
            x_optimal = [self.x[i].getAttr("x") for i in range(len(self.x))]
            y_optimal = [self.y[i].getAttr("x") for i in range(len(self.y))]
            print("non-chosen road nodes",len([n for i,n in enumerate(self.rindex) if y_optimal[i]<0.5]))
            print("chosen road nodes",len([n for i,n in enumerate(self.rindex) if y_optimal[i]>0.5]))
            return [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]
        

class MILP_primary_modified:
    """
    Contains methods and attributes to generate the optimal primary distribution
    network for covering a given set of local transformers through the edges of
    an existing road network.
    """
    def __init__(self,graph,tnodes):
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
        self.p = [1e-3*tnodes[self.nodes[i]] for i in self.tindex]
        
        self.model = grb.Model(name="Get Primary Network")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__powerflowvar()
        self.__radiality()
        self.__flowconstraint()
        self.__connectivity()
        self.__limit_feeder()
        self.__objective()
        self.solve()
        return
        
    
    def __variables(self):
        """
        Create variable initialization for MILP such that x is binary, y is a 
        binary variable and z is a continuous variable. 
        """
        print("Setting up variables")
        self.x = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="x_{0}".format(i)) \
                    for i in range(len(self.edges))}
        self.y = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="y_{0}".format(i)) \
                    for i in range(len(self.rindex))}
        self.z = {i:self.model.addVar(vtype=grb.GRB.BINARY,
                                      name="z_{0}".format(i)) \
                    for i in range(len(self.rindex))}
        self.f = {i:self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      lb=-grb.GRB.INFINITY,
                                      name="f_{0}".format(i)) \
                    for i in range(len(self.edges))}
        return
    
    def __powerflowvar(self,r=0.8625/39690,M=0.15):
        """
        """
        print("Setting up power flow variables and constraints")
        self.v = {i:self.model.addVar(vtype = grb.GRB.CONTINUOUS,
                                      lb = 0.9, ub = 1.0,
                                      name = "v_{0}".format(i)) \
                    for i in range(len(self.nodes))}
        for j in range(len(self.edges)):
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[:,j].toarray()[:,0],[self.v[i] \
                                    for i in range(len(self.nodes))])-\
                    (self.f[j]*r*self.c[j])-M*(1-self.x[j]),
                    sense=grb.GRB.LESS_EQUAL,rhs=0)
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[:,j].toarray()[:,0],[self.v[i] \
                                    for i in range(len(self.nodes))])-\
                    (self.f[j]*r*self.c[j])+M*(1-self.x[j]),
                    sense=grb.GRB.GREATER_EQUAL,rhs=0)
        for i,ind in enumerate(self.rindex):
            self.model.addLConstr(
                    lhs=1-self.v[ind],sense=grb.GRB.LESS_EQUAL,rhs=self.z[i])
        return
    
    def __radiality(self):
        """
        """
        print("Setting up radiality constraints")
        self.model.addLConstr(
                lhs=grb.quicksum(self.x[j] for j in range(len(self.edges))),
                sense=grb.GRB.EQUAL,rhs=len(self.nodes)\
                -grb.quicksum((1-self.y[j]) for j in range(len(self.rindex)))\
                -grb.quicksum((1-self.z[j]) for j in range(len(self.rindex))),
                name="radiality constraint")
        return
    
    def __connectivity(self):
        """
        """
        print("Setting up connectivity constraints")
        for j,ind in enumerate(self.rindex):
            expr = grb.LinExpr(self.I[ind,:].toarray()[0],
                               [self.x[i] for i in range(len(self.edges))])
            self.model.addLConstr(lhs=expr,
                                  sense=grb.GRB.LESS_EQUAL,
                                  rhs=len(self.edges)*self.y[j])
            self.model.addLConstr(lhs=expr,
                                  sense=grb.GRB.GREATER_EQUAL,
                                  rhs=self.y[j])
            self.model.addLConstr(lhs=expr,
                                  sense=grb.GRB.GREATER_EQUAL,
                                  rhs=2*(self.y[j]+self.z[j]-1))
        for i in range(len(self.rindex)):
            self.model.addLConstr(lhs=1-self.z[i],sense=grb.GRB.LESS_EQUAL,
                                  rhs=self.y[i])
        return
    
    def __flowconstraint(self,M=800):
        """
        """
        print("Setting up flow constraints for road nodes")
        for i,ind in enumerate(self.rindex):
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[ind,:].toarray()[0],
                    [self.f[j] for j in range(len(self.edges))])-(1-self.z[i])*M,
                    sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="node{0} zero injection".format(i))
        for i,ind in enumerate(self.rindex):
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[ind,:].toarray()[0],
                    [self.f[j] for j in range(len(self.edges))])+(1-self.z[i])*M,
                    sense=grb.GRB.GREATER_EQUAL,rhs=0,
                    name="node{0} root injection".format(i))
        print("Setting up flow constraints for transformer nodes")
        for i,ind in enumerate(self.tindex):
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.A[ind,:].toarray()[0],
                    [self.f[j] for j in range(len(self.edges))]),
                    sense=grb.GRB.EQUAL,rhs=-self.p[i],
                    name="node{0} flow constraint".format(i))
        print("Setting up Big M constraints; M=",M)
        for j in range(len(self.edges)):
            self.model.addLConstr(
                    lhs = self.f[j]-self.x[j]*M,
                    sense=grb.GRB.LESS_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 1".format(j))
            self.model.addLConstr(
                    lhs = self.f[j]+self.x[j]*M,
                    sense=grb.GRB.GREATER_EQUAL,rhs=0,
                    name="edge{0} Big-M constraint 2".format(j))
        return
    
    def __limit_feeder(self):
        """
        """
        print("Setting up constraint for number of feeders")
        self.model.addLConstr(
                lhs=grb.quicksum((1-self.z[j]) \
                                 for j in range(len(self.rindex))),
                sense=grb.GRB.LESS_EQUAL,rhs=10,name='feeder_constraint')
        return
    
    def __objective(self):
        """
        """
        print("Setting up objective function")
        self.model.setObjective(grb.quicksum(self.x[j]*self.c[j]\
                            for j in range(len(self.edges))) + \
                grb.quicksum(self.d[j]*(1-self.z[j]) \
                             for j in range(len(self.rindex))))
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
            x_optimal = [self.x[i].getAttr("x") for i in range(len(self.x))]
            y_optimal = [self.y[i].getAttr("x") for i in range(len(self.y))]
            z_optimal = [self.z[i].getAttr("x") for i in range(len(self.z))]
            print("non-chosen road edges",len([e for i,e in enumerate(self.edges) if x_optimal[i]<0.5]))
            print("chosen road edges",len([e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]))
            print("non-chosen road nodes",len([n for i,n in enumerate(self.rindex) if y_optimal[i]<0.5]))
            print("chosen road nodes",len([n for i,n in enumerate(self.rindex) if y_optimal[i]>0.5]))
            print("transfer nodes",len([n for i,n in enumerate(self.rindex) if z_optimal[i]>0.5]))
            print("root nodes",len([n for i,n in enumerate(self.rindex) if z_optimal[i]<0.5]))
            self.optimal_edges = [e for i,e in enumerate(self.edges) if x_optimal[i]>0.5]
            self.roots = [self.nodes[ind] for i,ind in enumerate(self.rindex) if z_optimal[i]<0.5]
            return
        
        
        

















