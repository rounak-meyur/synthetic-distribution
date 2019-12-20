# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:34:40 2019

@author: rounak
"""
import gurobipy as grb
import numpy as np
import networkx as nx

#%% Function for callback
def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if (time>300 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5% gap achieved')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1% gap achieved')
            model.terminate()
    return

#%% Optimal Power Flow

class Dispatch:
    """
    """
    def __init__(self,graph,gendata):
        """
        """
        self.bus = sorted(list(graph.nodes()))
        self.line = list(graph.edges())
        self.A = nx.incidence_matrix(graph,nodelist=self.bus,edgelist=self.line,
                                     oriented=True)
        xdata = nx.get_edge_attributes(graph,name='x')
        self.x = [xdata[e] for e in self.line]
        self.X = np.linalg.inv(np.diag(self.x))
        self.B = np.matmul(np.matmul(self.A.toarray(),self.X),
                           self.A.toarray().T)
        f = nx.get_edge_attributes(graph,name='fmax')
        self.fmax = [f[e] for e in self.line]
        dem = nx.get_node_attributes(graph,name='demand')
        self.d = [dem[n] for n in self.bus]
        self.gen = list(gendata.keys())
        self.pmax = [gendata[g]['pmax'] for g in self.gen]
        self.pmin = [gendata[g]['pmin'] for g in self.gen]
        self.cost = [gendata[g]['cost'] for g in self.gen]
        
        self.genbus = {g:gendata[g]['bus'] for g in self.gen}
        self.G = np.zeros(shape=(len(self.bus),len(self.gen)))
        
        for i,gen in enumerate(self.gen):
            self.G[self.bus.index(self.genbus[gen]),i] = 1
        self.model = grb.Model(name="Optimal Economic Dispatch")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.__variables()
        self.__genconstraint()
        self.__lineconstraint()
        self.__powerflowconstraint()
        self.__getobjective()
        return
    
    def __variables(self):
        """
        """
        self.p = [self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      name="p_{0}".format(i))\
                                        for i in range(len(self.gen))]
        self.theta = [self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=-grb.GRB.INFINITY,
                                          name="theta_{0}".format(i))\
                                        for i in range(len(self.bus))]
        self.model.addLConstr(lhs=self.theta[-1],sense=grb.GRB.EQUAL,rhs=0.0,
                              name='reference_bus')
        return
    
    def __genconstraint(self):
        """
        """
        print("Setting up generation capacity constraints")
        for i in range(len(self.gen)):
            self.model.addLConstr(
                    lhs=self.p[i],sense=grb.GRB.GREATER_EQUAL,rhs=self.pmin[i],
                    name="generator_mincap_constraint_{0}".format(i))
            self.model.addLConstr(
                    lhs=self.p[i],sense=grb.GRB.LESS_EQUAL,rhs=self.pmax[i],
                    name="generator_maxcap_constraint_{0}".format(i))
        return
    
    def __lineconstraint(self):
        """
        """
        print("Setting up line capacity constraints")
        for i in range(len(self.line)):
            self.model.addLConstr(
                    lhs=(1.0/self.x[i])*grb.LinExpr(self.A[:,i].toarray()[:,0],
                         self.theta),sense=grb.GRB.GREATER_EQUAL,
                         rhs=-self.fmax[i],
                         name="line_mincap_constraint_{0}".format(i))
            self.model.addLConstr(
                    lhs=(1.0/self.x[i])*grb.LinExpr(self.A[:,i].toarray()[:,0],
                         self.theta),sense=grb.GRB.LESS_EQUAL,
                         rhs=self.fmax[i],
                         name="line_maxcap_constraint_{0}".format(i))
        return
    
    def __powerflowconstraint(self):
        """
        """
        print("Setting up power flow constraints")
        for i in range(len(self.bus)):
            self.model.addLConstr(
                    lhs=grb.LinExpr(self.G[i,:],self.p)-self.d[i],
                    sense=grb.GRB.EQUAL,rhs=grb.LinExpr(self.B[i,:],self.theta),
                    name="powerflow_constraint_{0}".format(i))
        return
    
    def __getobjective(self):
        """
        """
        self.model.setObjective(grb.quicksum(self.cost[j]*self.p[j]\
                                           for j in range(len(self.bus))))
        return
    


#%% Network Data
G = nx.Graph()
edges = {(1,2):{'x':0.0281,'f':4.0},(1,4):{'x':0.0304,'f':5.0},
         (1,5):{'x':0.0064,'f':5.0},(2,3):{'x':0.0108,'f':5.0},
         (3,4):{'x':0.0297,'f':5.0},(4,5):{'x':0.0297,'f':2.4}}
edgelist = list(edges.keys())
xdata = {e:edges[e]['x'] for e in edgelist}
fdata = {e:edges[e]['f'] for e in edgelist}
demand = {1:0.0,2:3.0,3:3.0,4:4.0,5:0.0}
G.add_edges_from(edgelist)
nx.set_edge_attributes(G,xdata,'x')
nx.set_edge_attributes(G,fdata,'fmax')
nx.set_node_attributes(G,demand,'demand')

gendata = {1:{'cost':14.0,'pmin':0.0,'pmax':0.4,'bus':1},
           2:{'cost':15.0,'pmin':0.0,'pmax':1.7,'bus':1},
           3:{'cost':30.0,'pmin':0.0,'pmax':5.2,'bus':3},
           4:{'cost':40.0,'pmin':0.0,'pmax':2.0,'bus':4},
           5:{'cost':10.0,'pmin':0.0,'pmax':6.0,'bus':5}}

Dis = Dispatch(G,gendata)
Dis.model.optimize()
p_optimal = [Dis.p[i].getAttr("x") for i in range(len(Dis.p))]
theta_optimal = [Dis.theta[i].getAttr("x") for i in range(len(Dis.theta))]
print(p_optimal)
print(theta_optimal)
print(Dis.model.getAttr("pi"))