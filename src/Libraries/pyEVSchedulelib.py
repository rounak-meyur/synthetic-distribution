# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:36:14 2019

Author: Rounak Meyur
"""

import gurobipy as grb
import numpy as np

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
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
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