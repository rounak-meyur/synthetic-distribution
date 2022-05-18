# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:55:45 2022

Author: Rounak Meyur

Description: Creates balanced three phase network from positive sequence network.
Constructs the new network by creating new secondary network.
"""

import sys
import networkx as nx
from shapely.geometry import Point,LineString
from pyGeometrylib import Link
import gurobipy as grb
import numpy as np

#%% Solve optimization problem
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
    return

def get_3phase(loadlist,path):
    model = grb.Model(name="Get 3 phase Network")
    model.ModelSense = grb.GRB.MINIMIZE
    
    # Get home data
    p = np.array(loadlist)
    
    # Create variables
    uA = {}
    uB = {}
    uC = {}
    for i in range(len(loadlist)):
        uA[i] = model.addVar(vtype=grb.GRB.BINARY)
        uB[i] = model.addVar(vtype=grb.GRB.BINARY)
        uC[i] = model.addVar(vtype=grb.GRB.BINARY)
    
    dab = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='dab')
    dbc = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='dbc')
    dca = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='dca')
    
    pA = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='pa')
    pB = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='pb')
    pC = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name='pc')
    
    # Add constraints
    for i in range(len(loadlist)):
        model.addConstr(uA[i]+uB[i]+uC[i] == 1)
    
    model.addConstr(grb.quicksum([p[i]*uA[i] for i in range(len(loadlist))]) == pA)
    model.addConstr(grb.quicksum([p[i]*uB[i] for i in range(len(loadlist))]) == pB)
    model.addConstr(grb.quicksum([p[i]*uC[i] for i in range(len(loadlist))]) == pC)
    
    model.addConstr(pA-pB <= dab)
    model.addConstr(pA-pB >= -dab)
    model.addConstr(pB-pC <= dbc)
    model.addConstr(pB-pC >= -dbc)
    model.addConstr(pC-pA <= dca)
    model.addConstr(pC-pA >= -dca)
    
    # Set objective function
    model.setObjective(dab+dbc+dca)
    
    # Solve the MILP
    # grb.setParam('OutputFlag', 0)
    # grb.setParam('Heuristics', 0)
    
    # Open log file
    logfile = open(path+'3phase-gurobi.log', 'w')
    
    # Pass data into my callback function
    model._lastiter = -grb.GRB.INFINITY
    model._lastnode = -grb.GRB.INFINITY
    model._logfile = logfile
    model._vars = model.getVars()
    
    # Solve model and capture solution information
    model.optimize(mycallback)
    
    # Close log file
    logfile.close()
    print('')
    print('Optimization complete')
    if model.SolCount == 0:
        print('No solution found, optimization status = %d' % model.Status)
        sys.exit(0)
    else:
        print('Solution found, objective = %g' % model.ObjVal)
        uA_optimal = [uA[i].getAttr("x") for i in range(len(loadlist))]
        uB_optimal = [uB[i].getAttr("x") for i in range(len(loadlist))]
        uC_optimal = [uC[i].getAttr("x") for i in range(len(loadlist))]
        
        phase = []
        for i in range(len(loadlist)):
            if uA_optimal[i] > 0.8:
                phase.append('A')
            elif uB_optimal[i] > 0.8:
                phase.append('B')
            elif uC_optimal[i] > 0.8:
                phase.append('C')
            else:
                print("Error in solution... Check constraints")
                sys.exit(0)
        return phase


#%% Construct three phase secondary
def get_edges(t_node,nodes,phase):
    branch = {'A':[t_node],'B':[t_node],'C':[t_node]}
    for n in nodes:
        branch[phase[n]].append(n)
    g = nx.Graph()
    for ph in branch:
        nx.add_path(g,branch[ph])
    return list(g.edges)

def create_new_secondary(net,sub,phase):
    tree = nx.dfs_tree(net,sub)
    rem_edges = [e for e in tree.edges if net.edges[e]['label']!='S']
    tree.remove_edges_from(rem_edges)
    
    # Create new 3 phase network
    sec_edges = []
    t_nodes = [n for n in net if net.nodes[n]['label'] == 'T']
    for t in t_nodes:
        t_child = list(tree.successors(t))
        dfs_nodes = list(nx.dfs_preorder_nodes(tree, source=t))
        c_index = [dfs_nodes.index(c) for c in t_child]
        
        for idx in range(len(t_child)):
            if idx == len(t_child) - 1:
                nodelist = dfs_nodes[c_index[idx]:]
            else:
                nodelist = dfs_nodes[c_index[idx]:c_index[idx+1]]
            
            sec_edges += get_edges(t,nodelist,phase)
    return sec_edges


def construct_3phasenet(net,feedlines,primlines,seclines,phase):
    new_dist = nx.Graph()
    new_dist.add_edges_from(feedlines+primlines+seclines)
    # node attributes
    for n in new_dist:
        new_dist.nodes[n]['cord'] = net.nodes[n]['cord']
        new_dist.nodes[n]['geometry'] = Point(net.nodes[n]['cord'])
        new_dist.nodes[n]['load'] = net.nodes[n]['load']
        new_dist.nodes[n]['label'] = net.nodes[n]['label']
        if new_dist.nodes[n]['label'] == 'H':
            new_dist.nodes[n]['phase'] = phase[n]
        else:
            new_dist.nodes[n]['phase'] = 'ABC'
    # edge attributes
    for e in new_dist.edges:
        if (e in feedlines+primlines) or ((e[1],e[0]) in feedlines+primlines):
            new_dist.edges[e]['geometry'] = net.edges[e]['geometry']
            new_dist.edges[e]['length'] = net.edges[e]['length']
            new_dist.edges[e]['label'] = net.edges[e]['label']
            new_dist.edges[e]['phase'] = 'ABC'
        else:
            new_dist.edges[e]['geometry'] = LineString([net.nodes[e[0]]['cord'],
                                                        net.nodes[e[1]]['cord']])
            new_dist.edges[e]['length'] = Link(new_dist.edges[e]['geometry']).geod_length
            new_dist.edges[e]['label'] = 'S'
            new_dist.edges[e]['phase'] = [phase[n] for n in e \
                                          if net.nodes[n]['label']=='H'][0]
    return new_dist