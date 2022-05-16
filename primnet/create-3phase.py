# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import gurobipy as grb
import numpy as np


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
grbpath = workpath + "/out/gurobi/"
shappath = rootpath + "/output/optimal/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet

print("Imported modules")

#%% Functions to create 3 phase network

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

def create_3phase(loadlist,path):
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
    

#%% Load a network and save as shape file
sub = 121144
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

for sub in sublist:
    dist = GetDistNet(distpath,sub)
    tree = nx.dfs_tree(dist,sub)
    reg_nodes = list(nx.neighbors(dist,sub))

    
    for feeder in reg_nodes:
        homes = [n for n in nx.descendants(tree,feeder) \
                 if dist.nodes[n]['label']=='H']
        loadlist = [dist.nodes[h]['load'] for h in homes]
        phaselist = create_3phase(loadlist, grbpath)
        for i,h in enumerate(homes):
            with open(workpath+'/out/phase.txt','a') as f:
                f.write(str(h) + '\t' + phaselist[i]+'\n')
            






























