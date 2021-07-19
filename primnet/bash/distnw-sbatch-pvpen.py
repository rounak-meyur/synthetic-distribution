# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:00:21 2021

@author: Rounak
"""

import sys,os
import networkx as nx
import numpy as np


workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)

# Load scratchpath
scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inppath = scratchpath + "/input/"
figpath = scratchpath + "/figs/"
tmppath = scratchpath + "/temp/"
distpath = tmppath + "osm-prim-network/"

from pyBuildPrimNetlib import powerflow
from pyPowerNetworklib import GetDistNet,assign_linetype


#%% User defined functions
def run_powerflow(sub,homes,rating):
    """
    Runs the power flow on the network with the installed rooftop solar panels.

    Parameters
    ----------
    sub : integer type
        substation ID.
    homes : list type
        list of node IDs where solar panels are installed.
    rating : float type
        power rating of solar panel in kW.

    Returns
    -------
    voltages : TYPE
        DESCRIPTION.
    flows : TYPE
        DESCRIPTION.

    """
    graph = GetDistNet(distpath,sub)
    assign_linetype(graph)
    for h in homes:
        graph.nodes[h]['load'] = graph.nodes[h]['load'] - (rating*1e3)
    powerflow(graph)
    voltages = [graph.nodes[n]['voltage'] for n in graph]
    return voltages


#%% Get list of substations
with open(inppath+"rural-sublist.txt") as f:
    rural = f.readlines()
rural = [int(r) for r in rural[0].strip('\n').split(' ')]
f_done = [int(f.strip('-prim-dist.gpickle')) for f in os.listdir(distpath)]
rural_done = [r for r in f_done if r in rural]


#%% MV level penetration

pen = sys.argv[1][:-1]
level = sys.argv[1][-1]

if level == 'M':
    # Penetration at a single node in MV network
    prim_host = 3
    for sub in rural_done:
        synth_net = GetDistNet(distpath,sub)
        total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])
            
        # Get random location for MV penetration
        prim = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
        solar = np.random.choice(prim,prim_host,replace=False)
        rating = 1e-5*float(pen)*total_load/prim_host
        
        # Run powerflow and get node voltage
        v_list = run_powerflow(sub,solar,rating)
        
        # Write the voltages in a txt file
        with open(tmppath+"pv_pen_MV_"+pen+".txt",'a') as f:
            f.write('\n'.join([str(x) for x in v_list]))
        
elif level == 'L':
    # Penetration at multiple nodes in LV network
    res_host = 0.5
    for sub in rural_done:
        synth_net = GetDistNet(distpath,sub)
        total_load = sum([synth_net.nodes[n]['load'] for n in synth_net])
        
        # Get random locations for LV penetration
        res = [n for n in synth_net if synth_net.nodes[n]['label']=='H']
        n_solar = int(res_host*len(res))
        solar = np.random.choice(res,n_solar,replace=False)
        rating = 1e-5*float(pen)*total_load/n_solar
        
        # Run powerflow and get node voltage
        v_list = run_powerflow(sub,solar,rating)
        
        # Write the voltages in a txt file
        with open(tmppath+"pv_pen_LV_"+pen+".txt",'a') as f:
            f.write('\n'.join([str(x) for x in v_list]))

















