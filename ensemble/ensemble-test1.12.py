# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program compares the structural attributes of ensemble of networks.

"""

import sys,os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
enspath = workpath + "/out/osm-ensemble/"


sys.path.append(libpath)
from pyMiscUtilslib import powerflow,assign_linetype

print("Imported modules")



def GetDistNet(path,sub,i):
    return nx.read_gpickle(path+str(sub)+'-ensemble-'+str(i)+'.gpickle')


def run_powerflow(sub,i,homes,rating):
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
    graph = GetDistNet(enspath,sub,i)
    assign_linetype(graph)
    for h in homes:
        graph.nodes[h]['load'] = graph.nodes[h]['load'] - (rating*1e3)
    powerflow(graph)
    voltages = [graph.nodes[n]['voltage'] for n in graph]
    return voltages




                         zorder=-i, edgecolor="k",errwidth=5)
    

#%% Input
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

