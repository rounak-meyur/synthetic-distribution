# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:37:41 2020

@author: rounak
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



workpath = os.getcwd()
libpath = workpath + "/Libraries/"

scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
tmppath = scratchPath + "/temp/"
qgispath = scratchPath + "/validate/QGIS Files/"
figpath = scratchPath + "/figs/"


sys.path.append(libpath)
from pyValidationlib import Validate,GetSynthNet


#%% Input parameters

g = int(sys.argv[1])


sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]


areas = {'patrick_henry':194,'mcbryde':9001}


# Synthetic and Actual Network Data
synth_net = GetSynthNet(tmppath+'prim-network/',sublist)
V = Validate(qgispath,synth_net,areas)



# Spatial Distribution of Graphlets
target = nx.Graph()
nx.add_path(target,['a','b','c','d'])
V.graphlets(target,g,g,figpath)












