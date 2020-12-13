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

scratchPath = workpath
tmppath = scratchPath + "/temp/"
qgispath = scratchPath + "/validate/QGIS Files/"
figpath = scratchPath + "/figs/"


sys.path.append(libpath)
from pyValidationlib import Validate,GetSynthNet


#%% Input parameters

sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]


areas = {'patrick_henry':194,'mcbryde':9001}


# Synthetic and Actual Network Data
synth_net = GetSynthNet(tmppath+'prim-network/',sublist)
V = Validate(qgispath,synth_net,areas)

V.compare_length(5,5,figpath)
V.compare_length(7,7,figpath)
V.compare_length(10,10,figpath)
V.compare_length(15,15,figpath)




V.node_stats(5,5,figpath)
V.node_stats(7,7,figpath)
V.node_stats(10,10,figpath)
V.node_stats(15,15,figpath)









