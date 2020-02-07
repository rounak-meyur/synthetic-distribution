# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program displays results.
"""

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/results-dec12/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network

q_object = Query(csvPath)
_,homes = q_object.GetHomes()
sub = 34816
dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)

#%% Display the networks
from pyBuildNetworklib import Display
D = Display(dist_net)
D.plot_network(figPath,str(sub)+'-networktest')
D.plot_primary(homes,figPath,str(sub)+'-primarytest')
D.check_pf(figPath,str(sub)+'-voltagetest')

