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
from pyImgHelperlib import PFSol, read_network
q_object = Query(csvPath)
_,homes = q_object.GetHomes()
sub = 24664
dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)

#%% Display the networks

D = PFSol(dist_net)
D.run_pf()
D.volt_gif(figPath+"tmp/",figPath+str(sub)+"-voltage")
D.flow_gif(figPath+"tmp/",figPath+str(sub)+"-flows")

