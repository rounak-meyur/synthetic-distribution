# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program tries to generate ensemble of synthetic networks by varying
parameters in the optimization problem.
"""

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/prim-ensemble/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network


#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble

sub = 24665
theta_range = range(300,401,5)
phi_range = range(3,11)


Fdist = {}
Hops = {}
for theta in theta_range:
    for phi in phi_range:
        # Create the cumulative distribution for a given (theta,phi) pair
        fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
        graph = read_network(tmpPath+fname+'.txt',homes)
        
        
        