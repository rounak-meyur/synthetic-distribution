# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program tries to analyze ensemble of synthetic networks by considering
the power flow Jacobian and its eigen values
"""

import sys,os
import numpy as np
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/ensemble-stochastic/"
tmpPath = workPath + "/temp/ensemble-stochastic/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network
from pyRobustnesslib import FlatEVD



#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble 1

# sub = 24664
# theta_range = range(200,601,20)
# phi_range = [4,5,6,7]


# W = []
# for theta in theta_range:
#     for phi in phi_range:
#         print(theta,phi)
#         # Create the cumulative distribution for a given (theta,phi) pair
#         fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
#         graph = read_network(tmpPath+fname+'.txt',homes)
#         W.append(FlatEVD(graph))

# log_data = [np.log10(x) for x in W]

#%% Ensemble 2

# sub = 24664

# thetaphi_range = [(200,5),(400,4),(800,1)]
# W = []
# for theta,phi in thetaphi_range:
#     print(theta,phi)
#     # Create the cumulative distribution for a given (theta,phi) pair
#     fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
#     graph = read_network(tmpPath+fname+'.txt',homes)
#     W.append(FlatEVD(graph))

# log_data = [np.log10(x) for x in W]

#%% Ensemble 3

sub = 24664

W = []
for i in range(50):
    print(i+1)
    # Create the cumulative distribution for a given (theta,phi) pair
    fname = str(sub)+'-network-'+str(i+1)
    graph = read_network(tmpPath+fname+'.txt',homes)
    W.append(FlatEVD(graph))

log_data = [np.log10(x) for x in W]

#%% PLot the eigenvalues
import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
box = ax.boxplot(log_data,patch_artist=True)
plt.xticks([i+1 for i in range(50)], [str(i+1) for i in range(50)])
ax.set_xlabel("Synthetic networks",fontsize=20)
ax.set_ylabel("Log transformed eigenvalues of Jacobian",fontsize=20)
fig.savefig(figPath+str(sub)+"-stochastic-flateig.png")
