# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx
import numpy as np
from scipy import stats

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/prim-ensemble/"
tmpPath = workPath + "/temp/prim-ensemble/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network

#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble
sub = 24664
theta_range = range(300,401,5)
phi_range = range(3,11)

dict_hops = {}
for i,theta in enumerate(theta_range):
    for j,phi in enumerate(phi_range):
        print("Extracting:",i)
        fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
        dist_net = read_network(tmpPath+fname+'.txt',homes)
        dict_hops[(theta,phi)] = [nx.shortest_path_length(dist_net,n,sub) \
                                  for n in list(dist_net.nodes())]

print("Extracted Network Data")


#%% Heat map generation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=(20,12))
ax = fig.gca(projection='3d')

# Make data.
ref = dict_hops[(350,8)]
theta = np.arange(300, 401, 5)
phi = np.arange(3, 11, 1)
X, Y = np.meshgrid(theta, phi)

@np.vectorize
def kstat(x,y):
    return stats.ks_2samp(dict_hops[(x,y)],ref)[0]

Z = kstat(X,Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.set_zlim(0, 0.1)
ax.zaxis.set_major_locator(LinearLocator(5))
# ax.set_xlim(300, 400)
# ax.xaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_ylabel('Maximum number of feeders',fontsize=20)
ax.set_xlabel('Maximum substation rating',fontsize=20)
ax.set_zlabel('Kolmogorv Smirnov statistic',fontsize=20)