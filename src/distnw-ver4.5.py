# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program tries to analyze ensemble of synthetic networks by considering
the power flow Jacobian and its eigen values
"""

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/ensemble-nopf/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import read_network


#%% Compute Jacobian
import numpy as np
import networkx as nx

def getYbus(G):
    """
    Generates the Ybus matrix for the networkx graph

    Parameters
    ----------
    G : TYPE Networkx Graph with attributes 'r' and 'x'
        DESCRIPTION A networkx graph representing the distribution network.

    Returns
    -------
    Ybus:   TYPE Scipy sparse matrix
            DESCRIPTION The bus admittance matrix which is the weighted Laplacian of
            the network.
    """
    nodelabel = nx.get_node_attributes(G,'label')
    node_ind = [i for i,node in enumerate(G.nodes()) \
                if nodelabel[node] != 'S']
    dict_y = {e:1.0/(G[e[0]][e[1]]['r']+1j*G[e[0]][e[1]]['x']) \
              for e in list(G.edges())}
    nx.set_edge_attributes(G,dict_y,'y')
    Ybus = nx.laplacian_matrix(G, nodelist=list(G.nodes()), weight='y').toarray()
    return Ybus[node_ind,:][:,node_ind]

def check_pf(G):
    """
    Checks power flow solution and plots the voltage at different nodes in the 
    network through colorbars.
    """
    A = nx.incidence_matrix(G,nodelist=list(G.nodes()),
                            edgelist=list(G.edges()),oriented=True).toarray()
    
    nodelabel = nx.get_node_attributes(G,'label')
    nodeload = nx.get_node_attributes(G,'load')
    node_ind = [i for i,node in enumerate(G.nodes()) \
                if nodelabel[node] != 'S']
    nodelist = [node for node in list(G.nodes()) if nodelabel[node] != 'S']
    
    # Resistance data
    R = np.diag([1.0/G[e[0]][e[1]]['r'] for e in list(G.edges())])
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([nodeload[h] for h in nodelist])
    q = 0.328*p
    
    dv = np.matmul(np.linalg.inv(G),p)
    v = 1.0-dv
    return p,q,v

def getJacobian(Y,p,q,v,theta):
    """
    Gets the the Jacobian matrix: 
        H: partial derivative of real power with respect to the voltage angles.
        N: partial derivative of real power with respect to the voltage magnitudes.
        M: partial derivative of reactive power with respect to the voltage angles.
        L: partial derivative of reactive power with respect to the voltage magnitudes.

    Parameters
    ----------
    Y : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nb = v.shape[0]
    G = Y.real
    B = Y.imag
    H = np.zeros(shape=(nb,nb))
    N = np.zeros(shape=(nb,nb))
    M = np.zeros(shape=(nb,nb))
    L = np.zeros(shape=(nb,nb))
    for m in range(nb):
        for n in range(nb):
            if m!=n: 
                H[m,n] = v[m]*v[n]*((G[m,n]*np.sin(theta[m]-theta[n]))-\
                                    (B[m,n]*np.cos(theta[m]-theta[n])))
                N[m,n] = v[m]*v[n]*((G[m,n]*np.cos(theta[m]-theta[n]))+\
                                    (B[m,n]*np.sin(theta[m]-theta[n])))
                M[m,n] = -v[m]*v[n]*((G[m,n]*np.cos(theta[m]-theta[n]))+\
                                    (B[m,n]*np.sin(theta[m]-theta[n])))
                L[m,n] = v[m]*v[n]*((G[m,n]*np.sin(theta[m]-theta[n]))-\
                                    (B[m,n]*np.cos(theta[m]-theta[n])))
            else:
                H[m,m] = -q[m]-B[m,m]*v[m]*v[m]
                N[m,m] = p[m]+G[m,m]*v[m]*v[m]
                M[m,m] = p[m]-G[m,m]*v[m]*v[m]
                L[m,m] = q[m]-B[m,m]*v[m]*v[m]
    return np.concatenate((np.concatenate((H,N),axis=1),
                           np.concatenate((M,L),axis=1)),axis=0)

def get_EVD(graph):
    p,q,v = check_pf(graph)
    theta = np.zeros(shape=(v.shape[0],))
    Y = getYbus(graph)
    J = getJacobian(Y,p,q,v,theta)
    w,_ = np.linalg.eig(J)
    return np.sort(abs(w))


def FlatEVD(graph):
    nodelabel = nx.get_node_attributes(graph,'label')
    nodeload = nx.get_node_attributes(graph,'load')
    nodelist = [node for node in list(graph.nodes()) if nodelabel[node] != 'S']
    p = np.array([nodeload[h] for h in nodelist])
    q = 0.328*p
    Y = getYbus(graph)
    H = -Y.imag-np.diag(q)
    N = Y.real+np.diag(p)
    M = -Y.real+np.diag(p)
    L = -Y.imag+np.diag(q)
    J = np.concatenate((np.concatenate((H,N),axis=1),
                        np.concatenate((M,L),axis=1)),axis=0)
    w,_ = np.linalg.eig(J)
    return np.sort(abs(w))


#%% Create the plots
q_object = Query(csvPath)
_,homes = q_object.GetHomes()


#%% Ensemble

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

sub = 24664

thetaphi_range = [(200,5),(400,4),(800,1)]
W = []
for theta,phi in thetaphi_range:
    print(theta,phi)
    # Create the cumulative distribution for a given (theta,phi) pair
    fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
    graph = read_network(tmpPath+fname+'.txt',homes)
    W.append(FlatEVD(graph))

log_data = [np.log10(x) for x in W]

#%% PLot the eigenvalues
import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
box = ax.boxplot(log_data,patch_artist=True)
plt.xticks([1,2,3], ["Max limit: 200, Max feeder:5","Max limit: 400, Max feeder:5",
                     "Max limit: 800, Max feeder:1"])
# ax.tick_params(bottom=False,labelbottom=False)
ax.set_xlabel("Synthetic networks",fontsize=20)
ax.set_ylabel("Log transformed eigenvalues of Jacobian",fontsize=20)
fig.savefig(figPath+str(sub)+"-nopfflateigmod.png")

# import pandas as pd
# df = pd.DataFrame(data=W)
# df.to_csv(csvPath + "eigval-"+str(sub)+"all.csv",index=False)
