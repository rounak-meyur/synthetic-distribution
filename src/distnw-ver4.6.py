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
import pandas as pd
from pyBuildNetworklib import Initialize_Primary as init
from pyBuildNetworklib import InvertMap as imap
from collections import namedtuple as nt
from pyBuildNetworklib import Primary,Display

q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()
subs = q_object.GetSubstations()
tsfr = q_object.GetTransformers()

df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])
L2Home = imap(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])<=70]
secondary_network_file = inpPath + 'secondary-network.txt'

#%% Ensemble
masterG,S2Node = init(subs,roads,tsfr,links)

sub = 24664
substation = nt("local_substation",field_names=["id","cord","nodes"])
sub_data = substation(id=sub,cord=subs.cord[sub],nodes=S2Node[sub])
G = masterG.subgraph(sub_data.nodes)

edgeprob = {e:0 for e in list(G.edges())}

theta_range = range(200,601,20)
phi_range = [4,5,6,7]


count = 0
for theta in theta_range:
    for phi in phi_range:
        count += 1
        print(theta,phi)
        # Create the cumulative distribution for a given (theta,phi) pair
        fname = str(sub)+'-network-f-'+str(theta)+'-s-'+str(phi)
        graph = read_network(tmpPath+fname+'.txt',homes)
        
        for e in list(graph.edges):
            if graph[e[0]][e[1]]['label']=='P': 
                if (e[0],e[1]) in edgeprob:
                    edgeprob[(e[0],e[1])] += 1
                else:
                    edgeprob[(e[1],e[0])] += 1

# Compute edge selection probability
edgeprob = {e:edgeprob[e]/float(count) for e in edgeprob}

#%% Stochastic generation
figPath = workPath + "/figs/ensemble-stochastic/"
tmpPath = workPath + "/temp/ensemble-stochastic/"

for i in range(50):
    P = Primary(sub_data,homes,G)
    P.get_stochastic_network(secondary_network_file,edgeprob)
    dist_net = P.dist_net
    D = Display(dist_net)
    filename = str(sub)+'-network-f-'+str(i)
    D.plot_network(figPath,filename)
    D.save_network(tmpPath,filename)




























