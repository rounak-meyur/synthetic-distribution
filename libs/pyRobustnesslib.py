# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:22:05 2020

@author: rounak
"""

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