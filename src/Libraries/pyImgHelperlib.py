# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:46:27 2019

Author: Rounak Meyur
"""

import os
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from matplotlib import cm
import numpy as np
from math import log


def read_network(filename,homes):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        filename: name of the .txt file
        homes: named tuple for residential consumer data
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            load: load for each node for consumers, otherwise it is 0.0
            label: 'H' for home, 'T' for transformer, 'R' for road node, 'S' for subs
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
    """
    # Open file and readlines
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    
    # Create the list/dictionary of node/edge labels and attributes
    edges = []
    edgelabel = {}
    edge_r = {}
    edge_x = {}
    nodelabel = {}
    nodepos = {}
    for line in lines:
        data = line.strip('\n').split(' ')
        edges.append((int(data[0]),int(data[4])))
        nodepos[int(data[0])] = [float(data[2]),float(data[3])]
        nodepos[int(data[4])] = [float(data[6]),float(data[7])]
        nodelabel[int(data[0])] = data[1]
        nodelabel[int(data[4])] = data[5]
        edgelabel[(int(data[0]),int(data[4]))] = data[-1]
        edge_r[(int(data[0]),int(data[4]))] = float(data[-3])
        edge_x[(int(data[0]),int(data[4]))] = float(data[-2])
    
    # Create the networkx graph
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nodeload = {n:[val*0.001 for val in homes.profile[n]] if nodelabel[n]=='H' \
                else [0.0]*24 for n in list(graph.nodes())}
    nx.set_edge_attributes(graph,edgelabel,'label')
    nx.set_edge_attributes(graph,edge_r,'r')
    nx.set_edge_attributes(graph,edge_x,'x')
    nx.set_node_attributes(graph,nodelabel,'label')
    nx.set_node_attributes(graph,nodepos,'cord')
    nx.set_node_attributes(graph,nodeload,'load')
    return graph



class ImageHelper(object):
    """
    """
    def save_image(self,figure,name,directory,close=True):
        '''
        '''
        figure.savefig("{}{}.png".format(directory,name))
        if close: plt.close()
        return
    
    
    def makegif(self,src,dest):
        '''
        Input:  src : Source directory of images
                dest: Destination path of gif
        '''
        fnames = [f for f in os.listdir(src) if ".png" in f]
        fnames_sorted = [str(m)+'.png'for m in 
                         sorted([int(s.strip('.png')) for s in fnames])]
        
    
        with imageio.get_writer(dest+'.gif', mode='I') as writer:
            for f in fnames_sorted:
                image = imageio.imread(src+f)
                writer.append_data(image)
        
        for f in fnames:
            os.remove(src+f)
        return
    
class PFSol:
    """
    """
    def __init__(self,network):
        """
        """
        self.dist_net = network
        self.img_helper = ImageHelper()
        self.voltage = {n:[] for n in list(network.nodes())}
        self.flows = {e:[] for e in list(network.edges())}
        return
    
    def run_pf(self):
        """
        Checks power flow solution and plots the voltage at different nodes in the 
        network through colorbars.
        """
        
        A = nx.incidence_matrix(self.dist_net,nodelist=list(self.dist_net.nodes()),
                                edgelist=list(self.dist_net.edges()),oriented=True).toarray()
        
        nodelabel = nx.get_node_attributes(self.dist_net,'label')
        nodeload = nx.get_node_attributes(self.dist_net,'load')
        node_ind = [i for i,node in enumerate(self.dist_net.nodes()) \
                    if nodelabel[node] != 'S']
        nodelist = [node for node in list(self.dist_net.nodes()) if nodelabel[node] != 'S']
        edgelist = [edge for edge in list(self.dist_net.edges())]
        
        # Resistance data
        edge_r = nx.get_edge_attributes(self.dist_net,'r')
        R = np.diag([1.0/edge_r[e] if e in edge_r else 1.0/edge_r[(e[1],e[0])] \
             for e in list(self.dist_net.edges())])
        G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
        
        for d in range(24):
            p = np.array([nodeload[h][d] for h in nodelist])
            v = np.matmul(np.linalg.inv(G),p)
            f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
            subnodes = [node for node in list(self.dist_net.nodes()) \
                        if nodelabel[node] == 'S']
            for s in subnodes: self.voltage[s].append(1.0)
            for i,n in enumerate(nodelist):
                self.voltage[n].append(1.0-v[i])
            for i,e in enumerate(edgelist):
                self.flows[e].append(abs(f[i]))
        return
    
    def update_voltage(self,i,tmppath):
        """
        """
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        colors = [self.voltage[n][i] for n in list(self.dist_net.nodes())]
        # Generate visual representation
        plt.ioff()
        fig = plt.figure(figsize=(18,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net, nodepos, ax=ax, node_color=colors,
            node_size=10, cmap=plt.cm.plasma, with_labels=False, vmin=0.85, vmax=1.02)
        cobj = cm.ScalarMappable(cmap='plasma')
        cobj.set_clim(vmin=0.85,vmax=1.02)
        cbar = fig.colorbar(cobj,ax=ax)
        cbar.set_label('Voltage(pu)',size=30)
        cbar.ax.tick_params(labelsize=20)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.set_title('Time of Day:'+str(1+i),fontsize=30)
        fig.savefig(tmppath+str(i)+".png")
        plt.close()
        return
    
    def update_flows(self,i,tmppath):
        """
        """
        nodepos = nx.get_node_attributes(self.dist_net,'cord')
        colors = [log(self.flows[e][i],2.0) for e in list(self.dist_net.edges())]
        # Generate visual representation
        fmin = 0.2
        fmax = 400.0
        plt.ioff()
        fig = plt.figure(figsize=(18,15))
        ax = fig.add_subplot(111)
        nx.draw_networkx(self.dist_net, nodepos, ax=ax, edge_color=colors,node_color='black',
            node_size=1.0, edge_cmap=plt.cm.plasma, with_labels=False, 
            vmin=log(fmin), vmax=log(fmax), width=2)
        cobj = cm.ScalarMappable(cmap='plasma')
        cobj.set_clim(vmin=fmin,vmax=fmax)
        cbar = fig.colorbar(cobj,ax=ax)
        cbar.set_label('Flow along edge in kVA',size=30)
        cbar.ax.tick_params(labelsize=20)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.set_title('Time of Day:'+str(1+i),fontsize=30)
        fig.savefig(tmppath+str(i)+".png")
        plt.close()
        return
    
    def volt_gif(self,src,dest):
        """
        """
        for i in range(24):
            self.update_voltage(i,src)
        self.img_helper.makegif(src,dest)
        return
    
    def flow_gif(self,src,dest):
        """
        """
        for i in range(24):
            self.update_flows(i,src)
        self.img_helper.makegif(src,dest)
        return
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        