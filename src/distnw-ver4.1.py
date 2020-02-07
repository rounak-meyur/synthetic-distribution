# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:33:51 2020

@author: Rounak
"""

import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt

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
sub = 34780
dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)


def create_base():
    """
    """
    G = nx.Graph()
    
    G.add_edge('R1','ABC1',label='E')
    nx.add_path(G,['ABC1','ABC2','ABC3'],label='P')
    nx.add_path(G,['ABC1']+['ABC'+str(i) for i in range(4,22)],label='P')
    
    
    nx.add_path(G,['ABC2']+['A'+str(i) for i in range(1,12)],label='P')
    nx.add_path(G,['A9']+['A'+str(i) for i in range(12,14)],label='S')
    nx.add_path(G,['ABC2']+['B'+str(i) for i in range(1,10)],label='P')
    nx.add_path(G,['ABC2']+['C'+str(i) for i in range(1,8)],label='S')
    
    nx.add_path(G,['ABC1']+['A'+str(i) for i in range(52,57)],label='S')
    
    nx.add_path(G,['ABC4']+['A'+str(i) for i in range(14,23)],label='P')
    nx.add_path(G,['ABC4']+['B'+str(i) for i in range(10,32)],label='P')
    nx.add_path(G,['ABC4']+['C'+str(i) for i in range(8,14)],label='S')
    
    nx.add_path(G,['ABC8']+['B'+str(i) for i in range(32,40)],label='P')
    nx.add_path(G,['ABC19']+['C'+str(i) for i in range(14,26)],label='P')
    nx.add_path(G,['ABC19']+['A'+str(i) for i in range(23,35)],label='P')
    
    nx.add_path(G,['ABC20']+['C'+str(i) for i in range(26,34)],label='S')
    nx.add_path(G,['ABC20']+['C'+str(i) for i in range(34,42)],label='S')
    nx.add_path(G,['ABC20']+['A'+str(i) for i in range(35,45)],label='S')
    nx.add_path(G,['ABC20']+['B'+str(i) for i in range(40,50)],label='S')
    
    G.add_edge('A36','A45',label='S')
    G.add_edge('A37','A46',label='S')
    G.add_edge('A38','A47',label='S')
    
    G.add_edge('B42','B50',label='S')
    G.add_edge('B43','B51',label='S')
    
    nx.add_path(G,['ABC21']+['A'+str(i) for i in range(48,52)],label='S')
    nx.add_path(G,['ABC21']+['B'+str(i) for i in range(52,59)],label='S')
    nx.add_path(G,['ABC21']+['C'+str(i) for i in range(42,51)],label='S')
    
    node_label = {}
    for n in list(G.nodes()):
        if n[0]=='R':
            node_label[n] = 'S'
        elif n[0:3]=='ABC':
            node_label[n] = 'T'
        elif n[0]=='A':
            node_label[n] = 'H'
        elif n[0]=='B':
            node_label[n] = 'H'
        elif n[0]=='C':
            node_label[n] = 'H'
    nx.set_node_attributes(G,node_label,'nodelab')
    
    try:
        print("Number of cycles:",len(nx.find_cycle(G)))
    except:
        print("No cycles found!!!")
    return G

def display_base(G):
    """
    """
    nodecolor = []
    nodesize = []
    for n in list(G.nodes()):
        if n[0]=='R':
            nodecolor.append('magenta')
            nodesize.append(100)
        elif n[0:3]=='ABC':
            nodecolor.append('black')
            nodesize.append(50)
        elif n[0]=='A':
            nodecolor.append('red')
            nodesize.append(20)
        elif n[0]=='B':
            nodecolor.append('green')
            nodesize.append(20)
        elif n[0]=='C':
            nodecolor.append('blue')
            nodesize.append(20)
    
    nx.draw_networkx(G,with_labels=False,node_color=nodecolor,node_size=nodesize)
    return
    
def degree_dist(G1,G2):
    """
    """        
    degree_sequence_a = sorted([d for n, d in G1.degree()], reverse=True)  # degree sequence
    na = G1.number_of_nodes()
    degree_sequence_b = sorted([d for n, d in G2.degree()], reverse=True)  # degree sequence
    nb = G2.number_of_nodes()
    
    degreeCount_a = collections.Counter(degree_sequence_a)
    degreeCount_b = collections.Counter(degree_sequence_b)
    deg_a = degreeCount_a.keys()
    deg_b = degreeCount_b.keys()
    
    max_deg = max(max(list(deg_a)),max(list(deg_b)))
    cnt_a = []
    cnt_b = []
    for i in range(1,max_deg+1):
        if i in degreeCount_a:
            cnt_a.append(100.0*degreeCount_a[i]/na)
        else:
            cnt_a.append(0)
        if i in degreeCount_b:
            cnt_b.append(100.0*degreeCount_b[i]/nb)
        else:
            cnt_b.append(0)
    
    cnt_a = tuple(cnt_a)
    cnt_b = tuple(cnt_b)
    deg = np.arange(max_deg)+1
    width = 0.35
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(deg, cnt_a, width, color='royalblue')
    rects2 = ax.bar(deg+width, cnt_b, width, color='seagreen')
    ax.set_xticks(deg + width / 2)
    ax.set_xticklabels([str(x) for x in deg])
    ax.legend((rects1[0],rects2[0]),('Synthetic Network', 'Original Network'),
              prop={'size': 15})
    ax.set_ylabel("Percentage of nodes",fontsize=15)
    ax.set_xlabel("Degree of nodes",fontsize=15)
    
    ax.tick_params(axis='both', labelsize=15)
    return


G = create_base()
# degree_dist(dist_net,G)



#%% Edit Distance
def return_eq(edge1, edge2):
    return (edge1['label']==edge2['label'])
# print(nx.graph_edit_distance(dist_net, G, edge_match=return_eq))
# print(nx.graph_edit_distance(dist_net, G))

# def return_eq_node(node1, node2):
#     return (node1['nodelab']==node2['nodelab'])
# print(nx.graph_edit_distance(dist_net, G, node_match=return_eq_node))
    

#%%
def hop_dist(G,base,sub):
    """
    """        
    hops = [nx.shortest_path_length(G,n,sub) for n in list(G.nodes())]
    hops_base = [nx.shortest_path_length(base,n,'R1') for n in list(base.nodes())]
    hops = [hops,hops_base]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.hist(hops,density=True)
    ax.set_ylabel("Fraction of nodes",fontsize=15)
    ax.set_xlabel("Hops from root",fontsize=15)
    
    ax.tick_params(axis='both', labelsize=15)
    return  


# hop_dist(G,'R1')
for sub in [34780,34816,28228,28235,34810]:
    dist_net = read_network(tmpPath+str(sub)+'-network.txt',homes)
    hop_dist(dist_net,G,sub)