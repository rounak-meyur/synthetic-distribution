# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Checks the secondary distribution network in a given county.
"""

import sys,os
import shutil
import pandas as pd
import networkx as nx
import datetime,time

workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
# Load scratchpath
scratchPath = workPath
inpPath = scratchPath + "/input/"
csvPath = scratchPath + "/csv/"
tmpPath = scratchPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query

def read_network(filename):
    # Open file and readlines
    with open(filename,'r') as f:
        lines = f.readlines()
    
    # Create the list/dictionary of node/edge labels and attributes
    edges = []
    nodelabel = {}
    for line in lines:
        data = line.strip('\n').split('\t')
        edges.append((int(data[0]),int(data[4])))
        nodelabel[int(data[0])] = data[1]
        nodelabel[int(data[4])] = data[5]
    
    # Create the networkx graph
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph,nodelabel,'label')
    return graph


def check_area(area,path):
    g = read_network(path+area+'-data/'+area+'-sec-dist.txt')
    # check that number of components is equal to number of transformers
    num_comp = nx.number_connected_components(g)
    node_label = nx.get_node_attributes(g,'label')
    num_tsfr = len([n for n in list(g.nodes()) if node_label[n]=='T'])
    num_home = len([n for n in list(g.nodes()) if node_label[n]=='H'])
    num_edge = g.number_of_edges()
    with open(path+area+'-data/'+area+'-tsfr-data.csv') as f:
        line_tsfr = f.readlines()
    print(area,num_comp-num_tsfr,
          num_edge-num_home,
          num_tsfr-len(line_tsfr),)
    return

with open(inpPath+'arealist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

areas = ['121','161','013','610','145']
for area in areas:
    check_area(area,csvPath)