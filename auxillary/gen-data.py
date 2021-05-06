# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 08:11:34 2021

@author: rouna
"""

import sys,os
import pandas as pd
from shapely import wkt
import networkx as nx


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/"
datapath = workpath + "/out/"


sys.path.append(libpath)
from pyPowerNetworklib import GetDistNet
print("Imported modules")


from math import exp
import matplotlib.pyplot as plt

def get_edge_names(synth_net):
    prim_amps = {e:2.2*exp(synth_net[e[0]][e[1]]['flow'])/6.8 \
                 for e in synth_net.edges if synth_net[e[0]][e[1]]['label']=='P'}
    sec_amps = {e:1.5*exp(synth_net[e[0]][e[1]]['flow'])/0.12 \
                for e in synth_net.edges if synth_net[e[0]][e[1]]['label']=='S'}
    
    edge_name = {}
    for e in synth_net.edges:
        # names of secondary lines
        if synth_net[e[0]][e[1]]['label']=='S':
            if sec_amps[e]<=95:
                edge_name[e] = 'OH_Voluta'
            elif sec_amps[e]<=125:
                edge_name[e] = 'OH_Periwinkle'
            elif sec_amps[e]<=165:
                edge_name[e] = 'OH_Conch'
            elif sec_amps[e]<=220:
                edge_name[e] = 'OH_Neritina'
            elif sec_amps[e]<=265:
                edge_name[e] = 'OH_Runcina'
            else:
                edge_name[e] = 'OH_Zuzara'
        # names of primary lines
        elif synth_net[e[0]][e[1]]['label']=='P':
            if prim_amps[e]<=140:
                edge_name[e] = 'OH_Swanate'
            elif prim_amps[e]<=185:
                edge_name[e] = 'OH_Sparrow'
            elif prim_amps[e]<=240:
                edge_name[e] = 'OH_Raven'
            elif prim_amps[e]<=315:
                edge_name[e] = 'OH_Pegion'
            else:
                edge_name[e] = 'OH_Penguin'
        else:
            edge_name[e] = 'OH_Penguin'
    names = [edge_name[e] for e in edge_name]
    return names



mont_sub = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

for code in mont_sub[:1]:
    graph = GetDistNet(distpath,code)
    # Node data
    node_data = {n:[graph.nodes[n]['label'],graph.nodes[n]['cord'][0],
                    graph.nodes[n]['cord'][1],graph.nodes[n]['load']] \
                 for n in graph.nodes}
    node_cols = ["label","longitude","latitude","load(W)"]
    df_nodes = pd.DataFrame.from_dict(node_data, orient='index',columns=node_cols)
    df_nodes.index.names = ["node"]
    df_nodes.to_csv(datapath+"node-data-"+str(code)+".txt",sep=" ")
    # Edge data
    edgelist = graph.edges()
    nodeA = [e[0] for e in edgelist]
    nodeB = [e[1] for e in edgelist]
    label = [graph[e[0]][e[1]]['label'] for e in edgelist]
    names = get_edge_names(graph)
    # egeom = [graph[e[0]][e[1]]['geometry'].apply(lambda x: wkt.dumps(x))\
    #          for e in edgelist]
    
    edge_data = {"nodeA":nodeA,"nodeB":nodeB,"label":label,"name":names,
                 }
    df_edges = pd.DataFrame.from_dict(edge_data)
    df_edges.to_csv(datapath+"edge-data-"+str(code)+".txt",sep=" ")