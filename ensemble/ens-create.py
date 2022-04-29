# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program uses a Markov chain to create synthetic networks which
are solutions of the optimization program. Reformulates the original MILP with
altered constraints.
"""

import sys,os
import networkx as nx
import numpy as np
from shapely.geometry import LineString



workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
tsfrpath = rootpath + "/secnet/out/osm-prim-road/"
distpath = rootpath + "/primnet/out/osm-primnet/"
outpath = workpath + "/out/osm-ensemble-mc/"
sys.path.append(libpath)

from pyGeometrylib import Link
from pyExtractDatalib import GetDistNet,GetPrimRoad
from pyMiscUtilslib import get_load, create_variant_network
from pyEnsemblelib import reduced_MILP_primary


#%% Load the network data
sub = 121144
synth_net = GetDistNet(distpath,sub)
road_net = GetPrimRoad(tsfrpath,sub)
load = get_load(synth_net)

# Update road network data
for e in road_net.edges:
    road_net.edges[e]['geometry'] = LineString((road_net.nodes[e[0]]['cord'],
                                                road_net.nodes[e[1]]['cord']))
    road_net.edges[e]['length'] = Link(road_net.edges[e]['geometry']).geod_length
for n in road_net:
    if road_net.nodes[n]['label'] == 'R':
        road_net.nodes[n]['load'] = 0.0
    else:
        road_net.nodes[n]['load'] = load[n]



#%% Create ensemble of networks
# Initial edgelist
hv_edges = [e for e in synth_net.edges if synth_net.edges[e]['label'] == 'E']
prim_edges = [e for e in synth_net.edges if synth_net.edges[e]['label'] == 'P']
edgelist = [e for e in prim_edges if synth_net.nodes[e[0]]['label']=='T' \
            and synth_net.nodes[e[1]]['label']=='T']

np.random.seed(1234)
i = 0
# Run loop to create networks
while(i<10):
    # dummy graph for the synthetic network of current state
    # this graph changes for each state in the Markov chain
    new_synth = nx.Graph()
    new_synth.add_edges_from(hv_edges+prim_edges)
    
    # Get a valid edge to delete
    while(1):
        rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
        new_road = road_net.__class__()
        new_road.add_edges_from(road_net.edges)
        new_road.remove_edge(*rand_edge)
        if nx.number_connected_components(new_road)==1:
            break
    
    # Solve the restricted MILP
    M = reduced_MILP_primary(road_net,grbpath=workpath+"/out/")
    M.restrict(sub,new_synth,[rand_edge])
    prim_edges = M.solve()
    if prim_edges != []:
        create_variant_network(synth_net,road_net,prim_edges,[rand_edge])
        print("Variant Network",i+1,"constructed\n\n")
        edgelist = [e for e in prim_edges if synth_net.nodes[e[0]]['label']=='T' \
            and synth_net.nodes[e[1]]['label']=='T']
        nx.write_gpickle(synth_net,outpath+str(sub)+'-ensemble-'+str(i+1)+'.gpickle')
        i += 1

