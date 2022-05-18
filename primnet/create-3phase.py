# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak Meyur

Description: This program loads a networkx gpickle file from the repo and 
stores information as a shape file for edgelist and nodelist
"""

import sys,os
import networkx as nx


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"
grbpath = workpath + "/out/gurobi/"
shappath = rootpath + "/output/optimal/"
outpath = workpath + "/out/3-phase-net/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from py3PhaseNetlib import get_3phase,create_new_secondary,construct_3phasenet

print("Imported modules")

    

#%% Load a network and save as shape file
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

for sub in sublist:
    dist = GetDistNet(distpath,sub)
    prm_edges = [e for e in dist.edges if dist.edges[e]['label']=='P']
    hvf_edges = [e for e in dist.edges if dist.edges[e]['label']=='E']
    tree = nx.dfs_tree(dist,sub)
    reg_nodes = list(nx.neighbors(dist,sub))

    # Get phase of each residence 
    phase = {}
    for feeder in reg_nodes:
        homes = [n for n in nx.descendants(tree,feeder) \
                 if dist.nodes[n]['label']=='H']
        loadlist = [dist.nodes[h]['load'] for h in homes]
        phaselist = get_3phase(loadlist, grbpath)
        for i,h in enumerate(homes):
            phase[h] = phaselist[i]
    
    # Create the 3 phase network
    sec_edges = create_new_secondary(dist,sub,phase)
    new_dist = construct_3phasenet(dist,hvf_edges,prm_edges,sec_edges,phase)
    nx.write_gpickle(new_dist,outpath+str(sub)+'-3phase-net.gpickle')
            

