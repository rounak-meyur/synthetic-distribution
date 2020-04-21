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
    filename = str(sub)+'-network-'+str(i+1)
    D.plot_network(figPath,filename)
    D.save_network(tmpPath,filename)




























