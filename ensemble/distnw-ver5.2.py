# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network. Thereafter, it generates ensembles by varying parameters
of the optimization framework.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import namedtuple as nt



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/prim-ensemble/"
tmpPath = workPath + "/temp/prim-ensemble/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Primary,Display
from pyBuildNetworklib import MeasureDistance as dist
from pyBuildNetworklib import Initialize_Primary as init
from pyBuildNetworklib import InvertMap as imap
from pyBuildNetworklib import plot_graph


#%% Get transformers and store them in csv
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


#%% Initialize Primary Network Generation Process
graph,S2Node = init(subs,roads,tsfr,links)

sub = 24664
substation = nt("local_substation",field_names=["id","cord","nodes"])
sub_data = substation(id=sub,cord=subs.cord[sub],nodes=S2Node[sub])

#%% Generate primary distribution network

for fmax in range(405,601,5):
    P = Primary(sub_data,homes,graph)
    P.get_sub_network(secondary_network_file,flowmax=fmax,feedermax=8)
    dist_net = P.dist_net
    D = Display(dist_net)
    filename = str(sub)+'-network-f-'+str(fmax)+'-s-8'
    D.plot_network(figPath,filename)
    D.save_network(tmpPath,filename)

    try:
        print("Number of cycles:",len(nx.find_cycle(dist_net)))
    except:
        print("No cycles found!!!")
        pass

    print("##############################################################")
    print("##############################################################")
    print("##############################################################")


