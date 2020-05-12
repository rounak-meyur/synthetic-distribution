# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network.
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
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Primary
from pyBuildNetworklib import MeasureDistance as dist
from pyBuildNetworklib import Initialize_Primary as init
from pyBuildNetworklib import InvertMap as imap


#%% Get transformers and store them in csv
fis=161
q_object = Query(csvPath,inpPath)
homes,roads = q_object.GetDataset(fislist=[161,770,775])
subs = q_object.GetSubstations(fis=161)
tsfr = q_object.GetTransformers(fis=161)

fiscode = '%03.f'%(fis)
df_hmap = pd.read_csv(csvPath+fiscode+'-home2link.csv')
H2Link = dict([(t.hid, (t.source, t.target)) for t in df_hmap.itertuples()])
L2Home = imap(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]
secondary_network_file = csvPath + fiscode + '-sec-dist.txt'


#%% Initialize Primary Network Generation Process
graph,S2Node = init(subs,roads,tsfr,links)

sub = 146410
substation = nt("local_substation",field_names=["id","cord","nodes"])
sub_data = substation(id=sub,cord=subs.cord[sub],nodes=S2Node[sub])
sys.exit(0)

#%% Generate primary distribution network
P = Primary(sub_data,homes,graph,min_node=300)
P.get_sub_network(secondary_network_file,flowmax=800,feedermax=10)
dist_net = P.dist_net

#%% Display network and save png
from pyBuildNetworklib import Display
D = Display(dist_net)
filename = str(sub)+'-network'
D.plot_network(figPath,filename)
D.save_network(tmpPath,filename)
D.plot_primary(homes,figPath,str(sub)+'-primary')

try:
    print("Number of cycles:",len(nx.find_cycle(dist_net)))
except:
    print("No cycles found!!!")
    pass

#run power flow
filename = str(sub)+'-voltage'
voltage = D.check_pf(figPath,filename)
filename = str(sub)+'-flows'
flows = D.check_flows(figPath,filename)





