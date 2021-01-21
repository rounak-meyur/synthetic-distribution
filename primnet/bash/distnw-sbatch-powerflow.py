# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:14:58 2021

Author: Rounak
"""

import sys,os
import networkx as nx


workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inppath = scratchPath + "/input/"
primpath = scratchPath + "/temp/prim-network/"
secpath = scratchPath + "/temp/sec-network/"
distpath = scratchPath + "/temp/dist-network/"

sys.path.append(libPath)
from pyDrawNetworklib import GetPrimNet,GetSecnet,combine_networks
from pyDrawNetworklib import GetHomes, check_flows, check_voltage
print("Imported modules")


# Primary Network
sub = sys.argv[1]
primnet = GetPrimNet(primpath,sub)

# Secondary network
master_secnet = GetSecnet(secpath,['121','161','071','063','045','155'])

# Combine networks
synth_net = combine_networks(primnet,master_secnet)

#%% Voltage and power flows
homes = GetHomes(inppath,['121','161','071','063','045','155'])
voltage = check_voltage(synth_net,homes)
flows = check_flows(synth_net,homes)
nx.set_node_attributes(synth_net,voltage,'voltage')
nx.set_edge_attributes(synth_net,flows,'flow')
nx.write_gpickle(synth_net,distpath+sub+'-distnet.gpickle')
