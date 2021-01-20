# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:14:58 2021

Author: Rounak
"""

import sys,os
import networkx as nx


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
primpath = workpath + "/out/prim-network/"
secpath = rootpath + "/secnet/out/sec-network/"

sys.path.append(libpath)
from pyDrawNetworklib import GetSynthNet,Link,GetSecnet,combine_networks
from pyDrawNetworklib import GetHomes, check_flows, check_voltage
from pyDrawNetworklib import plot_network, color_nodes, color_edges
print("Imported modules")




sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
primnet = GetSynthNet(primpath,sublist)
sgeom = nx.get_edge_attributes(primnet,'geometry')
synthgeom = {e:Link(sgeom[e]) for e in sgeom}
glength = {e:synthgeom[e].geod_length for e in sgeom}
nx.set_edge_attributes(primnet,glength,'geo_length')
print("Primary network extracted")

#%% Secondary network
master_secnet = GetSecnet(secpath,['121','161','071','063','045','155'])
print("Secondary network extracted")

#%% Combine primary and secondary network
synth_net = combine_networks(primnet,master_secnet)
small_primnet1 = GetSynthNet(primpath,[121143])
small_synth_net1 = combine_networks(small_primnet1,master_secnet)

small_primnet2 = GetSynthNet(primpath,[147793])
small_synth_net2 = combine_networks(small_primnet2,master_secnet)


dict_inset = {121143:{'graph':small_synth_net1,'loc':2,
                      'loc1':1,'loc2':3,'zoom':1.5},
              147793:{'graph':small_synth_net2,'loc':3,
                      'loc1':1,'loc2':4,'zoom':1.2}}

##%% Plot the network with inset figure
plot_network(synth_net,dict_inset,figpath,with_secnet=True)

#%% Voltage and power flows
homes = GetHomes(inppath,['121','161','071','063','045','155'])
voltage = check_voltage(synth_net,homes)
flows = check_flows(synth_net,homes)

color_nodes(synth_net,voltage,dict_inset,figpath)
color_edges(synth_net,flows,dict_inset,figpath)



