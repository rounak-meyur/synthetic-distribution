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
outpath = workpath + "/out/"
secpath = rootpath + "/secnet/out/osm-sec-network/"


sys.path.append(libpath)
from pyBuildPrimNetlib import Primary
from pyDrawNetworklib import plot_network, color_nodes, color_edges
from pyMiscUtilslib import powerflow
print("Imported modules")

#%% Primary network generation

# Extract all substations in the region
with open(outpath+"subdata.txt") as f:
    lines = f.readlines()

data = [temp.strip('\n').split('\t') for temp in lines]
subs = {int(d[0]):{"id":int(d[0]),"near":int(d[1]),
                   "cord":[float(d[2]),float(d[3])]} for d in data}


sub=121144
sub_data = subs[sub]

# Generate primary distribution network by partitions
P = Primary(sub_data,outpath+"osm-prim-master/")

dist_net = P.get_sub_network(secpath,inppath,outpath)

powerflow(dist_net)
sys.exit(0)


#%% Other stuff
def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    graph = nx.Graph()
    for c in code:
        g = nx.read_gpickle(path+str(c)+'-prim-dist.gpickle')
        graph = nx.compose(graph,g)
    return graph

sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]


#%% Combine primary and secondary network
synth_net = GetDistNet(outpath,sublist)
small_synth_net1 = GetDistNet(outpath,[121143])
small_synth_net2 = GetDistNet(outpath,[147793])

dict_inset = {121143:{'graph':small_synth_net1,'loc':2,
                      'loc1':1,'loc2':3,'zoom':1.5},
              147793:{'graph':small_synth_net2,'loc':3,
                      'loc1':1,'loc2':4,'zoom':1.2}}

##%% Plot the network with inset figure
plot_network(dist_net,with_secnet=True)
plot_network(synth_net,dict_inset,figpath,with_secnet=True)

#%% Voltage and flow plots
color_nodes(synth_net,dict_inset,figpath)
color_edges(synth_net,dict_inset,figpath)



