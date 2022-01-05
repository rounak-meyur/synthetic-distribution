# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:14:58 2021

Author: Rounak
"""

import sys,os
workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
secpath = rootpath + "/secnet/out/osm-sec-network/"


sys.path.append(libpath)
print("Imported modules")

#%% Primary network generation
# import networkx as nx
# from pyBuildPrimNetlib import Primary
# from pyMiscUtilslib import powerflow

# Extract all substations in the region
with open(outpath+"subdata.txt") as f:
    lines = f.readlines()

data = [temp.strip('\n').split('\t') for temp in lines]
subs = {int(d[0]):{"id":int(d[0]),"near":int(d[1]),
                    "cord":[float(d[2]),float(d[3])]} for d in data}


# sub=121144
# sub_data = subs[sub]

# # Generate primary distribution network by partitions
# P = Primary(sub_data,outpath+"osm-prim-master/")

# dist_net = P.get_sub_network(secpath,inppath,outpath)

# powerflow(dist_net)
# sys.exit(0)


#%% Display example network
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
#%% Road network
from pyDrawNetworklib import plot_road_network
from pyExtractDatalib import GetPrimRoad
from pyDrawNetworklib import plot_network
from pyExtractDatalib import GetDistNet

distpath = outpath + "osm-primnet/"
roadpath = outpath + "osm-prim-master/"
sub = 121143
subdata = {s:subs[s] for s in [sub]}
roadnet = GetPrimRoad(roadpath,sub)
synth_net = GetDistNet(distpath,sub)
plot_road_network(roadnet,subdata,path=figpath+str(sub))
plot_network(synth_net,{},figpath+str(sub)+"-prim",with_secnet=False)
plot_network(synth_net,{},figpath+str(sub)+"-all",with_secnet=True)
sys.exit(0)
#%% Combine primary and secondary network

distpath = outpath + "osm-primnet/"
synth_net = GetDistNet(distpath,sublist)
small_synth_net1 = GetDistNet(distpath,121143)
small_synth_net2 = GetDistNet(distpath,147793)

dict_inset = {121143:{'graph':small_synth_net1,'loc':2,
                      'loc1':1,'loc2':3,'zoom':1.5},
              147793:{'graph':small_synth_net2,'loc':3,
                      'loc1':1,'loc2':4,'zoom':1.1}}

##%% Plot the network with inset figure
plot_network(synth_net,dict_inset,figpath+"prim",with_secnet=False)


#%% Voltage and flow plots
# from pyDrawNetworklib import color_nodes, color_edges
# color_nodes(synth_net,dict_inset,figpath)
# color_edges(synth_net,dict_inset,figpath)



