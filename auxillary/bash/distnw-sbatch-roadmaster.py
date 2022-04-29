# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:29:49 2020

Author: Rounak Meyur
Description: This program creates a visualization of how the local transformers 
are distributed among the substations.
"""

import sys,os
import networkx as nx
workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)



# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
figPath = scratchPath + "/figs/"
tmpPath = scratchPath + "/temp/"
roadpath = tmpPath + "osm-prim-road/"



# mastpath = tmpPath + "osm-prim-master/"
# filelist = os.listdir(mastpath)
# for f in filelist:
#     master_net = nx.read_gpickle(mastpath+f)
#     road = master_net.__class__()
#     road.add_edges_from(master_net.edges)
#     for n in road.nodes:
#         road.nodes[n]['cord'] = master_net.nodes[n]['cord']
#         road.nodes[n]['label'] = master_net.nodes[n]['label']
#         road.nodes[n]['load'] = master_net.nodes[n]['load']
    
#     sub = f.strip('-master.gpickle')
#     nx.write_gpickle(road,roadpath+sub+'-road.gpickle')

# sys.exit(0)

sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

from pyExtractDatalib import GetPrimRoad
from pyDrawNetworklib import plot_road_network

roadnet = GetPrimRoad(roadpath,sublist)



small_road_net1 = GetPrimRoad(roadpath,121143)
small_road_net2 = GetPrimRoad(roadpath,147793)
dict_inset = {121143:{'graph':small_road_net1,'loc':2,
                      'loc1':1,'loc2':3,'zoom':1.5},
              147793:{'graph':small_road_net2,'loc':3,
                      'loc1':1,'loc2':4,'zoom':1.1}}


# Extract all substations in the region
with open(tmpPath+"subdata.txt") as f:
    lines = f.readlines()
data = [temp.strip('\n').split('\t') for temp in lines]
subs = {int(d[0]):{"id":int(d[0]),"near":int(d[1]),
                    "cord":[float(d[2]),float(d[3])]} for d in data}
subdata = {s:subs[s] for s in sublist}
plot_road_network(roadnet,subdata,inset=dict_inset,path=figPath+"all")

# with open(inpPath+'sublist.txt') as f:
#     sublist = [int(x) for x in f.readlines()[0].split(' ')]

# G = nx.Graph()
# color_list = ['crimson','royalblue','seagreen','magenta','gold',
#               'cyan','olive','maroon']*132
# for i,s in enumerate(sublist):
#     new_g = read_master_graph(tmpPath+'prim-master/',str(s))
#     ncol = {n:color_list[i] for n in list(new_g.nodes())}
#     nx.set_node_attributes(new_g,ncol,'color')
#     G = nx.union(G,new_g)


# nodepos = nx.get_node_attributes(G,'cord')
# nodecol = nx.get_node_attributes(G,'color')
# nodelist = list(G.nodes())
# nodecollist = [nodecol[n] for n in nodelist]
# fig = plt.figure(figsize=(100,50))
# ax = fig.add_subplot(111)
# nx.draw_networkx(G,pos=nodepos,ax=ax,nodelist=nodelist,node_color=nodecollist,
#                  node_size=20.0,with_labels=False,edge_color='black',width=1)


# axins = zoomed_inset_axes(ax, 20, loc=2)
# subind = 1041#int(sys.argv[1])
# subind = sublist.index(150692)
# sub = sublist[subind]

# axins.scatter(subs.cord[sub][0],subs.cord[sub][1],s=5000.0,c='black',
#               marker='*')
# sgraph = read_master_graph(tmpPath+'prim-master/',str(sub))
# snodes = list(sgraph.nodes())
# axins.set_aspect(1.2)
# xpts = [nodepos[r][0] for r in snodes]
# ypts = [nodepos[r][1] for r in snodes]
# axins.scatter(xpts,ypts,s=100.0,c='royalblue')

# axins.set_xlim(min(xpts),max(xpts))
# axins.set_ylim(min(ypts),max(ypts))
# axins.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

# mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
# fig.savefig("{}{}.png".format(figPath,'partition-sub'),bbox_inches='tight')
