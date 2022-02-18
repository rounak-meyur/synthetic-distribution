# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program plots the distribution network of Virginia state.
"""

# import sys,os
# import networkx as nx
# import matplotlib.pyplot as plt
# import geopandas as gpd
# import numpy as np
# from itertools import combinations as comb

# workpath = os.getcwd()
# rootpath = os.path.dirname(workpath)
# libpath = rootpath + "/libs/"
# inppath = rootpath + "/input/"
# figpath = workpath + "/figs/"
# distpath = rootpath + "/primnet/out/osm-primnet/"


# sys.path.append(libpath)
# from pyExtractDatalib import GetDistNet
# from pyDrawNetworklib import DrawNodes, DrawEdges
# from pyResiliencelib import count_motif
# print("Imported modules")




# sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
# with open(inppath+"urban-sublist.txt") as f:
#     urban_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]
# with open(inppath+"rural-sublist.txt") as f:
#     rural_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]

# rural_sub = [s for s in sublist if s in rural_sublist]
# urban_sub = [s for s in sublist if s in urban_sublist]



#%%
import itertools
import networkx as nx
import numpy as np
def get_motifs(g):
    nodes = g.nodes()
    quadlets = list(itertools.combinations(nodes,4))
    
    return

tree = nx.random_tree(n=5, seed=0)
print(nx.forest_str(tree, sources=[0]))
get_motifs(tree)



# dist = GetDistNet(distpath,121144)























#%%
# mont_sub = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

# sw_sub = [109418, 109426, 109430, 109487, 113962, 113968, 113969]





# plot_network(ax,rural_sub)
# plot_network(ax,urban_sub)



# hops1 = []
# hops2 = []
# dist1 = []
# dist2 = []
# for s in rural_sub:
#     net = GetDistNet(distpath,s)
#     nodes = [n for n in net if net.nodes[n]['label']!='H']
#     hops1.extend([nx.shortest_path_length(net,n,int(s)) for n in net.nodes \
#             if net.nodes[n]['label']!='H'])
#     dist1.extend([nx.shortest_path_length(net,n,int(s),'geo_length') for n in net.nodes \
#             if net.nodes[n]['label']!='H'])

# w1 = np.ones_like(hops1)/float(len(hops1))

# for s in urban_sub:
#     net = GetDistNet(distpath,s)
#     nodes = [n for n in net if net.nodes[n]['label']!='H']
#     hops2.extend([nx.shortest_path_length(net,n,int(s)) for n in net.nodes \
#             if net.nodes[n]['label']!='H'])
#     dist2.extend([nx.shortest_path_length(net,n,int(s),'geo_length') for n in net.nodes \
#             if net.nodes[n]['label']!='H'])

# w2 = np.ones_like(hops2)/float(len(hops2))

# hops = [hops1,hops2]
# w = [w1,w2]

# #%% Hop distribution
# from matplotlib.ticker import FuncFormatter
# def to_percent(y, position):
#     # Ignore the passed in position. This has the effect of scaling the default
#     # tick locations.
#     s = "{0:.1f}".format(100*y)
#     return s

# colors = ['blue','orangered']
# labels = ['Rural Areas','Urban Areas']
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# ax.hist(hops,weights=w,label=labels,color=colors)
# ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
# ax.set_ylabel("Percentage of nodes",fontsize=20)
# ax.set_xlabel("Hops from root node",fontsize=20)
# ax.legend(prop={'size': 20})
# ax.tick_params(axis='both', labelsize=20)
# fig.savefig("{}{}.png".format(figpath,'hop-comp'),bbox_inches='tight')

# #%% Distance
# dist = [[d/1e3 for d in dist1],[d/1.6e3 for d in dist2]]
# w = [w1,w2]


# from matplotlib.ticker import FuncFormatter
# def to_percent(y, position):
#     # Ignore the passed in position. This has the effect of scaling the default
#     # tick locations.
#     s = "{0:.1f}".format(100*y)
#     return s

# colors = ['blue','orangered']
# labels = ['Rural Areas','Urban Areas']
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# ax.hist(dist,weights=w,label=labels,color=colors)
# ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
# ax.set_ylabel("Percentage of nodes",fontsize=20)
# ax.set_xlabel("Distance (in miles) from root node",fontsize=20)
# ax.legend(prop={'size': 20})
# ax.tick_params(axis='both', labelsize=20)
# fig.savefig("{}{}.png".format(figpath,'dist-comp'),bbox_inches='tight')