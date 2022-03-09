# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak

Description: This program plots the results of motifs in distribution network 
of Virginia.
"""

import sys,os
import matplotlib.pyplot as plt



workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
distpath = workpath + "/out/osm-primnet/"

sys.path.append(libpath)
print("Imported modules")


#%% Get substations in rural and urban regions
sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
with open(inppath+"urban-sublist.txt") as f:
    urban_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]
with open(inppath+"rural-sublist.txt") as f:
    rural_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]

# with open(inppath+"sublist.txt") as f:
#     rem_sublist = [int(x) for x in f.readlines()[0].strip('\n').split(' ')]

rural_sub = [s for s in sublist if s in rural_sublist]
urban_sub = [s for s in sublist if s in urban_sublist]

# remlist = {'R':len([s for s in rem_sublist if s in rural_sublist]),
#            'U':len([s for s in rem_sublist if s in urban_sublist])}


#%% Get motif data in rural and urban areas
with open(workpath+"/out/4star-motif.txt") as f:
    lines = f.readlines()
kstar_motif = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    kstar_motif[int(temp[0])] = (int(temp[1]),int(temp[2]))
    
with open(workpath+"/out/4path-motif.txt") as f:
    lines = f.readlines()
kpath_motif = {}
for line in lines:
    temp = line.strip('\n').split('\t')
    kpath_motif[int(temp[0])] = (int(temp[1]),int(temp[2]))


def scatter_plot(subs,ax,motifs,color='r',size=10,label='all areas'):
    all_nodes = [motifs[s][0] for s in subs]
    star_motifs = [motifs[s][1] for s in subs]
    ax.scatter(all_nodes,star_motifs,c=color,s=size,label=label)
    return


#%% 
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)

scatter_plot(urban_sub,ax,kstar_motif,color='red',size=300,label='urban areas')
scatter_plot(rural_sub,ax,kstar_motif,color='blue',size=300,label='rural areas')

ax.set_xlabel("Number of nodes",fontsize=60)
ax.set_ylabel("Number of star motifs",fontsize=60)
ax.set_title("4-node star motifs",fontsize=60)

ax.legend(prop={'size': 50})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,'4star-motif-comp'),bbox_inches='tight')





fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)

scatter_plot(urban_sub,ax,kpath_motif,color='red',size=300,label='urban areas')
scatter_plot(rural_sub,ax,kpath_motif,color='blue',size=300,label='rural areas')

ax.set_xlabel("Number of nodes",fontsize=60)
ax.set_ylabel("Number of star motifs",fontsize=60)
ax.set_title("4-node path motifs",fontsize=60)

ax.legend(prop={'size': 50})
ax.tick_params(axis='both', labelsize=50)
fig.savefig("{}{}.png".format(figpath,'4path-motif-comp'),bbox_inches='tight')












#%% 4-node Motif Counter Visualization on Map

# import geopandas as gpd
# import networkx as nx
# from shapely.geometry import Point
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import cm

# with open(workpath+"/motif-count.txt",'r') as f:
#     data = f.readlines()
# motif = {}
# for temp in data:
#     sub = int(temp.strip("\n").split('\t')[0])
#     motif_count = temp.strip("\n").split('\t')[1].split(' ')
#     motif[sub] = {'path':int(motif_count[0]),'star':int(motif_count[1])}

# nodelist = []
# edgelist = []
# node_geom = []
# edge_geom = []
# path_motif_node = []
# star_motif_node = []
# path_motif_edge = []
# star_motif_edge = []

# for sub in motif:
#     dist = GetDistNet(distpath,sub)
#     nodelist.extend(dist.nodes)
#     edgelist.extend(dist.edges)
#     node_geom.extend([Point(dist.nodes[n]['cord']) for n in dist])
#     edge_geom.extend([dist.edges[e]['geometry'] for e in dist.edges])
#     path_motif_node.extend([motif[sub]["path"] for _ in dist])
#     star_motif_node.extend([motif[sub]["star"] for _ in dist])
#     path_motif_edge.extend([motif[sub]["path"] for _ in dist.edges])
#     star_motif_edge.extend([motif[sub]["star"] for _ in dist.edges])


# df_nodes = gpd.GeoDataFrame({'nodes':nodelist,'geometry':node_geom,
#                              'path_motif':path_motif_node,
#                              'star_motif':star_motif_node},crs="EPSG:4326")
# df_edges = gpd.GeoDataFrame({'edges':edgelist,'geometry':edge_geom,
#                              'path_motif':path_motif_edge,
#                              'star_motif':star_motif_edge},crs="EPSG:4326")

# #%% Plot the motif count
# fig = plt.figure(figsize=(35,30),dpi=72)
# ax = fig.add_subplot(1,1,1)
# df_edges.plot(ax=ax,color="red",linewidth=2)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.001)
# df_nodes.plot(ax=ax,column='path_motif',markersize=20.0,cmap=cm.plasma,
#               vmin=100,vmax=2000,cax=cax,legend=True)
# df_edges.plot(column='path_motif',ax=ax,cmap=cm.plasma,vmin=100,vmax=2000)
# cax.set_ylabel("Number of 4-node path motifs",fontsize=50)
# cax.tick_params(labelsize=30)
# ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

#%% Hop counter from root node

# nodelist = []
# node_geom = []
# node_hop = []
# node_reach = []

# mont_sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 
#                 148723, 150353, 150589, 150638, 150692, 150722, 150723, 150724, 
#                 150725, 150726, 150727, 150728]

# sublist = [int(x.strip('-dist-net.gpickle')) for x in os.listdir(distpath)]


# for sub in sublist:
#     print("Working on network: "+str(sub))
#     dist = GetDistNet(distpath,sub)
#     nodelist.extend(dist.nodes)
#     node_geom.extend([Point(dist.nodes[n]['cord']) for n in dist])
#     node_hop.extend([nx.shortest_path_length(dist,sub,n) for n in dist])
#     node_reach.extend([(1.0/1609.34)*nx.shortest_path_length(dist,sub,n,weight="length") \
#                        for n in dist])


# df_nodes = gpd.GeoDataFrame({'nodes':nodelist,'geometry':node_geom,
#                               'hop':node_hop,'reach':node_reach},crs="EPSG:4326")

# df_nodes.to_file(workpath+"va-stat-989.shp")
#%% Plot the motif count
# fig = plt.figure(figsize=(100,60),dpi=72)
# ax = fig.add_subplot(1,1,1)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.001)
# df_nodes.plot(ax=ax,column='hop',markersize=20.0,cmap=cm.viridis,
#               vmin=0,vmax=max(node_hop),cax=cax,legend=True)
# cax.set_ylabel("Number of hops from substation",fontsize=50)
# cax.tick_params(labelsize=50)
# ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# fig.savefig("{}{}.png".format(figpath,'va-hops'),bbox_inches='tight')

# fig = plt.figure(figsize=(100,60),dpi=72)
# ax = fig.add_subplot(1,1,1)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.001)
# df_nodes.plot(ax=ax,column='reach',markersize=20.0,cmap=cm.viridis,
#               vmin=0,vmax=max(node_reach),cax=cax,legend=True)
# cax.set_ylabel("Distance from substation along network (in miles)",fontsize=50)
# cax.tick_params(labelsize=50)
# ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# fig.savefig("{}{}.png".format(figpath,'va-reach'),bbox_inches='tight')




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