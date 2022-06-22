# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:34:58 2022

Author: Rounak Meyur

Description: Small example of creating synthetic distribution network
"""

import sys,os
from pyBuildNetlib import GetOSMRoads,GetHomes,GetSubstations,MapOSM,groups
from pyBuildNetlib import generate_optimal_topology as generate

workpath = os.getcwd()
tmppath = workpath + "/temp/"

homes = GetHomes('homes-interest.csv')
roads = GetOSMRoads(homes)
subs = GetSubstations('Electric_Substations.csv',homes)

#%% Step 1a: Mapping

# Map the residence to nearest road network link
H2Link = MapOSM(roads).map_point(homes)

# Reverse mapping
L2Home = groups(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]


#%% Step 1b: Secondary network creation
prefix = ''  # Use a prefix for the state and county to avoid confusion
count = 5000001

for link in links:
    # Get the link geometry and home data
    linkgeom = roads.edges[link]['geometry']
    dict_homes = {h:{'cord':homes.cord[h],'load':homes.average[h]} \
              for h in L2Home[link]}
    # Solve the optimization problem and generate forest
    homelist = L2Home[link]
    if len(homelist)>350:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=100,
                                tsfr_max=150,heuristic=3,path=tmppath)
    elif len(homelist)>150:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=80,
                                tsfr_max=100,heuristic=5,path=tmppath)
    elif len(homelist)>80: 
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=40,
                                tsfr_max=60,heuristic=15,path=tmppath)
    else:
        forest,roots = generate(linkgeom,dict_homes,minsep=50,hops=10,
                                tsfr_max=25,heuristic=None,path=tmppath)
    
    # Additional network data
    cord = nx.get_node_attributes(forest,'cord')
    load = nx.get_node_attributes(forest,'load')
    
    # Iterate over nodes to record transformers
    tsfr_data = ''
    dict_tsfr = {}
    rename = {} # to rename the transformers with node IDs
    tsfrlist = [] # to list the transformers
    for n in sorted(list(forest.nodes())):
        if n in roots:
            dict_tsfr[int(prefix+str(count))] = [cord[n][0],cord[n][1],load[n]]
            rename[n] = int(prefix+str(count))
            tsfrlist.append(int(prefix+str(count)))
            tsfr_data += ','.join([str(rename[n]),str(cord[n][0]),str(cord[n][1]),
                                  str(load[n])])+'\n'
            count += 1
    
    # Get edgelist for primary network design
    start_tsf = dict_tsfr[tsfrlist[0]][:2]
    road_cord1 = [roads.nodes[link[0]]['x'],roads.nodes[link[0]]['y']]
    road_cord2 = [roads.nodes[link[1]]['x'],roads.nodes[link[1]]['y']]
    if geodist(start_tsf,road_cord1)<geodist(start_tsf,road_cord2):
        start = link[0]; end = link[1]
    else:
        start = link[1]; end = link[0]
    nodes = [start]+tsfrlist+[end]
    g = nx.Graph()
    nx.add_path(g,nodes)
    edgelist = list(g.edges())
    edgelist_data = '\n'.join([','.join([str(x) for x in edge]) \
                               for edge in edgelist])+'\n'
    
    # Iterate over edges to record secondary network
    network = ''
    for e in list(forest.edges()):
        node1 = str(rename[e[0]]) if e[0] in roots else str(e[0])
        type1 = 'T' if e[0] in roots else 'H'
        cord1a = str(cord[e[0]][0])
        cord1b = str(cord[e[0]][1])
        node2 = str(rename[e[1]]) if e[1] in roots else str(e[1])
        type2 = 'T' if e[1] in roots else 'H'
        cord2a = str(cord[e[1]][0])
        cord2b = str(cord[e[1]][1])
        network += '\t'.join([node1,type1,cord1a,cord1b,
                             node2,type2,cord2a,cord2b])+'\n'
    
    


#%% Check
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
d = {'nodes':[n for n in homes.cord],
     'geometry':[Point(homes.cord[n]) for n in homes.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='crimson',markersize=1,alpha=0.8)

d = {'nodes':[n for n in subs.cord],
     'geometry':[Point(subs.cord[n]) for n in subs.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='blue',markersize=50,alpha=0.8)


edgelist = list(roads.edges(keys=True))
d = {'edges':edgelist,
     'geometry':[roads.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor='black',linewidth=1.0,linestyle='dashed',alpha=0.8)

ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)