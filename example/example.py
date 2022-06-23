# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:34:58 2022

Author: Rounak Meyur

Description: Small example of creating synthetic distribution network. This code
was developed as part of NSF EAGER grant. To reference the synthetic network
generation approach, use:
    
    "Rounak Meyur, Madhav Marathe, Anil Vullikanti, Henning Mortveit, 
    Samarth Swarup, Virgilio Centeno, Arun Phadke, "Creating Realistic Power 
    Distribution Networks using Interdependent Road Infrastructure," in 
    2020 IEEE International Conference on Big Data (Big Data), 2020, 
    pp. 1226-1235, doi: 10.1109/BigData50022.2020.9377959."
    
"""

import sys,os
import networkx as nx
from shapely.geometry import LineString
from pyBuildNetlib import GetOSMRoads,GetHomes,GetSubstations,MapOSM
from pyBuildNetlib import groups, geodist, Link
from pyBuildNetlib import generate_optimal_topology as generate
from pyBuildNetlib import create_voronoi
from pyBuildNetlib import Primary, add_secnet, powerflow, assign_linetype
from pyBuildNetlib import create_shapefile

workpath = os.getcwd()
tmppath = workpath + "/gurobi/"
outpath = workpath + "/output/"


#%% Step 0: Extract dataset
homes = GetHomes('homes.csv')
roads = GetOSMRoads(homes)
subs = GetSubstations('Electric_Substations.csv',homes)

#%% Step 1a: Mapping residences to roads

# Map the residence to nearest road network link
H2Link = MapOSM(roads).map_point(homes)

# Reverse mapping
L2Home = groups(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])]


#%% Step 1b: Secondary network creation
prefix = '51121'  # Use a prefix for the state and county to avoid confusion
count = 5000001

# Initialize list for storing candidate edges for primary network
prim_edges = []
tsfr_data = {}

for k,link in enumerate(links):
    print("Creating secondary network for link",k+1,"out of",len(links))
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
    
    edge_comp = {r:list(nx.subgraph(forest,c).edges) for r in roots \
                 for c in nx.connected_components(forest) if r in c }
    
    # Iterate over nodes to record transformers
    start_tsf = sorted([n for n in forest if n in roots])[0]
    tsfrlist = []
    for n in sorted(forest.nodes):
        if n in roots:
            # Rename edges in the components
            newID = int(prefix+str(count))
            renamed_edges = []
            for e in edge_comp[n]:
                if e[0] == n: renamed_edges.append((newID,e[1]))
                elif e[1] == n: renamed_edges.append((e[0],newID))
                else: renamed_edges.append((e[0],e[1]))
            tsfr_data[newID] = {'secnet':renamed_edges,
                                'cord':forest.nodes[n]['cord']}
            
            tsfrlist.append(newID)
            count += 1
    
    # Get edgelist for primary network design
    start_tsf_loc = forest.nodes[start_tsf]['cord']
    road_cord1 = [roads.nodes[link[0]]['x'],roads.nodes[link[0]]['y']]
    road_cord2 = [roads.nodes[link[1]]['x'],roads.nodes[link[1]]['y']]
    if geodist(start_tsf_loc,road_cord1)<geodist(start_tsf_loc,road_cord2):
        start = link[0]; end = link[1]
    else:
        start = link[1]; end = link[0]
    nodes = [start]+tsfrlist+[end]
    
    # Construct the graph
    g = nx.Graph()
    nx.add_path(g,nodes)
    prim_edges += [(e[0],e[1],0) for e in g.edges]
    print("\n\n\n")
    
    
#%% Step 2a: Mapping roads/transformers to substations
new_road = roads.__class__()
new_road.add_edges_from(roads.edges)
new_road.remove_edges_from(links)
new_road.add_edges_from(prim_edges)
road_edges = [(e[0],e[1]) for e in new_road.edges]

primroad = nx.Graph()
primroad.add_edges_from(road_edges)

# Add node attributes
nodelabel = {}
nodecord = {}
nodeload = {}
for n in primroad:
    if n in tsfr_data:
        nodelabel[n] = 'T'
        nodecord[n] = tsfr_data[n]['cord']
        res_nodes = [k for edge in tsfr_data[n]['secnet'] for k in list(edge) if k!=n]
        nodeload[n] = sum([homes.average[h] for h in res_nodes])
    else:
        nodelabel[n] = 'R'
        nodecord[n] = [roads.nodes[n]['x'],roads.nodes[n]['y']]
        nodeload[n] = 0.0

nx.set_node_attributes(primroad, nodelabel, 'label')
nx.set_node_attributes(primroad, nodecord, 'cord')
nx.set_node_attributes(primroad, nodeload, 'load')

for e in road_edges:
    if ((e[0],e[1],0) in prim_edges) or ((e[1],e[0],0) in prim_edges):
        primroad[e[0]][e[1]]['geometry'] = LineString((primroad.nodes[e[0]]["cord"],
                                                    primroad.nodes[e[1]]["cord"]))
    else:
        ident = list(roads[e[0]][e[1]].keys())[0]
        primroad[e[0]][e[1]]['geometry'] = roads[e[0]][e[1]][ident]['geometry']
    primroad[e[0]][e[1]]['length'] = Link(primroad[e[0]][e[1]]['geometry']).geod_length

# Perform the Voronoi partitioning
S2Near,S2Node = create_voronoi(subs,primroad)

#%% Step 2b: Primary network creation
prim_net = nx.Graph()
for s in S2Node:
    sub_data = {"id":int(s),"near":S2Near[s],"cord":subs.cord[s]}
    master_graph = nx.subgraph(primroad,S2Node[s])

    # Generate primary distribution network by partitions
    P = Primary(sub_data,master_graph)
    dist_net = P.get_sub_network(grbpath=tmppath)
    prim_net = nx.compose(prim_net,dist_net)


#%% Step 2c: Generate shape files
# Add secondary network to primary
dist_net = add_secnet(prim_net,tsfr_data,homes)
# Run power flow and add line types
powerflow(dist_net)
assign_linetype(dist_net)

# Save shapefiles
create_shapefile(dist_net,outpath)

#%% Check by ploting
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
d = {'nodes':[n for n in homes.cord],
     'geometry':[Point(homes.cord[n]) for n in homes.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='crimson',markersize=50,alpha=0.8)

d = {'nodes':[n for n in subs.cord],
     'geometry':[Point(subs.cord[n]) for n in subs.cord]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_nodes.plot(ax=ax,color='blue',markersize=100,alpha=0.8)


edgelist = list(roads.edges(keys=True))
d = {'edges':edgelist,
     'geometry':[roads.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
df_edges.plot(ax=ax,edgecolor='black',linewidth=1.0,linestyle='dashed',alpha=0.8)

ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
