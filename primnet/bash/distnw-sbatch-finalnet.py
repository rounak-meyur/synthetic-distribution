# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:44:07 2021

Author: Rounak

Description: Removes road nodes from the created synthetic network and just 
creates a network consisting of transformer nodes. 
"""

import sys,os
import networkx as nx
from shapely.geometry import LineString, MultiLineString

workPath = os.getcwd()
libPath = workPath + "/Libraries/"
sys.path.append(libPath)
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
distpath = scratchPath + "/temp/osm-prim-network/"



from pyExtractDatalib import GetDistNet
from pyBuildPrimNetlib import assign_linetype, powerflow
from pyGeometrylib import Link

print("Imported modules")



def create_final_network(path,sub):
    """
    Creates the final version of the network with only the transformer nodes, 
    substation node and the road nodes where the voltage regulator are placed.
    The residence nodes are also connected to the transformer.

    Parameters
    ----------
    path : string
        path for the network gpickle file.
    sub : integer
        substation ID of the network.

    Returns
    -------
    graph : networkx Graph
        Final graph with all required edge and node properties.

    """
    # Load network created by optimization framework
    synth_net = GetDistNet(path,sub)
    reg_nodes = list(nx.neighbors(synth_net, sub))
    tnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='T']
    rnodes = [n for n in synth_net if synth_net.nodes[n]['label']=='R' and n not in reg_nodes]
    
    # Separate the secondary network edges
    sec_edges = [e for e in synth_net.edges if synth_net.edges[e]['label']=='S']
    
    # Remove unnecessary road nodes
    nodelist = [sub]
    prim_edges = []
    
    for t in tnodes:
        if t not in nodelist:
            nodes = [v for v in nx.shortest_path(synth_net,sub,t) if v not in rnodes]
            edges = [(nodes[i],nodes[i+1]) for i,_ in enumerate(nodes[:-1])]
            nodelist.extend(nodes[1:])
            prim_edges.extend(edges)
    
    # Construct final graph
    graph = nx.Graph()
    graph.add_edges_from(prim_edges+sec_edges)
    
    
    # Add edge properties of primary network
    for edge in prim_edges:
        path = nx.shortest_path(synth_net,edge[0],edge[1])
        path_geom = MultiLineString([synth_net[path[i]][path[i+1]]['geometry'] \
                     for i,_ in enumerate(path[:-1])])
        out_coords = [list(i.coords) for i in path_geom]
        
        # Edge data
        graph.edges[edge]['geometry'] = LineString([i for sublist in out_coords for i in sublist])
        graph.edges[edge]['geo_length'] = Link(graph.edges[edge]['geometry']).geod_length
        if sub in edge:
            graph.edges[edge]['label'] = 'E'
            graph.edges[edge]['r'] = 1e-12 * graph.edges[edge]['geo_length']
            graph.edges[edge]['x'] = 1e-12 * graph.edges[edge]['geo_length']
        else:
            graph.edges[edge]['label'] = 'P'
            graph.edges[edge]['r'] = 0.8625/39690 * graph.edges[edge]['geo_length']
            graph.edges[edge]['x'] = 0.4154/39690 * graph.edges[edge]['geo_length']
    
    # Add edge properties of secondary network
    for edge in sec_edges:
        graph.edges[edge]['geometry'] = synth_net.edges[edge]['geometry']
        graph.edges[edge]['label'] = 'S'
        graph.edges[edge]['geo_length'] = Link(graph.edges[edge]['geometry']).geod_length
        graph.edges[edge]['r'] = 0.81508/57.6 * graph.edges[edge]['geo_length']
        graph.edges[edge]['x'] = 0.3496/57.6 * graph.edges[edge]['geo_length']
        
    for n in graph.nodes:
        graph.nodes[n]['cord'] = synth_net.nodes[n]['cord']
        graph.nodes[n]['label'] = synth_net.nodes[n]['label']
        graph.nodes[n]['load'] = synth_net.nodes[n]['load']
    
    # Run power flow and store the flows and voltages
    powerflow(graph)
    
    # Assign line types
    assign_linetype(graph)
    return graph


sub=int(sys.argv[1])
prim_net = create_final_network(distpath,sub)
nx.write_gpickle(prim_net,distpath+str(sub)+'-prim-dist.gpickle')
    