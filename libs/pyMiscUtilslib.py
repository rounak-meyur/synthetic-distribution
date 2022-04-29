# -*- coding: utf-8 -*-
"""
Created on Tue Sept 15 10:19:05 2021

@author: Rounak Meyur
Description: Library of functions for miscellaneous applications
"""


from collections import defaultdict
from shapely.geometry import LineString
import networkx as nx
import numpy as np
from math import log,exp
from pyGeometrylib import Link
from pyExtractDatalib import GetSecnet,GetHomes


#%% Functions on data structures
def groups(many_to_one):
    """Converts a many-to-one mapping into a one-to-many mapping.

    `many_to_one` must be a dictionary whose keys and values are all
    :term:`hashable`.

    The return value is a dictionary mapping values from `many_to_one`
    to sets of keys from `many_to_one` that have that value.

    """
    one_to_many = defaultdict(set)
    for v, k in many_to_one.items():
        one_to_many[k].add(v)
    D = dict(one_to_many)
    return {k:list(D[k]) for k in D}


#%% Functions on total distribution network
def get_load(graph):
    """
    Get a dictionary of loads corresponding to each transformer node in the 
    graph. This data will be used as a proxy for the secondary network.

    Parameters
    ----------
    graph : networkx Graph
        The optimal distribution network comprising of primary and secondary.

    Returns
    -------
    LOAD : dictionary of loads
        The load supplied by each transformer node.
    """
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H']
    sub = [n for n in graph if graph.nodes[n]['label']=='S'][0]
    res2tsfr = {h:[n for n in nx.shortest_path(graph,h,sub) if n in tnodes][0] \
                for h in hnodes}
    tsfr2res = groups(res2tsfr)
    LOAD = {t:sum([graph.nodes[n]['load'] for n in tsfr2res[t]]) for t in tnodes}
    return LOAD

def get_secnet(graph,secpath,homepath):
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    fislist = list(set([str(x)[2:5] for x in tnodes]))
    
    # Get all secondary networks
    secnet = nx.Graph()
    hcord = {}; hload = {}
    for fis in fislist:
        g = GetSecnet(secpath, fis)
        secnet = nx.compose(secnet,g)
        h = GetHomes(homepath,fis)
        hcord.update(h.cord); hload.update(h.average)
    
    # Extract only associated secondaries
    sec_graph = nx.Graph()
    comps = list(nx.connected_components(secnet))
    for t in tnodes:
        comp = [c for c in comps if t in c][0]
        g = secnet.subgraph(comp)
        sec_graph = nx.compose(sec_graph,g)
    
    # Add the secondary network to the primary network
    graph = nx.compose(graph,sec_graph)
    
    # Add new node attributes
    hnodes = [n for n in sec_graph if n not in tnodes]
    for n in hnodes:
        graph.nodes[n]['cord'] = secnet.nodes[n]['cord']
        graph.nodes[n]['label'] = 'H'
        if n in hload:
            graph.nodes[n]['load'] = hload[n]
        else:
            # use a sample house as the load
            graph.nodes[n]['load'] = hload[[n for n in hload][0]]
    
    for e in graph.edges:
        if e in sec_graph.edges:
            graph.edges[e]['geometry'] = LineString((graph.nodes[e[0]]["cord"],
                                                     graph.nodes[e[1]]["cord"]))
            graph.edges[e]['length'] = Link(graph.edges[e]['geometry']).geod_length
            graph.edges[e]['label'] = 'S'
            graph.edges[e]['r'] = 0.81508/57.6 * graph.edges[e]['length']
            graph.edges[e]['x'] = 0.34960/57.6 * graph.edges[e]['length']
    return graph

def powerflow(graph,v0=1.0):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    # Pre-processing to rectify incorrect code
    hv_lines = [e for e in graph.edges if graph.edges[e]['label']=='E']
    for e in hv_lines:
        try:
            length = graph.edges[e]['length']
        except:
            length = graph.edges[e]['geo_length']
        graph.edges[e]['r'] = (0.0822/363000)*length*1e-3
        graph.edges[e]['x'] = (0.0964/363000)*length*1e-3
    
    # Main function begins here
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(graph.edges())]
    
    # Resistance data
    edge_r = []
    for e in graph.edges:
        try:
            edge_r.append(1.0/graph.edges[e]['r'])
        except:
            edge_r.append(1.0/1e-14)
    R = np.diag(edge_r)
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*graph.nodes[n]['load'] for n in nodelist])
    
    # Voltages and flows
    v = np.matmul(np.linalg.inv(G),p)
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    voltage = {n:v0-v[i] for i,n in enumerate(nodelist)}
    flows = {e:log(abs(f[i])+1e-10) for i,e in enumerate(edgelist)}
    subnodes = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = v0
    nx.set_node_attributes(graph,voltage,'voltage')
    nx.set_edge_attributes(graph,flows,'flow')
    return

def assign_linetype(graph):
    prim_amps = {e:2.2*exp(graph[e[0]][e[1]]['flow'])/6.3 \
                 for e in graph.edges if graph[e[0]][e[1]]['label']=='P'}
    sec_amps = {e:1.5*exp(graph[e[0]][e[1]]['flow'])/0.12 \
                for e in graph.edges if graph[e[0]][e[1]]['label']=='S'}
    
    
    edge_name = {}
    for e in graph.edges:
        # names of secondary lines
        if graph[e[0]][e[1]]['label']=='S':
            if sec_amps[e]<=95:
                edge_name[e] = 'OH_Voluta'
                r = 0.661/57.6; x = 0.033/57.6
            elif sec_amps[e]<=125:
                edge_name[e] = 'OH_Periwinkle'
                r = 0.416/57.6; x = 0.031/57.6
            elif sec_amps[e]<=165:
                edge_name[e] = 'OH_Conch'
                r = 0.261/57.6; x = 0.03/57.6
            elif sec_amps[e]<=220:
                edge_name[e] = 'OH_Neritina'
                r = 0.164/57.6; x = 0.03/57.6
            elif sec_amps[e]<=265:
                edge_name[e] = 'OH_Runcina'
                r = 0.130/57.6; x = 0.029/57.6
            else:
                edge_name[e] = 'OH_Zuzara'
                r = 0.082/57.6; x = 0.027/57.6
        
        # names of primary lines
        elif graph[e[0]][e[1]]['label']=='P':
            if prim_amps[e]<=140:
                edge_name[e] = 'OH_Swanate'
                r = 0.407/39690; x = 0.113/39690
            elif prim_amps[e]<=185:
                edge_name[e] = 'OH_Sparrow'
                r = 0.259/39690; x = 0.110/39690
            elif prim_amps[e]<=240:
                edge_name[e] = 'OH_Raven'
                r = 0.163/39690; x = 0.104/39690
            elif prim_amps[e]<=315:
                edge_name[e] = 'OH_Pegion'
                r = 0.103/39690; x = 0.0992/39690
            else:
                edge_name[e] = 'OH_Penguin'
                r = 0.0822/39690; x = 0.0964/39690
        else:
            edge_name[e] = 'OH_Penguin'
            r = 0.0822/363000; x = 0.0964/363000
        
        # Assign new resitance and reactance
        try:
            l = graph.edges[e]['length']
        except:
            l = graph.edges[e]['geo_length']
        graph.edges[e]['r'] = r * l * 1e-3
        graph.edges[e]['x'] = x * l * 1e-3
    
    # Add new edge attribute
    nx.set_edge_attributes(graph,edge_name,'type')
    return


#%% Creating variant networks
def create_variant_network(net,road,prim_edges):
    """
    Alters the input network to create a variant version of it. It is used to
    create variant networks in the ensemble of networks

    Parameters
    ----------
    old_net : networkx graph
        input distribution network.
    road : networkx graph
        underlying road network.
    prim_edges : list of edge tuples
        list of primary network edges in the new network.

    Returns
    -------
    None.
    Essentially returns the altered graph.

    """
    # Get node and edge list with attributes
    sec_edges = [e for e in net.edges if net.edges[e]['label']=='S']
    hv_edges = [e for e in net.edges if net.edges[e]['label']=='E']
    
    # Construct the variant network
    variant = nx.Graph()
    variant.add_edges_from(sec_edges+hv_edges+prim_edges)
    
    # Add node attributes
    for n in variant:
        if n in net:
            variant.nodes[n]['cord'] = net.nodes[n]['cord']
            variant.nodes[n]['load'] = net.nodes[n]['load']
            variant.nodes[n]['label'] = net.nodes[n]['label']
        elif n in road:
            variant.nodes[n]['cord'] = road.nodes[n]['cord']
            variant.nodes[n]['load'] = road.nodes[n]['load']
            variant.nodes[n]['label'] = road.nodes[n]['label']
        else:
            print("Warning!!! Unknown node information")
    for e in variant.edges:
        # Edge geometry and length
        variant.edges[e]['geometry'] = LineString((variant.nodes[e[0]]['cord'],
                                                 variant.nodes[e[1]]['cord']))
        variant.edges[e]['length'] = Link(variant.edges[e]['geometry']).geod_length
        length = variant.edges[e]['length']
        # Edge labels
        if (e in sec_edges) or ((e[1],e[0]) in sec_edges):
            variant.edges[e]['label'] = 'S'
            variant.edges[e]['r'] = (0.082/57.6) * length*1e-3
            variant.edges[e]['x'] = (0.027/57.6) * length*1e-3
        elif (e in prim_edges) or ((e[1],e[0]) in prim_edges):
            variant.edges[e]['label'] = 'P'
            variant.edges[e]['r'] = (0.0822/39690)*length*1e-3
            variant.edges[e]['x'] = (0.0964/39690)*length*1e-3
        elif (e in hv_edges) or ((e[1],e[0]) in hv_edges):
            variant.edges[e]['label'] = 'E'
            variant.edges[e]['r'] = (0.0822/363000)*length*1e-3
            variant.edges[e]['x'] = (0.0964/363000)*length*1e-3
        else:
            print("Warning!!! Unknown edge information")
    
    # Run powerflow
    powerflow(variant)
    assign_linetype(variant)
    return variant


