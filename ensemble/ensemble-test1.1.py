# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:10:18 2021

@author: rouna
"""

import sys,os
import networkx as nx
from pyqtree import Index
import numpy as np
from shapely.geometry import Point,LineString
from geographiclib.geodesic import Geodesic
from math import log


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
distpath = rootpath + "/primnet/out/prim-network/"
outpath = workpath + "/out/"
sys.path.append(libpath)


from pyExtractDatalib import GetDistNet


# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]
sub = 121144
synth_net = GetDistNet(distpath,sub)
# plot_network(synth_net,with_secnet=True)

#%% Functions
def geodist(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates. The format of each point 
    is (longitude,latitude).
    pt1: shapely point geometry of point 1
    pt2: shapely point geometry of point 2
    '''
    lon1,lat1 = pt1.x,pt1.y
    lon2,lat2 = pt2.x,pt2.y
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

def bounds(pt,radius):
    """
    Returns the bounds for a point geometry. The bound is a square around the
    point with side of 2*radius units.
    
    pt:
        TYPE: shapely point geometry
        DESCRIPTION: the point for which the bound is to be returned
    
    radius:
        TYPE: floating type 
        DESCRIPTION: radius for the bounding box
    """
    return (pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius)
    
def find_nearest_node(center_cord,node_cord):
    """
    Computes the nearest node in the dictionary 'node_cord' to the point denoted
    by the 'center_cord'
    
    center_cord: 
        TYPE: list of two entries
        DESCRIPTION: geographical coordinates of the center denoted by a list
                     of two entries
    
    node_cord: 
        TYPE: dictionary 
        DESCRIPTION: dictionary of nodelist with values as the geographical 
                     coordinate
    """
    xmin,ymin = np.min(np.array(list(node_cord.values())),axis=0)
    xmax,ymax = np.max(np.array(list(node_cord.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    nodes = []
    for i,n in enumerate(list(node_cord.keys())):
        node_geom = Point(node_cord[n])
        node_bound = bounds(node_geom,0.0)
        idx.insert(i,node_bound)
        nodes.append((node_geom, node_bound, n))
    
    pt_center = Point(center_cord)
    center_bd = bounds(pt_center,0.1)
    matches = idx.intersect(center_bd)
    closest_node = min(matches,key=lambda i: geodist(nodes[i][0],pt_center))
    return nodes[closest_node][-1]

class Link(LineString):
    """
    Derived class from Shapely LineString to compute metric distance based on 
    geographical coordinates over geometric coordinates.
    """
    def __init__(self,line_geom):
        """
        """
        super().__init__(line_geom)
        self.geod_length = self.__length()
        return
    
    
    def __length(self):
        '''
        Computes the geographical length in meters between the ends of the link.
        '''
        if self.geom_type != 'LineString':
            print("Cannot compute length!!!")
            return None
        # Compute great circle distance
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length


def create_network(graph,master):
    sec_edges = [e for e in master.edges() \
                 if master[e[0]][e[1]]['label']=='S']
    graph.add_edges_from(sec_edges)
    
    # node attributes
    nodepos = {n:master.nodes[n]['cord'] for n in graph}
    nodelab = {n:master.nodes[n]['label'] for n in graph}
    nodeload = {n:master.nodes[n]['load'] for n in graph}
    nx.set_node_attributes(graph,nodepos,'cord')
    nx.set_node_attributes(graph,nodelab,'label')
    nx.set_node_attributes(graph,nodeload,'load')
    
    # edge attributes
    edge_geom = {}
    edge_label = {}
    edge_r = {}
    edge_x = {}
    glength = {}
    for e in list(graph.edges()):
        if e in master.edges():
            edge_geom[e] = master[e[0]][e[1]]['geometry']
            edge_label[e] = master[e[0]][e[1]]['label']
            edge_r[e] = master[e[0]][e[1]]['r']
            edge_x[e] = master[e[0]][e[1]]['x']
            glength[e] = master[e[0]][e[1]]['geo_length']   
        else:
            edge_geom[e] = LineString((nodepos[e[0]],nodepos[e[1]]))
            glength[e] = Link(edge_geom[e]).geod_length
            edge_label[e] = 'P'
            length = glength[e] if glength[e] != 0.0 else 1e-12
            edge_r[e] = 0.8625/39690 * length
            edge_x[e] = 0.4154/39690 * length
            
    nx.set_edge_attributes(graph, edge_geom, 'geometry')
    nx.set_edge_attributes(graph, edge_label, 'label')
    nx.set_edge_attributes(graph, edge_r, 'r')
    nx.set_edge_attributes(graph, edge_x, 'x')
    nx.set_edge_attributes(graph, glength,'geo_length')
    return

def powerflow(graph):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(graph.edges())]
    nodeload = nx.get_node_attributes(graph,'load')
    
    # Resistance data
    edge_r = nx.get_edge_attributes(graph,'r')
    R = np.diag([1.0/edge_r[e] if e in edge_r else 1.0/edge_r[(e[1],e[0])] \
         for e in list(graph.edges())])
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*nodeload[n] for n in nodelist])
    
    # Voltages and flows
    v = np.matmul(np.linalg.inv(G),p)
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    voltage = {h:1.0-v[i] for i,h in enumerate(nodelist)}
    flows = {e:log(abs(f[i])) for i,e in enumerate(edgelist)}
    subnodes = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = 1.0
    nx.set_node_attributes(graph,voltage,'voltage')
    nx.set_edge_attributes(graph,flows,'flow')
    return

#%% Create new networks
prim_nodes = [n for n in synth_net if synth_net.nodes[n]['label']!='H']
prim_edges = [e for e in synth_net.edges() \
              if synth_net[e[0]][e[1]]['label']!='S']

prim_length = sum([synth_net[e[0]][e[1]]['geo_length'] \
                    for e in prim_edges])
edgelist = [e for e in prim_edges if synth_net[e[0]][e[1]]['label']!='E']

count = 0
num_nets = 1
while(count<num_nets):    
    # Copy graph with shallow copy referencing data from original
    # attributes are referenced but not the structure
    new_network = synth_net.__class__()
    new_network.add_nodes_from(prim_nodes)
    new_network.add_edges_from(prim_edges)
    
    # Select edge at random from the primary network
    
    rand_edge = edgelist[np.random.choice(range(len(edgelist)))]
    new_network.remove_edge(*rand_edge)
    
    # Get the end point unconnected to root node
    # end node: node connected to root
    # other node: node unconnected to root
    # Note that connected node must be a transformer node
    if nx.has_path(new_network,sub,rand_edge[0]):
        end_node = rand_edge[0]
    else:
        end_node = rand_edge[1]
    other_node = rand_edge[0] if end_node==rand_edge[1] else rand_edge[1]
    if synth_net.nodes[end_node]['label']=='R':
        print("Leaf node is road node. Choose another edge.")
    else:
        print("Found an edge")
        # Connect the unconnected node to another nearby node
        comps = list(nx.connected_components(new_network))
        connected_nodes = list(comps[0]) \
            if end_node in list(comps[0]) else list(comps[1])
        dict_node = {n:synth_net.nodes[n]['cord'] for n in connected_nodes \
                     if n!=end_node}
        center_node = synth_net.nodes[other_node]['cord']
        near_node = find_nearest_node(center_node,dict_node)
        new_edge = (near_node,other_node)
        new_network.add_edge(*new_edge)
        count += 1
        create_network(new_network,synth_net)
        # powerflow(new_network)
        # plot_network(new_network,with_secnet=True,
        #              path=figpath+'ensemble-'+str(count)+'-')
        # nx.write_gpickle(new_network,
        #                  outpath+str(sub)+'-ensemble-'+str(count)+'.gpickle')

































