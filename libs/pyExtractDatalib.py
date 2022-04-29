# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:46:36 2018

Authors: Rounak Meyur

Description: This library contains classes, methods etc to extract data from
the noldor database with required credentials. 
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString,Point
from collections import namedtuple as nt
from pyqtree import Index
from pyGeometrylib import Link, MeasureDistance



#%% Navteq HERE data
def combine_components(graph,cords,radius = 0.01):
    """
    Combines network components by finding nearest nodes
    based on a QD Tree Approach.

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Initialize QD Tree
    xmin,ymin = np.min(np.array(list(cords.values())),axis=0)
    xmax,ymax = np.max(np.array(list(cords.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    # Differentiate large and small components
    comps = [c for c in list(nx.connected_components(graph))]
    lencomps = [len(c) for c in list(nx.connected_components(graph))]
    indlarge = lencomps.index(max(lencomps))
    node_main = list(graph.subgraph(comps[indlarge]).nodes())
    del(comps[indlarge])
    
    # keep track of lines so we can recover them later
    nodes = []
    
    # create bounding box around each point in large component
    for i,node in enumerate(node_main):
        pt = Point(cords[node])
        pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
        idx.insert(i, pt_bounds)
        nodes.append((pt, pt_bounds, node))
        
    # find intersection and add edges
    edgelist = []
    for c in comps:
        node_comp = list(graph.subgraph(c).nodes())
        nodepairs = []
        for n in node_comp:
            pt = Point(cords[n])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = idx.intersect(pt_bounds)
            closest_pt = min(matches,key=lambda i: nodes[i][0].distance(pt))
            nodepairs.append((n,nodes[closest_pt][-1]))
        dist = [MeasureDistance(cords[p[0]],cords[p[1]]) for p in nodepairs]
        edgelist.append(nodepairs[np.argmin(dist)])
    return edgelist


def GetRoads(path,fis,level=[1,2,3,4,5],thresh=0):
    """
    """
    corefile = "nrv/core-link-file-" + fis + ".txt"
    linkgeom = "nrv/link-file-" + fis + ".txt"
    nodegeom = "nrv/node-geometry-" + fis + ".txt"
    
    datalink = {}
    roadcord = {}
    edgelist = []
    
    # Get edgelist from the core link file
    df_core = pd.read_table(path+corefile,header=0,
                    names=['src','dest','fclass'])
    for i in range(len(df_core)):
        edge = (df_core.loc[i,'src'],df_core.loc[i,'dest'])
        fclass = df_core.loc[i,'fclass']
        if (edge not in edgelist) and ((edge[1],edge[0]) not in edgelist):
            edgelist.append(edge)
            datalink[edge] = {'level':fclass,'geometry':None}
    
    # Get node coordinates from the node geometry file            
    df_node = pd.read_table(path+nodegeom,header=0,
                            names=['id','long','lat'])
    roadcord = dict([(n.id, (n.long, n.lat)) \
                     for n in df_node.itertuples()])
    
    # Get link geometry from the link geometry file
    colnames = ['ref_in_id','nref_in_id','the_geom']
    coldtype = {'ref_in_id':'Int64','nref_in_id':'Int64','the_geom':'str'}
    df_link = pd.read_table(path+linkgeom,sep=',',
                            usecols=colnames,dtype=coldtype)
    for i in range(len(df_link)):
        edge = (df_link.loc[i,'ref_in_id'],df_link.loc[i,'nref_in_id'])
        pts = [tuple([float(x) for x in pt.split(' ')]) \
                for pt in df_link.loc[i,'the_geom'].lstrip('MULTILINESTRING((').rstrip('))').split(',')]
        geom = LineString(pts)
        if (edge in edgelist):
            datalink[edge]['geometry']=geom
        elif ((edge[1],edge[0]) in edgelist):
            datalink[(edge[1],edge[0])]['geometry']=geom
        else:
            print(','.join([str(x) for x in list(edge)])+": not in edgelist")
    
    
    # Update the link dictionary with missing geometries    
    for edge in datalink:
        if datalink[edge]['geometry']==None:
            pts = [tuple(roadcord[r]) for r in list(edge)]
            geom = LineString(pts)
            datalink[edge]['geometry'] = geom
    
    # Create networkx graph from edges in the level list
    listedge = [edge for edge in edgelist if datalink[edge]['level'] in level]
    graph = nx.Graph(listedge)
    
    # Join disconnected components in the road network
    new_edges = combine_components(graph,roadcord,radius=0.1)
    graph.add_edges_from(new_edges)
    
    # Update graph attributes
    datalink.update({edge:{'level':5,'geometry':LineString([tuple(roadcord[r]) \
                      for r in list(edge)])} for edge in new_edges})
    road_x = {r:roadcord[r][0] for r in graph}
    road_y = {r:roadcord[r][1] for r in graph}
    edge_geom = {e:datalink[e]['geometry'] for e in datalink}
    
    nx.set_node_attributes(graph,road_x,'x')
    nx.set_node_attributes(graph,road_y,'y')
    nx.set_edge_attributes(graph,edge_geom,'geometry')
    return graph


#%% Open Street Maps data
def combine_osm_components(road,radius = 0.01):
    """
    Combines network components by finding nearest nodes
    based on a QD Tree Approach.

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Initialize QD Tree
    longitudes = [road.nodes[n]['x'] for n in road.nodes()]
    latitudes = [road.nodes[n]['y'] for n in road.nodes()]
    xmin = min(longitudes); xmax = max(longitudes)
    ymin = min(latitudes); ymax = max(latitudes)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    # Differentiate large and small components
    comps = [c for c in list(nx.connected_components(road))]
    lencomps = [len(c) for c in list(nx.connected_components(road))]
    indlarge = lencomps.index(max(lencomps))
    node_main = list(road.subgraph(comps[indlarge]).nodes())
    del(comps[indlarge])
    
    # keep track of nodes so we can recover them later
    nodes = []
    
    # create bounding box around each point in large component
    for i,node in enumerate(node_main):
        pt = Point([road.nodes[node]['x'],road.nodes[node]['y']])
        pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
        idx.insert(i, pt_bounds)
        nodes.append((pt, pt_bounds, node))
        
    # find intersection and add edges
    edgelist = []
    for c in comps:
        node_comp = list(road.subgraph(c).nodes())
        nodepairs = []
        for n in node_comp:
            pt = Point([road.nodes[n]['x'],road.nodes[n]['y']])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = idx.intersect(pt_bounds)
            closest_pt = min(matches,key=lambda i: nodes[i][0].distance(pt))
            nodepairs.append((n,nodes[closest_pt][-1]))
        
        # Get the geodesic distance
        dist = [MeasureDistance([road.nodes[p[0]]['x'],road.nodes[p[0]]['y']],
                                [road.nodes[p[1]]['x'],road.nodes[p[1]]['y']]) \
                for p in nodepairs]
        edgelist.append(nodepairs[np.argmin(dist)]+tuple([0]))
    return edgelist


def GetOSMRoads(path,fis,statefp="51"):
    county_file = "tl_2018_us_county.shp"
    data_counties = gpd.read_file(path+"census/"+county_file)
    data_va = data_counties.loc[data_counties["STATEFP"]==statefp]
    
    # Get the OSM links within the county polygon
    county_polygon = list(data_va[data_va.COUNTYFP==fis].geometry.items())[0][1]
    osm_graph = ox.graph_from_polygon(county_polygon, retain_all=True,
                                  truncate_by_edge=True)
    
    G = osm_graph.to_undirected()
    
    # Add geometries for links without it
    edge_nogeom = [e for e in G.edges(keys=True) if 'geometry' not in G.edges[e]]
    for e in edge_nogeom:
        pts = [(G.nodes[e[0]]['x'],G.nodes[e[0]]['y']),
               (G.nodes[e[1]]['x'],G.nodes[e[1]]['y'])]
        link_geom = LineString(pts)
        G.edges[e]['geometry'] = link_geom
    
    # Join disconnected components in the road network
    new_edges = combine_osm_components(G,radius=0.1)
    G.add_edges_from(new_edges)
    for i,e in enumerate(new_edges):
        pts = [(G.nodes[e[0]]['x'],G.nodes[e[0]]['y']),
               (G.nodes[e[1]]['x'],G.nodes[e[1]]['y'])]
        link_geom = LineString(pts)
        G.edges[e]['geometry'] = link_geom
        G.edges[e]['length'] = Link(link_geom).geod_length
        G.edges[e]['oneway'] = float('nan')
        G.edges[e]['highway'] = 'extra'
        G.edges[e]['name'] = 'extra'
        G.edges[e]['osmid'] = fis+str(80000+i)
    return G
    

def GetHomes(path,fis):
    '''
    '''
    df_home = pd.read_csv(path+'load/'+fis+'-home-load.csv')
    df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","average"])
    dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
    dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
    dict_peak = dict(zip(df_home.hid,df_home.peak))
    dict_avg = dict(zip(df_home.hid,df_home.average))
    homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
    return homes


def GetSubstations(path,areas=None,state_fis='51',
                   sub_file='Electric_Substations.shp',
                    county_file='tl_2018_us_county.shp',
                    state_file = 'states.shp'):
    """
    Gets the list of substations within the area polygons
    
    Parameters
    ----------
    path : string
        The path where all input data (eia, county etc.) is stored.
    areas : list of strings, optional
        The list of county fips code for which the substations are to be 
        extracted. The default is ['121'] for Montgomery county.
    state_fis : string, optional
        The state fips code for which the substation is to be extracted.
        The default is '51' for state of VIRGINIA.
    sub_file : string, optional
        The shape file with geographic information about the substations.
        The default is 'Electric_Substations.shp'.
    county_file : string, optional
        The shape file with geographic information about the county boundaries.
        The default is 'tl_2018_us_county.shp'.
    
    Returns
    -------
    subs : named tuple of substation.
        A record of substations and their corresponding coordinates.
    
    """
    subs = nt("substation",field_names=["cord"])
    data_substations = gpd.read_file(path+'eia/'+sub_file)
    data_counties = gpd.read_file(path+'census/'+county_file)
    data_states = gpd.read_file(path+'census/'+state_file)
    
    if areas == None:
        state_polygon = list(data_states[data_states.STATE_FIPS == state_fis].geometry.items())[0][1]
        df_subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
        cord = dict([(t.ID, (t.LONGITUDE, t.LATITUDE)) for t in df_subs.itertuples()])
        dict_cord = {int(k):cord[k] for k in cord}
    else:
        dict_cord = {}
        data_county_state = data_counties.loc[data_counties['STATEFP']==state_fis]
        for area in areas:
            county_polygon = list(data_county_state[data_county_state.COUNTYFP == area].geometry.items())[0][1]
            df_subs = data_substations.loc[data_substations.geometry.within(county_polygon)]
            cord = dict([(t.ID, (t.LONGITUDE, t.LATITUDE)) for t in df_subs.itertuples()])
            cord = {int(k):cord[k] for k in cord}
            dict_cord.update(cord)
    return subs(cord=dict_cord)


def GetTransformers(path,fis):
    """
    Gets the network of local transformers in the county/city
    """
    df_tsfr = pd.read_csv(path+fis+'-tsfr-data.csv',
                          header=None,names=['tid','long','lat','load'])
    tsfr = nt("Transformers",field_names=["cord","load","graph"])
    dict_cord = dict([(t.tid, (t.long, t.lat)) for t in df_tsfr.itertuples()])
    dict_load = dict([(t.tid, t.load) for t in df_tsfr.itertuples()])
    df_tsfr_edges = pd.read_csv(path+fis+'-tsfr-net.csv',
                                header=None,names=['source','target'])
    g = nx.from_pandas_edgelist(df_tsfr_edges)
    return tsfr(cord=dict_cord,load=dict_load,graph=g)

def GetMappings(path,fis):
    """
    Gets the mapping between residences and road network links in the area

    Parameters
    ----------
    path : TYPE string
        The path where the txt file is saved and is to be read.
    fis : TYPE string
        The FIPS ID of the area under consideration.

    Returns
    -------
    links : TYPE list of tuples
        list of edges formated as tuples.

    """
    with open(path+fis+'-link2home.txt') as f:
        lines = f.readlines()
    edgedata = [line.strip('\n').split('\t')[0] for line in lines]
    links = [tuple([int(x) for x in edge.split(',')]) for edge in edgedata]
    return links

def GetVASubstations(path,sub_file='Electric_Substations.shp',
                    state_file='states.shp'):
    """
    Gets the list of substations within the county polygon
    
    Parameters
    ----------
    fis : TYPE
        DESCRIPTION.
    sub_file : TYPE, optional
        DESCRIPTION. The default is 'Electric_Substations.shp'.
    state_file : TYPE, optional
        DESCRIPTION. The default is 'states.shp'.
    
    Returns
    -------
    None.
    
    """
    subs = nt("substation",field_names=["cord"])
    data_substations = gpd.read_file(path+'eia/'+sub_file)
    data_states = gpd.read_file(path+'census/'+state_file)
    
    state_polygon = list(data_states[data_states.STATE_ABBR == 
                              'VA'].geometry.items())[0][1]
    df_subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
    cord = dict([(t.ID, (t.LONGITUDE, t.LATITUDE)) \
                  for t in df_subs.itertuples()])
    cord = {int(k):cord[k] for k in cord}
    return subs(cord=cord)


def GetSecnet(path,fis):
    """
    Extracts the generated secondary network information for the area and
    constructs the networkx graph. The attributes are listed below.
    Node attributes: 
        cord: geographical coordinates
        label: node label, H: residence, T: transformer
        resload: average load at residence
    """
    # Extract the secondary network data from all areas
    nodelabel = {}
    nodepos = {}
    edgelist = []
    
    with open(path+str(fis)+'-sec-dist.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip('\n').split('\t')
        n1 = int(data[0]); n2 = int(data[4])
        edgelist.append((n1,n2))
        # nodedata
        nodepos[n1]=[float(data[2]),float(data[3])]
        nodepos[n2]=[float(data[6]),float(data[7])]
        nodelabel[n1] = data[1] 
        nodelabel[n2] = data[5]
    # Construct the graph
    secnet = nx.Graph()
    secnet.add_edges_from(edgelist)
    nx.set_node_attributes(secnet,nodepos,'cord')
    nx.set_node_attributes(secnet,nodelabel,'label')
    return secnet

#%% Get road and transformer network (output of Step 1)
def GetMaster(path,code):
    """
    Extracts the road network with local transformer nodes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'T' for transformer, 'R' for road node
        edge attributes of graph:
            None
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-master.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-master.gpickle')
    return graph


def GetPrimRoad(path,code):
    """
    Extracts the road network with local transformer nodes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'T' for transformer, 'R' for road node
        edge attributes of graph:
            None
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-road.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-road.gpickle')
    
    # Add edge attributes for geometry
    for e in graph.edges:
        graph.edges[e]['geometry'] = LineString((graph.nodes[e[0]]['cord'],
                                                 graph.nodes[e[1]]['cord']))
        graph.edges[e]['length'] = Link(graph.edges[e]['geometry']).geod_length
    return graph

#%% Get output synthetic networks (output of Step 2)
def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
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
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-dist-net.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-dist-net.gpickle')
    return graph

#%% Actual Network and Synthetic Network data
def get_geometry(path,area):
    """
    Loads the shape files for edges and nodes in the region and returns the 
    geometry of the network.

    Parameters
    ----------
    path : string
        path of directory where the shape files are stored.
    area : dictionary
        data keyed by area ID and value as the root node.

    Returns
    -------
    df_buses: geopandas dataframe of nodes
    df_lines: geopandas dataframe of edges
    act_graph: networkx graph of actual network
    limits: tuple of limits
    """
    # Get dataframe for buses and lines
    df_lines = gpd.read_file(path+area+'/'+area+'_edges.shp')
    df_buses = gpd.read_file(path+area+'/'+area+'_nodes.shp')
    
    # Get dictionary of node geometries and edgelist
    edgelist = [tuple([int(x) for x in e.split('_')]) for e in df_lines['ID']]
    
    # Create actual graph network
    act_graph = nx.Graph()
    act_graph.add_edges_from(edgelist)
    
    # Compute edge lengths
    geo_length = {}
    for e in edgelist:
        edge = str(e[0])+'_'+str(e[1])
        edge_geom = df_lines.loc[df_lines.ID == edge]['geometry'].values[0]
        geo_length[e] = Link(edge_geom).geod_length
    nx.set_edge_attributes(act_graph,geo_length,'geo_length')

    # Get coordinate limits
    busgeom = df_buses['geometry']
    x_bus = [geom.x for geom in busgeom]
    y_bus = [geom.y for geom in busgeom]
    
    # Get axes limits
    buffer = 0.0
    left = min(x_bus)-buffer
    right = max(x_bus)+buffer
    bottom = min(y_bus)-buffer
    top = max(y_bus)+buffer
    limits = (left,right,bottom,top)
    
    return df_buses,df_lines,act_graph,limits


def get_network(synth,limits):
    """
    Gets the actual network, extracts the limits of the region and loads the
    synthetic network of the region.

    Parameters
    ----------
    synth : networkx graph
        total synthetic primary network
    limits: tuple of float data
        axes limits for the area under consideration

    Returns
    -------
    act_graph: networkx graph of the actual network
    synth_graph: networkx graph of the synthetic network in the axes limits
    limits: axes limits for the area under consideration

    """
    # Get the primary network in the region from the synthetic network
    left,right,bottom,top = limits
    nodelist = [n for n in synth.nodes \
                if left<=synth.nodes[n]['cord'][0]<=right \
                 and bottom<=synth.nodes[n]['cord'][1]<=top \
                     and synth.nodes[n]['label'] != 'H']
    synth_graph = nx.subgraph(synth,nodelist)
    
    # Get the dataframe for node and edge geometries
    d = {'nodes':list(synth_graph.nodes()),
         'geometry':[Point(synth_graph.nodes[n]['cord']) \
                     for n in synth_graph.nodes()]}
    df_cords = gpd.GeoDataFrame(d, crs="EPSG:4326")
    
    d = {'edges':list(synth_graph.edges()),
         'geometry':[synth_graph[e[0]][e[1]]['geometry'] \
                     for e in synth_graph.edges()]}
    df_synth = gpd.GeoDataFrame(d, crs="EPSG:4326")
    return synth_graph,df_cords,df_synth

def get_areadata(path,area,root,synth):
    """
    Extracts the data for an area and returns them in a dictionary which
    is labeled with the required entries.

    Parameters
    ----------
    path : string type
        directory path for the shape files of the actual network
    area : string type
        ID of the area for the shape file
    root : integer type
        root node for the area in the shape file
    synth : networkx graph
        complete synthetic primary distribution network

    Returns
    -------
    data: dictionary of area data
        root: root node ID of actual network
        df_lines: geopandas data frame of actual network edges
        df_buses: geopandas data frame of actual network nodes
        actual: networkx graph of actual network
        synthetic: networkx graph of synthetic network
        limits: axes limits of the region of comparison
        df_synth: geopandas data frame of synthetic network edges
        df_cords: geopandas data frame of synthetic network nodes

    """
    df_buses,df_lines,act_graph,limits = get_geometry(path,area)
    synth_graph,df_cords,df_synth = get_network(synth,limits)
    # Store the data in the dictionary
    data = {'root':root,'limits':limits,'df_lines':df_lines,
            'df_buses':df_buses,'actual':act_graph,
            'synthetic':synth_graph,'df_synth':df_synth,'df_cords':df_cords}
    return data