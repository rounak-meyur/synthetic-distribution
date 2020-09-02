# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:46:36 2018

Authors: Rounak Meyur

Description: This library contains classes, methods etc to extract data from
the noldor database with required credentials. 
"""

from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString,Point
from collections import namedtuple as nt
from geographiclib.geodesic import Geodesic
from pyqtree import Index


#%% Functions


def MeasureDistance(pt1,pt2):
    '''
    The format of each point is (longitude,latitude).
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

def combine_components(graph,cords,radius = 0.01):
    """
    

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


#%% Classes

class Query:
    """
    """
    def __init__(self,csvpath,inppath):
        '''
        '''
        self.inppath = inppath
        self.csvpath = csvpath
        return
    
    def GetRoads(self,fis='121',level=[1,2,3,4,5],thresh=0):
        """
        """
        corefile = "nrv/core-link-file-" + fis + ".txt"
        linkgeom = "nrv/link-file-" + fis + ".txt"
        nodegeom = "nrv/node-geometry-" + fis + ".txt"
        
        datalink = {}
        roadcord = {}
        edgelist = []
        
        # Get edgelist from the core link file
        df_core = pd.read_table(self.inppath+corefile,header=0,
                        names=['src','dest','fclass'])
        for i in range(len(df_core)):
            edge = (df_core.loc[i,'src'],df_core.loc[i,'dest'])
            fclass = df_core.loc[i,'fclass']
            if (edge not in edgelist) and ((edge[1],edge[0]) not in edgelist):
                edgelist.append(edge)
                datalink[edge] = {'level':fclass,'geometry':None}
        
        # Get node coordinates from the node geometry file            
        df_node = pd.read_table(self.inppath+nodegeom,header=0,
                                names=['id','long','lat'])
        roadcord = dict([(n.id, (n.long, n.lat)) \
                         for n in df_node.itertuples()])
        
        # Get link geometry from the link geometry file
        colnames = ['ref_in_id','nref_in_id','the_geom']
        coldtype = {'ref_in_id':'Int64','nref_in_id':'Int64','the_geom':'str'}
        df_link = pd.read_table(self.inppath+linkgeom,sep=',',
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
        datalink.update({edge:{'level':5,'geometry':LineString([tuple(roadcord[r]) \
                          for r in list(edge)])} for edge in new_edges})
        road = nt("road",field_names=["graph","cord","links"])
        return road(graph=graph,cord=roadcord,links=datalink)
    
    
    def GetSubstations(self,fis='121',sub_file='Electric_Substations.shp',
                        county_file='county.shp'):
        """
        Gets the list of substations within the county polygon
        
        Parameters
        ----------
        fis : TYPE
            DESCRIPTION.
        sub_file : TYPE, optional
            DESCRIPTION. The default is 'Electric_Substations.shp'.
        county_file : TYPE, optional
            DESCRIPTION. The default is 'county.shp'.
        
        Returns
        -------
        None.
        
        """
        subs = nt("substation",field_names=["cord"])
        data_substations = gpd.read_file(self.inppath+'eia/'+sub_file)
        data_counties = gpd.read_file(self.inppath+'eia/'+county_file)
        
        county_polygon = list(data_counties[data_counties.COUNTYFP == fis].geometry.items())[0][1]
        df_subs = data_substations.loc[data_substations.geometry.within(county_polygon)]
        cord = dict([(t.ID, (t.LONGITUDE, t.LATITUDE)) \
                     for t in df_subs.itertuples()])
        cord = {int(k):cord[k] for k in cord}
        return subs(cord=cord)
    
    
    def GetHomes(self,fis='121'):
        '''
        '''
        df_home = pd.read_csv(self.csvpath+fis+'-home-load.csv')
        df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
        df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
        
        home = nt("home",field_names=["cord","profile","peak","average"])
        dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
        dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
        dict_peak = dict(zip(df_home.hid,df_home.peak))
        dict_avg = dict(zip(df_home.hid,df_home.average))
        homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
        return homes
    
    
    def GetTransformers(self,fis='121'):
        """
        """
        df_tsfr = pd.read_csv(self.csvpath+fis+'-data/'+fis+'-tsfr-data.csv',
                              header=None,names=['tid','long','lat','load'])
        tsfr = nt("Transformers",field_names=["cord","load","graph"])
        dict_cord = dict([(t.tid, (t.long, t.lat)) for t in df_tsfr.itertuples()])
        dict_load = dict([(t.tid, t.load) for t in df_tsfr.itertuples()])
        df_tsfr_edges = pd.read_csv(self.csvpath+fis+'-data/'+fis+'-tsfr-net.csv',
                                    header=None,names=['source','target'])
        g = nx.from_pandas_edgelist(df_tsfr_edges)
        return tsfr(cord=dict_cord,load=dict_load,graph=g)
    
    
    def GetDataset(self,fis='121'):
        """
        """
        homes = self.GetHomes(fis=fis)
        roads = self.GetRoads(fis=fis)
        return homes,roads
    
    def GetAllRoads(self,areas):
        """
        """
        road_graph = nx.Graph()
        road_cord = {}
        road_data = {}
        for area in areas:
            roads = self.GetRoads(fis=area)
            road_graph = nx.compose(road_graph,roads.graph)
            road_cord.update(roads.cord)
            road_data.update(roads.links)
        road = nt("road",field_names=["graph","cord","links"])
        return road(graph=road_graph,cord=road_cord,links=road_data)
    
    def GetAllTransformers(self,areas):
        tsfr_cord = {}
        tsfr_load = {}
        tsfr_graph = nx.Graph()
        for area in areas:
            tsfr = self.GetTransformers(fis=area)
            tsfr_graph = nx.compose(tsfr_graph,tsfr.graph)
            tsfr_cord.update(tsfr.cord)
            tsfr_load.update(tsfr.load)
        tsfr = nt("Transformers",field_names=["cord","load","graph"])
        return tsfr(cord=tsfr_cord,load=tsfr_load,graph=tsfr_graph)
    
    def GetAllMappings(self,areas):
        links = []
        for area in areas:
            with open(self.csvpath+area+'-data/'+area+'-link2home.txt') as f:
                lines = f.readlines()
            edgedata = [line.strip('\n').split('\t')[0] for line in lines]
            links += [tuple([int(x) for x in edge.split(',')]) for edge in edgedata]
        return links
    
    def GetAllSubstations(self,sub_file='Electric_Substations.shp',
                        state_file='states.shp'):
        """
        Gets the list of substations within the county polygon
        
        Parameters
        ----------
        fis : TYPE
            DESCRIPTION.
        sub_file : TYPE, optional
            DESCRIPTION. The default is 'Electric_Substations.shp'.
        county_file : TYPE, optional
            DESCRIPTION. The default is 'county.shp'.
        
        Returns
        -------
        None.
        
        """
        subs = nt("substation",field_names=["cord"])
        data_substations = gpd.read_file(self.inppath+'eia/'+sub_file)
        data_states = gpd.read_file(self.inppath+'eia/'+state_file)
        
        state_polygon = list(data_states[data_states.STATE_ABBR == 
                                 'VA'].geometry.items())[0][1]
        df_subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
        cord = dict([(t.ID, (t.LONGITUDE, t.LATITUDE)) \
                     for t in df_subs.itertuples()])
        cord = {int(k):cord[k] for k in cord}
        return subs(cord=cord)
    
    def GetAllSecondary(self,areas):
        """
        """
        lines = []
        for area in areas:
            with open(self.csvpath+area+'-data/'+area+'-sec-dist.txt') as f:
                lines += f.readlines()
        secnet = nx.Graph()
        edgelist = []
        nodepos = {}
        nodelabel = {}
        for temp in lines:
            e = temp.strip('\n').split('\t')
            edgelist.append((int(e[0]),int(e[4])))
            nodepos[int(e[0])]=[float(e[2]),float(e[3])]
            nodepos[int(e[4])]=[float(e[6]),float(e[7])]
            nodelabel[int(e[0])]=e[1]
            nodelabel[int(e[4])]=e[5]
        secnet.add_edges_from(edgelist)
        nx.set_node_attributes(secnet,nodelabel,'label')
        nx.set_node_attributes(secnet,nodepos,'cord')
        return secnet
        
        
        
    
   
        
        




