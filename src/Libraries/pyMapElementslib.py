# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:41:09 2019

Author: Rounak Meyur
"""

import os
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from shapely.geometry import Point,LineString
from pyqtree import Index
from scipy.spatial import Voronoi,cKDTree,voronoi_plot_2d
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from collections import namedtuple
import networkx as nx
from pyImgHelperlib import ImageHelper

#%% Functions
def MeasureDistance(Point1,Point2):
    '''
    The format of each point is (longitude,latitude).
    '''
    # Approximate radius of earth in km
    R = 6373.0
    
    # Get the longitude and latitudes of the two points
    lat1 = radians(Point1[1])
    lon1 = radians(Point1[0])
    lat2 = radians(Point2[1])
    lon2 = radians(Point2[0])
    
    # Measure the long-lat difference
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Calculate distance between points in km
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance*1000




#%% Classes
class MapLink:
    """
    This class consists of attributes and methods to evaluate the nearest road
    network link to a point. The point may be a home location or a substation.
    The algorithm uses a QD-Tree approach to evaluate a bounding box for each
    point and link then finds the nearest link to each point.
    """
    def __init__(self,links,radius=0.01):
        '''
        '''
        xmin,ymin = np.min(np.array(list(links.cord.values())),axis=0)
        xmax,ymax = np.max(np.array(list(links.cord.values())),axis=0)
        bbox = (xmin,ymin,xmax,ymax)
    
        # keep track of lines so we can recover them later
        all_link = list(links.graph.edges())
        self.lines = []
    
        # initialize the quadtree index
        self.idx = Index(bbox)
        
        # add edge bounding boxes to the index
        for i, path in enumerate(all_link):
            # create line geometry
            line = LineString([links.cord[path[0]],links.cord[path[1]]])
        
            # bounding boxes, with padding
            x1, y1, x2, y2 = line.bounds
            bounds = x1-radius, y1-radius, x2+radius, y2+radius
        
            # add to quadtree
            self.idx.insert(i, bounds)
        
            # save the line for later use
            self.lines.append((line, bounds, path))
        return
    
    
    def map_point(self,points,path=os.getcwd(),name=None,
                  radius=0.01,projection=False):
        '''
        '''
        Map2Link = {}
        if projection: Proj = {}
        for h in points.cord:
            pt = Point(points.cord[h])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = self.idx.intersect(pt_bounds)
            
            # find closest path
            try:
                closest_path = min(matches, 
                                   key=lambda i: self.lines[i][0].distance(pt))
                Map2Link[h] = self.lines[closest_path][-1]
                if projection:
                    p = self.lines[closest_path][0].project(pt,normalized=True)
                    pt_proj = self.lines[closest_path][0].interpolate(p,
                        normalized=True)
                    Proj[h] = (pt_proj.x,pt_proj.y)
            except:
                Map2Link[h] = None
                if projection: Proj[h] = None
        
        # Delete unmapped points
        unmapped = [p for p in Map2Link if Map2Link[p]==None]
        for p in unmapped:
            del Map2Link[p]
            if projection: del Proj[h]
        
        # Save as a csv file
        df_map = pd.DataFrame.from_dict(Map2Link,orient='index',
                                        columns=['source','target'])
        df_map.index.names = [name[0].upper()+'ID']
        if projection:
            df_proj = pd.DataFrame.from_dict(Proj,orient='index',
                                         columns=['longitude','latitude'])
            df_proj.index.names = [name[0].upper()+'ID']
            df_map = pd.merge(df_map,df_proj,on=name[0].upper()+'ID')
        df_map.to_csv(path+name+'2link.csv')
        return
    


class MapSub2Road:
    """
    """
    def __init__(self,subs,roads,road_to_home):
        """
        """
        self.roads = roads
        self.subs = subs
        self.RW = [r for r in road_to_home if road_to_home[r]!=[]]
        self.RW_pts = [(roads.cord[r][0],roads.cord[r][1]) for r in self.RW]
        self.sub_pts = list(subs.cord.values())
        
        # Find number of road nodes mapped to each substation
        voronoi_kdtree = cKDTree(self.sub_pts)
        _, RW_regions = voronoi_kdtree.query(self.RW_pts, k=1)
        sub_map = {s:RW_regions.tolist().count(s) for s in range(len(subs.cord))}
        
        # Get substations with minimum number of road nodes
        sub_region = [list(subs.cord.keys())[s] for s in sub_map if sub_map[s]>30]
        self.sub_pts = [subs.cord[s] for s in sub_region]
        
        # Recompute the Voronoi regions and generate the final map
        voronoi_kdtree = cKDTree(self.sub_pts)
        _, RW_regions = voronoi_kdtree.query(self.RW_pts, k=1)
        indS2R = [np.argwhere(i==RW_regions)[:,0] for i in np.unique(RW_regions)]
        self.S2Road = {sub_region[i]:[self.RW[j] for j in indS2R[i]] for i in range(len(indS2R))}
        
        S2RNear = {}
        for sub in self.S2Road:
            road_nodes = self.S2Road[sub]
            dist = [MeasureDistance(subs.cord[sub],roads.cord[r]) for r in road_nodes]
            S2RNear[sub] = road_nodes[dist.index(min(dist))]
        return
    
    
    def plot_voronoi(self,S,V):
        """
        """
        # Plot Voronoi regions with mapped road network nodes
        vor = Voronoi([self.subs.cord[s] for s in S])
        RW_xval = [self.roads.cord[r][0] for r in V]
        RW_yval = [self.roads.cord[r][1] for r in V]
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.scatter(RW_xval,RW_yval,c='r',s=1)
        voronoi_plot_2d(vor,ax=ax,show_vertices=False)
        plt.show()

        


class Cluster:
    """
    """
    def __init__(self,homes,roads):
        """
        Perform weighted k-means clustering of the homes whre the weights are the 
        average hourly load demand at each home.
        Inputs: homes: named tuple of type 'Homes'
        """
        self.X = np.array(list(homes.cord.values()))
        self.w = np.array(list(homes.average.values()))
        self.cluster = self.__clusters()
        self.roads = roads
        self.home_id = list(homes.cord.keys())
        return
    
    
    def __clusters(self,Pmax=100e3,kmax=600):
        """
        """
        #total_demand = np.sum(self.w)
        #k_min = int(max(total_demand/Pmax,1))-1
        kmin = 600
        k_clusters = int((kmin+kmax)/2)
        # Perform k-means clustering with corresponding weights
        cluster = k_means(self.X,n_clusters=k_clusters,
                          sample_weight=self.w,return_n_iter=True)
        return cluster
    
    
    def get_tsfr(self,path=os.getcwd()):
        """
        """
        centroid = {k:self.cluster[0][k] for k in range(len(self.cluster[0]))}
        tsfr = namedtuple("Transformers",field_names=["cord"])
        transformers = tsfr(cord=centroid)
        map_object = MapLink(self.roads)
        map_object.map_point(transformers,path=path,name='tsfr',
                             projection=True)
        
        tsfr_id = list(centroid.keys())
        c_labels = self.cluster[1]
        home_to_tsfr = {self.home_id[i]:tsfr_id[c_labels[i]] \
                        for i in range(len(c_labels))}
        df_map = pd.DataFrame.from_dict(home_to_tsfr,orient='index',
                                        columns=['TID'])
        df_map.index.names = ['HID']
        df_map.to_csv(path+'home2tsfr.csv')
        return
    
    
    def plot_clusters(self,path=os.getcwd()):
        """
        """
        fig = plt.figure(figsize=(13,10))
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,0],self.X[:,1],c=self.cluster[1],s=1.0,
                   cmap='plasma')
        nx.draw_networkx(self.roads.graph, pos=self.roads.cord, node_size=1.0, 
                         with_labels=False,ax=ax,edge_color='k',width=0.5)
        plt.close()
        ImageHelper().save_image(fig,"clusters",path)
        return fig


















