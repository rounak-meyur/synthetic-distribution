# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:41:09 2019

Author: Rounak Meyur
"""

import os
import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from shapely.geometry import Point,LineString
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
            #if projection: del Proj[h]
        
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
    
