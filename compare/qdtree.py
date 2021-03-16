# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:59:01 2021

Author: Rounak
"""

from geographiclib.geodesic import Geodesic
from pyqtree import Index
import numpy as np
from shapely.geometry import Point

#%% Functions
def geodist(pt1,pt2):
    '''
    Computes the geodesic distance between two shapely points
    '''
    lon1 = pt1.x; lat1 = pt1.y
    lon2 = pt2.x; lat2 = pt2.y
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

def nearest(main_points,aux_points,radius=1):
    """
    Objective is to find the nearest point to the points in the aux_points set.
    The nearest point is to be identified among the points in the main_points
    set.
    """
    xmin,ymin = np.min(np.array(list(main_points.values())),axis=0)
    xmax,ymax = np.max(np.array(list(main_points.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    idx = Index(bbox)
    
    # keep track of points so we can recover them later
    points = []
    
    # create bounding box around each point in main_points set
    for i,p in enumerate(main_points):
        pt_main = Point(main_points[p])
        pt_bounds_main = pt_main.x-radius, pt_main.y-radius, \
            pt_main.x+radius, pt_main.y+radius
        idx.insert(i, pt_bounds_main)
        points.append((pt_main, pt_bounds_main, p))
        
    # find intersection with bounding box around aux point set
    for n in aux_points:
        pt_aux = Point(aux_points[n])
        pt_bounds_aux = pt_aux.x-radius, pt_aux.y-radius, \
            pt_aux.x+radius, pt_aux.y+radius
        matches = idx.intersect(pt_bounds_aux)
        ind_closest = min(matches,key=lambda i: geodist(points[i][0],pt_aux))
        print(points[ind_closest][-1])
    return


# Example code
main_pts = {'A':[0,1],'B':[0,2],'C':[0,3]}
aux_pts = {'a':[1,1],'b':[-1,1],'c':[-1,2]}
nearest(main_pts,aux_pts)