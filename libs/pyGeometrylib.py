# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:34:56 2021

@author: rouna
"""

from shapely.geometry import Point,LineString,Polygon
from geographiclib.geodesic import Geodesic
import numpy as np
from pyqtree import Index

#%% Class Link to define network geometry
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
    
    def InterpolatePoints(self,sep=20):
        """
        """
        points = []
        length = self.geod_length
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(Point(xy))
        if len(points)==0: 
            points.append(Point((self.xy[0][0],self.xy[1][0])))
        # return {i:[pt.x,pt.y] for i,pt in enumerate(MultiPoint(points))}
        return points


#%% Classes for grid geometry
class Grid(Polygon):
    """A rectangular grid with limits."""

    def __init__(self, poly_geom):
        super().__init__(poly_geom)
        self.west_edge, self.east_edge = poly_geom[0][0],poly_geom[1][0]
        self.north_edge, self.south_edge = poly_geom[0][1], poly_geom[2][1]
        return

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)
    
    def __hash__(self):
        return hash((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

#%% Functions
def partitions(limits,kx,ky,x0=0,y0=0):
    """
    kx,ky: number of demarcations along x and y axes.
    x0,y0: percentage x and y shifts for the grid demarcations.
    """
    LEFT,RIGHT,BOTTOM,TOP = limits
    xmin = []; xmax = []; ymin = []; ymax = []
    width = RIGHT-LEFT
    height = TOP-BOTTOM
    x_shift = x0*width
    y_shift = y0*height
    for t in range(kx):
        xmin.append(LEFT+(t/kx)*(RIGHT-LEFT)+x_shift)
        xmax.append(LEFT+((1+t)/kx)*(RIGHT-LEFT)+x_shift)
    for t in range(ky):
        ymin.append(BOTTOM+(t/ky)*(TOP-BOTTOM)+y_shift)
        ymax.append(BOTTOM+((1+t)/ky)*(TOP-BOTTOM)+y_shift)
    # For shifted origins
    if x0>0:
        xmax = [xmin[0]]+xmax
        xmin = [LEFT]+xmin
    if x0<0:
        xmin = xmin+[xmax[-1]]
        xmax = xmax+[RIGHT]
    if y0>0:
        ymax = [ymin[0]]+ymax
        ymin = [BOTTOM]+ymin
    if y0<0:
        ymin = ymin+[ymax[-1]]
        ymax = ymax+[TOP]
    # get the grid polygons
    grid = []
    for i in range(len(xmin)):
        for j in range(len(ymin)):
            vertices = [(xmin[i],ymin[j]),(xmax[i],ymin[j]),
                        (xmax[i],ymax[j]),(xmin[i],ymax[j])]
            grid.append(Grid(vertices))
    return grid


# def hausdorff(main_points,aux_points,radius=0.001):
#     """
#     Objective is to find the nearest point to the points in the aux_points set.
#     The nearest point is to be identified among the points in the main_points
#     set.
#     """
#     xpts = [p.x for p in main_points] + [p.x for p in aux_points]
#     ypts = [p.y for p in main_points] + [p.y for p in aux_points]
#     bbox = (min(xpts),min(ypts),max(xpts),max(ypts))
#     idx = Index(bbox)
    
#     # keep track of points so we can recover them later
#     points = []
    
#     # create bounding box around each point in main_points set
#     for i,pt_main in enumerate(main_points):
#         pt_bounds_main = pt_main.x-radius, pt_main.y-radius, \
#             pt_main.x+radius, pt_main.y+radius
#         idx.insert(i, pt_bounds_main)
#         points.append((pt_main, pt_bounds_main))
        
#     # find intersection with bounding box around aux point set
#     dist = []
#     for pt_aux in aux_points:
#         pt_bounds_aux = pt_aux.x-radius, pt_aux.y-radius, \
#             pt_aux.x+radius, pt_aux.y+radius
#         matches = idx.intersect(pt_bounds_aux)
#         ind_closest = min(matches,key=lambda i: geodist(points[i][0],pt_aux))
#         dist.append(geodist(points[ind_closest][0],pt_aux))
#     return max(dist)


def geodist(geomA,geomB):
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']

























