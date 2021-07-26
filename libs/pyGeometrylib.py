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
def MeasureDistance(pt1,pt2):
    '''
    Measures the geodesic distance between two coordinates. The format of each point 
    is (longitude,latitude).
    pt1: (longitude,latitude) of point 1
    pt2: (longitude,latitude) of point 2
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']

def geodist(geomA,geomB):
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']
    
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
        xmin.append(LEFT+(t/kx)*(RIGHT-LEFT))
        xmax.append(LEFT+((1+t)/kx)*(RIGHT-LEFT))
    for t in range(ky):
        ymin.append(BOTTOM+(t/ky)*(TOP-BOTTOM))
        ymax.append(BOTTOM+((1+t)/ky)*(TOP-BOTTOM))
    # For shifted origins
    xmin = [xmin[0]] + [x+x_shift for x in xmin[1:]]
    xmax = [x+x_shift for x in xmax[:-1]] + [xmax[-1]]
    ymin = [ymin[0]] + [y+y_shift for y in ymin[1:]]
    ymax = [y+y_shift for y in ymax[:-1]] + [ymax[-1]]
    
    # get the grid polygons
    grid = []
    for i in range(len(xmin)):
        for j in range(len(ymin)):
            vertices = [(xmin[i],ymin[j]),(xmax[i],ymin[j]),
                        (xmax[i],ymax[j]),(xmin[i],ymax[j])]
            grid.append(Grid(vertices))
    return grid

