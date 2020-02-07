# Creating Synthetic Power Distribution Networks from Road Network Infrastructure
## Motivation
Physical inter-dependencies between networked civil infrastructures such as transportation and power system network are well known. In order to analyze complex non-linear co-relations between such networks, datasets pertaining to such real infrastructures are required. Such data are not readily available due to their sensitive nature. The aim of this project is to generate realistic synthetic distribution network for a given geographical region. The generated network is not the actual distribution system but is very similar to the real distribution network. The synthetic network connects high voltage substations to individual residential consumers through primary and secondary distribution networks. The distribution network is generated by solving an optimization problem which minimizes the overall length of network and is subject to the usual structural and power flow constraints. The network generation algorithm is applied to create synthetic distribution networks in Montgomery county of south-west Virginia, USA.

## Datasets used
**Roads** The road network represented in the form of a graph <img src="https://render.githubusercontent.com/render/math?math=\mathcal{R}=(\mathcal{V}_\mathcal{R},\mathcal{L}_\mathcal{R})">, where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{V}_\mathcal{R}"> and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_\mathcal{R}"> are respectively the sets of nodes and links of the network. Each road link <img src="https://render.githubusercontent.com/render/math?math=l\in\mathcal{L}_\mathcal{R}"> is represented as an unordered pair of terminal nodes <img src="https://render.githubusercontent.com/render/math?math=(u,v)"> with <img src="https://render.githubusercontent.com/render/math?math=u,v\in\mathcal{V}_\mathcal{R}">. Each road node has a spatial embedding in form of longitude and latitude. Therefore each node <img src="https://render.githubusercontent.com/render/math?math=v\in\mathcal{V}_\mathcal{R}"> can be represented in two dimensional space as <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p_v}\in\mathbb{R}^2">. Similarly, a road link <img src="https://render.githubusercontent.com/render/math?math=l=(u,v)"> can be represented as a vector <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p_u}-\mathbf{p_v}">.
	
**Substations** The set of substations <img src="https://render.githubusercontent.com/render/math?math=\mathsf{S}=\{s_1,s_2,\cdots,s_M\}">, where the area consists of $M$ substations and their respective geographical location data. Each substation can be represented by a point in the 2-D space as $\mathbf{p_s}\in\mathbb{R}^2$.
	
**Residences** The set of residential buildings with geographical coordinates $\mathsf{H}=\{h_1,h_2,\cdots,h_N\}$, where the area consists of $N$ home locations. Each residential building can be represented by a point in the 2-D space as $\mathbf{p_h}\in\mathbb{R}^2$.

## Map points in region to network links
The first task is to define a function to compute the geodesic distance of two given points on the earth's surface. We use the methods in *Geodesic* of the *geographiclib* module. Note the input format of the geographic coordinates. The first coordinate is the longitude and followed by latitude.
```python
# Import geodesic from geographiclib module
from geographiclib.geodesic import Geodesic
def MeasureDistance(pt1,pt2):
    '''
    This function measures the geodesic distance of two points.
    The format of each point is (longitude,latitude).
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']
```
Now we define a python class to link each point of interest to the nearest link. This is accomplished using a QD-Tree approach. The primary aim is to reduce the computation time of the entire process. First, we try to form a bounding box around each network link in 
```python
# Import required modules
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point,LineString
from pyqtree import Index
# Class definition to link each point to nearest nearest link
class MapLink:
    """
    This class consists of attributes and methods to evaluate the nearest road
    network link to a point. The point may be a home location or a substation.
    The algorithm uses a QD-Tree approach to evaluate a bounding box for each
    point and link then finds the nearest link to each point.
    """
    def __init__(self,links,radius=0.01):
        '''
	Form bounding boxes around each network link and store indices.
        '''
	# Get the end points of each network link
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
```
