# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:20:45 2019

Author: Dr Anil Vullikanti
        Rounak Meyur
        
Description: Generates primary distribution network for Montgomery county
"""


import sys,os
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import networkx as nx
import pandas as pd
import geopandas as gpd
import utm
import matplotlib.pyplot as plt
from collections import namedtuple as nt
from shapely.geometry import Point,LineString
from pyqtree import Index

workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)



#%% Functions

def MeasureDistance(Point1,Point2):
    '''
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
    return distance


def RoadData(csvpath,level=[3,4,5]):
    '''
    Method to create a named tuple to store the road network data.
    '''
    df_all = pd.read_csv(csvpath+'road-graph.csv')
    df_road = df_all.loc[df_all['level'].isin(level)]
    df_cord = pd.read_csv(csvpath+'road-cord.csv')
    graph = nx.from_pandas_edgelist(df_road,'source','target','level')
    cords = df_cord.set_index('node').T.to_dict('list')
    utm_cord = {k:[utm.from_latlon(cords[k][1],cords[k][0])[0],
                   utm.from_latlon(cords[k][1],cords[k][0])[1]] for k in cords}
    road = nt("road",field_names=["graph","cord","UTM"])
    return road(graph=graph,cord=cords,UTM=utm_cord)


def SubData(csvpath):
    '''
    Method to create a named tuple to store the substation location data which 
    is obtained from EIA database.
    '''
    df_cord = pd.read_csv(csvpath+'subs-cord.csv')
    cords = df_cord.set_index('node').T.to_dict('list')
    utm_cord = {k:[utm.from_latlon(cords[k][1],cords[k][0])[0],
                   utm.from_latlon(cords[k][1],cords[k][0])[1]] for k in cords}
    subs = nt("substation",field_names=["cord","UTM"])
    return subs(cord=cords,UTM=utm_cord)


def GetHourlyDemand(inppath,csvpath,filename,cordfile):
    '''
    '''
    df_load = pd.read_csv(inppath+filename)
    df_cord = pd.read_csv(inppath+cordfile,
                            usecols=['HID','Longitude','Latitude'])
    df_cord = df_cord[['HID','Longitude','Latitude']]
    df_cord.to_csv(csvpath+'home-cord.csv',index=False)
    df_demand = pd.read_csv(inppath+cordfile,usecols=['HID'])
    #df_base = (df_load['hotWaterWH'] + df_load['standbyWH'])/24.0
    df_base = (df_load['standbyWH'])/24.0
    for i in range(1,25):
        df_demand['hour'+str(i)] = df_load['P'+str(i)]+df_load['A'+str(i)]+df_base
    df_demand.to_csv(csvpath+'home-load.csv',index=False)
    return


def HomeGeoDF(csvpath):
    '''
    '''
    df_home = pd.merge(pd.read_csv(csvpath+'home-cord.csv'),
                       pd.read_csv(csvpath+'home-load.csv'),on='HID')
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","UTM"])
    dict_cord = pd.read_csv(csvpath+'home-cord.csv').set_index('HID').T.to_dict('list')
    dict_load = pd.read_csv(csvpath+'home-load.csv').set_index('HID').T.to_dict('list')
    dict_peak = dict(zip(df_home.HID,df_home.peak))
    utm_cord = {k:[utm.from_latlon(dict_cord[k][1],dict_cord[k][0])[0],
                   utm.from_latlon(dict_cord[k][1],dict_cord[k][0])[1]] for k in dict_cord}
    homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,UTM=utm_cord)
    
    gdf_home = gpd.GeoDataFrame(df_home.loc[:,['HID','Longitude','Latitude','peak']],
                                geometry=gpd.points_from_xy(df_home.Longitude,
                                                            df_home.Latitude))
    return gdf_home,homes
    
    
def MapHome2Link(roads,homes,radius=0.01):
    '''
    '''
    xmin,ymin = np.min(np.array(list(roads.cord.values())),axis=0)
    xmax,ymax = np.max(np.array(list(roads.cord.values())),axis=0)
    bbox = (xmin,ymin,xmax,ymax)
    
    # keep track of lines so we can recover them later
    links = list(roads.graph.edges())
    lines = []
    
    # initialize the quadtree index
    idx = Index(bbox)
    
    # add edge bounding boxes to the index
    for i, path in enumerate(links):
        # create line geometry
        line = LineString([roads.cord[path[0]],roads.cord[path[1]]])
    
        # bounding boxes, with padding
        x1, y1, x2, y2 = line.bounds
        bounds = x1-radius, y1-radius, x2+radius, y2+radius
    
        # add to quadtree
        idx.insert(i, bounds)
    
        # save the line for later use
        lines.append((line, bounds, path))
    
    
    Home2Link = {}
    for h in homes.cord:
        pt = Point(homes.cord[h])
        pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
        matches = idx.intersect(pt_bounds)
        
        # find closest path
        try:
            closest_path = min(matches, key=lambda i: lines[i][0].distance(pt))
            Home2Link[h] = lines[closest_path][-1]
        except:
            Home2Link[h] = None
    return Home2Link


def SeparateHomeLink(Home2Link,HomeCoord,RWCoord):
    '''
    '''
    Link2Home = {}
    # Generate the opposite mapping
    for h in Home2Link:
        if Home2Link[h] in Link2Home.keys():
            Link2Home[Home2Link[h]].append(h)
        else:
            Link2Home[Home2Link[h]]=[h]
    # for each link, separate the homes
    HomeSideA = {l:[] for l in Link2Home}
    HomeSideB = {l:[] for l in Link2Home}
    for l in Link2Home:
        d = [(HomeCoord[h][0]-RWCoord[l[0]][0])*(RWCoord[l[1]][1]-RWCoord[l[0]][1]) \
             -(HomeCoord[h][1]-RWCoord[l[0]][1])*(RWCoord[l[1]][0]-RWCoord[l[0]][0]) \
             for h in Link2Home[l]]
        for i in range(len(Link2Home[l])):
            if d[i]>=0: HomeSideA[l].append(Link2Home[l][i])
            else: HomeSideB[l].append(Link2Home[l][i])
    return HomeSideA,HomeSideB


def MapHomeRoad(HomeCoord,RWCoord,HomeSide,RW2Home):
    '''
    '''
    for l in HomeSide:
        dist1 = [MeasureDistance(HomeCoord[h],RWCoord[l[0]]) for h in HomeSide[l]]
        dist2 = [MeasureDistance(HomeCoord[h],RWCoord[l[1]]) for h in HomeSide[l]]
        home_list_p1 = []; home_dist_p1 = []; home_list_p2 = []; home_dist_p2 = []
        for i in range(len(HomeSide[l])):
            if dist1[i]<=dist2[i]:
                home_list_p1.append(HomeSide[l][i])
                home_dist_p1.append(dist1[i])
            else:
                home_list_p2.append(HomeSide[l][i])
                home_dist_p2.append(dist2[i]) 
        RW2Home[l[0]].append([x for _,x in sorted(zip(home_dist_p1,home_list_p1))])
        RW2Home[l[1]].append([x for _,x in sorted(zip(home_dist_p2,home_list_p2))])
    
    return RW2Home


#%% Main function goes here
subs = SubData(csvPath)
gdf_home,homes = HomeGeoDF(csvPath)
roads = RoadData(csvPath,level=[4,5])





#H2Link = MapHome2Link(roads,homes,radius=0.01)
df_map = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_map.itertuples()])
del H2Link[511210214001525]

HA,HB = SeparateHomeLink(H2Link,homes.cord,roads.cord)

RW2Home = {k:[] for k in roads.cord}
RW2Home = MapHomeRoad(homes.cord,roads.cord,HA,RW2Home)
RW2Home = MapHomeRoad(homes.cord,roads.cord,HB,RW2Home)


#home_x = [homes.cord[k][0] for k in homes.cord]
#home_y = [homes.cord[k][1] for k in homes.cord]
plt.figure(figsize=(10,10))
nx.draw_networkx(roads.graph,pos=roads.cord,with_labels=False,node_size=1)
plt.scatter(homes.cord[511210214001525][0],homes.cord[511210214001525][1],
            marker='.',s=100,c='r')
plt.show()

















