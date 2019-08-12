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
import matplotlib.pyplot as plt
from collections import namedtuple as nt

workPath = os.getcwd()
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


def RoadData(csvpath):
    '''
    Method to create a named tuple to store the road network data.
    '''
    df_road = pd.read_csv(csvpath+'road-graph.csv')
    df_cord = pd.read_csv(csvpath+'road-cord.csv')
    graph = nx.from_pandas_edgelist(df_road,'source','target','level')
    cords = df_cord.set_index('node').T.to_dict('list')
    road = nt("road",field_names=["graph","cords"])
    return road(graph=graph,cords=cords)
    

def HomeData(csvpath):
    '''
    Method to create a named tuple to store the home/activity location data
    which is obtained from noldor database.
    '''
    df_cord = pd.read_csv(csvpath+'home-cord.csv')
    cords = df_cord.set_index('node').T.to_dict('list')
    home = nt("home",field_names=["cords"])
    return home(cords=cords)


def SubData(csvpath):
    '''
    Method to create a named tuple to store the substation location data which 
    is obtained from EIA database.
    '''
    df_cord = pd.read_csv(csvpath+'subs-cord.csv')
    cords = df_cord.set_index('node').T.to_dict('list')
    subs = nt("substation",field_names=["cords"])
    return subs(cords=cords)




#%% Main function goes here
subs = SubData(csvPath)
home = HomeData(csvPath)
home_x = [home.cords[k][0] for k in home.cords]
home_y = [home.cords[k][1] for k in home.cords]
subs_x = [subs.cords[k][0] for k in subs.cords]
subs_y = [subs.cords[k][1] for k in subs.cords]


plt.figure(figsize=(10,10))
plt.scatter(home_x,home_y,marker='.',s=10,c='r')
plt.scatter(subs_x,subs_y,marker='^',s=20,c='g')
plt.show()
















