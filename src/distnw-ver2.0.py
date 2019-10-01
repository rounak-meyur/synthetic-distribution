# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak
"""

import sys,os
import osmnx as ox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)


place_name = "Blacksburg, Virginia, USA"

graph = ox.graph_from_place(place_name)
fig = plt.figure(figsize=(10,10))
fig, ax = ox.plot_graph(graph)

#area = ox.gdf_from_place(place_name)
#buildings = ox.buildings_from_place(place_name)