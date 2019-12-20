# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program computes the mapping between homes and road network by
finding the nearest road link to a residential building.
"""

import sys,os
import pandas as pd



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyMapElementslib import MapLink
from pyBuildNetworklib import InvertMap as imap

#%% Initialization of data sets and mappings
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads(level=[3,4,5])
MapLink(roads).map_point(homes,path=csvPath,name='home')

print("DONE")

#%% Check the output
df_hmap = pd.read_csv(csvPath+'home2link.csv')
H2Link = dict([(t.HID, (t.source, t.target)) for t in df_hmap.itertuples()])

L2Home = imap(H2Link)
links = [l for l in L2Home if 0<len(L2Home[l])<=70]
