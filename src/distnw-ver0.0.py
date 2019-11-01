# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:20:45 2019

Author: Dr Anil Vullikanti
        Rounak Meyur
        
Description: Generates secondary distribution network for Montgomery county.
Particularly identifies the location of transformers using clustering method.
"""


import sys,os
import time
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyExtractDatalib import Query


#%% Main function goes here
    
start = time.time()
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()

#%% Create the mapping between homes and transformers/links
from pyMapElementslib import Cluster
cluster_obj = Cluster(homes,roads,k=1500)
cluster_obj.get_tsfr(path=csvPath)
print("Clustering done in",cluster_obj.cluster[3],"iterations.")
f = cluster_obj.plot_clusters(path=figPath)


#%%
end = time.time()
print ("Time taken:",end-start)
sys.exit(0)