# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program creates attempts to formulate the problem for creating
primary distribution network.
"""

import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
from pyExtractDatalib import Query
from pyBuildNetworklib import Spider
from pyBuildNetworklib import MeasureDistance as dist



#%% Get transformers and store them in csv
q_object = Query(csvPath)
gdf_home,homes = q_object.GetHomes()
roads = q_object.GetRoads()
subs = q_object.GetSubstations()













