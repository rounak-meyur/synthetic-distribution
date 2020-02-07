# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:16:26 2020

@author: Rounak Meyur
Description: This program is version 2 which uses Open Street Maps to create road
network data.
"""

import sys,os
import pandas as pd
import osmnx


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"

sys.path.append(libPath)
