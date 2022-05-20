# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:41:29 2022

author: Rounak

Description: Creates a OpenDSS file for the network with necessary models.
Version 1: 
    Line wire data: conductor data for one type only
    Line geometry data: 4 wire 3 phase at a single road side pole geometry
"""

import os,sys



workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet


with open(workpath+"/LineGeometry.dss",'w') as f:
    f.write('s')