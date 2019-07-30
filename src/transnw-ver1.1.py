# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:02:21 2019

Author: Rounak Meyur
"""

import sys,os
import pandas as pd

# Get the different directory loaction into different variables
workPath = os.getcwd()
pathLib = workPath + "\\Libraries\\"
pathFig = workPath + "\\figs\\"
pathInp = workPath + "\\input\\"

# User defined libraries
sys.path.append(pathLib)

lines = pd.read_csv(pathInp+'line-data.csv')
voltages = set(lines['kV'].values)