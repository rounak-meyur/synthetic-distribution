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

acsr = pd.read_csv(pathInp+'acsr.csv')
lines = pd.read_csv(pathInp+'line-data.csv')

#%% Determine ACSR Conductor
#import numpy as np
#acsr_code = {k:acsr[acsr['n']==k]['code'].values for k in range(1,5)}
#line_kv = lines['volt'].values
#thresh = [0.0,50.0,280.0,480.0,1000.0]
#bundles = sum([i*((line_kv>thresh[i-1]) & (line_kv<=thresh[i])) for i in range(1,5)])
#line_code = pd.Series([np.random.choice(acsr_code[k]) for k in bundles])
#lines['code'] = line_code
#lines.to_csv(pathInp+'lines.csv',index=False)

#%% Determine ACSR Conductor current capacity
line_code = lines['code'].values
line_icap = [acsr[acsr['code']==code]['icap'].values[0] for code in line_code]
lines['icap'] = line_icap
lines.to_csv(pathInp+'lines.csv',index=False)
