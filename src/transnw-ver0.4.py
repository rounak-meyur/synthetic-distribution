# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:22:57 2019

@author: meyu507
"""

import sys,os
import pandas as pd
import networkx as nx
import numpy as np



# Get the different directory loaction into different variables
workPath = os.getcwd()
pathLib = workPath + "\\Libraries\\"
pathFig = workPath + "\\figs\\"
pathInp = workPath + "\\input\\"

# User defined libraries
sys.path.append(pathLib)
from pyBuildNetworklib import MeasureDistance as dist



objs = pd.read_csv(pathInp+'Data-Rachel/Object_Longitude_Latitude.csv')
plants = pd.read_csv(pathInp+'Data-Rachel/Plant_MWProduce_PrimarySource.csv')
edges = pd.read_csv(pathInp+'Data-Rachel/Edge_List.csv')
evolt = pd.read_csv(pathInp+'Data-Rachel/Line_Length_Voltage.csv',na_filter=False)

nbus = len(objs)
nodelist = objs['Object_ID'].values
nodelong = objs['Longitude'].values
nodelat = objs['Latitude'].values
Node = {nodelist[k]:[nodelong[k],nodelat[k]] for k in range(nbus)}


edge_length = {evolt['Line_ID'].values[k]:evolt['Line_Length_Meters'].values[k]\
               for k in range(len(evolt))}
edge_volt = {evolt['Line_ID'].values[k]:evolt['Voltage'].values[k] \
             for k in range(len(evolt))}


line_volt = [float(edge_volt[i]) if edge_volt[i]!='-999999' \
             and edge_volt[i] != 'NA' else 0.0\
             for i in edges['Full_Line_ID'].values]



nlines = len((edges))
full = edges['Full_Line_ID'].values
fbus = edges['Start_Object_ID'].values
tbus = edges['End_Object_ID'].values
lineid = edges['Line_ID'].values
length = [dist(Node[fbus[k]],Node[tbus[k]]) for k in range(nlines)]

Volt = {n:0.0 for n in nodelist}
count = 0
for k in range(nlines):
    if length[k]>100.0:
        if Volt[fbus[k]] in [0.0,line_volt[k]] and Volt[tbus[k]] in [0.0,line_volt[k]]:
            Volt[fbus[k]]=line_volt[k]
            Volt[tbus[k]]=line_volt[k]
        else:
            count += 1
fv = [Volt[fbus[k]] for k in range(nlines)]
tv = [Volt[tbus[k]] for k in range(nlines)]
line_volt = [max(fv[k],tv[k],line_volt[k]) for k in range(nlines)]

Full2Ind = {k:[] for k in list(set(full))}
for k in range(nlines):
    Full2Ind[full[k]].append(lineid[k])

Ind2Full = {lineid[k]:full[k] for k in range(nlines)}
Ind2Volt = {lineid[k]:line_volt[k] for k in range(nlines)}

for k in range(nlines):
    if line_volt[k]==0.0:
        line_volt[k] = max([Ind2Volt[f] for f in Full2Ind[Ind2Full[lineid[k]]]])

line_volt = np.array(line_volt)
acsr = pd.read_csv(pathInp+'acsr.csv')
acsr_code = {k:acsr[acsr['n']==k]['code'].values for k in range(1,5)}
acsr_code[0] = np.array(['transformer'])
thresh = [0.0,50.0,280.0,480.0,1000.0]
bundles = sum([i*((line_volt>thresh[i-1]) & (line_volt<=thresh[i]))\
               for i in range(1,5)])
line_code = pd.Series([np.random.choice(acsr_code[k]) for k in bundles])
line_icap = [acsr[acsr['code']==code]['icap'].values[0] \
             if code != 'transformer' else 0.0 for code in line_code]



lines = pd.DataFrame()
lines['fbus']=fbus
lines['tbus']=tbus
lines['id']=lineid
lines['volt']=line_volt
lines['meters']=length
lines['code']=line_code
lines['icap']=line_icap
lines['mva'] = [np.sqrt(3)*line_volt[k]*line_icap[k]*0.001 if line_volt[k]!=0.0\
     else float('inf') for k in range(nlines)]
lines.to_csv(pathInp+'lines.csv',index=False)

























