# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:02:21 2019

Author: Rounak Meyur
"""

import sys,os
import pandas as pd
import networkx as nx



# Get the different directory loaction into different variables
workPath = os.getcwd()
pathLib = workPath + "\\Libraries\\"
pathFig = workPath + "\\figs\\"
pathInp = workPath + "\\input\\"

# User defined libraries
sys.path.append(pathLib)
from pyBuildNetworklib import MeasureDistance as dist

tsfr = pd.read_csv(pathInp+'tx-data.csv')
nodes = pd.read_csv(pathInp+'bus-data.csv')
lines = pd.read_csv(pathInp+'line-data.csv')
Node = {nodes['node'].values[k]:[nodes['long'].values[k],nodes['lat'].values[k]]\
        for k in range(len(nodes))}

Volt = {nodes['node'].values[k]:nodes['volts'].values[k] for k in range(len(nodes))}

fbus = tsfr['fbus'].values
tbus = tsfr['tbus'].values

tsfr_len = [dist(Node[fbus[t]],Node[tbus[t]]) for t in range(len(tsfr))]
tsfr_l = [t for t in range(len(tsfr)) if tsfr_len[t]>100.0]
tsfr_nl = [t for t in range(len(tsfr)) if t not in tsfr_l]


line_len = lines['metres'].values
line_t = [l for l in range(len(lines)) if line_len[l]<100.0]
line_nt = [l for l in range(len(lines)) if l not in line_t]

tsfr_new1 = tsfr.iloc[tsfr_nl,:]
tsfr_new2 = lines.iloc[line_t,:]

line_new1 = lines.iloc[line_nt,:]

#%%
edgelist_file = pathInp + 'Data-Rachel/' + 'Edge_List.csv'
f = open(edgelist_file,'r')
edge_data = [e.strip('\n').split(',') for e in f.readlines()[1:]]
f.close()
G = nx.MultiGraph()
for e in edge_data:
    G.add_edge(int(e[1]),int(e[2]),str(e[0]))
known = [n for n in Volt if Volt[n]!=0.0]


#%%
line_new2 = tsfr.iloc[tsfr_l,:]
fb = line_new2['fbus'].values
tb = line_new2['tbus'].values
lid = line_new2['id'].values
fv = line_new2['kvfbus'].values
tv = line_new2['kvtbus'].values
volt = []
for k in range(len(line_new2)):
    nvolt = max(fv[k],tv[k])
    volt.append(nvolt)
    if fv[k]==nvolt: 
        tv[k]=nvolt
        Volt[tb[k]]=nvolt
    else:
        fv[k]=nvolt
        Volt[fb[k]]=nvolt
    if volt[k]==0.0:
        nvolt = max([Volt[n] for n in list(G.neighbors(fb[k]))+list(G.neighbors(tb[k]))])
        fv[k]=nvolt
        tv[k]=nvolt
        Volt[fb[k]]=nvolt
        Volt[tb[k]]=nvolt
        volt[k] = nvolt
    



#%%
volt = pd.Series([max(fv[k],tv[k]) for k in range(len(line_new2))])
metres = [dist(Node[fb[k]],Node[tb[k]]) for k in range(len(line_new2))]
import numpy as np
acsr = pd.read_csv(pathInp+'acsr.csv')
acsr_code = {k:acsr[acsr['n']==k]['code'].values for k in range(1,5)}
thresh = [0.0,50.0,280.0,480.0,1000.0]
bundles = sum([i*((volt>thresh[i-1]) & (volt<=thresh[i])) for i in range(1,5)])
line_code = pd.Series([np.random.choice(acsr_code[k]) for k in bundles])
line_icap = [acsr[acsr['code']==code]['icap'].values[0] for code in line_code]
line_new2 = pd.DataFrame()
line_new2['fbus']=fb
line_new2['tbus']=tb
line_new2['id'] = lid
line_new2['volt'] = volt
line_new2['metres']=metres
line_new2['code']=line_code
line_new2['icap']=line_icap

























