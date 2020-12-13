# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:12:30 2020

Author: Rounak Meyur
Description: This program reads the road link geometry and the link files to get the
total structure of the road network.
    a. The total road network structure would be read as a geopandas dataframe.
    b. The geopandas dataframe has a column for the geometry.
    c. Final objective is to place transformers at regular intervals on the geopandas
    data object.
"""

import sys,os
import pandas as pd
from shapely.geometry import LineString,MultiLineString

workPath = os.getcwd()
inpPath = workPath + "/input/nrv/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/nrv/"
tmpPath = workPath + "/temp/nrv/"

sys.path.append(libPath)


with open(workPath+"/input/areacode.txt",'r') as f:
    AREAS = {}
    for d in f.readlines():
        data = d.strip('\n').split('\t')
        AREAS[data[0]] = data[1]


def createlist(path,filelist):
    lines = []
    for filename in filelist:
        with open(path+filename,'r') as f:
            if filename.endswith('1.txt'): lines += f.readlines()
            else: lines += f.readlines()[1:]
    return lines

def createfile(area,path,prefix):
    suffix = AREAS[area]
    filelist = [f for f in os.listdir(path) if f.startswith(prefix+suffix)]
    datalist = createlist(path,filelist)
    if len(filelist)==0:
        print("No such file:",area)
        return
    with open(path+prefix+area+'.txt','w') as f:
        f.write(''.join(datalist))
    return
    
# for area in AREAS:
#     createfile(area,inpPath,'node-geometry-')
#     createfile(area,inpPath,'core-link-file-')
#     createfile(area,inpPath,'link-file-')



sys.exit(0)

#%% Extract data from each file
corefile = "core-link-file-071.txt"
linkgeom = "link-file-071.txt"
nodegeom = "node-geometry-071.txt"



datalink = {}
edgelist = []

df_node = pd.read_table(inpPath+nodegeom,header=0,names=['id','long','lat'])
roadcord = dict([(n.id, (n.long, n.lat)) for n in df_node.itertuples()])

df_core = pd.read_table(inpPath+corefile,header=0,
                        names=['src','dest','fclass'])
for i in range(len(df_core)):
    edge = (df_core.loc[i,'src'],df_core.loc[i,'dest'])
    fclass = df_core.loc[i,'fclass']
    if (edge not in edgelist) and ((edge[1],edge[0]) not in edgelist):
        edgelist.append(edge)
        datalink[edge] = {'level':fclass,'geometry':None}

colnames = ['ref_in_id','nref_in_id','the_geom']
coldtype = {'ref_in_id':'Int64','nref_in_id':'Int64','the_geom':'str'}
df_link = pd.read_table(inpPath+linkgeom,sep=',',
                        usecols=colnames,dtype=coldtype)
for i in range(len(df_link)):
    edge = (df_link.loc[i,'ref_in_id'],df_link.loc[i,'nref_in_id'])
    pts = [tuple([float(x) for x in pt.split(' ')]) \
            for pt in df_link.loc[i,'the_geom'].lstrip('MULTILINESTRING((').rstrip('))').split(',')]
    geom = LineString(pts)
    if (edge in edgelist):
        datalink[edge]['geometry']=geom
    elif ((edge[1],edge[0]) in edgelist):
        datalink[(edge[1],edge[0])]['geometry']=geom
    else:
        print(','.join([str(x) for x in list(edge)])+": not in edgelist")

for edge in datalink:
    if datalink[edge]['geometry']==None:
        print(edge,datalink[edge]['level'])
        pts = [tuple(roadcord[r]) for r in list(edge)]
        geom = LineString(pts)
        datalink[edge]['geometry'] = geom