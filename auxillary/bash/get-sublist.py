# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:50:33 2020

@author: rounak
"""

# Get list of substations
import os
import geopandas as gpd
from pyqtree import Index


# Load scratchpath
scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inppath = scratchpath + "/input/"
tmppath = scratchpath + "/temp/"


filelist = [f for f in os.listdir(tmppath+'osm-prim-master') \
            if f.endswith('-master.gpickle')]
sublist = [f.split('-')[0] for f in filelist]



sub_file="eia/Electric_Substations.shp"
state_file="census/states.shp"
block_file = "census/tl_2018_51_tabblock10.shp"

   

data_substations = gpd.read_file(inppath+sub_file)
data_states = gpd.read_file(inppath+state_file)
data_blocks = gpd.read_file(inppath+block_file)

state_polygon = list(data_states[data_states.STATE_FIPS == '51'].geometry.items())[0][1]
subs = data_substations.loc[data_substations.geometry.within(state_polygon)]


#%% Rural and Urban differentiation
rural_blocks = data_blocks.loc[data_blocks.UR10=='R']["geometry"].values
urban_blocks = data_blocks.loc[data_blocks.UR10=='U']["geometry"].values



#%% Using QDTree
subx = [subs.iloc[i]["geometry"].coords[0][0] for i in range(len(subs))]
suby = [subs.iloc[i]["geometry"].coords[0][1] for i in range(len(subs))]
xmax = max(subx)
xmin = min(subx)
ymax = max(suby)
ymin = min(suby)
bbox = (xmin,ymin,xmax,ymax)

# Urban substations
idx = Index(bbox)
for pos, poly in enumerate(urban_blocks):
    idx.insert(pos, poly.bounds)

#iterate through points
urban_subs = []
for i in range(len(subs)):
    point = subs.iloc[i]["geometry"]
    # iterate through spatial index
    for j in idx.intersect(point.coords[0]):
        if point.within(urban_blocks[j]):
            urban_subs.append(subs.iloc[i]["ID"])

# Rural substations
idx = Index(bbox)
for pos, poly in enumerate(rural_blocks):
    idx.insert(pos, poly.bounds)

#iterate through points
rural_subs = []
for i in range(len(subs)):
    point = subs.iloc[i]["geometry"]
    # iterate through spatial index
    for j in idx.intersect(point.coords[0]):
        if point.within(rural_blocks[j]):
            rural_subs.append(subs.iloc[i]["ID"])



rural_sublist = sorted([int(s) for s in sublist if s in rural_subs])
urban_sublist = sorted([int(s) for s in sublist if s in urban_subs])
rural_sublist = [str(x) for x in rural_sublist]
urban_sublist = [str(x) for x in urban_sublist]


data = ' '.join(rural_sublist)
with open(inppath+'rural-sublist.txt','w') as f:
    f.write(data)
data = ' '.join(urban_sublist)
with open(inppath+'urban-sublist.txt','w') as f:
    f.write(data)
    
    
print("Total number of substations in database",len(subs))
print("Total number of substations in directory",len(sublist))
print("Partitioned substations",len(rural_sublist)+len(urban_sublist))