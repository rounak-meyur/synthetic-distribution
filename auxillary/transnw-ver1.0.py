# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:54:45 2020

@author: rounak
"""

import os,sys
import geopandas as gpd
from shapely.geometry import MultiPolygon
from pyqtree import Index

state = 'VA'

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
path = rootpath+"/input/"
figpath = workpath+'/figs/'
line_file="eia/Electric_Power_Transmission_Lines.shp"
sub_file="eia/Electric_Substations.shp"
state_file="census/states.shp"
block_file = "census/tl_2018_51_tabblock10.shp"

   
data_lines = gpd.read_file(path+line_file)
data_substations = gpd.read_file(path+sub_file)
data_states = gpd.read_file(path+state_file)
data_blocks = gpd.read_file(path+block_file)

state_polygon = list(data_states[data_states.STATE_ABBR == 
                         state].geometry.items())[0][1]

subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
lines1 = data_lines.loc[data_lines.geometry.intersects(state_polygon)]
lines2 = data_lines.loc[data_lines.geometry.within(state_polygon)]

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


with open(path+"sublist.txt") as f:
    sublist = [temp.strip('\n').split(' ') for temp in f.readlines()][0]

rural_sublist = [s for s in sublist if s in rural_subs]
urban_sublist = [s for s in sublist if s in urban_subs]


data = ' '.join(rural_sublist)
with open(path+'rural-sublist.txt','w') as f:
    f.write(data)
data = ' '.join(urban_sublist)
with open(path+'urban-sublist.txt','w') as f:
    f.write(data)
sys.exit(0)
#%% Discard lines which are partially within the state
# sub_list = subs['NAME'].values.tolist()
# idx1 = [i for i,x in enumerate(lines['SUB_1'].values) if x not in sub_list]
# idx2 = [i for i,x in enumerate(lines['SUB_2'].values) if x not in sub_list]
# line_idx = list(set(idx1).union(set(idx2)))
# lines.drop(lines.index[line_idx], inplace=True)

#%% Check voltage level of individual lines
# voltage = lines['VOLTAGE'].tolist()

#%% Plot the transmission network
linecolor = 'royalblue'
subcolor = 'brown'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
ax.set_xlim(-128,-65)
ax.set_ylim(22,52)
data_lines.plot(ax=ax,edgecolor=linecolor,linewidth=0.8)
data_substations.plot(ax=ax,edgecolor=subcolor,markersize=2.0)
ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

axins = zoomed_inset_axes(ax,2.4,3)
axins.set_aspect(1.3)

# Draw nodes and edges
subs.plot(ax=axins,edgecolor=subcolor,markersize=2.0)
lines1.plot(ax=axins,edgecolor=linecolor,linewidth=0.8)
lines2.plot(ax=axins,edgecolor=linecolor,linewidth=0.8)

axins.tick_params(bottom=False,left=False,
                  labelleft=False,labelbottom=False)
mark_inset(ax, axins, loc1=2, 
           loc2=4, fc="none", ec="0.5")

leghands = [Line2D([0], [0], color=linecolor, markerfacecolor=linecolor, 
                   marker='o',markersize=0,label='transmission network'),
            Line2D([0], [0], color='white', markerfacecolor=subcolor, 
                   marker='o',markersize=10,label='transmission substations')]
ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 20})
fig.savefig("{}{}.png".format(figpath,'eia-data'),bbox_inches='tight')



#%% Plot substations
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
subs.plot(ax=ax,edgecolor=subcolor,markersize=10.0)
lines1.plot(ax=ax,edgecolor=linecolor,linewidth=1.2,linestyle='dotted')
lines2.plot(ax=ax,edgecolor=linecolor,linewidth=1.2,linestyle='dotted')
ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 20})
fig.savefig("{}{}.png".format(figpath,'eia-va-data'),bbox_inches='tight')














