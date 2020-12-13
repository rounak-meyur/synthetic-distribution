# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:19:25 2020

@author: rounak
"""

import sys,os
import matplotlib.pyplot as plt
import numpy as np


workPath = os.getcwd()
inpPath = workPath + "/input/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/"
libPath = workPath + "/Libraries/"
sys.path.append(libPath)

with open(inpPath+'arealist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

time_stat = []
for area in areas:
    with open(csvPath+area+'-data/'+area+'-time-stat.txt','r') as f:
        times = [[float(x) for x in lines.strip('\n').split('\t')] \
                 for lines in f.readlines()]
    time_stat.extend(times)

#%% Plot points
from matplotlib.colors import LogNorm
xpts = [time[0] for time in time_stat]
ypts = [time[1] for time in time_stat]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax.scatter(xpts,ypts,s=20,marker='*',c='royalblue')

#%% Box Plot
_, xbins = np.histogram(np.log10(np.array(xpts)),bins=8)
ypts_dict = {'grp'+str(i+1):[np.log10(time[1]) for time in time_stat \
                             if 10**xbins[i]<=time[0]<=10**xbins[i+1]] \
             for i in range(len(xbins)-1)}

xmean = [int(round(10**((xbins[i]+xbins[i+1])/2))) for i in range(len(xbins)-1)]
xmean = [1,3,7,15,31,63,127,255]
ytick_old = np.linspace(-3,4,num=8)
ytick_new = 10**ytick_old

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax.boxplot(ypts_dict.values())
ax.set_xticklabels(xmean)
ax.set_yticklabels(ytick_new)
ax.set_xlabel('Number of residences to be connected',fontsize=15)
ax.set_ylabel('Time (in seconds) to create secondary network',
              fontsize=15)
fig.savefig("{}{}.png".format(figPath,'secnet-time'),bbox_inches='tight')

#%% Get residence/road stats
with open(inpPath+'fislist.txt') as f:
    areas = f.readlines()[0].strip('\n').split(' ')

from pyExtractDatalib import Query

q_object = Query(csvPath,inpPath)
area_stat = {}
for i,f in enumerate(areas[119:]):
    print(f)
    roads = q_object.GetRoads(fis=f)
    homes = q_object.GetHomes(fis=f)
    area_stat[f] = [len(homes.cord),roads.graph.number_of_nodes(),
                    roads.graph.number_of_edges()]


numhome = [area_stat[f][0] for f in areas]
numnode = [area_stat[f][1] for f in areas]
numedge = [area_stat[f][2] for f in areas]


print("Residences:",min(numhome),max(numhome),np.mean(numhome),np.median(numhome))
print("Road nodes:",min(numnode),max(numnode),np.mean(numnode),np.median(numnode))
print("Road edges:",min(numedge),max(numedge),np.mean(numedge),np.median(numedge))


data = '\n'.join([str(f)+'\t'+'\t'.join([str(x) for x in area_stat[f]]) for f in areas])
with open(tmpPath+'area-stat.txt','w') as f:
    f.write(data)