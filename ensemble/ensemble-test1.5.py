# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:10:18 2021

@author: rounak
Description: Compares the efficiency of the ensemble of networks
"""

import sys,os
import matplotlib.pyplot as plt
# import seaborn as sns


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
figpath = workpath + "/figs/"
outpath = workpath + "/out/"


with open(outpath+"osm-ensemble-efficiency.txt") as f:
    lines = f.readlines()

data = [temp.strip("\n").split("\t") for temp in lines]
dict_eff = {int(d[0]):[float(x) for x in d[1].split(',')] for d in data}

sublist = sorted([k for k in dict_eff])
efflist = [dict_eff[k][1:] for k in sublist]
eff_opt = [dict_eff[k][0] for k in sublist]


DPI = 72    
fig = plt.figure(figsize=(1000/DPI, 600/DPI), dpi=DPI)
ax = fig.add_subplot(111)

bp = ax.boxplot(efflist)
ax.scatter(range(1,len(sublist)+1),eff_opt,marker='^',s=50,c='red')

ax.set_xticklabels([str(x) for x in sublist],rotation=90)
ax.set_xlabel("Substation ID for the network ensemble",fontsize=14)
ax.set_ylabel("Efficiency",fontsize=14)

fig.savefig("{}{}.png".format(figpath,"ensemble-efficiency"),
            bbox_inches='tight')

