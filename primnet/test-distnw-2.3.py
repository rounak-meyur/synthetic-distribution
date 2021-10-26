# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

@author: Rounak

Description: This program provides codes to the distribution lines depending on
the catalogue for primary and secondary lines.
"""

import sys,os
import networkx as nx


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
distpath = workpath + "/out/osm-primnet/"


sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyMiscUtilslib import powerflow
print("Imported modules")



sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]


#%% Plot variation in voltage
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import numpy as np

# sublist = [121144]

for sub in sublist[1:]:
    dist = GetDistNet(distpath,sub)
    powerflow(dist,v0=1.02)
    volt = {n:dist.nodes[n]['voltage'] for n in dist}
    reach = {n:1e-3*nx.shortest_path_length(dist,sub,n,weight='length') for n in dist}
    vlist = [dist.nodes[n]['voltage'] for n in dist]
    N = dist.number_of_nodes()
    
    # Voltage Histogram
    fig1 = plt.figure(figsize=(20,12))
    ax1 = fig1.add_subplot(111)
    ax1.hist(vlist,bins=50,weights=np.ones(N)/N,color='royalblue',edgecolor='black')
    ax1.set_xlim(left=0.95,right=1.05)
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.set_xlabel("Node voltage (in p.u.)",fontsize=25)
    ax1.set_ylabel("Percentage of nodes",fontsize=25)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    # ax1.grid(color='black',linestyle='dashed',linewidth=0.8)
    fig1.savefig("{}{}.png".format(figpath,str(sub)+'-voltage-hist'),
                bbox_inches='tight')
    
    # Voltage tree
    # fig2 = plt.figure(figsize=(20,12))
    # ax2 = fig2.add_subplot(111)
    # for e in dist.edges:
    #     xpt = [reach[e[0]],reach[e[1]]]
    #     ypt = [volt[e[0]],volt[e[1]]]
    #     if dist.edges[e]['label']=='E':
    #         ax2.plot(xpt,ypt,color='dodgerblue',
    #                  marker='o',markersize=10.0,linewidth=3.0)
    #     elif dist.edges[e]['label']=='P':
    #         ax2.plot(xpt,ypt,color='black',
    #                  marker='o',markersize=5.0,linewidth=2.0)
    #     else:
    #         ax2.plot(xpt,ypt,color='red',
    #                  marker='o',markersize=2.5,linewidth=0.5)
    
    # ax2.set_xlabel("Distance from substation (in km)",fontsize=25)
    # ax2.set_ylabel("Voltage at node (in p.u.)",fontsize=25)
    # ax2.grid(color='black',linestyle='dashed',linewidth=0.3)
    
    # ax2.tick_params(axis="x", labelsize=20)
    # ax2.tick_params(axis="y", labelsize=20)
    
    # # Legends
    # leghands = [Line2D([0], [0], color='dodgerblue', marker='o',markersize=10,
    #                    label='HV feeder lines',linewidth=3.0),
    #             Line2D([0], [0], color='black', linewidth=2.0,
    #                marker='o',markersize=10,label='MV primary lines'),
    #             Line2D([0], [0], color='red', linewidth=0.5,
    #                marker='o',markersize=5,label='LV secondary lines')]
    # ax2.legend(handles=leghands,loc='lower left',ncol=1,prop={'size': 25})
    
    # fig2.savefig("{}{}.png".format(figpath,str(sub)+'-voltage-profile'),
    #             bbox_inches='tight')
    
    
sys.exit(0)

#%% Catalogue of distribution lines
from math import exp
import matplotlib.pyplot as plt

synth_net = GetDistNet(distpath,sublist)
prim_amps = {e:2.2*exp(synth_net[e[0]][e[1]]['flow'])/6.8 \
             for e in synth_net.edges if synth_net[e[0]][e[1]]['label']=='P'}
sec_amps = {e:1.5*exp(synth_net[e[0]][e[1]]['flow'])/0.12 \
            for e in synth_net.edges if synth_net[e[0]][e[1]]['label']=='S'}


edge_name = {}
for e in synth_net.edges:
    
    # names of secondary lines
    if synth_net[e[0]][e[1]]['label']=='S':
        if sec_amps[e]<=95:
            edge_name[e] = 'OH_Voluta'
        elif sec_amps[e]<=125:
            edge_name[e] = 'OH_Periwinkle'
        elif sec_amps[e]<=165:
            edge_name[e] = 'OH_Conch'
        elif sec_amps[e]<=220:
            edge_name[e] = 'OH_Neritina'
        elif sec_amps[e]<=265:
            edge_name[e] = 'OH_Runcina'
        else:
            edge_name[e] = 'OH_Zuzara'
    
    # names of primary lines
    elif synth_net[e[0]][e[1]]['label']=='P':
        if prim_amps[e]<=140:
            edge_name[e] = 'OH_Swanate'
        elif prim_amps[e]<=185:
            edge_name[e] = 'OH_Sparrow'
        elif prim_amps[e]<=240:
            edge_name[e] = 'OH_Raven'
        elif prim_amps[e]<=315:
            edge_name[e] = 'OH_Pegion'
        else:
            edge_name[e] = 'OH_Penguin'
    else:
        edge_name[e] = 'OH_Penguin'


names = [edge_name[e] for e in edge_name]
name_order = ['OH_Voluta','OH_Periwinkle','OH_Conch','OH_Neritina','OH_Runcina',
              'OH_Zuzara','OH_Swanate','OH_Sparrow','OH_Raven','OH_Pegion',
              'OH_Penguin']

#%% Bar plot of line name statistics
import pandas as pd
from collections import Counter

DPI = 72    
fig1 = plt.figure(figsize=(1000/DPI, 1000/DPI), dpi=DPI)
fig2 = plt.figure(figsize=(1000/DPI, 1000/DPI), dpi=DPI)
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

name_counts = Counter(names)
data = {n:name_counts[n] for n in name_order[:6]}
df = pd.DataFrame.from_dict(data, orient='index',columns=['Number of lines'])
df.plot(y='Number of lines',kind='pie',legend=True,
        ax=ax1,autopct='%1.1f%%',fontsize=14,startangle=0,
        title="Pie chart of overhead LV secondary distribution lines")
fig1.savefig("{}{}.png".format(figpath,'sec-lines'),bbox_inches='tight')

data = {n:name_counts[n] for n in name_order[6:]}
df = pd.DataFrame.from_dict(data, orient='index',columns=['Number of lines'])
df.plot(y='Number of lines',kind='pie',legend=True,
        ax=ax2,autopct='%1.1f%%',fontsize=14,startangle=0,
        title="Pie chart of overhead MV primary distribution lines")
fig2.savefig("{}{}.png".format(figpath,'prim-lines'),bbox_inches='tight')

















