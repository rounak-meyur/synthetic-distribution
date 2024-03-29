# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:37:57 2021

@author: rouna
"""

import os,sys
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
inppath = rootpath + "/input/"
figpath = workpath + "/figs/"
qgispath = inppath + "/actual/"
distpath = rootpath + "/primnet/out/osm-primnet/"

sys.path.append(libpath)
from pyDrawNetworklib import plot_gdf
from pyExtractDatalib import GetOSMRoads,get_areadata,GetDistNet
print("Imported modules")


sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(distpath,sublist)
roads = GetOSMRoads(inppath,'121')

areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
# areas = {'patrick_henry':194,'mcbryde':9001}
area_data = {area:get_areadata(qgispath,area,root,synth_net) \
                      for area,root in areas.items()}

#%% Plot comparison
DPI = 72    
fig = plt.figure(figsize=(10,15), dpi=DPI)
ax = plt.subplot()    
for area in area_data:
    plot_gdf(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],
                 'orangered')
    plot_gdf(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],
                 'blue')
ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)

ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()

roadnodes = [n for n in roads if xmin<=roads.nodes[n]['x']<=xmax \
                 and ymin<=roads.nodes[n]['y']<=ymax]

nodepos = {n:[roads.nodes[n]['x'],roads.nodes[n]['y']] for n in roads}
nx.draw_networkx_edges(roads,pos=nodepos,ax=ax,
                 style='dashed',width=0.8,edge_color='black')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

# Inset figure
sxmin,sxmax = -80.435,-80.42
symin,symax = 37.24,37.255
axins = zoomed_inset_axes(ax, 1.2, 4)
axins.set_aspect(1.3)
for area in area_data:
    plot_gdf(axins,area_data[area]['df_lines'],area_data[area]['df_buses'],
                 'orangered')
    plot_gdf(axins,area_data[area]['df_synth'],area_data[area]['df_cords'],
                 'blue')
axins.set_xlim(sxmin,sxmax)
axins.set_ylim(symin,symax)
axins.tick_params(bottom=False,left=False,
                  labelleft=False,labelbottom=False)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

leglines = [Line2D([0], [0], color='orangered', markerfacecolor='orangered', 
                   marker='o',markersize=10, label='Actual Network'),
            Line2D([0], [0], color='blue', markerfacecolor='blue',
                   marker='o',markersize=10, label='Synthetic Network'),
            Line2D([0], [0], color='black', markerfacecolor='black',linestyle='dashed',
                   marker='o',markersize=0, label='Road Network')]
ax.legend(handles=leglines,loc='best',ncol=1,prop={'size': 9})
fig.savefig("{}{}.png".format(figpath,
        'visual-comparison-blacksburg'),bbox_inches='tight')