# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:05:36 2021

Author: Rounak

Description: This program plots the motifs in distribution network of Virginia.
"""

import sys,os


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
figpath = workpath + "/figs/"
inppath = rootpath + "/input/"
distpath = workpath + "/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet
from pyResiliencelib import star,path
print("Imported modules")


sublist = [int(x.strip("-dist-net.gpickle")) for x in os.listdir(distpath)]
k = 5

#%% k-path motif

data = ''
for sub in sublist:
    dist = GetDistNet(distpath,sub)
    motif = path(dist,k)
    num_nodes = dist.number_of_nodes()
    data += '\t'.join([str(sub), str(num_nodes), str(motif)])+'\n'


with open(workpath+"/out/"+str(k)+"path-motif.txt",'w') as f:
    f.write(data)


#%% k-star motif

data = ''
for sub in sublist:
    dist = GetDistNet(distpath,sub)
    count = star(dist,k)
    data += '\t'.join([str(sub), str(dist.number_of_nodes()), str(int(count))])+'\n'

with open(workpath+"/out/"+str(k)+"star-motif.txt",'w') as f:
    f.write(data)