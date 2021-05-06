# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:50:33 2020

@author: rounak
"""

# Get list of substations
import os


# Load scratchpath
scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inppath = scratchpath + "/input/"
tmppath = scratchpath + "/temp/"


filelist = [f for f in os.listdir(tmppath+'osm-prim-master') \
            if f.endswith('-master.gpickle')]
sublist = [f.split('-')[0] for f in filelist]
sublist = sorted([int(x) for x in sublist])



data = ' '.join([str(x) for x in sublist])
with open(inppath+'osm-sublist.txt','w') as f:
    f.write(data)
