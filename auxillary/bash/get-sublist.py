# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:50:33 2020

@author: rounak
"""

# Get list of substations
import os
# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
tmpPath = scratchPath + "/temp/"

filelist = [f for f in os.listdir(tmpPath+'prim-master') \
            if f.endswith('-master.gpickle')]
print(len(filelist))
sublist = [f.split('-')[0] for f in filelist]
print(len(sublist))
data = ' '.join(sublist)
with open(inpPath+'sublist.txt','w') as f:
    f.write(data)