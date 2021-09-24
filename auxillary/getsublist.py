# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:24:17 2021

@author: rm5nz
"""



import os
# Load scratchpath
scratchPath = "/sfs/lustre/bahamut/scratch/rm5nz/synthetic-distribution"
inpPath = scratchPath + "/input/"
tmpPath = scratchPath + "/temp/"
dirname = 'osm-prim-master/'

flist = os.listdir(tmpPath+dirname)
slist = ' '.join([str(y) for y in sorted([int(x.strip("-master.gpickle")) for x in flist])])
with open(inpPath+"sublist.txt",'w') as f:
    f.write(slist)