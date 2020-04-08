# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:15 2019

Author: Rounak Meyur
Description: This program tries to generate ensemble of synthetic networks by varying
parameters in the optimization problem.
"""

import sys,os
workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"
tmpPath = workPath + "/temp/prim-ensemble/"


sub = 24665
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(csvPath+"eigval-"+str(sub)+".csv")
data = df.to_numpy().tolist()

log_data = [np.array([x for i,x in enumerate(eiglist) if str(x)!='nan' and i<=100]) \
            for eiglist in data]

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)
box = ax.boxplot(log_data,patch_artist=True)
ax.tick_params(bottom=False,labelbottom=False)
ax.set_xlabel("Synthetic networks",fontsize=20)
ax.set_ylabel("Minimum 100 eigenvalues of Jacobian",fontsize=20)
fig.savefig(figPath+str(sub)+"-eigval_small.png")
