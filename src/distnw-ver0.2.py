# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the home csv from Swapna's output directory.
"""

import os
import pandas as pd

workPath = os.getcwd()
inpPath = workPath + "/input/"
csvPath = workPath + "/csv/"


def GetHourlyDemand(inppath,csvpath):
    '''
    '''
    filelist = [f for f in os.listdir(inppath) if f.endswith('.csv')]
    for i,filename in enumerate(filelist):
        print(i,filename)
        fiscode = filename.lstrip('VA').rstrip('.csv')
        df_all = pd.read_csv(inppath+filename)
        df_home = df_all[['hid','longitude','latitude']]
        for i in range(1,25):
            df_home['hour'+str(i)] = df_all['P'+str(i)]+df_all['A'+str(i)]
        df_home.to_csv(csvpath+fiscode+'-home-load.csv',index=False)
    return


GetHourlyDemand(inpPath,csvPath)
print("DONE")
