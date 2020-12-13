# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:19:25 2020

@author: Rounak Meyur
Description: Creates the home csv from Swapna's output directory.
"""

import os
import pandas as pd


swapnaPath = "/sfs/lustre/bahamut/scratch/st6ua/"


def GetHourlyDemand(path):
    """
    Generates the load data as required by rest of the program from the output
    of synthetic load generation module developed by Swapna.

    Parameters
    ----------
    path : string
        directory path of Swapna's output csv files.

    Returns
    -------
    None.

    """
    filelist = [f for f in os.listdir(path) if f.endswith('.csv')]
    for i,filename in enumerate(filelist):
        print(i,filename)
        fiscode = filename.lstrip('VA').rstrip('.csv')
        df_all = pd.read_csv(path+filename)
        df_home = df_all[['hid','longitude','latitude']]
        for i in range(1,25):
            df_home['hour'+str(i)] = df_all['P'+str(i)]+df_all['A'+str(i)]
        df_home.to_csv(path+'load/'+fiscode+'-home-load.csv',index=False)
    return


GetHourlyDemand(swapnaPath)
print("DONE")
