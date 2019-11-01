# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:03:48 2019

@author: Rounak Meyur
"""


import sys,os
import numpy as np


workPath = os.getcwd()
inpPath = workPath + "/input/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/"

sys.path.append(libPath)
from pyMILPlib import MILP_EV
#%% Main Function
        
        
T=24
n1 = np.random.randint(low=0,high=10,size=(T,))
n2 = np.random.randint(low=0,high=5,size=(T,))

Sch = np.zeros(shape=(4000,T))
Psh = np.zeros(shape=(4000,T))

s = np.array([])
a = np.array([],dtype=int)
c = np.array([])
ev_chosen = 0
for t in range(T):
    s1 = np.random.randint(low=10,high=30,size=(n1[t],))
    s2 = np.random.randint(low=10,high=30,size=(n2[t],))
    a1 = np.random.randint(low=1,high=T-t+1,size=(n1[t],))
    a2 = np.random.randint(low=1,high=T-t+1,size=(n2[t],))
    c = np.hstack((c,np.array([15]*n1[t]+[10]*n2[t])))
    
    old_ev = len(s)
    s = np.hstack((s,np.hstack((s1,s2))))
    a = np.hstack((a,np.hstack((a1,a2))))
    
    m = MILP_EV(s,a,c,old_ev,t,T=T)
    W = m.W_opt
    P = m.P_opt
    u = m.u_opt
    
    
    chosen_ind = np.where(u==1)[0]
    ev_chosen = len(chosen_ind)
    
    Sch[:ev_chosen,t] = W[chosen_ind,0]
    Psh[:ev_chosen,t] = P[chosen_ind,0]
    
    s = s[chosen_ind]-P[chosen_ind,0]
    a = a[chosen_ind] - np.where(a[chosen_ind]>0,1,0)
    c = c[chosen_ind]
    print("Number of vehicles chosen",ev_chosen)
    
        
        
        
        
        
        
        