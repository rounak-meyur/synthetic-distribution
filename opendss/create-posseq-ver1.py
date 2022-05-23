# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:41:29 2022

author: Rounak

Description: Creates a OpenDSS file for the network with necessary models.
Version 1: 
    Line wire data: conductor data for one type only
    Line geometry data: 4 wire 3 phase at a single road side pole geometry
"""

import os,sys
import networkx as nx



workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = rootpath + "/libs/"
distpath = rootpath + "/primnet/out/osm-primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet


#New	Line.05410_10000339601OH	
# bus1=N284033_lo.3	
# bus2=N284031.3	
# geometry=OH3P_FR8_N56_OH_1/0_AAAC_6201_OH_1/0_AAAC_6201_CN	
# phases=1	
# length=22.9889	
# units=ft

sub = 121144
dist = GetDistNet(distpath,sub)

geom_spec = "OH3P_FR8_N56_OH_477_AAC_OH_336_AAC_ABCN"

tnodes = {n:len([m for m in nx.neighbors(dist,n) if dist.nodes[m]['label']=='H']) \
          for n in dist if dist.nodes[n]['label']=='T'}
t_dict = {t:[str(x+1) for x in range(tnodes[t])] for t in tnodes}

edge_data = []
for e in dist.edges:
    if dist.edges[e]['label'] in ['P','E']:
        # Bus IDs
        bus1 = "Bus1="+str(e[0])+'.1.2.3'
        bus2 = "Bus2="+str(e[1])+'.1.2.3'
        # Line ID
        lineID = "Line."+'_'.join([str(x) for x in e])+'_OH'
    elif dist.edges[e]['label'] == 'S':
        # Bus 1 ID: if transformer, label it as LV side with identifier
        if dist.nodes[e[0]]['label'] == 'T':
            bus1 = "Bus1="+str(e[0])+"_T"+t_dict[e[0]].pop(0)+"_lv.1.2.3"
        else:
            bus1 = "Bus1="+str(e[0])+".1.2.3"
        # Bus 2 ID: if transformer, label it as LV side with identifier
        if dist.nodes[e[1]]['label'] == 'T':
            bus2 = "Bus2="+str(e[1])+"_T"+t_dict[e[1]].pop(0)+"_lv.1.2.3"
        else:
            bus2 = "Bus2="+str(e[1])+".1.2.3"
        # Line ID
        lineID = "Line."+bus1.lstrip("Bus1=").rstrip(".1.2.3")+"_"+\
           bus2.lstrip("Bus2=").rstrip(".1.2.3") +'_OH'
    # Line Geometry
    geom = "Geometry="+geom_spec
    # Phases
    phase = "Phases=3"
    # Length of line
    length = "Length="+str(dist.edges[e]["length"]*3.28084)
    unit = "Units=ft"
    edge_data.append('\t'.join(["New",lineID,bus1,bus2,geom,phase,length,unit]))

t_dict = {t:[str(x+1) for x in range(tnodes[t])] for t in tnodes}
for t in t_dict:
    for i in t_dict[t]:
        # Line ID
        lineID = "Line."+str(t)+"_T"+str(i)+'_COND'
        # Bus IDs
        bus1 = "Bus1="+str(t)+'.1.2.3'
        bus2 = "Bus2="+str(t)+"_T"+str(i)+'_hv.1.2.3'
        
        # Line Geometry
        code = "Linecode=Busbar"
        # Phases
        phase = "Phases=3"
        # Length of line
        length = "Length=0"
        unit = "Units=ft"
        edge_data.append('\t'.join(["New",lineID,bus1,bus2,geom,phase,length,unit]))



data = '\n'.join(edge_data)
with open("Lines_"+str(sub)+".dss",'w') as f:
    f.write(data)





















