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





#%% Functions
def get_lines(graph):
    
    # geometry specifications
    # need to update in the future versions
    # geom_spec = "OH3P_FR8_N56_OH_477_AAC_OH_336_AAC_ABCN"
    
    # transformer nodes
    # connect multiple pole top transformers at these locations
    tnodes = {n:len([m for m in nx.neighbors(graph,n) \
                     if graph.nodes[m]['label']=='H']) \
              for n in graph if graph.nodes[n]['label']=='T'}
    t_dict = {t:[str(x+1) for x in range(tnodes[t])] for t in tnodes}
    
    
    # Construct the edge data
    edge_data = []
    for e in graph.edges:
        # Length of line and units
        l = graph.edges[e]["length"]*3.28084
        length = "Length="+str(l)
        unit = "Units=ft"
        if graph.edges[e]['label'] == 'E':
            # Bus IDs
            bus1 = "Bus1="+str(e[0])+'.1.2.3'
            bus2 = "Bus2="+str(e[1])+'.1.2.3'
            # Line ID
            lineID = "Line."+'_'.join([str(x) for x in e])+'_OH'
            # Line geometry
            if l <= 5.0:
                geom = "Linecode=Busbar"
            else:
                geom = "Geometry=feeder"
            # Phases
            phase = "Phases=3"
        if graph.edges[e]['label'] == 'P':
            # Bus IDs
            bus1 = "Bus1="+str(e[0])+'.1.2.3'
            bus2 = "Bus2="+str(e[1])+'.1.2.3'
            # Line ID
            lineID = "Line."+'_'.join([str(x) for x in e])+'_OH'
            # Line Geometry
            if l <= 5.0:
                geom = "Linecode=Busbar"
            else:
                geom = "Geometry=primary"
            # Phases
            phase = "Phases=3"
        elif graph.edges[e]['label'] == 'S':
            # Bus 1 ID: if transformer, label it as LV side with identifier
            if graph.nodes[e[0]]['label'] == 'T':
                bus1 = "Bus1="+str(e[0])+"_T"+t_dict[e[0]].pop(0)+"_lv.1.2.3"
            else:
                bus1 = "Bus1="+str(e[0])+".1.2.3"
            # Bus 2 ID: if transformer, label it as LV side with identifier
            if graph.nodes[e[1]]['label'] == 'T':
                bus2 = "Bus2="+str(e[1])+"_T"+t_dict[e[1]].pop(0)+"_lv.1.2.3"
            else:
                bus2 = "Bus2="+str(e[1])+".1.2.3"
            # Line ID
            lineID = "Line."+bus1.lstrip("Bus1=").rstrip(".1.2.3")+"_"+\
               bus2.lstrip("Bus2=").rstrip(".1.2.3") +'_OH'
            # Line Geometry
            geom = "Geometry=secondary"
            # Phases
            phase = "Phases=3"
        
        edge_data.append('\t'.join(["New",lineID,bus1,bus2,
                                    geom,phase,length,unit]))
    
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
            edge_data.append('\t'.join(["New",lineID,bus1,bus2,
                                        code,phase,length,unit]))
    
    return edge_data


def get_tsfr(graph,s):
    # Construct a tree from the network
    tree = nx.dfs_tree(graph,s)
    # transformer nodes
    # connect multiple pole top transformers at these locations
    tnodes = {n:[m for m in nx.neighbors(graph,n) if graph.nodes[m]['label']=='H'] \
              for n in graph if graph.nodes[n]['label']=='T'}
    t_dict = {t:[str(x+1) for x in range(len(tnodes[t]))] for t in tnodes}
    
    # Get load for each transformer
    prim_edges = [e for e in tree.edges if graph.edges[e]['label']!='S']
    tree.remove_edges_from(prim_edges)
    
    # Get load per each branch originating from transformer node
    t_load = {t:[graph.nodes[n]['load'] + sum([graph.nodes[m]['load'] \
            for m in nx.descendants(tree,n)]) for n in tnodes[t]] \
            for t in tnodes}
    
    # Get the appropriate transformer rating
    kva_ratings = []
    
    
    
    tsfr_data = []
    
        
    
    return tsfr_data


#%% Test code
# sub = 121144
# dist = GetDistNet(distpath,sub)
# filename = "Lines_"+str(sub)+".dss"

g = nx.Graph()
edge_label = {(0,1):'E',(0,2):'E',(1,3):'P',(2,4):'P',(3,5):'P',
              (3,6):'S',(3,7):'S',(4,8):'S',(5,9):'S'}
edge_length = {e:10.0 for e in edge_label}
edgelist = [e for e in edge_label]
node_label = {0:'S',1:'R',2:'R',3:'T',4:'T',5:'T',6:'H',7:'H',8:'H',9:'H'}

g.add_edges_from(edgelist)
nx.set_edge_attributes(g, edge_label,'label')
nx.set_edge_attributes(g, edge_length,'length')
nx.set_node_attributes(g, node_label,'label')


line_data = get_lines(g)
data = '\n'.join(line_data)
filename = "Lines_test.dss"
with open(filename,'w') as f:
    f.write(data)




































































#%%
# with open("transformers_ckt24.dss") as f:
#     lines = f.readlines()

# tsfr_info = [temp.strip('\n').split('\t') for temp in lines]
# tsfr_rating = {}
# for t_data in tsfr_info:
#     key = [t_data[2],t_data[3],t_data[5],t_data[6],
#            t_data[7],t_data[9],t_data[10]]
#     tsfr_rating["\t".join(key)] = "\t".join(t_data[11:])


# data = '\n'.join(sorted(['\t'.join([t_data,tsfr_rating[t_data]]) for t_data in tsfr_rating]))
# with open("transformer-catalog.txt",'w') as f:
#     f.write(data)















