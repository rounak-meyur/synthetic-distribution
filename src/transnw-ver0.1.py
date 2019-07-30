# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:09:06 2019

Author: Rounak Meyur
        Srijan Sengupta
        Rachel Szabo
        Shivani Garg
"""

import sys,os
import networkx as nx


# Get the different directory loaction into different variables
workPath = os.getcwd()
pathLib = workPath + "\\Libraries\\"
pathBin = workPath + "\\tmp\\"
pathFig = workPath + "\\figs\\"
pathInp = workPath + "\\input\\"

# User defined libraries
sys.path.append(pathLib)

def UpdateNeighborVoltage(Graph,g,volt,genlist,ref_volt):
    '''
    Find voltage at neighboring buses for generators and try to estimate voltage
    of generator 'g'.
    
    Input:  Graph: Networkx graph
            g: generator node
            volt: dictionary of node voltages
    '''
    nodes = {0:[g]}
    for hop in range(1,5):
        node_src = nodes[hop-1]
        node_neighbors = []
        for nd in node_src:
            if hop == 1:
                node_neighbors.extend(Graph.neighbors(nd))
            else:
                node_neighbors.extend([n for n in Graph.neighbors(nd) \
                                       if (n not in nodes[hop-2]) and (n not in genlist)])
        nodes[hop] = node_neighbors
        
        nonzero_voltage = [volt[n] for n in node_neighbors if volt[n] > 0.0]
        if nonzero_voltage != []:
            base = min(nonzero_voltage)
        else:
            base = ref_volt[hop]
        for n in nodes[hop]:
            if volt[n] == 0.0: volt[n] = base
    return




sys.exit(0)
#%% Determine voltage of edges
edgevoltage_file = pathInp + 'Data-Rachel/' + 'Line_Length_Voltage.csv'
f = open(edgevoltage_file,'r')
edge_voltage_data = [e.strip('\n').split(',') for e in f.readlines()[1:]]
f.close()
edge_voltage = {int(e[0]):float(e[1]) if (e[1]!='-999999') and (e[1]!='NA') else 0.0 \
                for e in edge_voltage_data}
edge_length = {int(e[0]):float(e[2]) if (e[1]!='-999999') and (e[1]!='NA') else 0.0 \
               for e in edge_voltage_data}


edgelist_file = pathInp + 'Data-Rachel/' + 'Edge_List.csv'
f = open(edgelist_file,'r')
edge_data = [e.strip('\n').split(',') for e in f.readlines()[1:]]
f.close()
attr_edge_voltage = {(int(e[1]),int(e[2]),str(e[0])):[edge_voltage[int(e[3])],\
                     edge_length[int(e[3])]] for e in edge_data}

#%% Create Graph with voltage as an attribute
G = nx.MultiGraph()
for e in edge_data:
    G.add_edge(int(e[1]),int(e[2]),str(e[0]))
nx.set_edge_attributes(G,attr_edge_voltage,'voltage')

#%% Node information
nodelist_file = pathInp + 'Object_Longitude_Latitude.csv'
f = open(nodelist_file,'r')
node_data = [n.strip('\n').split(',') for n in f.readlines()[1:]]
f.close()
dict_node = {int(n[0]):(float(n[1]),float(n[2]))for n in node_data}

#%% Generting station information
genlist_file = pathInp + 'Plant_MWProduce_PrimarySource.csv'
f = open(genlist_file,'r')
gen_data = [g.strip('\n').split(',') for g in f.readlines()[1:]]
f.close()
dict_gen = {int(g[0]):float(g[1]) for g in gen_data}
gen_type = {int(g[0]):g[2] for g in gen_data}
genlist = dict_gen.keys()

#%% Determine voltage of nodes
#node_voltage = {nd: 0.0 for nd in dict_node}
#for e in attr_edge_voltage:
#    if (attr_edge_voltage[e] != 0.0):
#        node_voltage[e[0]] = float(attr_edge_voltage[e])
#        node_voltage[e[1]] = float(attr_edge_voltage[e])
#
## Base Generator Voltage Update
#basegen = [g for g in genlist if gen_type[g] not in ['solar','wind','batteries','biomass']]
#for g in basegen:
#    if dict_gen[g] < 2.0: 
#        node_voltage[g] = 4.16
#        ref_volt = {1:69.0,2:115.0,3:138.0,4:230.0}
#    elif dict_gen[g] < 100.0: 
#        node_voltage[g] = 6.90
#        ref_volt = {1:69.0,2:115.0,3:138.0,4:230.0}
#    elif dict_gen[g] < 500.0: 
#        node_voltage[g] = 13.8
#        ref_volt = {1:115.0,2:138.0,3:230.0,4:345.0}
#    else: 
#        node_voltage[g] = 22.0
#        ref_volt = {1:115.0,2:138.0,3:230.0,4:345.0}
#    UpdateNeighborVoltage(G,g,node_voltage,genlist,ref_volt)
#
## Peak Generator Voltage Update
#peakgen = [g for g in genlist if gen_type[g] in ['solar','wind','batteries','biomass']]
#for g in peakgen:
#    if dict_gen[g] < 2.0: 
#        node_voltage[g] = 0.69
#    elif dict_gen[g] < 300.0: 
#        node_voltage[g] = 4.16
#    else: 
#        node_voltage[g] = 6.9
#    ref_volt = {1:69.0,2:115.0,3:138.0,4:230.0}
#    UpdateNeighborVoltage(G,g,node_voltage,genlist,ref_volt)
#
## check voltages
#unknown_voltage = [k for k in node_voltage if node_voltage[k] == 0.0 and k not in genlist]
#known_voltage = [k for k in node_voltage if node_voltage[k] != 0.0]
#print "DONE"









