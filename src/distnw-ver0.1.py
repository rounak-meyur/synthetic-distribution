# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:10:54 2018

@author: rounakm8
"""

#from __future__ import print_function

import cx_Oracle
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx


# -----------------------------------------------------------------------------
# Command line argument parsing

def CreateParser():
    parser = argparse.ArgumentParser(
            prog = "%prog <username> <password> [options]", 
            
            description = '''This program connects with the noldor database of
            NDSSL and extracts information from the synthetic population data''',
            
            version = "%prog 1.0")
    
    parser.add_argument("username", type = str, metavar = "username", 
                        help = "Username to connect to database")
    
    parser.add_argument("password", type = str, metavar = "password", 
                        help = "Password for the username")
    
    parser.add_argument("-t", "--host", type = str, action = "store",
                        default = "noldor-db.vbi.vt.edu", dest = "host_name", 
                        help = "Host name for the database")
    
    parser.add_argument("-p", "--port", type = int, action = "store",
                        default = 1521, dest = "port_no",
                        help = "Port number to connect to the database")
    
    parser.add_argument("-s", "--service", type = str, action = "store",
                        default = "ndssl.bioinformatics.vt.edu", dest = "serv_name", 
                        help = "Service name to connect to database")
    
    return parser


# -----------------------------------------------------------------------------
# Classes to raise error messages
    
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DBError(Error):
    """Exception raised for errors in database handling.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


# -----------------------------------------------------------------------------
# Connect to noldor database

def CreateConnection(username, passwd, host, port, service) :
    try :
        dsn_tns = cx_Oracle.makedsn(host, port, service_name=service)
    except :
        raise DBError('CreateConnection', 'dsn_tns error')
        return None

    try :
        conn = cx_Oracle.connect(user=username, password=passwd, dsn=dsn_tns)
    except :
        raise DBError('CreateConnection',  'error authenticating')
        return None

    return conn


# -----------------------------------------------------------------------------
# Create cursor object to talk with database
    
def CreateCursor(connection) :
    try :
        c = connection.cursor()
    except :
        raise DBError('CreateCursor', 'Error creating cursor')
        return None

    return c


# -----------------------------------------------------------------------------
# Extract data for substations in the required county
    
def GetConditionHomeDat(counties):
    
    cond = []
    for x in counties:
        cond.append("COUNTY='"+str(x)+"'")
    if len(cond)!=1:
        condition = ' OR '.join(cond)
    else:
        condition = cond[0]
    return condition


# -----------------------------------------------------------------------------
# Extract data for required county
    
def ExtractInfo(cursor,ID,Long,Lat,Zone,Table,Condition):
    
    query = "SELECT %s,%s FROM %s WHERE %s" % (Long, Lat, Table, Condition)
    try:
        cursor.execute(query)
    except:
        raise DBError('Query Error', 'Error in the query sent')
        return None
    rVal1 = cursor.fetchall()
    
    query = "SELECT %s FROM %s WHERE %s" % (ID, Table, Condition)
    try:
        cursor.execute(query)
    except:
        raise DBError('Query Error', 'Error in the query sent')
        return None
    rVal2 = cursor.fetchall()
    
    # Get the IDs in suitable list format
    locID = [x[0] for x in rVal2]
    
    query = "SELECT %s FROM %s WHERE %s" % (Zone, Table, Condition)
    try:
        cursor.execute(query)
    except:
        raise DBError('Query Error', 'Error in the query sent')
        return None
    rVal3 = cursor.fetchall()
    zones = [x[0] for x in list(set(rVal3))]
    
    return (rVal1,locID,zones)


# -----------------------------------------------------------------------------
# Extract data for substations in the required county
    
def GetCondition(zones):
    
    cond = []
    for z in zones:
        cond.append("ZIPCODE='"+str(z)+"'")
    condition = ' OR '.join(cond)
    return condition
    

# -------------------------------------------------------------------------
# Form vornoi regions for the substation centers
    
def GetVornoiRegions(centers,flag):
    
    points = np.array(centers)
    vor = Voronoi(points)
    # Show vornoi regions on a plot if requested
    if flag == True:
        voronoi_plot_2d(vor)
        plt.show()
    return
    

# -----------------------------------------------------------------------------
# Remove substations located at remote locations
    
def GetRegIndex(dict_subs,dict_locs):
    
    # Get the XY coordinates for the substations and homes
    subs = dict_subs.values()
    locs = dict_locs.values()
    
    # Get the Voronoi Regions and region indices for the home/activity locations
    points = np.array(subs)
    voronoi_kdtree = cKDTree(points)
    regions = voronoi_kdtree.query(locs, k=1)
    regInd = list(regions[1])
    
    # Uncomment this line to visualize the vornoi regions
    #GetVornoiRegions(subs,True)
    
    # Form dictionary with keys as substation ID and values as list of location IDs in region
    D = {k: [] for k in dict_subs.keys()}
    for k in range(len(regInd)):
        index_loc = regInd[k]
        D[dict_subs.keys()[index_loc]].append(dict_locs.keys()[k])
    
    return D


# -----------------------------------------------------------------------------
# Cluster the data points using k-means clustering  
def KmeansCluster(HomeActvLocs, MastNet, pts_per_tsfr):
    
    # Help for function
    '''Takes input as the points in the vornoi region and outputs the cluster 
    index in which each point belongs'''
    
    # Initialize Network Dictionary
    FullNet = {}
    TranLocs = {}
    ID = 1000
    for p in MastNet.keys():
        LocsID = MastNet[p]
        
        if len(LocsID)!=0:
            points = [HomeActvLocs[k] for k in LocsID]
            
            # Make an array of points for clustering algorithm
            data = np.array(points)
            
            # Identify the number of centers
            centers = (len(points)/pts_per_tsfr)+1
            
            # Perform the k-means algorithm and identify the center which each point belongs to
            kmeans = KMeans(n_clusters=centers, random_state=0).fit(data)
            kmeans_ind = kmeans.predict(data)
            
            # Form a dictionary for transformer locations
            kmeans_cen = [tuple(i) for i in (kmeans.cluster_centers_)]
            TsfrID = []
            for c in kmeans_cen:
                # Update the transformer dictionary
                TranLocs[ID] = c
                # Update the transformer list for the current region
                TsfrID.append(ID)
                ID = ID + 1
            
            # Initialize a dictionary with keys as cluster number and value as the points in the cluster
            SecNet = {l:[] for l in TsfrID}
            for m in range(len(kmeans_ind)):
                index_loc = kmeans_ind[m]
                SecNet[TsfrID[index_loc]].append(LocsID[m])
            
            # Update the full network
            FullNet[p]=SecNet
    return (FullNet,TranLocs)


# -----------------------------------------------------------------------------
# Create the graph for the distribution network
def create_graph(Net,Subs_pos,HomeActv_pos,Trans_pos):
    G = nx.Graph()
    # Add nodes based on the longitude-latitude information
    for s in Subs_pos.keys():
        G.add_node(s,pos=Subs_pos[s],node_color='r')
    for s in HomeActv_pos.keys():
        G.add_node(s,pos=HomeActv_pos[s],node_color='g')
    for s in Trans_pos.keys():
        G.add_node(s,pos=Trans_pos[s],node_color='b')
    # Add edges based on the connection in the dictionary
    for sub in Net.keys():
        for tsfr in Net[sub].keys():
            G.add_edge(sub,tsfr,edge_color='r')
            for pts in Net[sub][tsfr]:
                G.add_edge(tsfr,pts,edge_color='b')
    return G


# -----------------------------------------------------------------------------
# main function

def main():
    
    # -------------------------------------------------------------------------
    # Parser Construction
    parser = CreateParser()
    args = parser.parse_args()
    if len(vars(args)) != 5:
        parser.print_usage()
        sys.exit(0)
    
    CountyList = [121]
    # -------------------------------------------------------------------------
    # Connect with database
    try :
        conn = CreateConnection(args.username, args.password, args.host_name,
                               args.port_no, args.serv_name)
    except DBError as dbe :
        print dbe.message
        sys.exit(-1)
    
    try :
        c = CreateCursor(conn)
    except DBError as dbe :
        print dbe.message
        sys.exit(-1)
    
    
    # -------------------------------------------------------------------------
    # Get Home Locations
    table1 = 'PROTOPOP.VA_HOME_LOCS_2015_V3'
    Col_0 = 'ID'
    Col_1 = 'X'
    Col_2 = 'Y'
    Col_3 = 'ZONE'
    Cond1 = GetConditionHomeDat(CountyList)
    
    try :
        (Home_Long_Lat,HomeID,Home_Zip) = ExtractInfo(c,Col_0,Col_1,Col_2,Col_3,table1,Cond1)
    except DBError as dbe :
        print dbe.message
        sys.exit(-1)
        
    
    # -------------------------------------------------------------------------
    # Get Activity Locations
    table2 = 'PROTOPOP.VA_ACT_LOCS_2015_V3'
    Col_0 = 'ID'
    Col_1 = 'X'
    Col_2 = 'Y'
    Col_3 = 'ZONE'
    Cond2 = '('+Cond1+') AND STATE=51 AND ZONE!=14472'
    
    try :
        (Actv_Long_Lat,ActvID,Actv_Zip) = ExtractInfo(c,Col_0,Col_1,Col_2,Col_3,table2,Cond2)
    except DBError as dbe :
        print dbe.message
        sys.exit(-1)
    
    
    # -------------------------------------------------------------------------
    # Create a dictionary with key as Home/Activity ID and value as longitude latitude tuple
    Position_HomeActv = {}
    for k in HomeID:
        ind = HomeID.index(k)
        Position_HomeActv[k]=Home_Long_Lat[ind]
    for k in ActvID:
        ind = ActvID.index(k)
        Position_HomeActv[k]=Actv_Long_Lat[ind]

    
    # -------------------------------------------------------------------------
    # Get the list of all zipcodes (homes and activity locations)
    zones = []
    zones.extend(Home_Zip)
    zones.extend(Actv_Zip)
     
    
    # -------------------------------------------------------------------------
    # Get substation locations in the zipcodes
    table3 = 'PROTOPOP.SUBSTATIONS_LOCS'
    Col_0 = 'OBJECTID'
    Col_1 = 'X'
    Col_2 = 'Y'
    Col_3 = 'ZIPCODE'
    Cond3 = GetCondition(zones)
    
    try :
        (Substation_Long_Lat,SubsID,Subs_Zip) = ExtractInfo(c,Col_0,Col_1,Col_2,Col_3,table3,Cond3)
    except DBError as dbe :
        print dbe.message
        sys.exit(-1)
    
    
    # -------------------------------------------------------------------------
    # Create a dictionary with key as Home/Activity ID and value as longitude latitude tuple
    Position_Subs = {}
    for k in SubsID:
        ind = SubsID.index(k)
        Position_Subs[k]=Substation_Long_Lat[ind]
    
    # -------------------------------------------------------------------------
    # Close connection with database
    conn.close()
    
    
    # -------------------------------------------------------------------------
    # Get vornoi region for homes and activity centers
    # Net1 is a dictionary with key as the substation ID and value as an array of
    # location IDs of the homes/activity points
    MasterNet = GetRegIndex(Position_Subs,Position_HomeActv)
    
    # -------------------------------------------------------------------------
    # Get the tree network for the distribution system
    (Network,Position_Trans) = KmeansCluster(Position_HomeActv, MasterNet, 20)
    
    # -------------------------------------------------------------------------
    # Create the network
    GRAPH = create_graph(Network,Position_Subs,Position_HomeActv,Position_Trans)
    
    # -------------------------------------------------------------------------
    # print the graph statistics
    print GRAPH.number_of_nodes()
    print GRAPH.number_of_edges()
    
    Coordinates = {}
    Coordinates.update(Position_Subs)
    Coordinates.update(Position_HomeActv)
    Coordinates.update(Position_Trans)
    
    n_col = []
    for n in GRAPH.nodes():
        if n in Position_Subs.keys():
            n_col.append('r')
        elif n in Position_Trans.keys():
            n_col.append('b')
        elif n in Position_HomeActv.keys():
            n_col.append('g')
    
    e_col = []
    for e in GRAPH.edges():
        if (e[0] in Position_Subs.keys()) or (e[1] in Position_Subs.keys()):
            e_col.append('r')
        else:
            e_col.append('b')
    nx.draw_networkx(GRAPH,pos=Coordinates,with_labels=False,node_size=1,
                     nodelist=GRAPH.nodes(),node_color=n_col,edgelist=GRAPH.edges(),
                     edge_color=e_col)
    plt.show()
     
    
    
    
    return



main()
sys.exit(0)











