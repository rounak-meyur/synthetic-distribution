# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:10:54 2018

@author: rounakm8
"""

#from __future__ import print_function

#import cx_Oracle
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy as np
import argparse
import sys,os,os.path
import datetime
from sklearn.cluster import KMeans
import networkx as nx
import pickle
import csv
import networkx as nx
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    #print type(rVal1[0])
    #for i in range(len(rVal1)):
      #print "bbb", i, rVal1[i]
    
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
    
def print_k_items(D, k):
  numitems=1
  for i in D:
    print "aaa", i, D[i]
    numitems += 1
    if (numitems ==k): return

# -----------------------------------------------------------------------------
# Remove substations located at remote locations
    
def GetRegIndex(dict_subs,dict_locs):
    
    # Get the XY coordinates for the substations and homes
    subs = dict_subs.values()
    locs = dict_locs.values()
    #print_k_items(dict_subs, 5)
    #print_k_items(dict_locs, 5)
    #print "ccc", subs[0:4]
    
    # Get the Voronoi Regions and region indices for the home/activity locations
    points = np.array(subs)
    voronoi_kdtree = cKDTree(points)
    regions = voronoi_kdtree.query(locs, k=1)
    regInd = list(regions[1])
    #print regions
    
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


def get_csvlist(G,dict_point):
    
    path = os.getcwd()
    subdir_path = os.path.join(path,"Output_data")
    if os.path.exists(subdir_path)==False:
        os.mkdir("Output_data")
    
    suffix = datetime.datetime.now().isoformat()
    suffix = suffix.replace(':','-')
    suffix = suffix.replace('.','-')
    
    f_node = open(os.path.join(subdir_path,"nodelist"+suffix+".csv"),'w')
    f_edge = open(os.path.join(subdir_path,"edgelist"+suffix+".csv"),'w')
    
    for node in dict_point.keys():
        temp1 = [node,dict_point[node][0],dict_point[node][1]]
        f_node.write(','.join(str(x) for x in temp1)+'\n')
        
    for edge in list(G.edges()):
        temp4 = [edge[0],edge[1]]
        f_edge.write(','.join(str(x) for x in temp4)+'\n')
    
    f_node.close()
    f_edge.close()
    
    return


# -----------------------------------------------------------------------------
# main function

def get_files_from_db(homeact_out_fname, subpos_out_fname):
    
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

    homeact_output = open(homeact_out_fname, 'wb')
    pickle.dump(Position_HomeActv, homeact_output)

    
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

    subpos_output = open(subpos_out_fname, 'wb')
    pickle.dump(Position_Subs, subpos_output)
    subpos_output.close()
    
    # -------------------------------------------------------------------------
    # Close connection with database
    conn.close()

def map_homes_to_substations(homeact_out_fname, subpos_out_fname):
    pkl_file1 = open(homeact_out_fname, 'rb')
    Position_HomeActv = pickle.load(pkl_file1)
    pkl_file1.close()
    pkl_file2 = open(subpos_out_fname, 'rb')
    Position_Subs = pickle.load(pkl_file2)
    pkl_file2.close()
    
    # -------------------------------------------------------------------------
    # Get vornoi region for homes and activity centers
    # Net1 is a dictionary with key as the substation ID and value as an array of
    # location IDs of the homes/activity points
    return GetRegIndex(Position_Subs,Position_HomeActv)

#return coordinates of home locs
def get_home_coord(homeact_out_fname):
    pkl_file1 = open(homeact_out_fname, 'rb')
    Position_HomeActv = pickle.load(pkl_file1)
    pkl_file1.close()
    #for i in Position_HomeActv: print "aaa", i, Position_HomeActv[i]
    return Position_HomeActv

#return coordinates of home locs
def get_substation_coord(subpos_out_fname):
    pkl_file2 = open(subpos_out_fname, 'rb')
    Position_Subs = pickle.load(pkl_file2)
    pkl_file2.close()
    #for i in Position_Subs: print "aaa", i, Position_Subs[i]
    return Position_Subs
    
def voronoi_regions(homeact_out_fname, subpos_out_fname):
    pkl_file1 = open(homeact_out_fname, 'rb')
    Position_HomeActv = pickle.load(pkl_file1)
    pkl_file1.close()
    pkl_file2 = open(subpos_out_fname, 'rb')
    Position_Subs = pickle.load(pkl_file2)
    pkl_file2.close()
    
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
    #nx.draw_networkx(GRAPH,pos=Coordinates,with_labels=False,node_size=1,
                     #nodelist=GRAPH.nodes(),node_color=n_col,edgelist=GRAPH.edges(),
                     #edge_color=e_col)
    #plt.show()
    
     
    get_csvlist(GRAPH,Coordinates)
    
    
    return

def read_road_network(road_network_fname, road_coord_fname):
  road_f_reader = csv.reader(open(road_network_fname), delimiter = '\t')
  G=nx.Graph()
  V={}
  Vinv={}
  for line in road_f_reader:
    if ('#' in line[0]): continue
    #G.add_edge(line[0], line[1], weight=line[2])
    v = line[0]; w=line[1]
    if (line[0] not in V): 
      #v=G.add_node(line[0])
      G.add_node(line[0])
      #print "added ", line[0], v
      V[line[0]]=1
      #Vinv[v] = line[0]
    #else: v=V[line[0]]
    if (line[1] not in V): 
      G.add_node(line[1])
      #print "added ", line[1], w
      V[line[1]]=1
      #Vinv[w] = line[1]
    #else: w=V[line[1]]
    if (line[2] == '4' or line[2] == '5'): G.add_edge(v, w, weight=line[2])
    #print "added ", v, w
    #if (line[2] == '4' or line[2] == '5'): G.add_edge(line[0], line[1], weight=line[2])  
  weight = nx.get_edge_attributes(G, 'weight')
  #drop isolated nodes
  #for v in G.nodes(): print G.degree(v)
  #for e in G.edges(): print e, weight[e]
  for v in G.nodes():
    if (G.degree(v)==0): G.remove_node(v)
  
  RoadNW_coord={}
  coord_f_reader = csv.reader(open(road_coord_fname), delimiter = '\t')
  for line in coord_f_reader:
    if ('#' in line[0]): continue
    RoadNW_coord[line[0]] = float(line[1]), float(line[2])
  
  return V, Vinv, RoadNW_coord, G

#homeact_out_fname: pkl file with home locations
#RoadNW_coord: dictionary with coordinates of each node in the RoadNW
def map_homes_to_RoadNW(homeact_out_fname, RoadNW_coord):
    pkl_file1 = open(homeact_out_fname, 'rb')
    Position_HomeActv = pickle.load(pkl_file1)
    pkl_file1.close()

    #voronoi_kdtree = cKDTree(np.array(RoadNW_coord.values()))
    #regions = voronoi_kdtree.query(np.array(Position_HomeActv.values()), k=1)

    L1=[]; M1={}; N1={}
    for v in RoadNW_coord:
      L1.append(RoadNW_coord[v])
      M1[v] = len(L1)-1
      N1[len(L1)-1]=v
    L2=[]; M2={}
    for v in Position_HomeActv:
      L2.append(Position_HomeActv[v])
      M2[v] = len(L2)-1
    voronoi_kdtree = cKDTree(np.array(L1))
    regions = voronoi_kdtree.query(np.array(L2), k=1)
    Home2RW={}; RW2Home={}
    for v in Position_HomeActv:
      nbr = (regions[1])[M2[v]]
      Home2RW[v] = N1[nbr]
      if (N1[nbr] not in RW2Home): RW2Home[N1[nbr]]=[v]
      else: RW2Home[N1[nbr]].append(v)
      #print v, N1[nbr]
    #for i in RW2Home:
      #print "aaa", i, RW2Home[i]
    return Home2RW, RW2Home

#    Z={}
#    for i in regions[1]:
#      if (i not in Z): Z[i]=1
#      else: Z[i] = Z[i]+1
#    for i in Z: print i, Z[i]

#run DFS on G to get a forest F
#within each tree T in F, run pre_order
#in pre_order, return sequence of RW nodes in RW2Home
def  make_forest(G, RW2Home, RoadNW_coord):
  #print list(nx.dfs_preorder_nodes(G))
#  G_tmp = nx.Graph()
#  for v in G.nodes():
#    #G_tmp.add_node(v)
#    for w in G.neighbors(v):
#      if (int(v) > int(w)): continue
#      dist = np.sqrt((RoadNW_coord[v][0] - RoadNW_coord[w][0])*(RoadNW_coord[v][0] - RoadNW_coord[w][0])+ (RoadNW_coord[v][1] - RoadNW_coord[w][1])*(RoadNW_coord[v][1] - RoadNW_coord[w][1]))
#      G_tmp.add_edge(v, w, weight = dist)
#  T = nx.minimum_spanning_tree(G_tmp)
#  print "bbb", T.number_of_nodes(), T.number_of_edges()

#  for v in G.nodes():
#    T=nx.dfs_tree(G, v)
#    draw_graph(T, RoadNW_coord, "tmp/foo1.pdf", justplot=0)
#    break
  
  Traversed={}
  D_preorder_lists={}
  comp_n = 0
  L=[]; L1=[]

#  for v in T.nodes():
#    if (v in Traversed): continue
#    L=list(nx.dfs_preorder_nodes(T, v))
#    L1=[]
#    for w in L: 
#      Traversed[w]=1
#      if (w in RW2Home): L1.append(w)
#    if (len(L1)>0):
#      D_preorder_lists[comp_n] = copy.copy(L1)
#      comp_n += 1

  #f = plt.figure(figsize=(10,10))
  for v in G.nodes():
    if (v in Traversed): continue
    L=list(nx.dfs_preorder_nodes(G, v))
    L1=[]
    for w in L: 
      Traversed[w]=1
      if (w in RW2Home): L1.append(w)
    if (len(L1)>0):
      D_preorder_lists[comp_n] = copy.copy(L1)
      H=nx.Graph()
      for i in range(1, len(L1)):
        H.add_edge(L1[i-1], L1[i])
      #draw_graph(H, RoadNW_coord, "tmp/foo2.pdf", justplot=1)
      comp_n += 1
  #f.savefig("tmp/foo2.pdf", bbox_inches='tight')

  return D_preorder_lists

#Home2sub: dictionary where Home2sub[i] gives the homes associated with substation i
#D_preorder is a set of lists, each corresponding to preorder
def path_cover(Home2RW, Homes2sub, D_preorder, RW2Home, RoadNW_coord, Sub_coord):
  RW_nodes_list={}
  for i in D_preorder:
    for j in D_preorder[i]: RW_nodes_list[j]=i

  Sub_map={}
  for i in Homes2sub:
    #print i, Homes2sub[i]
    for j in Homes2sub[i]: 
      if (Home2RW[j] in RW_nodes_list): 
        Sub_map[j] = i
        Sub_map[Home2RW[j]] = i
      #if (Home2RW[j] not in RW_nodes_list): print "bbb", j

  #create Paths with segment size k
  Paths={}
  k=10
  for s in Sub_map.values():
    Paths[s]={}
  for i in D_preorder:
    L=D_preorder[i]
    L1=[]; L1.append(L[0])
    for j in range(1, len(L)):
      if (L[j] not in RW_nodes_list): continue
      if (Sub_map[L[j]] != Sub_map[L[j-1]]): #end previous list, create new one
        s = Sub_map[L[j-1]]; n=len(Paths[s]); Paths[s][n+1] = copy.copy(L1)
        L1=[]; L1.append(L[j])
      else: #continue in previous L1, unless segment length reached
        if (len(L1) < k): L1.append(L[j])
        else:
          s = Sub_map[L[j-1]]; n=len(Paths[s]); Paths[s][n+1] = copy.copy(L1)
          L1=[]; L1.append(L[j]) 

    
  #################
  ## create Paths without bound on segment size
#  Paths={}
#  for s in Sub_map.values():
#    Paths[s]={}
#  for i in D_preorder:
#    current_sub=-1
#    L=[]
#    for j in D_preorder[i]:
#      if (j not in RW_nodes_list): continue
#      if (current_sub < 0): current_sub = Sub_map[j]
#      elif (Sub_map[j] == current_sub): L.append(j)
#      else:
#        #print "aaa", j
#        if (len(L)==0):
#          current_sub=Sub_map[j]
#          L.append(j)
#        else: #new list starting
#          #print Sub_map[L[0]], L
#          s=Sub_map[L[0]]
#          n=len(Paths[s])
#          Paths[s][n+1] = copy.copy(L)
#          L=[]
#          current_sub=Sub_map[j]
#          L.append(j)
########## end of Paths

    #if (len(L)): print L
#  for s in Paths:
#    for n in Paths[s]:
#      L=Paths[s][n]
#      for i in range(len(L)):
#        print "aaa", RoadNW_coord[L[i]]
  return Paths


def draw_graph(G, Coords, fname, justplot, shape=0, color=0):
  if (justplot==0): f = plt.figure(figsize=(10,10))
  for e in G.edges():
    #x=[RoadNW_coord[e[0]][0], RoadNW_coord[e[1]][0]]
    #y=[RoadNW_coord[e[0]][1], RoadNW_coord[e[1]][1]]
    x=[Coords[e[0]][0], Coords[e[1]][0]]
    y=[Coords[e[0]][1], Coords[e[1]][1]]
      #print "s=", s, "n=", n, "i=", i, "x=", x, "y=", y
    if (shape==0 and color==0): plt.plot(x, y, 'r-')
    if (shape==1 and color==0): plt.plot(x, y, 'ro-')
    if (shape==1 and color==1): plt.plot(x, y, color = 'grey', linestyle='-', marker='o', alpha=0.3)
  if (justplot==0): f.savefig(fname, bbox_inches='tight')

def draw_paths(Paths, RoadNW_coord, Sub_coord):
  fname = "tmp/dist-minus-substation.pdf"
  #fname = "tmp/dist-with-substation.pdf"
  f = plt.figure(figsize=(10,10))
  outf = open("tmp/dist.txt", 'w')
  #map RW coords to ids
  RW_ids={}; id=1
  for i in RoadNW_coord:
    RW_ids[i] = id; id += 1
  #map substation ids
  Sub_ids={}
  for i in Sub_coord:
    Sub_ids[i] = id; id +=1
  #plt.plot(range(10), range(10), "o")
  #plt.show()
  for s in Paths:
    for n in Paths[s]:
      L=Paths[s][n]
      #draw line from substation to transformer
      H=nx.Graph(); H.add_edge(s, L[0])
      coord={}; coord[s] = Sub_coord[s]; coord[L[0]] = RoadNW_coord[L[0]]
      outf.write(str(Sub_ids[s]) + ' S ' + str(coord[s][0])+ ' ' + str(coord[s][1])+' '+ str(RW_ids[L[0]]) + ' T ' + str(coord[L[0]][0])+' '+str(coord[L[0]][1])+'\n')
      #draw_graph(H, coord, fname, justplot=1, shape=1, color=1)
      #draw remaining segments
      H=nx.Graph()
      for i in range(1, len(L)):
        H.add_edge(L[i-1], L[i])
        outf.write(str(RW_ids[L[i]]) + ' L ' + str(RoadNW_coord[L[i-1]][0])+ ' '+str(RoadNW_coord[L[i-1]][1])+' '+ str(RW_ids[L[i]]) + ' L ' + str(RoadNW_coord[L[i]][0])+' '+str(RoadNW_coord[L[i]][1])+'\n')
      draw_graph(H, RoadNW_coord, fname, justplot=1)
#      x = [Sub_coord[s][0], RoadNW_coord[L[0]][0]]
#      y = [Sub_coord[s][1], RoadNW_coord[L[0]][1]]
#      #plt.plot(x, y, 'ro-')
#      for i in range(1, len(L)):
#        x = [RoadNW_coord[L[i-1]][0], RoadNW_coord[L[i]][0]]
#        y = [RoadNW_coord[L[i-1]][1], RoadNW_coord[L[i]][1]]
#        #print "s=", s, "n=", n, "i=", i, "x=", x, "y=", y
#        print RoadNW_coord[L[i]]
#        plt.plot(x, y, 'b-')
  f.savefig(fname, bbox_inches='tight')
  #f.savefig("tmp/foo.pdf", bbox_inches='tight')

if __name__ == "__main__":
  #######overall structure
  #read substations and home locations into pkl files
  #read road network
  #drop level 1-3 links
  #run DFS
  #drop isolated nodes
  #map each home location to road network node
  #map each road network node to closest substation
  #for each voronoi region of substation
  #	construct subtrees for this region
  #	run pre-order traversal
  #	break into segments and add transformer every k steps
  #	connect transformer to substation

  road_network_fname = "input/core-link-file-Montgomery-VA.txt"
  road_coord_fname = "input/node-geometry-Montgomery-VA.txt"

  homeact_out_fname = "tmp/homeact_pkl_file.dat"
  subpos_out_fname = "tmp/subpos_pkl_file.dat"
  #get substations and home locations
  #get_files_from_db(homeact_out_fname, subpos_out_fname)
  Home_coord = get_home_coord(homeact_out_fname)
  Sub_coord = get_substation_coord(subpos_out_fname)


  #read road network and coordinates
  #drop level 1-3 links
  V, Vinv, RoadNW_coord, G = read_road_network(road_network_fname, road_coord_fname) 
  #run DFS

  #map home locations to closest road network nodes
  Home2RW, RW2Home = map_homes_to_RoadNW(homeact_out_fname, RoadNW_coord)

  #run DFS on RoadNW and shortcut
  #D_preorder is a dictionary where each value is a list of RW nodes
  #	(in preorder) having homes mapped to it
  D_preorder=make_forest(G, RW2Home, RoadNW_coord)

  #voronoi_regions(homeact_out_fname, subpos_out_fname)
  Homes2sub = map_homes_to_substations(homeact_out_fname, subpos_out_fname)

  #convert the voronoi region of each substation into lists, based on preorder
  Paths = path_cover(Home2RW, Homes2sub, D_preorder, RW2Home, RoadNW_coord, Sub_coord)

  #draw_paths(Paths, RoadNW_coord, Sub_coord)

#main()
  sys.exit(0)

