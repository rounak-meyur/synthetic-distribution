# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:46:36 2018

Authors: Dr Anil Vullikanti
         Dr Henning Mortveit
         Rounak Meyur

Description: This library contains classes, methods etc to extract data from
the noldor database with required credentials. 
"""

from __future__ import print_function

import sys
import argparse
import cx_Oracle
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from collections import namedtuple as nt


# -----------------------------------------------------------------------------
# Classes
    
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

#%% Oracle related functions
# -----------------------------------------------------------------------------
# Functions

def CreateParser():
    '''
    '''
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


def CreateConnection(username, passwd, host, port, service):
    '''
    Connect with the noldor Oracle database with necessary information
    
    Inputs: username: Username for the user account in the noldor synthetic 
                      population database
            passwd: Password for the user account in the noldor database
            host: host ID for the account
            port: port number
            service: service ID
    
    Output: conn: connection object
    '''
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


def CreateCursor(connection):
    '''
    '''
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
        print (dbe.message)
        sys.exit(-1)
    
    try :
        c = CreateCursor(conn)
    except DBError as dbe :
        print (dbe.message)
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
        print (dbe.message)
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
        print (dbe.message)
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
        print (dbe.message)
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



#%% Classes

class Query:
    """
    """
    def __init__(self,csvpath):
        '''
        '''
        self.csvpath = csvpath
        return
    
    
    def GetRoads(self,level=[3,4,5],thresh=10):
        '''
        Method to create a named tuple to store the road network data.
        '''
        df_all = pd.read_csv(self.csvpath+'road-graph.csv')
        df_road = df_all.loc[df_all['level'].isin(level)]
        df_cord = pd.read_csv(self.csvpath+'road-cord.csv')
        
        # Networkx graph construction
        root = nx.from_pandas_edgelist(df_road,'source','target','level')
        comps = [c for c in list(nx.connected_components(root)) if len(c)>thresh]
        graph = nx.Graph()
        for c in comps: graph=nx.compose(graph,root.subgraph(c))
        
        cords = df_cord.set_index('node').T.to_dict('list')
        road = nt("road",field_names=["graph","cord"])
        return road(graph=graph,cord=cords)
    
    
    def GetSubstations(self):
        '''
        Method to create a named tuple to store the substation location data which 
        is obtained from EIA database.
        '''
        df_cord = pd.read_csv(self.csvpath+'subs-cord.csv')
        cords = df_cord.set_index('node').T.to_dict('list')
        subs = nt("substation",field_names=["cord"])
        return subs(cord=cords)
    
    
    def GetHourlyDemand(self,inppath,filename,cordfile):
        '''
        '''
        df_load = pd.read_csv(inppath+filename)
        df_cord = pd.read_csv(inppath+cordfile,
                                usecols=['HID','Longitude','Latitude'])
        df_cord = df_cord[['HID','Longitude','Latitude']]
        df_cord.to_csv(self.csvpath+'home-cord.csv',index=False)
        df_demand = pd.read_csv(inppath+cordfile,usecols=['HID'])
        #df_base = (df_load['hotWaterWH'] + df_load['standbyWH'])/24.0
        df_base = (df_load['standbyWH'])/24.0
        for i in range(1,25):
            df_demand['hour'+str(i)] = df_load['P'+str(i)]+df_load['A'+str(i)]+df_base
        df_demand.to_csv(self.csvpath+'home-load.csv',index=False)
        return
    
    
    def GetHomes(self):
        '''
        '''
        df_home = pd.merge(pd.read_csv(self.csvpath+'home-cord.csv'),
                           pd.read_csv(self.csvpath+'home-load.csv'),on='HID')
        df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:].values,axis=1))
        df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:].values,axis=1))
        
        home = nt("home",field_names=["cord","profile","peak","average"])
        dict_cord = pd.read_csv(self.csvpath+'home-cord.csv').set_index('HID').T.to_dict('list')
        dict_load = pd.read_csv(self.csvpath+'home-load.csv').set_index('HID').T.to_dict('list')
        dict_peak = dict(zip(df_home.HID,df_home.peak))
        dict_avg = dict(zip(df_home.HID,df_home.average))
        homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
        
        gdf_home = gpd.GeoDataFrame(df_home.loc[:,['HID','Longitude','Latitude','peak','average']],
                                    geometry=gpd.points_from_xy(df_home.Longitude,
                                                                df_home.Latitude))
        return gdf_home,homes
    
    
    def get_tsfr_to_link(self):
        """
        """
        df_tsfr = pd.read_csv(self.csvpath+'tsfr2link.csv')
        tsfr = nt("Transformers",field_names=["cord","link"])
        dict_cord = dict([(t.TID, (t.longitude, t.latitude)) \
                          for t in df_tsfr.itertuples()])
        dict_link = dict([(t.TID, (t.source, t.target)) \
                          for t in df_tsfr.itertuples()])
        return tsfr(cord=dict_cord,link=dict_link)
    
    
    def GetTransformers(self):
        """
        """
        df_tsfr = pd.read_csv(self.csvpath+'tsfr-cord-load.csv')
        tsfr = nt("Transformers",field_names=["cord","load","graph"])
        dict_cord = dict([(t.TID, (t.long, t.lat)) for t in df_tsfr.itertuples()])
        dict_load = dict([(t.TID, t.load) for t in df_tsfr.itertuples()])
        df_tsfr_edges = pd.read_csv(self.csvpath+'tsfr-net.csv')
        g = nx.from_pandas_edgelist(df_tsfr_edges)
        return tsfr(cord=dict_cord,load=dict_load,graph=g)
        





class EIA:
    """
    """
    def __init__(self,path,line_file="Electric_Power_Transmission_Lines.shp",
                 sub_file="Electric_Substations.shp",state_file="states.shp"):
        """
        """
        data_lines = gpd.read_file(path+line_file)
        data_substations = gpd.read_file(path+sub_file)
        data_states = gpd.read_file(path+state_file)
        
        state_polygon = list(data_states[data_states.STATE_ABBR == 
                                 'VA'].geometry.items())[0][1]
        self.subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
        self.lines = data_lines.loc[data_lines.geometry.intersects(state_polygon)]
        
        sub_list = self.subs['NAME'].values.tolist()
        idx1 = [i for i,x in enumerate(self.lines['SUB_1'].values) if x not in sub_list]
        idx2 = [i for i,x in enumerate(self.lines['SUB_2'].values) if x not in sub_list]
        line_idx = list(set(idx1).union(set(idx2)))
        self.lines.drop(self.lines.index[line_idx], inplace=True)



