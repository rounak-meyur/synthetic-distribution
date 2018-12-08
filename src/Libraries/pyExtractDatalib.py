# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:46:36 2018

Authors: Dr Anil Vullikanti
         Dr Henning Mortveit
         Rounak Meyur
"""

import sys
import argparse
import cx_Oracle
import pickle


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