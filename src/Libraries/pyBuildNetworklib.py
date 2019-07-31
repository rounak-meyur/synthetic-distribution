# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:41:09 2019

Author: Rounak Meyur
"""


from math import sin, cos, sqrt, atan2, radians

#%% Functions
def MeasureDistance(Point1,Point2):
    '''
    The format of each point is (longitude,latitude).
    '''
    # Approximate radius of earth in km
    R = 6373.0
    
    # Get the longitude and latitudes of the two points
    lat1 = radians(Point1[1])
    lon1 = radians(Point1[0])
    lat2 = radians(Point2[1])
    lon2 = radians(Point2[0])
    
    # Measure the long-lat difference
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Calculate distance between points in km
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance*1000




#%% Classes
class iBus(list):
    """
    """
    def __init__(self,aList):
        '''
        A method to initialize the data for a bus element in the power network.
        '''
        super(iBus,self).__init__(aList)
        self.__dict__['d'] = {'number':0, 'long':1, 'lat':2, 'kv':3, 'type':4}
        self.__dict__['data'] = aList
    
    def __getitem__(self,key):
        '''
        A method to get item from class using the key defined in the constructor.
        '''
        if isinstance(key,str):
            return self.data[self.d[key.lower()]]
        else:
            return self.data[key]
    
    def __setitem__(self,key,value):
        '''
        A method to get set value for a particular attribute of the instance.
        '''
        if isinstance(key,str):
            self.data[self.d[key.lower()]] = value
        else:
            self.data[key] = value



class Bus(list):
    """
    """
    def __getitem__(self,key):
        '''
        Returns the bus instance corresponding to key
        '''
        return iBus(self.data[key])
    
    def __iter__(self):
        '''
        Returns as class objects during iterations
        '''
        for p in self.data:
            yield iBus(p)
    
    def __init__(self,csvfile):
        '''
        '''
        f = open(csvfile,'r')
        cdata = [line.strip('\n').split(',') for line in f.readlines()]
        f.close()
        # Append each line of data one at a time
        self.data = []
        dicref = []
        for k in range(1,len(cdata)):
            L = [int(cdata[k][0]),int(cdata[k][1]),float(cdata[k][2]),
                 float(cdata[k][3]),float(cdata[k][4]),float(cdata[k][5]),
                 float(cdata[k][6]),float(cdata[k][7]),float(cdata[k][8]),
                 float(cdata[k][9]),float(cdata[k][10]),float(cdata[k][11]),
                 float(cdata[k][12])]
            self.__dict__['data'].append(L)
            dicref.append(L[0])
        super(Bus,self).__init__(self.__dict__['data'])
        self.identify = dict(zip(dicref,range(len(cdata))))