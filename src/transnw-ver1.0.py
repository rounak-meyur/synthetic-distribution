# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:54:45 2020

@author: rounak
"""

import os
import geopandas as gpd


state = 'OK'

path = os.getcwd()+"/input/eia/"
line_file="Electric_Power_Transmission_Lines.shp"
sub_file="Electric_Substations.shp"
state_file="states.shp"
    
data_lines = gpd.read_file(path+line_file)
data_substations = gpd.read_file(path+sub_file)
data_states = gpd.read_file(path+state_file)

state_polygon = list(data_states[data_states.STATE_ABBR == 
                         state].geometry.items())[0][1]
subs = data_substations.loc[data_substations.geometry.within(state_polygon)]
lines = data_lines.loc[data_lines.geometry.intersects(state_polygon)]

#%% Discard lines which are partially within the state
# sub_list = subs['NAME'].values.tolist()
# idx1 = [i for i,x in enumerate(lines['SUB_1'].values) if x not in sub_list]
# idx2 = [i for i,x in enumerate(lines['SUB_2'].values) if x not in sub_list]
# line_idx = list(set(idx1).union(set(idx2)))
# lines.drop(lines.index[line_idx], inplace=True)

#%% Check voltage level of individual lines
voltage = lines['VOLTAGE'].tolist()