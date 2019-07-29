library(data.table)
library(geosphere)
library(dplyr)

setwd("~/College/Research/Redone with Updated Data")
line = read.csv('Electric_Power_Transmission_Lines.csv')
colnames(line)[1] = "OBJECTID"
line = line[,-which(colnames(line) == "ID")]
object = read.csv('Sub_Plant_Merge.csv')
sub = subset(object, object$Substation == 1)
pp = subset(object, object$Substation == 0)
zip_county = read.csv('~/College/Research/Redone with Updated Data/Helper Data/zipcodes.csv')

#####################################

#1) Delete network points with power plant connections - find network of substations only
net = read.csv('Sub_Plant_Line_Connections_Reduce.csv')
#SUB_PLANT_OBJECTID_1 in net :: OBJECTID_1.. in object
net_sub = subset(net, net$SUB_PLANT_OBJECTID_1 %in% sub$OBJECTID_1..)

#Pull in county data for connections without county data
net_sub = merge(net_sub, zip_county[,c("COUNTYNAME", "ZIP")], by.x = "Zipcode", by.y = "ZIP", all.x =T)
net_sub <- data.frame(lapply(net_sub, as.character), stringsAsFactors=FALSE)
#Fill in missing county data with original county data
na_county = which(is.na(net_sub$COUNTYNAME))
net_sub$COUNTYNAME[na_county] = paste(net_sub$County[na_county], "County", sep = " ")
net_sub = net_sub[,-which(colnames(net_sub) %in% "County")]
colnames(net_sub)[which(colnames(net_sub) == "COUNTYNAME")] = "County"
#Replace NA County with NA
net_sub$County[net_sub$County == "NA County"] <- NA

#Identify points without full state names
stateabb_idx = which(!net_sub$State %in% state.name)
#Change State abbreviation to State Name
net_sub$State[stateabb_idx] = state.name[match(net_sub[stateabb_idx, "State"],state.abb)]
#Remove connections in non-states (PR)
net_sub = net_sub[-which(is.na(net_sub$State)),]

#####################################

#2) Create edge list of substations and line IDs
#Subset net_sub to just SUB_PLANT_OBJECTID_1, LINE_OBJECTID_1
d1 = net_sub[,c("SUB_PLANT_OBJECTID_1", "LINE_OBJECTID_1")]
#Remove duplicates
d1 = d1[!duplicated(d1),]
#Create edge-list of nodes and line ID
edge_list = data.frame()
count = 1
#For each unique Line ID, list the objects that intersect that line
for(i in unique(d1$LINE_OBJECTID_1)) {
  obj = d1[d1$LINE_OBJECTID_1 == i,"SUB_PLANT_OBJECTID_1"]
  #If there is more than one object on the line, save connections to edge_list
  if(length(obj) > 1) {
    #For each object, create connection with next in list and save to edge_list
    for(j in 1:(length(obj)-1)) {
      conn = c(as.numeric(i), as.numeric(obj[j]), as.numeric(obj[j+1]))
      edge_list = rbind(edge_list, conn)
    }
  }
  print(count/length(unique(d1$LINE_OBJECTID_1))*100)
  count = count + 1
}
colnames(edge_list) = c("Line_ID", "Start_Object_ID", "End_Object_ID")

letters_suffix = c(letters, "aa", "bb", "cc", "dd", "ee", "ff")
count= 1
#Split edge list edges into subsets: If edge has multiple connections, make into ID_A, ID_B, etc
for(i in unique(edge_list$Line_ID)) {
  idx = which(edge_list$Line_ID == i)
  if(length(idx) > 1) {
    for(j in 1:length(idx)) {
      edge_list$Line_ID[idx[j]] = paste(edge_list$Line_ID[idx[j]], letters_suffix[j], sep='')
    }
  }
  print(count / length(unique(edge_list$Line_ID))*100)
  count = count + 1
}

#Update sub file to only include substations in the network
sub = sub[which(sub$OBJECTID_1.. %in% c(edge_list$Start_Object_ID, edge_list$End_Object_ID)),]
#####################################

#3) Incorporate power plants by finding substations within 5 miles, find more accurate threshold by analyzing distribution. 
#Threshold can be different between power plants.
#If more than one substation is within 10 miles of pp, then pp is connected to multiple substations

#Longitude: 55 miles ~ 1 dd, 10 mile ~ .18 dd
#Latitude: 69 miles ~ 1 dd, 10 mile ~ .14 dd
#Find all substations within 5 miles :: longitude = .09, latitude = .07 of power plants
#Empty dataframe to append powerplant and closest substation connections
closest_subs = data.frame()
for(i in 1:nrow(pp)) {
  for(j in which((abs(sub$Latitude-pp[i,]$Latitude) < .07) & (abs(sub$Longitude-pp[i,]$Longitude) < .09))) {
    distance = distm(c(pp[i,]$Longitude, pp[i,]$Latitude), c(sub[j,]$Longitude, sub[j,]$Latitude), fun = distHaversine)
    subconn = c(pp[i,]$OBJECTID_1.., sub[j,]$OBJECTID_1.., distance)
    closest_subs = rbind(closest_subs, subconn)
  }
  print(i/nrow(pp)*100)
}
colnames(closest_subs) = c("PowerPlant", "Substation", "Distance")
#91% of power plants have a substation within 5 miles

# #See closest substations- based on closest substations, use 1000 distance as threshold for multiple connections
# min_dist = as.data.frame(closest_subs %>%
#   group_by(PowerPlant) %>%
#   slice(which.min(Distance)))

#Connect Powerplant to closest substations
pp_sub_connect = data.frame()
count = 1
for(i in unique(closest_subs$PowerPlant)) {
  #Subset power plant, substation, distance data to power plant of interest
  temp_pp = subset(closest_subs, closest_subs$PowerPlant == i)
  #If there is only one closest substation within 1000 distance, connect pp to that
  if(sum(temp_pp$Distance < 1000) == 1) {
    pp_sub_connect = rbind(pp_sub_connect, c(i, temp_pp$Substation[which(temp_pp$Distance == min(temp_pp$Distance))], 
                                             temp_pp$Distance[which(temp_pp$Distance == min(temp_pp$Distance))]))
  } else if(sum(temp_pp$Distance < 1000) > 1) {
    #If there is >1 substation within 1000, find if there are any other substations within 1.5 times the distance of the closest
    closest = min(temp_pp$Distance)
    closest_15 = closest*1.5
    #If so, the power plant is connected to both (all) substations
    if(sum(temp_pp$Distance <= closest_15) > 1) {
      for(j in which(temp_pp$Distance <= closest_15)) {
        pp_sub_connect = rbind(pp_sub_connect, c(i, temp_pp$Substation[j], temp_pp$Distance[j]))
      }
    } else {
      #If only 1 within 1.5xs minimum, then just connect to the minimum
      pp_sub_connect = rbind(pp_sub_connect, c(i, temp_pp$Substation[which(temp_pp$Distance == min(temp_pp$Distance))], 
                                                 temp_pp$Distance[which(temp_pp$Distance == min(temp_pp$Distance))]))
    }
  } else {
    #If the closest substation is not within 1000 distance, it is connected to its closest station
    pp_sub_connect = rbind(pp_sub_connect, c(i, temp_pp$Substation[which(temp_pp$Distance == min(temp_pp$Distance))], 
                                             temp_pp$Distance[which(temp_pp$Distance == min(temp_pp$Distance))]))
  }
  print(count/length(unique(closest_subs$PowerPlant))*100)
  count = count + 1
}
colnames(pp_sub_connect) = c("PowerPlant", "Substation", "Distance")
nrow(pp_sub_connect) #8453 (91%) of power plants are connected
length(unique(pp_sub_connect$PowerPlant))/length(unique(pp$OBJECTID_1..)) #87% of the power plants are connected

#####################################

#4) Create artificial lines going from power plant to substation(s) within 5 miles
#All artificial lines start at 74554 (1 above highest recorded line)
pp_sub_line_start = max(net$LINE_OBJECTID_1)
#PP and sub connection in pp_sub_connect
#Format 1) GIS Format: Line | Power Plant | Substation
new_lines_pp = data.frame()
counter = 1
for(i in unique(pp_sub_connect$PowerPlant)) {
  #Number of substations connected to powerplant
  pp_subset = subset(pp_sub_connect, pp_sub_connect$PowerPlant == i)
  num_sub = nrow(pp_subset)
  #For each substation connected to power plant, add line | Plant and line | Sub connection to new_lines_pp
  for(j in 1:num_sub) {
    conn_pp = c(pp_sub_line_start + counter, pp_subset[j,"PowerPlant"], pp_subset[j,"Substation"])
    #Append to new_lines_pp
    new_lines_pp = rbind(new_lines_pp, conn_pp)
  }
  counter = counter + 1
}

colnames(new_lines_pp) = c("Line_ID", "Start_Object_ID", "End_Object_ID")

#Add data to new lines pp to add to network (edge_list)
edge_list = rbind(edge_list, new_lines_pp)

#Add artifical lines data to line file
artificial_lines = unique(new_lines_pp$Line_ID)
artificial_lines_df = data.frame('OBJECTID' = artificial_lines, 'TYPE' = rep("ARTIFICIAL", length(artificial_lines)))
line = bind_rows(line, artificial_lines_df) #Ignore warnings


#####################################

#5) Estimate demand for counties using population data and CA data
setwd("~/College/Research/Redone with Updated Data/Helper Data")
ca_demand = read.csv("Demand_by_County_CA.csv")
us_pop = read.csv("population_by_county_2016.csv")[,1:3] #2016 only data

ca_demand$State = "California"
ca_demand_ref = merge(ca_demand, us_pop, by = c("County", "State"), all.x = T, all.y = F)
ca_demand_ref$usage_by_person = ca_demand_ref$Total.Usage / ca_demand_ref$pop_2016
#Find average usage by person
use_by_person = mean(ca_demand_ref$usage_by_person) #.00858
#Estimate usage by county for all US
us_pop$Estimated_Usage_2016 = round(us_pop$pop_2016 * use_by_person,2)

#####################################

#6) Substations within a county delivers power to that county.
#If a county doesn't have a substation in it, find county center and find substation closest to county center. 

#Identify substations in the network 
s_network = sub[sub$OBJECTID_1.. %in% unique(c(edge_list$Start_Object_ID, edge_list$End_Object_ID)),"OBJECTID_1.."]

#Change county with tilde to non-tilde
us_pop <- data.frame(lapply(us_pop, as.character), stringsAsFactors=FALSE)
us_pop$County[us_pop$County == "Doña Ana County"] = 'Dona Ana County'

#Identify counties with substations in network
net_sub_in = net_sub[net_sub$SUB_PLANT_OBJECTID_1 %in% s_network,c("State", "County", "SUB_PLANT_OBJECTID_1")]

#Identify counties without substations in network
net_sub_out = anti_join(us_pop[,c("County", "State")], net_sub_in[,c("County", "State")], by = c("County", "State"))

#Remove counties in Alaska and Hawaii since those aren't in substation data
net_sub_out = net_sub_out[-which(net_sub_out$State %in% c("Alaska", "Hawaii")),]

#Find County Centers
county_centers = read.csv('~/College/Research/Redone with Updated Data/Helper Data/County_Centers.csv')

#Find County Centers for counties with missing substations
net_sub_out = merge(net_sub_out, county_centers, by = c("State", "County"), all.x = T, all.y = F)

#Pull in long/lat data for substations in network
net_sub_in = merge(net_sub_in, sub[,c('OBJECTID_1..', 'Longitude', 'Latitude')], by.x = 'SUB_PLANT_OBJECTID_1', by.y = 'OBJECTID_1..', all.x = T, all.y = F)

#Create empty dataframe to save substations that serve counties without other substations in them
implied_sub_demand = data.frame()

#Find closest counties without substations to substations in network
for(i in 1:nrow(net_sub_out)) {
  dist_lat = .09
  dist_long = .07
  found = FALSE
  #If there are substations within (.07, .09) decimal degres of county center, then find the closest substation
  while(found == FALSE) {
    close_subs = which((abs(net_sub_in$Latitude-net_sub_out[i,]$Latitude) < dist_lat) & (abs(net_sub_in$Longitude-net_sub_out[i,]$Longitude) < dist_long))
    if(length(close_subs) > 0) {
      #Create empty dataframe to save closest index and distances
      close_subs_info = data.frame()
      for(j in close_subs) {
        #Calculate distance between center and each substation within radius, save distance to substation ID
        distance = distm(c(net_sub_out[i,]$Longitude, net_sub_out[i,]$Latitude), c(net_sub_in[j,]$Longitude, net_sub_in[j,]$Latitude), fun = distHaversine)
        close_subs_info = rbind(close_subs_info, c(j, distance))
      }
      #Find the index of the closest sub
      colnames(close_subs_info) = c("idx", "dist")
      min_idx = close_subs_info$idx[which.min(close_subs_info$dist)]
      #Save the information of the closest sub: Sub_ID | County | State 
      og_info = net_sub_in[min_idx,]
      #Remove long/lat and Update county/state to reflect new substation
      mod_info = og_info[,-which(colnames(og_info) %in% c("Longitude", "Latitude"))]
      mod_info$County = net_sub_out[i, "County"]
      mod_info$State = net_sub_out[i, "State"]
      #Save new connection to implied_sub_demand
      implied_sub_demand = rbind(implied_sub_demand, mod_info)
      found = TRUE
    } else {
      #If no substations within range are found, expand range by 2
      dist_lat = dist_lat * 2
      dist_long = dist_long * 2
    }
    print(i/nrow(net_sub_out)*100)
  }
}

#Join implied_sub_demand with substations and their original counties
sub_counties = rbind(implied_sub_demand, net_sub_in[,-which(colnames(net_sub_in) %in% c("Longitude", "Latitude"))])
#Remove duplicates
sub_counties = sub_counties[!duplicated(sub_counties),]

#For substations with multiple states/counties, combine in [array]
sub_counties_joined = data.frame()
count = 1
for(i in unique(sub_counties$SUB_PLANT_OBJECTID_1)) {
  sub_counties_joined = data.frame(lapply(sub_counties_joined, as.character), stringsAsFactors=FALSE)
  s1 = sub_counties[sub_counties$SUB_PLANT_OBJECTID_1 == i,]
  if(nrow(s1) > 1) {
    #Aggregate multiple counties and states as county = [c1, c2, c3]
    counties = "["
    states = "["
    for(j in 1:nrow(s1)) {
      if(j == 1) {
        counties = paste(counties, paste(as.character(s1$County[j]), ',', sep=''), sep = "")
        states = paste(states, paste(as.character(s1$State[j]), ',', sep=''), sep = "")
      } else if(j == nrow(s1)) {
        counties = paste(counties, as.character(s1$County[j]), sep = " ")
        states = paste(states, as.character(s1$State[j]), sep = " ")
      } else {
        counties = paste(counties, paste(as.character(s1$County[j]), ',', sep=''), sep = " ")
        states = paste(states, paste(as.character(s1$State[j]), ',', sep=''), sep = " ")
      }
    }
    counties = paste(counties, ']', sep = "")
    states = paste(states, ']', sep = "")
    sub_counties_joined = rbind(sub_counties_joined, c(as.numeric(i), as.character(states), as.character(counties)))
  } else {
    #Aggregate original row
    conn = c(as.numeric(s1$SUB_PLANT_OBJECTID_1), as.character(s1$State), as.character(s1$County))
    sub_counties_joined = rbind(sub_counties_joined, conn)
  }
  print(count / length(unique(sub_counties$SUB_PLANT_OBJECTID_1))*100)
  count = count + 1
}
colnames(sub_counties_joined) = c("Sub_ID", "State", "County")


#####################################
#FINAL DATAASETS

#1) Line | Start | End  (edge_list)
#write.csv(edge_list, "Edge_List.csv")

#2) Line | Line Length | Voltage
d2 = data.frame("Line_ID" = unique(edge_list$Line_ID))
d2 = merge(d2, line[,c("VOLTAGE", "SHAPE__Length", "OBJECTID")], by.x = "Line_ID", by.y = "OBJECTID", all.x = T, all.y = F)
colnames(d2) = c("Line_ID", "Voltage", "Line_Length_Meters")
#write.csv(d2, "Line_Length_Voltage.csv")

#3) Plant/Sub ID | Long | Lat 
d3 = data.frame("Object_ID" = unique(c(edge_list$Start_Object_ID, edge_list$End_Object_ID)))
d3 = merge(d3, object[,c("Longitude", "Latitude", "OBJECTID_1..")], by.x = "Object_ID", by.y = "OBJECTID_1..", all.x = T, all.y = F)
colnames(d3) = c("Object_ID", "Longitude", "Latitude")
#write.csv(d3, 'Object_Longitude_Latitude.csv')

#4) Substation ID | County Served | State Served | County_State Served
#If sub serves more than one county: County = [c1, c2], State = [S1, S2], County_State = [c1_s1, c2_s2]
d4 = sub_counties_joined
#write.csv(d4, "Sub_Counties_States.csv")

#5) County | State | Demand 
d5 = us_pop[,-which(colnames(us_pop) == "pop_2016")]
d5$County_State = paste(d5$County, d5$State, sep = "_")
#write.csv(d5, "County_State_Demand.csv")

#6) Plant ID | MW produced | Primsource
d6 = data.frame("Object_ID" = unique(c(edge_list$Start_Object_ID, edge_list$End_Object_ID)))
d6 = merge(d6, pp[,c("Total_MW", "PrimSource", "OBJECTID_1..")], by.x = "Object_ID", by.y = "OBJECTID_1..", all.x = F, all.y = F)
colnames(d6) = c("Plant_ID", "Total_MW", "PrimSource")
#write.csv(d6, "Plant_MWProduce_PrimarySource.csv")

