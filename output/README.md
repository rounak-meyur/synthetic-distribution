## Output Data

### Optimal Network Dataset
The "optimal" directory consists of shapefiles for nodelist and edgelist of the optimal power network. 
The node attributes are listed below.
- node ID
- node label
	- H  residence
	- T  local transformer
	- S  substation
	- R  auxilary node
- node average hourly load
- node geometry (longitude, latitude)

The edge attributes are listed below.
- node IDs
- edge label
	- E  HV feeder edges
	- P  primary network edges
	- S  secondary network edges
- edge resistance
- edge reactance
- edge line type
- edge length (in meters)
- edge geometry (shapely LineString format)

### Ensembles of Networks 
The "optimal" directory consists of shapefiles for nodelist and edgelist of the optimal power network. 
The node attributes are listed below.
- node ID
- node label
	- H  residence
	- T  local transformer
	- S  substation
	- R  auxilary node
- node average hourly load
- node geometry (longitude, latitude)

The edge attributes are listed below.
- node IDs
- edge label
	- E  HV feeder edges
	- P  primary network edges
	- S  secondary network edges
- edge resistance
- edge reactance
- edge line type
- edge length (in meters)
- edge geometry (shapely LineString format)