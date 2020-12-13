# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:07:26 2020

Author: Rounak Meyur
"""

import networkx as nx
import geopandas as gpd
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
import seaborn as sns
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString, Point
from matplotlib import cm
from matplotlib.lines import Line2D
import itertools


#%% Functions and Classes

def MeasureDistance(pt1,pt2):
    '''
    The format of each point is (longitude,latitude).
    '''
    lon1,lat1 = pt1
    lon2,lat2 = pt2
    geod = Geodesic.WGS84
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12']


def GetSynthNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            load: load for each node for consumers, otherwise it is 0.0
            label: 'H' for home, 'T' for transformer, 'R' for road node, 'S' for subs
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
    """
    graph = nx.Graph()
    for c in code:
        g = nx.read_gpickle(path+str(c)+'-prim-dist.gpickle')
        graph = nx.compose(graph,g)
    return graph




#%% Map homes to the actual network
class Link(LineString):
    """
    Derived class from Shapely LineString to compute metric distance based on 
    geographical coordinates over geometric coordinates.
    """
    def __init__(self,line_geom):
        """
        """
        super().__init__(line_geom)
        self.geod_length = self.__length()
        return
    
    
    def __length(self):
        '''
        Computes the geographical length in meters between the ends of the link.
        '''
        if self.geom_type != 'LineString':
            print("Cannot compute length!!!")
            return None
        # Compute great circle distance
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length


#%% Get area information
class Validate:
    """
    """
    def __init__(self,path,synth,dict_areas):
        """
        Initializes the object for validating a given area.
        Inputs: path: the path where the edge and node shape files are stored
                synth: networkx graph of synthetic primary network
                areas: dictionary of area with root nodes as values
        """
        self.area_data = {area:self.get_areadata(path,area,root,synth) \
                          for area,root in dict_areas.items()}
        return
    
    def get_geometry(self,path,area):
        """
        Loads the shape files for edges and nodes in the region and returns the 
        geometry of the network.

        Parameters
        ----------
        path : string
            path of directory where the shape files are stored.
        area : dictionary
            data keyed by area ID and value as the root node.

        Returns
        -------
        df_buses: geopandas dataframe of nodes
        df_lines: geopandas dataframe of edges
        bus_geom: dictionary of node geometries
        line_geom: dictionary of edge geometries
        """
        # Get dataframe for buses and lines
        df_lines = gpd.read_file(path+area+'/'+area+'_edges.shp')
        df_buses = gpd.read_file(path+area+'/'+area+'_nodes.shp')
        
        # Get dictionary of node geometries and edgelist
        line_geom = {}
        bus_geom = {int(df_buses.loc[i,'id']):df_buses.loc[i,'geometry'].coords[0] \
                      for i in range(len(df_buses))}
        for i in range(len(df_lines)):
            e = df_lines.loc[i,'ID']
            edge = tuple([int(x) for x in e.split('_')])
            geom = [bus_geom[edge[0]]]+df_lines.loc[i,'geometry'].coords[:]\
                +[bus_geom[edge[1]]]
            line_geom[edge]=Link(geom)
        
        return df_buses,df_lines,bus_geom,line_geom
        
    
    def get_network(self,line_geom,bus_geom,synth):
        """
        Gets the actual network, extracts the limits of the region and loads the
        synthetic network of the region.

        Parameters
        ----------
        line_geom : dictionary
            keys are edge IDs and values are edge geometries in actual network
        bus_geom : dictionary
            keys are node IDs and values are node geometries in actual network
        synth : networkx graph
            total synthetic primary network

        Returns
        -------
        act_graph: networkx graph of the actual network
        synth_graph: networkx graph of the synthetic network in the axes limits
        limits: axes limits for the area under consideration

        """
        # Create actual graph network
        edgelist = (line_geom.keys())
        act_graph = nx.Graph()
        act_graph.add_edges_from(edgelist)
    
        # Get coordinate limits
        x_bus = [bus_geom[n][0] for n in bus_geom]
        y_bus = [bus_geom[n][1] for n in bus_geom]
        
        # Get axes limits
        buffer = 0.0
        left = min(x_bus)-buffer
        right = max(x_bus)+buffer
        bottom = min(y_bus)-buffer
        top = max(y_bus)+buffer
        limits = (left,right,bottom,top)
        
        # Get the primary network in the region from the synthetic network
        nodepos = nx.get_node_attributes(synth,'cord')
        nodelist = [n for n in nodepos if left<=nodepos[n][0]<=right \
                     and bottom<=nodepos[n][1]<=top]
        synth_graph = nx.subgraph(synth,nodelist)
        return act_graph,synth_graph,limits


    def get_areadata(self,path,area,root,synth):
        """
        Extracts the data for an area and returns them in a dictionary which
        is labeled with the required entries.

        Parameters
        ----------
        path : string type
            directory path for the shape files of the actual network
        area : string type
            ID of the area for the shape file
        root : integer type
            root node for the area in the shape file
        synth : networkx graph
            complete synthetic primary distribution network

        Returns
        -------
        data: dictionary of area data
            root: root node ID of actual network
            df_lines: geopandas data frame of actual network edges
            df_buses: geopandas data frame of actual network nodes
            bus_geom: geometry of nodes
            line_geom: geometry of edges
            actual: networkx graph of actual network
            synthetic: networkx graph of synthetic network
            limits: axes limits of the region of comparison

        """
        df_buses,df_lines,bus_geom,line_geom = self.get_geometry(path,area)
        act_graph,synth_graph,limits = self.get_network(line_geom,bus_geom,
                                                        synth)
        # Store the data in the dictionary
        data = {'root':root,'limits':limits,'df_lines':df_lines,
                'df_buses':df_buses,'bus_geom':bus_geom,
                'line_geom':line_geom,'actual':act_graph,
                'synthetic':synth_graph}
        return data

    def compare_networks(self,ax,df_lines,df_buses,sgraph):
        """
        Compares the actual and synthetic networks visually.
        """
        # Colors for networks
        act_color = 'orangered'
        synth_color = 'blue'
        
        # Plot the networks
        df_lines.plot(ax=ax,edgecolor=act_color,linewidth=1.0)
        df_buses.plot(ax=ax,color=act_color,markersize=10)
        
        nodepos = nx.get_node_attributes(sgraph,'cord')
        d = {'nodes':list(nodepos.keys()),
             'geometry':[Point(nodepos[n]) for n in nodepos]}
        df_cords = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_cords.plot(ax=ax,color=synth_color,markersize=10)
        # nx.draw_networkx(sgraph,pos=nodepos,with_labels=False,
        #                   ax=ax,node_size=10.0,node_color=synth_color,
        #                   edge_color=synth_color,width=1.0)
        
        line_geom = nx.get_edge_attributes(sgraph,'geometry')
        d = {'edges':list(line_geom.keys()),
             'geometry':list(line_geom.values())}
        df_synth = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_synth.plot(ax=ax,edgecolor=synth_color,linewidth=1.0)
        
        # Define legends for the plot
        leglines = [Line2D([0], [0], color=act_color, markerfacecolor=act_color, 
                           marker='o',markersize=10),
                    Line2D([0], [0], color=synth_color, markerfacecolor=synth_color,
                           marker='o',markersize=10)]
        leglabels = ['Actual distribution network',
                     'Synthetic distribution network']
            
        # Legends
        ax.legend(leglines,leglabels,loc='best',ncol=1,prop={'size': 8})
        ax.tick_params(bottom=False,left=False,labelleft=False,labelbottom=False)
        return
    
    def get_partitions(self,nX,nY):
        """
        Generates partitions in the geographical region and returns the grids. 

        Parameters
        ----------
        nX, nY : integer
            Number of partitions along the X and Y axes.

        Returns
        -------
        LIMITS: tuple of floating point data
            the extreme axes limits of the total area
        GRID: tuple of list of floating point
            the grid partitions with the axes

        """
        # Get the limits of the total area to be analyzed
        lims = np.empty(shape=(len(self.area_data),4))
        for i,area in enumerate(self.area_data):
            lims[i,:] = np.array(self.area_data[area]['limits'])
        LEFT = np.min(lims[:,0]); RIGHT = np.max(lims[:,1])
        BOTTOM = np.min(lims[:,2]); TOP = np.max(lims[:,3])
        
        # Partition into required areas
        xmin = []; xmax = []; ymin = []; ymax = []
        for t in range(nX):
            xmin.append(LEFT+(t/nX)*(RIGHT-LEFT))
            xmax.append(LEFT+((1+t)/nX)*(RIGHT-LEFT))
        for t in range(nY):
            ymin.append(BOTTOM+(t/nY)*(TOP-BOTTOM))
            ymax.append(BOTTOM+((1+t)/nY)*(TOP-BOTTOM))
        LIMITS = (LEFT,RIGHT,BOTTOM,TOP)
        GRID = (xmin,xmax,ymin,ymax)
        return LIMITS,GRID


    def node_stats(self,nX,nY,path):
        """
        Plots the spatial statistical distribution of nodes in a geographical 
        region. 
        """
        LIMITS,GRID = self.get_partitions(nX,nY)
        (LEFT,RIGHT,BOTTOM,TOP) = LIMITS
        (xmin,xmax,ymin,ymax) = GRID
        
        # Initialize figure
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Partition the nodes
        A = np.zeros(shape=(nY,nX))
        B = np.zeros(shape=(nY,nX))
        for area in self.area_data:
            primnet = self.area_data[area]['synthetic']
            nodelabel = nx.get_node_attributes(primnet,'label')
            synthpos = nx.get_node_attributes(primnet,'cord')
            tnodes = {n for n in list(primnet.nodes()) if nodelabel[n]=='T'}
            actpos = self.area_data[area]['bus_geom']
        
            for i in range(nY):
                for j in range(nX):
                    A[i,j] += len([n for n in actpos if xmin[j]<=actpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=actpos[n][1]<=ymax[nY-1-i]])
                    B[i,j] += len([n for n in tnodes if xmin[j]<=synthpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=synthpos[n][1]<=ymax[nY-1-i]])
        
        
            self.compare_networks(ax1,self.area_data[area]['df_lines'],
                             self.area_data[area]['df_buses'],
                             self.area_data[area]['synthetic'])
        
        for i in range(nX-1):
            ax1.plot([xmin[1+i]]*2,[BOTTOM,TOP],color='seagreen',linestyle='dotted')
        for i in range(nY-1):
            ax1.plot([LEFT,RIGHT],[ymin[1+i]]*2,color='seagreen',linestyle='dashed')
        
        # Statistics Figure
        colormap = cm.RdBu
        nan_color = 'white'
        colormap.set_bad(nan_color,1.)
        C = A/np.sum(A) - B/np.sum(B)
        for i in range(nY):
            for j in range(nX):
                if A[i,j]!=0:
                    C[i,j]=100.0*C[i,j]/(A[i,j]/np.sum(A))
                else:
                    C[i,j]=np.nan
        C_masked = np.ma.array(C, mask=np.isnan(C))
        ax2.matshow(C_masked,cmap=colormap,vmin=-100,vmax=100)
        ax2.tick_params(top=False,left=False,labelleft=False,labeltop=False)
        ax2.set_title("Percentage deviation in the spatial distribution of nodes",fontsize=10)
        cobj = cm.ScalarMappable(cmap=colormap)
        cobj.set_clim(vmin=-100,vmax=100)
        cbar = fig.colorbar(cobj,ax=ax2)
        cbar.set_label('Percentage Deviation',size=20)
        cbar.ax.tick_params(labelsize=20)
        
        # Save figure
        fig.savefig("{}{}.png".format(path,
                    'spatial-comparison-'+str(nX)+'-'+str(nY)),
                    bbox_inches='tight')
        return


    def count_motifs(self,g,target):
        count = 0
        for sub_nodes in itertools.combinations(g.nodes(),len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                count += 1
        return count
    
    
    def graphlets(self,target,nX,nY,path):
        """
        """
        LIMITS,GRID = self.get_partitions(nX,nY)
        (LEFT,RIGHT,BOTTOM,TOP) = LIMITS
        (xmin,xmax,ymin,ymax) = GRID
        
        # Initialize figure
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Partition the nodes
        A = np.zeros(shape=(nY,nX))
        B = np.zeros(shape=(nY,nX))
        for area in self.area_data:
            primnet = self.area_data[area]['synthetic']
            actnet = self.area_data[area]['actual']
            synthpos = nx.get_node_attributes(primnet,'cord')
            actpos = self.area_data[area]['bus_geom']
        
            for i in range(nY):
                for j in range(nX):
                    act_nodes = [n for n in actpos if xmin[j]<=actpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=actpos[n][1]<=ymax[nY-1-i]]
                    synth_nodes = [n for n in synthpos if xmin[j]<=synthpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=synthpos[n][1]<=ymax[nY-1-i]]
                    sg_act = nx.subgraph(actnet,act_nodes)
                    sg_synth = nx.subgraph(primnet,synth_nodes)
                    A[i,j] += self.count_motifs(sg_act,target)
                    B[i,j] += self.count_motifs(sg_synth,target)
        
            self.compare_networks(ax1,self.area_data[area]['df_lines'],
                             self.area_data[area]['df_buses'],
                             self.area_data[area]['synthetic'])
        
        for i in range(nX-1):
            ax1.plot([xmin[1+i]]*2,[BOTTOM,TOP],color='seagreen',linestyle='dotted')
        for i in range(nY-1):
            ax1.plot([LEFT,RIGHT],[ymin[1+i]]*2,color='seagreen',linestyle='dashed')
        
        # Figure 2
        colormap = 'RdBu'
        C = A/np.sum(A) - B/np.sum(B)
        for i in range(nY):
            for j in range(nX):
                if A[i,j]!=0:
                    C[i,j]=100.0*C[i,j]/(A[i,j]/np.sum(A))
                else:
                    C[i,j]=np.nan
        C_masked = np.ma.array(C, mask=np.isnan(C))
        ax2.matshow(C_masked,cmap=colormap,vmin=-100,vmax=100)
        ax2.tick_params(top=False,left=False,labelleft=False,labeltop=False)
        ax2.set_title("Percentage Deviation in the spatial distribution of graphlets",
                      fontsize=12)
        cobj = cm.ScalarMappable(cmap=colormap)
        cobj.set_clim(vmin=-100,vmax=100)
        cbar = fig.colorbar(cobj,ax=ax2)
        cbar.set_label('Percentage Deviation',size=20)
        cbar.ax.tick_params(labelsize=20)
        
        # Save figure
        n = target.number_of_nodes()
        fig.savefig("{}{}.png".format(path,
                    str(n)+'-motif-comparison-graphlet-'+str(nX)+'-'+str(nY)),
                    bbox_inches='tight')
        return
    
    
    def degree_dist(self,path):
        """
        Creates the degree distribution of the networks. The synthetic network is compared
        with a base network. The degree distribution of both the networks is plotted 
        together in a stacked plot.
        
        Inputs: graph: synthetic network graph
                base: original network graph
                sub: substation ID
                path: path to save the plot
        """
        for area in self.area_data:
            synth = self.area_data[area]['synthetic']
            act = self.area_data[area]['actual']
            degree_sequence_a = sorted([d for n, d in synth.degree()],
                                       reverse=True)
            degree_sequence_b = sorted([d for n, d in act.degree()],
                                       reverse=True)
            na = synth.number_of_nodes()
            nb = act.number_of_nodes()
            sub = self.area_data[area]['root']
        
            degreeCount_a = collections.Counter(degree_sequence_a)
            degreeCount_b = collections.Counter(degree_sequence_b)
            deg_a = degreeCount_a.keys()
            deg_b = degreeCount_b.keys()
            
            max_deg = min(max(list(deg_a)),max(list(deg_b)))
            cnt_a = []
            cnt_b = []
            for i in range(1,max_deg+1):
                if i in degreeCount_a:
                    cnt_a.append(100.0*degreeCount_a[i]/na)
                else:
                    cnt_a.append(0)
                if i in degreeCount_b:
                    cnt_b.append(100.0*degreeCount_b[i]/nb)
                else:
                    cnt_b.append(0)
            
            cnt_a = tuple(cnt_a)
            cnt_b = tuple(cnt_b)
            deg = np.arange(max_deg)+1
            width = 0.35
            
            # Create the degree distribution comparison
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)
            rects1 = ax.bar(deg, cnt_a, width, color='royalblue')
            rects2 = ax.bar(deg+width, cnt_b, width, color='seagreen')
            ax.set_xticks(deg + width / 2)
            ax.set_xticklabels([str(x) for x in deg])
            ax.legend((rects1[0],rects2[0]),('Synthetic Network', 'Original Network'),
                      prop={'size': 20})
            ax.set_ylabel("Percentage of nodes",fontsize=20)
            ax.set_xlabel("Degree of nodes",fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_title("Degree distribution comparison for network",
                         fontsize=20)
            
            # Save the figure
            filename = "degree-distribution-"+str(sub)
            fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
        return
    
    def compare_length(self,nX,nY,path):
        """
        

        Parameters
        ----------
        nX : TYPE
            DESCRIPTION.
        nY : TYPE
            DESCRIPTION.
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        LIMITS,GRID = self.get_partitions(nX,nY)
        (LEFT,RIGHT,BOTTOM,TOP) = LIMITS
        (xmin,xmax,ymin,ymax) = GRID
        
        # Initialize figure
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Partition the nodes
        A = np.zeros(shape=(nY,nX))
        B = np.zeros(shape=(nY,nX))
        for area in self.area_data:
            primnet = self.area_data[area]['synthetic']
            sgeom = nx.get_edge_attributes(primnet,'geometry')
            synthgeom = {e:Link(sgeom[e]) for e in sgeom}
            synthpos = nx.get_node_attributes(primnet,'cord')
            
            actnet = self.area_data[area]['actual']
            ageom = self.area_data[area]['line_geom']
            actgeom = {e:Link(ageom[e]) for e in ageom}
            actpos = self.area_data[area]['bus_geom']
        
            for i in range(nY):
                for j in range(nX):
                    act_nodes = [n for n in actpos if xmin[j]<=actpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=actpos[n][1]<=ymax[nY-1-i]]
                    synth_nodes = [n for n in synthpos if xmin[j]<=synthpos[n][0]<=xmax[j] and\
                                  ymin[nY-1-i]<=synthpos[n][1]<=ymax[nY-1-i]]
                    act_edges = nx.edges(actnet,act_nodes)
                    synth_edges = nx.edges(primnet,synth_nodes)
                    A[i,j] += sum([actgeom[e].geod_length if e in actgeom \
                               else actgeom[(e[1],e[0])].geod_length \
                                   for e in act_edges])
                    B[i,j] += sum([synthgeom[e].geod_length if e in synthgeom \
                               else synthgeom[(e[1],e[0])].geod_length \
                                   for e in synth_edges])
        
        
            self.compare_networks(ax1,self.area_data[area]['df_lines'],
                             self.area_data[area]['df_buses'],
                             self.area_data[area]['synthetic'])
        
        for i in range(nX-1):
            ax1.plot([xmin[1+i]]*2,[BOTTOM,TOP],color='seagreen',linestyle='dotted')
        for i in range(nY-1):
            ax1.plot([LEFT,RIGHT],[ymin[1+i]]*2,color='seagreen',linestyle='dashed')
        
        # Statistics Figure
        colormap = cm.RdBu
        nan_color = 'white'
        colormap.set_bad(nan_color,1.)
        C = A-B
        for i in range(nY):
            for j in range(nX):
                if A[i,j]!=0:
                    C[i,j]=100.0*C[i,j]/A[i,j]
                else:
                    C[i,j]=np.nan
        C_masked = np.ma.array(C, mask=np.isnan(C))
        ax2.matshow(C_masked,cmap=colormap,vmin=-100,vmax=100)
        ax2.tick_params(top=False,left=False,labelleft=False,labeltop=False)
        ax2.set_title("Deviation in the length of edges",fontsize=14)
        cobj = cm.ScalarMappable(cmap=colormap)
        cobj.set_clim(vmin=-100,vmax=100)
        cbar = fig.colorbar(cobj,ax=ax2)
        cbar.set_label('Percentage Deviation',size=20)
        cbar.ax.tick_params(labelsize=20)
        
        # Save figure
        fig.savefig("{}{}.png".format(path,
                    'percent-length-comparison-'+str(nX)+'-'+str(nY)),
                    bbox_inches='tight')
        return
    
    



#%% Statistical comparisons



def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{0:.1f}".format(100*y)
    return s





def hop_density(graph_list,sub):
    """
    """
    # nodelabel = nx.get_node_attributes(graph,'label')
    for g in graph_list:
        h = [nx.shortest_path_length(g,n,sub) for n in list(g.nodes())]
        sns.distplot(h,hist=False,kde=True,kde_kws = {'shade': False, 'linewidth': 2})
    return


# def hop_dist(self,path,fname=None,name1='Synthetic Network',
    #          name2='Original Network'):
    #     """
    #     Creates the hop distribution of the networks. The synthetic network is compared
    #     with a base network. The hop distribution of both the networks is plotted 
    #     together in a stacked plot/histogram.
        
    #     Inputs: graph: synthetic network graph
    #             base: original network graph
    #             sub: substation ID
    #             path: path to save the plot
    #     """
    #     for area in self.area_data:
            
    #         synth = self.area_data[area]['synthetic']
    #         act = self.area_data[area]['actual']
    #         h1 = [nx.shortest_path_length(graph,n,sub) for n in list(graph.nodes())]
    #         w1 = np.ones_like(h1)/float(len(h1))
    #         h2 = [nx.shortest_path_length(base,n,111) for n in list(base.nodes())]
    #         w2 = np.ones_like(h2)/float(len(h2))
    #     hops = [h1,h2]
    #     w = [w1,w2]
    #     bins = range(0,80,2)
    #     colors = ['lightsalmon','turquoise']
    #     labels = [name1,name2]
    #     fig = plt.figure(figsize=(10,6))
    #     ax = fig.add_subplot(111)
    #     ax.hist(hops,bins=bins,weights=w,label=labels,color=colors)
    #     ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    #     ax.set_ylabel("Percentage of nodes",fontsize=20)
    #     ax.set_xlabel("Hops from root node",fontsize=20)
    #     ax.legend(prop={'size': 20})
    #     ax.tick_params(axis='both', labelsize=20)
    #     if fname == None:
    #         ax.set_title("Hop distribution comparison for network rooted at "+str(sub),
    #                      fontsize=20)
    #     elif fname=='compare':
    #         ax.set_title("Hop distribution for two generated synthetic networks",
    #                      fontsize=20)
        
    #     # Save the figure
    #     if fname==None:filename = str(sub)+'-hop-dist'
    #     else: filename=fname
    #     fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    #     return