# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:07:26 2020

Author: Rounak Meyur
"""

import networkx as nx
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
import seaborn as sns

def create_base(path,filename='hethwood',thresh=2):
    """
    """
    df_network = pd.read_csv(path+filename+'.csv')
        
    # Networkx graph construction
    root = nx.from_pandas_edgelist(df_network,'node_a','node_b')
    comps = [c for c in list(nx.connected_components(root)) if len(c)>thresh]
    graph = nx.Graph()
    for c in comps: graph=nx.compose(graph,root.subgraph(c))
    
    try:
        print("Number of cycles:",len(nx.find_cycle(graph)))
    except:
        print("No cycles found!!!")
    return graph


def degree_dist(graph,base,sub,path,fname=None):
    """
    Creates the degree distribution of the networks. The synthetic network is compared
    with a base network. The degree distribution of both the networks is plotted 
    together in a stacked plot.
    
    Inputs: graph: synthetic network graph
            base: original network graph
            sub: substation ID
            path: path to save the plot
    """
    degree_sequence_a = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    na = graph.number_of_nodes()
    degree_sequence_b = sorted([d for n, d in base.degree()], reverse=True)  # degree sequence
    nb = base.number_of_nodes()
    
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
    if fname == None:
        ax.set_title("Degree distribution comparison for network rooted at "+str(sub),
                     fontsize=20)
    else:
        ax.set_title("Degree distribution for "+str(sub)+" with maximum flow:"+fname.split('-')[3],
                     fontsize=20)
    
    # Save the figure
    if fname == None:filename = str(sub)+'-degree-dist'
    else: filename = fname
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    print("Kolmogorov Smimnov test result:",stats.ks_2samp(degree_sequence_a,
                                                           degree_sequence_b))
    return


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{0:.1f}".format(100*y)
    return s


def hop_dist(graph,base,sub,path,fname=None,name1='Synthetic Network',
             name2='Original Network'):
    """
    Creates the hop distribution of the networks. The synthetic network is compared
    with a base network. The hop distribution of both the networks is plotted 
    together in a stacked plot/histogram.
    
    Inputs: graph: synthetic network graph
            base: original network graph
            sub: substation ID
            path: path to save the plot
    """
    # nodelabel = nx.get_node_attributes(graph,'label')
    h1 = [nx.shortest_path_length(graph,n,sub) for n in list(graph.nodes())]
    w1 = np.ones_like(h1)/float(len(h1))
    h2 = [nx.shortest_path_length(base,n,sub) for n in list(base.nodes())]
    w2 = np.ones_like(h2)/float(len(h2))
    hops = [h1,h2]
    w = [w1,w2]
    bins = range(0,80,2)
    colors = ['lightsalmon','turquoise']
    labels = [name1,name2]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.hist(hops,bins=bins,weights=w,label=labels,color=colors)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_ylabel("Percentage of nodes",fontsize=20)
    ax.set_xlabel("Hops from root node",fontsize=20)
    ax.legend(prop={'size': 20})
    ax.tick_params(axis='both', labelsize=20)
    if fname == None:
        ax.set_title("Hop distribution comparison for network rooted at "+str(sub),
                     fontsize=20)
    elif fname=='compare':
        ax.set_title("Hop distribution for two generated synthetic networks",
                     fontsize=20)
    else:
        ax.set_title("Hop distribution for "+str(sub)+" with maximum flow:"+fname.split('-')[3],
                     fontsize=20)
    
    # Save the figure
    if fname==None:filename = str(sub)+'-hop-dist'
    else: filename=fname
    fig.savefig("{}{}.png".format(path,filename),bbox_inches='tight')
    return  


def hop_density(graph_list,sub):
    """
    """
    # nodelabel = nx.get_node_attributes(graph,'label')
    for g in graph_list:
        h = [nx.shortest_path_length(g,n,sub) for n in list(g.nodes())]
        sns.distplot(h,hist=False,kde=True,kde_kws = {'shade': False, 'linewidth': 2})
    return

