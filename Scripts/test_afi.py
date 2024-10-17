# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:23:55 2024

@author: sarab
"""

import matplotlib.pyplot as plt
import networkx as nx

# Tests AFI

def plot_network(G, pc4d_crop, typ = False): #TODO: maybe define typ as an iterative to plot multiple layers
    pos = coord_list(G)
    nodes = list(G._node)
    edges = list(G.edges)
    nd_colors = [G._node[u]['color'] for u in nodes]
    edg_colors = [G[u][v]['color'] for u,v in edges]
    if typ:
        nd_size = [0.5 if G._node[u]['type'] == typ else 0 for u in nodes]
        edg_width = [1 if G[u][v]['type'] == typ else 0 for u,v in edges]
    else:
        nd_size = 3
        edg_width = 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    pc4d_crop.boundary.plot(ax=ax, linewidth=0.5, color='black',zorder=1)
    nx.draw_networkx(G, 
                     ax=ax, 
                     pos=pos, 
                     with_labels=False, 
                     node_size=nd_size, 
                     width=edg_width, 
                     arrows=False, 
                     nodelist=nodes, 
                     node_color=nd_colors, 
                     edgelist=edges, 
                     edge_color=edg_colors)
    ax.set_axis_off()
    return ax


def coord_list(G):
    """
    Collect the coordinates of all nodes in a graph, type dict().
    Supported types: FlexibleGraph, DiGraph.
    """
    coordList = {}
    for idx in G._node.keys():
        coordList[idx] = [G._node[idx]['x'], G._node[idx]['y']]
    return coordList


plot_network(G_o, pc4d_crop)


