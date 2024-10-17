# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:57:08 2024

@author: sarab
"""

from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D
import os

def GetCoordList(graph):
    """
    Collect the coordinates of all nodes in a graph, type dict().
    Supported types: FlexibleGraph, DiGraph.
    """
    # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
    
    coordList = {}
    for idx in graph._node.keys():
        coordList[idx] = [graph._node[idx]['x'], graph._node[idx]['y']]
    return coordList

def plot_paths(paths,D,edges_matlab,title,nodes,edges,ods,G,type_od,fp,form,net,ax):#,start=0):
    # plot_paths(paths_mintt,D,edges_matlab,'path Min TT',nodes,edges,j,'worsen')!
    color_blue = (0, 0.4470, 0.7410)
    color_red = (0.8500, 0.3250, 0.0980)
    color_yellow = (0.9290, 0.6940, 0.1250)
    color_purple  = (0.4940, 0.1840, 0.5560)
    
    pos_all = GetCoordList(G)
    
    nd_colors = [G._node[u]['color'] 
                 for u in nodes]
    
    switching_edges = [edge for edge in edges
                        if G[edge[0]][edge[1]]['type'][0] == 's']
    
    edg_colors = [G[u][v]['color'] for u,v in edges]
    
    edg_width = [0.1 if edg in switching_edges #0.01
                 # else 0.2 for edg in edges]
                 else 0.7 for edg in edges] #0.5
    
    

    # fig_g, ax_g = plt.subplots(figsize=(10, 10))
    pc4d_crop.boundary.plot(ax=ax, linewidth=1, color='gray',zorder=1)
    if net:
        nx.draw_networkx(G, 
                          ax=ax, 
                          pos=pos_all, 
                          with_labels=False, 
                          node_size=5,#1, 
                          width=edg_width, 
                          alpha=0.6, #0.4
                          arrows=False, 
                          nodelist=nodes, 
                          node_color=nd_colors, 
                          edgelist=edges, 
                          edge_color=edg_colors)
    for i in ods:#range(start,D.shape[1],100):
        i = i - 1
        
        node_o = nodes[np.where(D[:,i]<0)[0][0]]
        node_d = nodes[np.where(D[:,i]>0)[0][0]]
        edges_in_path =  np.nonzero(paths[:,i])[0]
        path_draw = [edges[ind_edge]
                      for ind_edge in edges_in_path]
        
        edg_width_path = [4.5 if edg in path_draw 
                          else 0 for edg in edges]
        # nd_size_path = [50 if node in node == node_o 
        #                 else 50 if node == node_d
        #                 else 0 for node in nodes]
        nd_colors = [G._node[node]['color']
                     for node in nodes]
        # nd_shape = ["s" if node == node_d 
        #              else "o" for node in nodes]
        
        nx.draw_networkx(G, 
                         ax=ax, 
                         pos=pos_all, 
                         with_labels=False, 
                         node_size=0, 
                         width=edg_width_path, 
                         arrows=False, 
                         nodelist=nodes, 
                         node_color=nd_colors, 
                         edgelist=edges, 
                         edge_color=edg_colors)
        nx.draw_networkx_nodes(G,ax=ax,pos=pos_all,nodelist=[node_o],
                               node_shape="o",node_size=50,node_color='k')
        nx.draw_networkx_nodes(G,ax=ax,pos=pos_all,nodelist=[node_d],
                               node_shape="s",node_size=50,node_color='k')
    
    mode_color = [mpatches.Patch(color=color_blue, label='$\\textrm{Car}$'),
                  mpatches.Patch(color=color_red, label='$\\textrm{Bike}$'),
                  mpatches.Patch(color=color_yellow, label='$\\textrm{Walk}$'),
                  mpatches.Patch(color=color_purple, label='$\\textrm{PT}$'),
                  Line2D([0],[0],label='$\\textrm{Origin}$',marker="o", 
                          markeredgecolor='k',markersize=10, markerfacecolor='k',linestyle=''),
                  Line2D([0],[0],label='$\\textrm{Destination}$',marker="s", 
                          markeredgecolor='k',markersize=10, markerfacecolor='k',linestyle='')]
    plt.legend(handles=mode_color,fontsize=25,loc=(0.08,0.58))#'upper left') 
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # title_str = '$' + title + ' i=' + str(i) + '$'
    # ax_g.set_title(title_str, fontsize=20)
    # ax_g.legend()
    ax.set_axis_off()
    plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

    # plt.savefig(fp +"\\" + title + '.' + form,#".svg",
    #             bbox_inches = "tight",
    #             pad_inches = 0, 
    #             transparent = True,
    #             format = form, #'svg', 
    #             dpi = 1200)
        # plt.close()

    


os.chdir('..')

fp = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
     multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
     G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, 
     data_matlab, G_pt] = pickle.load(f)

color_pink = (204/255, 121/255, 167/255)

fp_matlab = os.path.join(os.getcwd(), 'model')

D_matlab = loadmat(fp_matlab + '\\data_g.mat')
D = D_matlab['D']

nodes = list(G_ocbwptd._node.keys())

orig_per_pc = {}
for pc in pc4_info['unique']:
    orig_per_pc[pc] = [node for node in G_o._node 
                       if G_o._node[node]['postcode'] == pc]

    
plt.rcParams.update({"text.usetex": True,'font.size': 25})
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

orig_map=plt.colormaps['RdBu'] 
reversed_map = orig_map.reversed() 

fp = os.path.join(os.getcwd(), 'output')
# form = 'jpg'
form = 'png'
# form = 'svg'
# form = 'pdf'
dpi = 500

folder_ = os.path.join(os.getcwd(), 'output\\paper\\figures\\')


Tmax = '20'

load_m = loadmat(fp + '\\matlab_python_T' + Tmax + '.mat')

edges_matlab = load_m['edgesMatlab']
demand = load_m['demand']
D_total = sum(sum(demand))

edges = [(nodes[edge[0]-1],nodes[edge[1]-1]) 
          for edge in edges_matlab]


paths_mintt = load_m['X_mintt']
paths_maxAcc = load_m['X_maxAcc']
paths_pathAcc = load_m['X_pathAcc']

fp = folder_

fig, (ax1, ax2) = plt.subplots(1, 2)

plot_paths(paths_mintt,D,edges_matlab,'Min TT',nodes,edges,[438,1085],G_ocbwptd,'improve',fp,form,0,ax1)
# plot_paths(paths_maxAcc,D,edges_matlab,'Max Acc',nodes,edges,[438,1085],G_ocbwptd,'improve',fp,form,0)
plot_paths(paths_pathAcc,D,edges_matlab,'Path Acc',nodes,edges,[438,1085],G_ocbwptd,'improve',fp,form,0,ax2)

# plot_paths(paths_maxAcc,D,edges_matlab,'network',nodes,edges,[],G_ocbwptd,'improve',fp,form,1)

plt.savefig(fp +"\\resource_comparison" + form,#".svg",
            bbox_inches = "tight",
            pad_inches = 0, 
            transparent = True,
            format = form, #'svg', 
            dpi = 1200)