# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:32:32 2024

@author: sarab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:30 2024

@author: sarab
"""

from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size#, SubplotDivider
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

def plot_paths(paths,D,edges_matlab,title,nodes,edges,ods,G,type_od,fp,form,net):#,start=0):
    # plot_paths(paths_mintt,D,edges_matlab,'path Min TT',nodes,edges,j,'worsen')!
    color_blue = (0, 0.4470, 0.7410)
    color_red = (0.8500, 0.3250, 0.0980)
    color_green = (0.4660, 0.6740, 0.1880)
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
    
    

    fig_g, ax_g = plt.subplots(figsize=(10, 10))
    pc4d_crop.boundary.plot(ax=ax_g, linewidth=1, color='gray',zorder=1)
    if net:
        nx.draw_networkx(G, 
                          ax=ax_g, 
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
        
        # remove almost zero flow on arcs
        flow_od = paths[:,i]
        demand_od = D[np.where(D[:,i]>0)[0][0],i]
        eps_od_sl = demand_od/100
        flow_od_ = np.where(flow_od < eps_od_sl, 0, flow_od)
                
        # edges_in_path =  np.nonzero(flow_od_)[0]
        # edg_path = [edges[i] for i in edges_in_path]
        # path_draw = [edges[ind_edge]
        #               for ind_edge in edges_in_path]
        
        # edg_width_path = [flow_od_[i] * 5/demand_od if edg in path_draw 
        #                   else 0 for i, edg in enumerate(edges)]
        
        edg_width_path = [flow_od_[j] * 10/demand_od for j, edg in enumerate(edges)]
        
        nd_colors = [G._node[node]['color']
                     for node in nodes]
        
        nx.draw_networkx(G, 
                         ax=ax_g, 
                         pos=pos_all, 
                         with_labels=False, 
                         node_size=0, 
                         width=edg_width_path, 
                         arrows=False, 
                         nodelist=nodes, 
                         node_color=nd_colors, 
                         edgelist=edges, 
                         edge_color=edg_colors)
        nx.draw_networkx_nodes(G,ax=ax_g,pos=pos_all,nodelist=[node_o],
                               node_shape="o",node_size=50,node_color='k')
        nx.draw_networkx_nodes(G,ax=ax_g,pos=pos_all,nodelist=[node_d],
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
    ax_g.set_axis_off()
    plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
    
    plt.savefig(fp +"\\" + title + '.' + form,#".svg",
                bbox_inches = "tight",
                pad_inches = 0, 
                transparent = True,
                format = form, #'svg', 
                dpi = 100)
    # plt.close()

    


os.chdir('..')

fp = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
     multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
     G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, 
     data_matlab, G_pt] = pickle.load(f)

# # predefined variables maybe worth saving if not on .txt file outside
# fp_save_vars = os.path.join(os.getcwd(), 'variables\\data_Eindhoven_vars.pkl')
# with open(fp_save_vars, 'wb') as f:  
#     pickle.dump([average_speed_car, average_speed_walk, bbox, crs, 
#                  default_wait_pt, dict_weird_nodes, flag_bike, 
#                  flag_create_pt, flag_load, flag_parking, max_nodes_pc, 
#                  modes, modes_percentage, motives_to_remove, parking_time, 
#                  share_nodes, target_nodes, total_av_trips], f)

color_pink = (204/255, 121/255, 167/255)

fp_matlab = os.path.join(os.getcwd(), 'model')

D_matlab = loadmat(fp_matlab + '\\data_g.mat')
D = D_matlab['D']
PC_matlab = loadmat(fp_matlab + '\\data_shortPaths.mat')
pc_order = PC_matlab['pc_unique']

nodes = list(G_ocbwptd._node.keys())

orig_per_pc = {}
for pc in pc4_info['unique']:
    orig_per_pc[pc] = [node for node in G_o._node 
                       if G_o._node[node]['postcode'] == pc]

dict_weird_nodes = {'5647': '5646',
                    '5582': '5644',
                    '5674': '5632',
                    '5656': '5654',
                    '5658': '5657', 
                    '5617': '5616', 
                    '5684': '5688', 
                    '5685': '5688', 
                    '5681': '5683', 
                    '5692': '5629', 
                    '5691': '5633', 
                    '5513': '5657', 
                    '5614': '5613', 
                    '5511': '5505', 
                    '5581': '5656', 
                    '5583': '5644',
                    '5645': '5646'} 
    
plt.rcParams.update({"text.usetex": True,'font.size': 25})
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

# orig_map=plt.cm.get_cmap('RdBu')
orig_map=plt.colormaps['RdBu'] 
reversed_map = orig_map.reversed() 

fp = os.path.join(os.getcwd(), 'output')
# form = 'png'
# form = 'svg'
form = 'pdf'
dpi = 1000

min_lst = []
max_lst = []

pc_info = pc4d_join
nr = pc4_info['population']

Tmax = '20'

# nCarRange = [1e3, 3e3, 4e3]
nCarRange = [3e3]

for nCar in nCarRange:
    
    str_folder = 'output\\nCar\\' + str(int(nCar)) + '\\figures\\ods reg'
    
    folder_plots = os.path.join(os.getcwd(), str_folder)
    
    str_folder = 'output\\nCar\\' + str(int(nCar))
    
    fp_load = os.path.join(os.getcwd(), str_folder)
    
    load_m = loadmat(fp_load + '\\matlab_python_T' + Tmax + '.mat')

    edges_matlab = load_m['edgesMatlab']
    demand = load_m['demand']
    D_total = sum(sum(demand))

    edges = [(nodes[edge[0]-1],nodes[edge[1]-1]) 
              for edge in edges_matlab]

    paths_mintt = load_m['X_mintt']
    # paths_maxAcc = load_m['X_maxAcc']
    paths_maxAcc = load_m['X_maxAcc_reg']
    paths_pathAcc = load_m['X_pathAcc']

    fp = folder_plots
    
    color_yellow = (0.9290, 0.6940, 0.1250)
    
    G_plot = G_ocbwptd.copy()
    
    switch_edg = [(ed1,ed2) for ed1, ed2 in G_plot.edges.keys() if G_plot[ed1][ed2]['type'][0]=='s']
    
    for ed1, ed2 in switch_edg:
        G_plot[ed1][ed2]['color'] = color_yellow
    
    # ods_bc = [19,176,212,218,278,337,365,386,388,448,460,462,465,470,487,595,
    #           711,715,827,831,1050,1055,1068,1073,1117,1215]
    # ods_cb = [271,321,361,1018]
    
    # for od in ods_bc:#range(demand.shape[1]):
    #     plot_paths(paths_mintt,D,edges_matlab,'bc_'+str(od)+'_Min TT',nodes,edges,[od],G_plot,'improve',fp,form,0)
    #     plot_paths(paths_maxAcc,D,edges_matlab,'bc_'+str(od)+'_Max Acc_reg',nodes,edges,[od],G_plot,'improve',fp,form,0)
    #     plot_paths(paths_pathAcc,D,edges_matlab,'bc_'+str(od)+'_Path Acc',nodes,edges,[od],G_plot,'improve',fp,form,0)

    # for od in ods_cb:#range(demand.shape[1]):
    #     plot_paths(paths_mintt,D,edges_matlab,'cb_'+str(od)+'_Min TT',nodes,edges,[od],G_plot,'improve',fp,form,0)
    #     plot_paths(paths_maxAcc,D,edges_matlab,'cb_'+str(od)+'_Max Acc_reg',nodes,edges,[od],G_plot,'improve',fp,form,0)
    #     plot_paths(paths_pathAcc,D,edges_matlab,'cb_'+str(od)+'_Path Acc',nodes,edges,[od],G_plot,'improve',fp,form,0)
    
    # plot_paths(paths_mintt,D,edges_matlab,'Min TT',nodes,edges,[361,1215],G_plot,'improve',fp,form,0)
    # plot_paths(paths_maxAcc,D,edges_matlab,'Max Acc_reg',nodes,edges,[361,1215],G_plot,'improve',fp,form,0)
    # plot_paths(paths_pathAcc,D,edges_matlab,'Path Acc',nodes,edges,[361,1215],G_plot,'improve',fp,form,0)
    plot_paths(paths_maxAcc,D,edges_matlab,'Max Acc_1215',nodes,edges,[1215],G_plot,'improve',fp,form,0)

