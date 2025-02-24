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
        
        # plt.savefig(fp + "\\fig\\" + type_od + "_" + str(i) + title + ".jpg",
        # # plt.savefig(fp + "\\fig\\" + str(i) + "_" + title + ".svg",
        # # plt.savefig(fp + "\\fig2\\" + str(i) + "_" + title + ".pdf",
        # plt.savefig(fp + "\\fig\\" + type_od + "_" + str(i) + title + ".svg",
    plt.savefig(fp +"\\" + title + '.' + form,#".svg",
                bbox_inches = "tight",
                pad_inches = 0, 
                transparent = True,
                format = form, #'svg', 
                dpi = 1200)
        # plt.close()

    

def calculate_afi(orig_per_pc, D, avg_eps, path_eps, weird_nodes, pc_info, 
                  col_name, nr):
            # (orig_per_pc, D, acc_maxAcc, acc_mintt, acc_pathAcc, 
            #  eps_maxAcc, eps_mintt, eps_pathAcc,
            #  T_max,D_total,dict_weird_nodes,new_postcodes_4d):
    
    afi_pc = {}
    for pc, origins in orig_per_pc.items():
        afi_acum = {'avg': 0,
                    'path': 0}
        # afi_acum = {'mintt': 0, 
        #             'eps_mintt': 0,}
        for origin in origins:
            i = nodes.index(origin)
            # Find demand columns nonzero for this origin
            cols = np.where(D[i,:]<0)[0]
            d = -1 * D[i,cols]
            avg_ori = [avg_eps[col][0] for col in cols]
            path_ori = [path_eps[col][0] for col in cols]
            # acc_mintt_ = [acc_mintt[col][0] for col in cols]
            # eps_mintt_ = [eps_mintt[col][0] for col in cols]
            
            afi_acum['avg'] = afi_acum['avg'] + sum(d*avg_ori)
            afi_acum['path'] = afi_acum['path'] + sum(d*path_ori)
            # afi_acum['mintt'] = afi_acum['mintt'] + sum(d*acc_mintt_)
            # afi_acum['eps_mintt'] = afi_acum['eps_mintt'] + sum(d*eps_mintt_)
        
        alpha = nr[pc]
        afi_pc[pc] = {'avg': afi_acum['avg']/alpha,
                      'path': afi_acum['path']/alpha,}
                      # 'mintt': afi_acum['mintt']/afi_acum['alpha'],
                      # 'eps_mintt': afi_acum['eps_mintt']/afi_acum['alpha'],}
    
    for pc1, pc2 in weird_nodes.items():
        afi_pc[pc1] = afi_pc[pc2]
    
    col_avg = col_name + '_avg'
    col_path = col_name + '_path'
    pc_info[col_avg] = np.nan
    pc_info[col_path] = np.nan
    
    # new_postcodes_4d['mintt'] = np.nan
    # new_postcodes_4d['eps_mintt'] = np.nan

    pc_info = pc_info.reset_index(drop=True)
    
    for pc in afi_pc.keys():
        ind = pc_info.loc[pc_info['postcode4'] == pc].index[0]
        pc_info.loc[ind, col_avg] = afi_pc[pc]['avg']
        pc_info.loc[ind, col_path] = afi_pc[pc]['path']
        # ind = new_postcodes_4d.loc[new_postcodes_4d['postcode4'] == pc].index[0]
        # new_postcodes_4d.loc[ind, 'mintt'] = afi_pc[pc]['mintt']
        # new_postcodes_4d.loc[ind, 'eps_mintt'] = afi_pc[pc]['eps_mintt']
    
    pc_info = geopandas.GeoDataFrame(pc_info, crs="EPSG:4326")
    
    min_v = min(min(pc_info[col_avg]),min(pc_info[col_path]))
    max_v = max(max(pc_info[col_avg]),max(pc_info[col_path]))
    
    return pc_info, min_v, max_v

def plot_afi_heatmap(pc_info_df, col, leg_label, min_value, max_value, color_map,
                     fig_name, folder, form, dpi, close = False):
    
    leg = {"label": leg_label}
    
    fig_, ax_ = plt.subplots(figsize=(20, 20))
    ax_._xmargin = 0
    ax_._ymargin = 0
    divider = make_axes_locatable(ax_)
    divider.set_vertical([0.9*Size.AxesY(ax_)])
    cax = divider.append_axes("right", size="4%", pad=-4)
    ax_.set_axis_off()
    pc_info_df.plot(ax=ax_, 
                    linewidth = 1, 
                    column=col,
                    legend=True,
                    cax = cax,
                    legend_kwds=leg,
                    vmin = min_value,
                    vmax = max_value,
                    zorder = 1,
                    cmap=color_map)
    pc_info_df.boundary.plot(ax=ax_, 
                             color='w', 
                             zorder=2, 
                             linewidth=0.2);
    plt.savefig(folder + fig_name + '.' + form,#"ur_TT_paths_heatmap.pdf", #"ur_tt_paths_heatmap.svg", #
                bbox_inches = "tight",
                pad_inches = 0, 
                transparent = True,
                format = form, 
                dpi = dpi)
    if close:
        plt.close()
    
def calculate_AFI_MILP(pc_info, col_name, Nmin, delta_n, weird_nodes):
    
    col = col_name + Nmin
    
    # AFI MILP
    pc_info[col] = np.nan
    afi_milp = {str(pc[0]): delta_n[i][0] for i, pc in enumerate(pc_order)}
    for pc1, pc2 in weird_nodes.items():
        afi_milp[pc1] = afi_milp[pc2]
    for pc in afi_milp.keys():
        ind = pc_info.loc[pc_info['postcode4'] == pc].index[0]
        pc_info.loc[ind, col] = afi_milp[pc]
    
    min_v = min(pc_info[col])
    max_v = max(pc_info[col])
    
    return pc_info, min_v, max_v

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
form = 'svg'
# form = 'pdf'
dpi = 500

min_lst = []
max_lst = []

pc_info = pc4d_join
nr = pc4_info['population']

folder_ = os.path.join(os.getcwd(), 'output\\figures\\')

Tmax = '20'

load_m = loadmat(fp + '\\matlab_python_T' + Tmax + '_S.mat')

edges_matlab = load_m['edgesMatlab']
demand = load_m['demand']
D_total = sum(sum(demand))

edges = [(nodes[edge[0]-1],nodes[edge[1]-1]) 
          for edge in edges_matlab]

ur_path_maxAcc = load_m['ur_avgAccS_path']

ur_avg_maxAcc = load_m['ur_avgAccS_avg']

pc_unique = load_m['pc_unique']

ur = {str(v[0]): {'avgAcc_avg': ur_avg_maxAcc[i][0],
                  'avgAcc_path': ur_path_maxAcc[i][0],}
      for i, v in enumerate(pc_unique)}

for pc1, pc2 in dict_weird_nodes.items():
    ur[pc1] = ur[pc2]

pc_info['avgAcc_avg'] = np.nan
pc_info['avgAcc_path'] = np.nan


pc_info = pc_info.reset_index(drop=True)

for pc in ur.keys():
    ind = pc_info.loc[pc_info['postcode4'] == pc].index[0]
    pc_info.loc[ind, 'avgAcc_avg'] = ur[pc]['avgAcc_avg']
    pc_info.loc[ind, 'avgAcc_path'] = ur[pc]['avgAcc_path']

pc_info = geopandas.GeoDataFrame(pc_info, crs="EPSG:4326")

min_v = min(min(ur_path_maxAcc),min(ur_avg_maxAcc))
max_v = max(max(ur_path_maxAcc),max(ur_avg_maxAcc))

folder = os.path.join(folder_, 'sq_')

leg_label = "$u_{r}\\ [\\mathrm{min}^2]$"

# max Acc avg
col_name = 'avgAcc_avg'
plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                  'ur_AccS_OD_heatmap', folder, form, dpi)

# max Acc path
col_name = 'avgAcc_path'
plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                  'ur_AccS_paths_heatmap', folder, form, dpi)

