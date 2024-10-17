# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:22:32 2024

@author: sarab
"""

import pickle
import copy
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def coord_list(G):
    """
    Collect the coordinates of all nodes in a graph, type dict().
    Supported types: FlexibleGraph, DiGraph.
    """
    coordList = {}
    for idx in G._node.keys():
        coordList[idx] = [G._node[idx]['x'], G._node[idx]['y']]
    return coordList


def plot_paths(G, path, pc4d_crop, type_graph, nodes, edges, od, pos_all, 
               flag_save):
    G = copy.deepcopy(G)
    
    node_o = path[0]
    node_d = path[-1]
    nodes_in_path = path
    path_draw = [(nodes_in_path[i], nodes_in_path[i+1]) 
                  for i in range(len(nodes_in_path)-1)]
    
    switching_edges = [edge for edge in edges
                       if G[edge[0]][edge[1]]['type'][0] == 's']
    
    nd_colors = [G._node[u]['color'] for u in nodes]
    edg_colors = [G[u][v]['color'] for u,v in edges]

    edg_width = [5 if edg in path_draw 
                  else 0 if edg in switching_edges 
                  else 0.1 for edg in edges]
    nd_size = [20 if node in  nodes_in_path else 0.1 for node in nodes]
    
    fig_g, ax_g = plt.subplots(figsize=(10, 10))
    pc4d_crop.boundary.plot(ax=ax_g, linewidth=0.2, color='k')
    nx.draw_networkx(G, 
                     ax=ax_g, 
                     pos=pos_all, 
                     with_labels=False, 
                     node_size=nd_size, 
                     width=edg_width, 
                     arrows=False, 
                     nodelist=nodes, 
                     node_color=nd_colors, 
                     edgelist=edges, 
                     edge_color=edg_colors)
    nx.draw_networkx_nodes(G, 
                           ax=ax_g, 
                           pos=pos_all, 
                           nodelist=[node_o], 
                           node_shape="o", 
                           node_size=50, 
                           node_color='k')
    nx.draw_networkx_nodes(G, 
                           ax=ax_g, 
                           pos=pos_all, 
                           nodelist=[node_d], 
                           node_shape="s", 
                           node_size=50, 
                           node_color='k')
    ax_g.set_axis_off()
    
    if flag_save:
        fp = os.path.join(os.getcwd(), f'figures_check_again\\od{od}_{type_graph}.jpg')
        plt.savefig(fp,
                    bbox_inches = "tight",
                    pad_inches = 0, 
                    transparent = True,
                    dpi = 100)
        plt.close()
    print('OD: ' + str(od))

os.chdir('..')

plt.rcParams.update({"text.usetex": True,'font.size': 25})
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

fp_save = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp_save, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, multiplier_low_income, 
     G_cbw, G_o, G_ocbw, pc4_info, G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, 
     full_demand, data_matlab, G_pt] = pickle.load(f)
    
# predefined variables maybe worth saving if not on .txt file outside
fp_save_vars = os.path.join(os.getcwd(), 'variables\\data_Eindhoven_vars.pkl')
with open(fp_save_vars, 'rb') as f:  
    [average_speed_car, average_speed_walk, bbox, crs, default_wait_pt, 
     dict_weird_nodes, flag_bike, flag_create_pt, flag_load, flag_parking, 
     max_nodes_pc, modes, modes_percentage, motives_to_remove, parking_time, 
     share_nodes, target_nodes, total_av_trips] = pickle.load(f)


color_blue = (0, 0.4470, 0.7410)
color_red = (0.8500, 0.3250, 0.0980)
color_green = (0.4660, 0.6740, 0.1880)
color_yellow = (0.9290, 0.6940, 0.1250)
color_purple  = (0.4940, 0.1840, 0.5560)
color_light_blue = (0.3010, 0.7450, 0.9330)
color_burgundy = (0.6350, 0.0780, 0.1840)

# #%%%
# nodes = list(data_matlab['nodes'][0])
# edges = [(data_matlab['edges'].loc[i,0], data_matlab['edges'].loc[i,1]) 
#           for i in range(len(data_matlab['edges']))]

# origin_nodes_ind = data_matlab['origin_nodes_ind']
# destination_nodes_ind = data_matlab['destination_nodes_ind']


# od_pairs = np.where(full_demand>0)
# demand_o_notin_origins = [ori for ori in od_pairs[0] 
#                           if ori not in origin_nodes_ind]
# demand_d_notin_origins = [dest for dest in od_pairs[1] 
#                           if dest not in destination_nodes_ind]

# print(demand_o_notin_origins, demand_d_notin_origins)

# short_path_dict = dict()
# type_graph = ['full','no_car']
# for k in range(2):
#     shortest_paths = []
#     if k == 0:
#         G = copy.deepcopy(G_ocbwptd)
#     if k == 1:
#         G = copy.deepcopy(G_obwptd)
#     for od in range(len(od_pairs[0])):
#         origin = nodes[od_pairs[0][od]]
#         destination = nodes[od_pairs[1][od]]
#         path = nx.shortest_path(G, source=origin, target=destination, 
#                                 weight='weight')
#         shortest_paths.append(path)
#         print('k: ' + str(k) + '. Origin ' + str(origin))
#     short_path_dict[type_graph[k]] = shortest_paths

# print(len(od_pairs[0]))
# print(len(short_path_dict['full']))
# print(len(short_path_dict['no_car']))

# mask_edges_nocar = data_matlab['mask_edges_nocar']
# mask_nodes_nocar = data_matlab['mask_nodes_nocar']

# edges_nocar = [edg for (edg, mask) in zip(edges, mask_edges_nocar) if mask]
# nodes_nocar = [node for (node, mask) in zip(nodes, mask_nodes_nocar) if mask]

# # Check type of edges used in path, which paths use bike and take longer than 20 mins?
# typ_nodes_paths = defaultdict(list)
# with_bike = defaultdict(list)
# times_paths = defaultdict(list)
# for k in range(2):
#     if k == 0: 
#         G = G_ocbwptd
#         edges_ = edges.copy()
#         nodes_ = nodes.copy()
#     if k == 1: 
#         G = G_obwptd
#         edges_ = edges_nocar.copy()
#         nodes_ = nodes_nocar.copy()
#     paths_list = short_path_dict[type_graph[k]]
#     for od in range(len(paths_list)):
#         nodes_in_path = paths_list[od]
#         path_draw = [(nodes_in_path[i], nodes_in_path[i+1]) 
#                       for i in range(len(nodes_in_path)-1)]
#         time = sum([G[u][v]['weight'] for u, v in path_draw])
#         typ_edges = [G[u][v]['type'] for u, v in path_draw]
#         times_paths[type_graph[k]].append(time)
#         if (('b' in typ_edges or 's-o-b' in typ_edges or 's-b-w' in typ_edges) 
#             and time > 20*60):
#             with_bike[type_graph[k]].append(od)


# # Check which paths from case1 (graph with cars) use bike by default and take 
# # more than 20 mins are also on case 2 with same characteristics
# print([od in with_bike[type_graph[1]] for od in with_bike[type_graph[0]]])


# # Plot paths with bike > 20 mins
# # for k in range(2):
# #     if k == 0: 
# #         G = G_ocbwptd
# #         edges_ = edges.copy()
# #         nodes_ = nodes.copy()
# #     if k == 1: 
# #         G = G_obwptd
# #         edges_ = edges_nocar.copy()
# #         nodes_ = nodes_nocar.copy()
# #     paths_list = short_path_dict[type_graph[k]]
# #     for od in with_bike[type_graph[k]]: #range(0, len(paths_list), 100):
# #         path = paths_list[od]
# #         plot_paths(G, path, pc4d_crop, type_graph[k], nodes_, edges_, od)

# # After checking plots: following od pairs are not problematic
# no_prob_nc = [7, 9, 11, 42, 88, 102, 103, 105, 154, 175, 177, 178, 181, 184, 
#             204, 208, 209, 211, 215, 230, 232, 268, 274, 307, 343, 345, 347, 
#             351, 368, 369, 427, 428, 429, 430, 495, 522, 523, 524, 525, 542, 
#             556, 602, 667, 668, 669, 673, 674, 675, 706, 730, 734, 801, 802, 
#             901, 909, 912, 914, 920, 923, 938, 1026, 1033, 1066, 1096, 1168, 
#             1183, 1228, 1258, 1260, 1266]
# no_prob_full = [521, 526, 527, 528, 603, 604, 606, 663, 665, 666]

# # Find ods w/ bike > 20 mins to be checked again
# check_bike_nc = [od for od in with_bike[type_graph[1]] 
#                   if od not in no_prob_nc]
# check_bike_full = [od for od in with_bike[type_graph[0]] 
#                     if od not in no_prob_full]

# # Create new graph w/out car&bike to compare od paths w/ pt only
# G_owptd = copy.deepcopy(G_obwptd)
# G_owptd.remove_nodes_from(list(G_b._node.keys()))


# # check = check_bike_nc
# # check.extend(check_bike_full)

# # Get shortest path for problematic ods on pt only graph
# G = G_owptd
# edges_ = list(G_owptd.edges)
# nodes_ = list(G_owptd._node.keys())
# type_g = 'no bike'
# prob_od_paths = []
# prob_od_times = []
# for od in check_bike_nc: 
#     origin = nodes[od_pairs[0][od]]
#     destination = nodes[od_pairs[1][od]]
#     path = nx.shortest_path(G, source=origin, target=destination, 
#                             weight='weight')
#     # plot_paths(G, path, pc4d_crop, type_g, nodes_, edges_, od)
#     prob_od_paths.append(path)
#     path_draw = [(path[i], path[i+1]) 
#                   for i in range(len(path)-1)]
#     time = sum([G[u][v]['weight'] for u, v in path_draw])
#     prob_od_times.append(time)


# # Compare times of bike & pt
# compare_t_ptb = []
# for ind, od in enumerate(check_bike_nc):
#     ind_bike = with_bike[type_graph[1]].index(od)
#     t_bike = times_paths[type_graph[1]][ind_bike]
#     t_pt = prob_od_times[ind]
#     compare_t_ptb.append(t_bike > t_pt)

# od_pt_faster = [check_bike_nc[ind] 
#                 for ind, value in enumerate(compare_t_ptb) 
#                 if value]


# # # Check and plot again
# # compare_t_ptb_dict = defaultdict(list)
# # for k in range(2):
# #     if k == 0: 
# #         G = G_obwptd
# #         type_g = 'no car'
# #     if k == 1: 
# #         G = G_owptd
# #         type_g = 'no bike'
# #     for od in od_pt_faster:
# #         origin = nodes[od_pairs[0][od]]
# #         destination = nodes[od_pairs[1][od]]
# #         path = nx.shortest_path(G, source=origin, target=destination, 
# #                                 weight='weight')
# #         path_draw = [(path[i], path[i+1]) 
# #                       for i in range(len(path)-1)]
# #         time = sum([G[u][v]['weight'] for u, v in path_draw])
# #         compare_t_ptb_dict[type_g].append(time)
# #         nodes_ = list(G._node.keys())
# #         edges_ = list(G.edges)
# #         pos_all = coord_list(G)
# #         plot_paths(G, path, pc4d_crop, type_g, nodes_, edges_, od, pos_all)

# # for ind, od in enumerate(od_pt_faster):
# #     t_b = round(compare_t_ptb_dict['no car'][ind]/60)
# #     t_pt = round(compare_t_ptb_dict['no bike'][ind]/60)
# #     print("od: {}, b: {:.2f}, pt: {:.2f}".format(od, round(t_b, 2),round(t_pt, 2)))



# # fp_save_checks = os.path.join(os.getcwd(), 'variables\\check_vars.pkl')
# # with open(fp_save_checks, 'wb') as f:  
# #     pickle.dump([od_pt_faster, G_owptd], f)

#%%%

nodes = list(data_matlab['nodes'][0])
edges = [(data_matlab['edges'].loc[i,0], data_matlab['edges'].loc[i,1]) 
          for i in range(len(data_matlab['edges']))]

# od_pairs = np.where(full_demand>0)

fp_save_checks = os.path.join(os.getcwd(), 'variables\\check_vars.pkl')
with open(fp_save_checks, 'rb') as f:  
    [od_pt_faster, G_owptd, od_pairs] = pickle.load(f)

# Check and plot again
compare_t_ptb_dict = defaultdict(list)
for k in range(2):
    if k == 0: 
        G = G_obwptd
        type_g = 'no car'
    if k == 1: 
        G = G_owptd
        type_g = 'no bike'
    for od in od_pt_faster:
        origin = nodes[od_pairs[0][od]]
        destination = nodes[od_pairs[1][od]]
        path = nx.shortest_path(G, source=origin, target=destination, 
                                weight='weight')
        path_draw = [(path[i], path[i+1]) 
                      for i in range(len(path)-1)]
        time = sum([G[u][v]['weight'] for u, v in path_draw])
        compare_t_ptb_dict[type_g].append(time)
        nodes_ = list(G._node.keys())
        edges_ = list(G.edges)
        pos_all = coord_list(G)
        plot_paths(G, path, pc4d_crop, type_g, nodes_, edges_, od, pos_all, 
                   False)

for ind, od in enumerate(od_pt_faster):
    t_b = round(compare_t_ptb_dict['no car'][ind]/60)
    t_pt = round(compare_t_ptb_dict['no bike'][ind]/60)
    print("od: {}, b: {:.2f}, pt: {:.2f}".format(od, round(t_b, 2),round(t_pt, 2)))


