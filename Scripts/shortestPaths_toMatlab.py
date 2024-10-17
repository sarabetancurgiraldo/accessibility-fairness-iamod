# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:49:28 2024

@author: sarab
"""

import os
import pickle
import scipy.io
import numpy as np
from collections import defaultdict
# import copy
import networkx as nx


def save_paths(data):
    fp_save = os.path.join(os.getcwd(), 'variables\\od_paths.pkl')
    with open(fp_save, 'wb') as f:  
        pickle.dump([data], f)
    save_mat = os.path.join(os.getcwd(), 
                            'variables\\matlab_data_od_paths.mat')
    scipy.io.savemat(save_mat,data)


os.chdir('..')

fp_open = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp_open, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, multiplier_low_income, 
     G_cbw, G_o, G_ocbw, pc4_info, G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, 
     full_demand, data_matlab, G_pt] = pickle.load(f)
   


#%%% Shortest paths

nodes = list(data_matlab['nodes'][0])
edges = [(data_matlab['edges'].loc[i,0], data_matlab['edges'].loc[i,1]) 
         for i in range(len(data_matlab['edges']))]

origin_nodes_ind = data_matlab['origin_nodes_ind']
origins = [nodes[i] for i in origin_nodes_ind]
origins_from_graph = list(G_o._node.keys())

destination_nodes_ind = data_matlab['destination_nodes_ind']
destinations = [nodes[i] for i in destination_nodes_ind]
destinations_from_graph = list(G_d._node.keys())

od_pairs = np.where(full_demand>0)

short_path_dict = defaultdict(list)
type_graph = ['full','no_car']#, 'no_bike']

# G_owptd = copy.deepcopy(G_obwptd)
# G_owptd.remove_nodes_from(list(G_b._node.keys()))

for od in range(len(od_pairs[0])): 
    for k in range(len(type_graph)):
        if k == 0:
            G = G_ocbwptd
        if k == 1:
            G = G_obwptd
        # if k == 2:
        #     G = G_owptd
        origin = nodes[od_pairs[0][od]]
        destination = nodes[od_pairs[1][od]]
        path = nx.shortest_path(G, source=origin, target=destination, 
                                weight='weight')
        short_path_dict[type_graph[k]].append(path)
        path_draw = [(path[i], path[i+1]) 
                      for i in range(len(path)-1)]
        time = sum([G[u][v]['weight'] for u, v in path_draw])
    print(od)


data = {"od_pairs": od_pairs, 
        "paths_dict": short_path_dict}

save_paths(data)




