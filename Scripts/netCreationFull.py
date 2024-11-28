# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:47:51 2024

@author: sarab
"""


import pickle
import networkx as nx
# import FlexibleGraph6 as fg
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import osmnx as ox
import geopy.distance
from collections import defaultdict
import random
import copy
import os
import matplotlib.pyplot as plt
import shapely.geometry as geo
import scipy.io
import numpy as np
from PTnetwork import create_pt_network
import zipfile

#%%% Functions

def remov(G):
    to_remove = []
    for node in G._node.keys():
        if not any(G._node[node]):
            to_remove.append(node)
    for node in to_remove:
        G.remove_node(node)
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self loops
    return G

def random_color_generator():
    r = random.uniform(0, 0.8)
    g = random.uniform(0, 0.8)
    b = random.uniform(0, 0.8)
    return (r, g, b)

def remove_negative_weights(G):
    G = copy.deepcopy(G)
    negative_weight = []
    
    for edge in G.edges:
        weight = G[edge[0]][edge[1]]['weight']
        if weight < 0:
            if weight < 0.00001:
                G[edge[0]][edge[1]]['weight'] = 0
            else:
                negative_weight.append(edge)
    # Hopefully no more negative weights. Otherwise, set to 0 
    if len(negative_weight) > 0:
        for edge in negative_weight: G[edge[0]][edge[1]]['weight'] = 0
    return G

def weights(G, mode):
    G = copy.deepcopy(G)
    edgeList = list(G.edges)
    
    if mode == 'w':
        speed = 4.5
    elif mode == 'b':
        speed = 20
    if mode == 'w' or mode == 'b':
        for edge in edgeList:
            s, t = edge
            length = G[s][t]["length"]
            G[s][t]["weight"] = ( (length/1000) / speed ) * 3600
    else:
        for edge in edgeList:
            s, t = edge
            G[s][t]["weight"] = G[s][t]['travel_time']
    return G

def get_networks(bbox):
    H_b = ox.graph_from_bbox(bbox = bbox, network_type = 'bike')
    H_w = ox.graph_from_bbox(bbox = bbox, network_type = 'walk')
    H_c = ox.graph_from_bbox(bbox = bbox, network_type = 'drive')
    
    H_c = ox.speed.add_edge_speeds(H_c)
    H_c = ox.speed.add_edge_travel_times(H_c)
    
    # Graphs on networkx
    G_b = nx.Graph(H_b)
    G_w = nx.Graph(H_w)
    G_c = nx.Graph(H_c)
    
    # Add weights to graphs
    G_b = weights(G_b, 'b')
    G_w = weights(G_w, 'w')
    G_c = weights(G_c, 'c')
    
    return G_b, G_w, G_c

def get_pc_data(crs, box_gpd):
    pc4_path = os.path.join(os.getcwd(), 
                            'variables\\2023-CBS_pc4_2022_v1\\cbs_pc4_2022_v1.gpkg')
    postcodes_4d = gpd.read_file(pc4_path, bbox=box_gpd)
    postcodes_4d = postcodes_4d.to_crs(crs)
    postcodes_4d_crop = gpd.clip(postcodes_4d.geometry,box_gpd)
    
    pc_geo_join = pd.DataFrame(postcodes_4d_crop)
    pc_geo_join = pc_geo_join.join(postcodes_4d['postcode4'], 
                                   rsuffix='_', sort=False, 
                                   validate=None)
    
    fp_2019 = os.path.join(os.getcwd(), 
                           'variables\\2023-cbs_pc4_2019_vol\\cbs_pc4_2019_vol.gpkg')
    Eind_2019 = gpd.read_file(fp_2019, bbox=box_gpd)
    
    multiplier_low_income = {pc: Eind_2019.loc[Eind_2019['postcode4']==pc, 
                                                'percentage_laag_inkomen_huishouden']
                              for pc in Eind_2019['postcode4']}
    
    return postcodes_4d_crop, pc_geo_join, postcodes_4d, multiplier_low_income 

def set_type_color_attributes(G, typ, color):
    G = copy.deepcopy(G)
    new_names = dict()
    
    for key in G._node.keys():
        new_names[key] = str(key) + typ
    
    nx.relabel_nodes(G, new_names, copy=False)
    
    nx.set_edge_attributes(G, color, 'color')
    nx.set_edge_attributes(G, typ, 'type')
    nx.set_node_attributes(G, color, 'color')
    nx.set_node_attributes(G, typ, 'type')
    
    return G

def graph_copy(G, typ, color, flag_nodes = True):
    G = copy.deepcopy(G)
    
    if flag_nodes: 
        G_new = nx.empty_graph(create_using=nx.DiGraph)
        nodes = defaultdict(dict)
    else: G_new = G
    
    eq_names_nodes = dict()
    for key, item in G._node.items():
        name = str(key[:-1]) + typ
        eq_names_nodes[key] = name
        if flag_nodes:
            nodes[name]['x'] = item['x']
            nodes[name]['y'] = item['y']
    
    if flag_nodes: 
        G_new.add_nodes_from([(k, v) for k, v in nodes.items()])
    else: 
        nx.relabel_nodes(G_new, eq_names_nodes, copy=False)
    
    nx.set_node_attributes(G_new, color, 'color')
    nx.set_node_attributes(G_new, typ, 'type')
    
    return G_new, eq_names_nodes

def get_parking_graph(box_gpd, bbox, typ, color):
    tags_parkingLots = {'amenity' : 'parking'}
    parkingLots = ox.features.features_from_bbox(bbox=bbox, tags=tags_parkingLots)
    parkingLots_ = parkingLots.to_crs(crs)
    parkingLots_ = gpd.clip(parkingLots_.geometry, box_gpd)
    
    G = nx.empty_graph(create_using=nx.DiGraph)
    parking_nodes = defaultdict(dict)
    
    for ind, item in parkingLots_.items():
        parking_nodes[str(ind[1]) + typ]['x'] = item.centroid.bounds[0]
        parking_nodes[str(ind[1]) + typ]['y'] = item.centroid.bounds[1]
        parking_nodes[str(ind[1]) + typ]['type'] = typ
        parking_nodes[str(ind[1]) + typ]['color'] = color
        if pd.isna(parkingLots.loc[ind].capacity):
            parking_nodes[str(ind[1]) + typ]['capacity'] = 1
        else:
            parking_nodes[str(ind[1]) + typ]['capacity'] = int(parkingLots.loc[ind].capacity)
    
    G.add_nodes_from([(k, v) for k, v in parking_nodes.items()])
    
    return G

def get_nodes_under_d(G, typ1, typ2, max_dist, flag_one):
    G = copy.deepcopy(G)
    
    nodes_typ1 = {k: v 
                  for k, v in G._node.items() if v['type'] == typ1}
    geopos = pd.DataFrame()
    
    geopos['node'] = nodes_typ1.keys()
    geopos['x'] = [v['x'] for v in nodes_typ1.values()]
    geopos['y'] = [v['y'] for v in nodes_typ1.values()]
    geopos_2 = [(k, v['x'], v['y']) for k, v in G._node.items() if v['type'] == typ2]
    
    nodes_to_connect = defaultdict(dict)
    for x in geopos_2:
        node2 = x[0]
        point = (x[1], x[2])
        below_d_nodes = get_near_nodes(geopos, point, max_dist)
        if len(below_d_nodes) != 0:
            if flag_one:
                i = below_d_nodes['dist'].idxmin()
                node1 = below_d_nodes.loc[i,'node']
                nodes_to_connect[(node1, node2)]['dist'] = below_d_nodes.loc[i,'dist']
            else:
                for i, row in below_d_nodes.iterrows():
                    node1 = row['node']
                    nodes_to_connect[(node1, node2)]['dist'] = row['dist'] 
    
    return nodes_to_connect

def get_near_nodes(df_pos, point, max_dist):
    df_pos['ref_x'] = point[0]
    df_pos['ref_y'] = point[1]
    df_pos['dist'] = [geopy.distance.geodesic((row['y'],row['x']), 
                                              (row['ref_y'],row['ref_x'])).m
                      for i, row in df_pos.iterrows()]
    mask = (df_pos['dist'] < max_dist)
    nearest_nodes = df_pos[mask]
    
    return nearest_nodes

def merge_parking(G, typ, max_dist):
    G = copy.deepcopy(G)
    
    distances = get_nodes_under_d(G, typ, typ, max_dist, False)
    nodes_to_merge = defaultdict(list)
    
    for node1, node2 in distances.keys():
        if node1 != node2:
            nodes_to_merge[node1].append(node2)
    
    #Check nodes that are closer to each other, to merge first
    priority = pd.DataFrame(columns=['node','distance'])
    for node_p1, list_p2 in nodes_to_merge.items():
        total_dist = 0
        for node_p2 in list_p2:
            total_dist += distances[(node_p1, node_p2)]['dist']
        priority.loc[len(priority)] = [node_p1,total_dist]
    
    priority.sort_values(by='distance', inplace=True)
    
    resulting_p = defaultdict(dict)
    merged = list()
    merging = defaultdict(list)
    for p1 in priority.node:
        # Check if node to merge has not been merged already
        if p1 not in merged:
            #List of candidates to merge with node p1
            p2_to_merge = nodes_to_merge[p1] 
            for p2 in p2_to_merge:
                # Check if node has been merged, if not, add to merged and merging with p1
                if p2 not in merged:
                    merged.append(p2)
                    merging[p1].append(p2)
            if len(merging[p1]) != 0:
                # Initialize x, y positiion and capacity with p1 info
                x_sum =  G.nodes[p1]['x']
                y_sum =  G.nodes[p1]['y']
                capacity_sum = G.nodes[p1]['capacity'] 
                for p2 in merging[p1]:
                    x_sum += G.nodes[p2]['x']
                    y_sum += G.nodes[p2]['y']
                    capacity_sum += G.nodes[p2]['capacity']
                # Assign x, y as average position of all merged nodes
                resulting_p[p1]['x'] = x_sum/(len(merging[p1])+1)
                resulting_p[p1]['y'] = y_sum/(len(merging[p1])+1)
                resulting_p[p1]['type'] = typ
                resulting_p[p1]['color'] = 'purple'
                # Assign capacity as sum of each node capacity
                resulting_p[p1]['capacity'] = capacity_sum
    
    for p1, list_merge in merging.items():
        if len(list_merge) != 0:
            for p2 in list_merge:
                nx.contracted_nodes(G, p1, p2, self_loops=False, copy=False)
            nx.set_node_attributes(G, {p1:resulting_p[p1]})
    
    return G

def connect_layers(G, typ1, typ2, average_speed, max_dist, color, 
                   extra_weight = 0, default_weight = 0, flag_2_ways = False,
                   flag_pt_boarding = False, flag_one = True):
    G = copy.deepcopy(G)
    
    type_arc = 's-' + typ1 + '-' + typ2
    distances = get_nodes_under_d(G, typ1, typ2, max_dist, flag_one)
    nodes_to_connect = defaultdict(dict)
    for node1, node2 in distances.keys():
        if node1 != node2:
            if default_weight:
                weight = default_weight
            else:
                weight = distances[(node1, node2)]['dist']/average_speed_walk 
            if flag_pt_boarding:
                extra_weight = G._node[node2]['boarding_cost']
            nodes_to_connect[(node1,node2)]['weight'] = weight + extra_weight
            nodes_to_connect[(node1,node2)]['type'] = type_arc
            nodes_to_connect[(node1,node2)]['color'] = color
            if flag_2_ways:
                type_arc = 's-' + typ2 + '-' + typ1
                nodes_to_connect[(node2,node1)]['weight'] = weight
                nodes_to_connect[(node2,node1)]['type'] = type_arc
                nodes_to_connect[(node2,node1)]['color'] = color
    
    G.add_edges_from(nodes_to_connect.keys())
    nx.set_edge_attributes(G, nodes_to_connect)
    
    nodes_dict = {n1: n2 for n1, n2 in nodes_to_connect.keys()}
    return G, nodes_dict

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

def trips_motives(pc4_unique, motives_to_remove): #TODO: check this properly
    #Motives albatross
    motives_path = os.path.join(os.getcwd(), 
                    'variables\\20230112_EhvMetro_Fabio_sched_singleDay-gen.zip')
    archive = zipfile.ZipFile(motives_path, 'r')
    motives_file = archive.open('20230112_EhvMetro_Fabio_sched_singleDay-gen.txt')
    # motives_path = os.path.join(os.getcwd(), 
    #                'variables\\20230112_EhvMetro_Fabio_sched_singleDay-gen.txt')
    motives = pd.read_csv(motives_file)
    # motives = pd.read_csv(motives_path)
    mask = [motive not in motives_to_remove for motive in motives['ActivityType']]
    filtered_motives = motives.loc[mask, ['ActivityType','OrigLoc','DestLoc']]
    
    mask1 = [str(pc) in pc4_unique for pc in filtered_motives['OrigLoc']]
    mask2 = [str(pc) in pc4_unique for pc in filtered_motives['DestLoc']]
    mask = mask1 and mask2
    filtered_motives_pc = filtered_motives.loc[mask]
    
    ActivityType = ['Business','Groceries','Leisure','Services','Social','Touring','Work']
    Eq = ['Work','Groceries','Leisure','Services','Leisure','Leisure','Work']
    data = {'ActivityType': ActivityType, 'ResumedActivity': Eq}
    to_join = pd.DataFrame(data=data)
    
    motives_final = pd.merge(filtered_motives_pc, to_join, how="left")
    
    summary_pcs = motives_final[['OrigLoc','DestLoc']].value_counts()
    
    share_trips_all = np.zeros((len(pc4_unique), len(pc4_unique)))
    for i in range(len(pc4_unique)):
        pc_origin = int(pc4_unique[i])
        for j in range(len(pc4_unique)):
            pc_destination = int(pc4_unique[j])
            if i != j:
                try: 
                    share_trips_all[i][j] = summary_pcs[pc_origin][pc_destination]
                except:
                      share_trips_all[i][j] = 0
    
    trips_per_origin = np.zeros((len(pc4_unique), len(pc4_unique)))
    for i in range(len(pc4_unique)):
        all_trips_ori = sum(share_trips_all[i])
        if all_trips_ori == 0:
            trips_per_origin[i] = 0
        else:
            trips_per_origin[i] = share_trips_all[i] / all_trips_ori
    
    return trips_per_origin

def connect_copied_layers(G, typ1, typ2, eq_names_nodes, default_weight = 0):
    G = copy.deepcopy(G)
    
    arctype = 's-' + typ1 + '-' + typ2
    connect = {(out_edge[:-1] + typ1, in_edge[:-1] + typ2): 
                {'weight': default_weight, 
                'type': arctype, 
                'color': 'k'}
                for in_edge, out_edge in eq_names_nodes.items()}
    
    G.add_edges_from(connect.keys())
    nx.set_edge_attributes(G, connect)
    
    return G
    
    # return G, nodes_to_connect

def pc_attribute(G, pc4d_crop, weird_nodes):
    G = copy.deepcopy(G)
    
    pc4d_crop = pc4d_crop.copy()
    nodes = defaultdict(dict)
    for key, item in G._node.items():
        # Get geo position of node
        point = geo.Point(item['x'], item['y'])
        # Find pc4 to which the node belongs to
        pc4d_crop['in'] = [True if point.within(polygon) else False 
                            for polygon in pc4d_crop['geometry']] #pandas mask
        ind_pc = pc4d_crop['in'][pc4d_crop['in'] == True].index[0]
        pc = pc4d_crop.loc[ind_pc]['postcode4']
        if pc in weird_nodes.keys(): pc = weird_nodes[pc]
        nodes[key]['postcode'] = str(pc)
    
    nx.set_node_attributes(G, nodes)
    
    return G

def create_origin_layer(Gw,
                        Gb, 
                        pc4d_crop, 
                        color, 
                        weird_nodes, 
                        full_G, 
                        flag_bike,
                        typ_from = 'w', 
                        typ_res = 'o',
                        typ_extra = 'b'):
    Gw = copy.deepcopy(Gw)
    Gb = copy.deepcopy(Gb)
    pc4d_crop = pc4d_crop.copy()
    
    #Create graph with nodes
    G_new, eq_names = graph_copy(Gw, typ_res, color)
    G_new = pc_attribute(G_new, pc4d_crop, weird_nodes)
    
    # List of origin nodes
    origin_nodes = list(G_new._node)
    # List of pc associated to origin nodes as nodes order from graph
    pc4_nodes = [G_new._node[node]['postcode'] for node in origin_nodes]
    # List of unique pc
    pc4_unique = list(set(pc4_nodes))
    # List of number of nodes belonging to each pc as in pc4_unique order
    num_nodes_per_pc4 = [pc4_nodes.count(x) for x in pc4_unique]
    
    pc4_info = {'pc_nodes': pc4_nodes,
                'unique': pc4_unique,
                'num_nodes_pc': num_nodes_per_pc4,
                'nodes': origin_nodes}
    
    G_new, population_pc4 = population(G_new, pc4_info)
    
    pc4_info['population'] = population_pc4
    
    full_G = nx.union(full_G, G_new)
    full_G = connect_copied_layers(full_G, typ_res, typ_from, eq_names)
    default_weight = 1*60
    if flag_bike:
        full_G = connect_copied_layers(full_G, typ_res, typ_extra, eq_names, 
                                       default_weight = default_weight)
    else:
        arctype = 's-' + typ_extra + '-' + typ_from
        edg_bw = {edg: v for edg, v in full_G.edges.items() 
                  if v['type'] == arctype}
        arctype_res = 's-' + typ_res + '-' + typ_extra
        connect = {(out_edge[:-1] + typ_res, in_edge): 
                   {'weight': default_weight, 
                    'type': arctype_res, 
                    'color': 'k'}
                   for in_edge, out_edge in edg_bw.keys()}
        full_G.add_edges_from(connect.keys())
        nx.set_edge_attributes(full_G, connect)
        
    
    return G_new, full_G, pc4_info, eq_names

def bike_graph(flag_bike, 
               G_b, 
               G_w, 
               color, 
               factor_wb,
               typ_res = 'b', 
               typ_from = 'w'):
    G_b = copy.deepcopy(G_b)
    G_w = copy.deepcopy(G_w)
    
    if flag_bike:
        G_b, eq_nodes_wb = graph_copy(G_w, typ_res, color, flag_nodes = False)
        default_weight_bw = 60
    else:
        max_dist_bw = 100
        default_weight_bw = 0
    
    weight = defaultdict(dict)
    for edge in G_b.edges:
        weight[edge]['weight'] = G_b[edge[0]][edge[1]]['weight'] / factor_wb
    
    nx.set_edge_attributes(G_b, weight)
    nx.set_edge_attributes(G_b, color, 'color')
    nx.set_edge_attributes(G_b, typ_res, 'type')
    
    G_b = G_b.to_directed()
    G_cbw = nx.union(G_b, G_cw)
    
    if flag_bike:
        G_cbw = connect_copied_layers(G_cbw, 
                                      typ_res, 
                                      typ_from, 
                                      eq_nodes_wb, 
                                      default_weight = default_weight_bw)
    else:
        G_cbw, _ = connect_layers(G_cbw, 
                                  typ_res, 
                                  typ_from, 
                                  average_speed_walk, 
                                  max_dist_bw, 
                                  'k', 
                                  default_weight = default_weight_bw)
    
    return G_b, G_cbw

def public_transport_layer(flag_create_pt, 
                           default_wait_pt, 
                           G_full, 
                           average_speed_walk, 
                           typ1 = 'pt',
                           typ2 = 'w'):
    G_full = copy.deepcopy(G_full)
    
    if flag_create_pt:
        # Take between 6 to 9 in the morning
        start = 6*60*60
        end = 9*60*60
        create_pt_network(os.getcwd(),start,end,box)
    
    pt_path = os.path.join(os.getcwd(), 'variables\\pt_data.pkl')
    with open(pt_path, 'rb') as f:
        G = pickle.load(f)
    
    G.remove_nodes_from(list(nx.isolates(G)))
    # Change attribute of length (default from creation) to weight
    att_edg = {(u, v): {'weight': G[u][v]['length']} for u, v in G.edges}
    nx.set_edge_attributes(G, att_edg)
    G = set_type_color_attributes(G, typ1, color_purple)
    
    if default_wait_pt:
        flag_pt_boarding = False
        extra_weight = default_wait_pt
    else: 
        flag_pt_boarding = True
        extra_weight = 0
    
    G_full = nx.union(G_full, G)
    G_full, _ = connect_layers(G_full, 
                               typ2, 
                               typ1, 
                               average_speed_walk, 
                               500, 
                               color_green, 
                               extra_weight = extra_weight,
                               flag_2_ways = True, 
                               flag_pt_boarding = flag_pt_boarding, 
                               flag_one = False)
    
    return G, G_full

def connect_car(flag_parking, bbox, parking_time, 
                G_full, color, dict_sow, default_wait_pt):
    
    G_full = copy.deepcopy(G_full)
    
    if flag_parking:
        G_p = get_parking_graph(box_gpd, bbox, 'p', color)
        G_p = merge_parking(G_p, 'p', 500)
        G_full = nx.union(G_full, G_p)
        # Car to parking
        G_full, _ = connect_layers(G_full, 
                                   'c', 
                                   'p', 
                                   average_speed_car, 
                                   1000, 
                                   'k',
                                   extra_weight = parking_time)
        # Parking to walking
        G_full, dict_spw = connect_layers(G_full, 
                                          'p', 
                                          'w', 
                                          average_speed_walk, 
                                          1000,
                                          'k')
        # TODO: check here
        G_full = connect_copied_layers(G_full, 
                                       'o', 
                                       'p', 
                                       dict_spw, 
                                       default_weight = 3*60)
    else:
        # Car to walking and walking to car
        G_full, _ = connect_layers(G_full, 
                                   'c', 
                                   'w', 
                                   average_speed_walk, 
                                   1000, 
                                   'k', 
                                   default_weight = 3*60, 
                                   flag_2_ways = True)
        # Car to pt and pt to car
        G_full, _ = connect_layers(G_full, 
                                   'c', 
                                   'pt', 
                                   average_speed_walk, 
                                   300, 
                                   'k', 
                                   extra_weight = default_wait_pt, 
                                   flag_2_ways = True)
    return G_full

def population(G, pc4_info):
    G = copy.deepcopy(G)
    pc4_unique = pc4_info['unique']
    
    population_pc4 = {}
    for pc in pc4_unique: 
        pop = pc4d_data.loc[pc4d_data['postcode4'] == pc, 
                            'aantal_inwoners'].iloc[0]
        population_pc4[pc] = pop
    
    origin_nodes = pc4_info['nodes']
    att_pop_origins = {node: {'population': 
                              population_pc4[G._node[node]['postcode']]}
                        for node in origin_nodes}
    
    nx.set_node_attributes(G, att_pop_origins)
    
    return G, population_pc4

def create_destinations_layer(G, G_full, color, dict_sow, typ_res = 'd'):
    G = copy.deepcopy(G)
    G_full = copy.deepcopy(G_full)
    
    # Create destination layer from G (origins)
    G_new, eq_names = graph_copy(G, typ_res, color, flag_nodes = False)
    G_full = nx.union(G_full, G_new)
    G_full = connect_copied_layers(G_full, 'w', 'd', dict_sow)
    
    return G_new, G_full 

def create_demand_matrix(G, 
                         pc4_info, 
                         share_nodes, 
                         motives_to_remove, 
                         total_av_trips):
    #Create demand matrix between origins and destinations
    G = copy.deepcopy(G)
    
    pc_list = pc4_info['unique']
    num_nodes_pc = pc4_info['num_nodes_pc']
    pc_origin = pc4_info['pc_nodes']
    population_pc4 = pc4_info['population']
    origins = pc4_info['nodes']
    destinations = [node[:-1] + 'd' for node in origins]
    nodes = list(G._node.keys())
    
    num_nodes_pc4 = {pc: max(1, min(round(num_nodes_pc[ind]*share_nodes),max_nodes_pc)) 
                                  for ind, pc in enumerate(pc_list)}
    
    orig_per_pc = {}
    dest_per_pc = {}
    for pc in pc_list:
        orig_per_pc[pc] = [node for i, node in enumerate(origins) if pc_origin[i] == pc]
        dest_per_pc[pc] = [node for i, node in enumerate(destinations) if pc_origin[i] == pc]
    
    trips_per_origin = trips_motives(pc_list, motives_to_remove)
    
    full_demand = np.zeros((len(nodes), len(nodes)))
    # For each origin node and its pc related
    for ind_opc, pc_ori in enumerate(pc_list):
        # According to travel data: pc4 destination from origin pc (indices according to pc_list)
        possible_dest_pc = np.nonzero(trips_per_origin[ind_opc,:])[0]
        # For each index of possible pc destinations
        for ind_dpc in possible_dest_pc:
            pc_dest = pc_list[ind_dpc]
            # List of nodes on origin pc
            possible_nodes_orig = orig_per_pc[pc_ori]
            # Choose random node from destinations
            random_orig = random.sample(possible_nodes_orig, num_nodes_pc4[pc_ori])
            # List of nodes on destination pc
            possible_nodes_dest = dest_per_pc[pc_dest]
            # Choose random node from destinations
            random_dest = random.sample(possible_nodes_dest, num_nodes_pc4[pc_dest])
            for orig_node in random_orig:
                ind_origin = nodes.index(orig_node)
                for dest_node in random_dest:
                    ind_destination = nodes.index(dest_node)
                    population_origin = population_pc4[pc_ori] / len(random_orig)
                    trips_origin = population_origin * total_av_trips
                    share_trips_pc_o_d = trips_per_origin[ind_opc][ind_dpc]
                    demand = trips_origin * share_trips_pc_o_d / len(random_dest)
                    full_demand[ind_origin][ind_destination] = demand
    
    return full_demand

def save_matlab(G, pc4_info, G_no_car, full_demand):
    origin_pc = pc4_info['pc_nodes']
    origins = pc4_info['nodes']
    destinations = [node[:-1] + 'd' for node in origins]
    
    nodes = list(G._node.keys())
    nodes_pd = pd.DataFrame(nodes)
    nodes_nocar = list(G_no_car._node.keys())
    
    edges = list(G.edges)
    edges_pd = pd.DataFrame(edges)
    edges_nocar = list(G_no_car.edges)
    
    mask_nodes_nocar = [(node in nodes_nocar) for node in nodes]
    mask_edges_nocar = [(edge in edges_nocar) for edge in edges]
    
    times = [G[n1][n2]['weight'] for n1, n2 in edges]
    
    incidence = nx.incidence_matrix(G, 
                                    nodelist=nodes, 
                                    edgelist=edges, 
                                    oriented=True)
    adjacency = nx.adjacency_matrix(G, nodelist=nodes)

    origin_nodes_ind = [nodes.index(node) 
                        for node in nodes if node in origins]
    destination_nodes_ind = [nodes.index(node) 
                             for node in nodes if node in destinations]
    
    positions = pd.DataFrame([[G._node[node]['x'], G._node[node]['y']] 
                              for node in nodes])
    nodes_type = [G._node[node]['type'] for node in nodes]
    edges_type = [G[n1][n2]['type'] for n1, n2 in edges]
    
    data = {"inc": incidence,
            "adj": adjacency,
            "nodes": nodes_pd,
            "edges": edges_pd,
            "times": times,
            # "costs": G1_costs,
            "origin_nodes_ind": origin_nodes_ind,
            "destination_nodes_ind": destination_nodes_ind,
            "origin_pc": origin_pc,
            "positions": positions,
            "nodes_type": nodes_type,
            "edges_type": edges_type,
            "mask_nodes_nocar": mask_nodes_nocar,
            "mask_edges_nocar": mask_edges_nocar,
            "full_demand": full_demand}
    
    save_mat = os.path.join(os.getcwd(), 
                            'variables\\matlab_data_Eindhoven.mat')
    scipy.io.savemat(save_mat,data)
    
    with open(save_mat, 'wb') as f:  
        pickle.dump(data, f)
    
    return data

def save_python(G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
                multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
                G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, data_matlab, 
                G_pt, average_speed_car, average_speed_walk, bbox, crs, 
                default_wait_pt, dict_weird_nodes, flag_bike, flag_create_pt, 
                flag_load, flag_parking, max_nodes_pc, modes, 
                modes_percentage, motives_to_remove, parking_time, 
                share_nodes, target_nodes, total_av_trips):
    
    fp_save = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
    with open(fp_save, 'wb') as f:  
        pickle.dump([G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
                     multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
                     G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, 
                     data_matlab, G_pt], f)
        
    # predefined variables maybe worth saving if not on .txt file outside
    fp_save_vars = os.path.join(os.getcwd(), 'variables\\data_Eindhoven_vars.pkl')
    with open(fp_save_vars, 'wb') as f:  
        pickle.dump([average_speed_car, average_speed_walk, bbox, crs, 
                     default_wait_pt, dict_weird_nodes, flag_bike, 
                     flag_create_pt, flag_load, flag_parking, max_nodes_pc, 
                     modes, modes_percentage, motives_to_remove, parking_time, 
                     share_nodes, target_nodes, total_av_trips], f)

def no_car_graph(flag_parking, G, G_c, G_p=None):
    G = copy.deepcopy(G)
    to_remove = list(G_c._node)
    if flag_parking:
        park_nodes = [n for n, v in G._node.items() if v['type']=='p']
        to_remove.extend(park_nodes)
    G.remove_nodes_from(to_remove)
    
    return G

def _0_weight(G):
    G = copy.deepcopy(G)
    for edg in G.edges: 
        if G[edg[0]][edg[1]]['weight'] == 0:
            G[edg[0]][edg[1]]['weight'] = 0.001
    return G

#%%% Variables definition
# distances in meters 
# times in seconds
# weights in time units (seconds)

# TODO: define these variables on a .txt? so they can be modified without accessing the code

os.chdir('..')

share_nodes = 0.25
max_nodes_pc = 1
average_speed_walk = 5 / 3.6 
average_speed_car = 25 / 3.6 
parking_time = 36 
target_nodes = 20
factor_wb = 3

# TODO: color object (maybe initialize from other file)
# TODO: color_yellow for walking switchs

# class color:
color_blue = (0, 0.4470, 0.7410)
color_red = (0.8500, 0.3250, 0.0980)
color_green = (0.4660, 0.6740, 0.1880)
color_yellow = (0.9290, 0.6940, 0.1250)
color_purple  = (0.4940, 0.1840, 0.5560)
color_light_blue = (0.3010, 0.7450, 0.9330)
color_burgundy = (0.6350, 0.0780, 0.1840)


north, east, south, west = 51.497082, 5.518988, 51.404946, 5.354779 
bbox = (north, south, east, west)

crs = "EPSG:4326"
box = Polygon([[west, south],
                [west, north],
                [east, north],
                [east, south]])
box_gpd = gpd.GeoDataFrame(pd.DataFrame(['p1'], columns = ['geom']),
                            crs = crs,
                            geometry = [box])

flag_bike = True # True to have bike graph as walking graph copy
flag_parking = False # True to have a parking graph
flag_create_pt = False # True to create from scratch PT graph
default_wait_pt = 4*60
flag_load = True # True to load B, W, C from previous execution

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
                    '5583': '5644'} 

motives_to_remove = ['Home', 'NonGroc', 'BringGet', 'Other']

modes = {'total':              2.75, 
         'car (driver)':       1.06, 
         'car (passenger)':    0.26, 
         'Train':              0.04, 
         'Bus/tram/metro':     0.03, 
         'Bike':               0.70, 
         'Walking':            0.56, 
         'Other':              0.11}

modes_percentage = {'car': ((modes['car (driver)'] + modes['car (passenger)']) 
                            * 100 / modes['total']),
                    'bike': modes['Bike'] * 100 / modes['total'],
                    'PT-walking': ((modes['Train'] + modes['Bus/tram/metro'] + 
                                    modes['Walking'] + modes['Other']) 
                                    * 100 / modes['total'])}

total_av_trips = modes['total']


#%%% Execute functions

#TODO: Check if its better to create all of the layers and then connect

graphs_path = os.path.join(os.getcwd(), 
                           f'variables\\graphs_shrink{target_nodes}.pkl')
if flag_load:
    # Get graphs saved from previous execution
    with open(graphs_path, 'rb') as f:
       [G_w, G_b, G_c] = pickle.load(f)
else:
    # Get graphs from OSM
    G_b, G_w, G_c = get_networks(bbox)
    
    # Shrink graphs
    shrink_graphs(target_nodes, G_b, G_w, G_c) # Reduce to 20 nodes
    
    # Remove nodes with no coordinate info and self loop edges
    G_b = remov(G_b)
    G_w = remov(G_w)
    G_c = remov(G_c)
    with open(graphs_path, 'wb') as f:
        pickle.dump([G_w, G_b, G_c], f)

# Postcodes
pc4d_crop, pc4d_join, pc4d_data, multiplier_low_income = get_pc_data(crs, box_gpd)

# Assign attributes graph
G_b = set_type_color_attributes(G_b, 'b', color_red)
G_w = set_type_color_attributes(G_w, 'w', color_yellow)
G_c = set_type_color_attributes(G_c, 'c', color_blue)

G_w = G_w.to_directed()
G_b = G_b.to_directed()
G_c = G_c.to_directed()

G_cw = nx.union(G_c, G_w)

#Bike connections
G_b, G_cbw = bike_graph(flag_bike, G_b, G_w, color_red, factor_wb)

# Function for creating origins and connecting to walking
G_o, G_ocbw, pc4_info, dict_sow = create_origin_layer(G_w, 
                                                      G_b, 
                                                      pc4d_join, 
                                                      color_burgundy, 
                                                      dict_weird_nodes, 
                                                      G_cbw,
                                                      flag_bike,)

# Create or load pt and connect w to pt and pt to w
G_pt, G_ocbwpt = public_transport_layer(flag_create_pt, 
                                        default_wait_pt, 
                                        G_ocbw, 
                                        average_speed_walk)

G_ocbwpt = connect_car(flag_parking, 
                       bbox, 
                       parking_time, 
                       G_ocbwpt, 
                       color_light_blue, 
                       dict_sow, 
                       default_wait_pt)

G_d, G_ocbwptd = create_destinations_layer(G_o, G_ocbwpt, color_burgundy, dict_sow)

full_demand = create_demand_matrix(G_ocbwptd, 
                                   pc4_info, 
                                   share_nodes, 
                                   motives_to_remove, 
                                   total_av_trips)

G_obwptd = no_car_graph(flag_parking, G_ocbwptd, G_c)

G_ocbwptd = _0_weight(G_ocbwptd)
G_obwptd = _0_weight(G_obwptd)

data_matlab = save_matlab(G_ocbwptd, pc4_info, G_obwptd, full_demand)

save_python(G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
            multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, G_ocbwpt, 
            G_d, G_ocbwptd, G_obwptd, full_demand, data_matlab, G_pt, 
            average_speed_car, average_speed_walk, bbox, crs, 
            default_wait_pt, dict_weird_nodes, flag_bike, flag_create_pt, 
            flag_load, flag_parking, max_nodes_pc, modes, modes_percentage, 
            motives_to_remove, parking_time, share_nodes, target_nodes, 
            total_av_trips)






