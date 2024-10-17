# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:55:22 2024

@author: sarab
"""

import peartree as pt
import partridge as ptg
# import matplotlib.pyplot as plt
import pickle
import networkx as nx
import pandas as pd
# from shapely.geometry import Polygon
import geopandas as gpd
from collections import defaultdict
import random
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

def random_color_generator():
    r = random.uniform(0, 0.8)
    g = random.uniform(0, 0.8)
    b = random.uniform(0, 0.8)
    return (r, g, b)

def create_pt_network(fp,start,end,box):
    
    path = os.path.join(fp, 'variables\gtfs-nl.zip')
    
    service_ids = ptg.read_busiest_date(path)[1]
    view = {
        'agency.txt': {'agency_id': 'BRAVO:CXX'},
        'trips.txt': {'service_id': service_ids}
    }
    print('Loading feed')
    feed = ptg.load_feed(path, view)
    
    print('Feed loaded')
    
    trips = feed.trips
    stops = feed.stops
    stop_times = feed.stop_times
    
    print('Constructing graph')
    G = pt.load_feed_as_graph(feed, start, end, connection_threshold=0)
    
    crs = "EPSG:4326"
    
    nodes_pd = pd.DataFrame.from_dict(G._node,orient="index")
    
    nodes_gdf = gpd.GeoDataFrame(nodes_pd,
                                 geometry=gpd.points_from_xy(nodes_pd.x, nodes_pd.y),
                                 crs=crs)
    
    
    # north, east, south, west = 51.497082, 5.518988, 51.404946, 5.354779 
    
    # box = Polygon([[west, south],
    #                [west, north],
    #                [east, north],
    #                [east, south]])
    
    box_gpd = gpd.GeoDataFrame(pd.DataFrame(['p1'], columns = ['geom']),
                               crs = crs,
                               geometry = [box])
    
    nodes_gdf_inbbox = gpd.clip(nodes_gdf.geometry,box_gpd)
    
    nodes_in_bbox_list = list(nodes_gdf_inbbox.index)
    
    nodes_to_remove = [x for x in G.nodes if x not in nodes_in_bbox_list]
    
    G.remove_nodes_from(nodes_to_remove)
    
    nodes_inbbox_info = nodes_gdf.loc[nodes_in_bbox_list]
    nodes_inbbox_info['node_ref'] = nodes_inbbox_info.index
    
    data_stops = stops.merge(nodes_inbbox_info, how='left', 
                              left_on=['stop_lat', 'stop_lon'], right_on=['y', 'x'], 
                              suffixes=('_x', '_y'))
    
    data_stops = data_stops.loc[data_stops['boarding_cost'].notna()]
    
    stop_times_inbbox = stop_times[stop_times['stop_id'].isin(list(data_stops['stop_id']))]
    
    to_join_routeid = pd.DataFrame(data=trips[['route_id','trip_id']],
                                    columns=['route_id','trip_id'], index=range(len(trips)))
    stop_times_inbbox_routeid = stop_times_inbbox.merge(to_join_routeid, how='left',
                                                        on='trip_id')
    
    stop_trip_route = pd.DataFrame(stop_times_inbbox_routeid[['route_id','trip_id','stop_id']])
    
    unique_routes = list(stop_trip_route['route_id'].unique())
    colors_routes = defaultdict()
    
    for i in unique_routes:
        flag_color = True
        while flag_color:
            color = random_color_generator()
            if color not in colors_routes.values(): flag_color = False 
        colors_routes[i] = color
    
    
    edges_pt = list(G.edges())
    # weird_edges = []
    edg_route_att = defaultdict(dict)
    
    for node_in, node_out in edges_pt:
        stop_in = data_stops.loc[data_stops['node_ref'] == node_in,'stop_id']#.item()
        stop_out = data_stops.loc[data_stops['node_ref'] == node_out,'stop_id']#.item()
        if len(stop_in) > 1:
            routes_nodein = []
            for stop in stop_in:
                routes_ = stop_trip_route.loc[stop_trip_route['stop_id'] == stop, 'route_id']
                routes_nodein.extend(routes_)
        else:
            stop_in = stop_in.item()
            routes_nodein = stop_trip_route.loc[stop_trip_route['stop_id'] == stop_in, 'route_id']
        if len(stop_out) > 1:
            routes_nodeout = []
            for stop in stop_out:
                routes_ = stop_trip_route.loc[stop_trip_route['stop_id'] == stop, 'route_id']
                routes_nodeout.extend(routes_)
        else:
            stop_out = stop_out.item()
            routes_nodeout = stop_trip_route.loc[stop_trip_route['stop_id'] == stop_out, 'route_id']
        route = [route for route in list(routes_nodein) if route in list(routes_nodeout)]
        route = list(set(route))
        edg_route_att[(node_in, node_out)]['routes'] = route
        edg_route_att[(node_in, node_out)]['color'] = colors_routes[route[0]]
    
    
    G_ = nx.DiGraph(G)
    nx.set_edge_attributes(G_, edg_route_att)
    
    # Save    
    path_save = os.path.join(fp, 'variables')
    
    with open(path_save + '\pt_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(G_, f)
    

    
    
    
    
    
    
    
    
    
    
    
    
