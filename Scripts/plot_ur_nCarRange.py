# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:20:49 2024

@author: sarab
"""


from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size#, SubplotDivider
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
                    '5691': '5632', 
                    '5513': '5657', 
                    '5614': '5613', 
                    '5511': '5505', 
                    '5581': '5656', 
                    '5583': '5644',
                    '5645': '5646',
                    '5633': '5632'} 
    
plt.rcParams.update({"text.usetex": True,'font.size': 25})
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

# orig_map=plt.cm.get_cmap('RdBu')
orig_map=plt.colormaps['RdBu'] 
reversed_map = orig_map.reversed() 

fp = os.path.join(os.getcwd(), 'output')
# form = 'svg'
form = 'pdf'
dpi = 500

min_lst = []
max_lst = []

pc_info = pc4d_join
nr = pc4_info['population']

# nCarRange = [1e3, 3e3, 4e3]
nCarRange = [3e3]

for nCar in nCarRange:
    
    str_folder = 'output\\nCar\\' + str(int(nCar)) + '\\figures\\new plts'
    
    folder_ = os.path.join(os.getcwd(), str_folder)
    
    
    str_folder = 'output\\nCar\\' + str(int(nCar))
    
    fp = os.path.join(os.getcwd(), str_folder)
    
    Tmax = '20'
    
    load_m = loadmat(fp + '\\matlab_python_T' + Tmax + '.mat')
    load_m_MIQP = loadmat(fp + '\\matlab_python_T' + Tmax + '_MIQP.mat')
    
    edges_matlab = load_m['edgesMatlab']
    demand = load_m['demand']
    D_total = sum(sum(demand))
    
    edges = [(nodes[edge[0]-1],nodes[edge[1]-1]) 
              for edge in edges_matlab]
        
    ur_path_mintt = load_m['ur_tt_path']
    ur_path_maxAcc = load_m['ur_avgAcc_path']
    ur_path_pathAcc = load_m['ur_pathAcc_path']
    
    ur_avg_mintt = load_m['ur_tt_avg']
    ur_avg_maxAcc = load_m['ur_avgAcc_avg']
    ur_avg_pathAcc = load_m['ur_pathAcc_avg']
    
    pc_unique = load_m['pc_unique']
    
    ur = {str(v[0]): {'tt_avg': ur_avg_mintt[i][0],
                      'tt_path': ur_path_mintt[i][0],
                      'avgAcc_avg': ur_avg_maxAcc[i][0],
                      'avgAcc_path': ur_path_maxAcc[i][0],
                      'pathAcc_avg': ur_avg_pathAcc[i][0],
                      'pathAcc_path': ur_path_pathAcc[i][0],}
          for i, v in enumerate(pc_unique)}
    
    for pc1, pc2 in dict_weird_nodes.items():
        ur[pc1] = ur[pc2]
    
    pc_info['tt_avg'] = np.nan
    pc_info['tt_path'] = np.nan
    pc_info['avgAcc_avg'] = np.nan
    pc_info['avgAcc_path'] = np.nan
    pc_info['pathAcc_avg'] = np.nan
    pc_info['pathAcc_path'] = np.nan
    
    pc_info = pc_info.reset_index(drop=True)
    
    for pc in ur.keys():
        ind = pc_info.loc[pc_info['postcode4'] == pc].index[0]
        pc_info.loc[ind, 'tt_avg'] = ur[pc]['tt_avg']
        pc_info.loc[ind, 'tt_path'] = ur[pc]['tt_path']
        pc_info.loc[ind, 'avgAcc_avg'] = ur[pc]['avgAcc_avg']
        pc_info.loc[ind, 'avgAcc_path'] = ur[pc]['avgAcc_path']
        pc_info.loc[ind, 'pathAcc_avg'] = ur[pc]['pathAcc_avg']
        pc_info.loc[ind, 'pathAcc_path'] = ur[pc]['pathAcc_path']
    
    pc_info = geopandas.GeoDataFrame(pc_info, crs="EPSG:4326")
    
    min_v = min(min(ur_path_mintt),min(ur_avg_mintt),
                min(ur_path_maxAcc),min(ur_avg_maxAcc),
                min(ur_path_pathAcc),min(ur_avg_pathAcc))
    max_v = max(max(ur_path_mintt),max(ur_avg_mintt),
                max(ur_path_maxAcc),max(ur_avg_maxAcc),
                max(ur_path_pathAcc),max(ur_avg_pathAcc))
    
    Tmax = '20'
    
    # folder = os.path.join(folder_, Tmax + '\\sq_')
    folder = os.path.join(folder_, 'sq_')
    
    leg_label = "$u_{r}\\ [\\mathrm{min}^2]$"
    
    # min TT avg
    col_name = 'tt_avg'
    plot_afi_heatmap(pc_info, col_name, leg_label, min_v, max_v, reversed_map, 
                      'ur_TT_OD_heatmap', folder, form, dpi)
    
    # min TT path
    col_name = 'tt_path'
    plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_TT_paths_heatmap', folder, form, dpi)
    
    # max Acc avg
    col_name = 'avgAcc_avg'
    plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_Acc_OD_heatmap', folder, form, dpi)
    
    # max Acc path
    col_name = 'avgAcc_path'
    plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_Acc_paths_heatmap', folder, form, dpi)
    
    # path Acc avg
    col_name = 'pathAcc_avg'
    plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_pathAcc_OD_heatmap', folder, form, dpi)
    
    # path Acc path
    col_name = 'pathAcc_path'
    plot_afi_heatmap(pc_info, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_pathAcc_paths_heatmap', folder, form, dpi)
    
    
    ur_minTT_avg_dest = load_m['ur_minTT_avg_dest']
    ur_minTT_path_dest = load_m['ur_minTT_path_dest']
    ur_avgAcc_avg_dest = load_m['ur_avgAcc_avg_dest']
    ur_avgAcc_path_dest = load_m['ur_avgAcc_path_dest']
    ur_pathAcc_avg_dest = load_m['ur_pathAcc_avg_dest']
    ur_pathAcc_path_dest = load_m['ur_pathAcc_path_dest']
    ur_dest_def_path_dest = load_m['ur_dest_def_path_dest']
    ur_pathAcc_avg_dest_MIQP = load_m_MIQP['dest_def_OD_pathAcc']
    ur_pathAcc_path_dest_MIQP = load_m_MIQP['dest_def_path_pathAcc']
    ur_dest_def_path_dest_MIQP = load_m_MIQP['ur_dest_def_path_dest']
    
    pc_unique = load_m['pc_unique']
    
    ur = {str(v[0]): {'ur_minTT_avg_dest': ur_minTT_avg_dest[i][0],
                      'ur_minTT_path_dest': ur_minTT_path_dest[i][0],
                      'ur_avgAcc_avg_dest': ur_avgAcc_avg_dest[i][0],
                      'ur_avgAcc_path_dest': ur_avgAcc_path_dest[i][0],
                      'ur_pathAcc_avg_dest': ur_pathAcc_avg_dest[i][0],
                      'ur_pathAcc_path_dest': ur_pathAcc_path_dest[i][0],
                      'ur_dest_def_path_dest': ur_dest_def_path_dest[i][0],
                      'ur_pathAcc_avg_dest_MIQP': ur_pathAcc_avg_dest_MIQP[i][0],
                      'ur_pathAcc_path_dest_MIQP': ur_pathAcc_path_dest_MIQP[i][0],
                      'ur_dest_def_path_dest_MIQP': ur_dest_def_path_dest_MIQP[i][0],}
          for i, v in enumerate(pc_unique)}
    
    for pc1, pc2 in dict_weird_nodes.items():
        ur[pc1] = ur[pc2]
    
    pc_info_dest = pc_info.copy()
    
    pc_info_dest['ur_minTT_avg_dest'] = np.nan
    pc_info_dest['ur_minTT_path_dest'] = np.nan
    pc_info_dest['ur_avgAcc_avg_dest'] = np.nan
    pc_info_dest['ur_avgAcc_path_dest'] = np.nan
    pc_info_dest['ur_pathAcc_avg_dest'] = np.nan
    pc_info_dest['ur_pathAcc_path_dest'] = np.nan
    pc_info_dest['ur_dest_def_path_dest'] = np.nan
    pc_info_dest['ur_pathAcc_avg_dest_MIQP'] = np.nan
    pc_info_dest['ur_pathAcc_path_dest_MIQP'] = np.nan
    pc_info_dest['ur_dest_def_path_dest_MIQP'] = np.nan
    
    pc_info_dest = pc_info_dest.reset_index(drop=True)
    
    for pc in ur.keys():
        ind = pc_info_dest.loc[pc_info['postcode4'] == pc].index[0]
        pc_info_dest.loc[ind, 'ur_minTT_avg_dest'] = ur[pc]['ur_minTT_avg_dest']
        pc_info_dest.loc[ind, 'ur_minTT_path_dest'] = ur[pc]['ur_minTT_path_dest']
        pc_info_dest.loc[ind, 'ur_avgAcc_avg_dest'] = ur[pc]['ur_avgAcc_avg_dest']
        pc_info_dest.loc[ind, 'ur_avgAcc_path_dest'] = ur[pc]['ur_avgAcc_path_dest']
        pc_info_dest.loc[ind, 'ur_pathAcc_avg_dest'] = ur[pc]['ur_pathAcc_avg_dest']
        pc_info_dest.loc[ind, 'ur_pathAcc_path_dest'] = ur[pc]['ur_pathAcc_path_dest']
        pc_info_dest.loc[ind, 'ur_dest_def_path_dest'] = ur[pc]['ur_dest_def_path_dest']
        pc_info_dest.loc[ind, 'ur_pathAcc_avg_dest_MIQP'] = ur[pc]['ur_pathAcc_avg_dest_MIQP']
        pc_info_dest.loc[ind, 'ur_pathAcc_path_dest_MIQP'] = ur[pc]['ur_pathAcc_path_dest_MIQP']
        pc_info_dest.loc[ind, 'ur_dest_def_path_dest_MIQP'] = ur[pc]['ur_dest_def_path_dest_MIQP']
    
    pc_info_dest = geopandas.GeoDataFrame(pc_info_dest, crs="EPSG:4326")
    
    min_v = min(min(ur_minTT_avg_dest),min(ur_minTT_path_dest),
                min(ur_avgAcc_avg_dest),min(ur_avgAcc_path_dest),
                min(ur_pathAcc_avg_dest),min(ur_pathAcc_path_dest),
                min(ur_dest_def_path_dest),)
    
    min_MIQP = min(min(ur_pathAcc_avg_dest_MIQP),
                   min(ur_pathAcc_path_dest_MIQP),
                   min(ur_dest_def_path_dest_MIQP))
    
    max_v = max(max(ur_minTT_avg_dest),max(ur_minTT_path_dest),
                max(ur_avgAcc_avg_dest),max(ur_avgAcc_path_dest),
                max(ur_pathAcc_avg_dest),max(ur_pathAcc_path_dest),
                max(ur_dest_def_path_dest))
    
    max_MIQP = max(max(ur_pathAcc_avg_dest_MIQP),
                   max(ur_pathAcc_path_dest_MIQP),
                   max(ur_dest_def_path_dest_MIQP))
    
    Tmax = '20'
    
    folder = os.path.join(folder_, 'dest_')
    
    leg_label = "$u_{r}$"#"\\ [\\mathrm{N\ destinations}]$"
    
    
    col_name = 'ur_minTT_avg_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label, min_v, max_v, reversed_map, 
                      'ur_minTT_avg_dest', folder, form, dpi)
    
    col_name = 'ur_minTT_path_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label, min_v, max_v, reversed_map, 
                      'ur_minTT_path_dest', folder, form, dpi)
    
    col_name = 'ur_avgAcc_avg_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label, min_v, max_v, reversed_map, 
                      'ur_avgAcc_avg_heatmap', folder, form, dpi)
    
    col_name = 'ur_avgAcc_path_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label, min_v, max_v, reversed_map, 
                      'ur_avgAcc_path_dest', folder, form, dpi)
    
    col_name = 'ur_pathAcc_avg_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_pathAcc_avg_dest', folder, form, dpi)
    
    col_name = 'ur_pathAcc_path_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_pathAcc_path_heatmap', folder, form, dpi)
    
    
    col_name = 'ur_dest_def_path_dest'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_v, max_v, reversed_map, 
                      'ur_dest_def_path_heatmap', folder, form, dpi)

    
    
    col_name = 'ur_pathAcc_avg_dest_MIQP'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_MIQP, max_MIQP, reversed_map, 
                      'ur_pathAcc_avg_dest_MIQP', folder, form, dpi)

    
    col_name = 'ur_pathAcc_path_dest_MIQP'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_MIQP, max_MIQP, reversed_map, 
                      'ur_pathAcc_path_dest_MIQP', folder, form, dpi)

    
    col_name = 'ur_dest_def_path_dest_MIQP'
    plot_afi_heatmap(pc_info_dest, col_name, leg_label,  min_MIQP, max_MIQP, reversed_map, 
                      'ur_dest_def_path_MIQP_heatmap', folder, form, dpi)


# population

pc_population = load_m_MIQP['population_region']
pc_info_pop = pc_info.copy()
pc_info_pop = pc_info_dest.reset_index(drop=True)

pop = {str(v[0]): {'population_region': pc_population[i][0],}
       for i, v in enumerate(pc_unique)}

for pc1, pc2 in dict_weird_nodes.items():
    pop[pc1] = pop[pc2]

pc_info_pop['population_region'] = np.nan

pc_info_pop = pc_info_pop.reset_index(drop=True)

for pc in pop.keys():
    ind = pc_info_dest.loc[pc_info['postcode4'] == pc].index[0]
    pc_info_pop.loc[ind, 'population_region'] = pop[pc]['population_region']
    
pc_info_pop = geopandas.GeoDataFrame(pc_info_pop, crs="EPSG:4326")

min_v = min(pc_population)
max_v = max(pc_population)

col_name = 'population_region'
plot_afi_heatmap(pc_info_pop, col_name, leg_label,  min_v, max_v, reversed_map, 
                  'population_region', folder, form, dpi)

