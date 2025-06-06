# %%
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

# %%
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

color_kthBlue = '#004791'  # Default color for the plot
color_kthNavy = '#000061'  # Default color for the plot

os.chdir('..')

fp = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
     multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
     G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, 
     data_matlab, G_pt] = pickle.load(f)


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
    

# %%

size_oris = np.zeros(len(G_o._node.keys()))
size_dest = np.zeros(len(size_oris))

pc_or = '5623'
mask_region = np.zeros(len(nodes), dtype=bool)
# Create a mask for the nodes in the selected postcode region
for i, node in enumerate(nodes):
    if G_ocbwptd._node[node]['type'] == 'o':
        if G_o._node[node]['postcode'] == pc_or:
            mask_region[i] = True

demand_region = D[mask_region,:]
od_region = np.where(demand_region<0)[1]
node_size = np.sum(D[:,od_region],axis=1)

o_region = np.where(node_size<0)[0]
demand_region = abs(sum(node_size[o_region]))
node_size[o_region] = demand_region / len(o_region) * 0.1

pos_all = GetCoordList(G_ocbwptd)

fig_g, ax_g = plt.subplots(figsize=(10, 10))
pc4d_crop.boundary.plot(ax=ax_g, linewidth=1, color='gray',zorder=1)

nx.draw_networkx(G_ocbwptd, 
                ax=ax_g, 
                pos=pos_all, 
                with_labels=False, 
                node_size=node_size*3, 
                arrows=False, 
                nodelist=nodes, 
                node_color=color_kthBlue, 
                edgelist=[], )

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
ax_g.set_axis_off()
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})


plt.savefig('output\\figures\\demand_region_blue.svg', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

# %%