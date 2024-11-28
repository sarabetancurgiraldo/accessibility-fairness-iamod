# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:35:54 2024

@author: sarab
"""

import os
import pandas as pd
import zipfile
import pickle


os.chdir('..')

fp = os.path.join(os.getcwd(), 'variables\\data_Eindhoven.pkl')
with open(fp, 'rb') as f:  
    [G_w, G_b, G_c, pc4d_crop, pc4d_join, pc4d_data, 
     multiplier_low_income, G_cbw, G_o, G_ocbw, pc4_info, 
     G_pt, G_ocbwpt, G_d, G_ocbwptd, G_obwptd, full_demand, 
     data_matlab, G_pt] = pickle.load(f)

motives_path = os.path.join(os.getcwd(), 
                'variables\\20230112_EhvMetro_Fabio_sched_singleDay-gen.zip')
archive = zipfile.ZipFile(motives_path, 'r')
motives_file = archive.open('20230112_EhvMetro_Fabio_sched_singleDay-gen.txt')
motives = pd.read_csv(motives_file)

pc4_unique = pc4_info['unique']

weird_pc = 5633


# mask1 = [str(pc) in pc4_unique for pc in motives['OrigLoc']]
# mask2 = [str(pc) in pc4_unique for pc in motives['DestLoc']]
# mask = mask1 and mask2
mask = [pc == weird_pc for pc in motives['OrigLoc']]
motives_pc = motives.loc[mask, ['OrigLoc','DestLoc','ActivityType', 'Mode']]

mask_trips_outside = [str(pc) not in pc4_unique for pc in motives_pc['DestLoc']]
motives_trips_outside = motives_pc.loc[mask_trips_outside]

mask_trips_inside = [str(pc) in pc4_unique for pc in motives_pc['DestLoc']]
motives_trips_inside = motives_pc.loc[mask_trips_inside]


unique_dest = motives_pc.DestLoc.unique()
dest_out = motives_trips_outside.DestLoc.unique()
dest_in = motives_trips_inside.DestLoc.unique()


motives_to_remove = ['Home', 'NonGroc', 'BringGet', 'Other']

mask_motives = [activity not in motives_to_remove for activity in motives_trips_inside['ActivityType']]
motives_trips_in_filter = motives_trips_inside[mask_motives]

dest_in_filter = motives_trips_in_filter.DestLoc.unique()


mask_pcs_all = [str(pc) in pc4_unique for pc in motives['OrigLoc']]
mask_motives_all = [activity not in motives_to_remove for activity in motives['ActivityType']]
mask_all = mask_pcs_all and mask_motives_all
motives_all_filter = motives[mask_all]

trips_pc = motives_all_filter.OrigLoc.value_counts()



#%%%%%%%


mask = [motive not in motives_to_remove for motive in motives['ActivityType']]
filtered_motives = motives.loc[mask, ['ActivityType','OrigLoc','DestLoc']]

mask1 = [str(pc) in pc4_unique for pc in filtered_motives['OrigLoc']]
mask2 = [str(pc) in pc4_unique for pc in filtered_motives['DestLoc']]
mask = [o and d for o, d in zip(mask1, mask2)]
filtered_motives_pc = filtered_motives.loc[mask]

ActivityType = ['Business','Groceries','Leisure','Services','Social','Touring','Work']
Eq = ['Work','Groceries','Leisure','Services','Leisure','Leisure','Work']
data = {'ActivityType': ActivityType, 'ResumedActivity': Eq}
to_join = pd.DataFrame(data=data)

motives_final = pd.merge(filtered_motives_pc, to_join, how="left")

summary_pcs = motives_final[['OrigLoc','DestLoc']].value_counts()



