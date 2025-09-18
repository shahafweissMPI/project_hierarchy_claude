# -*- coding: utf-8 -*-
"""
Created on: 2025-07-30

Author: Dylan Festa


PCA analysis for lagged neural data visualization.

The goal is to perform PCA on neural data with lags—using different preprocessing methods—and
visualize the results, using labeled behaviors as colors in the PCA scatter plots.

If it looks good I might turn it into a dashboard :-P
"""



#%%

import numpy as np
import h5py
import os

import read_data_light as rdl


# date string in the format YYYYMMDD
import datetime
date_str = datetime.datetime.now().strftime("%Y%m%d")

#%%

good_paths_df = rdl.get_good_paths()

#%% select only first 2 rows
#good_paths_df = good_paths_df.loc[0:1]

#%%


hfile = h5py.File(f'/tmp/all_rasters_{date_str}.hdf5', 'w')

# %% Groups for each mouse/session
k_row = 0
tot_rows = len(good_paths_df)
for mouse, session in good_paths_df[['Mouse_ID', 'session']].drop_duplicates().values:
    k_row += 1
    # try reading, skip if error
    try:
        read_dict = rdl.load_preprocessed_dict(mouse, session)
    except Exception as e:
        print(f"\n Error reading {mouse}/{session}: {e} \n")
        continue
    group_name = f"{mouse}/{session}"
    group = hfile.create_group(group_name)
    print(f"Processing {group_name}, row {k_row} of {tot_rows}")
    print(f"Loaded data for {group_name}")
    trains=read_dict['spike_times']
    units=read_dict['cluster_index']
    unit_locations = read_dict['region_index']
    # save untis as dataset
    group.create_dataset('units', data=np.array(units, dtype='int32'))
    # unit locations are strings
    group.create_dataset('unit_locations', data=np.array(unit_locations, dtype='S'))
    # for each element in trains, create a dataset
    for i, train in enumerate(trains):
        dataset_name = f"{i}"
        group.create_dataset(dataset_name, data=np.array(train, dtype='float32'))
    print(f"Saved {len(trains)} spike trains for {group_name}")

#%%

print(f"******\nAll data saved to {hfile.filename}. Closing file! DONE!\n******")

hfile.close()


# #%%
# animal_test='afm16924'
# session_test='240522'


# paths_test=rdl.get_paths(animal_test, session_test)
# # %%
