# -*- coding: utf-8 -*-
"""
Created on 2025-08-26

@author: Dylan Festa

I am focusing narrowly on four behaviours: chase, pup_run, pup_retrieve, (escape)

The first step is to navigate the dataset, and find the sessions where they are mostly expressed.
"""
#%%

import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json
from joblib import load


# impot local modules in PopulationMethods/lib 
import read_data_light as rdl

#%%

interesting_behaviours = ['chase', 'pup_run', 'pup_retrieve','escape','escape_switch']
required_behaviours = [behaviour\
        for behaviour in interesting_behaviours if (behaviour != 'escape') and (behaviour != 'escape_switch')]

def is_good_session(timestamps_df):
    # required are interesting behaviours minus escape and escape_switch
    all_behaviours = timestamps_df['behaviour'].unique()
    return all(behaviour in all_behaviours for behaviour in required_behaviours)


#%%

animals_all = rdl.get_good_animals()
print(f"Found {len(animals_all)} animals.")
print("Animals:", animals_all)

#%%
# dictionary of list of sessions for each animal
dict_sessions = {}
for _animal in animals_all:
    sessions_for_animal = rdl.get_good_sessions(_animal)
    dict_sessions[_animal] = sessions_for_animal

# exclude sessions with things in their name
exclude_things=['test','Kilosort', 'coded','overlap' ]

for _animal in animals_all:
    dict_sessions[_animal] = [session for session in dict_sessions[_animal] if not any(thing in session for thing in exclude_things)]

#%% now, single beh timestamps df
time_start = time.time()
behaviour_timestamps_dfs = []
for _animal in animals_all:
    for _session in dict_sessions[_animal]:
        _data_dict = rdl.load_preprocessed_dict(_animal, _session)
        _beh = _data_dict['behaviour']
        _beh_ts_df = rdl.convert_to_behaviour_timestamps(_animal, _session, _beh)
        behaviour_timestamps_dfs.append(_beh_ts_df)

behaviour_timestamps_df = pd.concat(behaviour_timestamps_dfs)
time_end = time.time()
time_duration_str = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
print(f"Time taken to create behaviour timestamps df: {time_duration_str}")

#%% Filter sessions that contain all required behaviours
behaviour_timestamps_good_df = behaviour_timestamps_df.groupby(['mouse', 'session']).filter(is_good_session)
n_total = behaviour_timestamps_df[['mouse','session']].drop_duplicates().shape[0]
n_good = behaviour_timestamps_good_df[['mouse','session']].drop_duplicates().shape[0]
print(f"Good sessions (with all required behaviours): {n_good}/{n_total}")
#%% Simple per-behaviour summary (assumes 'duration' column exists)
_beh = behaviour_timestamps_good_df[behaviour_timestamps_good_df['behaviour'].isin(interesting_behaviours)].copy()


for (m, s), df_ms in _beh.groupby(['mouse','session']):
    print(f"\nMouse: {m}  Session: {s}")
    for beh in interesting_behaviours:
        row = df_ms[df_ms['behaviour'] == beh]
        if row.empty:
            print(f"  {beh:12s} n_trials:   0 total_duration: 0.00")
        else:
            print(f"  {beh:12s} n_trials: {int(row.iloc[0].n_trials):2d} total_duration: {row.iloc[0].total_duration:.2f}")
#%%


