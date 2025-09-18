# -*- coding: utf-8 -*-
"""
Created on 2025-08-15

@author: Dylan Festa

Reads and processes data for hierarchical clustering analysis. 
The goal is to have simple averages of neural activity across multiple hunting sessions.
The values are baseline-subtracted. 

"""
#%%
from __future__ import annotations
from typing import List, Dict, Tuple, Union
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains


#%%

all_mice = rdl.get_good_animals()
print(f"Found {len(all_mice)} animals.")
print("Animals:", all_mice)

animal = 'afm16924'
sessions_for_animal = rdl.get_good_sessions(animal)
print(f"Found {len(sessions_for_animal)} sessions for animal {animal}.")
print("Sessions:", sessions_for_animal)

#%%

hunting_sessions = ['240522', '240523_0', '240529']

#%%

def plot_behaviour_labels_from_session(theanimal:str, thesession:str,
                    show_only: list[str] | None = None): 

    all_data_dict = rdl.load_preprocessed_dict(theanimal, thesession)
    behaviour = all_data_dict['behaviour']
    time_index = all_data_dict['time_index']
    time_start = 0.0
    time_stop = time_index[-1]

    behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(theanimal, thesession, behaviour)
    # keep only behaviours that is start_stop
    behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['is_start_stop']]
    # if show_only is not None, filter the dataframe
    if show_only is not None:
        behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['behaviour'].isin(show_only)]
    # short alphabetically by behaviour
    behaviour_timestamps_df.sort_values('behaviour', inplace=True)
    # inverse dictionary for convenience
    dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
    # PLOT!
    beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df,dict_behavior_label_to_index)
    #beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

    n_beh_keys = len(beh_plot_dict)

    thefig = go.Figure()
    thefig.update_layout(
    title="time labels for each behavior",
    xaxis_title="time (s)",
    yaxis_title="beh. idx"
    )
    bar_height = 0.8

    # define color dictionary
    color_map = px.colors.qualitative.Plotly
    beh_color_dict = {}
    for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
        beh_color_dict[beh_idx] = color_map[idx % len(color_map)]

    # Add custom legend
    for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
        thefig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=beh_color_dict[beh_idx]),
            legendgroup=beh,
            showlegend=True,
            name=str(beh)
        ))

    for k, (x, y) in enumerate(beh_plot_xy):
        # Prepare color mapping for legend and rectangles
        offset = bar_height / 2.0  # half the height of the bar
        for i in range(len(x) - 1):
            if not (np.isnan(x[i]) or np.isnan(x[i+1]) or np.isnan(y[i]) or np.isnan(y[i+1])):
                beh_idx = y[i]
                fillcolor = beh_color_dict.get(beh_idx, "#000000")
                thefig.add_shape(
                    type="rect",
                    x0=x[i], y0=y[i] - offset,
                    x1=x[i + 1],
                    y1=y[i] + offset,
                    fillcolor=fillcolor,
                    line=dict(color=fillcolor),
                    opacity=1.0,
                    layer='above',
                )
            i += 1

        
    thefig.update_xaxes(range=[time_start,time_stop])
    thefig.update_yaxes(range=[0, n_beh_keys + 1])

    return thefig

#%%
# let's test the function above

figure_22 = plot_behaviour_labels_from_session(animal,hunting_sessions[0])
figure_22.show(renderer="vscode")

#%%
figure_23 = plot_behaviour_labels_from_session(animal,hunting_sessions[1])
figure_23.show(renderer="vscode")

#%%

figure_29 = plot_behaviour_labels_from_session(animal,hunting_sessions[2])
figure_29.show(renderer="vscode")

#%%

my_show_only = ['chase','attack','approach','eat']

figure_22_less = plot_behaviour_labels_from_session(animal,hunting_sessions[0],show_only=my_show_only)
figure_22_less.show(renderer="vscode")

#%%

figure_23_less = plot_behaviour_labels_from_session(animal,hunting_sessions[1],show_only=my_show_only)
figure_23_less.show(renderer="vscode")

#%%
figure_29_less = plot_behaviour_labels_from_session(animal,hunting_sessions[2],show_only=my_show_only)
figure_29_less.show(renderer="vscode")

#%%
#%%

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(animal, session)
# unpack the data
behaviour = all_data_dict['behaviour']
spike_times = all_data_dict['spike_times']
time_index = all_data_dict['time_index']
cluster_index = all_data_dict['cluster_index']
region_index = all_data_dict['region_index']
channel_index = all_data_dict['channel_index']

t_data_load_seconds = time.time() - t_data_load_start
t_data_load_minutes = t_data_load_seconds / 60
print(f"Data loaded successfully in {t_data_load_seconds:.2f} seconds ({t_data_load_minutes:.2f} minutes).")

t_start_all = 0.0
t_stop_all = time_index[-1]


spiketrains=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')

n_units = spiketrains.n_units
print(f"Number of PAG units: {n_units}.")
#%% Now, process behaviour data to get labels

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter labels for training, myst be only start_stop, at least 15 trials and no pursuit or run_away behaviours
# keep only behaviours that occur at least 15 times
behaviour_timestamps_df = behaviour_timestamps_df[ (behaviour_timestamps_df['n_trials'] >= 15)
                                                  & (behaviour_timestamps_df['is_start_stop'])
                                                  & (behaviour_timestamps_df['behaviour'] != 'pursuit')
                                                  & (behaviour_timestamps_df['behaviour'] != 'run_away')
                                                  ]

#%% add a "loiter" behavior, in the form of 
# k randomly arranged, non superimposing 10 second intervals
# after the first 5 min but before the 5 min preceding the first labeled behavior


t_first_behav = behaviour['frames_s'].min()
t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

#%%

t_loiter_start = 5 * 60.0  # 5 minutes after the start of the session
t_loiter_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

k_intervals = 20  # number of intervals to generate
t_interval_duration = 20.0  # duration of each interval in seconds

intervals_loiter_fromzero = rdl.generate_random_intervals_within_time(
        t_loiter_end,k_intervals,t_interval_duration)

intervals_loiter = [(_start+ t_loiter_start, _end + t_loiter_start) for _start, _end in intervals_loiter_fromzero]


# add a row to behaviour dataframe
new_row = pd.DataFrame([{
    'mouse': animal,
    'session': session,
    'behavioural_category': 'loiter',
    'behaviour': 'loiter',
    'n_trials': k_intervals,
    'is_start_stop': True,
    'total_duration': t_interval_duration * k_intervals,
    'start_stop_times': intervals_loiter,
    'point_times': []
}])
behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)
                                    

#%%
# now generate a dictionary
# add 'none'
dict_behavior_label_to_index['none'] = -1
## pup grab and pup retrieve should have same label
##dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']

#%% for convenience, have a dictionary that maps index to label, merging labels with same index
# so the above becomes pup_grab_and_pup_retrieve
# dict_classindex_to_behavior={}
# Build inverse with merging of duplicate indices
_inv = {}
for label, idx in dict_behavior_label_to_index.items():
    _inv.setdefault(idx, []).append(label)


# Deduplicate and merge with "_and_" for multi-label indices
dict_classindex_to_behaviour = {
    idx: (labels[0] if len(labels) == 1 else "_and_".join(labels))
    for idx, labels in ((i, lst) for i, lst in _inv.items())
}

#%%
# build behaviour_representation_df
# keys are label and class index as in dict_classindex_to_behaviour
_rows_temp = []
for class_index, label in dict_classindex_to_behaviour.items():
    if label == 'none':
        continue
    _intervals = []
    for (_key,_val) in dict_behavior_label_to_index.items():
        if _val == class_index:
            #print(f"Adding intervals for label: {label} (class index: {class_index}), key: {_key}")
            _to_add = behaviour_timestamps_df[behaviour_timestamps_df['behaviour'] == _key]['start_stop_times'].values[0]
            _intervals.append(_to_add)
    # merge if needed
    if len(_intervals) == 2:
        _intervals = rdl.merge_time_interval_list(_intervals[0], _intervals[1])
    elif len(_intervals) > 2:
        raise ValueError("More than 2 intervals found")
    else:
        _intervals=_intervals[0]
    _rows_temp.append({'label':label,'class_index':class_index,'intervals':_intervals})

behaviour_representation_df = pd.DataFrame(_rows_temp,columns=['label','class_index','intervals'])
behaviour_representation_df['plot_index'] = behaviour_representation_df.index

#%% Plot the timestamps of the selected labels


beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df,dict_behavior_label_to_index)
#beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

n_beh_keys = len(beh_plot_dict)

thefig = go.Figure()
thefig.update_layout(
    title="time labels for each behavior",
    xaxis_title="time (s)",
    yaxis_title="beh. idx"
)
bar_height = 0.8

# define color dictionary
color_map = px.colors.qualitative.Plotly
beh_color_dict = {}
for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
    beh_color_dict[beh_idx] = color_map[idx % len(color_map)]

# Add custom legend
for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
    thefig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color=beh_color_dict[beh_idx]),
        legendgroup=beh,
        showlegend=True,
        name=str(beh)
    ))

for k, (x, y) in enumerate(beh_plot_xy):
    # Prepare color mapping for legend and rectangles
    offset = bar_height / 2.0  # half the height of the bar
    for i in range(len(x) - 1):
        if not (np.isnan(x[i]) or np.isnan(x[i+1]) or np.isnan(y[i]) or np.isnan(y[i+1])):
            beh_idx = y[i]
            fillcolor = beh_color_dict.get(beh_idx, "#000000")
            thefig.add_shape(
                type="rect",
                x0=x[i], y0=y[i] - offset,
                x1=x[i + 1],
                y1=y[i] + offset,
                fillcolor=fillcolor,
                line=dict(color=fillcolor),
                opacity=1.0,
                layer='above',
            )
        i += 1

    
thefig.update_xaxes(range=[0, t_stop_all])
thefig.update_yaxes(range=[0, n_beh_keys + 1])
    
#thefig.show(renderer="browser")
#thefig.show(renderer="vscode")
thefig.show()
