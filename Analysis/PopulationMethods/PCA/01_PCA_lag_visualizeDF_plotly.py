# -*- coding: utf-8 -*-
"""
Created on 2025-07-28

Author: Dylan Festa


PCA analysis for lagged neural data visualization.

The goal is to perform PCA on neural data with lags—using different preprocessing methods—and
visualize the results, using labeled behaviors as colors in the PCA scatter plots.

If it looks good I might turn it into a dashboard :-P
"""
import os
import numpy as np, pandas as pd, xarray as xr
import time
import plotly.express as px

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

# import stuff from sklear: pipeline, lagged data, Z-score, PCA
from sklearn.pipeline import make_pipeline
from sktime.transformations.series.lag import Lag
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer

#%%Loading of neural data

animal = 'afm16924'
session = '240524'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
[_, 
 behaviour, 
 _, 
 n_spike_times,
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 _, 
 _, 
 _, 
 _] = rdl.load_preprocessed(animal, session)
t_data_load_seconds = time.time() - t_data_load_start
t_data_load_minutes = t_data_load_seconds / 60
print(f"Data loaded successfully in {t_data_load_seconds:.2f} seconds ({t_data_load_minutes:.2f} minutes).")

t_start_all = 0.0
t_stop_all = n_time_index[-1]

#%%
# key parameters here: time bin size, and how many steps compose the lag

dt= 100*1E-3  # time bin size in seconds
n_lag_steps = 1  # number of lag steps

print(f"Parameters set: dt = {dt} seconds, n_lag_steps = {n_lag_steps}, total time-window size: {dt * n_lag_steps} seconds.")


def drop_nan_lag_rows_from_X(X):
    """
    Drop rows with NaN values in the lagged data.
    This is necessary because the first n_lag_steps-1 rows will have NaN values after lagging.
    """
    k = n_lag_steps - 1
    if k == 0:
        return X
    elif k < 0:
        raise ValueError("n_lag_steps must be greater than 0.")
    elif k >= X.shape[0]:
        raise ValueError("n_lag_steps is too large for the number of samples in X.")
    return X[k:-k]

# this is for the label or time series vector. In this case I just need to remvoe the first n_lag_steps-1 elements
def drop_nan_lag_rows_from_y(y):
    k = n_lag_steps - 1
    if k == 0:
        return y
    elif k < 0:
        raise ValueError("n_lag_steps must be greater than 0.")
    elif k >= len(y):
        raise ValueError("n_lag_steps is too large for the number of samples in y.")
    return y[k:]

#%%

# use library function
spiketrains=pre.SpikeTrains.from_spike_list(n_spike_times,
                                units=n_cluster_index,
                                unit_location=n_region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')
# compute iFRs too
iFRs = pre.IFRTrains.from_spiketrains(spiketrains)

#%% Let's compute X

#binneddatax=pre.do_binning_operation(spiketrains,'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)
binneddatax=pre.do_binning_operation(iFRs,'mean',dt=dt,t_start=t_start_all,t_stop=t_stop_all)


#%%
bin_centers = binneddatax.coords['time_bin_center'].values
bin_edges = binneddatax.attrs['time_bin_edges']
#print(f"feature dimension (i.e. neurons) expanded from {binneddatax.shape[1]} to {lagged_shape[1]} after lag expansion.")

#%%
pipe = make_pipeline(
    Lag(lags=list(range(n_lag_steps)), index_out='extend'),
    FunctionTransformer(drop_nan_lag_rows_from_X),
    StandardScaler(),
    PCA(n_components=2)
)

# Fit-transform the pipeline
binneddatax_lagged_pca = pipe.fit_transform(binneddatax.values)

#%% Now, let's get labels!

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter all behaviors that occurr less than 10 times
behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['n_trials'] >= 10]

dict_beh ={ beh_:k
    for (k, beh_) in enumerate(behaviour_timestamps_df['behaviour'])}
    

behaviour_labels_xarray = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                        t_start =0.0,t_stop= n_time_index[-1],
                                        dt=dt,
                                        behaviour_labels_dict=dict_beh)


# Get labels for coloring
binneddatay = drop_nan_lag_rows_from_y(behaviour_labels_xarray.values)

#%%
# check that size matches
if binneddatax_lagged_pca.shape[0] != len(binneddatay):
    raise AssertionError("Mismatch between number of lagged PCA samples and behavior labels.")


#%% Scatter plot of first two PCs colored by behavior label
# slect only behaviors that are not none!
idx_valid_behaviors = binneddatay != -1  # assuming -1 is the 'none' label

inv_dict_beh = {v: k for k, v in dict_beh.items()}
df_plot = pd.DataFrame({
    'PC1': binneddatax_lagged_pca[idx_valid_behaviors, 0],
    'PC2': binneddatax_lagged_pca[idx_valid_behaviors, 1],
    'Behavior': [inv_dict_beh[label] for label in binneddatay[idx_valid_behaviors]]
})

fig = px.scatter(
    df_plot,
    x='PC1',
    y='PC2',
    color='Behavior',
    title='PCA of Lagged Neural Data (colored by behavior)',
    opacity=0.7,
    height=700,  # Add this line to make the figure taller
)
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
#fig.show(renderer="vscode")
fig.show()

#%%
# Here I repeat everything, but using a shuffled version of the data, as a control


spiketrains_shuffled = spiketrains.generate_shuffled_control()
# compute iFRs too
iFRs_shuffled = pre.IFRTrains.from_spiketrains(spiketrains_shuffled)

#%% Let's compute X

binneddatax=pre.do_binning_operation(spiketrains_shuffled,'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)
#binneddatax=pre.do_binning_operation(iFRs_shuffled,'mean',dt=dt,t_start=t_start_all,t_stop=t_stop_all)


#%%
bin_centers = binneddatax.coords['time_bin_center'].values
bin_edges = binneddatax.attrs['time_bin_edges']
#print(f"feature dimension (i.e. neurons) expanded from {binneddatax.shape[1]} to {lagged_shape[1]} after lag expansion.")

#%%
pipe = make_pipeline(
    Lag(lags=list(range(n_lag_steps)), index_out='extend'),
    FunctionTransformer(drop_nan_lag_rows_from_X),
    StandardScaler(),
    PCA(n_components=2)
)

# Fit-transform the pipeline
binneddatax_lagged_pca = pipe.fit_transform(binneddatax.values)

#%% Now, let's get labels!

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter all behaviors that occurr less than 10 times
behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['n_trials'] >= 10]

dict_beh ={ beh_:k
    for (k, beh_) in enumerate(behaviour_timestamps_df['behaviour'])}
    

behaviour_labels_xarray = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                        t_start =0.0,t_stop= n_time_index[-1],
                                        dt=dt,
                                        behaviour_labels_dict=dict_beh)


# Get labels for coloring
binneddatay = drop_nan_lag_rows_from_y(behaviour_labels_xarray.values)

#%%
# check that size matches
if binneddatax_lagged_pca.shape[0] != len(binneddatay):
    raise AssertionError("Mismatch between number of lagged PCA samples and behavior labels.")


#%% Scatter plot of first two PCs colored by behavior label
# slect only behaviors that are not none!
idx_valid_behaviors = binneddatay != -1  # assuming -1 is the 'none' label

inv_dict_beh = {v: k for k, v in dict_beh.items()}
df_plot = pd.DataFrame({
    'PC1': binneddatax_lagged_pca[idx_valid_behaviors, 0],
    'PC2': binneddatax_lagged_pca[idx_valid_behaviors, 1],
    'Behavior': [inv_dict_beh[label] for label in binneddatay[idx_valid_behaviors]]
})

fig = px.scatter(
    df_plot,
    x='PC1',
    y='PC2',
    color='Behavior',
    title='PCA of Lagged Neural Data (colored by behavior)',
    opacity=0.7,
    height=700,  # Add this line to make the figure taller
)
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
#fig.show(renderer="vscode")
fig.show()



# %%
