# -*- coding: utf-8 -*-
"""
Created on 2025-07-30

@author: Dylan Festa

Similar to `01_lda_withoutPCA.py`, but with a grid search 
for hyperparameters such as bin size and number of lags.
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer

#%% Utility functions

def select_specific_idxs_from_X_y(X,idxs):
    """
    Select specific indices from X. 
    X and idxs are supposed to be numpy arrays.
    """
    return X[idxs]


#%%Loading of neural data

animal = 'afm16924'
session = '240524'

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
# keep only behaviours that occur at least 15 times
behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['n_trials'] >= 15]
# now generate a dictionary
dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# remove 'loom'
dict_behavior_label_to_index.pop('loom', None)
# add 'none'
dict_behavior_label_to_index['none'] = -1
# pup grab and pup retrieve should have same label
dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']


#%% Plot the timestamps of the selected labels
import plotly.graph_objects as go


beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df,dict_behavior_label_to_index)
#beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

n_beh_keys = len(beh_plot_dict)

#%%

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


#%%
n_lags_max = 20
# Define hyperparameters for grid search
# lag zero means to consider only the current time, lag 1 means to double the dimensions, etc

lags_hyper = [ 0,1,4,9,14,n_lags_max-1]  # 0 means no lag, 1 means one lag, etc.
bins_hyper = [ k * 1E-3 for k in [10, 20,30,50,100,150,200] ]

the_lda = LDA(solver='lsqr', shrinkage=None)

#%%


# Different pipelines for different bin sizes, since this affects the labels too!

pipes_results = []

time_start_gridsearch = time.time()
print("Starting grid search...")

for (kdt,dt) in enumerate(bins_hyper):
    print(f"bin step {kdt+1}/{len(bins_hyper)}: Processing dt = {dt} seconds...")

    # bin the data according to dt
    X_alltime=pre.do_binning_operation(spiketrains,
                    'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)
    # get labels
    beh_labels_xr = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                        t_start =0.0,t_stop= t_stop_all,
                                        dt=dt,
                                        behaviour_labels_dict=dict_behavior_label_to_index)
    y_alltime = beh_labels_xr.values
    if len(y_alltime) != X_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after binning.")
    # apply max lag to the data outside
    print(f"Applying max lag of {n_lags_max} to the data...")
    X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion(X_alltime.values, n_lags_max)
    print("Max lag applied successfully.")
    # drop NaN rows from X and y
    X_alltime_maxlagged = X_alltime_maxlagged[n_lags_max:,:]  # keep only the last n_lags_max rows
    y_alltime = y_alltime[n_lags_max:]  # keep only the last n_lags_max rows
    if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")
    # now, select only behavioural labels
    y_idx_behavior = y_alltime != -1  # -1 is the 'none' label
    y_behaviour = y_alltime[y_idx_behavior]
    X_behaviour_maxlagged = X_alltime_maxlagged[y_idx_behavior, :]
    if X_behaviour_maxlagged.shape[0] != y_behaviour.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after 'select_behaviour_idxs' step.")
    # now define a function to apply the actual lag to X, for the pipeline
    def apply_lag(X, n_lag_steps):
        X_new = np.copy(X[:,:(n_lag_steps+1) * n_units])  # keep only the first n_lag_steps*n_units columns
        return X_new
   
    # Create a pipeline for the current bin size
    pipe = Pipeline([
        ('apply_lag', FunctionTransformer(apply_lag, kw_args={'n_lag_steps': None})),
        ('scaler', StandardScaler()),
        ('lda', the_lda)
    ])


    # Add scaler on/off as a hyperparameter
    cv = GridSearchCV(
        estimator=pipe,
        param_grid={
            'apply_lag__kw_args': [{'n_lag_steps': lag} for lag in lags_hyper],
            'scaler': [StandardScaler(), 'passthrough'],
            'lda__shrinkage': [None, 'auto'],
        },
        scoring='accuracy',
        #cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        n_jobs=-1,
        verbose=1
    )

    cv.fit(X_behaviour_maxlagged, y_behaviour)

    best_params = cv.best_params_
    best_score = cv.best_score_
    all_scores = cv.cv_results_

    print(f"Best parameters for dt={dt}: {best_params}")
    print(f"Best cross-validated score for dt={dt}: {best_score:.4f}")

    # Save all scores as well
    pipes_results.append({
        'dt': dt,
        'best_params': best_params,
        'best_score': best_score,
        'all_scores': all_scores
    })

#%%
#check total time taken
time_end_gridsearch = time.time()
time_gridsearch_string = time.strftime("%H:%M:%S", time.gmtime(time_end_gridsearch - time_start_gridsearch))
print(f"Grid search completed in: {time_gridsearch_string}")


#%%%
#

for pip in pipes_results:
    _pip_dt = pip['dt']
    _pip_best_params = pip['best_params']['apply_lag__kw_args']['n_lag_steps']
    print(f"dt={_pip_dt}, best lag steps={_pip_best_params}, best score={pip['best_score']:.4f}")

#%% heatmap of score, with dt and lag steps as x and y 

bins_hyper_str= [f"dt={k}" for k in bins_hyper]
lags_hyper_str = [f"lags={k}" for k in lags_hyper]

#%% xarray.DataArray to hold the scores
# HELP

# Prepare a 2D array to hold the best scores for each (dt, lag) combination
score_matrix = np.full((len(bins_hyper), len(lags_hyper)), np.nan)

for i, pipe in enumerate(pipes_results):
    all_scores = pipe['all_scores']
    for j, lag in enumerate(lags_hyper):
        # Find the index in cv_results_ where n_lag_steps == lag
        for k, params in enumerate(all_scores['param_apply_lag__kw_args']):
            if params['n_lag_steps'] == lag:
                score_matrix[i, j] = all_scores['mean_test_score'][k]
                break

score_da = xr.DataArray(
    score_matrix,
    coords={'dt': bins_hyper, 'n_lag_steps': lags_hyper},
    dims=['dt', 'n_lag_steps']
)

#%%

import plotly.express as px

fig = px.imshow(
    score_da.values,
    x=[f"lags={k}" for k in lags_hyper],
    y=[f"dt={k*1e3:.0f}ms" for k in bins_hyper],
    color_continuous_scale='Viridis',
    labels={'x': 'Lag steps', 'y': 'Bin size (ms)', 'color': 'Score'},
    aspect='equal'
)
fig.update_layout(title="Grid Search Accuracy Heatmap")
fig.show()


#%%
# Test using max lab early on, and then cutting at the lag level desired

# dt_test = 10*1E-3  # 10 ms

# X_alltime=pre.do_binning_operation(spiketrains,
#                     'count',dt=dt_test,t_start=t_start_all,t_stop=t_stop_all)
# y_alltime = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
#                                         t_start =0.0,t_stop= t_stop_all,
#                                         dt=dt_test,
#                                         behaviour_labels_dict=dict_behavior_label_to_index).values

# the_max_lag = Lag(lags=list(range(n_lags_max)))
# X_alltime_lagged = the_max_lag.fit_transform(X_alltime.values)

# # cut NaN rows from X and y
# X_alltime_lagged = drop_nan_lag_rows_from_X(X_alltime_lagged, n_lags_max)
# y_alltime = drop_nan_lag_rows_from_y(y_alltime, n_lags_max)

# if X_alltime_lagged.shape[0] != y_alltime.shape[0]:
#     raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")

# #%%
# lag_level_here = 5  # example lag level to test

# if X_alltime_lagged.shape[1] != n_lags_max*n_units:
#     raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.") 

# X_at_laglevel = X_alltime_lagged[:,:lag_level_here*n_units]

# #%%
# X_at_laglevel.shape

# #%%
# X_at_laglevel.shape[1] // n_units

# # %%
