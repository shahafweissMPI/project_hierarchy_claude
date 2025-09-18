# -*- coding: utf-8 -*-
"""
Created on 2025-08-01

@author: Dylan Festa

Applies logistic regression with regularizers that promote sparsity
(elasticnet)

WARNING: running this code takes 11 hours and 15 minutes!
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
from sklearn.linear_model import LogisticRegressionCV

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer

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
# these are fixed
n_lags = 9
dt = 20*1E-3  # 10 ms

# and parameters for the pipeline
C_for_pipeline= np.logspace(-3, 3, 15)  # regularization strength
l1_ratios_for_pipeline = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
cv_for_pipeline = StratifiedKFold(n_splits=5, shuffle=False)

#%%

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
# apply lag
X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion(X_alltime.values, n_lags)
# drop NaN rows from X and y
X_alltime_maxlagged = X_alltime_maxlagged[n_lags:,:]  # keep only the last n_lags_max rows
y_alltime = y_alltime[n_lags:]  # keep only the last n_lags_max rows
if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
    raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")
# now, select only behavioural labels
y_idx_behavior = y_alltime != -1  # -1 is the 'none' label
y_behaviour = y_alltime[y_idx_behavior]
X_behaviour_maxlagged = X_alltime_maxlagged[y_idx_behavior, :]


#%% Data is ready! Now build the pipeline and apply the grid search
pipe= Pipeline([
    ('regularizer',StandardScaler()),
    ('lda', LogisticRegressionCV(
                    penalty='elasticnet', 
                    solver='saga',class_weight='balanced',
                    refit=True,verbose=True,
                    n_jobs=-1,
                    l1_ratios=l1_ratios_for_pipeline,
                    Cs=C_for_pipeline,
                    scoring='neg_log_loss',
                    max_iter=5000,
                    tol=1e-4,
                    random_state=0,
                    cv=cv_for_pipeline )),])

time_start_gridsearch = time.time()
print("Starting grid search...")

pipe.fit(X_behaviour_maxlagged, y_behaviour)

time_end_gridsearch = time.time()
time_gridsearch_string = time.strftime("%H:%M:%S", time.gmtime(time_end_gridsearch - time_start_gridsearch))
print(f"Grid search completed in: {time_gridsearch_string}")


#%% heatmap of score, with li_ratios and C values lables as axes

C_for_pipeline_str = [f"C={c:.3f}" for c in C_for_pipeline]
l1_ratios_for_pipeline_str = [f"l1_ratio={l1:.2f}" for l1 in l1_ratios_for_pipeline]


#%% xarray.DataArray to hold the scores

# # Prepare a 2D array to hold the best scores for each (l1_ratio, C) combination
score_matrix = np.full((len(C_for_pipeline), len(l1_ratios_for_pipeline)), np.nan)

lda_cv = pipe.named_steps['lda']

#%%

lda_cv.scores_[1].shape

#%%
len(C_for_pipeline_str)

#%%
len(l1_ratios_for_pipeline_str)

#%%

# LogisticRegressionCV stores scores in lda_cv.scores_ (dict: class -> ndarray)
# We'll average across classes for multiclass, or use the single class for binary
scores_dict = lda_cv.scores_
if len(scores_dict) == 1:
    # Binary classification: get the only class
    scores_arr = list(scores_dict.values())[0]  # shape: (n_folds,n_l1_ratios, n_Cs)
    mean_scores = np.mean(scores_arr, axis=0)   # mean over folds
else:
    # Multiclass: average over classes
    scores_arrs = [np.mean(arr, axis=0) for arr in scores_dict.values()]
    mean_scores = np.mean(scores_arrs, axis=0)  # shape: (n_Cs, n_l1_ratios)

# Fill score_matrix with mean_scores
score_matrix[:, :] = mean_scores

#%%

import plotly.express as px

fig = px.imshow(
    score_matrix,
    y=C_for_pipeline_str,
    x=l1_ratios_for_pipeline_str,
    color_continuous_scale='Viridis',
    labels={'x': 'C values', 'y': 'L1 ratio', 'color': 'Score'},
    aspect='equal',
    height= 900,
)
fig.update_layout(title="Grid Search Accuracy Heatmap")
fig.show()
# %%
