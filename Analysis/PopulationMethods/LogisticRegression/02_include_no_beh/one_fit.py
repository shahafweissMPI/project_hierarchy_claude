# -*- coding: utf-8 -*-
"""
Created on 2025-08-08

@author: Dylan Festa

Applies logistic regression with regularizers that promote sparsity
(elasticnet) Single fit, measure speed when using cuML and RAPIDS on the GPU

Here I also include a label for "no behaviour", corresponding to time intervals before
the pups are introduced, while the mouse is simply moving around (exploring?)

"""
#%%
%load_ext cuml.accel
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

# import stuff from sklear: pipeline, lagged data, Z-score, PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer



#%%


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


#%% add a "loiter" behavior, in the form of 
# k randomly arranged, non superimposing 10 second intervals
# after the first 5 min but before the 5 min preceding the first labeled behavior


t_first_behav = behaviour['frames_s'].min()
t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

#%%

t_loiter_start = 5 * 60.0  # 5 minutes after the start of the session
t_loiter_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

k_intervals = 18  # number of intervals to generate
t_interval_duration = 10.0  # duration of each interval in seconds

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
dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# remove 'loom'
dict_behavior_label_to_index.pop('loom', None)
# add 'none'
dict_behavior_label_to_index['none'] = -1
# pup grab and pup retrieve should have same label
dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']

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
import plotly.graph_objects as go


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


#%%
# these are fixed
n_lags = 29
dt = 10*1E-3  # 10 ms
l1_ratio = 1.0
C = 0.02


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
X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion_xr(X_alltime, n_lags)
# drop NaN rows from X and y
X_alltime_maxlagged = X_alltime_maxlagged[n_lags:,:]  # keep only the last n_lags_max rows
y_alltime = y_alltime[n_lags:]  # keep only the last n_lags_max rows
if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
    raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")
t_alltime = X_alltime_maxlagged.coords['time_bin_center'].values
# now, select only behavioural labels
y_idx_behavior = y_alltime != -1  # -1 is the 'none' label
y_behaviour = y_alltime[y_idx_behavior]
X_behaviour_maxlagged = X_alltime_maxlagged.values[y_idx_behavior, :]


units_fit = X_alltime.coords['neuron'].values


#%% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_behaviour_maxlagged, y_behaviour, test_size=0.2, random_state=1, stratify=y_behaviour
)



#%% Data is ready! Now build the pipeline and apply the grid search
pipe= Pipeline([
    ('regularizer',StandardScaler()),
    ('lda', LogisticRegression(
                    penalty='elasticnet', 
                    solver='saga',class_weight='balanced',
                    verbose=True,
                    n_jobs=-1,
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=5000,
                    tol=1e-4,
                    random_state=0,
                    )),])

time_start_onefit = time.time()
print("Starting logistic regression...")
pipe.fit(X_train, y_train)
time_end_onefit = time.time()
time_onefit_string = time.strftime("%H:%M:%S", time.gmtime(time_end_onefit - time_start_onefit))
print(f"Logistic regression completed in: {time_onefit_string}")

#%% Check performance on test set
from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipe.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))



#%% Dataframe of best 10 units for each label

lda = pipe.named_steps['lda']
rows_best_units = []
for beh_idx,beh_name in dict_classindex_to_behaviour.items():
    # avoid behaviours not used for training
    if beh_idx not in lda.classes_:
        continue
    # get all coefficients for the behavior, and reshape them
    coefficients_beh = lda.coef_[lda.classes_ == beh_idx]
    coefficients_beh_reshaped = coefficients_beh.reshape(n_lags+1,n_units)
    lag_seconds = np.arange(n_lags+1) * dt
    coefficients_xr = xr.DataArray(
        coefficients_beh_reshaped,
        coords=[lag_seconds, units_fit],
        dims=["lag", "unit"],
    )
    # get the units with the top sums for each behavior
    summed_units_xr = np.abs(coefficients_xr).sum(dim='lag')  # (unit,)
    # sort by value desc and pick top-10
    top10 = summed_units_xr.sortby(summed_units_xr, ascending=False).isel(unit=slice(0, 10))
    top_units = np.sort(top10['unit'].values)
    for unit in top_units:
        # get unit coefficients (length n_lags+1)
        unit_coefficients = coefficients_xr.sel(unit=unit)
        # sum of coefficients (consistent with ranking)
        coef_sum = unit_coefficients.values.sum()
        rows_best_units.append({
            'unit': unit,
            'behaviour_name': beh_name,
            'behaviour_idx': int(beh_idx),
            'coefficients': unit_coefficients.values,
            'coefficients_lag': unit_coefficients.coords['lag'].values,
            'summed_coefficients': coef_sum,
        })

best10_units_per_behaviour_df = (
    pd.DataFrame(rows_best_units)
      .sort_values(['behaviour_idx', 'summed_coefficients'], ascending=[True, False])
      .reset_index(drop=True)
)

#%%
# count how many units are repeated more than once, how many are unique
unit_counts = best10_units_per_behaviour_df['unit'].value_counts()
n_repeated_units = (unit_counts > 1).sum()
n_unique_units = (unit_counts == 1).sum()
print(f"Number of repeated units: {n_repeated_units}")
print(f"Number of unique units: {n_unique_units}")


#%%
# get probability estimate for each label for the full dataset
y_predprob_full_1 = pipe.predict_proba(X_alltime_maxlagged[:100000,:].values)
y_predprob_full_2 = pipe.predict_proba(X_alltime_maxlagged[100000:200000,:].values)
y_predprob_full_3 = pipe.predict_proba(X_alltime_maxlagged[200000:300000,:].values)
y_predprob_full_4 = pipe.predict_proba(X_alltime_maxlagged[300000:,:].values)

y_predprob_full = np.concatenate([y_predprob_full_1, y_predprob_full_2, y_predprob_full_3, y_predprob_full_4], axis=0)

## turn it into data array, coordinates are time and label index
# adds as an attribute the dictionary that matches class and behavior labels
predictions_xr = xr.DataArray(
        y_predprob_full,
        coords=[t_alltime, pipe.classes_],
        dims=["time", "label"],
        attrs={
            "description": "Predicted probabilities for each behavior label",
            "units": "probability",
            "class_labels_dict": dict_classindex_to_behaviour
        }
    )


#%%
dict_save = {
    "best10_units_per_behaviour": best10_units_per_behaviour_df,
    "predictions": predictions_xr,
    "behaviour_timestamps": behaviour_timestamps_df,
    "index_to_behaviour": dict_behavior_label_to_index,
    "behaviour_representation_df": behaviour_representation_df,
}

# savename with mouse and session
saveaname = f"{animal}_{session}_one_fit.pkl"

with open(saveaname, "wb") as f:
    pickle.dump(dict_save, f)

#%%
# exit the script
exit()

#######################################
# Older plot stuff for single behaviors
#%% Make a nice heatmap of the confusion matrix, with numbers in each cell
# using ploty express
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=beh_plot_dict.keys(), columns=beh_plot_dict.keys())

fig = px.imshow(conf_matrix_df,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=conf_matrix_df.columns,
                y=conf_matrix_df.index,
                text_auto=True,
                color_continuous_scale='Blues',
                title="Confusion Matrix Heatmap")
fig.update_layout(
    xaxis_title="Predicted Behaviour",
    yaxis_title="True Behaviour",
    title_x=0.5
)
#fig.show(renderer="browser")
#fig.show(renderer="vscode")
fig.show()

#%% Now again, but using percentages in the confusion matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
conf_matrix_normalized_rounded = np.round(conf_matrix_normalized, 2)
conf_matrix_normalized_df = pd.DataFrame(conf_matrix_normalized_rounded, index=beh_plot_dict.keys(), columns=beh_plot_dict.keys())

fig_norm = px.imshow(conf_matrix_normalized_df,
                     labels=dict(x="Predicted Label", y="True Label", color="Percentage"),
                     x=conf_matrix_normalized_df.columns,
                     y=conf_matrix_normalized_df.index,
                     text_auto=True,
                     color_continuous_scale='Blues',
                     title="Normalized Confusion Matrix Heatmap")
fig_norm.update_layout(
    xaxis_title="Predicted Behaviour",
    yaxis_title="True Behaviour",
    title_x=0.5
)
#fig_norm.show(renderer="browser")
#fig_norm.show(renderer="vscode")
fig_norm.show()




# %% 
# Now, take a look at the coefficients of the logistic regression model
lda = pipe.named_steps['lda']
coefficients_all = lda.coef_

coefficients_pup_run = coefficients_all[lda.classes_ == dict_behavior_label_to_index['pup_run']]

# %%

coefficients_pup_run_reshaped = coefficients_pup_run.reshape(n_lags+1,n_units)
lag_seconds = np.arange(n_lags+1) * dt
# Print as heatmap
fig_heatmap_pup_run = px.imshow(
    coefficients_pup_run_reshaped,
    labels=dict(x="units", y="bin lag (s)", color="LDA coefficient"),
    y=lag_seconds,  # Use numerical labels for y-axis
    x=[f"unit {u}" for u in units_fit],
    title="LDA Coefficients for 'pup_run' Behavior",
    color_continuous_scale='Oxy',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

fig_heatmap_pup_run.show(renderer="vscode")
#fig_heatmap_pup_run.show()

#%%

pup_run_coefs_summary = np.sum(np.abs(coefficients_pup_run_reshaped), axis=0)
# histogram of coefficients
fig_hist = px.histogram(
    pup_run_coefs_summary,
    labels={'value': 'Sum of absolute coefficients'},
    title='Sum of absolute coefficients for each unit in "pup_run" behavior',
    nbins=50,
    color_discrete_sequence=['#636EFA']  # Use a single color for the histogram
)
fig_hist.update_layout(
    xaxis_title='Sum of absolute coefficients',
    yaxis_title='Count',
    bargap=0.2
)

#fig_hist.show(renderer="vscode")
fig_hist.show()


#%%
# same for loiter
coefficients_loiter = coefficients_all[lda.classes_ == dict_behavior_label_to_index['loiter']]
coefficients_loiter_reshaped = coefficients_loiter.reshape(n_lags+1,n_units)
lag_seconds = np.arange(n_lags+1) * dt
# Print as heatmap
fig_heatmap_loiter = px.imshow(
    coefficients_loiter_reshaped,
    labels=dict(x="units", y="bin lag (s)", color="LDA coefficient"),
    y=lag_seconds,  # Use numerical labels for y-axis
    x=[f"unit {u}" for u in units_fit],
    title="LDA Coefficients for 'loiter' Behavior",
    color_continuous_scale='Oxy',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

fig_heatmap_loiter.show(renderer="vscode")
#fig_heatmap_loiter.show()

#%%

loiter_coefs_summary = np.sum(np.abs(coefficients_loiter_reshaped), axis=0)
# histogram of coefficients
fig_hist = px.histogram(
    loiter_coefs_summary,
    labels={'value': 'Sum of absolute coefficients'},
    title='Sum of absolute coefficients for each unit in "loiter" behavior',
    nbins=50,
    color_discrete_sequence=['#636EFA']  # Use a single color for the histogram
)
fig_hist.update_layout(
    xaxis_title='Sum of absolute coefficients',
    yaxis_title='Count',
    bargap=0.2
)

#fig_hist.show(renderer="vscode")
fig_hist.show()


# %%
