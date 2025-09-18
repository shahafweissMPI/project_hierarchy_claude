# -*- coding: utf-8 -*-
"""
Created on 2025-07-28

@author: Dylan Festa

LDA analysis similar to Petros' script, but avoiding PCA intermediate step.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer

#%%Loading of neural data

animal = 'afm16924'
session = '240524'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(animal, session)
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

#%%
# key parameters here: time bin size, and how many steps compose the lag

dt= 10*1E-3  # time bin size in seconds

# selection of number of lag steps. 0 means only present time. So 9 means present + 9 previous time steps, 
# 10 in total, etc.
n_lag_steps = 19  # number of lag steps



print(f"Parameters set: dt = {dt} seconds, n_lag_steps = {n_lag_steps}, total time-window size: {dt * (n_lag_steps+1)} seconds.")

#%%

# use library function
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
# compute iFRs too
iFRs = pre.IFRTrains.from_spiketrains(spiketrains)

#%% Let's compute X

# WARNING: this controls whether the data is spike counts or instantaneous firing rates
# comment out the data type you don't want to use

binneddatax=pre.do_binning_operation(spiketrains,'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)

# Warning: with the current parameters, the iFRs don't work so well. More debugging and inspections needed!
#binneddatax=pre.do_binning_operation(iFRs,'mean',dt=dt,t_start=t_start_all,t_stop=t_stop_all)


#%% Now, process behaviour data to get labels

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)

# keep only behaviours that occur at least 15 times
behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['n_trials'] >= 15]

# now generate a dictionary
dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}

# now use the dictionary to get the labels in the y vector

beh_xarray = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                        t_start =0.0,t_stop= t_stop_all,
                                        dt=dt,
                                        behaviour_labels_dict=dict_behavior_label_to_index)


#%%
# Plot of time labels associated with each behavior
import plotly.graph_objects as go


beh_plot_xy, beh_plot_dict = rdl.behaviour_startstop_df_to_segments(behaviour_timestamps_df)
beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

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
thefig.update_yaxes(range=[0, 8])
    
#thefig.show(renderer="browser")
#thefig.show(renderer="vscode")
thefig.show()

#%%
bin_centers = binneddatax.coords['time_bin_center'].values
bin_edges = binneddatax.attrs['time_bin_edges']
#print(f"feature dimension (i.e. neurons) expanded from {binneddatax.shape[1]} to {lagged_shape[1]} after lag expansion.")

bin_neurons = binneddatax.coords['neuron'].values


#%% Get test and train data, get labels

X_start = binneddatax.values
ystart = beh_xarray.values

# Must apply lag transformation to both X and y before splitting, because
# splitting might destroy time order!
X_start_lagged =  rdl.generate_lag_dimensions_expansion(X_start, n_lag_steps)
ystart_lagged = ystart

# cut off the NaN part of the lagged data
X_start_lagged = X_start_lagged[n_lag_steps:, :]
ystart_lagged = ystart_lagged[n_lag_steps:]


#%% Wait a second: I want only behaviour samples, not samples without beahaviour
idx_behavior_only = np.where(ystart_lagged != dict_behavior_label_to_index['none'])[0]

X_start_lagged_beh = X_start_lagged[idx_behavior_only]
ystart_lagged_beh = ystart_lagged[idx_behavior_only]

#%%


X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_start_lagged_beh, ystart_lagged_beh, stratify=ystart_lagged_beh, test_size=0.5)

print(f"Training is using {X_train_s.shape[0]} samples and {X_train_s.shape[1]} features.")
print(f"Testing is using {X_test_s.shape[0]} samples and {X_test_s.shape[1]} features.")

#%%
pipe = make_pipeline(
    StandardScaler(),
    LDA(solver='svd', n_components=2, shrinkage=None)
    #LDA(solver='eigen', shrinkage='auto')  # try with shrinkage to avoid overfitting
    #LDA(solver='lsqr', shrinkage='auto')  # try with shrinkage to avoid overfitting
)

#%%
# Fit-transform the pipeline
pipe.fit(X_train_s, y_train_s)
y_pred_s = pipe.predict(X_test_s)
#%%
# Get the accuracy
accuracy = np.mean(y_pred_s == y_test_s)
print(f"Accuracy of LDA model: {accuracy:.2f}")
#%% Now, trainsform the behavior data using LDA and do a 2D scatter plot
X_transformed = pipe.transform(X_start_lagged_beh)



#%% Scatter plot of first two PCs colored by behavior label

inv_dict_beh = {v: k for k, v in dict_behavior_label_to_index.items()}
beh_order = sorted(inv_dict_beh.values())
df_plot = pd.DataFrame({
    'LDA1': X_transformed[:, 0], 
    'LDA2': X_transformed[:, 1],
    'Behavior': [inv_dict_beh[label] for label in ystart_lagged_beh]
})

df_plot.sort_values(by='Behavior', inplace=True)

fig = px.scatter(
    df_plot,
    x='LDA1',
    y='LDA2',
    color='Behavior',
    category_orders={'Behavior': beh_order},  # Ensure behaviors are ordered
    title='All data (with possible artifacts)',
    opacity=0.7,
    height=700,  # Add this line to make the figure taller
)
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
#fig.show(renderer="vscode")
fig.show()

#%% Redo, but only for test data

X_test_transformed = pipe.transform(X_test_s)

# Scatter plot of first two LDA components colored by behavior label

df_test_plot = pd.DataFrame({
    'LDA1': X_test_transformed[:, 0],
    'LDA2': X_test_transformed[:, 1],
    'Behavior': [inv_dict_beh[label] for label in y_test_s]
})

# sort by behavior for better visualization
df_test_plot.sort_values(by='Behavior', inplace=True)

fig_test = px.scatter(
    df_test_plot,
    x='LDA1',
    y='LDA2',
    color='Behavior',
    category_orders={'Behavior': beh_order},  # Ensure behaviors are ordered
    title='Honest test data',
    opacity=0.7,
    height=700,  # Add this line to make the figure taller
)
fig_test.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
#fig.show(renderer="vscode")
fig_test.show()
# %%
# get LDA coefficients
lda = pipe.named_steps['lineardiscriminantanalysis']
lda_coefficients = lda.coef_

pup_run_coef_idx = dict_behavior_label_to_index['pup_run'] - 1
pup_run_coefs = lda_coefficients[pup_run_coef_idx]
#%%
# reshape coefficient into n_units x n_lag_steps
pup_run_coefs_reshaped = pup_run_coefs.reshape(n_lag_steps+1,n_units)
print(f"pup_run_coefs_reshaped shape: {pup_run_coefs_reshaped.shape}")

fig_heatmap_pup_run = px.imshow(
    pup_run_coefs_reshaped,
    labels=dict(x="lag steps", y="units", color="LDA coefficient"),
    y=np.arange(n_lag_steps+1),  # Use numerical labels for y-axis
    x=[f"unit {str(neu)}" for neu in bin_neurons],
    title="LDA coefficients for 'pup_run' Behavior",
    color_continuous_scale='Oxy',
    color_continuous_midpoint=0.0,
    height=800  # Add this line to make the figure taller
)

fig_heatmap_pup_run.show()


#%% Trying another measure: sum of absolute values of coefficients

pup_run_coefs_summary = np.sum(np.abs(pup_run_coefs_reshaped), axis=0)

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

# add vertical line at 1.5
#fig_hist.add_vline(x=1.5, line_width=2, line_dash="dash", line_color="red")

#fig_hist.show(renderer="vscode")
fig_hist.show()

#%%

# generate dataframe with neuron and importance
df_pup_run_coefs = pd.DataFrame({
    'neuron': bin_neurons,
    'importance': pup_run_coefs_summary
})

# sort by importance
df_pup_run_coefs.sort_values(by='importance', ascending=False, inplace=True, ignore_index=True)

#np.count_nonzero(pup_run_coefs_summary > 1)

# %%
# Now same, but for pup grab

pup_grab_coef_idx = dict_behavior_label_to_index['pup_grab'] - 1
pup_grab_coefs = lda_coefficients[pup_grab_coef_idx]
pup_grab_coefs_reshaped = pup_grab_coefs.reshape(n_lag_steps+1,n_units)
# Print as heatmap
fig_heatmap_pup_grab = px.imshow(
    pup_grab_coefs_reshaped,
    labels=dict(x="lag steps", y="units", color="LDA coefficient"),
    #y=[f"lag {i}" for i in range(n_lag_steps+1)],
    y=np.arange(n_lag_steps+1),  # Use numerical labels for y-axis
    x=[f"unit {i+1}" for i in range(n_units)],
    title="LDA Coefficients for 'pup_grab' Behavior",
    color_continuous_scale='Oxy',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

#fig_heatmap.show(renderer="vscode")
fig_heatmap_pup_grab.show()

#%%

pup_grab_coefs_summary = np.sum(np.abs(pup_grab_coefs_reshaped), axis=0)
# histogram of coefficients
fig_hist = px.histogram(
    pup_grab_coefs_summary,
    labels={'value': 'Sum of absolute coefficients'},
    title='Sum of absolute coefficients for each unit in "pup_grab" behavior',
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

# generate dataframe with neuron and importance
df_pup_grab_coefs = pd.DataFrame({
    'neuron': bin_neurons,
    'importance': pup_grab_coefs_summary
})

#%%
