# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:05:43 2025

@author: chalasp (edited by Dylan Festa)
"""
import os
os.environ["CUPY_NVRTC_OPTIONS"] = "-std=c++17"
import numpy as np, pandas as pd, xarray as xr
import time
import matplotlib
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plottingFunctions as pf
import helperFunctions as hf
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import pearsonr, spearmanr, wilcoxon
from scipy.stats import expon, kurtosis
from scipy.integrate import quad
import math
import seaborn as sns
import plottingFunctions as pf
from numba import njit
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as cpu_gaussian_filter1d
from cupyx.scipy.ndimage import gaussian_filter1d as gpu_gaussian_filter1d
from scipy.signal import detrend
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
os.environ['CUPY_NVRTC_OPTIONS'] = '--std=c++17'

import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains


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

#%%

isi_array = []
mask = []
for i in range(len(n_spike_times)):
    isi_array.append(np.diff(np.append(0, n_spike_times[i])) if np.all(n_spike_times[i])>0 else np.nan)
    mask_i = ( (1/isi_array[i]) <= 200)
    mask_i[0]= True  # No point in removing first spike, since the first ISI is referred to 0.0
    mask.append(mask_i)

isi_array = []
firing_rates=[]
spike_times_old=[]
for i in range(len(n_spike_times)):
    if n_spike_times[i].any() < 1:
        isi_array.append(0)
        firing_rates.append(0)
        spike_times_old.append(np.nan)
    else:
        isi_array.append(np.diff(np.append(0, np.array(n_spike_times[i][mask[i]]))))
        firing_rates.append(1/isi_array[i] if np.all(n_spike_times[i])>0 else np.nan)
        spike_times_old.append(n_spike_times[i][mask[i]])

#%% spiketimes from cool library function
spiketrains=pre.SpikeTrains.from_spike_list(n_spike_times,
                                units=n_cluster_index,
                                unit_location=n_region_index,
                                isi_minimum=1/200.0, 
                                t_start=0.0,
                                t_stop=n_time_index[-1])
spiketrains_onlypag = spiketrains.filter_by_unit_location('PAG')
iFRs = pre.IFRTrains.from_spiketrains(spiketrains)
iFRs_onlypag = iFRs.filter_by_unit_location('PAG')
#%%
# test: spike_times and spike_times2 should be the same
for (k,_train) in enumerate(spiketrains):
    spiketrain_old = spike_times_old[k]
    assert len(_train) == len(spiketrain_old), f"Spike train lengths differ for neuron {k}: {len(_train)} vs {len(spiketrain_old)}"
    assert np.allclose(_train, spiketrain_old), f"Spike trains differ for neuron {k}"


#%% Mean firing rates

def mean_firing_rate_old(n_spike_times, firing_rates, stop_time, start_time=0, bin_size_ms=10):
    """
    Compute the binned mean firing rate for each neuron, with spike times recast to the nearest time bin.

    Parameters:
    - n_spike_times: list of arrays, each containing spike times for a neuron
    - firing_rates: list of arrays, each containing instantaneous firing rates (1/ISI) for a neuron
    - stop_time: float, end of the analysis window
    - start_time: float, start of the analysis window (default=0)
    - bin_size_ms: int, bin size in milliseconds (default=10ms)

    Returns:
    - binned_firing_rates: 2D array (neurons x bins), mean firing rates per bin
    - mean_rates: array of mean firing rates per neuron (ignoring zero bins)
    - nan_firing_rates: 2D array (neurons x bins) with zeros replaced by NaN
    - mean_rates_ifr: array of mean firing rates per neuron (ignoring NaN bins)
    - time_bins: array of bin edges
    - recast_spike_times: list of arrays, spike times mapped to bin centers
    """
    
    # Convert bin size to seconds
    bin_size = bin_size_ms / 1000  # Convert 10ms to 0.01s
    time_bins = np.arange(start_time, stop_time + bin_size, bin_size)  # Bin edges
    bin_centers = time_bins[:-1] + bin_size / 2  # Bin centers
    num_neurons = len(firing_rates)
    num_bins = len(time_bins)  # Number of bins (edges - 1)

    # Preallocate arrays
    binned_firing_rates = np.zeros((num_neurons, num_bins))
    nan_firing_rates = np.full((num_neurons, num_bins), np.nan)  # Initialize with NaN
    recast_spike_times = []
    
    for n in range(num_neurons):
        spike_times = np.array(n_spike_times[n])
        firing_rate = np.array(firing_rates[n])

        # Select spikes within the time window
        valid_mask = (spike_times >= start_time) & (spike_times < stop_time)
        if not np.any(valid_mask):
            recast_spike_times.append([])
            continue  # Skip neurons with no spikes in the window
        
        spike_times = spike_times[valid_mask]
        firing_rate = firing_rate[valid_mask]
        
        # Assign each spike to its corresponding bin
        bin_indices = np.searchsorted(time_bins, spike_times, side='right') - 1
        # with this, no element can actually be equal to num_bins OR num_bins-1!
        assert np.all(bin_indices < (num_bins-1)), "Oh no, I lost my bet!"
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid indices

        recast_spike_times.append(bin_centers[bin_indices])  # Recast spike times to bin centers
        
        # Use np.bincount to compute sum of firing rates in each bin
        bin_sums = np.bincount(bin_indices, weights=firing_rate, minlength=num_bins)
        # so last bin is always zero
        assert bin_sums[-1] == 0, "Oh no, I lost my bet on the last bin being zero!"
        bin_counts = np.bincount(bin_indices, minlength=num_bins)

        # Compute mean firing rate per bin (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_firing_rates[n, :] = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
        
        # Convert zero values to NaN for mean firing rate calculation
        nan_firing_rates[n, :] = np.where(binned_firing_rates[n, :] == 0, np.nan, binned_firing_rates[n, :])
    
     
    # Compute mean firing rates, ignoring zero and NaN bins
    mean_rates = np.mean(binned_firing_rates, axis=1)
    mean_rates_ifr = np.nanmean(nan_firing_rates, axis=1)
    
    return binned_firing_rates, mean_rates, nan_firing_rates, mean_rates_ifr, time_bins, recast_spike_times


binned_firing_rates_old, mean_rates_old, binned_ifr_old,  mean_rates_ifr_old, time_bins, recast_spike_times =  mean_firing_rate_old(spike_times_old, firing_rates, stop_time=n_time_index[-1], bin_size_ms=100.0) #whole session


# Get binned firing rates using library function

#%% Get binned spike counts using library function

# here the bin_size is in seconds, not milliseconds and is called dt
binned_spike_counts = pre.do_binning_operation(spiketrains,'count',dt=0.1,t_start=0.0,t_stop=n_time_index[-1])
binned_spike_rates = pre.do_binning_operation(spiketrains,'rate',dt=0.1,t_start=0.0,t_stop=n_time_index[-1])
iFRs_binned_mean = pre.do_binning_operation(iFRs,'mean',dt=0.1,t_start=0.0,t_stop=n_time_index[-1])
#%%

time_bins_edgesx,time_bins_centersx=pre.get_bin_edges_and_centers_from_xarray(iFRs_binned_mean)


#%%
assert np.allclose(time_bins[:-1],time_bins_edgesx), "Time bins do not match between custom and library function."

# %% Compare mean iFRs, library vs Petros

# remove last two bins, one because the last edge is incomplete, another 
# because the length in the function above is just wrong
# so the last bin is always zero, and the second to last refers to the incomplete interval
binned_firing_rates_old_compare = np.copy(binned_firing_rates_old[:,:-2].T)

# second issue, the 0.0 added to each spiketrain means the first element is trash
# so for each row, we need to ignore the first non-zero element along the row, and store its index
mask = binned_firing_rates_old_compare != 0.0

# First nonzero index in each row
first_nonzero_index = np.argmax(mask, axis=0)

# Set to -1 if the whole row is zero
all_zero = ~mask.any(axis=0)
first_nonzero_index[all_zero] = -1

# Modify in place
for i, idx in enumerate(first_nonzero_index):
    if idx != -1:
        binned_firing_rates_old_compare[idx,i] = 123.456


iFRs_binned_mean_compare = np.copy(iFRs_binned_mean.values)
# now set some elements to NaN
for k in range(iFRs.n_units):
    idx_bad = first_nonzero_index[k]
    if idx_bad != -1:
        iFRs_binned_mean_compare[idx_bad,k] = 123.456  # set to same recognizable value


assert np.allclose(binned_firing_rates_old_compare,iFRs_binned_mean_compare), "Binned firing rates do not match between custom and library function."


#%% Okay compute again binned iFRs, but only for PAG

iFRs_binned_mean_pag = pre.do_binning_operation(iFRs_onlypag,'mean',dt=0.1,t_start=0.0,t_stop=n_time_index[-1])




#%% LDA with PCA pipeline

def plot_stage(data, title, neuron_idx=0):
    plt.plot(data[neuron_idx])
    plt.title(f"{title} (Neuron {neuron_idx})")
    plt.xlabel("Time")
    plt.ylabel("Firing Rate / Value")
    plt.grid(True)

def export_behaviour_timestamps(behaviour):
    behavior_events = {b: {} for b in behaviour['behaviours'].unique()}
    for idx, behavior in enumerate(behavior_events):
        rows = behaviour[behaviour['behaviours'] == behavior]
        starts = rows[rows['start_stop'] == 'START']['frames_s'].values
        stops = rows[rows['start_stop'] == 'STOP']['frames_s'].values
        behavior_events[behavior]['start_stop'] = [(s, e) for s, e in zip(starts, stops) if e > s]
        behavior_events[behavior]['index'] = idx
    return behavior_events

def create_label_vector(behaviour, time_bins):
    bin_size = time_bins[1]
    behavior_events = export_behaviour_timestamps(behaviour)
    
    index = -1 * np.ones(len(time_bins), dtype=int)
    label_to_index = {}
    current_index = 0

    for key, data in behavior_events.items():
        label_to_index[key] = current_index
        for start, end in data['start_stop']:
            start_idx = int(np.floor(start / bin_size))
            end_idx = int(np.ceil(end / bin_size))
            index[start_idx:end_idx] = current_index
        current_index += 1

    return index, label_to_index

def merge_label_indices(index, label_to_index, merge_dict):
    """
    Remap label indices according to merge_dict, preserving unmerged labels.

    Parameters:
        index: np.ndarray — original label vector (e.g. from create_label_vector)
        label_to_index: dict — mapping from label name to index
        merge_dict: dict — mapping of {new_label_name: [original_label_names]}

    Returns:
        new_index: np.ndarray — new remapped index vector
        new_label_to_index: dict — new label name to new index
    """
    used_original_indices = set()
    new_index = -1 * np.ones_like(index)
    new_label_to_index = {}
    next_new_idx = 0

    # First: assign indices to merged groups
    for new_label, group in merge_dict.items():
        group_indices = [label_to_index[name] for name in group if name in label_to_index]
        new_label_to_index[new_label] = next_new_idx
        used_original_indices.update(group_indices)
        for gi in group_indices:
            new_index[index == gi] = next_new_idx
        next_new_idx += 1

    # Then: assign indices to unmerged labels
    for label, orig_idx in label_to_index.items():
        if orig_idx not in used_original_indices:
            new_label_to_index[label] = next_new_idx
            new_index[index == orig_idx] = next_new_idx
            next_new_idx += 1

    return new_index, new_label_to_index

# HELP! Why average again on a new bin size? Better to set the desired bin size only once!
def export_design_matrix(binned_firing_rates, n_region_index, time_bins, n_cluster_index,
                         gaussian=False, pag=False, sigma=1, bin_size=0.01, use_gpu=True):
    """
    Export design matrix of firing rates (timepoints x neurons), with optional PAG filtering and downsampling,
    and return the mapping of output neuron columns back to original cluster IDs.

    Returns
    -------
    X : ndarray, shape (n_timepoints, n_neurons_selected)
        The firing-rate matrix for analysis.
    selected_clusters : ndarray, shape (n_neurons_selected,)
        The original cluster IDs corresponding to each column of X.
    """
    import numpy as np
    import cupy as cp

    # Move data to GPU if requested
    if use_gpu:
        binned_firing_rates = cp.asarray(binned_firing_rates)
        n_cluster_index    = cp.asarray(n_cluster_index)

    # Determine bin size from provided time_bins
    original_bin_size = time_bins[1]

    # Select PAG regions if requested
    if pag:
        mask = np.isin(n_region_index, ['DMPAG', 'DLPAG', 'LPAG'])
        if use_gpu:
            mask = cp.asarray(mask)
        selected_idx = cp.where(mask)[0] if use_gpu else np.where(mask)[0]
        rates = binned_firing_rates[selected_idx]
        selected_clusters = n_cluster_index[selected_idx]
    else:
        # Keep all neurons
        rates = binned_firing_rates
        selected_clusters = n_cluster_index

    # Downsample in time by averaging over `factor` original bins
    factor = int(round(bin_size / original_bin_size))
    if factor > 1:
        # rates shape: (n_neurons_selected, n_time_bins)
        n_neurons_sel, n_time = rates.shape
        # Truncate to a multiple of factor then reshape and mean
        truncated = rates[:, : (n_time // factor) * factor]
        rates = truncated.reshape((n_neurons_sel, -1, factor)).mean(axis=2)

    # Move back to CPU and transpose to (timepoints, neurons)
    if use_gpu:
        rates = cp.asnumpy(rates)
        selected_clusters = cp.asnumpy(selected_clusters)
    X = rates.T

    return X, selected_clusters

def create_sliding_window_data(X, y, window_size, stride):
    X_windows, y_windows = [], []
    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        window_y = y[start:end]
        
        # Exclude -1s when choosing the majority label
        valid_labels = window_y[window_y != -1]
        if len(valid_labels) > 0:
            label = np.bincount(valid_labels.astype(int)).argmax()
        else:
            label = -1  # If all labels are -1
        
        X_windows.append(X[start:end].flatten())
        y_windows.append(label)
        
    return np.array(X_windows), np.array(y_windows)


# --- Main pipeline ---
original_bin_size = time_bins[1]
X_raw, selected_clusters = export_design_matrix(
    binned_firing_rates_old, n_region_index, time_bins, n_cluster_index,
    pag=True, bin_size=original_bin_size, use_gpu=True
)

plt.figure(figsize=(14, 12))
plt.subplot(5, 1, 1)
plot_stage(X_raw, "Raw Data")

X_smoothed = gpu_gaussian_filter1d(cp.asarray(X_raw), sigma=0.1, axis=1).get()
plt.subplot(5, 1, 2)
plot_stage(X_smoothed, "Gaussian-Smoothed Data")

X_detrend = detrend(X_smoothed)
plt.subplot(5, 1, 3)
plot_stage(X_detrend, "After Detrending")

X_scaled = StandardScaler().fit_transform(X_detrend)
plt.subplot(5, 1, 4)
plot_stage(X_scaled, "After Normalization")

y, label_to_index = create_label_vector(behaviour, time_bins)

plt.show()

#%% Test labels
# to-do: use label_to_index to map labels to integers, and call library function
# then make broader library function that accounts for bin edges! Taking non-null behavior 
# that most occur in each edge, among those selected!

dict_behavior_label_to_index= label_to_index.copy()  # make a copy to avoid modifying the original

dict_behavior_label_to_index['none']=-1  # add a label for no behavior

behaviour_timestamps = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)

time_bins_centers = time_bins[:-1] + original_bin_size / 2  # bin centers

# Create label vector
beh_xarray = rdl.generate_behaviour_labels(
    behaviour_timestamps, 
    time_bins_centers, 
    dict_behavior_label_to_index
)

beh_xarray_inc = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps,
                                        t_start =0.0,t_stop= n_time_index[-1],
                                        dt=original_bin_size,
                                        behaviour_labels_dict=dict_behavior_label_to_index)
                                        
                                                        

#%%

#merging of behavioural classes
# merge_dict = {'hunting': ['approach', 'attack', 'chase', 'eat', 'grab_play', 'pursuit'],
#               'pupping': ['pup_grab', 'pup_run', 'bed_retrieve', 'pup_grab', 'pup_retrieve'],
#               'escape': ['loom', 'run_away', 'escape', 'escape_switch']}
# y, new_label_to_index = merge_label_indices(y1, label_to_index, merge_dict)

y = y[:len(X_scaled)]
# valid_idx = y != -1 
X_ready, y_ready = X_scaled, y

# Sliding window parameters
window_size = int(3.0 / original_bin_size)
stride = int(0.1 / original_bin_size)

# Step 1: Create sliding window data (already includes -1-labeled windows)
X_windows, y_windows = create_sliding_window_data(X_ready, y_ready, window_size, stride)

# Step 2: Fit PCA to *all* windows (including those with -1)
pca = PCA(n_components=100, whiten=False, svd_solver='randomized', random_state=40)
X_pca_all = pca.fit_transform(X_windows)

# check variance explained by PCA components
explained_variance = pca.explained_variance_ratio_.sum()
print("Total explained variance by PCA components: {:.1f}%".format(explained_variance * 100))

#%%

# Step 3: Select only valid labels for LDA
valid_lda_idx = y_windows != -1
X_pca_valid = X_pca_all[valid_lda_idx]
y_valid = y_windows[valid_lda_idx]

# Step 4: Train/test split only on valid data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_valid, y_valid, stratify=y_valid, test_size=0.25, random_state=40
)


#%% Test lag from sktime
from sktime.transformations.series.lag import Lag

lags_test = list(range(3))
small_lag = Lag(lags=lags_test,index_out='extend')
n_samples_test,n_features_test = 5,2
Xlagtest = np.random.rand(n_samples_test, n_features_test)  # Example data
ylagtest = np.random.randint(0, 2, n_samples_test)  # Example binary labels

small_lag.get_tags()

Xlagtest_lagged = small_lag.fit_transform(Xlagtest)
klag_fix = len(lags_test)-1

print("Original shape:", Xlagtest.shape)
print("Lagged shape:", Xlagtest_lagged.shape)

rows_idx_with_nan=[]
for k in range(Xlagtest_lagged.shape[0]):
    if np.any(np.isnan(Xlagtest_lagged[k])):
        rows_idx_with_nan.append(k)

Xfix = Xlagtest_lagged[klag_fix:-klag_fix]
yfix = ylagtest[klag_fix:]

print("Rows with NaN after lagging:", rows_idx_with_nan)

#%%
# Step 5: Fit LDA
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
lda.fit(X_train, y_train)

# Step 6: Evaluate
print("LDA Accuracy:", lda.score(X_test, y_test))

# Optional: Predict for all time windows, including -1-labeled ones (e.g., for visualizations)
y_pred_all = lda.predict(X_pca_all)

# ─── Step 7: Map LDA weights back to original features ─────────────────────────

# A) Get weight vector in original windowed‐feature space
#    lda.coef_ shape: (n_classes, n_components)
#    pca.components_ shape: (n_components, window_size * n_neurons)
W_orig = lda.coef_.dot(pca.components_)  
# now shape = (n_classes, F) where F = window_size * n_neurons_selected

# B) Sum absolute across classes to get a single weight per feature
w_feat = np.sum(np.abs(W_orig), axis=0)   # shape (F,)

# ─── Step 8: Neuron‐level importance ────────────────────────────────────────────

T_sel, N_sel = X_raw.shape   # timepoints × selected_neurons
F = w_feat.size              # should equal window_size * N_sel

# 1. Compute each neuron's total importance
imp_neurons = [
    np.sum(w_feat[n::N_sel])  # sum over that neuron's features across all time‐offsets
    for n in range(N_sel)
]

thr_neuron = np.mean(imp_neurons) + 2*np.std(imp_neurons)
important_idx = [n for n,imp in enumerate(imp_neurons) if imp > thr_neuron]
important_clusters = selected_clusters[important_idx]

print("Important clusters:", important_clusters)

# ─── Step 9: Window‐level importance per neuron ────────────────────────────────

# Precompute window start times in seconds
# windows were built as X_windows = create_sliding_window_data(X_ready, …)
n_windows = X_windows.shape[0]
starts    = np.arange(0, T_sel - window_size + 1, stride)
win_times = starts * original_bin_size   # in seconds

# Compute each window’s per‐feature contribution
# shape (n_windows, F)
contrib = X_windows * w_feat[np.newaxis, :]

# For each important neuron, find which windows exceed 2 SD of its own contributions
neuron_time_periods = {}
for n in important_idx:
    # feature indices for neuron n: t*N_sel + n, for t=0…window_size-1
    feat_inds = np.arange(n, F, N_sel)
    # sum the contributions over those features
    c_n = contrib[:, feat_inds].sum(axis=1)  # shape (n_windows,)
    thr_win = np.mean(c_n) + 2*np.std(c_n)
    idxs = np.where(c_n > thr_win)[0]
    neuron_time_periods[int(selected_clusters[n])] = win_times[idxs]

# Save or inspect:
for clu, times in neuron_time_periods.items():
    print(f"Cluster {clu} important at windows starting (s): {np.round(times,2)}")

#%%
import numpy as np
import pandas as pd

# 1) Find overall importances of each PC in the LDA
#    For multiclass LDA, coef_ has shape (n_classes-1, n_PCs)
pc_importance = np.sum(np.abs(lda.coef_), axis=0)   # sum over discriminants
pc_ranking = np.argsort(pc_importance)[::-1]       # descending

# top K PCs you care about:
K = 10
top_pcs = pc_ranking[:K]
print("Top PCs by LDA weight:", top_pcs, 
      "with weights", pc_importance[top_pcs])

# 2) Project those PCs back into original feature space
#    pca.components_ has shape (n_PCs, n_features)
#    where n_features = n_neurons * window_size
orig_loadings = pca.components_[top_pcs] * pc_importance[top_pcs, None]
# sum across selected PCs to get one combined importance map
combined = np.sum(np.abs(orig_loadings), axis=0)  # length = n_features

# 3) Reshape back to (n_neurons, window_size)
n_neurons = X_raw.shape[1]
window_size = int(3.0 / original_bin_size)
feat_map = combined.reshape(n_neurons, window_size)

# 4a) Which neurons matter most?  Sum across time
neuron_scores = np.sum(feat_map, axis=1)
top_neurons = np.argsort(neuron_scores)[::-1]
print("Top neurons:", selected_clusters[top_neurons[:10]])

# 4b) Which time‐bins matter most? Sum across neurons
time_scores = np.sum(feat_map, axis=0)
top_times = np.argsort(time_scores)[::-1]
print("Top time‐bins (within window):", top_times[:10])

# 5) (Optional) pack into DataFrame for inspection
df = pd.DataFrame({
    'neuron': np.repeat(selected_clusters[np.arange(n_neurons)], window_size),
    'time_bin': np.tile(np.arange(window_size), n_neurons),
    'importance': combined
})
print(df.sort_values('importance', ascending=False).head(20))

#%% Predictors plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif

# --- assume X_pca_valid, y_valid, lda are defined as in your pipeline ---

# 1) Compute the LDA transform
X_lda = lda.transform(X_pca_valid)  # shape (n_samples, n_classes-1)

# 2) Compute univariate feature–class statistics
F_vals, p_vals = f_classif(X_pca_valid, y_valid)
n_classes = len(np.unique(y_valid))
n_samples = len(y_valid)
dfb = n_classes - 1
dfw = n_samples - n_classes
eta2 = F_vals * dfb / (F_vals * dfb + dfw)

feature_stats = pd.DataFrame({
    "feature":   np.arange(X_pca_valid.shape[1]),
    "F":         F_vals,
    "R2 (eta²)": eta2,
    "p-value":   p_vals
})
print(feature_stats.round(4))

# 3) Biplot: samples + predictor loadings
fig, ax = plt.subplots(figsize=(8, 6))
classes = np.unique(y_valid)

# choose a colormap with exactly len(classes) colors
cmap = plt.get_cmap('tab20', len(classes))

# scatter points by class
for i, cls in enumerate(classes):
    idx = (y_valid == cls)
    ax.scatter(
        X_lda[idx, 0], X_lda[idx, 1],
        label=f"class {cls}",
        alpha=0.6,
        c=[cmap(i)]
    )

# draw loading vectors for the first two LDA dimensions
loadings = lda.scalings_[:, :2]
arrow_scale = np.max(np.abs(X_lda)) * 0.4

for i, vec in enumerate(loadings):
    ax.arrow(
        0, 0,
        vec[0] * arrow_scale,
        vec[1] * arrow_scale,
        head_width=0.02 * arrow_scale,
        head_length=0.02 * arrow_scale,
        fc='k', ec='k'
    )
    ax.text(
        vec[0] * arrow_scale * 1.1,
        vec[1] * arrow_scale * 1.1,
        f"PC{i}",
        color='k',
        ha='center', va='center'
    )

ax.set_xlabel("LDA Component 1")
ax.set_ylabel("LDA Component 2")
ax.set_title("LDA Biplot: Classes + Predictor Loadings")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.show()


#%%PCA visualisation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_two_classes_pca(X_pca_all, y_windows, label_to_index, class_names):
    """
    Scatter‐plot PC1 vs PC2 for exactly two behavior classes.
    
    Parameters
    ----------
    X_pca_all : array, shape (n_samples, n_components)
        Your PCA‐transformed window data.
    y_windows : array, shape (n_samples,)
        Integer labels (including -1).
    label_to_index : dict
        Mapping from behavior name to integer label.
    class_names : list of str, length 2
        The two behavior names you want to plot.
    """
    # reverse map
    index_to_label = {v: k for k, v in label_to_index.items()}

    # get integer codes for those two
    class_idxs = [label_to_index[c] for c in class_names]
    
    # mask out everything except those two *and* drop -1 automatically
    mask = np.isin(y_windows, class_idxs)
    X_sel = X_pca_all[mask, :2]     # first two PCs
    y_sel = y_windows[mask]

    # convert back to names for legend
    y_sel_names = [index_to_label[i] for i in y_sel]

    # palette for 2 classes
    palette = sns.color_palette("tab10", 2)
    color_map = {class_names[i]: palette[i] for i in range(2)}

    plt.figure(figsize=(7, 6))
    for cname in class_names:
        pts = X_sel[np.array(y_sel_names) == cname]
        plt.scatter(pts[:, 0], pts[:, 1],
                    label=cname,
                    alpha=0.7,
                    s=50,
                    color=color_map[cname])
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"PCA: PC1 vs PC2 for “{class_names[0]}” vs “{class_names[1]}”")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- example usage ---
# say you want to compare "hunting" vs "escape":
plot_two_classes_pca(X_pca_all, y_windows, label_to_index, ["escape", 'pursuit'])

#%% Shuffled blocks
# Shuffle behavioral blocks
X_shuffled, y_shuffled = shuffle_behavioural_blocks(y_ready, X_ready)

# Visual check: original vs. shuffled label timeline
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 2))
plt.plot(y_ready, label="Original", alpha=0.5)
plt.plot(y_shuffled, label="Shuffled", alpha=0.5)
plt.legend()
plt.title("Label Sequence Before and After Block Shuffling")
plt.tight_layout()
plt.show()

# Sliding window parameters
window_size = int(3.0 / original_bin_size)
stride = int(0.1 / original_bin_size)

X_windows, y_windows = create_sliding_window_data(X_shuffled, y_shuffled, window_size, stride)

# Step 2: Fit PCA to *all* windows (including those with -1)
pca = PCA(n_components=100, whiten=False, svd_solver='randomized', random_state=42)
X_pca_all = pca.fit_transform(X_windows)

# Step 3: Select only valid labels for LDA
valid_lda_idx = y_windows != -1
X_pca_valid = X_pca_all[valid_lda_idx]
y_valid = y_windows[valid_lda_idx]

# Step 4: Train/test split only on valid data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_valid, y_valid, stratify=y_valid, test_size=0.25, random_state=42
)

# Step 5: Fit LDA
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
lda.fit(X_train, y_train)

# Step 6: Evaluate
print("LDA Accuracy:", lda.score(X_test, y_test))


#%%
def plot_lda_and_confusion(pca, lda, X_windows, y_windows, label_to_index):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Reverse mapping
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Transform all windows with PCA
    X_pca_all = pca.transform(X_windows)

    # Select valid label windows
    valid_idx = y_windows != -1
    X_test_lda = lda.transform(X_pca_all[valid_idx])
    y_test = y_windows[valid_idx]
    y_pred = lda.predict(X_pca_all[valid_idx])

    # Convert indices to names
    y_test_labels = np.array([index_to_label.get(int(i), f"unknown_{i}") for i in y_test])
    y_pred_labels = np.array([index_to_label.get(int(i), f"unknown_{i}") for i in y_pred])
    unique_labels = np.unique(np.concatenate([y_test_labels, y_pred_labels]))

    # Color palette
    plt.figure(figsize=(8, 6))
    num_labels = len(unique_labels)
    palette = sns.color_palette("tab10") if num_labels <= 10 else sns.color_palette("tab20", num_labels)
    label_to_color = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    # --- LDA projection plot ---
    if X_test_lda.shape[1] == 1:
        for label in unique_labels:
            plt.hist(X_test_lda[y_test_labels == label],
                     label=label, bins=30, alpha=0.6, color=label_to_color[label], density=True)
        plt.xlabel("LDA Component 1")
        plt.title("LDA Projection (1D)")
    else:
        for label in unique_labels:
            plt.scatter(X_test_lda[y_test_labels == label, 1],
                        X_test_lda[y_test_labels == label, 2],
                        label=label, alpha=0.7, color=label_to_color[label])
        plt.xlabel("LDA Component 1")
        plt.ylabel("LDA Component 2")
        plt.title("LDA Projection (2D)")

    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    ConfusionMatrixDisplay(confusion_matrix(normalize='pred', y_true=y_test_labels, y_pred=y_pred_labels)).from_predictions(
        y_test_labels, y_pred_labels,
        display_labels=unique_labels,
        xticks_rotation=45
    )
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()


plot_lda_and_confusion(pca, lda, X_windows, y_windows, label_to_index)

#%%
# Assuming:
# - window_size = number of time bins in a window
# - n_neurons = number of neurons
# - X_windows.shape[1] = window_size * n_neurons

pag=True

if pag:
    pag_ind = np.where((n_region_index == 'DMPAG') | 
                       (n_region_index == 'DLPAG') | 
                       (n_region_index == 'LPAG'))[0]
    n_neurons = len(pag_ind)
else:
    n_neurons = binned_firing_rates.shape[0]


lda_weights = lda.coef_[0]  # shape: (window_size * n_neurons,)

# Get correct shape from actual training data
n_features = X_windows.shape[1]
n_neurons = X_ready.shape[1]
window_size = n_features // n_neurons

assert window_size * n_neurons == n_features, "Shape mismatch! Double-check how X_windows was created."

# Reshape into (time, neuron)
weight_matrix = lda_weights.reshape(window_size, n_neurons)

# Step 3: Visualize as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(weight_matrix.T, cmap='RdBu_r', center=0,
            xticklabels=int(window_size / 10), yticklabels=5)

plt.xlabel("Time bin within window")
plt.ylabel("Neuron ID")
plt.title("LDA Weights: Time vs Neuron Contribution")
plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def shuffle_behavioural_blocks(y, X, min_block_length=2):
    """
    Shuffles contiguous blocks of identical labels of at least `min_block_length`.
    Blocks can have any label (including -1).
    """
    assert len(y) == len(X), "X and y must have same length"

    blocks = []
    start_idx = 0

    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            # New label started
            block_len = i - start_idx
            if block_len >= min_block_length:
                blocks.append((X[start_idx:i], y[start_idx:i]))
            start_idx = i

    # Handle the last block
    block_len = len(y) - start_idx
    if block_len >= min_block_length:
        blocks.append((X[start_idx:], y[start_idx:]))

    print(f"Found {len(blocks)} valid blocks with len >= {min_block_length}.")

    # Shuffle blocks
    np.random.shuffle(blocks)

    # Flatten back into arrays
    X_shuffled = np.concatenate([block[0] for block in blocks], axis=0)
    y_shuffled = np.concatenate([block[1] for block in blocks], axis=0)

    # Pad if shorter than original
    if len(X_shuffled) < len(X):
        pad_len = len(X) - len(X_shuffled)
        X_pad = np.zeros((pad_len,) + X[0].shape, dtype=X.dtype)
        y_pad = -1 * np.ones(pad_len, dtype=y.dtype)
        X_shuffled = np.concatenate([X_shuffled, X_pad], axis=0)
        y_shuffled = np.concatenate([y_shuffled, y_pad], axis=0)

    return X_shuffled, y_shuffled



def run_permutation_test_advanced(X_input, y_input, pipeline, window_size, stride, n_permutations=100,
                                  method='full', correlation_threshold=0.8, random_state=None,
                                  plot=True, verbose=True):
    
    from sklearn.utils.multiclass import unique_labels

    null_conf_matrices = []

    if random_state is not None:
        np.random.seed(random_state)

    y_input = y_input[:len(X_input)]
    valid_idx = y_input != -1
    X = X_input[valid_idx]
    y = y_input[valid_idx]

    X_win, y_win = create_sliding_window_data(X, y, window_size, stride)

    X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, stratify=y_win, test_size=0.2, random_state=42)

    pca = pipeline.named_steps['pca']
    lda = pipeline.named_steps['lineardiscriminantanalysis']

    X_train_pca = pca.fit_transform(X_train)
    corr_matrix = np.corrcoef(X_train_pca.T)
    keep_idx = ~np.any(np.abs(corr_matrix - np.eye(len(corr_matrix))) > correlation_threshold, axis=0)

    if verbose:
        print(f"[Info] Removed {np.sum(~keep_idx)} correlated PCs > {correlation_threshold:.2f}")

    X_train_decorr = X_train_pca[:, keep_idx]
    X_test_decorr = pca.transform(X_test)[:, keep_idx]

    lda.fit(X_train_decorr, y_train)
    y_pred = lda.predict(X_test_decorr)

    acc_orig = accuracy_score(y_test, y_pred)
    f1_orig = f1_score(y_test, y_pred, average='weighted')
    cm_orig = confusion_matrix(y_test, y_pred)

    null_accuracies = []
    null_f1s = []

    for i in range(n_permutations):
        if method == 'full':
            y_shuffled = np.random.permutation(y_input)
            X_s = X_input
        elif method == 'block':
            X_s, y_shuffled = shuffle_behavioural_blocks(y_input, X_input)
        else:
            raise ValueError("Method must be 'full' or 'block'.")
            
        cm_s = confusion_matrix(y_test, y_pred, labels=unique_labels(y_test, y_pred))
        null_conf_matrices.append(cm_s)

        y_shuffled = y_shuffled[:len(X_s)]
        valid_idx = y_shuffled != -1
        X_valid = X_s[valid_idx]
        y_valid = y_shuffled[valid_idx]

        try:
            X_win_s, y_win_s = create_sliding_window_data(X_valid, y_valid, window_size, stride)

            if len(np.unique(y_win_s)) < 2:
                if verbose:
                    print(f"[Warning] Permutation {i} skipped: only one class present.")
                continue

            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_win_s, y_win_s, stratify=y_win_s, test_size=0.2)
            X_train_s_pca = pca.transform(X_train_s)[:, keep_idx]
            X_test_s_pca = pca.transform(X_test_s)[:, keep_idx]
            lda.fit(X_train_s_pca, y_train_s)
            y_pred_s = lda.predict(X_test_s_pca)

            null_accuracies.append(accuracy_score(y_test_s, y_pred_s))
            null_f1s.append(f1_score(y_test_s, y_pred_s, average='weighted'))

        except Exception as e:
            if verbose:
                print(f"[Warning] Permutation {i} skipped due to error: {e}")
            continue
        
        if null_conf_matrices:
            mean_cm = np.mean(null_conf_matrices, axis=0)
        else:
            mean_cm = None

    null_accuracies = np.array(null_accuracies)
    null_f1s = np.array(null_f1s)

    p_val_acc = (np.sum(null_accuracies >= acc_orig) + 1) / (len(null_accuracies) + 1)
    p_val_f1 = (np.sum(null_f1s >= f1_orig) + 1) / (len(null_f1s) + 1)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(null_accuracies, bins=20, color='gray', alpha=0.7)
        plt.axvline(acc_orig, color='red', linestyle='--', label=f"Original = {acc_orig:.3f}")
        plt.title(f"Accuracy Permutation Test\np = {p_val_acc:.4f}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(null_f1s, bins=20, color='blue', alpha=0.7)
        plt.axvline(f1_orig, color='red', linestyle='--', label=f"Original = {f1_orig:.3f}")
        plt.title(f"F1-score Permutation Test\np = {p_val_f1:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        ConfusionMatrixDisplay(cm_orig).plot(cmap='Blues')
        plt.title("Confusion Matrix (Original Classifier)")
        plt.show()

    if method == 'block':
        plt.plot(y_shuffled, label="Shuffled y")
        plt.title("Shuffled Behavioral Sequence (Block-Wise)")
        plt.xlabel("Time")
        plt.ylabel("Label")
        #plt.legend()
        plt.show()
        
    if mean_cm is not None:
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(mean_cm, annot=True, fmt=".2f", cmap="Purples",
                         xticklabels=np.unique(y_test),
                         yticklabels=np.unique(y_test),
                         cbar_kws={"label": "Mean Count"})
        ax.set_title("Mean Confusion Matrix (Shuffled Classifiers)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.show()

    return {
        'original_accuracy': acc_orig,
        'original_f1': f1_orig,
        'confusion_matrix': cm_orig,
        'null_accuracies': null_accuracies,
        'null_f1s': null_f1s,
        'p_value_accuracy': p_val_acc,
        'p_value_f1': p_val_f1,
        'classes': np.unique(y_test)
    }




pipeline = make_pipeline(
    PCA(n_components=500, svd_solver='randomized', random_state=42),
    LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
)

results = run_permutation_test_advanced(
    X_ready, y_ready,
    window_size=window_size,
    stride=stride,
    pipeline=pipeline,
    n_permutations=100,
    method='block',  # or 'full'
    correlation_threshold=0.9,
    random_state=42,
    plot=True,
    verbose=True
)

#%% Clustering performance
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    fowlkes_mallows_score
)
import pandas as pd

def clustering_vs_labels(X, y_true, k_range=(2, 11)):
    results = []

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        y_pred = kmeans.fit_predict(X)

        results.append({
            "n_clusters": k,
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "Homogeneity": homogeneity_score(y_true, y_pred),
            "Completeness": completeness_score(y_true, y_pred),
            "V-Measure": v_measure_score(y_true, y_pred),
            "Fowlkes-Mallows": fowlkes_mallows_score(y_true, y_pred)
        })

    return pd.DataFrame(results).sort_values(by="ARI", ascending=False).reset_index(drop=True)

metrics_df = clustering_vs_labels(X_test, y_test)
print(metrics_df)


#%%
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

# Function to generate all set partitions into up to max_groups
def partitions(set_):
    if not set_:
        yield []
        return
    first = set_[0]
    for smaller in partitions(set_[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller

def filter_partitions(set_, max_groups=4):
    all_parts = list(partitions(set_))
    return [p for p in all_parts if 2 <= len(p) <= max_groups]

def remap_labels(y, grouping):
    label_map = {label: i for i, group in enumerate(grouping) for label in group}
    return np.array([label_map[l] for l in y])

# Inputs: X_test_lda is your reduced data (n_samples x n_features)
# y_test is your true label vector
# Make sure these are defined in your environment

unique_labels = list(np.unique(y_test))
all_groupings = filter_partitions(unique_labels, max_groups=4)

best_ari = -1
best_grouping = None

for grouping in all_groupings:
    y_grouped = remap_labels(y_test, grouping)
    kmeans = KMeans(n_clusters=len(grouping), random_state=42).fit(X_test)
    score = adjusted_rand_score(y_grouped, kmeans.labels_)
    if score > best_ari:
        best_ari = score
        best_grouping = grouping

print("Best ARI:", best_ari)
print("Best Grouping:", best_grouping)
