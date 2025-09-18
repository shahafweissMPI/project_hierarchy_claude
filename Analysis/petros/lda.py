# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:05:43 2025

@author: chalasp
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
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

#%%

#Parameters
animal = 'afm16924'
session = '240526'

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
 _] = hf.load_preprocessed(animal, session)

#%%
#pag_ind = np.where((n_region_index=='DMPAG') | (n_region_index=='DLPAG') | (n_region_index=='LPAG'))[0]

#remove firings above 200Hz
isi_array = []
mask = []
for i in range(len(n_spike_times)):
    isi_array.append(np.diff(np.append(0, n_spike_times[i])) if np.all(n_spike_times[i])>0 else np.nan)
    mask.append(1 / isi_array[i] <= 200)

isi_array = []
firing_rates=[]
spike_times=[]
for i in range(len(n_spike_times)):
    if n_spike_times[i].any() < 1:
        isi_array.append(0)
        firing_rates.append(0)
        spike_times.append(np.nan)
    else:
        isi_array.append(np.diff(np.append(0, np.array(n_spike_times[i][mask[i]]))))
        firing_rates.append(1/isi_array[i] if np.all(n_spike_times[i])>0 else np.nan)
        spike_times.append(n_spike_times[i][mask[i]])

#%% Mean firing rates

def mean_firing_rate(n_spike_times, firing_rates, stop_time, start_time=0, bin_size_ms=10):
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
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid indices

        recast_spike_times.append(bin_centers[bin_indices])  # Recast spike times to bin centers
        
        # Use np.bincount to compute sum of firing rates in each bin
        bin_sums = np.bincount(bin_indices, weights=firing_rate, minlength=num_bins)
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

        
binned_firing_rates, mean_rates, binned_ifr,  mean_rates_ifr, time_bins, recast_spike_times = mean_firing_rate(spike_times, firing_rates, stop_time=n_time_index[-1]) #whole session
binned_baseline_rates, mean_baseline, _, _,time_bins_baseline,_ = mean_firing_rate(spike_times, firing_rates, stop_time=420) #baseline

#%%
from sklearn.model_selection import train_test_split


def export_behaviour_timestamps(behaviour):
    behavior_events = {b: {} for b in behaviour['behaviours'].unique()}
    index = 0
    for behavior in behavior_events.keys():
        behavior_rows = behaviour[behaviour['behaviours'] == behavior]
        starts = behavior_rows[behavior_rows['start_stop'] == 'START']['frames_s'].values
        stops = behavior_rows[behavior_rows['start_stop'] == 'STOP']['frames_s'].values
        behavior_events[behavior]['start_stop'] = [(s, e) for s, e in zip(starts, stops) if e > s]
        behavior_events[behavior]['index'] = index
        index += 1
    return behavior_events


def create_label_vector(behaviour, time_bins):
    original_bin_size = time_bins[1]  # assume uniform spacing
    behavior_events = export_behaviour_timestamps(behaviour)
    index = -1 * np.ones(len(time_bins))  # -1 means "no label"

    for key in behavior_events:
        if not behavior_events[key]['start_stop']:
            continue
        class_label = behavior_events[key]['index']
        for start, end in behavior_events[key]['start_stop']:
            start_idx = int(np.floor(start / original_bin_size))
            end_idx = int(np.ceil(end / original_bin_size))
            index[start_idx:end_idx] = class_label

    return index


def export_design_matrix(binned_firing_rates, n_region_index, time_bins, n_cluster_index,
                         gaussian=False, pag=False, sigma=1, bin_size=0.01):
    from scipy.ndimage import gaussian_filter1d
    original_bin_size = time_bins[1]  # in seconds

    # 1. Select regions
    if pag:
        pag_mask = (n_region_index == 'DMPAG') | (n_region_index == 'DLPAG') | (n_region_index == 'LPAG')
        pag_ind = np.where(pag_mask)[0]
        selected_firing_rates = binned_firing_rates[pag_ind]
        selected_cluster_ids = n_cluster_index[pag_ind]  # Keep only PAG cluster IDs
    else:
        selected_firing_rates = binned_firing_rates
        selected_cluster_ids = n_cluster_index

    # 2. Apply Gaussian smoothing (optional)
    if gaussian:
        smoothed = gaussian_filter1d(selected_firing_rates, sigma=sigma, axis=1)
    else:
        smoothed = selected_firing_rates

    # 3. Downsample based on bin size
    factor = int(round(bin_size / original_bin_size))
    if factor > 1:
        n_neurons, n_time = smoothed.shape
        n_down = n_time // factor
        smoothed = smoothed[:, :n_down * factor]  # trim to fit
        smoothed = smoothed.reshape(n_neurons, n_down, factor).mean(axis=2)

    design_matrix = smoothed.T  # shape: (n_timepoints, n_neurons)
    return design_matrix, selected_cluster_ids


def create_sliding_window_data(X, y, window_size, stride):
    X_windows = []
    y_windows = []

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        window_X = X[start:end].flatten()
        window_y = y[start:end]

        # Skip if any undefined labels (-1) are in the window
        if -1 in window_y:
            continue

        # Assign label: most frequent in window
        label = np.bincount(window_y.astype(int)).argmax()

        X_windows.append(window_X)
        y_windows.append(label)

    return np.array(X_windows), np.array(y_windows)


# Determine original bin size
original_bin_size = time_bins[1]  # 0.01 for 10ms bins

X, selected_clusters = export_design_matrix(binned_firing_rates, n_region_index, time_bins, n_cluster_index, gaussian=True, pag=True, sigma=1, bin_size=original_bin_size)
y = create_label_vector(behaviour, time_bins)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Sliding window parameters
bin_size = time_bins[1]  # 0.01 by default (10 ms bins)
window_size_sec = 3.0  # 500 ms window
stride_sec = 0.2      # 500 ms stride

window_size = int(window_size_sec / bin_size)
stride = int(stride_sec / bin_size)

# Step 2: Filter invalid time points
y = y[:len(X)]  # ensure y and X match in length
valid_idx = y != -1
X = X[valid_idx]
y = y[valid_idx]

# Step 3: Create windowed data
X_windows, y_windows = create_sliding_window_data(X, y, window_size, stride)

# Step 4: Train/test LDA
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, stratify=y_windows, test_size=0.20, random_state=42)
clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
clf.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", clf.score(X_test, y_test))


#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Create mapping from index to behavior label
behavior_events = export_behaviour_timestamps(behaviour)
index_to_label = {v['index']: k for k, v in behavior_events.items()}

# Predict and transform
y_pred = clf.predict(X_test)
X_test_lda = clf.transform(X_test)

# Map indices to behavior names
y_test_labels = np.array([index_to_label[int(i)] for i in y_test])
y_pred_labels = np.array([index_to_label[int(i)] for i in y_pred])
unique_labels = np.unique(y_test_labels)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(8, 6))

# Generate high-contrast color palette
num_labels = len(np.unique(y_test_labels))
palette = sns.color_palette("tab10")  # use tab10 as base

# If more labels than base palette, extend using husl or tab20
if num_labels > len(palette):
    palette = sns.color_palette("tab20", num_labels)

label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

if X_test_lda.shape[1] == 1:
    for label in unique_labels:
        plt.hist(X_test_lda[y_test_labels == label],
                 label=label, bins=30, alpha=0.6, color=label_to_color[label])
    plt.xlabel("LDA Component 1")
    plt.title("LDA Projection (1D)")
else:
    for label in unique_labels:
        plt.scatter(X_test_lda[y_test_labels == label, 0],
                    X_test_lda[y_test_labels == label, 1],
                    label=label, alpha=0.7, color=label_to_color[label])
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.title("LDA Projection (2D)")

plt.legend(title="Class")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot confusion matrix using string labels
ConfusionMatrixDisplay.from_predictions(
    y_test_labels, y_pred_labels,
    display_labels=unique_labels,
    xticks_rotation=45
)
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()

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


lda_weights = clf.coef_[0]  # shape: (window_size * n_neurons,)

# Get correct shape from actual training data
n_features = X_windows.shape[1]
n_neurons = X.shape[1]
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
