# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:07:01 2025

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
session = '240525'

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
#delta change
import numpy as np
from numba import njit

# ----- Numba-accelerated parts -----

@njit
def compute_delta(pre_rate, during_rate, eps=1e-10):
    return ((during_rate - pre_rate))#*100 / (during_rate + pre_rate + eps))

@njit
def generate_random_deltas(firing_rate_array, num_trials, duration_bins, t_max_bins, eps=1e-10):
    random_deltas = []
    max_idx = len(firing_rate_array)
    attempts = 0
    max_attempts = num_trials * 10  # Prevent infinite loop

    while len(random_deltas) < num_trials and attempts < max_attempts:
        attempts += 1
        rand_start_bin = np.random.randint(0, int(t_max_bins))
        rand_mid_bin = rand_start_bin + duration_bins
        rand_stop_bin = rand_mid_bin + duration_bins

        if rand_stop_bin >= max_idx:
            continue

        # Skip windows with NaNs
        if np.any(np.isnan(firing_rate_array[rand_start_bin:rand_stop_bin])):
            continue

        pre_rate = np.nanmean(firing_rate_array[rand_start_bin:rand_mid_bin])
        during_rate = np.nanmean(firing_rate_array[rand_mid_bin:rand_stop_bin])

        delta = compute_delta(pre_rate, during_rate, eps)
        random_deltas.append(delta)

    return random_deltas

# ----- Main function -----

def extract_delta_change_with_random_baseline(
    firing_rates, spike_times, behaviour_df,
    n_cluster_index, n_region_index,
    bin_size=0.01,
    num_random_trials=1000, p_value_threshold=0.01,
    random_time_limit_s=None,
    min_behavior_duration=0.5
):
    eps = 1e-10
    allowed_regions = ["DMPAG", "DLPAG", "LPAG"]

    # Parse behavior events 
    behavior_events = {b: [] for b in behaviour_df['behaviours'].unique()}
    for behavior in behavior_events.keys():
        rows = behaviour_df[behaviour_df['behaviours'] == behavior]
        starts = rows[rows['start_stop'] == 'START']['frames_s'].values
        stops = rows[rows['start_stop'] == 'STOP']['frames_s'].values
        behavior_events[behavior] = [(s, e) for s, e in zip(starts, stops) if e > s]

    results = {b: {} for b in behavior_events.keys()}

    for i, neuron_firing in enumerate(firing_rates):
        neuron_region = n_region_index[i]
        if neuron_region not in allowed_regions:
            continue

        neuron_label = str(n_cluster_index[i])
        neuron_duration_s = len(neuron_firing) * bin_size
        max_random_time = min(random_time_limit_s if random_time_limit_s else neuron_duration_s, neuron_duration_s)

        for behavior, windows in behavior_events.items():
            real_deltas = []
            random_deltas = []

            for start, stop in windows:
                duration = stop - start
                if duration < min_behavior_duration:
                    continue

                pre_start = start - duration
                prior_behaviors = [(s, e) for s, e in windows if e <= start]
                if prior_behaviors:
                    _, last_prior_end = prior_behaviors[-1]
                    if last_prior_end > pre_start:
                        pre_start = last_prior_end
                if pre_start < 0:
                    continue

                pre_start_idx = int(pre_start / bin_size)
                pre_stop_idx = int(start / bin_size)
                start_idx = int(start / bin_size)
                stop_idx = int(stop / bin_size)

                if stop_idx > len(neuron_firing):
                    continue

                pre_rate = np.nanmean(neuron_firing[pre_start_idx:pre_stop_idx])
                during_rate = np.nanmean(neuron_firing[start_idx:stop_idx])
                if np.isnan(pre_rate) or np.isnan(during_rate):
                    continue

                delta = compute_delta(pre_rate, during_rate, eps)
                real_deltas.append(delta)

                duration_bins = int(duration / bin_size)
                t_max_bins = (max_random_time - 2 * duration) / bin_size
                if t_max_bins > 0:
                    rand_deltas = generate_random_deltas(
                        neuron_firing, num_random_trials, duration_bins, t_max_bins, eps
                    )
                    random_deltas.extend(rand_deltas)

            if len(real_deltas) >= 5 and len(random_deltas) >= 5:
                real_mean = np.nanmean(real_deltas)
                null_dist = np.array(random_deltas)
  
                # One-tailed test (increase in firing)
                #p_val = np.mean(null_dist >= real_mean)
  
                # Optional: two-tailed
                p_val = np.mean(np.abs(null_dist - np.mean(null_dist)) >= np.abs(real_mean - np.mean(null_dist)))
  
                z_score = (real_mean - np.mean(null_dist)) / (np.std(null_dist) + eps)
            else:
                p_val = np.nan
                z_score = np.nan

            if not np.isnan(p_val):
                results[behavior][neuron_label] = {
                    "delta_changes": real_deltas,
                    "random_deltas": random_deltas,
                    "region": neuron_region,
                    "cluster": n_cluster_index[i],
                    "p_value": p_val,
                    "real_mean": real_mean,
                    "random_mean": np.nanmean(random_deltas) if len(random_deltas) > 0 else np.nan,
                    "z_score": z_score,
                }

    return results

# Example call
results = extract_delta_change_with_random_baseline(
    binned_firing_rates,
    recast_spike_times,
    behaviour,
    n_cluster_index,
    n_region_index,
    bin_size=0.01,
    num_random_trials=10000,
    p_value_threshold=0.05,
    random_time_limit_s=420  # 7 minutes
)


#%% Visualisation

def prepare_neuron_dataframe(results_dict):
    """Flatten results into a DataFrame."""
    data = []
    for behavior, neuron_dict in results_dict.items():
        for neuron, vals in neuron_dict.items():
            deltas = vals["delta_changes"]
            mean_delta = vals["real_mean"]
            mean_random = vals['random_mean']
            direction = "increase" if mean_delta > 0 else "decrease"
            data.append({
                "Behavior": behavior,
                "Neuron": neuron,
                "Region": vals["region"],
                "DeltaChanges": deltas,
                "MeanDelta": mean_delta,
                'MeanRandom': mean_random,
                "Direction": direction,
                "p_value": vals["p_value"]
            })
    return pd.DataFrame(data)

def plot_real_responses_per_region_sorted_with_significance(
    results, 
    behaviors_of_interest=None, 
    p_value_threshold=0.05, 
    behavior_color_map=None  # Add parameter for color map
):
    """
    Plot real delta changes per neuron across behaviors, one plot per region,
    sorted by strongest response, adding significance markers.
    """

    # Prepare tidy DataFrame
    data = []
    for behavior, neurons_dict in results.items():
        if behaviors_of_interest and behavior not in behaviors_of_interest:
            continue
        for neuron, neuron_data in neurons_dict.items():
            mean_delta = np.nanmean(neuron_data["delta_changes"])
            region = neuron_data["region"]
            p_value = neuron_data.get("p_value", np.nan)  # grab p_value if available
            data.append({
                "Neuron": neuron,
                "Behavior": behavior,
                "MeanDelta": mean_delta,
                "Region": region,
                "p_value": p_value
            })
    
    if not data:
        print("⚠️ No data found matching the specified behaviors.")
        return
    
    df = pd.DataFrame(data)
    
    # Plot separately for each region
    regions = df["Region"].unique()
    for region in regions:
        region_df = df[df["Region"] == region]
        
        # 🧠 Find the max response across behaviors per neuron
        neuron_max_deltas = (
            region_df.groupby("Neuron")["MeanDelta"]
            .max()
            .sort_values(ascending=False)
        )
        sorted_neurons = neuron_max_deltas.index.tolist()
        
        # Apply sorting to dataframe
        region_df.loc[:, "Neuron"] = pd.Categorical(region_df["Neuron"].astype(str), categories=sorted_neurons, ordered=True)
        region_df = region_df.sort_values("Neuron")

        plt.figure(figsize=(14, 6))
        sns.set(style="whitegrid")

        # Get colors from behavior_color_map, ensuring same color across all plots for each behavior
        if behavior_color_map is not None:
            palette = behavior_color_map  # Pass dict directly
        else:
            unique_behaviors = df["Behavior"].unique()
            palette = dict(zip(unique_behaviors, sns.color_palette("Set2", n_colors=len(unique_behaviors))))
            
        # Determine the full list of behaviors to fix the hue order globally
        if behaviors_of_interest:
            hue_order = behaviors_of_interest
        else:
            hue_order = sorted(df["Behavior"].unique())
        
        # Define the palette
        if behavior_color_map is not None:
            palette = behavior_color_map
        else:
            palette = dict(zip(hue_order, sns.color_palette("Set2", n_colors=len(hue_order))))
        


        # Barplot
        ax = sns.barplot(
            x="Neuron",
            y="MeanDelta",
            hue="Behavior",
            data=region_df,
            dodge=True,
            palette=palette,
            hue_order=hue_order,  # ✅ Fixes inconsistent coloring
            errorbar=None
        )


        # Add asterisks for significant neurons (positioned correctly above each behavior)
        for idx, row in region_df.iterrows():
            if row["p_value"] < p_value_threshold:
                neuron_idx = sorted_neurons.index(row["Neuron"])
                
                behaviors = region_df["Behavior"].unique()
                behavior_idx = list(behaviors).index(row["Behavior"])
                
                total_width = 0.8  # default bar group width
                n_behaviors = len(behaviors)
                bar_width = total_width / n_behaviors
                
                x = neuron_idx - total_width / 2 + bar_width / 2 + behavior_idx * bar_width
        
                # Decide how many asterisks
                if row["p_value"] < 0.001:
                    star = "***"
                elif row["p_value"] < 0.01:
                    star = "**"
                else:  # p < 0.05
                    star = "*"
        
                ax.text(
                    x,
                    row["MeanDelta"] + (2 if row["MeanDelta"] >= 0 else -2),
                    star,
                    ha="center",
                    va="bottom" if row["MeanDelta"] > 0 else "top",
                    fontsize=12,
                    fontweight="bold",
                    color="black"
                )

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"Neural Responses per Behavior in {region} (sorted)", fontsize=16)
        ax.set_ylabel("Mean Δ firing (%)", fontsize=14)
        ax.set_xlabel("Neuron", fontsize=14)
        plt.xticks(rotation=90)
        plt.legend(title="Behavior", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

data_pd = prepare_neuron_dataframe(results)
plot_real_responses_per_region_sorted_with_significance(results, behaviors_of_interest=['chase','escape','pup_retrieve'])

#%%write to csv
data_pd.to_csv(rf'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\analysis\neuron_selection\afm16924\{animal}_{session}_delta_firing.csv')

