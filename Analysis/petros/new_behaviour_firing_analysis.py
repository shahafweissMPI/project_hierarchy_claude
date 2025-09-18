# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:38:48 2025

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
session = '240524'

[_, 
 behaviour, 
 _, 
 n_spike_times,
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
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

from scipy.ndimage import gaussian_filter1d

def mean_firing_rate(
    n_spike_times, firing_rates, stop_time, start_time=0.,
    bin_size_ms=10, smooth=True, smooth_sigma_bins=1
):
    # Convert to seconds
    bin_size = bin_size_ms / 1e3
    # Edges and centers
    time_bins = np.arange(start_time, stop_time + bin_size, bin_size)
    bin_centers = time_bins[:-1] + bin_size / 2
    num_neurons = len(firing_rates)
    num_bins = len(time_bins) - 1

    binned_rates = np.zeros((num_neurons, num_bins))
    nan_rates    = np.full((num_neurons, num_bins), np.nan)
    recast_spikes = []

    for n in range(num_neurons):
        spikes = np.array(n_spike_times[n])
        rates  = np.array(firing_rates[n])

        # Restrict to window
        mask = (spikes >= start_time) & (spikes < stop_time)
        spikes = spikes[mask]
        rates  = rates[mask]

        if spikes.size == 0:
            recast_spikes.append(np.array([]))
            continue

        # 1) Bin index via floor
        idx = np.floor((spikes - start_time) / bin_size).astype(int)
        idx = np.clip(idx, 0, num_bins-1)

        # 2) Spike recast at bin center
        recast_spikes.append(start_time + (idx + 0.5) * bin_size)

        # 3) Weighted rate per bin
        sum_w    = np.bincount(idx, weights=rates, minlength=num_bins)
        count    = np.bincount(idx, minlength=num_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            binned = np.where(count>0, sum_w/count, 0)

        # 4) Optional smoothing
        if smooth:
            binned = gaussian_filter1d(binned, sigma=smooth_sigma_bins)

        binned_rates[n] = binned
        nan_rates[n]    = np.where(binned==0, np.nan, binned)

    mean_rates      = np.mean(binned_rates, axis=1)
    mean_ifr_rates  = np.nanmean(nan_rates, axis=1)

    return (
        binned_rates, mean_rates,
        nan_rates, mean_ifr_rates,
        time_bins, recast_spikes
    )

        
binned_firing_rates, mean_rates, binned_ifr,  mean_rates_ifr, time_bins, recast_spike_times = mean_firing_rate(spike_times, firing_rates, stop_time=n_time_index[-1], bin_size_ms=100, smooth=True, smooth_sigma_bins=0.1) #whole session
binned_baseline_rates, mean_baseline, _, _,time_bins_baseline,_ = mean_firing_rate(spike_times, firing_rates, stop_time=420) #baseline


#%%
# Updated detect_rolling_modulations_fast with requested features:
# - Null distribution computed only from time before first behavior
# - Export all statistical parameters per neuron per behavior
# - Export per-behavior null distributions

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from tqdm.notebook import tqdm
from numba import njit, prange

def get_valid_null_windows(fr_length, bin_size, window_bins, behavior_intervals, exclusion_buffer=1.0):
    """
    Efficiently computes valid null window starts by excluding buffered behavior periods.
    """
    # Step 1: Convert behavior intervals to bin indices and apply buffer
    exclusion_intervals = []
    for start, stop in behavior_intervals:
        bin_start = max(0, int((start - exclusion_buffer) / bin_size))
        bin_stop = min(fr_length, int((stop + exclusion_buffer) / bin_size))
        exclusion_intervals.append((bin_start, bin_stop))

    # Step 2: Merge overlapping exclusion intervals
    if not exclusion_intervals:
        merged = []
    else:
        exclusion_intervals.sort()
        merged = [exclusion_intervals[0]]
        for current in exclusion_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))  # merge
            else:
                merged.append(current)

    # Step 3: Identify valid start indices by walking through gaps
    valid_starts = []
    prev_end = 0
    for start, end in merged:
        # Window must fit in the gap before this exclusion
        for i in range(prev_end, start - window_bins + 1):
            valid_starts.append(i)
        prev_end = end

    # Check final gap after last exclusion
    for i in range(prev_end, fr_length - window_bins + 1):
        valid_starts.append(i)

    return np.array(valid_starts)



@njit(parallel=True)
def compute_null_distributions(fr, starts, window_bins):
    n = len(starts)
    results = np.empty(n)
    for i in prange(n):
        start = starts[i]
        results[i] = np.mean(fr[start:start+window_bins])
    return results

def check_mod_hit(start, stop, bin_size, window_bins, mod_mask, offset=0):
    idx_start = int(start / bin_size) - offset
    idx_stop = int(stop / bin_size) - offset

    # Boundary check
    if idx_start < 0 or idx_stop > len(mod_mask) or idx_start >= idx_stop:
        return 'none'

    window_slice = mod_mask[idx_start:idx_stop]

    if np.any(window_slice == 1) and not np.any(window_slice == -1):
        return 'increase'
    elif np.any(window_slice == -1) and not np.any(window_slice == 1):
        return 'decrease'
    elif np.any(window_slice != 0):
        return 'mixed'
    return 'none'



def detect_rolling_modulations_fast(
    binned_firing_rates,
    behaviour_df,
    n_cluster_index,
    n_region_index=None,
    region_filter=None,
    bin_size=0.1,
    window_s=3.0,
    modulation_percentile=95,
    baseline_samples_per_neuron=1000,
    exclude_behaviors_from_null=True,
    require_consistent_direction=False,
    run_permutation_if_binomial_significant=True,
    n_permutations=1000,
    significance_level=0.05,
    n_jobs=-1
):
    window_bins = int(window_s / bin_size)
    rolling_offset = window_bins // 2  # midpoint of window
    cluster_to_index = {str(label): idx for idx, label in enumerate(n_cluster_index)}
    behavior_events = behaviour_df.copy()
    behavior_list = behavior_events['behaviours'].unique()

    # Gather behavior intervals and determine pre-behavior segment
    behavior_intervals = []
    for _, row in behavior_events.iterrows():
        if row['start_stop'] == 'START':
            current_start = row['frames_s']
        elif row['start_stop'] == 'STOP':
            current_stop = row['frames_s']
            if current_stop > current_start:
                behavior_intervals.append((current_start, current_stop))

    if behavior_intervals:
        first_behavior_start = min([start for start, _ in behavior_intervals])
    else:
        first_behavior_start = float('inf')

    if region_filter is not None and n_region_index is not None:
        region_mask = np.isin(n_region_index, region_filter)
        filtered_neurons = np.array(n_cluster_index)[region_mask]
    else:
        filtered_neurons = n_cluster_index

    def process_neuron(neuron_label):
        
        neuron_idx = cluster_to_index[str(neuron_label)]
        fr = binned_firing_rates[neuron_idx]
        if len(fr) < window_bins:
            return str(neuron_label), {}, {}

        rolling_means, rolling_std, rolling_slope, rolling_entropy, rolling_times = [], [], [], [], []

        for start in range(len(fr) - window_bins + 1):
            window = fr[start:start + window_bins]
            if np.isnan(window).any():
                continue
            rolling_means.append(np.mean(window))
            rolling_std.append(np.std(window))
            rolling_slope.append(np.polyfit(np.arange(window_bins), window, 1)[0])
            hist, _ = np.histogram(window, bins=10, density=True)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log(hist))
            rolling_entropy.append(entropy)
            rolling_times.append((start + window_bins // 2) * bin_size)

        rolling_means = np.array(rolling_means)
        rolling_std = np.array(rolling_std)
        rolling_slope = np.array(rolling_slope)
        rolling_entropy = np.array(rolling_entropy)
        rolling_times = np.array(rolling_times)

        valid_starts = get_valid_null_windows(len(fr), bin_size, window_bins, behavior_intervals, exclusion_buffer=1.0)

        if len(valid_starts) < baseline_samples_per_neuron:
            return str(neuron_label), {}, {}

        sampled_starts = np.random.choice(valid_starts, size=baseline_samples_per_neuron, replace=False)
        null_samples = compute_null_distributions(fr, sampled_starts, window_bins)

        upper_thresh = np.percentile(null_samples, modulation_percentile)
        lower_thresh = np.percentile(null_samples, 100 - modulation_percentile)

        mod_mask = np.zeros_like(rolling_means, dtype=int)
        mod_mask[rolling_means > upper_thresh] = 1
        mod_mask[rolling_means < lower_thresh] = -1

        mod_up = rolling_means > upper_thresh
        mod_down = rolling_means < lower_thresh
        mod_indices_up = np.where(mod_up)[0]
        mod_indices_down = np.where(mod_down)[0]
        all_mod_times = rolling_times[np.concatenate((mod_indices_up, mod_indices_down))]
        all_mod_indices = np.concatenate((mod_indices_up, mod_indices_down))

        neuron_results = {
            '_rolling_features': {
                'time': rolling_times.tolist(),
                'mean': rolling_means.tolist(),
                'std': rolling_std.tolist(),
                'slope': rolling_slope.tolist(),
                'entropy': rolling_entropy.tolist()
            }
        }
        neuron_stats = {}

        for behavior in behavior_list:
            starts = behavior_events[
                (behavior_events['behaviours'] == behavior) & 
                (behavior_events['start_stop'] == 'START')
            ]['frames_s'].values
            stops = behavior_events[
                (behavior_events['behaviours'] == behavior) & 
                (behavior_events['start_stop'] == 'STOP')
            ]['frames_s'].values

            trials = []
            for s, e in zip(starts, stops):
                if e <= s:
                    continue
                if behavior == "pup_grab" and (e - s) > 1.0:
                    trials.append((s, s + 0.5))
                    trials.append((e - 0.5, e))
                else:
                    trials.append((s, e))
            if not trials:
                continue

            trial_hits = []
            trial_directions = []
            
            for start, stop in trials:
                direction = check_mod_hit(start, stop, bin_size, window_bins, mod_mask, offset=rolling_offset)
                if direction and direction != 'none':
                    trial_hits.append(True)
                else:
                    trial_hits.append(False)
                trial_directions.append(direction)

            direction_counts = pd.Series(trial_directions).value_counts()
            if require_consistent_direction:
                if 'increase' in direction_counts and len(direction_counts) == 1:
                    overall_direction = 'increase'
                elif 'decrease' in direction_counts and len(direction_counts) == 1:
                    overall_direction = 'decrease'
                else:
                    overall_direction = 'none'
            else:
                if direction_counts.get('increase', 0) > direction_counts.get('decrease', 0):
                    overall_direction = 'increase'
                elif direction_counts.get('decrease', 0) > direction_counts.get('increase', 0):
                    overall_direction = 'decrease'
                elif direction_counts.get('mixed', 0) > 0:
                    overall_direction = 'mixed'
                else:
                    overall_direction = 'none'

            n_trials = len(trials)
            n_hits = sum(trial_hits)
            fraction_modulated = n_hits / n_trials
            p_null = 1.0 - (1.0 - modulation_percentile / 100.0)

            binom_result = binomtest(n_hits, n_trials, p_null, alternative='two-sided')
            binom_pval = binom_result.pvalue
            binom_std = np.sqrt(p_null * (1 - p_null) / n_trials)
            z_binomial = (fraction_modulated - p_null) / binom_std if binom_std > 0 else 0.0
            binom_z_pval = 2 * (1 - norm.cdf(abs(z_binomial)))

            perm_pval, z_permutation, effect_size = None, None, None
            null_samples_this_behavior = []

            if n_permutations > 0 and (not run_permutation_if_binomial_significant or binom_pval < significance_level):
                permuted_fractions = []
                for _ in range(n_permutations):
                    shuffled_hits = 0
                    for s, e in trials:
                        trial_len = e - s
                        valid_windows = [interval for interval in behavior_intervals if interval[1] - interval[0] >= trial_len]
                        if not valid_windows:
                            continue
                        rand_window = valid_windows[np.random.randint(len(valid_windows))]
                        rand_start = np.random.uniform(rand_window[0], rand_window[1] - trial_len)
                        rand_end = rand_start + trial_len

                        idx_start = int(rand_start / bin_size)
                        idx_end = int(rand_end / bin_size)
                        if idx_end > len(fr):
                            continue
                        null_val = np.mean(fr[idx_start:idx_end])
                        null_samples_this_behavior.append(null_val)

                        hit = any((rolling_times[mod_indices_up] >= rand_start) & (rolling_times[mod_indices_up] <= rand_end)) or \
                              any((rolling_times[mod_indices_down] >= rand_start) & (rolling_times[mod_indices_down] <= rand_end))
                        shuffled_hits += int(hit)

                    permuted_fractions.append(shuffled_hits / n_trials)

                permuted_fractions = np.array(permuted_fractions)
                perm_pval = np.mean(permuted_fractions >= fraction_modulated)
                perm_mean = np.mean(permuted_fractions)
                perm_std = np.std(permuted_fractions, ddof=1)
                if perm_std > 0:
                    z_permutation = (fraction_modulated - perm_mean) / perm_std
                    effect_size = z_permutation
                else:
                    z_permutation = 0.0
                    effect_size = 0.0

            is_significant = perm_pval < significance_level if perm_pval is not None else binom_pval < significance_level

            neuron_results[behavior] = {
                'fraction_modulated_trials': float(fraction_modulated),
                'direction': overall_direction,
                'trial_hits': trial_hits,
                'trial_directions': trial_directions,
                'n_trials': len(trials),
                'rolling_mod_times_s': all_mod_times.tolist(),
                'mod_window_indices': all_mod_indices.tolist(),
                'thresholds': {'upper': float(upper_thresh), 'lower': float(lower_thresh)},
                'binomial_pval': float(binom_pval),
                'binomial_z': float(z_binomial),
                'binomial_z_pval': float(binom_z_pval),
                'permutation_pval': float(perm_pval) if perm_pval is not None else None,
                'permutation_z': float(z_permutation) if z_permutation is not None else None,
                'significant': bool(is_significant),
                'null_distribution': null_samples_this_behavior,
                'effect_size': float(effect_size) if effect_size is not None else None
            }


            neuron_stats[behavior] = {
                'binomial_pval': float(binom_pval),
                'binomial_z': float(z_binomial),
                'binomial_z_pval': float(binom_z_pval),
                'permutation_pval': float(perm_pval) if perm_pval is not None else None,
                'permutation_z': float(z_permutation) if z_permutation is not None else None,
                'effect_size': float(effect_size) if effect_size is not None else None,
                'n_trials': n_trials,
                'n_hits': n_hits,
                'p_null': float(p_null)
            }

        return str(neuron_label), neuron_results, neuron_stats

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_neuron)(neuron_label)
        for neuron_label in tqdm(filtered_neurons, desc="Processing neurons")
    )

    mod_results = {}
    stats_results = {}

    for neuron_id, res, stat in results:
        mod_results[neuron_id] = res
        stats_results[neuron_id] = stat

    return mod_results, stats_results


mod_results, stats_results = detect_rolling_modulations_fast(
    binned_firing_rates,
    behaviour,
    n_cluster_index,
    n_region_index=n_region_index,
    region_filter=['DMPAG', 'DLPAG','LPAG'],
    bin_size=0.1,
    window_s=0.5,
    modulation_percentile=90,
    baseline_samples_per_neuron=1000,
    run_permutation_if_binomial_significant=True,
    n_permutations=1000,
    significance_level=0.05,
    n_jobs=4  # adjust based on your CPU
)


#%%
def get_behavior_intervals(behaviour_df):
    intervals = []
    for behavior, group in behaviour_df.groupby('behaviours'):
        starts = group[group['start_stop']=='START'].index
        stops  = group[group['start_stop']=='STOP'].index
        n_int  = min(len(starts), len(stops))
        for i in range(n_int):
            s_idx, e_idx = starts[i], stops[i]
            s_time = behaviour_df.loc[s_idx].name if behaviour_df.index.dtype!='int64' else s_idx
            e_time = behaviour_df.loc[e_idx].name if behaviour_df.index.dtype!='int64' else e_idx
            intervals.append({'behaviour':behavior,'start':s_time,'stop':e_time})
    return pd.DataFrame(intervals)


def plot_all_neuron_modulations(
    session,
    animal,
    mod_results,
    behaviour,
    binned_firing_rates,
    spike_times,
    n_cluster_index,
    n_region_index,
    region_filter=None,
    output_dir=None,
    bin_size=0.5,
    buffer_s=0.5,
    plots_per_fig=10,
    figsize_per_trial=(8, 2),
    baseline_duration=7*60,
    dpi=100
):
    # --- Defaults & setup ---
    if region_filter is None:
        region_filter = ['DMPAG','DLPAG','LPAG']
    if output_dir is None:
        output_dir = os.path.join('modulation_plots', session, animal)
    os.makedirs(output_dir, exist_ok=True)

    intervals_df = get_behavior_intervals(behaviour)
    summary_records = []

    # filter neurons by region
    neurons = [nid for nid, reg in zip(n_cluster_index, n_region_index) if reg in region_filter]

    for neuron_label in neurons:
        mod_data = mod_results.get(str(neuron_label), {})
        idxs = np.where(n_cluster_index==neuron_label)[0]
        if len(idxs)==0: 
            continue
        i0 = idxs[0]
        trace = binned_firing_rates[i0]
        spikes = spike_times[i0]
        if trace is None or spikes is None or len(spikes)==0:
            continue

        # --- Baseline metrics (first baseline_duration seconds) ---
        max_bin = int(min(len(trace), baseline_duration/bin_size))
        baseline_trace = trace[:max_bin]
        baseline_spikes = spikes[spikes<baseline_duration]
        mean_baseline = np.nanmean(baseline_trace) if baseline_trace.size else np.nan

        for behavior in sorted(behaviour['behaviours'].unique()):
            info = mod_data.get(behavior, {})
            if not info.get('significant', False):
                continue

            trials = intervals_df[intervals_df['behaviour']==behavior].head(plots_per_fig)
            if trials.empty:
                continue

            # --- Gather trial windows (including buffer) ---
            trial_data = []
            for _, t in trials.iterrows():
                w0 = t['start'] - buffer_s
                w1 = t['stop']  + buffer_s
                b0 = int(max(0, w0/bin_size))
                b1 = int(min(len(trace), w1/bin_size))
                sub_trace = trace[b0:b1]
                sub_spikes = spikes[(spikes>=w0)&(spikes<=w1)] - w0
                trial_dur = t['stop'] - t['start']
                window_dur = w1 - w0
                mean_trial = np.nanmean(sub_trace) if sub_trace.size else np.nan
                trial_data.append({
                    'trace':sub_trace,
                    'spikes':sub_spikes,
                    'trial_dur':trial_dur,
                    'window_dur':window_dur,
                    'mean_rate':mean_trial
                })

            # --- Compute a common y-lim across baseline + all trials ---
            all_max = [np.max(d['trace']) for d in trial_data] + ([np.max(baseline_trace)] if baseline_trace.size else [])
            all_std = [np.std(d['trace']) for d in trial_data] or [0]
            y_lim = max(all_max) + max(all_std)

            # --- Prepare figure: baseline + N trials ---
            n_panels = 1 + len(trial_data)
            w, h = figsize_per_trial
            # if >5 trials, scale up height a bit
            scale = 1.2 if len(trial_data)>5 else 1.0
            fig, axes = plt.subplots(
                    n_panels, 1,
                    figsize=(w, h * n_panels * scale),
                    dpi=dpi,
                    constrained_layout=True
                )

            # ----- Baseline panel -----
            ax0 = axes[0]
            tb = np.arange(0, baseline_duration, bin_size)[:len(baseline_trace)]
            ax0.plot(tb, baseline_trace, alpha=0.7, label='Baseline rate')
            ax0.set_title(
                    f'Neuron {neuron_label} — {behavior} — Baseline (first {baseline_duration/60:.0f} min)',
                    fontsize=10,   # slightly smaller
                    pad=8          # extra space above the plot
                )
            ax0.set_ylabel('Hz')
            ax0.set_xlim(0, baseline_duration)
            ax0.set_ylim(0, y_lim)
            ax0.text(0.95, 0.8, f'μ = {mean_baseline:.2f} Hz',
                     transform=ax0.transAxes, ha='right', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', alpha=0.3))

            # ----- Trial panels -----
            for i, d in enumerate(trial_data, start=1):
                ax = axes[i]
                x = np.linspace(0, d['window_dur'], len(d['trace']))
                ax.plot(x, d['trace'], alpha=0.7, label='Firing rate')
                # shade the actual behavior interval
                ax.axvspan(buffer_s, buffer_s + d['trial_dur'],
                           color='orange', alpha=0.3,
                           label='Behavior' if i==1 else None)
                # raster
                for sp in d['spikes']:
                    ax.vlines(sp, y_lim*0.9, y_lim,
                              linewidth=0.5,
                              label='Spike' if (i==1 and sp==d['spikes'][0]) else None)
                ax.set_ylabel('Hz')
                ax.set_ylim(0, y_lim)
                ax.set_title(
                    f'Trial {i}: μ = {d["mean_rate"]:.2f} Hz',
                    fontsize=10,  # match baseline
                    pad=6         # give a bit more room
                )
                if i == 1:
                    ax.legend(loc='upper right', fontsize='small')
                if i == n_panels - 1:
                    ax.set_xlabel('Time (s)', labelpad=6)

            fname = f"{session}_{animal}_neuron{neuron_label}_{behavior}.png"
            fpath = os.path.join(output_dir, fname)
            fig.savefig(fpath)
            plt.close(fig)

            # save summary
            summary_records.append({
                'neuron': neuron_label,
                'behavior': behavior,
                'mean_baseline_rate': mean_baseline,
                **{k: info.get(k) for k in (
                    'direction','fraction_modulated_trials','binomial_pval',
                    'binomial_z','binomial_z_pval','permutation_pval',
                    'permutation_z','effect_size','significant','n_trials'
                )}
            })

    # write out summary
    summary_df = pd.DataFrame(summary_records)
    summary_file = os.path.join(output_dir, f'summary_{session}_{animal}.csv')
    summary_df.to_csv(summary_file, index=False)

    return summary_df

plot_all_neuron_modulations(
    session,
    animal,
    mod_results,
    behaviour,
    binned_firing_rates,
    spike_times,
    n_cluster_index,
    n_region_index,
    region_filter=['DMPAG', 'DLPAG', 'LPAG'],
    output_dir=rf'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\analysis\modulation_plots\{session}',
    bin_size=0.1,
    buffer_s=1.0,
    plots_per_fig=20,
    figsize_per_trial=(8, 2)
)