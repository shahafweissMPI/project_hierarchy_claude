# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:48:45 2025

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
session = '240522'

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
pag_ind = np.where((n_region_index=='DMPAG') | (n_region_index=='DLPAG') | (n_region_index=='LPAG'))[0]

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

        
binned_firing_rates, mean_rates, binned_ifr,  mean_rates_ifr, time_bins, recast_spike_times = mean_firing_rate(spike_times, firing_rates, stop_time=n_time_index[-1], bin_size_ms=100, smooth=False, smooth_sigma_bins=0.1) #whole session
binned_baseline_rates, mean_baseline, _, _,time_bins_baseline,_ = mean_firing_rate(spike_times, firing_rates, stop_time=420) #baseline

#%%
import numpy as np
import pandas as pd
import math, traceback
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import poisson, chi2
from scipy.special import betaln, gammaln

# ---------------- Helpers ----------------

def get_valid_null_windows_bins(fr_length_bins, window_bins, behavior_intervals_s, bin_size, exclusion_buffer=1.0):
    """Return valid start indices (in bins) for windows of length `window_bins` excluding behavior_intervals (in sec)."""
    exclusion_intervals = []
    for s, e in behavior_intervals_s:
        b0 = int(math.floor((s - exclusion_buffer) / bin_size))
        b1 = int(math.ceil((e + exclusion_buffer) / bin_size))
        b0 = max(0, b0)
        b1 = min(fr_length_bins, b1)
        if b1 > b0:
            exclusion_intervals.append((b0, b1))
    if not exclusion_intervals:
        merged = []
    else:
        exclusion_intervals.sort(key=lambda x: x[0])
        merged = [exclusion_intervals[0]]
        for cur in exclusion_intervals[1:]:
            last = merged[-1]
            if cur[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], cur[1]))
            else:
                merged.append(cur)
    gaps = []
    prev_end = 0
    for (s_ex, e_ex) in merged:
        if s_ex - prev_end >= window_bins:
            gaps.append((prev_end, s_ex))
        prev_end = e_ex
    if fr_length_bins - prev_end >= window_bins:
        gaps.append((prev_end, fr_length_bins))
    valid_starts = []
    for g0, g1 in gaps:
        last_allowed = g1 - window_bins
        if last_allowed >= g0:
            valid_starts.extend(range(g0, last_allowed + 1))
    return np.array(valid_starts, dtype=int), gaps

def sliding_window_counts(spike_counts, window_bins):
    """Return sliding-window sums (counts) (valid mode)."""
    if len(spike_counts) < window_bins:
        return np.array([], dtype=float)
    ones = np.ones(window_bins, dtype=float)
    sums = np.convolve(spike_counts.astype(float), ones, mode='valid')
    return sums

def _run_parallel_safe(worker_fn, indices, n_jobs=1, verbose=False):
    """Try joblib Parallel (loky), fallback to threading then serial. Print tracebacks on error."""
    indices = list(indices)
    try:
        if verbose:
            print("[parallel] trying joblib Parallel (default backend)...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(worker_fn)(i) for i in indices
        )
        return results
    except Exception as e:
        print("[parallel] default Parallel failed:", str(e))
        traceback.print_exc()
    try:
        if verbose:
            print("[parallel] retrying with backend='threading' ...")
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(worker_fn)(i) for i in indices
        )
        return results
    except Exception as e:
        print("[parallel] Parallel (threading) failed:", str(e))
        traceback.print_exc()
    if verbose:
        print("[parallel] falling back to serial execution; printing per-worker tracebacks on error.")
    results = []
    for i in indices:
        try:
            results.append(worker_fn(i))
        except Exception:
            print(f"[serial-fallback] worker failed for index {i}. Traceback:")
            traceback.print_exc()
            results.append((str(i), {}, {}))
    return results

# ---------------- Poisson-Gamma (posterior predictive) supporting test ----------------

def poisson_gamma_ppp(obs_count, null_counts, min_null_windows=4, prior_a=1.0, prior_b=1.0):
    """
    Posterior predictive p-value P(X >= obs_count) using Poisson likelihood and Gamma prior estimated from null_counts.
    - If null_counts.size < min_null_windows, returns conservative p=1.0 (insufficient null).
    - Uses Gamma(shape=a, rate=b) prior (rate parametrization).
    Returns: (p_ge_obs, a_post, b_post, note)
    """
    null_counts = np.asarray(null_counts, dtype=float)
    if null_counts.size < min_null_windows:
        return 1.0, None, None, 'insufficient_null'
    # posterior parameters (Gamma prior with rate)
    a_post = prior_a + null_counts.sum()
    b_post = prior_b + null_counts.size
    # posterior predictive marginal for counts is Negative-Binomial-like:
    # P(X=k) = choose(k + a_post -1, k) * (b_post/(1+b_post))^a_post * (1/(1+b_post))^k
    # We'll compute P(X >= obs_count) = 1 - sum_{k=0}^{obs_count-1} pmf(k)
    k_vec = np.arange(0, int(max(0, obs_count)))
    if k_vec.size == 0:
        # obs_count == 0: P(X>=0) = 1
        return 1.0, float(a_post), float(b_post), 'ok_obs0'
    # compute log pmf for numerical stability
    logpmf = gammaln(k_vec + a_post) - gammaln(a_post) - gammaln(k_vec + 1) + a_post * np.log(b_post / (1.0 + b_post)) + k_vec * np.log(1.0 / (1.0 + b_post))
    # stabilize
    logpmf -= logpmf.max()
    pmf = np.exp(logpmf)
    # normalize (helpful for numerical safety)
    pmf_sum = pmf.sum()
    if pmf_sum <= 0:
        return 1.0, float(a_post), float(b_post), 'numeric_issue'
    pmf /= pmf_sum
    cdf = pmf.sum()
    p_ge = max(0.0, 1.0 - cdf)
    return float(p_ge), float(a_post), float(b_post), 'ok'

# ---------------- Main detector ----------------

def detect_rolling_modulations_spiketimes_timeblocked_poissongamma(
    spike_times,
    behaviour_df,
    n_cluster_index,
    n_region_index,
    region_filter=None,
    bin_size=0.1,
    window_s=0.5,
    exclusion_buffer=1.0,
    n_permutations=2000,
    significance_level=0.05,
    multiple_test_method='bonferroni',   # 'bonferroni' (default) or 'fdr'
    n_jobs=1,
    random_seed=None,
    verbose=False,
    min_trials=3,
    min_baseline_spikes=5,
    min_spikes_per_trial=1,
    require_min_trials_with_spikes=2,
    min_mean_rate_diff=0.05,
    block_size_s=300.0,
    min_null_windows_for_ppp=4
):
    """
    Time-blocked permutation (primary) + Poisson-Gamma supporting test.
    Returns: mod_results (dict), stats_results (dict), correction_df (DataFrame)
    """
    rng = np.random.default_rng(random_seed)

    # normalize & validate indices
    n_cluster_index = np.asarray(n_cluster_index)
    n_region_index = np.asarray(n_region_index)
    if len(n_cluster_index) != len(n_region_index):
        raise ValueError("n_cluster_index and n_region_index must be same length")
    n_neurons = len(n_cluster_index)

    def _norm(x):
        try:
            if x is None: return 'unknown'
            s = str(x).strip().lower()
            if s == '' or s in ('nan','none'): return 'unknown'
            return s
        except Exception:
            return 'unknown'

    region_norm = np.array([_norm(r) for r in n_region_index], dtype=object)

    # region selection
    if region_filter is None:
        selected_idx = np.arange(n_neurons)
    else:
        if isinstance(region_filter, str):
            rf_list = [region_filter]
        else:
            rf_list = list(region_filter)
        rf_norm = [_norm(x) for x in rf_list]
        rf_set = set([x for x in rf_norm if x != 'unknown'])
        mask_exact = np.array([r in rf_set for r in region_norm], dtype=bool)
        if np.any(mask_exact):
            selected_idx = np.where(mask_exact)[0]
        else:
            patterns = [p for p in [str(p).strip().lower() for p in rf_list if p is not None] if p!='']
            def _substr(r):
                for pat in patterns:
                    if '*' in pat:
                        pat2 = pat.replace('*','')
                        if pat2 == '' or pat2 in r:
                            return True
                    else:
                        if pat in r:
                            return True
                return False
            mask_sub = np.array([_substr(r) for r in region_norm], dtype=bool)
            selected_idx = np.where(mask_sub)[0]

    if selected_idx.size == 0:
        if verbose:
            print("[detect] no neurons matched region_filter -> returning empty")
        return {}, {}, pd.DataFrame(columns=['neuron','behaviour','pval_perm','pval_adj','significant'])

    # parse behaviour intervals
    behavior_events = behaviour_df.copy()
    behavior_list = behavior_events['behaviours'].unique()
    behaviour_intervals = []
    cur_start = None
    for _, row in behavior_events.iterrows():
        if row['start_stop'] == 'START':
            cur_start = float(row['frames_s'])
        elif row['start_stop'] == 'STOP':
            if cur_start is None:
                continue
            cur_stop = float(row['frames_s'])
            if cur_stop > cur_start:
                behaviour_intervals.append((cur_start, cur_stop))
            cur_start = None

    # determine session end
    last_spike = 0.0
    if isinstance(spike_times, dict):
        for lab in n_cluster_index[selected_idx]:
            if lab in spike_times:
                s = spike_times[lab]
            elif str(lab) in spike_times:
                s = spike_times[str(lab)]
            else:
                s = np.array([])
            if len(s) > 0:
                last_spike = max(last_spike, np.max(s))
    else:
        for i in selected_idx:
            if i < len(spike_times):
                s = np.asarray(spike_times[i])
                if len(s) > 0:
                    last_spike = max(last_spike, np.max(s))
    last_beh = max([e for (_, e) in behaviour_intervals]) if behaviour_intervals else 0.0
    session_end = max(last_spike, last_beh) + 1e-6
    if session_end <= 0:
        if verbose:
            print("[detect] session_end <= 0 -> nothing to compute")
        return {}, {}, pd.DataFrame(columns=['neuron','behaviour','pval_perm','pval_adj','significant'])

    n_bins = int(math.ceil(session_end / bin_size))
    window_bins = max(1, int(round(window_s / bin_size)))
    bin_edges = np.arange(0.0, n_bins+1) * bin_size

    # block function
    def block_from_time(t):
        return int(math.floor(t / block_size_s))

    mod_results = {}
    stats_results = {}
    all_pvals = []
    all_keys = []

    indices_to_process = list(map(int, selected_idx.tolist()))

    # Worker
    def _worker(i):
        neuron_label = n_cluster_index[i]
        # fetch spikes
        if isinstance(spike_times, dict):
            if neuron_label in spike_times:
                spikes = np.sort(np.asarray(spike_times[neuron_label], dtype=float))
            elif str(neuron_label) in spike_times:
                spikes = np.sort(np.asarray(spike_times[str(neuron_label)], dtype=float))
            else:
                spikes = np.array([])
        else:
            spikes = np.sort(np.asarray(spike_times[i], dtype=float)) if i < len(spike_times) else np.array([])

        counts, _ = np.histogram(spikes, bins=bin_edges)
        if counts.size < 1:
            return str(neuron_label), {}, {}

        # build behaviour->trials
        behaviour_trials = {}
        for behavior in behavior_list:
            starts = behavior_events[(behavior_events['behaviours'] == behavior) & (behavior_events['start_stop'] == 'START')]['frames_s'].values
            stops  = behavior_events[(behavior_events['behaviours'] == behavior) & (behavior_events['start_stop'] == 'STOP')]['frames_s'].values
            trials = []
            for s,e in zip(starts, stops):
                s = float(s); e = float(e)
                if e <= s: continue
                trials.append((s,e))
            if trials:
                behaviour_trials[behavior] = trials

        if not behaviour_trials:
            return str(neuron_label), {}, {}

        neuron_res = {'_rolling_features': {'time': [], 'mean_counts': []}}
        neuron_stats = {}

        for behavior, trials in behaviour_trials.items():
            n_trials = len(trials)
            if n_trials < min_trials:
                neuron_res[behavior] = {'n_trials': n_trials, 'note': f'not-enough-trials (min={min_trials})'}
                neuron_stats[behavior] = {'n_trials': n_trials}
                continue

            # compute trial lengths and bin-lengths
            trial_lens = [e - s for (s,e) in trials]
            trial_bins_list = [max(1, int(round(L / bin_size))) for L in trial_lens]

            # group trial indices by tb
            tb_to_indices = {}
            for tidx, tb in enumerate(trial_bins_list):
                tb_to_indices.setdefault(tb, []).append(tidx)

            # per-trial observed counts & rates
            per_trial_counts = np.zeros(n_trials, dtype=int)
            per_trial_rates = np.zeros(n_trials, dtype=float)
            total_spikes_across_trials = 0

            # prepare permutation matrix
            perm_rates_matrix = np.zeros((n_permutations, n_trials), dtype=float) if n_permutations > 0 else None

            # keep null_counts_by_tb_and_block to support Poisson-Gamma
            null_counts_indexer = {}  # tb -> dict with keys: valid_vs_tb, trial_stat_map_counts, block_to_valid_indices

            # Precompute trial_stat_map_counts and valid starts for each tb
            for tb in tb_to_indices.keys():
                rolling_sums_tb = sliding_window_counts(counts, tb)
                if rolling_sums_tb.size == 0:
                    trial_stat_map_counts = np.zeros(1, dtype=float)
                else:
                    trial_stat_map_counts = rolling_sums_tb
                valid_vs_tb, _gaps = get_valid_null_windows_bins(n_bins, tb, behaviour_intervals, bin_size, exclusion_buffer=exclusion_buffer)
                valid_vs_tb_times = valid_vs_tb * bin_size if valid_vs_tb.size > 0 else np.array([], dtype=float)
                # block -> indices into valid_vs_tb array (not direct counts index)
                block_to_valid_indices = {}
                for idx_start_in_valid_list, st in enumerate(valid_vs_tb):
                    start_time = st * bin_size
                    blk = block_from_time(start_time)
                    block_to_valid_indices.setdefault(blk, []).append(idx_start_in_valid_list)
                null_counts_indexer[tb] = {
                    'trial_stat_map_counts': trial_stat_map_counts,
                    'valid_vs_tb': valid_vs_tb,
                    'valid_vs_tb_times': valid_vs_tb_times,
                    'block_to_valid_indices': block_to_valid_indices
                }

            # fill observed per-trial counts & rates
            for t_idx, (s, e) in enumerate(trials):
                tb = trial_bins_list[t_idx]
                start_bin_obs = int(math.floor(s / bin_size))
                end_bin_obs = int(math.ceil(e / bin_size))
                start_bin_obs = max(0, min(start_bin_obs, len(counts)-1))
                end_bin_obs = max(0, min(end_bin_obs, len(counts)))
                count_obs = int(np.sum(counts[start_bin_obs:end_bin_obs])) if end_bin_obs > start_bin_obs else 0
                per_trial_counts[t_idx] = count_obs
                trial_dur = max(1e-12, e - s)
                per_trial_rates[t_idx] = count_obs / trial_dur
                total_spikes_across_trials += count_obs

            if total_spikes_across_trials < 1:
                neuron_res[behavior] = {
                    'n_trials': n_trials,
                    'note': f'no spikes in trials (total_spikes={total_spikes_across_trials})',
                    'per_trial_counts': per_trial_counts.tolist(),
                    'per_trial_rates': per_trial_rates.tolist()
                }
                neuron_stats[behavior] = {'n_trials': n_trials}
                continue

            # Build permuted rates: for each trial sample candidate counts from its block (or global fallback)
            for tb, trial_idx_list in tb_to_indices.items():
                info = null_counts_indexer[tb]
                trial_stat_map_counts = info['trial_stat_map_counts']
                valid_vs_tb = info['valid_vs_tb']
                block_to_valid_indices = info['block_to_valid_indices']
                # prepare global candidate counts (mapped from valid_vs_tb)
                if valid_vs_tb.size > 0:
                    clip_idx = np.clip(valid_vs_tb, 0, trial_stat_map_counts.size - 1).astype(int)
                    global_candidate_counts = trial_stat_map_counts[clip_idx]
                else:
                    global_candidate_counts = np.array([], dtype=float)

                for rel_pos, t_idx in enumerate(trial_idx_list):
                    s, e = trials[t_idx]
                    trial_mid = 0.5 * (s + e)
                    blk = block_from_time(trial_mid)
                    # choose block-specific candidates indices (indices into valid_vs_tb array)
                    block_idxs_in_valid = block_to_valid_indices.get(blk, [])
                    if len(block_idxs_in_valid) > 0:
                        counts_indices = np.clip(valid_vs_tb[np.array(block_idxs_in_valid)], 0, trial_stat_map_counts.size - 1).astype(int)
                        candidate_counts = trial_stat_map_counts[counts_indices]
                    else:
                        # fallback to global candidates
                        candidate_counts = global_candidate_counts

                    if candidate_counts.size == 0:
                        # no nulls for this tb anywhere -> fill with zeros
                        sampled_counts = np.zeros(n_permutations, dtype=float) if n_permutations>0 else np.array([], dtype=float)
                    else:
                        # sample with replacement for permutations
                        rand_idx = rng.integers(0, candidate_counts.size, size=n_permutations)
                        sampled_counts = candidate_counts[rand_idx]

                    trial_dur = max(1e-12, trials[t_idx][1] - trials[t_idx][0])
                    if n_permutations > 0:
                        sampled_rates = sampled_counts / trial_dur
                        perm_rates_matrix[:, t_idx] = sampled_rates

            # observed aggregate: mean rate across trials
            obs_mean_rate = float(np.mean(per_trial_rates))

            # permutation aggregated stat
            if n_permutations > 0:
                permuted_means = perm_rates_matrix.mean(axis=1)
                perm_mean = float(np.mean(permuted_means))
                perm_std = float(np.std(permuted_means, ddof=1)) if n_permutations>1 else 0.0
                diff_obs = abs(obs_mean_rate - perm_mean)
                diffs = np.abs(permuted_means - perm_mean)
                perm_pval = float((np.sum(diffs >= diff_obs) + 1) / (n_permutations + 1))
                perm_z = (obs_mean_rate - perm_mean) / perm_std if perm_std > 0 else 0.0
                effect_size = perm_z
            else:
                perm_pval = None; perm_mean = None; perm_std = None; perm_z = None; effect_size = None

            # Poisson-Gamma supporting test: compute per-trial PPP from matched null_counts (block-preferred)
            per_trial_ppps = []
            for t_idx, (s, e) in enumerate(trials):
                tb = trial_bins_list[t_idx]
                info = null_counts_indexer[tb]
                trial_stat_map_counts = info['trial_stat_map_counts']
                valid_vs_tb = info['valid_vs_tb']
                block_to_valid_indices = info['block_to_valid_indices']

                # pick block of trial midpoint
                trial_mid = 0.5 * (s + e)
                blk = block_from_time(trial_mid)
                block_idxs_in_valid = block_to_valid_indices.get(blk, [])
                if len(block_idxs_in_valid) > 0:
                    counts_indices = np.clip(valid_vs_tb[np.array(block_idxs_in_valid)], 0, trial_stat_map_counts.size - 1).astype(int)
                    candidate_counts = trial_stat_map_counts[counts_indices]
                else:
                    # fallback to global candidates
                    if valid_vs_tb.size > 0:
                        counts_indices = np.clip(valid_vs_tb, 0, trial_stat_map_counts.size - 1).astype(int)
                        candidate_counts = trial_stat_map_counts[counts_indices]
                    else:
                        candidate_counts = np.array([], dtype=float)

                if candidate_counts.size < min_null_windows_for_ppp:
                    # not enough nulls for per-trial PPP -> use conservative p=1.0
                    per_trial_ppps.append(1.0)
                else:
                    obs_count = int(per_trial_counts[t_idx])
                    p_ppp, a_post, b_post, note = poisson_gamma_ppp(obs_count, candidate_counts, min_null_windows=min_null_windows_for_ppp)
                    # guard against p_ppp==0 (numerical); set min positive
                    per_trial_ppps.append(max(p_ppp, 1.0/(n_permutations+1e3)))
            # combine per-trial p-values via Fisher's method
            per_trial_ppps = np.asarray(per_trial_ppps, dtype=float)
            # Avoid zeros in log
            per_trial_ppps[per_trial_ppps <= 0] = 1e-12
            # Fisher statistic
            chisq = -2.0 * np.sum(np.log(per_trial_ppps))
            df = 2 * per_trial_ppps.size
            try:
                combined_ppp = float(chi2.sf(chisq, df))
            except Exception:
                combined_ppp = 1.0

            # baseline sufficiency for null (pooled)
            total_baseline_spikes = 0
            total_null_windows = 0
            for tb, info in null_counts_indexer.items():
                valid_vs_tb = info['valid_vs_tb']
                if valid_vs_tb.size > 0:
                    clip_idx = np.clip(valid_vs_tb, 0, info['trial_stat_map_counts'].size - 1).astype(int)
                    null_counts = info['trial_stat_map_counts'][clip_idx]
                    total_baseline_spikes += int(np.sum(null_counts))
                    total_null_windows += int(null_counts.size)
            baseline_ok = (total_baseline_spikes >= min_baseline_spikes)

            # result assembly
            beh_entry = {
                'trial_stat': 'rate_mean',
                'per_trial_counts': per_trial_counts.tolist(),
                'per_trial_rates': per_trial_rates.tolist(),
                'n_trials': n_trials,
                'obs_mean_rate': obs_mean_rate,
                'permutation_pval': perm_pval,
                'permutation_mean': perm_mean,
                'permutation_std': perm_std,
                'permutation_z': perm_z,
                'effect_size': effect_size,
                'poisson_gamma_pval': combined_ppp,
                'baseline_total_spikes': total_baseline_spikes,
                'baseline_ok': baseline_ok
            }

            neuron_res[behavior] = beh_entry
            neuron_stats[behavior] = {
                'n_trials': n_trials,
                'obs_mean_rate': obs_mean_rate,
                'perm_pval': perm_pval,
                'perm_mean': perm_mean,
                'perm_std': perm_std,
                'effect_size': effect_size,
                'poisson_gamma_pval': combined_ppp
            }

        return str(neuron_label), neuron_res, neuron_stats

    # run parallel with fallback
    results = _run_parallel_safe(_worker, indices_to_process, n_jobs=n_jobs, verbose=verbose)

    # gather outputs & pvals
    for neuron_label, res, stat in results:
        if isinstance(res, dict) and len(res) > 0:
            mod_results[str(neuron_label)] = res
            stats_results[str(neuron_label)] = stat
            for beh, beh_dict in res.items():
                if beh.startswith('_'): continue
                pv = beh_dict.get('permutation_pval', None)
                if pv is not None:
                    all_pvals.append(pv)
                    all_keys.append((str(neuron_label), beh))

    # Multiple testing correction
    correction_rows = []
    if len(all_pvals) > 0:
        p_arr = np.asarray(all_pvals)
        m = p_arr.size
        if multiple_test_method == 'bonferroni':
            adj = np.minimum(p_arr * m, 1.0)
            rejected = adj <= significance_level
        elif multiple_test_method == 'fdr':
            order = np.argsort(p_arr)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, m+1)
            adj = p_arr * m / ranks
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.minimum(adj, 1.0)
            rejected = adj <= significance_level
        else:
            # fallback to bonferroni
            adj = np.minimum(p_arr * m, 1.0)
            rejected = adj <= significance_level

        for i, (neuron_label, beh) in enumerate(all_keys):
            p_adj = float(adj[i]); sig = bool(rejected[i])
            if neuron_label in mod_results and beh in mod_results[neuron_label]:
                mod_results[neuron_label][beh]['pval_adj'] = p_adj
                mod_results[neuron_label][beh]['significant'] = sig
            if neuron_label in stats_results and beh in stats_results[neuron_label]:
                stats_results[neuron_label][beh]['pval_adj'] = p_adj
                stats_results[neuron_label][beh]['significant'] = sig
            correction_rows.append({'neuron': neuron_label, 'behaviour': beh, 'pval_perm': float(all_pvals[i]), 'pval_adj': p_adj, 'significant': sig})

    correction_df = pd.DataFrame(correction_rows)

    # Post-hoc strict flag: require baseline_ok, effect size delta, and minimum strict trials (we used rate_mean so check spike counts)
    for neuron_label, res in mod_results.items():
        for beh, d in res.items():
            if beh.startswith('_'): continue
            base_sig = bool(d.get('significant', False))
            strict_sig = base_sig
            if not d.get('baseline_ok', True):
                strict_sig = False
            # require some minimal mean difference from perm mean
            if strict_sig:
                perm_mean = float(d.get('permutation_mean', 0.0)) if d.get('permutation_mean') is not None else 0.0
                if abs(float(d.get('obs_mean_rate', 0.0)) - perm_mean) < min_mean_rate_diff:
                    strict_sig = False
            d['significant_strict'] = bool(strict_sig)

    return mod_results, stats_results, correction_df







#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def get_behavior_intervals(behaviour_df):
    intervals = []
    # assume behaviour_df has rows with 'behaviours', 'start_stop', 'frames_s'
    for behavior, group in behaviour_df.groupby('behaviours'):
        starts = group[group['start_stop']=='START']['frames_s'].values
        stops  = group[group['start_stop']=='STOP']['frames_s'].values
        n_int  = min(len(starts), len(stops))
        for i in range(n_int):
            s_time = float(starts[i])
            e_time = float(stops[i])
            intervals.append({'behaviour': behavior, 'start': s_time, 'stop': e_time})
    return pd.DataFrame(intervals)


def plot_all_neuron_modulations(
    session,
    animal,
    mod_results,
    behaviour_df,
    binned_firing_rates,
    spike_times,
    n_cluster_index,
    n_region_index,
    region_filter=None,
    output_dir=None,
    bin_size=0.1,
    buffer_s=0.5,
    plots_per_fig=10,
    figsize_per_trial=(8, 2),
    baseline_duration=7*60,
    dpi=100,
    use_strict_flag=False
):
    """
    Plots per-neuron-per-behaviour figures and writes two PDFs:
      - significant results
      - non-significant results

    Parameters:
      - session, animal: strings used in filenames
      - mod_results: dict[neuron_label_str] -> dict   (contains per-behaviour dicts)
      - behaviour_df: DataFrame with START/STOP rows and column 'frames_s' (seconds)
      - binned_firing_rates: 2D array-like [n_neurons, n_bins] aligned with n_cluster_index
      - spike_times: list or dict of spike time arrays (seconds). If list-like, assumed aligned with n_cluster_index.
      - n_cluster_index: array-like of neuron labels (same type/values used as keys in mod_results)
      - n_region_index: array-like of region strings aligned with n_cluster_index
      - region_filter: list of regions to include (None -> all)
      - use_strict_flag: if True use 'significant_strict' to split; else 'significant'
    Returns:
      summary_df, (sig_pdf_path or None), (nonsig_pdf_path or None)
    """

    # --- imports & defaults ---
    if output_dir is None:
        output_dir = os.path.join('modulation_plots', session, animal)
    os.makedirs(output_dir, exist_ok=True)

    # Normalize indices to numpy arrays for easy searching
    n_cluster_index = np.asarray(n_cluster_index)
    n_region_index = np.asarray(n_region_index)

    if region_filter is None:
        region_mask = np.ones(len(n_cluster_index), dtype=bool)
    else:
        # case-insensitive match / substring fallback
        rf = [str(r).strip().lower() for r in region_filter]
        def match_region(rstr):
            if rstr is None:
                return False
            s = str(rstr).strip().lower()
            if s in rf:
                return True
            for pat in rf:
                if pat != '' and pat in s:
                    return True
            return False
        region_mask = np.array([match_region(r) for r in n_region_index], dtype=bool)

    selected_neurons = n_cluster_index[region_mask]

    # --- prepare behavior intervals and mapping ---
    intervals_df = get_behavior_intervals(behaviour_df)

    # containers for PDF pages and summary
    sig_pages = []
    nonsig_pages = []
    summary_records = []

    # find index lookup for binned traces and spike_times
    # build a mapping from str(label) -> index into n_cluster_index
    cluster_strs = np.array([str(x) for x in n_cluster_index])
    label_to_index = {cluster_strs[i]: i for i in range(len(cluster_strs))}

    # iterate neurons (only those in selected_neurons)
    for neuron_label in tqdm(selected_neurons, desc="Plotting neurons"):
        neuron_key = str(neuron_label)
        mod_data = mod_results.get(neuron_key, {})
        # find trace index
        if neuron_key not in label_to_index:
            # try matching as int/other type
            matches = np.where(cluster_strs == neuron_key)[0]
            if matches.size == 0:
                continue
            trace_idx = int(matches[0])
        else:
            trace_idx = int(label_to_index[neuron_key])

        # fetch trace and spikes
        trace = None
        try:
            # binned_firing_rates can be list-of-arrays or 2D numpy
            trace = np.asarray(binned_firing_rates)[trace_idx]
        except Exception:
            trace = None

        # spikes can be dict keyed by label or list aligned by index
        spikes = None
        if isinstance(spike_times, dict):
            if neuron_key in spike_times:
                spikes = np.sort(np.asarray(spike_times[neuron_key], dtype=float))
            elif str(neuron_label) in spike_times:
                spikes = np.sort(np.asarray(spike_times[str(neuron_label)], dtype=float))
            else:
                spikes = np.array([])
        else:
            if trace_idx < len(spike_times):
                spikes = np.sort(np.asarray(spike_times[trace_idx], dtype=float))
            else:
                spikes = np.array([])

        # baseline trace
        baseline_bins = int(max(0, int(np.floor(baseline_duration / bin_size))))
        baseline_trace = trace[:baseline_bins] if (trace is not None and len(trace) > 0) else np.array([])

        # iterate behaviours present in mod_data (skip internal keys)
        for behavior in sorted([b for b in mod_data.keys() if not b.startswith('_')]):
            info = mod_data.get(behavior, {})
            # require mod_data to be a dict with 'n_trials' at least
            if not isinstance(info, dict):
                continue

            # determine significance using chosen flag
            sig_flag_name = 'significant_strict' if use_strict_flag else 'significant'
            is_significant = bool(info.get(sig_flag_name, info.get('significant', False)))

            # get trials for this behaviour (up to plots_per_fig)
            trials = intervals_df[intervals_df['behaviour'] == behavior].head(plots_per_fig)
            if trials.empty:
                # still create a small page that reports no trials if it's significant? We'll skip plotting if no trials
                continue

            # prepare figure: baseline + N trials
            trial_data = []
            for _, t in trials.iterrows():
                s = float(t['start'])
                e = float(t['stop'])
                w0 = s - buffer_s
                w1 = e + buffer_s
                # convert to bins (floor for start, ceil for end)
                b0 = int(max(0, int(np.floor(w0 / bin_size))))
                b1 = int(min(len(trace) if trace is not None else 0, int(np.ceil(w1 / bin_size))))
                sub_trace = trace[b0:b1] if (trace is not None and b1 > b0) else np.array([])
                # select spikes in the time window
                sub_spikes = spikes[(spikes >= w0) & (spikes <= w1)] - w0 if (spikes is not None and spikes.size>0) else np.array([])
                trial_dur = e - s
                window_dur = w1 - w0
                mean_trial = float(np.nanmean(sub_trace)) if sub_trace.size else np.nan
                trial_data.append({
                    'start': s, 'stop': e, 'trace': sub_trace, 'spikes': sub_spikes,
                    'trial_dur': trial_dur, 'window_dur': window_dur, 'mean_rate': mean_trial
                })

            # compute y-limits robustly
            all_traces_vals = []
            for d in trial_data:
                if d['trace'].size:
                    all_traces_vals.append(np.max(d['trace']))
            if baseline_trace.size:
                all_traces_vals.append(np.max(baseline_trace))
            # if no data, skip plot
            if not all_traces_vals:
                continue
            all_max = max(all_traces_vals)
            # compute std to pad
            all_stds = [np.std(d['trace']) for d in trial_data if d['trace'].size] + ([np.std(baseline_trace)] if baseline_trace.size else [])
            pad = max(all_stds) if len(all_stds) > 0 else 0.0
            y_lim = max(all_max + pad, 1e-3)

            # figure sizing
            n_panels = 1 + len(trial_data)
            w, h = figsize_per_trial
            scale = 1.2 if len(trial_data) > 5 else 1.0
            fig, axes = plt.subplots(n_panels, 1, figsize=(w, h * n_panels * scale), dpi=dpi, constrained_layout=True)
            if n_panels == 1:
                axes = [axes]

            # Baseline panel
            ax0 = axes[0]
            if baseline_trace.size:
                tb = np.arange(0, baseline_trace.size) * bin_size
                ax0.plot(tb, baseline_trace, alpha=0.7, label='Baseline rate')
            ax0.set_title(f'Neuron {neuron_label} — {behavior} — Baseline (first {baseline_duration/60:.0f} min)')
            ax0.set_ylabel('Hz')
            ax0.set_xlim(0, baseline_duration)
            ax0.set_ylim(0, y_lim)
            mean_baseline = float(np.nanmean(baseline_trace)) if baseline_trace.size else np.nan
            ax0.text(0.95, 0.8, f'μ = {mean_baseline:.2f} Hz',
                     transform=ax0.transAxes, ha='right', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', alpha=0.3))

            # Trial panels
            for i, d in enumerate(trial_data, start=1):
                ax = axes[i]
                x = np.linspace(0, d['window_dur'], len(d['trace'])) if d['trace'].size else np.array([0.0])
                if d['trace'].size:
                    ax.plot(x, d['trace'], alpha=0.7, label='Firing rate')
                # shade behavior interval (buffer_s to buffer_s + trial_dur)
                ax.axvspan(buffer_s, buffer_s + d['trial_dur'], color='orange', alpha=0.25)
                # raster
                for sp in d['spikes']:
                    ax.vlines(sp, y_lim * 0.92, y_lim, linewidth=0.5)
                ax.set_ylabel('Hz')
                ax.set_ylim(0, y_lim)
                ax.set_title(f'Trial {i}: μ = {d["mean_rate"]:.2f} Hz', fontsize=9)
                if i == n_panels - 1:
                    ax.set_xlabel('Time (s)')

            # annotate figure with stats
            p_perm = info.get('permutation_pval', None)
            p_adj = info.get('pval_adj', None)
            p_pg = info.get('poisson_gamma_pval', None)
            obs_mean = info.get('obs_mean_rate', None)
            perm_mean = info.get('permutation_mean', None)
            effect = info.get('effect_size', None)
            baseline_ok = info.get('baseline_ok', None)
            text_lines = [
                f'perm_p = {p_perm:.3e}' if p_perm is not None else 'perm_p = None',
                f'p_adj = {p_adj:.3e}' if p_adj is not None else 'p_adj = None',
                f'poisson_gamma_p = {p_pg:.3e}' if p_pg is not None else 'poisson_gamma_p = None',
                f'obs_mean_rate = {obs_mean:.3f}' if obs_mean is not None else '',
                f'perm_mean = {perm_mean:.3f}' if perm_mean is not None else '',
                f'effect_z = {effect:.2f}' if effect is not None else '',
                f'baseline_ok = {bool(baseline_ok)}' if baseline_ok is not None else ''
            ]
            txt = "\n".join([ln for ln in text_lines if ln != ''])
            fig.text(0.02, 0.02, txt, fontsize=9, va='bottom', ha='left',
                     bbox=dict(boxstyle='round,pad=0.4', alpha=0.2))

            # finalize and append page to appropriate list
            fig.canvas.draw()
            if is_significant:
                sig_pages.append(fig)
            else:
                nonsig_pages.append(fig)

            # add summary record
            summary_records.append({
                'neuron': neuron_label,
                'behavior': behavior,
                'n_trials': info.get('n_trials'),
                'obs_mean_rate': info.get('obs_mean_rate'),
                'perm_pval': info.get('permutation_pval'),
                'pval_adj': info.get('pval_adj'),
                'poisson_gamma_pval': info.get('poisson_gamma_pval'),
                'significant': bool(info.get('significant', False)),
                'significant_strict': bool(info.get('significant_strict', False))
            })

            # close figure here? We delay closing until after saving to PDF (we'll close after)
            # plt.close(fig)  # DO NOT close now, we need to save these figs to PdfPages

    # --- write PDFs if we have pages ---
    sig_pdf_path = None
    nonsig_pdf_path = None

    if len(sig_pages) > 0:
        sig_pdf_path = os.path.join(output_dir, f'{session}_{animal}_significant.pdf')
        with PdfPages(sig_pdf_path) as pdf:
            for fig in sig_pages:
                pdf.savefig(fig)
                plt.close(fig)

    if len(nonsig_pages) > 0:
        nonsig_pdf_path = os.path.join(output_dir, f'{session}_{animal}_nonsignificant.pdf')
        with PdfPages(nonsig_pdf_path) as pdf:
            for fig in nonsig_pages:
                pdf.savefig(fig)
                plt.close(fig)

    # build summary_df and save CSV
    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_csv = os.path.join(output_dir, f'summary_{session}_{animal}.csv')
        summary_df.to_csv(summary_csv, index=False)
        
    plt.close(fig)

    return summary_df, sig_pdf_path, nonsig_pdf_path










#%%
mod_results, stats_results= detect_rolling_modulations_spiketimes(
    spike_times=spike_times,
    behaviour_df=behaviour,
    n_cluster_index=n_cluster_index,
    n_region_index=n_region_index,
    region_filter=['DMPAG', 'DLPAG', 'LPAG'],
    bin_size=0.1,
    window_s=0.5,
    trial_stat='max_z',
    n_permutations=1000,
    n_jobs=-1
)

mod_results, stats_results, correction_df = detect_rolling_modulations_spiketimes_timeblocked_poissongamma(
    spike_times=spike_times,                # your dict or list aligned to n_cluster_index
    behaviour_df=behaviour,              # your DataFrame with START/STOP rows and frames_s
    n_cluster_index=n_cluster_index,
    n_region_index=n_region_index,
    region_filter=['LPAG'], # choose regions you want
    bin_size=0.1,
    window_s=0.5,
    exclusion_buffer=1.0,
    block_size_s=300.0,                     # time-block size for time-blocked nulls (5 min)
    n_permutations=2000,                    # >=2000 recommended for stability
    multiple_test_method='fdr',      # conservative default
    significance_level=0.05,
    min_trials=4,                           # raise min trials for more reliable inference
    min_baseline_spikes=5,
    min_spikes_per_trial=4,
    require_min_trials_with_spikes=4,
    min_mean_rate_diff=0.05,                # minimal practical effect in Hz
    n_jobs=-1,
    random_seed=1234,
    verbose=True
)


#%%

# single neuron + behaviour plot
plot_all_neuron_modulations(
    session,
    animal,
    mod_results,
    behaviour,
    binned_firing_rates,
    spike_times,
    n_cluster_index,
    n_region_index,
    region_filter=['LPAG'],
    output_dir=r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\analysis\modulation_plots',
    bin_size=0.1,
    buffer_s=0.5,
    plots_per_fig=10,
    figsize_per_trial=(8, 2),
    baseline_duration=7*60,
    dpi=100,  
)
