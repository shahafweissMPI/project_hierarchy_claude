# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 18:23:20 2025

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
session = '240523_0' #'241211' 

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
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed


def get_behavior_intervals(behaviour_df):
    """
    Pair START/STOP events per behaviour using LIFO matching.
    Returns list of (behaviour, start, stop).
    """
    intervals = []
    stacks = {}
    for _, row in behaviour_df.sort_values('frames_s').iterrows():
        beh, t = row['behaviours'], row['frames_s']
        if row['start_stop'] == 'START':
            stacks.setdefault(beh, []).append(t)
        elif row['start_stop'] == 'STOP' and beh in stacks and stacks[beh]:
            s0 = stacks[beh].pop()
            if t > s0:
                intervals.append((beh, s0, t))
    # warn unmatched
    counts = behaviour_df.groupby('behaviours')['start_stop']\
                        .value_counts().unstack(fill_value=0)
    for beh, cnt in counts.iterrows():
        if cnt.get('START',0) != cnt.get('STOP',0):
            print(f"Warning: '{beh}' START={cnt.get('START')} STOP={cnt.get('STOP')}" )
    return intervals


def detect_and_label_segments(fr, bin_size, pen, min_seg_s, jump, downsample_factor, threshold):
    """
    Segment a firing-rate trace with bottom-up l2 and label segments by mean z-score.
    Returns seg_stats, fr_z, eff_bin, mod_mask.
    """
    fr = np.asarray(fr)
    fr_z = StandardScaler().fit_transform(fr.reshape(-1,1)).ravel()
    eff_bin = bin_size
    if downsample_factor and len(fr_z) > 100_000:
        L = len(fr_z)//downsample_factor
        fr_z = fr_z[:L*downsample_factor].reshape(L,downsample_factor).mean(axis=1)
        eff_bin *= downsample_factor
    T = len(fr_z)
    min_size = int(np.ceil(min_seg_s/eff_bin))
    algo = rpt.BottomUp(model='l2', min_size=min_size, jump=jump)
    algo.fit(fr_z.reshape(-1,1))
    cps = sorted(c for c in algo.predict(pen=pen) if 0 < c < T)
    bounds = [0.0] + [c*eff_bin for c in cps] + [T*eff_bin]
    seg_stats = []
    mod_mask = np.zeros(T, bool)
    for s, e in zip(bounds[:-1], bounds[1:]):
        i0, i1 = int(np.floor(s/eff_bin)), min(T, int(np.ceil(e/eff_bin)))
        if i1 <= i0:
            continue
        mz = fr_z[i0:i1].mean()
        label = 'modulated' if abs(mz) >= threshold else 'unmodulated'
        seg_stats.append({'start': s, 'end': e, 'mean_z': mz, 'label': label})
        if label == 'modulated':
            mod_mask[i0:i1] = True
    return seg_stats, fr_z, eff_bin, mod_mask


def compute_modulation_stats(
    binned_firing_rates, behaviour_df,
    n_cluster_index, n_region_index,
    region_filter=None,
    bin_size=0.1, pen=10, min_seg_s=0.5, jump=5,
    downsample_factor=None, threshold=1.0,
    n_shuffles=1000, trial_count_thresh=5, trial_signif_frac=0.1,
    bootstrap_iters=1000, alpha=0.05, n_jobs=1
):
    """
    Full pipeline: segment, per-trial permutation, across-trial mixed-threshold and FDR.
    Returns `results` and `summary_df`.
    """
    intervals = get_behavior_intervals(behaviour_df)
    behaviours = sorted({b for b,_,_ in intervals})
    bfr = np.asarray(binned_firing_rates)
    tasks = [(clu, idx) for idx, (clu, reg) in enumerate(zip(n_cluster_index, n_region_index))
             if (region_filter is None or reg in region_filter) and idx < bfr.shape[0]]

    def worker(clu, idx):
        fr = bfr[idx]
        segs, fr_z, eff_bin, mask = detect_and_label_segments(
            fr, bin_size, pen, min_seg_s, jump, downsample_factor, threshold
        )
        T = len(fr_z)
        # per-trial permutation test
        trial_data = []
        for beh, s0, e0 in intervals:
            i0, i1 = int(np.floor(s0/eff_bin)), min(T, int(np.ceil(e0/eff_bin)))
            L = i1 - i0
            if L <= 0:
                continue
            obs = mask[i0:i1].mean()
            count = 0
            for _ in range(n_shuffles):
                st = np.random.randint(0, T-L+1)
                if mask[st:st+L].mean() >= obs:
                    count += 1
            p_perm = (count + 1) / (n_shuffles + 1)
            trial_data.append({'behaviour': beh, 'start': s0, 'end': e0,
                               'fraction': obs, 'p_perm': p_perm})
        df_td = pd.DataFrame(trial_data)

        # across-trial summary with mixed-threshold
        cross = []
        for beh, grp in df_td.groupby('behaviour'):
            fracs = grp['fraction'].values
            p_bin = mask.mean()
            ntr = len(fracs)
            nsig = (grp['p_perm'] < alpha).sum()
            signif_frac = nsig / ntr if ntr > 0 else np.nan
            if ntr > trial_count_thresh:
                # require fraction of significant trials
                significant = signif_frac >= trial_signif_frac
                p_cross = np.nan
                eff = fracs.mean() - p_bin
            else:
                # bootstrap small n
                diffs = fracs - p_bin
                boots = []
                for _ in range(bootstrap_iters):
                    sample = np.random.choice(diffs, size=ntr, replace=True)
                    boots.append(np.mean(sample))
                p_cross = (np.sum(np.array(boots) <= 0) + 1) / (bootstrap_iters + 1)
                eff = diffs.mean() if ntr > 0 else np.nan
                significant = p_cross < alpha
            cross.append({
                'behaviour': beh,
                'n_trials': ntr,
                'n_sig_trials': int(nsig),
                'signif_frac': signif_frac,
                'p_cross': p_cross,
                'effect_size': eff,
                'significant': bool(significant)
            })
        # FDR correction on p_cross where defined
        pvals = np.array([c['p_cross'] for c in cross])
        valid = ~np.isnan(pvals)
        if valid.sum() > 0:
            rej, padj, _, _ = multipletests(pvals[valid], alpha=alpha, method='fdr_bh')
            j = 0
            for i, c in enumerate(cross):
                if valid[i]:
                    c['p_cross_fdr'] = float(padj[j])
                    c['significant'] = bool(rej[j])
                    j += 1
                else:
                    c['p_cross_fdr'] = np.nan
        else:
            for c in cross:
                c['p_cross_fdr'] = np.nan
        return {
            'cluster': clu,
            'segs': segs,
            'fr_z': fr_z,
            'eff_bin': eff_bin,
            'trial_data': trial_data,
            'cross': cross
        }

    results = Parallel(n_jobs=n_jobs)(delayed(worker)(clu, idx) for clu, idx in tasks)
    rows = []
    for r in results:
        for c in r['cross']:
            rows.append({'cluster': r['cluster'], **c})
    summary_df = pd.DataFrame(rows)
    return results, summary_df


def plot_all_neuron_modulations_v6(
    session, animal, mod_results_full, behaviour_df,
    binned_firing_rates, spike_times,
    n_cluster_index, n_region_index,
    region_filter=None, output_dir=None,
    bin_size=0.5, buffer_s=0.5,
    baseline_duration=7*60, mean_baseline=None,
    dpi=100, plots_per_page=6
):
    """
    v6-style plotting: baseline plus paginated trial panels.
    Uses raw trace and spikes to rebuild per-trial segments so each td has 'trace' and 'spikes'.
    """
    if region_filter is None:
        region_filter = ['DMPAG','DLPAG','LPAG']
    if output_dir is None:
        output_dir = os.path.join('modulation_plots', session, animal)
    os.makedirs(output_dir, exist_ok=True)

    # extract intervals
    intervals = get_behavior_intervals(behaviour_df)
    behaviors = sorted({b for b,_,_ in intervals})
    cmap = plt.get_cmap('tab20_r', len(behaviors))
    beh_colors = {b: cmap(i) for i,b in enumerate(behaviors)}
    legend_handles = [plt.Line2D([0],[0], color=beh_colors[b], lw=4, label=b)
                      for b in behaviors]
    baseline_bins = int(baseline_duration / bin_size)
    pdf_path = os.path.join(output_dir, f'{session}_{animal}_modulations.pdf')
    svg_paths = []

    with PdfPages(pdf_path) as pdf:
        for neuron_label, reg in zip(n_cluster_index, n_region_index):
            if region_filter and reg not in region_filter:
                continue
            res = mod_results_full.get(str(neuron_label), {})
            # find array index
            idxs = np.where(np.array(n_cluster_index)==neuron_label)[0]
            if len(idxs)==0:
                continue
            idx = idxs[0]
            trace = binned_firing_rates[idx]
            spikes = spike_times[idx]
            bl_trace = trace[:baseline_bins]
            bl_spikes = spikes[(spikes>=0)&(spikes<baseline_duration)]
            mean_bl = mean_baseline[idx] if mean_baseline is not None else np.nan

            for beh in behaviors:
                # only plot statistically significant behaviours (FDR corrected)
                beh_cross = next((c for c in res.get('cross', []) if c['behaviour']==beh), None)
                if beh_cross is None or not beh_cross.get('significant', False):
                    continue
                # build per-trial segments
                trials = [(b,s,e) for b,s,e in intervals if b==beh]
                trial_segs = []
                # compute window dims
                durations = [e-s for _,s,e in trials]
                max_dur = max(durations)
                window_dur = max_dur + 2*buffer_s
                w_bins = int(window_dur / bin_size)
                for b,s,e in trials:
                    pad = (window_dur - (e-s))/2
                    w_start = s - pad
                    # rebuild trace segment
                    seg = np.full(w_bins, np.nan)
                    for j in range(w_bins):
                        ti = int((w_start + j*bin_size)/bin_size)
                        if 0 <= ti < len(trace):
                            seg[j] = trace[ti]
                    spks = spikes[(spikes>=w_start)&(spikes< w_start+window_dur)]
                    # find p_perm in trial_data
                    p_perm = next((td['p_perm'] for td in res.get('trial_data', [])
                                   if td['behaviour']==beh and td['start']==s and td['end']==e), np.nan)
                    trial_segs.append({'trace': seg,
                                       'spikes': spks,
                                       'dur': e-s,
                                       'pad': pad,
                                       'start': s,
                                       'p_perm': p_perm,
                                       'fraction': next((td['fraction'] for td in res.get('trial_data', [])
                                                        if td['start']==s and td['end']==e), np.nan)})
                if not trial_segs:
                    continue
                # pagination
                n_trials = len(trial_segs)
                n_pages = math.ceil(n_trials/plots_per_page)
                # compute y-limit
                all_max = [np.nanmax(ts['trace']) for ts in trial_segs]
                if bl_trace.size:
                    all_max.append(np.nanmax(bl_trace))
                ymax = max(all_max) * 1.2

                # plot each page
                for page in range(n_pages):
                    fig, axes = plt.subplots(plots_per_page+1,1,
                                             figsize=(8,2*(plots_per_page+1)), dpi=dpi)
                    plt.subplots_adjust(top=0.85, hspace=0.5)
                    agg_p = beh_cross.get('p_cross_fdr', np.nan)
                    eff    = beh_cross.get('effect_size', np.nan)
                    fig.suptitle(
                        f"{session} {animal} | Neuron {neuron_label} | {beh} | "
                        f"p={agg_p:.3g} | Δ={eff:.2f}", fontsize=12)
                    # baseline
                    ax0 = axes[0]
                    t_bl = np.arange(len(bl_trace)) * bin_size
                    ax0.plot(t_bl, bl_trace, color='k', lw=1)
                    for sp in bl_spikes:
                        ax0.vlines(sp, ymax*0.85, ymax, lw=0.5, color='k')
                    ax0.set_xlim(0, baseline_duration)
                    ax0.set_ylim(0, ymax)
                    ax0.set_ylabel('Hz')
                    ax0.set_title(f"Baseline {mean_bl:.2f}Hz")

                    # trial panels
                    for i in range(plots_per_page):
                        ax = axes[i+1]
                        tid = page*plots_per_page + i
                        if tid < n_trials:
                            td = trial_segs[tid]
                            # time axis
                            t_rel = np.arange(len(td['trace']))*bin_size - td['pad']
                            # shade all behaviours
                            for bb,s0,e0 in intervals:
                                r0, r1 = s0-td['start'], e0-td['start']
                                sc, ec = max(r0, t_rel[0]), min(r1, t_rel[-1])
                                if ec>sc:
                                    ax.axvspan(sc, ec, color=beh_colors.get(bb,'gray'), alpha=0.2)
                            # highlight this trial
                            ax.axvspan(0, td['dur'], color=beh_colors[beh], alpha=0.8)
                            # plot trace
                            ax.plot(t_rel, td['trace'], color='black', lw=1)
                            # raster
                            for sp in td['spikes']:
                                rel = sp - td['start']
                                if t_rel[0] <= rel <= t_rel[-1]:
                                    ax.vlines(rel, 0.9*ymax, ymax, color='k', lw=0.7)
                            # trial boundaries
                            ax.axvline(0, ls='--', color='green')
                            ax.axvline(td['dur'], ls='--', color='green')
                            # annotate p_perm
                            ax.set_xlim(t_rel[0], t_rel[-1])
                            ax.set_ylim(0, ymax)
                            ax.set_ylabel('Hz')                            ax.legend(handles=legend_handles, loc='upper right', fontsize=6, ncol=2)
                            ax.set_title(f"Trial {tid+1}: p_perm={td['p_perm']:.3f}")
                        else:
                            ax.axis('off')
                    axes[-1].set_xlabel('Time (s)')
                    pdf.savefig(fig)
                    svg = os.path.join(output_dir, f'{session}_{animal}_neuron{neuron_label}_{beh}_page{page+1}.svg')
                    fig.savefig(svg, format='svg')
                    svg_paths.append(svg)
                    plt.close(fig)
    return pdf_path, svg_paths




results, summary_df = compute_modulation_stats(
    binned_firing_rates,
    behaviour,
    n_cluster_index,
    n_region_index,
    region_filter=['DMPAG','DLPAG','LPAG'],
    bin_size=0.1, pen=10, min_seg_s=0.1, jump=5,
    downsample_factor=2, threshold=1.0,
    n_shuffles=1000, trial_count_thresh=10, trial_signif_frac=0.2,
    bootstrap_iters=1000, alpha=0.05,
    n_jobs=-1
)

# Build the dict that the v6 plotter expects:
mod_results_full = { str(r['cluster']): r for r in results }


pdf_path, svg_paths = plot_all_neuron_modulations_v6(
    session=session,
    animal=animal,
    mod_results_full=mod_results_full,
    behaviour_df=behaviour,
    binned_firing_rates=binned_firing_rates,
    spike_times=spike_times,
    n_cluster_index=n_cluster_index,
    n_region_index=n_region_index,
    region_filter=['DMPAG','DLPAG','LPAG'],
    output_dir=rf'Z:\data\project_hierarchy\data\analysis\modulation_plots\ruptures\{session}',
    bin_size=0.1,
    buffer_s=0.5,
    baseline_duration=7*60,
    mean_baseline=mean_baseline,
    dpi=200,
    plots_per_page=6
)


print("Saved PDF to:", pdf_path)
print("SVG pages:", svg_paths)
print("Stats CSV:", csv_path)
