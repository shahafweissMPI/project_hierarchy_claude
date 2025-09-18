# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:58:42 2025

@author: chalasp
"""

# Select neuron by its cluster index.

# Extract its firing‐rate trace (fr) and z-score it: fr_z.

# Slide a 1 s window (window_size = 1 s) in steps of 0.1 s over fr_z:

# Convert to bins: ws = int(1/bin_size), step = int(0.1/bin_size).

# For each window, collect the segment seg = fr_z[i:i+ws].

# For each window, compute:

# Histogram (e.g. 50 bins, density=True) → hist, edges.

# Mode value = center of bin with largest hist.

# Mode probability = max(hist).

# Variance = np.var(seg).
# Store (start_time, end_time, mode, prob, var) in a list.

# Across all windows, compute mean_var & std_var.

# Flag windows where var > mean_var + 2*std_var or var < mean_var - 2*std_var.

# Plot full‐session: time vs fr_z, shading each flagged window.

# Return the DataFrame of all windows and the subset of “modulated” windows.

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

def test_neuron_change_points_shaded(
    cluster_id,
    binned_firing_rates,
    n_cluster_index,
    bin_size=0.1,
    pen=10,
    jump=10      # only consider breakpoints every `jump` samples
):
    """
    1) Z-score the firing trace for `cluster_id`
    2) Run PELT with L2 cost (O(N)) and a jump to accelerate
    3) Treat each interval between change points as a segment
    4) Plot the full trace with alternating shaded backgrounds per segment
    5) Return the list of change-point times (in seconds)
    """
    # locate and normalize
    idx = list(n_cluster_index).index(cluster_id)
    fr   = binned_firing_rates[idx]
    fr_z = StandardScaler().fit_transform(fr.reshape(-1,1)).ravel()
    T    = len(fr_z)
    total_time = T * bin_size

    # reshape into (n_samples, n_features)
    signal = fr_z.reshape(-1, 1)

    # detect change points with L2 cost and jump
    algo = rpt.Pelt(model="l2", jump=jump).fit(signal)
    cps  = algo.predict(pen=pen)
    cps  = [c for c in cps if c < T]  # drop the final endpoint
    cps_sec = sorted(c * bin_size for c in cps)

    # define segment boundaries
    bounds = [0.0] + cps_sec + [total_time]

    # plot with shaded segments
    t = np.arange(T) * bin_size
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(t, fr_z, color='black', lw=1, label=f'Cluster {cluster_id}')

    colors = ['#FFCCCC', '#CCCCFF']
    for i in range(len(bounds)-1):
        start, end = bounds[i], bounds[i+1]
        ax.axvspan(start, end, color=colors[i % 2], alpha=0.3)

    for cp in cps_sec:
        ax.axvline(cp, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-scored FR')
    ax.set_title(f'Cluster {cluster_id} — change‐point segments (pen={pen}, jump={jump})')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return cps_sec


# ─────────── Example usage ─────────────────────────────────────────────────────

# Let’s test on cluster 765:
cps = test_neuron_change_points_shaded(
    cluster_id=416,
    binned_firing_rates=binned_firing_rates,
    n_cluster_index=n_cluster_index,
    bin_size=0.1,
    pen=15)      

print("Detected change‐points (s):", cps)

#%%
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import ruptures as rpt
from sklearn.preprocessing import StandardScaler


def detect_bottomup_segments(
    fr,
    bin_size=0.1,
    pen=10,
    min_seg_s=0.5,
    jump=5,              # only test splits every 5 bins
    cost_model='l2'      # use 'l2' for linear cost
):
    # 1) z-score
    fr_z = StandardScaler().fit_transform(fr.reshape(-1,1)).ravel()
    N    = len(fr_z)
    min_size = int(np.ceil(min_seg_s / bin_size))

    # 2) bottom-up
    signal = fr_z.reshape(-1,1)
    algo   = rpt.BottomUp(model=cost_model, min_size=min_size, jump=jump).fit(signal)
    cps    = algo.predict(pen=pen)
    cps    = sorted(c for c in cps if 0 < c < N)

    # 3) build segments
    cps_sec = [c * bin_size for c in cps]
    bounds  = [0.0] + cps_sec + [N * bin_size]
    segments = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    return segments, fr_z


def detect_behavior_modulations_bottomup(
    binned_firing_rates,
    behaviour_df,
    n_cluster_index,
    n_region_index=None,
    region_filter=None,
    bin_size=0.1,
    pen=10,
    min_seg_s=0.5,
    jump=5,
    cost_model='l2',
    fdr_alpha=0.1,
    n_jobs=1
):
    # 1) parse behaviour START/STOP into intervals
    intervals, starts = [], {}
    for _, row in behaviour_df.iterrows():
        beh, t = row['behaviours'], row['frames_s']
        if row['start_stop']=='START':
            starts[beh] = t
        elif row['start_stop']=='STOP':
            s0 = starts.pop(beh, None)
            if s0 is not None and t > s0:
                intervals.append((beh, s0, t))
    behaviors = sorted({b for b,_,_ in intervals})

    # 2) region mask
    if region_filter is not None and n_region_index is not None:
        region_mask = np.isin(n_region_index, region_filter)
    else:
        region_mask = np.ones(len(n_cluster_index), bool)

    # 3) select indices to process
    idxs_to_run = [i for i, keep in enumerate(region_mask) if keep]

    def _process(i):
        clu = n_cluster_index[i]
        fr  = binned_firing_rates[i]
        N   = len(fr)

        # detect modulated segments
        segments, fr_z = detect_bottomup_segments(
            fr, bin_size, pen, min_seg_s, jump, cost_model
        )

        # build boolean mask of modulated bins
        mod_mask = np.zeros(N, bool)
        for start_s, end_s in segments:
            i0 = int(np.floor(start_s/bin_size))
            i1 = min(N, int(np.ceil(end_s/bin_size)))
            mod_mask[i0:i1] = True

        chance_p = mod_mask.mean()

        # now per-behavior stats
        per_beh = {}
        for beh in behaviors:
            trials = [(s,e) for b,s,e in intervals if b==beh]
            if not trials:
                continue

            obs_overlaps = 0
            fracs = []
            for s0, e0 in trials:
                i0 = max(0, int(np.floor(s0/bin_size)))
                i1 = min(N, int(np.ceil(e0/bin_size)))
                overlap = mod_mask[i0:i1].any()
                obs_overlaps += overlap
                fracs.append(mod_mask[i0:i1].mean())

            # binomial test
            bt = binomtest(k=obs_overlaps, n=len(trials), p=chance_p, alternative='two-sided')
            effect = np.mean(fracs) - chance_p

            per_beh[beh] = {
                'n_trials':      len(trials),
                'obs_overlaps':  int(obs_overlaps),
                'chance_p':      float(chance_p),
                'binom_p':       float(bt.pvalue),
                'mean_mod_frac': float(np.mean(fracs)),
                'effect_size':   float(effect)
            }

        # FDR correct per neuron
        ps = [v['binom_p'] for v in per_beh.values()]
        if ps:
            rej, p_adj, *_ = multipletests(ps, alpha=fdr_alpha, method='fdr_bh')
            for (beh, stats), rj, padj in zip(per_beh.items(), rej, p_adj):
                stats['fdr_p']       = float(padj)
                stats['significant'] = bool(rj)

        return clu, per_beh

    # 4) parallel run only on selected indices
    out = Parallel(n_jobs=n_jobs)(
        delayed(_process)(i) for i in idxs_to_run
    )

    # 5) aggregate results
    mod_results = {clu: beh for clu, beh in out}
    rows = []
    for clu, beh_map in out:
        for beh, st in beh_map.items():
            row = {'cluster': clu, 'behaviour': beh}
            row.update(st)
            rows.append(row)
    stats_df = pd.DataFrame(rows)

    return mod_results, stats_df


#%%
mod_results, stats_results = detect_behavior_modulations_bottomup(
    binned_firing_rates,
    behaviour,
    n_cluster_index,
    n_region_index,
    region_filter=['DMPAG', 'DLPAG', 'LPAG'],
    bin_size=0.1,
    pen=10,
    min_seg_s=0.1,
    jump=4,
    fdr_alpha=1.0,
    n_jobs=-2
)