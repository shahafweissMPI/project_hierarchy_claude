# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:25:24 2025

@author: chalasp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, mannwhitneyu, binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ─── Helper functions ──────────────────────────────────────────────────────────

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.nanvar(x, ddof=1), np.nanvar(y, ddof=1)
    pooled_sd = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2))
    return (np.nanmean(x) - np.nanmean(y)) / pooled_sd

def detect_baseline_perturbations(fr_z, bin_size, z_thresh=2.0, min_dur_s=0.1):
    periods = []
    for mask, direction in [(fr_z >= z_thresh, +1), (fr_z <= -z_thresh, -1)]:
        in_run = False
        start = None
        for t, m in enumerate(mask):
            if m and not in_run:
                in_run, start = True, t
            elif (not m) and in_run:
                end, in_run = t, False
                if (end - start)*bin_size >= min_dur_s:
                    periods.append((start, end, direction))
        if in_run:  # catch run to end
            end = len(fr_z)
            if (end - start)*bin_size >= min_dur_s:
                periods.append((start, end, direction))
    return periods

def analyze_neuron_modulations_extended(
    fr,
    behaviour_df,
    cluster_id,
    region_id,
    bin_size=0.1,
    z_thresh=2.0,
    min_dur_s=0.5,
    perm_iters=1000,
    fdr_alpha=0.05
):
    T = len(fr)
    session_time = T * bin_size

    # 1) Z-score
    fr_z = StandardScaler().fit_transform(fr.reshape(-1,1)).ravel()

    # 2) Detect modulation periods
    periods = detect_baseline_perturbations(fr_z, bin_size, z_thresh, min_dur_s)
    period_records = [
        {
            'cluster':    cluster_id,
            'region':     region_id,
            'start_s':    s * bin_size,
            'end_s':      e * bin_size,
            'duration_s': (e - s) * bin_size,
            'direction':  'up' if d>0 else 'down'
        }
        for s, e, d in periods
    ]
    cols = ['cluster','region','start_s','end_s','duration_s','direction']
    periods_df = pd.DataFrame(period_records, columns=cols)

    # 3) Pair START/STOP into intervals
    intervals, starts = [], {}
    for _, row in behaviour_df.iterrows():
        beh, t = row['behaviours'], row['frames_s']
        if row['start_stop'] == 'START':
            starts[beh] = t
        elif row['start_stop'] == 'STOP':
            s0 = starts.pop(beh, None)
            if s0 is not None and t > s0:
                intervals.append((beh, s0, t))

    # 4) Build trials_df and count overlaps
    trials = []
    for beh, s0, t1 in intervals:
        overlaps = periods_df[(periods_df.start_s < t1) & (periods_df.end_s > s0)]
        trials.append({
            'cluster':     cluster_id,
            'region':      region_id,
            'behaviour':   beh,
            'trial_start': s0,
            'trial_end':   t1,
            'overlap':     not overlaps.empty
        })
    trials_df   = pd.DataFrame(trials)
    n_trials    = len(trials_df)
    obs_overlap = trials_df['overlap'].sum()

    # 5) Binomial test vs. chance
    total_mod = periods_df['duration_s'].sum()
    chance_p   = total_mod / session_time if session_time>0 else np.nan
    bt         = binomtest(k=obs_overlap, n=n_trials, p=chance_p, alternative='greater')
    binom_p    = bt.pvalue

    # 6) Permutation test for overlap
    perm_counts = np.zeros(perm_iters, int)
    durations   = trials_df['trial_end'] - trials_df['trial_start']
    for i in range(perm_iters):
        starts = np.random.rand(n_trials) * (session_time - durations)
        ends   = starts + durations
        perm_counts[i] = sum(
            ((periods_df.start_s < e) & (periods_df.end_s > s)).any()
            for s, e in zip(starts, ends)
        )
    perm_p = (perm_counts >= obs_overlap).sum() / perm_iters

    # 7) KS / Mann–Whitney + Cohen’s d
    inside = np.zeros(T, bool)
    for s, e, _ in periods:
        inside[s:e] = True
    fr_in  = fr_z[inside]
    fr_out = fr_z[~inside]
    ks_p   = ks_2samp(fr_in, fr_out).pvalue if fr_in.size and fr_out.size else np.nan
    mw_p   = mannwhitneyu(fr_in, fr_out, alternative='two-sided').pvalue \
             if fr_in.size and fr_out.size else np.nan
    d_val  = cohen_d(fr_in, fr_out) if fr_in.size>1 and fr_out.size>1 else np.nan

    # 8) ROC AUC: trial vs non‐trial bins
    y_time = np.zeros(T, int)
    for _, tr in trials_df.iterrows():
        i0 = int(np.floor(tr['trial_start']/bin_size))
        i1 = int(np.ceil (tr['trial_end']  /bin_size))
        y_time[i0:i1] = 1
    try:
        roc_auc = roc_auc_score(y_time, fr_z)
    except ValueError:
        roc_auc = np.nan

    # 9) Collect stats
    stats = {
        'cluster':         cluster_id,
        'region':          region_id,
        'n_periods':       len(periods_df),
        'n_trials':        n_trials,
        'obs_overlap':     obs_overlap,
        'chance_p':        chance_p,
        'binom_p_overlap': binom_p,
        'perm_p_overlap':  perm_p,
        'ks_p':            ks_p,
        'mw_p':            mw_p,
        'cohens_d':        d_val,
        'roc_auc':         roc_auc
    }
    return periods_df, trials_df, stats

def plot_modulation_periods(fr_z, periods_df, bin_size, cluster_id):
    t = np.arange(len(fr_z)) * bin_size
    plt.figure(figsize=(15,3))
    plt.plot(t, fr_z, label=f'Cluster {cluster_id}')
    for _, rec in periods_df.iterrows():
        c = 'red' if rec.direction=='up' else 'blue'
        plt.axvspan(rec.start_s, rec.end_s, color=c, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-scored FR')
    plt.title(f'Cluster {cluster_id} modulation periods')
    plt.legend()
    plt.tight_layout()

def plot_trial_overlaps(trials_df, periods_df):
    for _, tr in trials_df[trials_df.overlap].iterrows():
        s, e = tr.trial_start, tr.trial_end
        plt.figure(figsize=(6,2))
        plt.axvspan(s, e, color='green', alpha=0.3, label=tr.behaviour)
        for _, rec in periods_df.iterrows():
            if rec.end_s > s and rec.start_s < e:
                o0 = max(s, rec.start_s)
                o1 = min(e, rec.end_s)
                c  = 'red' if rec.direction=='up' else 'blue'
                plt.axvspan(o0, o1, color=c, alpha=0.5)
        plt.xlabel('Time (s)')
        plt.yticks([])
        plt.title(f"{tr.behaviour} (cluster {tr.cluster})")
        plt.legend()
        plt.tight_layout()

# ─── Parallel execution ─────────────────────────────────────────────────────────

def _process_neuron(i):
    periods_df, trials_df, stats = analyze_neuron_modulations_extended(
        binned_firing_rates[i],
        behaviour,
        n_cluster_index[i],
        n_region_index[i],
        bin_size=0.1,
        z_thresh=2.5,
        min_dur_s=0.1,
        perm_iters=2000,
        fdr_alpha=0.05
    )
    # # Optional: plot here or after
    # plot_modulation_periods(
    #     fr_z=StandardScaler().fit_transform(
    #         binned_firing_rates[i].reshape(-1,1)
    #     ).ravel(),
    #     periods_df=periods_df,
    #     bin_size=0.1,
    #     cluster_id=n_cluster_index[i]
    # )
    # plt.show()
    # plot_trial_overlaps(trials_df, periods_df)
    # plt.show()

    return periods_df, trials_df, stats

# run on all neurons using all cores
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(_process_neuron)(i)
    for i in range(len(n_cluster_index))
)

# combine
all_periods = pd.concat([r[0] for r in results], ignore_index=True)
all_trials  = pd.concat([r[1] for r in results], ignore_index=True)
stats_df    = pd.DataFrame([r[2] for r in results])


#%%
import matplotlib.pyplot as plt

def plot_modulation_periods(fr_z, periods_df, bin_size, cluster_id, save_path):
    """
    Plot the full-session z-scored firing trace with shaded modulation periods,
    then save as PNG to save_path.
    """
    t = np.arange(len(fr_z)) * bin_size
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(t, fr_z, label=f'Cluster {cluster_id}')
    for _, rec in periods_df.iterrows():
        color = 'red' if rec['direction']=='up' else 'blue'
        ax.axvspan(rec['start_s'], rec['end_s'], color=color, alpha=0.3)
    ax.set_ylabel('Z-scored FR')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'High/modulated periods — cluster {cluster_id}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_trial_overlaps(trials_df, periods_df, bin_size, cluster_id, save_dir):
    """
    For each trial that overlaps a modulation period, plot the trial window
    and shade overlapping segments, then save each as PNG under save_dir.
    """
    for idx, tr in enumerate(trials_df[trials_df['overlap']].itertuples()):
        s, e = tr.trial_start, tr.trial_end
        fig, ax = plt.subplots(figsize=(6,2))
        ax.axvspan(s, e, color='green', alpha=0.3, label=tr.behaviour)
        for _, rec in periods_df.iterrows():
            if rec['end_s'] > s and rec['start_s'] < e:
                o0 = max(s, rec['start_s'])
                o1 = min(e, rec['end_s'])
                c  = 'red' if rec['direction']=='up' else 'blue'
                ax.axvspan(o0, o1, color=c, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_yticks([])
        ax.set_title(f"{tr.behaviour} — cluster {cluster_id}")
        ax.legend(loc='upper right')
        fig.tight_layout()

        fname = os.path.join(save_dir,
            f"cluster_{cluster_id}_trial_{idx}_{tr.behaviour}.png"
        )
        fig.savefig(fname, dpi=300)
        plt.close(fig)


#%%
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assume results, n_cluster_index, binned_firing_rates are already defined:
output_dir = r"Z:\data\project_hierarchy\data\analysis\modulation plots 250619\240529"
os.makedirs(output_dir, exist_ok=True)

alpha = 0.05  # significance threshold for binomial overlap
bin_size = 0.1

for periods_df, trials_df, stats in results:
    if stats['binom_p_overlap'] >= alpha:
        continue

    clu = stats['cluster']
    # find index of this cluster in your original list
    idx = list(n_cluster_index).index(clu)
    fr = binned_firing_rates[idx]
    fr_z = StandardScaler().fit_transform(fr.reshape(-1,1)).ravel()

    # full-session plot
    save_path = os.path.join(output_dir, f"cluster_{clu}_modulation_{session}_{animal}.png")
    plot_modulation_periods(fr_z, periods_df, bin_size, clu, save_path)

    # per-trial plots
    plot_trial_overlaps(trials_df, periods_df, bin_size, clu, output_dir)
