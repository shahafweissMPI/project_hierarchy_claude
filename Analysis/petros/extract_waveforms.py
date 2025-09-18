# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 01:44:44 2025

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
session = '240523'

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
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo.rawio.spikeglxrawio import SpikeGLXRawIO
import scipy.signal as signal

# Globals for filter and time limit
_SOS = None
_MAX_SAMPLES = None

def _process_unit_thread(
    i, ap_file, times, ch, sr, nchan,
    total_samples, pre_samps, post_samps, max_spikes,
    bp_sos, times_are_troughs, times_are_seconds,
    gains, offsets
):
    wlen = pre_samps + post_samps

    # Convert times → samples
    if times_are_troughs and not times_are_seconds:
        samp_idx = np.asarray(times, int)
    else:
        samp_idx = np.round(np.asarray(times, float) * sr).astype(int)

    # Keep valid spikes within recording & time cap
    valid = (
        (samp_idx - pre_samps >= 0) &
        (samp_idx + post_samps <= total_samples) &
        (samp_idx + post_samps <= _MAX_SAMPLES)
    )
    samp_idx = samp_idx[valid]
    if samp_idx.size > max_spikes:
        samp_idx = np.random.choice(samp_idx, max_spikes, replace=False)
    if samp_idx.size == 0:
        return i, np.empty((0, wlen), dtype=np.float32)

    snippets = []
    bytes_per_sample = nchan * 2
    read_bytes = wlen * bytes_per_sample

    gain = gains[ch]      # µV per bit
    offset = offsets[ch]  # µV

    with open(ap_file, 'rb') as f:
        for s in samp_idx:
            start = s - pre_samps
            f.seek(start * bytes_per_sample)
            block = f.read(read_bytes)
            if len(block) != read_bytes:
                continue
            arr = np.frombuffer(block, dtype='<i2').reshape(wlen, nchan)
            snippet = arr[:, ch].astype(np.float32)
            # Convert to microvolts
            snippet = snippet * gain + offset
            # Bandpass filter in µV
            snippet = signal.sosfilt(bp_sos, snippet)
            if not times_are_troughs:
                trough = np.argmin(snippet)
                shift = pre_samps - trough
                snippet = np.roll(snippet, shift)
            snippets.append(snippet)

    return i, (np.stack(snippets, axis=0) if snippets else np.empty((0, wlen), dtype=np.float32))


def collect_and_plot_all_units(
    ap_file,
    spike_times,
    channel_index,
    cluster_index,
    output_folder,
    max_spikes=500,
    pre_s=0.2,
    post_s=0.5,
    n_jobs=None,
    bp_low=300,
    bp_high=3000,
    bp_order=3,
    times_are_troughs=False,
    times_are_seconds=True,
    limit_minutes=7
):
    """
    Threaded extraction & plotting with single-channel per cluster,
    Butterworth bandpass, optional trough alignment, 7-min cap,
    and conversion from ADC to microvolts.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Ensure channel_index is 1D ints
    channel_index = np.asarray(channel_index)
    if channel_index.ndim > 1:
        channel_index = channel_index[:, 0]
    channel_index = channel_index.astype(int)

    # Parse header
    reader = SpikeGLXRawIO(dirname=os.path.dirname(ap_file), load_sync_channel=False)
    try:
        reader.parse_header()
    except KeyError:
        pass
    sr = reader.header['signal_channels'][0][2]
    sc = reader.header['signal_channels']
    nchan = sc.shape[0]
    gains = sc['gain']      # µV per bit
    offsets = sc['offset']  # µV

    # Design filter & time cap
    global _SOS, _MAX_SAMPLES
    _SOS = signal.butter(bp_order, [bp_low, bp_high], btype='band', fs=sr, output='sos')
    _MAX_SAMPLES = int(limit_minutes * 60 * sr)

    # Window sizes and time axis
    pre_samps = int(pre_s * sr)
    post_samps = int(post_s * sr)
    wlen = pre_samps + post_samps
    t = (np.arange(wlen) - pre_samps) / sr * 1e3  # ms relative to trough

    total_samples = os.path.getsize(ap_file) // (nchan * 2)

    n_units = len(spike_times)
    mean_wfs = np.zeros((n_units, wlen), dtype=np.float32)
    snippets = [None] * n_units

    # Prepare args
    args_list = [
        (
            i, ap_file, spike_times[i], channel_index[i],
            sr, nchan, total_samples, pre_samps, post_samps,
            max_spikes, _SOS, times_are_troughs, times_are_seconds,
            gains, offsets
        )
        for i in range(n_units)
    ]

    # Thread pool
    n_jobs = n_jobs or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=n_jobs) as exe:
        futures = [exe.submit(_process_unit_thread, *args) for args in args_list]
        for future in as_completed(futures):
            i, arr = future.result()
            snippets[i] = arr
            if arr.size:
                mean_wfs[i] = arr.mean(axis=0)

    # Save NPZ
    npz_path = os.path.join(output_folder, 'cluster_waveforms_threaded.npz')
    obj = np.empty(n_units, dtype=object)
    obj[:] = snippets
    np.savez_compressed(
        npz_path,
        mean_wfs=mean_wfs,
        snippets=obj,
        channel_index=channel_index,
        cluster_index=np.asarray(cluster_index, int),
        time_axis=t
    )

    # Save PDF (y-axis in µV)
    pdf_path = os.path.join(output_folder, 'cluster_waveforms_threaded.pdf')
    with PdfPages(pdf_path) as pdf:
        for i, arr in enumerate(snippets):
            if arr is None or arr.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(6,4))
            for sn in arr:
                ax.plot(t, sn, color='gray', alpha=0.3)
            ax.plot(t, mean_wfs[i], color='blue', linewidth=2, label='Mean')
            ax.set_title(f"Cluster {cluster_index[i]} (ch {channel_index[i]})")
            ax.set_xlabel("Time (ms) relative to trough")
            ax.set_ylabel("Amplitude (µV)")
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("Saved:", npz_path, pdf_path)
    return npz_path, pdf_path



collect_and_plot_all_units(
    r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\240524\trial0\ephys\preprocessed\catgt_afm16924_240524_pup_retrieval_g0\afm16924_240524_pup_retrieval_g0_imec0\afm16924_240524_pup_retrieval_g0_tcat.imec0.ap.bin",
    n_spike_times,
    n_channel_index,
    n_cluster_index,
    rf'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\analysis\channel_waveforms\afm16924\{session}',
    max_spikes=100,
    pre_s=0.001,
    post_s=0.002,
    n_jobs=3000
)