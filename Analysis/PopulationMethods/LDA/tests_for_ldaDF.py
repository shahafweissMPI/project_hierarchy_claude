import os
os.environ["CUPY_NVRTC_OPTIONS"] = "-std=c++17"
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
import os
import cupy as cp
import numpy as np
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

import time

# import local modules
import read_data_light as rdl

#%%Loading of neural data

#Parameters
animal = 'afm16924'
session = '240524'

print("Loading data...")
t_data_load_start = time.time()
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

#%%

behaviour_events = export_behaviour_timestamps(behaviour)

#%%

behaviour_timestamps = rdl.convert_to_behaviour_timestamps(animal, session, behaviour)

#%%



