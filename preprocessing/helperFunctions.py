import numpy as np
import preprocessFunctions as pp
import scipy.io
import IPython
try:
    import cv2
except ModuleNotFoundError:
    print("cv2 couldn't be imported, video processing won't work")
#import winsound
import sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import KFold
from scipy.stats import zscore
from numpy.random import choice
from scipy import stats
import os
import numpy as np
from numba import njit

from pathlib import Path

def set_page_format(page_format: str):
    """
    Set the figure size based on the given paper format (e.g., 'A2', 'A3', 'A4').
    
    The dimensions are based on ISO A-series paper sizes:
       A0: 841 x 1189 mm
       A1: 594 x 841 mm
       A2: 420 x 594 mm
       A3: 297 x 420 mm
       A4: 210 x 297 mm
       A5: 148 x 210 mm
       
    The dimensions in inches are computed assuming 1 inch = 25.4 mm.
    
    Parameters:
        page_format (str): The page format string ('A2', 'A3', 'A4', etc.)
        
    Raises:
        ValueError: If an unsupported page format is provided.
    """
    # Define paper sizes in millimeters
    sizes_mm = {
        'A0': (841, 1189),
        'A1': (594, 841),
        'A2': (420, 594),
        'A3': (297, 420),
        'A4': (210, 297),
        'A5': (148, 210)
    }
    
    pf = page_format.upper().strip()
    if pf not in sizes_mm:
        raise ValueError(f"Unsupported page format: {page_format}")
    
    width_mm, height_mm = sizes_mm[pf]
    # Convert mm to inches
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    
    # Update matplotlib rc parameters for figure size
    plt.rc('figure', figsize=(width_in, height_in))
    print(f"Set page format {pf}: {width_in:.2f}in x {height_in:.2f}in.")
    
def insert_event_times_into_behavior_df(behaviour,framerate,event_type=None,behavior_name=None,behavior_type=None,**kwargs):
    """
    Insert new behavioral events into the DataFrame `behaviour` with optional extra columns.

    Parameters:
      behaviour (pd.DataFrame): Original DataFrame with these columns:
        'behaviours', 'behavioural_category', 'start_stop', 'frames_s',
        'frames', 'video_start_s', 'video_end_s'
      event_type (str): The behavioral event type. can be on one of "baseline_random" /  "distance_to_shelter" / "speed"
      behavior_name (str): The event name to display should be capital letters
      behavior_type (str) : category name to display
      
      time_pairs (list of [start, stop]): List of timepoint pairs (in seconds).
      kwargs: Optional keyword arguments which can include:
          for "distance_to_shelter" / "speed":
              distance_to_shelter: an array of floats
              speed: an array of floats
              time_ax: an array of floats
              Threshold (float): a constant float. defaults to 20 for speed and 5 for distance from shelter
              diff_time_s (float): a constant float. defaults to 5 for speed and 1 for distance from shelter
          
          for baseline_random:
          
              baseline_time_s : a float, defaults is 7 minutes 7*60,
              n_trials: 
          
    Returns:
      pd.DataFrame: The updated DataFrame with the new rows appended.
    """
    
    
    video_start = behaviour.iloc[0]['video_start_s']
    video_end = behaviour.iloc[0]['video_end_s']
    baseline_time_s = kwargs.get('baseline_time_s',7*60)
    N=np.nanmin([behaviour.iloc[0,3],baseline_time_s])
    # Generate N random time points with at least 10 seconds between each.
    if event_type=='baseline_random':    # generate N random times from baseline    
        n_trials =      kwargs.get('n_trials',10)        
        time_points = generate_random_times(N, num_points=n_trials, min_gap=10)
        start_stop=['POINT'] * n_trials
        frames=time_points*framerate
        frames=frames.astype(int)
        # Create a new DataFrame with the 10 additional rows.
        new_rows = pd.DataFrame({
            'behaviours': ['random_baseline'] * n_trials,
            'behavioural_category': ['baseline_random'] * n_trials,
            'start_stop': start_stop,
             'frames_s' : time_points,
             'frames' :frames,
            'video_start_s': [behaviour.iloc[0,5]] * n_trials,
            'video_end_s': [behaviour.iloc[0,6]] * n_trials  # All rows have the same video_end_s as the first row
        })
    
    elif event_type=='distance_to_shelter' or  event_type=='speed' or event_type=='acceleration':
        # Extract extra parameters from kwargs.
        speed = kwargs.get('speed', None)
        distance_to_shelter = kwargs.get('distance_to_shelter', None)
        time_ax = kwargs.get('time_ax', None)
        Threshold = kwargs.get('Threshold', None)
        diff_time_s = kwargs.get('diff_time_s', None)
        
        time_ax = time_ax[time_ax <= np.nanmin([time_ax[-1], N])] # limit to baseline
        
        base_speed=speed[0:len(time_ax)]
        base_distance_to_shelter=distance_to_shelter[0:len(time_ax)]                       
        
        
        

        # Determine which threshold function to call.
        #generate
        print(event_type)
        if event_type=='distance_to_shelter':
            if event_type == 'distance_to_shelter':
                if base_distance_to_shelter is None or time_ax is None:
                    raise ValueError("Both 'distance_to_shelter' and 'time_ax' must be provided for distance_to_shelter events.")
                time_pairs = generate_shelter_distance_threshold_crossings(base_distance_to_shelter, time_ax,
                                                                           Threshold=Threshold,
                                                                           diff_time_s=diff_time_s)
        elif event_type=='speed':    
            if base_speed is None or time_ax is None:
                raise ValueError("Both 'speed' and 'time_ax' must be provided for speed events.")
            time_pairs = generate_speed_threshold_crossings(base_speed, time_ax,
                                                            Threshold=Threshold,
                                                            diff_time_s=diff_time_s)
        elif event_type=='acceleration':
            time_pairs=generate_acceleration_threshold_crossings(base_speed, time_ax, Threshold=Threshold, diff_time_s=diff_time_s)
            
    
   
        new_rows_list = []
        behavior_type= behavior_type.capitalize()
        for time_pair in time_pairs:
             start_time, stop_time = time_pair
             start_frame = start_time * framerate
             stop_frame = stop_time * framerate

             pair_df = pd.DataFrame({
                 'behaviours': [behavior_name] * 2,
                 'behavioural_category': [behavior_type] * 2,
                 'start_stop': ['START', 'STOP'],
                 'frames_s': [start_time, stop_time],
                 'frames': [start_frame, stop_frame],
                 'video_start_s': [video_start] * 2,
                 'video_end_s': [video_end] * 2
             })
             new_rows_list.append(pair_df)
         # Combine all new rows into one DataFrame.
        if len(new_rows_list)==0:
             raise ValueError("no periods selected,try lowering threshold.")
        new_rows = pd.concat(new_rows_list, ignore_index=True)
         
        # for time_pair in time_pairs:
        #     start_time, stop_time = time_pair
        #     start_frame = start_time * framerate
        #     stop_frame = stop_time * framerate
    
        #     new_rows = pd.DataFrame({
        #         'behaviours': [behavior_name] * 2,
        #         'behavioural_category': [behavior_type] * 2,
        #         'start_stop': ['START', 'STOP'],
        #         'frames_s': [start_time, stop_time],
        #         'frames': [start_frame, stop_frame],
        #         'video_start_s': [video_start] *2,
        #         'video_end_s': [video_end ] *2
        #     })

    else:        
           raise ValueError(f"unkown behavioral event_type argument")
    
        # Append the new rows to the existing DataFrame.
    behaviour = pd.concat([behaviour, new_rows], ignore_index=True)
        
        # Optional: display the updated DataFrame
    print(behaviour.tail(5))
    
    return behaviour
def generate_speed_threshold_crossings(speed, time_ax, Threshold=45, diff_time_s=10):
    """
    Identify segments where speed rises above a threshold and then falls back below 5.
    The start time is defined as the first time before the threshold crossing 
    when the speed was over 5, and the end time is when speed finally falls below 5.

    Parameters:
        speed (array-like): Array of speed values.
        time_ax (array-like): Array of corresponding time values.
        Threshold (float): The speed threshold for detecting the upward crossing.
        diff_time_s (float): Minimum time difference between events.

    Returns:
        list: A list of [start, stop] pairs, where 'start' is the time when 
              speed first exceeded 5 (in the contiguous block) before the upward crossing 
              of the Threshold, and 'stop' is the time when speed falls back below 5.
    """
    crossings = []
    in_segment = False
    start_time = None

    if Threshold is None:
        Threshold = 45
    if diff_time_s is None:
        diff_time_s = 5

    # Iterate over each sample with index for backwards lookup when needed.
    for i, (s, t) in enumerate(zip(speed, time_ax)):
        # Detect upward crossing: speed goes from below to above or equal to Threshold.
        if not in_segment and s >= Threshold:
            # Backtrack to find the first time before this event when speed was over 5.
            j = i
            while j > 0 and speed[j - 1] > 5:
                j -= 1
            start_time = time_ax[j]
            in_segment = True
        # Detect downward crossing: when speed falls below 5.
        elif in_segment and s < 5:
            crossings.append([start_time, t])
            in_segment = False

    # Remove events that are less than diff_time_s seconds apart.
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        if last_end is None or start - last_end >= diff_time_s:
            filtered.append(interval)
            last_end = end

    return filtered
def generate_speed_threshold_crossings2(speed, time_ax, Threshold=45, diff_time_s=10):
    """
    Identify segments where speed rises above a threshold and then falls back below it.
    The start time is defined as the first time before the threshold crossing 
    when the speed was over 5.
    
    Parameters:
        speed (array-like): Array of speed values.
        time_ax (array-like): Array of corresponding time values.
        Threshold (float): The speed threshold for detecting crossings.
        diff_time_s (float): Minimum time difference between events.
    
    Returns:
        list: A list of [start, stop] pairs, where 'start' is the time when 
              speed first exceeded 5 (in the contiguous block) before crossing above 
              the Threshold, and 'stop' is the time when speed falls back below the Threshold.
    """
    crossings = []
    in_segment = False
    start_time = None
    
    if Threshold is None:
        Threshold = 45
    if diff_time_s is None:
        diff_time_s = 5
    
    # Iterate over each sample using index to allow backwards lookup.
    for i, (s, t) in enumerate(zip(speed, time_ax)):
        # Detect upward crossing: speed goes from below to above or equal to threshold.
        if not in_segment and s >= Threshold:
            # Backtrack to find the first time prior to this event when speed was over 5.
            j = i
            while j > 0 and speed[j - 1] > 5:
                j -= 1
            start_time = time_ax[j]
            in_segment = True
        # Detect downward crossing: speed falls below threshold when already in a segment.
        elif in_segment and s < Threshold:
            crossings.append([start_time, t])
            in_segment = False
    
    # Remove events that are less than diff_time_s seconds apart.
    filtered = []
    last_end = None
    
    for interval in crossings:
        start, end = interval
        # Always include if no interval has been previously kept.
        if last_end is None or start - last_end >= diff_time_s:
            filtered.append(interval)
            last_end = end
    
    return filtered
def insert_random_times_to_behavior_df(behaviour,baseline_time_s=7*60,framerate=50,n_trials=10):
    N=np.nanmin([behaviour.iloc[0,3],baseline_time_s])
    # Generate 10 random time points with at least 10 seconds between each.
    time_points = generate_random_times(N, num_points=n_trials, min_gap=10)
    frames=time_points*framerate
    frames=frames.astype(int)
    
    

    # Create a new DataFrame with the 10 additional rows.
    new_rows = pd.DataFrame({
        'behaviours': ['random_baseline'] * 10,
        'behavioural_category': ['random_baseline_time'] * 10,
        'start_stop': ['POINT'] * 10,
         'frames_s' : time_points,
         'frames' :frames,
        'video_start_s': [behaviour.iloc[0,5]] *10,
        'video_end_s': [behaviour.iloc[0,6]] * 10  # All rows have the same video_end_s as the first row
    })
    
        # Append the new rows to the existing DataFrame.
    behaviour = pd.concat([behaviour, new_rows], ignore_index=True)
        
        # Optional: display the updated DataFrame
    print(behaviour.tail(15))
    
    return behaviour


def generate_random_times(N, num_points=10, min_gap=10,):
    """Generate sorted random times between 0 and N with a minimum gap between them."""
    while True:
        times = np.sort(np.random.uniform(0, N, num_points))
        if np.all(np.diff(times) >= min_gap):
            return times
def insert_random_times_to_behavior_df(behaviour,baseline_time_s=7*60,framrate=50,n_trials=10):
    N=np.min([behaviour.iloc[0,3],baseline_time_s])
    # Generate 10 random time points with at least 10 seconds between each.
    time_points = generate_random_times(N, num_points=n_trials, min_gap=10)
    frames=time_points*framrate
    frames=frames.astype(int)
    
    

    # Create a new DataFrame with the 10 additional rows.
    new_rows = pd.DataFrame({
        'behaviours': ['random_baseline'] * 10,
        'behavioural_category': ['random_baseline_time'] * 10,
        'start_stop': ['POINT'] * 10,
         'frames_s' : time_points,
         'frames' :frames,
        'video_start_s': [behaviour.iloc[0,5]] *10,
        'video_end_s': [behaviour.iloc[0,6]] * 10  # All rows have the same video_end_s as the first row
    })
    
        # Append the new rows to the existing DataFrame.
    behaviour = pd.concat([behaviour, new_rows], ignore_index=True)
        
        # Optional: display the updated DataFrame
    print(behaviour.tail(15))
    
    return behaviour






def cupy_autocorr(all_times, autocorr_window_ms=500,autocorr_bins=2):
         import cupy as cp
         import numpy as np
         all_times_ms=all_times*1000
         all_times_ms_gpu = cp.array()

         diffs = all_times_ms_gpu[:, None] - all_times_ms_gpu[None, :]
         diffs = diffs.flatten()
         diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
         autocorr_gpu = cp.histogram(diffs, bins=len(diffs))[0]        

         autocorr_gpu= autocorr_gpu/ cp.nanmax(autocorr_gpu) if cp.nanmax(autocorr_gpu) > 0 else autocorr_gpu #normalize
         autocorr_cpu = autocorr_gpu.get()  # Copy from GPU to CPU
         return autocorr_cpu

     
def cross_correlation_cpu(times1, times2, bin_size=1, max_lag=50):
    """
    Compute cross-correlation histogram between two spike trains with vectorized histogram.

    Parameters:
      times1, times2 : arrays of spike timestamps (floats)
      bin_size       : width of the histogram bins (e.g., ms)
      max_lag        : maximum lag to consider (ms)

    Returns:
      bin_centers : centers of time lag bins (NumPy array)
      corr        : counts in each bin of the histogram (NumPy array)
    """
    # Create bins and initialize histogram count array
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    corr = np.zeros(len(bins) - 1, dtype=int)
    
    # Process each spike in times1
    for t in times1:
        dt = times2 - t
        valid = (dt >= -max_lag) & (dt <= max_lag)
        corr += np.histogram(dt[valid], bins=bins)[0]
        
    return bin_centers, corr
import cupy as cp   

####################################
import numpy as np
import os
from joblib import Parallel, delayed
from scipy.signal import correlate
def _compute_normalized_autocorr_trimmed(spike_times_s, bin_size_ms, max_lag_ms, duration_s):
    """
    Helper function to compute the normalized autocorrelation for a single neuron.
    """
    num_spikes = len(spike_times_s)
    max_lag_in_bins = int(max_lag_ms / bin_size_ms)

    # 1. Bin the spike train into a histogram.
    bins = np.arange(0, duration_s * 1000 + bin_size_ms, bin_size_ms)
    binned_spikes, _ = np.histogram(spike_times_s * 1000, bins=bins)
    
    if num_spikes == 0:
        time_lags_ms = np.arange(-max_lag_in_bins, max_lag_in_bins + 1) * bin_size_ms
        return time_lags_ms, np.zeros_like(time_lags_ms, dtype=float)
    
    # 2. Compute the raw autocorrelation.
    raw_corr = correlate(binned_spikes.astype(float), binned_spikes.astype(float), mode='full')
    
    # 3. Trim the correlation to the desired max_lag.
    center_index = len(raw_corr) // 2
    # Ensure we don't try to access indices out of bound 
    lower = max(center_index - max_lag_in_bins, 0)
    upper = min(center_index + max_lag_in_bins + 1, len(raw_corr))
    corr_to_keep = raw_corr[lower:upper]
    
    # Create a time lag array that exactly matches the length of corr_to_keep.
    actual_lags = np.arange(lower - center_index, upper - center_index) * bin_size_ms
    
    # 4. Normalize to conditional firing rate (Hz).
    bin_size_s = bin_size_ms / 1000.0
    normalization_factor = num_spikes * bin_size_s
    normalized_corr_rate = corr_to_keep / normalization_factor

    # 5. Remove the trivial zero-lag peak.
    zero_lag_index = np.where(actual_lags == 0)[0]
    if zero_lag_index.size > 0:
        normalized_corr_rate[zero_lag_index] = 0.0
    # Compute min and max of the autocorrelation values
    min_val = normalized_corr_rate.min()
    max_val = normalized_corr_rate.max()
    
    # Avoid division by zero; if all values are identical, return zeros or ones as appropriate.
    if max_val - min_val > 0:
        normalized_corr_rate = (normalized_corr_rate - min_val) / (max_val - min_val)
    else:
        normalized_corr_rate = np.zeros_like(normalized_corr_rate)

            
    return actual_lags, normalized_corr_rate

def _compute_normalized_autocorr(spike_times_s, bin_size_ms, max_lag_ms, duration_s):
    """
    Helper function to compute the normalized autocorrelation for a single neuron.
    """
    num_spikes = len(spike_times_s)
    max_lag_in_bins = int(max_lag_ms / bin_size_ms)
    time_lags_ms = np.arange(-max_lag_in_bins, max_lag_in_bins + 1) * bin_size_ms
    if num_spikes == 0:
        return time_lags_ms, np.zeros_like(time_lags_ms, dtype=float)

    # 1. Bin the spike train into a histogram.
    bins = np.arange(0, duration_s * 1000 + bin_size_ms, bin_size_ms)
    binned_spikes, _ = np.histogram(spike_times_s * 1000, bins=bins)
    
    if len(binned_spikes) < 2 * max_lag_in_bins + 1:
        raise ValueError(f" duration_s ({duration_s} ms) is too short to compute the full autocorrelation range.")
    
    # 2. Compute the raw autocorrelation.
    raw_corr = correlate(binned_spikes.astype(float), binned_spikes.astype(float), mode='full')
    
    # 3. Trim the correlation to the desired max_lag.
    center_index = len(raw_corr) // 2
    corr_to_keep = raw_corr[center_index - max_lag_in_bins : center_index + max_lag_in_bins + 1]
    
    # 4. Normalize to conditional firing rate (Hz).
    bin_size_s = bin_size_ms / 1000.0
    normalization_factor = num_spikes * bin_size_s
    normalized_corr_rate = corr_to_keep / normalization_factor

    # 5. Remove the trivial zero-lag peak.
    zero_lag_index = np.where(time_lags_ms == 0)[0]
    if zero_lag_index.size > 0:
      #  IPython.embed()
        normalized_corr_rate[zero_lag_index] = 0.0
    
    min_val = normalized_corr_rate.min()
    max_val = normalized_corr_rate.max()

    # Avoid division by zero; if all values are identical, return zeros or ones as appropriate.
    if max_val - min_val > 0:
        normalized_corr_rate = (normalized_corr_rate - min_val) / (max_val - min_val)
    else:
        normalized_corr_rate = np.zeros_like(normalized_corr_rate)

            
    return time_lags_ms, normalized_corr_rate

def get_autocorrs_cursor(session, n_spike_times, n_cluster_index, duration_s, save_path=None, bin_size_ms=1, max_lag_ms=500, n_jobs=1):
    """
    Computes normalized autocorrelations for multiple neurons in parallel.
    """
    num_neurons = len(n_spike_times)
    print(f"Computing autocorrelations for {num_neurons} neurons...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_normalized_autocorr_trimmed)(
            n_spike_times[i], bin_size_ms, max_lag_ms, duration_s
        )
        for i in range(num_neurons)
    )

    print("...computation complete.")
    
    # Unpack results and format into a dictionary
    autocorrs_dict = {}
    t_centers = None
    if results:
        t_centers = results[0][0]
        for i, (lags, corr) in enumerate(results):
            key = str(n_cluster_index[i])
            autocorrs_dict[key] = corr

    if save_path is not None and t_centers is not None:
        save_dir = os.path.join(save_path, "autocorrs")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{session}_autocorrs.npz")
        np.savez(save_file, t_centers=t_centers, **autocorrs_dict)
        print(f"Autocorrelation data saved to '{save_file}'")
        
    return t_centers, autocorrs_dict
def cross_correlation_gpu(times1, times2, bin_size=1, max_lag=50):
    """
    Compute cross-correlation histogram between two spike trains using GPU acceleration.

    Parameters:
      times1, times2 : array-like of spike timestamps (floats)
      bin_size       : width of the histogram bins (e.g., ms)
      max_lag        : maximum lag to consider (ms)

    Returns:
      bin_centers : centers of time lag bins (NumPy array)
      corr        : counts in each bin of the histogram (NumPy array)
    """
    # Create bin edges on GPU
    bins = cp.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    # Transfer spike times to GPU
    times1_gpu = cp.asarray(times1)
    times2_gpu = cp.asarray(times2)

    # Compute all pairwise time differences.
    # This step creates a 2D array of shape (len(times1), len(times2)).
    # For very large arrays, consider processing in blocks.
    differences = times2_gpu[None, :] - times1_gpu[:, None]

    # Restrict differences to [-max_lag, max_lag]
    valid = (differences >= -max_lag) & (differences <= max_lag)
    valid_differences = differences[valid]

    # Compute the histogram on GPU; cupy.histogram supports similar syntax as numpy.histogram
    corr, _ = cp.histogram(valid_differences, bins=bins)
    

    # Bring the results back to CPU (NumPy arrays)
    return cp.asnumpy(bin_centers), cp.asnumpy(corr)
  

def cross_correlation(times1, times2, bin_size=1, max_lag=50):
    """
    Compute cross correlation histogram between two spike trains.

    Parameters:
      times1, times2 : arrays of spike timestamps (floats)
      bin_size       : width of the histogram bins (time units = ms)
      max_lag        : maximum lag to consider (time units = ms)

    Returns:
      bin_centers : centers of time lag bins
      corr        : counts in each bin of the histogram
    """
    # Create bins from -max_lag to max_lag (include rightmost edge)
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    corr = np.zeros(len(bins) - 1)
    # Transfer spike times to GPU
    # times1_cpu = np.array(times1)
    # times2_cpu = np.array(times2)

    # # Compute all pairwise time differences.
    # # This step creates a 2D array of shape (len(times1), len(times2)).
    # # For very large arrays, consider processing in blocks.
    # differences = times2_cpu[None, :] - times1_cpu[:, None]

    # # Restrict differences to [-max_lag, max_lag]
    # valid = (differences >= -max_lag) & (differences <= max_lag)
    # valid_differences = differences[valid]
    # corr, _ = np.histogram(valid_differences, bins=bins)
    
    # For each spike in times1, compute time differences to all spikes in times2
    for t in times1:
        dt = times2 - t
        # Restrict to differences within the desired range
        valid = np.logical_and(dt >= -max_lag, dt <= max_lag)
        counts, _ = np.histogram(dt[valid], bins)
        corr += counts         
    corr= corr/ np.nanmax(corr) if np.nanmax(corr) > 0 else corr #normalize
    
    # Optionally remove the zero-lag (self-spike, can be trivially high)
    # Compute bin centers for plotting
    
    
    
    return bin_centers, corr

from tqdm import tqdm
def get_autocorrs(session,n_spike_times,n_cluster_index,save_path=None,bin_size=1, max_lag=500):
    # Compute autocorrelations:
    num_neurons = len(n_spike_times)

    # Autocorrelations (for each neuron)
    autocorrs = {}
    t_centers_all = None # To store t_centers (assumed to be the same for all neurons)

    for i, times in tqdm(enumerate(n_spike_times), total=num_neurons, desc="Computing autocorrelations"):
        times_ms = times * 1000  # Convert spike times to milliseconds once.
        t_centers, corr = cross_correlation(times_ms, times_ms, bin_size, max_lag)
        
        # Save t_centers from the first iteration.
        if t_centers_all is None:
            t_centers_all = t_centers
        
        # Remove zero-lag peak (self-spikes) if present, to avoid trivial high values.
        zero_index = np.where(np.abs(t_centers) < bin_size/2)[0]
        if zero_index.size > 0:
            corr[zero_index] = 0
        
        # Create a unique key for each neuron using the cluster index and neuron index.
        key = f"{n_cluster_index[i]}"
        autocorrs[key] =  corr
    if save_path is not None:
       
        np.savez(save_path / "autocorrs" / f"{session}_autocorrs.npz",t_centers, **autocorrs)
        print(f"Autocorrelation data saved to '{save_path}\{session} autocorrs.npz'")
    return t_centers,autocorrs
        
    # plt.figure(figsize=(10, 6))
    # for i, times in enumerate(n_spike_times):
    #     t_centers, corr = cross_correlation(times, times, bin_size, max_lag)    
    #     plt.step(t_centers, corr, where='mid', label=f'Neuron {i}')
    
    # plt.xlabel('Time Lag')
    # plt.ylabel('Count')
    # plt.title('Autocorrelations for All Neurons')
    # plt.legend()
    # plt.show()
        
#def save autocorrelations(n_spike_times,out_path):
#        for spike_train in n_spike_times:
#    cupy_autocorr(all_times_ms, autocorr_window_ms=500,autocorr_bins=2)
    
def get_inst_FR(n_spike_times):    
    max_value = 300
    threshold = 250
    
    while max_value > threshold:        
        isi_array = []
        firing_rates = []
        
        # 1st pass
        for i in n_spike_times:
            isi = np.diff(i)
            isi_array.append(isi)
            rate = 1 / isi
            firing_rates.append(rate)
        
        
        # Obtain indices of values below threshold for each array in the list.
        indices_below_threshold = [np.where(arr < threshold)[0] for arr in firing_rates]
        n_spike_times = [
            n_spike_times[i][indices + 1] for i, indices in enumerate(indices_below_threshold)
        ]
        max_value = np.nanmax(np.concatenate(firing_rates))
   
    # Pad firing_rates to make it a 2D array
    max_length = max(len(rate) for rate in firing_rates)
    
    firing_rates_padded = [np.pad(rate, (0, max_length - len(rate)), constant_values=np.nan) for rate in firing_rates]
    firing_rates_array = np.array(firing_rates_padded)  # Convert to 2D numpy array
    
    iFR = firing_rates
    return iFR, firing_rates_array, n_spike_times   


def recalculate_ndata_firing_rates(n_spike_times, bin_size=0.001):
    """
    Recalculates firing rates from neurons' spike times and generates a binary spike matrix.

    This function processes a list of spike time arrays (one per neuron) to produce:
    1. Time bins based on the overall spike time range and a given bin size.
    2. A 2D firing rate array for each neuron over the computed time bins.
    3. A binary matrix indicating the occurrence of spikes at each unique spike time for each neuron.

    Parameters
    ----------
    n_spike_times : list of numpy.ndarray
        A list where each element is a NumPy array containing spike times for a neuron.
    bin_size : float
        The size of each time bin (e.g., in seconds) used for computing firing rates.

    Returns
    -------
    bins_time, ndata, 

    
    ndata = a 1 millisecond binned version of n_spike_times
    bins_time : numpy.ndarray should be 1 milliscond

        
    firing_rates : numpy.ndarray
        A 2D array of shape (n_neurons, num_time_bins) containing the firing rate (spikes per unit time)
        for each neuron in each time bin.
    neurons_by_all_spike_times_binary_array : numpy.ndarray
        A binary matrix of shape (n_neurons, num_unique_spike_times) where each entry is 1 if the neuron
        fired at the corresponding unique spike time, and 0 otherwise.
    neurons_by_all_spike_times_t_seconds: array of timestamps corresponding to neurons_by_all_spike_times_binary_array columns
    """
    
    n_neurons = len(n_spike_times)
    
    # Concatenate all spike times and create corresponding neuron indices
    all_spike_times = np.concatenate(n_spike_times)
    neuron_indices = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(n_spike_times)])
    
    # Determine the time range
    min_time = all_spike_times.min()
    max_time = all_spike_times.max()
    #bins_time = np.arange(min_time, max_time + bin_size, bin_size)
    bins_time = np.arange(0, max_time + bin_size, bin_size)
    
    
    
    
    # Initialize the firing rate matrix
    n_by_t = np.zeros((n_neurons, len(bins_time)))
    
    
    # Calculate the firing rate for each neuron and each time bin
    for i, n in enumerate(n_spike_times):
        neuron_spike_times = n
        spike_nums, _ = np.histogram(neuron_spike_times, bins=bins_time)
        n_by_t[i][:len(spike_nums)] = spike_nums   
    
    
    
    # Create bins for neuron indices; using offset bins so that neuron index i falls into bin [i-0.5, i+0.5)
    bins_neuron = np.arange(-0.5, n_neurons + 0.5, 1)
    
    # Use np.histogram2d to span time and neuron indices.
    # H has shape (len(bins_time) - 1, len(bins_neuron) - 1), where each cell corresponds to the count of spikes.
    H, _, _ = np.histogram2d(all_spike_times, neuron_indices, bins=[bins_time, bins_neuron])
    
    # Transpose so that rows correspond to neurons, and columns to time bins
    counts = H.T
    ndata=n_by_t
    
    # Compute firing rates (spikes per unit time) by dividing counts by bin_size.
    firing_rates = counts / bin_size
    
    # Generate a binary matrix indicating spike occurrences by neuron and unique spike times:
    
    # 1. Concatenate all spike times.
    all_values = np.concatenate(n_spike_times)
    # 2. Build a sorted time axis from unique spike times.
    time_axis = np.sort(np.unique(all_values))
    # 3. Create a lookup dictionary mapping each spike time to its index in time_axis.
    spike_to_index = {t: idx for idx, t in enumerate(time_axis)}
    # 4. Initialize the binary matrix (neurons x unique spike times).
    neurons_by_all_spike_times_binary_array = np.zeros((len(n_spike_times), len(time_axis)))
    
    # 5. Mark entries with 1 for each neuron's spike times.
    for neuron_idx, spikes in enumerate(n_spike_times):
        for spike in spikes:
            col_idx = spike_to_index[spike]
            neurons_by_all_spike_times_binary_array[neuron_idx, col_idx] = 1
    
    
    bin_centers_seconds = (bins_time[:-1] + bins_time[1:]) / 2
    neurons_by_all_spike_times_t_seconds=time_axis
    
        
    return bins_time, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds

def recalculate_ndata_firing_rates2(n_spike_times, bin_size=0.001, firing_rate_bin_size=0.01):
    """
    Recalculates firing rates from neurons' spike times and generates a binary spike matrix.

    This function processes a list of spike time arrays (one per neuron) to produce:
    1. Time bins based on the overall spike time range and a given bin size.
    2. A 2D firing rate array for each neuron over the computed time bins.
    3. A binary matrix indicating the occurrence of spikes at each unique spike time for each neuron.

    Parameters
    ----------
    n_spike_times : list of numpy.ndarray
        A list where each element is a NumPy array containing spike times for a neuron.
    bin_size : float
        The size of each time bin (e.g., in seconds) used for computing the binary spike matrix.
    firing_rate_bin_size : float
        The size of each time bin (e.g., in seconds) used for computing firing rates.

    Returns
    -------
    bins_time, ndata, firing_rates, neurons_by_all_spike_times_binary_array, neurons_by_all_spike_times_t_seconds
    """
    
    n_neurons = len(n_spike_times)
    
    # Concatenate all spike times and create corresponding neuron indices
    all_spike_times = np.concatenate(n_spike_times)
    neuron_indices = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(n_spike_times)])
    
    # Determine the time range
    min_time = all_spike_times.min()
    max_time = all_spike_times.max()
    
    # Bins for binary spike matrix
    bins_time = np.arange(0, max_time + bin_size, bin_size)
    
    # Initialize the binary spike matrix
    n_by_t = np.zeros((n_neurons, len(bins_time)))
    
    # Calculate the binary spike matrix for each neuron and each time bin
    for i, n in enumerate(n_spike_times):
        neuron_spike_times = n
        spike_nums, _ = np.histogram(neuron_spike_times, bins=bins_time)
        n_by_t[i][:len(spike_nums)] = spike_nums   
    
    # Bins for firing rates
    firing_rate_bins_time = np.arange(0, max_time + firing_rate_bin_size, firing_rate_bin_size)
    
    # Create bins for neuron indices; using offset bins so that neuron index i falls into bin [i-0.5, i+0.5)
    bins_neuron = np.arange(-0.5, n_neurons + 0.5, 1)
    
    # Use np.histogram2d to span time and neuron indices for firing rates
    H, _, _ = np.histogram2d(all_spike_times, neuron_indices, bins=[firing_rate_bins_time, bins_neuron])
    
    # Transpose so that rows correspond to neurons, and columns to time bins
    counts = H.T
    
    # Compute firing rates (spikes per unit time) by dividing counts by firing_rate_bin_size
    firing_rates = counts / firing_rate_bin_size
    
    # Generate a binary matrix indicating spike occurrences by neuron and unique spike times:
    
    # 1. Concatenate all spike times.
    all_values = np.concatenate(n_spike_times)
    # 2. Build a sorted time axis from unique spike times.
    time_axis = np.sort(np.unique(all_values))
    # 3. Create a lookup dictionary mapping each spike time to its index in time_axis.
    spike_to_index = {t: idx for idx, t in enumerate(time_axis)}
    # 4. Initialize the binary matrix (neurons x unique spike times).
    neurons_by_all_spike_times_binary_array = np.zeros((len(n_spike_times), len(time_axis)))
    
    # 5. Mark entries with 1 for each neuron's spike times.
    for neuron_idx, spikes in enumerate(n_spike_times):
        for spike in spikes:
            col_idx = spike_to_index[spike]
            neurons_by_all_spike_times_binary_array[neuron_idx, col_idx] = 1
    
    bin_centers_seconds = (bins_time[:-1] + bins_time[1:]) / 2
    neurons_by_all_spike_times_t_seconds = time_axis
    
    return bins_time, n_by_t, firing_rate_bins_time,firing_rates, neurons_by_all_spike_times_binary_array, neurons_by_all_spike_times_t_seconds


@njit
def compute_binary_spike_matrix(n_spike_times, bins_time, n_neurons):
    n_by_t = np.zeros((n_neurons, len(bins_time) - 1))
    for i in range(n_neurons):
        neuron_spike_times = n_spike_times[i]
        spike_nums, _ = np.histogram(neuron_spike_times, bins=bins_time)
        n_by_t[i, :len(spike_nums)] = spike_nums
    return n_by_t

@njit
def compute_firing_rates(all_spike_times, neuron_indices, firing_rate_bins_time, n_neurons, firing_rate_bin_size):
    bins_neuron = np.arange(-0.5, n_neurons + 0.5, 1)
    H = np.zeros((len(firing_rate_bins_time) - 1, len(bins_neuron) - 1))
    for i in range(len(all_spike_times)):
        time_idx = np.searchsorted(firing_rate_bins_time, all_spike_times[i]) - 1
        neuron_idx = int(neuron_indices[i])
        if 0 <= time_idx < H.shape[0] and 0 <= neuron_idx < H.shape[1]:
            H[time_idx, neuron_idx] += 1
    firing_rates = H.T / firing_rate_bin_size
    return firing_rates

@njit
def compute_binary_array(n_spike_times, time_axis):
    n_neurons = len(n_spike_times)
    neurons_by_all_spike_times_binary_array = np.zeros((n_neurons, len(time_axis)))
    for neuron_idx in range(n_neurons):
        spikes = n_spike_times[neuron_idx]
        for spike in spikes:
            col_idx = np.searchsorted(time_axis, spike)
            if col_idx < len(time_axis) and time_axis[col_idx] == spike:
                neurons_by_all_spike_times_binary_array[neuron_idx, col_idx] = 1
    return neurons_by_all_spike_times_binary_array

def recalculate_ndata_firing_rates3(n_spike_times, bin_size=0.001, firing_rate_bin_size=0.001):
    n_neurons = len(n_spike_times)
    all_spike_times = np.concatenate(n_spike_times)
    neuron_indices = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(n_spike_times)])
    min_time = all_spike_times.min()
    max_time = all_spike_times.max()
    
    bins_time = np.arange(0, max_time + bin_size, bin_size)
    n_by_t = compute_binary_spike_matrix(n_spike_times, bins_time, n_neurons)
    
    firing_rate_bins_time = np.arange(0, max_time + firing_rate_bin_size, firing_rate_bin_size)
    firing_rates = compute_firing_rates(all_spike_times, neuron_indices, firing_rate_bins_time, n_neurons, firing_rate_bin_size)
    
    time_axis = np.sort(np.unique(all_spike_times))
    neurons_by_all_spike_times_binary_array = compute_binary_array(n_spike_times, time_axis)
    
    neurons_by_all_spike_times_t_seconds = time_axis
    return bins_time, n_by_t, firing_rate_bins_time, firing_rates, neurons_by_all_spike_times_binary_array, neurons_by_all_spike_times_t_seconds


def unique_float(vector, precision=10):
    """ calculates unique values in vector. It ignores differences that are lower than 'precision' decimals """
    
    rounded_vector = np.round(vector, precision)
    unique_values = np.unique(rounded_vector)
    return unique_values

def mat2npy (filepath, savepath=None, openfile=True):
    """
    Loads mat data as numpy file, or saves it in numpy format
    
    
    Parameters
    ----------
    filepath : Where the .mat file is

    savepath : optional; under what path the new variable should be saved.
                If empty, same path as filepath will be used

    openfile : if True, opens the npy file WITHOUT saving it

    Returns
    -------
    npy data.

    """
    mat_data=scipy.io.loadmat(filepath)
    keys=list(mat_data.keys())
    extracted_data=mat_data[keys[3]]
    
    if savepath==None and openfile==False:        
        np.save(filepath[:-3]+'npy',extracted_data)
    elif savepath!=None and openfile==False:
        np.save(savepath, extracted_data)
    elif openfile==True:
        return np.squeeze(extracted_data)
    
def extract_numbers(string):
    numbers = ''.join(filter(str.isdigit, string))
    return numbers# if numbers else 0

def unique_legend(ax=None, loc='upper right'):
    # Dictionary to keep track of labels and handles
    if ax is None:
        ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    
    # Add a legend with unique labels
    ax.legend(label_dict.values(), label_dict.keys(), loc=loc)



# def endsound():
#     winsound.Beep(300, 150)
#     winsound.Beep(400, 150)
#     winsound.Beep(500, 150)
#     winsound.Beep(700, 500)

# def errorsound():
#     winsound.Beep(500, 150)
#     winsound.Beep(400, 150)
#     winsound.Beep(200, 700)
#     sys.exit()
    
def count_frames(videopath):
    """
    The filelist needs to be separately for cam0 and cam1

    """

    cap = cv2.VideoCapture(videopath)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return frame_count

    
def diff(array,fill=0):
    """
    Inserts a 0 at the start of the array, so that it has the same shape as the input array


    fill : default: 0, changes the value that is used at the first location

    """
    diff=np.diff(array)
    old_shape=np.hstack((fill,diff))
    return old_shape

def load_preprocessed_mac(animal, session, load_pd=False, load_lfp=False):
    """
    Loads the data output from preprocess_all.py
    
    Parameters
    ----------
    animal : str
        The animal identifier.
    session : str
        The session identifier.
    load_pd : bool, optional
        Whether to load photodiode neural data (default is False).
    load_lfp : bool, optional
        Whether to load LFP data (default is False).

    Returns
    -------
    all the stuff from preprocessing
    frames_dropped: frame difference between frame index and velocity vector 
        (positive numbers mean there is more in nidq frame index)
    """
    
    if load_pd and load_lfp:
        raise ValueError('The function is not adapted for loading both PD and LFP simultaneously.')
    
    paths = pp.get_paths_mac(animal, session)
    path = paths['preprocessed'].replace("\\", "/").replace('gpfs.corp.brain.mpg.de', 'Volumes').replace('//', '/')
    
    # Use os.path.join for cross-platform compatibility (Mac/Linux/Windows)
    behaviour = pd.read_csv(os.path.join(path, 'behaviour.csv'))   
    tracking = np.load(os.path.join(path, 'tracking.npy'), allow_pickle=True).item()
    
    velocity = tracking['velocity']
    locations = tracking['locations']
    node_names = tracking['node_names']
    frame_index_s = tracking['frame_index_s']    
    frames_dropped = int(float(paths['frame_loss']))
    
    if load_pd:
        ndata_pd = pd.read_csv(os.path.join(path, 'pd_neural_data.csv'))
        return behaviour, ndata_pd, velocity, locations, node_names, frame_index_s   
    
    ndata_dict = np.load(os.path.join(path, 'np_neural_data.npy'), allow_pickle=True).item()
    ndata = ndata_dict['n_by_t']
    n_time_index = ndata_dict['time_index']
    n_cluster_index = ndata_dict['cluster_index']
    n_region_index = ndata_dict['region_index']
    n_channel_index = ndata_dict['cluster_channels']
    n_spike_times = ndata_dict['n_spike_times']
    
    if load_lfp:
        lfp_dict = np.load(os.path.join(path, 'lfp.npy'), allow_pickle=True).item()
        lfp = lfp_dict['lfp']
        lfp_time = lfp_dict['lfp_time']
        lfp_framerate = lfp_dict['lfp_framerate']
        
        return (frames_dropped, behaviour, ndata, n_time_index, n_cluster_index, 
                n_region_index, n_channel_index, velocity, locations, node_names, 
                frame_index_s, lfp, lfp_time, lfp_framerate)
    else:
        return (frames_dropped, behaviour, ndata, n_time_index, n_cluster_index, 
                n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s)

def load_specific_preprocessed_data (animal, session, varname,load_pd=False ):
    """
    load a specific variable only
    """
    paths=pp.get_paths(animal, session)
    path=paths['preprocessed']
    match varname:# added in python >3.10
        case 'tracking':
            try:
                tracking=np.load(fr'{path}\tracking.npy', allow_pickle=True).item()
            except:
                    print('cannot load tracKing data check session name')
            velocity=tracking['velocity']
            locations=tracking['locations']
            print(f"{locations.ndim=}")
            if locations.ndim==3:
                locations=locations.squeeze()
            node_names=tracking['node_names']
            bottom_node_names=tracking['bottom_node_names']
            frame_index_s=tracking['frame_index_s']    
            frames_dropped=int(float(paths['frame_loss']))
            distance_to_shelter=tracking['distance_to_shelter']
            bottom_distance_to_shelter = tracking['bottom_distance_to_shelter']
            
            return velocity,locations,node_names,bottom_node_names,frame_index_s,frames_dropped,distance_to_shelter,bottom_distance_to_shelter
        
        case 'behaviour':
            behaviour=pd.read_csv(fr'{path}\behaviour.csv')
            return behaviour
        case 'out_path':
            path=paths['preprocessed']
            return path   

    
        case 'ndata':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
            ndata=ndata_dict['n_by_t']
        
        case 'n_time_index':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
            return ndata_dict['time_index']
        
        case 'n_cluster_index':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
            return ndata_dict['cluster_index']
        
        case 'n_region_index':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
            return ndata_dict['region_index']
    
        
        case 'n_channel_index':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
            return ndata_dict['cluster_channels']
        
        case 'df_irc':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()    
            n_spike_times=ndata_dict['n_spike_times']
            return ndata_dict['df_irc'], n_spike_times
        
        case 'n_spike_times':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()    
            return ndata_dict['n_spike_times']
        case    'avg_waveforms':
            ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()    
            return ndata_dict['avg_waveforms']
        
      
    if load_pd==True:
            ndata_pd=pd.read_csv(fr'{path}\pd_neural_data.csv')
            return behaviour,ndata_pd , velocity, locations, node_names, frame_index_s  
def load_preprocessed_mac(animal, session, load_pd=False, load_lfp=False):
    """
    Loads the data output from preprocess_all.py
    
    Parameters
    ----------
    animal : str
        The animal identifier.
    session : str
        The session identifier.
    load_pd : bool, optional
        Whether to load photodiode neural data (default is False).
    load_lfp : bool, optional
        Whether to load LFP data (default is False).

    Returns
    -------
    all the stuff from preprocessing
    frames_dropped: frame difference between frame index and velocity vector 
        (positive numbers mean there is more in nidq frame index)
    """
    
    if load_pd and load_lfp:
        raise ValueError('The function is not adapted for loading both PD and LFP simultaneously.')
    
    paths = pp.get_paths_mac(animal, session)
    path = paths['preprocessed'].replace("\\", "/").replace('gpfs.corp.brain.mpg.de', 'Volumes').replace('//', '/')
    
    # Use os.path.join for cross-platform compatibility (Mac/Linux/Windows)
    behaviour = pd.read_csv(os.path.join(path, 'behaviour.csv'))   
    tracking = np.load(os.path.join(path, 'tracking.npy'), allow_pickle=True).item()
    
    velocity = tracking['velocity']
    locations = tracking['locations']
    node_names = tracking['node_names']
    frame_index_s = tracking['frame_index_s']    
    frames_dropped = int(float(paths['frame_loss']))
    
    if load_pd:
        ndata_pd = pd.read_csv(os.path.join(path, 'pd_neural_data.csv'))
        return behaviour, ndata_pd, velocity, locations, node_names, frame_index_s   
    
    ndata_dict = np.load(os.path.join(path, 'np_neural_data.npy'), allow_pickle=True).item()
    ndata = ndata_dict['n_by_t']
    n_time_index = ndata_dict['time_index']
    n_cluster_index = ndata_dict['cluster_index']
    n_region_index = ndata_dict['region_index']
    n_channel_index = ndata_dict['cluster_channels']
    n_spike_times = ndata_dict['n_spike_times']
    
    if load_lfp:
        lfp_dict = np.load(os.path.join(path, 'lfp.npy'), allow_pickle=True).item()
        lfp = lfp_dict['lfp']
        lfp_time = lfp_dict['lfp_time']
        lfp_framerate = lfp_dict['lfp_framerate']
        
        return (frames_dropped, behaviour, ndata, n_spike_times, n_time_index, n_cluster_index, 
                n_region_index, n_channel_index, velocity, locations, node_names, 
                frame_index_s, lfp, lfp_time, lfp_framerate)
    else:
        return (frames_dropped, behaviour, ndata, n_spike_times, n_tispiketimes,me_index, n_cluster_index, 
                n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s)

def resort_data(sorted_indices):  
    global n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR

    n_region_index = n_region_index[sorted_indices]
    n_cluster_index = n_cluster_index[sorted_indices]
    n_channel_index=n_channel_index[sorted_indices]
    ndata=ndata[sorted_indices,:]
    neurons_by_all_spike_times_binary_array=neurons_by_all_spike_times_binary_array[sorted_indices,:]
    firing_rates=firing_rates[sorted_indices,:]
    
    n_spike_times = [n_spike_times[i] for i in sorted_indices]
    iFR_array=iFR_array[sorted_indices,:]
    iFR = [iFR[i] for i in sorted_indices]
    
    return n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR

def load_preprocessed (animal, session, load_pd=False, load_lfp=False):
    """
    loads the data output from preprocess_all.py
    Parameters
    ----------
    path : path to folder with preprocessed data    
    b_n_t : which output to load (b=behaviour; n=neural; t=tracking)
    Returns
    --------
    all the stuff from preprocessing
    frames_dropped: frame difference between frame index and velocity vector (positive numbers mean there is more in nidq frame index)
    """
    if load_pd and load_lfp:
        raise ValueError('the function is not adapted for that')
    paths=pp.get_paths(animal, session)
    preprocessed_path=str(paths['preprocessed'])
    if os.name == 'posix':
        path_behaviour = Path(preprocessed_path) / 'behaviour.csv'
        path_tracking = Path(preprocessed_path) / 'tracking.npy'
        if not path_behaviour.exists() or not path_tracking.exists():
            raise FileNotFoundError(f"Required files not found in {preprocessed_path}.")
        if not path_tracking.exists():
            raise FileNotFoundError(f"Tracking file not found in {preprocessed_path}.")
    else:  # For Windows
        path_behaviour = fr"{preprocessed_path}\behaviour.csv"
        path_tracking = fr"{preprocessed_path}\tracking.npy"    
    behaviour=pd.read_csv(path_behaviour)
    tracking=np.load(path_tracking, allow_pickle=True).item()
    velocity=tracking['velocity']
    locations=tracking['locations']
    node_names=tracking['node_names']
    frame_index_s=tracking['frame_index_s']    
    frames_dropped=int(float(paths['frame_loss']))
    
    if load_pd:
        if os.name == 'posix':  # For Mac/Linux
            path_ndata_pd = preprocessed_path / 'pd_neural_data.csv'
        else:  # For Windows
            path_ndata_pd = fr"{preprocessed_path}\pd_neural_data.csv"
        ndata_pd=pd.read_csv(path_ndata_pd)
        return behaviour,ndata_pd , velocity, locations, node_names, frame_index_s   
   
    if os.name == 'posix':  # For Mac/Linux
        path_ndata_dict = Path(preprocessed_path) / 'np_neural_data.npy' 
        if not path_ndata_dict.exists():
            raise FileNotFoundError(f"Neural data file not found in {preprocessed_path}.")
    else:  # For Windows
        path_ndata_dict = fr"{preprocessed_path}\np_neural_data.npy"
    ndata_dict=np.load(path_ndata_dict, allow_pickle=True).item()
    ndata=ndata_dict['n_by_t']
    n_time_index=ndata_dict['time_index']
    n_cluster_index=ndata_dict['cluster_index']
    n_region_index=ndata_dict['region_index']
    n_channel_index=ndata_dict['cluster_channels']
    n_spike_times=ndata_dict['n_spike_times']
    
    
    if load_lfp==True:
        if os.name == 'posix':
            path_lfp_dict = Path(preprocessed_path) / 'lfp.npy'
            if not path_lfp_dict.exists():
                raise FileNotFoundError(f"LFP file not found in {preprocessed_path}.")
        else:  # For Windows
            path_lfp_dict = fr"{preprocessed_path}\lfp.npy"
        lfp_dict=np.load(path_lfp_dict, allow_pickle=True).item()
        lfp=lfp_dict['lfp']
        lfp_time=lfp_dict['lfp_time']
        lfp_framerate=lfp_dict['lfp_framerate']
        
        return frames_dropped,behaviour, ndata,n_spike_times, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s, lfp, lfp_time, lfp_framerate
    else:

        return frames_dropped, behaviour, ndata,n_spike_times, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s

    
def mat_struct(file_path, struct_name):
    """
    Read a MATLAB struct from a .mat file.

    Parameters:
    - file_path: Path to the MATLAB file (.mat).
    - struct_name: Name of the struct to read.

    Returns:
    - struct_data: Dictionary containing the data from the MATLAB struct.
    """

    # Open the MATLAB file using h5py
    with h5py.File(file_path, 'r') as mat_file:

        # Check if the specified struct exists in the file
        if struct_name not in mat_file:
            raise ValueError(f"The struct '{struct_name}' does not exist in the MATLAB file.")

        # Extract the struct data
        mat_struct_group = mat_file[struct_name]

        # Convert structured array to dictionary for easier access
        if isinstance(mat_struct_group, h5py.Group):
            struct_data = {name: mat_struct(file_path, struct_name + '/' + name) for name in mat_struct_group.keys()}
        elif isinstance(mat_struct_group, h5py.Dataset):
            struct_data = mat_struct_group[()].tolist()
        else:
            raise ValueError(f"The specified struct '{struct_name}' is not a valid group or dataset in the MATLAB file.")

    return struct_data
    

    
    
def get_paths(session=None, animal=None):
    """
    Gets csv file where paths to datafiles are stored
    works bot if you just specify the animal or just the session
    if you specify nothing, the whole file is being returned
    
    animal/ session names need to be EXACTLY like they are on the csv
   
    Returns
    -------
    paths, either all sessions from all animals, all sessions from one animal,
    or just one session.

    """
    # already defined in preprocess functions!
    pp.get_paths(session=session, animal=animal) 
    # paths=pd.read_csv(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv")
    # if session is None:
    #     if animal is None:
    #         return paths
    #     else:
    #         return paths[paths['Mouse_ID']==animal]
    # else:
    #     if animal is None:
    #         sesline=paths[paths['session']==session]
    #     else:
    #         sesline=paths[(paths['session']==session) & (paths['Mouse_ID'==animal])]
    #     return sesline.squeeze().astype(str)

    
    
    
    
def padded_reshape(array, binfactor, axis=0):
    """
    reshapes array by taking binfactor number of entries and putting them in a new
    axis. Works with 1D or 2D arrays
    If reshape is not evenly possible, the end is padded

    Parameters
    ----------
    array : either vector or array

    binfactor : how many values should make up one new entry?
      
    axis : if array 2D, walong which axis should be reshaped?


    """
    binfactor=np.array(binfactor)
    if binfactor!= binfactor.astype(int):
        raise TypeError ('binfactor must be int value')
    binfactor=binfactor.astype(int).item()
    pad_size = np.ceil(array.shape[axis] / binfactor) * binfactor - array.shape[axis]
    pad_size=pad_size.astype(int)
    old_shape=list(array.shape)

    if (len(old_shape)==2) and (axis==0):

        padded = np.pad(array, ((0,pad_size),(0,0)), 'constant', constant_values=0)
        reshaped = padded.reshape(-1,old_shape[1], binfactor)
        print('check if padding is on correct spot')
    elif (len(old_shape)==2) and (axis==1):

        padded = np.pad(array, ((0,0),(0,pad_size)), 'constant', constant_values=0)
        reshaped = padded.reshape(old_shape[0],-1, binfactor)
    elif (len(old_shape)==1):

        padded = np.pad(array, (0,pad_size), 'constant', constant_values=0)
        reshaped = padded.reshape(-1, binfactor)
    else:
        raise ValueError('this function only works in 1d/2d')
        
    
    return reshaped
    
def resample_ndata(ndata, n_time_index, resolution, method='sum'):
    """
    takes ndata that is neurons*time, and resamples it to new resolution

    Parameters
    ----------
    ndata :  neurons*time, with number of spikes per time bins

    n_time_index : same shape[1] as ndata, with the time in s per timebin
        .
    resolution : the new size of timebins, in s
    
    method: whether to take the mean or the sum in resampling step

    Returns
    -------
    resampled_ndata : neurons*time, but in new resolution
        
    new_time_index : index for each timebin, what is the time

    """

    old_resolution=unique_float(np.diff(n_time_index),precision=10)
    
    if len(old_resolution)!= 1:
        raise ValueError('there is something wrong with the time index')
        
    binfactor=resolution/old_resolution
    
    if not binfactor == binfactor.astype(int):
        raise ValueError('new resolution needs to be a multiple of old resoluition')
    
    
    reshaped_ndata=padded_reshape(ndata, binfactor, axis=1)
    if method=='sum':
        resampled_ndata=np.sum(reshaped_ndata, axis=2)
    elif method=='mean':
        resampled_ndata=np.mean(reshaped_ndata, axis=2)
    
    new_time_index=padded_reshape(n_time_index, binfactor)[:,0]
    
    if np.sum(ndata)!= np.sum(resampled_ndata):
        print('the resampling went wrong')
    
    return resampled_ndata, new_time_index 
    
    
#%% Video things
def count_frames(videopath):
    """
    The filelist needs to be separately for cam0 and cam1

    """

    cap = cv2.VideoCapture(videopath)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return frame_count
    
    
def read_frames(video_path, desired_frames):
    """
    reads only the indicated frames from video, not all
    averages out color dimension. This reduces size, but now cv2 plotting doesn't work anymore
    
    Parameters
    ----------
    video_path : path to video
    desired_frames : list with frame numbers
    """
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError("Could not open video.")
    # Set the frame position

    for i,frame in enumerate(desired_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        # Read frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            cap.release()
            raise ValueError("Video was opened, but could not read frame.")
            
        # average out color dimension
        frame = np.mean(frame, axis=-1)
        
        # save frames to array 
        if i==0:
            all_frames=np.zeros((len(desired_frames),*frame.shape))
        all_frames[i]=frame

    cap.release()

    return np.squeeze(all_frames)
    
# def read_frames(video_path, start_frame, end_frame, bw=True):
#     """
#     reads pixel data for each frame

#     Parameters
#     ----------
#     video_path : path to video
#         DESCRIPTION.
#     start_frame : which should be the first frame?
#         .
#     end_frame : whch should be the last frame to be read?
#         .
#     bw : bool, optional
#         should the 3rd dimension of each frame (color) be averaged out?.
#         The default is True.

#     Returns
#     -------
#     frames_np : np array
#         frames*height*width.

#     """
#     # Load video
#     clip = VideoFileClip(video_path)

#     # Get the frame rate (fps) of the video
#     fps = clip.fps

#     # Convert frame number to time
#     start_time = start_frame / fps
#     end_time = end_frame / fps

#     # Get subclip (i.e., frames between start_time and end_time)
#     subclip = clip.subclip(start_time, end_time)

#     # Convert subclip to frames and then to numpy array
#     frames = [frame for frame in subclip.iter_frames()]
#     frames_np = np.array(frames)
    
#     if bw:
#         frames_np=np.mean(frames_np,axis=3)

#     return frames_np

def convert_s(time_in_seconds):
    """
    Converts time in seconds to hh:mm:ss:msmsms format

    Parameters
    ----------
    time_in_seconds : float number in s, can be array of floats

    Returns
    -------
    time_string : string, time in hh:mm:ss:msms format

    """
    if (not isinstance(time_in_seconds, list)) and (not  isinstance(time_in_seconds,np.ndarray)):
        time_in_seconds=[time_in_seconds]
        
    time_strings=[]
    for i, t in enumerate(time_in_seconds):
        # Calculate hours, minutes, seconds and milliseconds
        hours = t // 3600
        minutes = (t % 3600) // 60
        seconds = (t % 60) // 1
        milliseconds = np.floor((t % 1) * 1000)
    
        # Format the time
        time_strings.append( "{:02d}h:{:02d}m:{:02d}s:{:03d}ms".format(int(hours), int(minutes), int(seconds), int(milliseconds)))

    return np.array(time_strings)



def euclidean_distance_old(vector, point, axis):
    """
    calculates distance between vector and time at each point in time
    
    vector: dimension 0 has to be time, can have multiple other dimensions
    point: same dimensions as vector, dimensions that are misssing shou;d be filled with None
    axis: The x,y axis

    """
    
    sq_diff=np.square(vector-point)
    # axis=tuple(range(1,np.ndim(vector))) # this feels very error prone
    distances=np.sqrt(np.nansum(sq_diff, axis )) 
    return distances

def baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7):
    
    """
    Calculates baseline firing during initial idle period. This function throws an error if a behavioural event
    is annotated during the assigned initial period
    - that are during/ before/ after ANY state/ point behaviour
    - where the animal is not locomoting for some time 
    - 
    Parameters
    ----------
    behaviour : pd table from preprocessing        
    n_time_index :       from preprocessing       
    ndata :              from preprocessing. can be replaced with neurons_by_all_spike_times_array
    initial_period:      in minutes. Default = 7min    

    Raises
    ------
    ValueError
        'choose a shorter initial period'

    Returns
    -------
    mean_frg_hz : For each neuron, what is the average firing rate in Hz
        during the baseline period.


    """
    n_sampling=n_time_index[1]#bins size in seconds
    #choose the initial period in seconds
    initial_period_length = initial_period*60 #minutes to seconds
    
    if behaviour.iloc[0,3] < initial_period_length:
        initial_period_length=behaviour.iloc[0,3]-5
        import warnings
        warnings.warn('choose a shorter initial period', UserWarning)
        
    max_baseline_idx = len(n_time_index[n_time_index < initial_period_length])
    
    baseline_sum = np.sum(ndata[:, :max_baseline_idx], axis=1)
    baseline_time = max_baseline_idx * n_sampling
    mean_firing_hz = baseline_sum/baseline_time   
    return mean_firing_hz


def euclidean_distance(vector, point, axis):
    """
    calculates distance between vector and time at each point in time
    
    vector: dimension 0 has to be time, can have multiple other dimensions
    point: same dimensions as vector, dimensions that are misssing shou;d be filled with None
    axis: The x,y axis

    """

    Px=point[0]
    Py=point[1]
    try:
        x= np.array(vector[:,:,0])
        y= np.array(vector[:,:,1])
    except:
        x= np.array(vector[:,0])
        y= np.array(vector[:,1])
    
    dx = (x - Px)
    dy = (y - Py)
#    sq_diff=np.square(vector-point)
    distance = np.sqrt(dx**2 + dy**2)
    return (dx, dy), distance




def convert_csv_time_to_s(timestamps, start_time):
    # Convert start_time to datetime
    start_time = pd.to_datetime(start_time)

    # Convert timestamps to datetime
    timestamps = pd.to_datetime(timestamps)

    # Calculate time difference in seconds
    time_diff_seconds = (timestamps - start_time).total_seconds()

    return time_diff_seconds


def start_stop_array(behaviour, b_name, frame=False, merge_attacks=None, pre_time=7.5):
    """
    take behaviour pd for one behaviour and turns it into matrix
    with rows for one behaviour, and start, stop in the columns
    IMPORTANT: retruns 'turns' as escape START

    Parameters
    ----------
    behaviour : pd with all behaviours, directly from preprocessing
    b_name: the target behaviuour that should be converted e.g. 'escape'
        (NEEDS TO BE STATE BEHAVIOUR)
    frame: if true, framenumber will be returned, otherwise the s of the frame
    merge_attacks: whether attack periods that follow each other shortly should 
        be taken together. if yes, give the minimum distance in s that attacks 
        should be allowed to have
    pre_time: how many s before escape should loom have occured (note this is before running onset, not turn)

    Returns
    -------
    vec: np array with first column being starts, second column being stops. 
            each row is a new instance of the behaviour

    """
    hunting_bs=['approach','pursuit','attack','eat']
    
    if frame==True:
        f='frames'
    elif frame==False:
        f='frames_s'
    #Get escape frames
    
    try:
        b_pd=behaviour[behaviour['behaviours']==b_name] 
    except:
        b_pd=behaviour['behaviours']==b_name
    
    #Sanity check
    starts=b_pd['start_stop']=='START'
    stops=b_pd['start_stop']=='STOP'    
  
    if b_pd['start_stop'].iloc[0] =='POINT':
        if not b_pd['start_stop'].nunique() ==1:
            raise ValueError ('something is weird here')        
        # print('behaviour is point behaviour, returning vector')
        return b_pd[f].to_numpy()    
    if len(b_pd) ==0:
        raise ValueError(f'{b_name}: specified behaviour doesnt exist')
    if (starts.sum()==0) or (stops.sum()==0):
        
        raise ValueError(f'{b_name}: specified doesnt seem to be state behaviour')
    if not all(starts.iloc[::2]) and all(stops.iloc[1::2]):
        raise ValueError('starts and stops are not alternating')
    if len(b_pd['behaviours'].unique()) > 1:
        raise ValueError('there is more than one behaviour')
    #If behaviour is point behaviour, just return the frames like this
    

    # Make vector
    vec=[]
    acount=0
    for i in range(len(b_pd)):
        if b_pd['start_stop'].iloc[i]!='START':
            continue
        
        #Apppend [start, stop]
        if b_name== 'escape':
            e=b_pd['frames_s'].iloc[i]
            
        
            around_e=behaviour[(behaviour['frames_s']>e-pre_time) & (behaviour['frames_s']<e+.1)]   
            #t=around_e
            t=around_e[around_e['behaviours']=='turn']
            
            #Exclude trials where the loom is too far away
            
            # if not around_e.isin(['loom']).any().any():
            #     print('escape excluded, no loom before')
            #     continue
            
           
            
            # if all(e-t['frames_s']>10):
            #     print('detected turns are more than 10 seconds before escape')
            #     continue
            
            if len(t) >1:
                print('more than one turn before escape. taking the last one')
                t=t.iloc[-1]

                #raise ValueError('more than 1 turn before escape')

            if len(t) == 0: #If there is no turn before escape, use escape start
                t=b_pd.iloc[i]
            
            vec.append([t[f].squeeze(),
                       b_pd[f].iloc[i+1]])
        
           
        
        # elif b_name=='switch':
            
        #     s=b_pd['frames_s'].iloc[i]
            
        #     around_s=behaviour[(behaviour['frames_s']>s-7.5) & (behaviour['frames_s']<s+.1)]   
            
        #     #exclude escapes
        #     if np.sum(np.isin(around_s, hunting_bs))==0:
        #         continue
            
        #     #Exclude trials where the loom is too far away
        #     if not around_s.isin(['loom']).any().any():
        #         continue
            
            
       
            
        #     s=b_pd['frames_s'].iloc[i]
            
        #     around_s=behaviour[(behaviour['frames_s']>s) & (behaviour['frames_s']<s+7)]
            
        #     e_stop=around_s[(around_s['behaviours']=='escape') & 
        #                     (around_s['start_stop']=='STOP')]
            
        #     if len(e_stop)==0:
        #         print('a')
        #         continue #switch not counted, because there was no escape afterwards
            
        #     vec.append([b_pd[f].iloc[i],
        #                 e_stop[f].squeeze()])
        
        # elif (b_name== 'attack') and merge_attacks:
            
        #     # if too  little distance between attack start and previous stop
        #     if ((b_pd[f].iloc[i] - b_pd[f].iloc[i-1])> merge_attacks) or (i==0):            
        #         vec.append([b_pd[f].iloc[i], 
        #                     b_pd[f].iloc[i+1]])

        #     else:
        #         vec[-1][1]=b_pd[f].iloc[i+1]
        #         acount+=1
        else: 
           
                vec.append([b_pd[f].iloc[i], 
                            b_pd[f].iloc[i+1]])

    

    return np.array(vec)


def exclude_switch_trials(event_frames, behaviour , window=10, startstop=0, unit='s', return_mask=False):
    """"excludes those events where there is a switch happening in the window s
    before or after the event
    ___________________
    Parameters:
        event_frames: vector with event times, column 0 is starts, column 1 is stops (as output from hf.start_stop_array)
        behaviour: the behaviour df that is output from preprocessing
        window:  scalar of how long before and after the event a switch should be considered for exclusion of trial
            ALWAYS in s
            
        startstop: if 0, starts are used as eventtime, otherwise stops
        unit: if s: event_frames and window has to be in seconds
              if f: eventrframes and window has to be in frames
        return_mask: should the mask used to exclude switch trials be returned?
    """

    if unit=='f':
        window*=50
        if window> 30*50:
            raise ValueError('window seems a bit large, check that you have it in s')
    
    #Deal with point behaviours
    if len(event_frames.shape)==1:
        event_frames=event_frames[:,None]
    
    
    mask=np.ones(len(event_frames), dtype=bool)
    
    if unit=='s':
        frame_column='frames_s'
    elif unit=='f':
        frame_column='frames'
    else:
        raise ValueError("wrong value for 'unit'")
    
    for i,ev_frame in enumerate( event_frames[:,startstop]): # the 0 is to only take the starts
        
        #get behaviours around event
        windowstart=ev_frame-window
        windowstop=ev_frame+window
        around_event=behaviour[(behaviour[frame_column]>=windowstart) & (behaviour[frame_column]<=windowstop)]
    
        
        #excluse escapes with a switch 5 sec before or after
        if 'switch' in around_event['behaviours'].values:
            
            mask[i]=0
            
    if return_mask:
        return np.squeeze(event_frames[mask,:]), mask
    return np.squeeze(event_frames[mask,:])

def peak_escape_vel(behaviour, velocity, exclude_switches=True):
    """
    get peak velocity for each escape

    Parameters
    ----------
    behaviour : df from preprocesing
    velocity: np vector from preprocessing
    exclude_switches: should switch escapes be excluded?

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    start_stop=start_stop_array(behaviour, 'escape', frame=True)
    if exclude_switches:
        start_stop=exclude_switch_trials(start_stop, behaviour, 5*50, 0, unit='f')
    peak_vels=[]
    for escape in start_stop:
        vel=velocity[escape[0]:escape[1]]
        peak_vels.append(np.nanmax(vel))
    
    return start_stop, np.array(peak_vels)


def baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s, window=5, vel_cutoff=[7, 3]):
    """
    Calculates baseline firing during exploration periods. For this, 
    periods are excluded...
    - that are during/ before/ after ANY state/ point behaviour
    - where the animal is not locomoting for some time 
    - 
    Parameters
    ----------
    behaviour : pd table from preprocessing        
    n_time_index :       from preprocessing       
    ndata :              from preprocessing       
    velocity :           from preprocessing
    frame_index_s :      from preprocessing
    
    window : int, optional
        How many s before and after a point/ state behaviour should the neural activity 
        be excluded from baseline. The default is 5s.
    vel_cutoff : list, optional
        first number is below what value the velocity has to drop,
        second number is for what period of time, so that the neural activity is excluded.
        The default is [7 cm/s, 3s].

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    mean_frg_hz : For each neuron, what is the average firing rate in Hz
        during the baseline period.
    base_ind : boolean index, True during baseline period. Shape matches n_time_index


    """
    
    if n_time_index[0] != 0:
        raise ValueError('fix your n_time_index!!!!')
        
    n_sampling=n_time_index[1]
    
    #get baseline
    base_ind=np.ones(ndata.shape[1]).astype(int)
    
    #Cut out periods of behaviour
    for b_name in behaviour['behaviours'].unique():
        b=behaviour[behaviour['behaviours']==b_name]
        
        if b['start_stop'].iloc[0]=='START':
            start_stop=start_stop_array(behaviour, b_name, frame=False)
            for start_stop_s in start_stop:
                ind=(n_time_index>(start_stop_s[0]-window)) & (n_time_index<(start_stop_s[1]+window))
                base_ind-=ind
        
        elif b['start_stop'].iloc[0] =='POINT':
            for event_s in b['frames_s']:
                ind=(n_time_index>(event_s-window)) & (n_time_index<(event_s+window))
                base_ind-=ind
        
        else:
            raise ValueError('unexpected content of start_stop, check this')
    
    #Cut out periods of low velocity

    interp_vel=interp(frame_index_s, velocity, n_time_index)
    num_consecutive_samples = int(vel_cutoff[1] / n_sampling)
    
    # Create a rolling window of the required size and check if all velocities are below 7
    rolling_windows = sliding_window_view(interp_vel, num_consecutive_samples)
    mask = np.all(rolling_windows < vel_cutoff[0], axis=-1)
    mask = np.pad(mask, (num_consecutive_samples - 1, 0), mode='constant', constant_values=False)
    
    base_ind-= mask
    
    
    #Get final index/ mask
    base_ind[base_ind!=1]=0
    base_ind=base_ind.astype(bool)
    
    #calculate baseline firing
    base_sum=np.sum(ndata[:,base_ind], axis=1)
    base_time=np.sum(base_ind)*n_sampling
    mean_frg_hz=base_sum/ base_time
    
    print(f'baseline period covers {convert_s(base_time)[0]}')
    
    return mean_frg_hz, base_ind

def nanmean(array,fill=0, axis=None):
    if np.sum(np.isinf(array))!=0:
        raise ValueError('there are inf values in this array!')
    return np.mean(np.nan_to_num(array, nan=fill), axis)

def interp(t, vector, new_t):
    #Cut out periods of low velocity
    interp_func = interp1d(t, vector, bounds_error=False, fill_value="extrapolate")
    interpolated_vector= interp_func(new_t)
    return interpolated_vector


#%% Regression

def regression(X,Y):

    #Centering X and Y to the mean


    # Calculate covariance matrix of X
    CXX = np.dot(X.T,X) # + sigma * np.identity(np.size(X,1))
    # Calculate covariance matrix of X with Y
    CXY = np.dot(X.T,Y)
    # Regress Y onto X/ calculate the B that gets you best from X to Y
    B_OLS = np.dot(np.linalg.pinv(CXX), CXY)
    # Use the cacluated B to make a prediction on Y
    Y_OLS = np.dot(X,B_OLS)
    # Perform SVD on the predicted Y values
    # _U, _S, V = np.linalg.svd(Y_OLS, full_matrices=False)

    return Y_OLS, B_OLS


 
def OLS_regression(X,Y,nfolds=5, random_state=None, normalise=True):
    """
    

    Parameters
    ----------
    X : Predictor; time*neurons
        .
    Y : Predicted; time*neurons
        .
    nfolds : int; how often cross validation
        Data is divided into nfolds random splits (not consecutive; along time dimension)
        each of the folds is predicted separately by all the other folds together.
        The default is 5.
    random_state : int, optional
        For always having the same random split. The default is None.
    normalise : Bool, optional
        Whether the data should be zscored. The default is True.

    Returns
    -------
    perf_r2 : Vector
        r2 (explained variance) per factor.
    bs : 3d matrix
        regression weights fold*X*Y .

    """
    

    is_gpu = False
    sigma=0
    kfolds = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    taxis = range(X.shape[0])
    perf_r2=[]
    bs=[]

    for ifold, (train, test) in enumerate(kfolds.split(taxis)):
        Xtrain=X[train,:]
        Ytrain=Y[train,:]
        Xtest=X[test,:]
        Ytest=Y[test,:]
        
        if normalise:
            Xtrain=np.nan_to_num(zscore(Xtrain, axis=0))
            Ytrain=np.nan_to_num(zscore(Ytrain, axis=0))
            Xtest=np.nan_to_num(zscore(Xtest, axis=0))
            Ytest=np.nan_to_num(zscore(Ytest, axis=0))

        _, B_OLS = regression(Xtrain,Ytrain)
        # Calculate TEST error by multiplying unseen X values with B 
        Ytestpred = np.dot(Xtest,B_OLS)
        test_err = Ytest - Ytestpred
        mse=np.sum(np.square(test_err),axis=0)
        sst=np.sum(np.square(Ytest-np.mean(Ytest, axis=0)), axis=0)
        perf_r2.append(1-(mse/sst))
        bs.append(B_OLS)

    perf_r2=np.mean(np.array(perf_r2), axis=0)

    return perf_r2, np.array(bs)

def regression_no_kfolds(X,Y):

    #Centering
    mX = np.mean(X, axis = 0,keepdims=True)
    mY = np.mean(Y, axis = 0, keepdims=True)
    stdX=np.std(X,axis=0)
    stdY=np.std(Y, axis=0)
    X = (X - mX)/stdX
    Y = (Y - mY)/stdY


    Y_OLS, B_OLS = regression(X,Y)
    
    #Calculate r2
    test_err = Y_OLS - Y
    mse=np.sum(np.square(test_err),axis=0)
    sst=np.sum(np.square(Y-np.mean(Y, axis=0)), axis=0)
    r2=1-(mse/sst)
    

    return Y_OLS, B_OLS, r2

#%%Permutation
def shift_data(data, num_shift, column_direction=False):
    """
    assumes input to be 2d
    Positive shift values: Data is shifted forward, bottom values are moved on top
    negative shift: Data is shifted backwards, top values are moved to bottom

    Parameters
    ----------
    data : 2d
        .
    num_shift : TYPE
        DESCRIPTION.
    column_direction : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    shifted_data : TYPE
        DESCRIPTION.

    """
    
    if column_direction==True:
        data=data.T
    
    top=data[:-num_shift,:]
    bottom=data[-num_shift:,:]
    shifted_data=np.concatenate((bottom,top),axis=0)
    
    if column_direction==True:
        shifted_data=shifted_data.T
    
    return shifted_data


def rand_base_periods (base_ind, n_time_index, num_base_samples, base_period_length, overlapping=False):
    """
    gets random samples of non-overlapping baseline periods, as a control for behaviours

    Parameters
    ----------
    base_ind : Boolean 1D vector
        indicates periods of baseline; output from hf.baseline_firing() .
    n_time_index : float 1D vector
        indicates time in s for each entry in base_in; output from preprocessing.
    num_base_samples : int
        how many samples/ periods should be drawn?.
    base_period_length : float; in s
        how long should each period be?

    Returns
    -------
    base_start_stop_s : 1D float vector; in s
        in s; each row is a baseline period, column[0] is start, column [1] is stop .

    """
   
    
   
    #NOTE:
   
    """
   
    -You could solve your current problem (shit takes too long) by 
    calculating average necessary distance between two random periods 
    and then taking random sample within that period
    
    -otherwise do as you did before, it doesn't feel so bad
    
    
    """
    
    
    # initial computations
    n_srate=len(n_time_index)/n_time_index[-1]
    num_base_samples=int(num_base_samples)
    base_period_length = int(base_period_length*n_srate)
    
    # test if command is possible
    if (num_base_samples * 2* base_period_length > np.sum(base_ind)) and not overlapping:
        max_samples=np.sum(base_ind)/ (2*base_period_length)
        raise ValueError (f'With this sample length, you can have a maximum of {int(max_samples)} samples')
    
    # Exclude periods that are too close to behaviour
    n=base_period_length*2 # distance to the sides
    padded_base = np.convolve(base_ind, np.ones(n, dtype=int), mode='same') >= n
    
    # get random periods, that are at least  'base_period_length' apart
    diff=0 
    if not overlapping:
        while np.min(diff)<=((base_period_length)/n_srate):
            base_starts=np.sort(choice(n_time_index[padded_base], size=num_base_samples, replace=False ))
            diff=np.diff(base_starts)
    else:
        base_starts=np.sort(choice(n_time_index[padded_base], size=num_base_samples, replace=False ))
        diff=np.diff(base_starts)
        print(f'random samples average diff: {np.median(diff)}')
        
        #wrong unit, needs to be /srate
        
    
    base_stops=base_starts.copy() + (base_period_length/ n_srate)
    base_start_stop_s=np.array([base_starts,base_stops]).T
    return base_start_stop_s


def convert_loc_to_cm_big_arena(px_coords,py_coords): #this function is not in use
    radius_cm = float(input("Enter the radius of the arena in centimeters: "))
    
    # Calculate the factor to convert pixels to centimeters
    max_pixel_radius = np.max(np.sqrt(px_coords**2 + py_coords**2))
    conversion_factor = radius_cm*2 / max_pixel_radius
    
    return px_coords*conversion_factor,py_coords*conversion_factor


def plot_and_convert(locations, vframerate, cm_x=45, cm_y=45):
    """
    Plots coordinates and allows the user to select a shelter location by clicking on the plot.
    Returns the conversion factors and the selected shelter location.
    
    Parameters:
    - locations: numpy array of shape (N, 2, 2) with coordinates
    - vframerate: int, downsampling rate
    - cm_x, cm_y: dimensions of the arena in centimeters
    
    Returns:
    - conversion_factor: list with [x_conversion_factor, y_conversion_factor]
    - shelter_point: numpy array with [shelter_x, shelter_y]
    """
    # Downsample the data based on the frame rate
    vframerate = int(vframerate)
    x_coords = locations[::vframerate, 1, 0]
    y_coords = locations[::vframerate, 1, 1]
    
    # Initialize shelter_point
    shelter_point = None
    
    # Function to handle mouse clicks
    def onclick(event):
        nonlocal shelter_point
        shelter_point = np.array([event.xdata, event.ydata])
        print(f"Shelter location selected: ({shelter_point[0]:.2f}, {shelter_point[1]:.2f}) pixels")
        plt.close()  # Close the plot after selecting the point

    # Create and display the plot
    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, 'o', markersize=2)
    ax.set_title("Click to select shelter location")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("Please click on the plot to select the shelter location.")
    plt.show()  # Display the plot and wait for user interaction

    # If no point is selected (user closes the plot), raise an error
    if shelter_point is None:
        raise ValueError("No point was selected. Please run the function again and click on the plot.")
    
    # Prompt the user for arena dimensions in centimeters
    try:
        cm_x = float(input("Enter the X length of the arena in centimeters: "))
        cm_y = float(input("Enter the Y length of the arena in centimeters: "))
    except ValueError:
        print("Invalid input. Using default values for cm_x and cm_y.")
    
    # Calculate the conversion factors from pixels to centimeters
    max_x, min_x = np.max(x_coords), np.min(x_coords)
    max_y, min_y = np.max(y_coords), np.min(y_coords)
    sum_x, sum_y = max_x - min_x, max_y - min_y

    x_conversion_factor = cm_x  / sum_x
    y_conversion_factor = cm_y  / sum_y
    
    conversion_factor = [x_conversion_factor, y_conversion_factor]
    print(f"Conversion factor: {conversion_factor} cm/pixel")
    
    return conversion_factor, shelter_point



# def plot_and_convert(locations,vframerate,cm_x=45,cm_y=45):
#     # we assume that N pixels x == N pixels Y
#     # Plot every 50th x, y coordinate
#     vframerate = int(vframerate)
#     x_coords = locations[::vframerate, 1, 0]
#     y_coords = locations[::vframerate, 1, 1]
    
#     fig, ax = plt.subplots()
#     ax.plot(x_coords, y_coords, 'o', markersize=2)
#     ax.set_title('Click to select shelter location as the mean upper edge shelter')
#     ax.set_xlabel('X (pixels)')
#     ax.set_ylabel('Y (pixels)')
    
#     shelter_point = np.array([0, 0])  # Initialize shelter point
    
#     # Function to handle click event
#     def onclick(event):
#         nonlocal shelter_point
#         shelter_x, shelter_y = event.xdata, event.ydata
#         shelter_point = np.array([shelter_x, shelter_y])
#         print(f'Shelter location: ({shelter_x}, {shelter_y}) pixels')
#         cid = fig.canvas.mpl_connect('button_press_event', onclick)
#         fig.canvas.mpl_disconnect(cid)
#         plt.close(fig)  # Close the plot after clicking
    
#     # Connect the clicnnect('button_press_event', onclick)
    
#     plt.show(block=False)
    
#     # Wait for the user to click on the plot
#     plt.waitforbuttonpress()
    
#     # Ask user for the radius of the arena in centimeters or just hard coded for now
#     cm_x = float(input("Enter the X length of the arena in centimeters: "))
#     cm_y = float(input("Enter the Y length of the arena in centimeters: "))
    
#     # Calculate the factor to convert pixels to centimeters
#     circle_max_pixel = np.max(np.sqrt(x_coords**2 + y_coords**2))
#     max_x=np.max(x_coords);min_x=np.min(x_coords)
#     max_y=np.max(y_coords);min_y=np.min(y_coords)
#     sum_x=max_x-min_x;sum_y=max_y-min_y

#     x_conversion_factor = cm_x*2 / sum_x
#     y_conversion_factor = cm_y*2 / sum_y
    
    
#     conversion_factor=[x_conversion_factor,y_conversion_factor]
#     print(f'Conversion factor: {conversion_factor} cm/pixel')
    
#     return conversion_factor, shelter_point




# def plot_and_convert(locations):
#     # Plot every 50th x, y coordinate
#     x_coords = locations[::50, 1, 0]
#     y_coords = locations[::50, 1, 1]
    
#     fig, ax = plt.subplots()
#     ax.plot(x_coords, y_coords, 'o', markersize=2)
#     ax.set_title('Click to select shelter location')
#     ax.set_xlabel('X (pixels)')
#     ax.set_ylabel('Y (pixels)')
    
#     shelter_point = np.array([0, 0])  # Initialize shelter point
    
#     # Function to handle click event
#     def onclick(event):
#         nonlocal shelter_point
#         shelter_x, shelter_y = event.xdata, event.ydata
#         shelter_point = np.array([shelter_x, shelter_y])
#         print(f'Shelter location: ({shelter_x}, {shelter_y}) pixels')
#         fig.canvas.mpl_disconnect(cid)
    
#     # Connect the click event to the handler
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
#     plt.show()
    
#     # Ask user for the radius of the arena in centimeters or just hard coded for now
#     radius_cm = 45 #float(input("Enter the radius of the arena in centimeters: "))
    
#     # Calculate the factor to convert pixels to centimeters
#     max_pixel_radius = np.max(np.sqrt(x_coords**2 + y_coords**2))
#     conversion_factor = radius_cm*2 / max_pixel_radius
    
#     print(f'Conversion factor: {conversion_factor} cm/pixel')
    
#     return conversion_factor, shelter_point

def check_and_convert_variable(x):
    
    if  isinstance(x, np.ndarray):
        return x
    if isinstance(x, str): #x is a string

        if '.' in x:
            x = list(map(float, x.split(',')))
        else:            
            x = list(map(int, x.split(',')))
            
    elif isinstance(x, int):
        if np.isnan(x): #x is NaN
            x=[]            # Do nothing if x is NaN
        else:#x is a float
            x = list(map(int, str(x).split(',')))
    elif isinstance(x, list): # x is a list
        # No conversion needed for lists
        return x
    elif x is None: #x is None
        x=[]
    else: #x is of an unknown type
        x=[]

    return x


def get_shelterdist(paths, node_locations, vframerate, vpath, locate_shelter=True):
   
    if (locate_shelter == True):
        
        from trajectory_calibration3 import calibrate_trajectory
        
        csv_path=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv"
        row_number=paths.name
        
        
        if node_locations.ndim==3: 
            px_to_cm_x, px_to_cm_y, shelterpoint, distance2shelter = calibrate_trajectory(x_trajectory=node_locations[:,:,0], y_trajectory=node_locations[:,:,1],video_path=vpath,csv_path=csv_path,row_number=row_number)
        elif node_locations.ndim==2: #from preprocessed data
            px_to_cm_x, px_to_cm_y, shelterpoint, distance2shelter = calibrate_trajectory(x_trajectory=node_locations[:,0], y_trajectory=node_locations[:,1],video_path=vpath,csv_path=csv_path,row_number=row_number)
        else:
            raise Exception('Error: unexpected node_locations.ndim ')
        pixel2cm=[px_to_cm_x, px_to_cm_y]
        shelter_point = np.zeros((2))
        shelter_point[0] = shelterpoint[0] * px_to_cm_x
        shelter_point[1] = shelterpoint[1] * px_to_cm_y
       # node_locations[:,:,0]=node_locations[:,:,0] * float(pixel2cm[0])
#        node_locations[:,:,1]=node_locations[:,:,1] *float(pixel2cm[1])
        
    else:
       
        shelter_point = np.zeros(2)
        pixel2cm=paths['Cm2Pixel_xy']        
        numbers_str = pixel2cm.split()
        px, py = map(float, numbers_str) # Convert each element to float using map
        
        shelter_point[0], shelter_point[1]=map(float, paths['Shelter_xy'].strip('[]').split())
        
        #node_locations[:,:,0]=node_locations[:,:,0]* px
#        node_locations[:,:,1]=node_locations[:,:,1] *py
       
        _,distance2shelter=euclidean_distance(node_locations, shelter_point, axis=1) 
   
    # shelterpoint=np.ndarray((2,1))      
    # shelterpoint[0]=shelter_point[0]*pixel2cm[0]
    # shelterpoint[1]=shelter_point[1]*pixel2cm[0]
       
  
    return distance2shelter, pixel2cm, shelter_point



def sample_neurons(ndata,n_region_index, res, nmb_n,  target_frg_rate, tolerance, return_factors=None):
    """
    Draws semi-random sample from neurons. Final sample has mean: 'target_frg_rate'
    +/- tolerance. Optional: return SVD factors, calculated on sample of neurons

    Parameters
    ----------
    ndata : from preprocessing
    n_region_index : from preprocessing

    res : what is the resolution of the data (first entry of n_time_index)

    nmb_n : int
        how many neurons should be drawn as sample
   
    target_frg_rate : float
        what firing rate shoul dthe samples from each area have
        
    tolerance : float
        What deviation from this avg frg rate is allowed?

    return_factors : int, optional
        If specified, how many factors SVD factors should be returned
        Otherwise, nmb_n neurons are returned. The default is None.


    Returns
    -------
    n_sample:  time* factors / neurons*time matrix with chosen neurons
    
    areass_sample: areanames for neurons
    
    OPTIONAL:
        s2_factors: s^2 values for the returned factors

    """

    avg_hz=np.mean(ndata, axis=1)/res
    areas=np.unique(n_region_index)
    
    n_sample=[]
    areas_sample=[]
    s2_factors=[]
    for aname in areas:
        region_ind=np.isin(n_region_index, aname)
        
        
        #check if area has enough neurons
        if np.sum(region_ind)<nmb_n:
            print(f'{aname} skipped, too few neurons')
            continue
        
        #Choose samples with same avg firing
        pop_avg=target_frg_rate-2*tolerance# just to  have a value outside of alowed range
        i=0
        while np.abs(pop_avg-target_frg_rate)>tolerance:
            n_ind=np.random.choice(np.where(region_ind)[0], nmb_n, replace=False)
            pop_avg=np.mean(avg_hz[n_ind])
            
            #break the loop, if conditions cant be fulfilled by area
            i+=1
            if i>10000:
                raise ValueError(f'sample cant be taken from {aname} with current restrictions')
  
        
        # collect neurons in vector
        if return_factors is None:
            n_sample.append(ndata[n_ind,:])
            areas_sample.append(n_region_index[n_ind])
            
        # OR collect factors
        else:
            n=zscore(ndata[n_ind,:], axis=1).T
            U, s, VT = np.linalg.svd(n, full_matrices=False)
            
            n_sample.append(U[:,:return_factors])
            areas_sample.append([aname]*return_factors)
            s2_factors.append(s[:return_factors]**2)
            
            
            
    if return_factors is None:
        return np.vstack(n_sample), np.hstack(areas_sample)
    else:
        return np.hstack(n_sample), np.hstack(areas_sample), np.hstack(s2_factors)



# # choose neurons with  high firing rate
# region_ind=np.isin(n_region_index, aname)
# sort_ind=np.argsort(avg_hz)
# sort_region_ind=np.isin(ind, np.where(region_ind)[0])
# n_ind=sort_ind[sort_region_ind][:nmb_n]




        
def get_bigB(animal, locs=False, get_sessionstarts=False):
    """
    Collects all the boris annotations from one animal in one long pandas
    and adapts the frame_index_s to match this

    Parameters
    ----------
    animal : str
        EXACT animal name like it is in csv.

    Returns
    -------
    bigB : pd matrix 
        like behaviour.csv, but concatenated from all sessions.
        time values are counted continuously from the start of the first session.
    all_frame_index : concatenated frme_index_s, with continuously increasing time.
.

    """
    from pathlib import Path
    paths=get_paths(None, animal)
    sessionlist=paths['session']
    
    previous_dur_s=0
    previous_dur_f=0
    bigB=[]
    all_locs=[]
    all_frame_index=[]
    all_vel=[]
    sessionstarts=[]
    for ses in sessionlist:  
        
        #Load data
        datapath=paths.loc[paths['session']==ses, 'preprocessed'].values[0]
        if datapath is np.nan:
            print(f'{ses} skipped, no preprocessing path')
            continue
        if os.path.exists(datapath):
            print("The path exists.")
        else:
            print("The path does not exist.")
            continue
        if os.path.exists(fr'{datapath}/behaviour.csv')==False:
            print("The path does not exist.")
            continue
          
        dropped=int(paths.loc[paths['session']==ses, 'frame_loss'].values[0])

         
        behaviour=pd.read_csv(fr'{datapath}/behaviour.csv')
        
        tracking=np.load(fr'{datapath}\tracking.npy', allow_pickle=True).item()
        if dropped != 0:
            frame_index_s=tracking['frame_index_s'][:-dropped]  
            locations=tracking['locations'][:-dropped]
        else:
            frame_index_s=tracking['frame_index_s']
            locations=tracking['locations']
        velocity=tracking['velocity']

        #check if data is aligned
        if len(velocity)!=len(frame_index_s):
            raise ValueError(f'{ses} datastreams are misaligned by {len(velocity)-len(frame_index_s)} frames')
        
        behaviour['frames_s']+=previous_dur_s
        behaviour['frames']+=previous_dur_f

        
        bigB.append(behaviour)
        all_frame_index.append(frame_index_s+previous_dur_s)
        all_vel.append(velocity)
        all_locs.append(locations)
        
        previous_dur_f+=len(frame_index_s)
        previous_dur_s+=frame_index_s[-1]
        sessionstarts.append(previous_dur_s)
        
        
        # plt.figure();plt.title(ses);plt.plot(behaviour['frames_s'].diff())
        # print(f'{ses}\n{frame_index_s[-1]}\n{previous_dur}\n\n')
    print(bigB) 
    bigB=pd.concat(bigB)
    all_frame_index=np.hstack(all_frame_index)
    all_vel=np.hstack(all_vel)
    
    if locs:
        all_locs=np.concatenate(all_locs, axis=0)
        return bigB, all_frame_index, all_vel, all_locs
    if get_sessionstarts:
        return bigB, all_frame_index, all_vel, sessionstarts
    return bigB, all_frame_index, all_vel

def bigB_multiple_animals(animals, get_locs=False):
    """
    Collects all the boris annotations from one animal in one long pandas
    and adapts the frame_index_s to match this
    if multiple animals are given, it concatentes all sessions from all animals

    Parameters
    ----------
    animal : str
        EXACT animal name like it is in csv.

    Returns
    -------
    bigB : pd matrix 
        like behaviour.csv, but concatenated from all sessions.
        time values are counted continuously from the start of the first session.
    all_frame_index : concatenated frme_index_s, with continuously increasing time.
    animal_borders: when do annotations from new animal start

    """
    
    bigB=[]
    all_frame_index=[]
    all_vel=[]
    prev_num_frames=0
    animal_borders=[]
    all_locs=[]
    for i, animal in enumerate(animals):
        
        smallB, frame_index, vel, locs=get_bigB(animal, True)
        
        if i==0:
            all_frame_index.append(frame_index)
            last_t_end=0
            res=0
        else:
            all_frame_index.append(frame_index+last_t_end+res)
        
        smallB['frames_s']+= last_t_end+res
        smallB['frames']+= prev_num_frames
        bigB.append(smallB)
        all_vel.append(vel)
        all_locs.append(locs)
        
        last_t_end=all_frame_index[-1][-1]
        prev_num_frames+= len(frame_index)
        res=frame_index[1]
        animal_borders.append(prev_num_frames)
    if get_locs:
        return pd.concat(bigB), np.hstack(all_frame_index), np.hstack(all_vel), np.concatenate(all_locs, axis=0)
    return pd.concat(bigB), np.hstack(all_frame_index), np.hstack(all_vel), animal_borders

        
        
        

def divide_looms(all_frame_index, bigB, radiance=2):
    """
    divide looms into looms that happen during hunt vs 
    looms that happen outside of hunting

    Parameters
    ----------
    all_frame_index : from hf.get_bigB
        .
    bigB : from hf.get_bigB
        .
    radiance : float, in s
        how long before/ after a hunting event must a loom happen to be considered 'other'. The default is 2.

    Returns
    -------
    otherlooms : list, in s
        when do looms happen that are outside of hunting.

    huntlooms : list, in s
        when do looms happen that are during hunting.

    """
    
    b_names=bigB['behaviours'].to_numpy()
    frames_s=bigB['frames_s'].to_numpy()
    
    hunting_bs=['approach','pursuit','attack', 'eat']
    
    #Get index for when hunting is happening
    hunt_ind=np.zeros_like(all_frame_index)
    for i, b_name in enumerate(hunting_bs):
    
        start_stop=start_stop_array(bigB, b_name, frame=False)
    
            
                
        for b in start_stop:
            hunt_ind+=(all_frame_index>(b[0])) & (all_frame_index<b[1])
    hunt_ind=hunt_ind.astype(bool)
           
    
    #Divide looms in during hunt and other
    huntlooms=[]
    otherlooms=[]
    for loom in frames_s[b_names=='loom']:
        
        around_loom=hunt_ind[(all_frame_index>loom-radiance )& (all_frame_index< loom+radiance)]
    
        if np.sum(around_loom) == 0:
            otherlooms.append(loom)
        elif np.sum(around_loom) > 0:
            huntlooms.append(loom)
        else:
            raise ValueError ('sth is weird here')
    return otherlooms, huntlooms

import itertools

def combos(n, skip_singulars=False):
    # Create a list of variables
    variables = range(n)
    
    # Generate all combinations
    combinations = []
    for r in range(1, n+1):
        combinations.extend(itertools.combinations(variables, r))

    
    if skip_singulars:
        combinations=combinations[n:]
    
    
        
    return combinations

def test_normality(data):
    ad_test = stats.anderson(data)
    print(f"Anderson-Darling test statistic: {ad_test.statistic}")
    for i in range(len(ad_test.critical_values)):
        sl, cv = ad_test.significance_level[i], ad_test.critical_values[i]
        if ad_test.statistic < ad_test.critical_values[i]:
            print(f"Significance level: {sl}, Critical value: {cv}, data looks normal (fail to reject H0)")
        else:
            print(f"Significance level: {sl}, Critical value: {cv}, data does not look normal (reject H0)")
        
    shapiro_test = stats.shapiro(data)
    print(f"\nShapiro test statistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}")
    
    # Kolmogorov-Smirnov test
    ks_test = stats.kstest(data, 'norm')
    print(f"\nKolmogorov-Smirnov test statistic: {ks_test.statistic}, p-value: {ks_test.pvalue}")
    print('\n\n')
    
    

def load_tuning(savepath):
    files=os.listdir(savepath)
    
    all_change=[]
    all_regions=[]
    all_bs=[]
    for file in files:
        data=np.load(f'{savepath}/{file}', allow_pickle=True).item()
        all_change.append(data['region_change'])
        all_regions.append(data['regions'])
        all_bs.append(data['target_bs'])
    
    if not np.all(all_bs==all_bs[0]):
        raise ValueError ("""
                          For this to work all target_bs in all sessions must
                          be the same (at least in the current version of the code)
                          This doesn't mean that you can't have different 
                          behaviours in different sessions, just that the 
                          target_bs you specify in the s_num_permutations.py 
                          script must stay the same
                          """)
    
    unique_bs=np.unique(np.hstack(all_bs))
    unique_regions=np.unique(np.hstack(all_regions))
    
    
    all_region_change=[]
    for i, session in enumerate(all_change):
        region_change=[]
        for region in unique_regions:

                region_change.append(session[all_regions[i]==region])

        all_region_change.append(region_change)
    return all_region_change, unique_regions, all_bs[-1]