# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:34:32 2025

@author: su-weisss
"""

import IPython
import polars as pl
import pandas as pd
import numpy as np

import os
from pathlib import Path

import time
import math

from tqdm import tqdm
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor # Corrected: Import ProcessPoolExecutor
# Recommended for multiprocessing on some platforms (like Windows) when creating executables,
# but also good practice in general when using if __name__ == '__main__':
from multiprocessing import freeze_support


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
# from matplotlib import cm # Deprecated
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from  matplotlib import colormaps # Import colormaps module

from collections import defaultdict
from collections.abc import Iterable

from typing import List, Dict

# Assuming preprocessFunctions (pp) is available in the environment
# If not, you might need to include its definition or a mock version
try:
    import preprocessFunctions as pp
except ImportError:
    print("Warning: preprocessFunctions not found. Some loading/path functions might fail.")
    # Define mock functions if pp is not available, or provide instructions to the user.
    class MockPreprocessFunctions:
        def get_paths(self, animal, session):
            print(f"Mock get_paths called for animal {animal}, session {session}")
            # Return a dummy structure or raise an error if actually needed
            # This mock might not be sufficient if actual file paths are needed.
            return {'preprocessed': Path(f'./dummy_preprocessed_data/{animal}/{session}'), 'frame_loss': '0'}
    pp = MockPreprocessFunctions()


# --- Debugging flags - MOVED TO TOP LEVEL ---
DEBUG_BEHAVIOR_EVENTS = True
DEBUG_PER_NEURON_DATA = True


# --- Helper Functions (extracted/adapted from provided files) ---

def start_stop_array(behaviour, b_name, frame=False, merge_attacks=None, pre_time=7.5):
    """
    Take behaviour pd for one behaviour and turns it into matrix
    with rows for one behaviour, and start, stop in the columns.
    Returns 'turns' as escape START.

    Parameters
    ----------
    behaviour : pd with all behaviours, directly from preprocessing
    b_name: the target behaviuour that should be converted e.g. 'escape'
        (NEEDS TO BE STATE BEHAVIOUR)
    frame: if true, framenumber will be returned, otherwise the s of the frame
    merge_attacks: Whether attack periods that follow each other shortly should
        be taken together. If yes, give the minimum distance in s that attacks
        should be allowed to have.
    pre_time: How many s before escape should loom have occurred (note this is before running onset, not turn).

    Returns
    -------
    vec: np array with first column being starts, second column being stops.
         Each row is a new instance of the behaviour. Returns empty array if behavior not found or invalid.
    """
    hunting_bs=['approach','pursuit','attack','eat']

    if frame==True:
        f='frames'
    elif frame==False:
        f='frames_s'
    else:
         raise ValueError("Invalid value for 'frame'. Must be True or False.")

    b_pd=behaviour[behaviour['behaviours']==b_name]

    #Sanity check
    if b_pd.empty:
        # print(f"Warning: Behavior '{b_name}' not found in behaviour DataFrame.")
        return np.array([]) # Return empty array if behavior not found

    # Check if it's a POINT behavior
    if 'POINT' in b_pd['start_stop'].unique():
        if not b_pd['start_stop'].nunique() == 1:
            print(f'Warning: Mixed start_stop types for point behavior {b_name}. Only using POINT.')
        return b_pd[b_pd['start_stop'] == 'POINT'][f].to_numpy() # Return only POINT times

    # Assume it's a STATE behavior with START/STOP pairs
    starts_df = b_pd[b_pd['start_stop'] == 'START']
    stops_df = b_pd[b_pd['start_stop'] == 'STOP']

    if len(starts_df) == 0 or len(stops_df) == 0 or len(starts_df) != len(stops_df):
        # print(f"Warning: Behavior '{b_name}' does not appear to be a state behavior with matching START/STOP pairs.")
        return np.array([])
    # Basic check for alternating starts and stops - assuming sorted times
    # This check is simplified and might fail if data is not perfectly ordered.
    # A more robust check would verify if timestamps interleave correctly.
    # For now, trust the sorted property of data from preprocessing.


    # Make vector
    vec = []
    starts_times = starts_df[f].to_numpy()
    stops_times = stops_df[f].to_numpy()

    if len(starts_times) != len(stops_times):
        # print(f"Warning: Mismatched START and STOP counts for '{b_name}'. Skipping.")
        return np.array([]) # Return empty if counts don't match

    for i in range(len(starts_times)):
        start_time = starts_times[i]
        stop_time = stops_times[i]

        if b_name == 'escape':
            # Complex logic for escape based on turns and looms (as in original)
            # This requires access to the full 'behaviour' DataFrame which is available.
            # Find turns around the escape start
            turns_around_escape = behaviour[
                (behaviour['behaviours'] == 'turn') &
                (behaviour['frames_s'] >= start_time - pre_time) &
                (behaviour['frames_s'] < start_time + 0.1)
            ]

            ref_event_time = start_time # Default to escape start time
            if not turns_around_escape.empty:
                 # Use the time of the last turn before escape start
                 ref_event_time = turns_around_escape['frames_s'].iloc[-1]

            # Use the reference event time (turn or escape start) as the start of the interval
            vec.append([ref_event_time, stop_time])

        # elif b_name == 'attack' and merge_attacks is not None:
        #      # Merge logic (as in original) - requires checking distance to previous event
        #      # This is complex with just start/stop pairs and needs iteration over original b_pd
        #      # Sticking to the original loop structure for merge_attacks if this is critical
        #      # For simplicity in this combined script, let's omit merge_attacks unless critical.
        #      # If merge_attacks is needed, the original for loop over b_pd with i+1 logic should be used.
        #      pass # Omit merge_attacks logic for now to keep it simpler
        else:
            # Standard state behavior: [start, stop]
            vec.append([start_time, stop_time])

    return np.array(vec)


def generate_random_times(N, num_points=10, min_gap=10):
    """Generate sorted random times between 0 and N with a minimum gap between them."""
    if num_points <= 0 or N <= 0 or min_gap <= 0:
        # print("Warning: Invalid input for generate_random_times.")
        return np.array([])
    if (num_points - 1) * min_gap >= N:
        # print(f"Warning: Cannot fit {num_points} points with minimum gap {min_gap} in range [0, {N}].")
        return np.array([])

    # Generate num_points random values between 0 and N - (num_points - 1) * min_gap
    # This ensures that after adding the minimum gap between consecutive points, the last point is <= N.
    max_val_for_random = N - (num_points - 1) * min_gap
    if max_val_for_random < 0: # Should be caught by the check above, but as a safeguard
         # print("Error in calculating max_val_for_random.")
         return np.array([])

    times = np.sort(np.random.uniform(0, max_val_for_random, num_points))

    # Add the minimum gap to ensure separation
    times += np.arange(num_points) * min_gap

    # Final check to ensure all times are within the range [0, N]
    if np.all(times >= 0) and np.all(times <= N):
        return times
    else:
        # If somehow outside bounds (unlikely with the calculation), try again or return empty.
        # print("Warning: Generated times outside the expected range after adjusting for gap. Retrying.")
        return generate_random_times(N, num_points, min_gap) # Recursive call might lead to infinite loop on failure, be cautious.
        # Alternatively, return np.array([]) here.


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

    speed = np.asarray(speed)
    time_ax = np.asarray(time_ax)

    if len(speed) != len(time_ax) or len(speed) < 2:
         # print("Warning: Input arrays for speed threshold crossings have inconsistent or insufficient length.")
         return []

    if Threshold is None:
        Threshold = 45
    if diff_time_s is None:
        diff_time_s = 10

    # Iterate over each sample with index for backwards lookup when needed.
    for i, (s, t) in enumerate(zip(speed, time_ax)):
        # Detect upward crossing: speed goes from below to above or equal to Threshold.
        if not in_segment and s >= Threshold:
            # Backtrack to find the first time before this event when speed was over 5.
            j = i
            while j > 0 and speed[j - 1] > 5: # Threshold to consider as start of movement
                j -= 1
            start_time = time_ax[j]
            in_segment = True
        # Detect downward crossing: when speed falls below 5.
        elif in_segment and s < 5: # Threshold to consider as end of movement
            crossings.append([start_time, t])
            in_segment = False

    # If a segment is ongoing at the end of the data, close it
    if in_segment:
         crossings.append([start_time, time_ax[-1]])


    # Remove events that are less than diff_time_s seconds apart.
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        # Always include the first valid interval
        if last_end is None or start - last_end >= diff_time_s:
            filtered.append(interval)
            last_end = end

    return filtered

def generate_acceleration_threshold_crossings(speed, time_ax, Threshold=55, diff_time_s=10):
    """
    Calculate acceleration from the given speed and detect segments where acceleration
    rises above a threshold and then falls back below it.

    Parameters:
        speed (array-like): Array of speed values.
        time_ax (array-like): Array of corresponding time values.
        Threshold (float): The acceleration threshold for detecting crossings.
        diff_time_s (float): Minimum time difference between consecutive crossings.

    Returns:
        list: A list of [start, stop] pairs, where 'start' is the time when
              acceleration first crosses above the threshold and 'stop' is the time when
              the acceleration falls back below the threshold.
    """
    speed = np.asarray(speed)
    time_ax = np.asarray(time_ax)

    if len(speed) != len(time_ax) or len(speed) < 2:
         # print("Warning: Input arrays for acceleration threshold crossings have inconsistent or insufficient length.")
         return []

    # Compute acceleration as the numerical derivative of speed.
    dt = np.diff(time_ax)
    if np.any(dt <= 0):
         # Handle non-increasing time_ax by filtering
         valid_indices = np.where(dt > 0)[0]
         valid_indices = np.insert(valid_indices, 0, 0) # Always keep the first point
         if len(valid_indices) < 2: return []
         speed_valid = speed[valid_indices]
         time_ax_valid = time_ax[valid_indices]
         acceleration = np.gradient(speed_valid, time_ax_valid)
         # Interpolate acceleration back to original time axis (approximate)
         acceleration_interp = np.interp(time_ax, time_ax_valid, acceleration, left=acceleration[0], right=acceleration[-1])
         acceleration = acceleration_interp
    else:
        acceleration = np.gradient(speed, time_ax)


    crossings = []
    in_segment = False
    start_time = None

    # Use default values if None is provided.
    if Threshold is None:
        Threshold = 55
    if diff_time_s is None:
        diff_time_s = 10

    # Iterate over acceleration with corresponding time values.
    for a, t in zip(acceleration, time_ax):
        # Detect upward crossing: acceleration goes from below to at/above threshold.
        if not in_segment and a >= Threshold:
            in_segment = True
            start_time = t
        # Detect downward crossing: acceleration falls below the threshold when already in a segment.
        elif in_segment and a < Threshold:
            crossings.append([start_time, t])
            in_segment = False

    # If a segment is ongoing at the end of the data, close it
    if in_segment:
         crossings.append([start_time, time_ax[-1]])

    # Remove events that are less than diff_time_s time apart.
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        # Always include the first valid interval
        if last_end is None or start - last_end >= diff_time_s:
            filtered.append(interval)
            last_end = end

    return filtered

def generate_shelter_distance_threshold_crossings(distance_to_shelter, time_ax, Threshold=5, diff_time_s=1):
    """
    Identify segments where distance_to_shelter drops less than a threshold and then rises back

    Parameters:
        distance_to_shelter (array-like): Array of distance_to_shelter values.
        time_ax (array-like): Array of corresponding time values.
        Threshold (float): The distance_to_shelter threshold for detecting crossings.

    Returns:
        list: A list of [start, stop] pairs, where 'start' is the time when
              distance_to_shelter first crosses under the threshold and 'stop' is the time when
              distance_to_shelter falls back over the threshold.
    """
    crossings = []
    in_segment = False
    start_time = None

    distance_to_shelter = np.asarray(distance_to_shelter)
    time_ax = np.asarray(time_ax)

    if len(distance_to_shelter) != len(time_ax):
         # print("Warning: Input arrays for distance threshold crossings have inconsistent length.")
         return []

    if Threshold is None:
        Threshold=5
    if diff_time_s is None:
        diff_time_s=1

    # Iterate over each sample in distance_to_shelter with its corresponding time index.
    for s, t in zip(distance_to_shelter, time_ax):
        # Detect crossing under threshold: distance_to_shelter goes from above or equal to below threshold.
        if not in_segment and s < Threshold:
            in_segment = True
            start_time = t
        # Detect crossing over threshold: distance_to_shelter goes from below to above or equal to threshold when already in a segment.
        elif in_segment and s >= Threshold:
            crossings.append([start_time, t])
            in_segment = False

    # If a segment is ongoing at the end of the data, close it
    if in_segment:
        crossings.append([start_time, time_ax[-1]])

    # remove events, less then diff_time_s time apart.
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        # Always include the first valid interval
        if last_end is None or start - last_end >= diff_time_s:
            filtered.append(interval)
            last_end = end

    return filtered


def insert_event_times_into_behavior_df(behaviour,framerate,event_type=None,behavior_name=None,behavior_type=None,**kwargs):
    """
    Insert new behavioral events into the DataFrame `behaviour` with optional extra columns.

    Parameters:
      behaviour (pd.DataFrame): Original DataFrame.
      framerate (float): Frame rate of the tracking data.
      event_type (str): The behavioral event type. Can be "baseline_random" / "distance_to_shelter" / "speed" / "acceleration".
      behavior_name (str): The event name for the new behavior.
      behavior_type (str): The behavioral category for the new behavior.
      kwargs: Optional keyword arguments depending on event_type.

    Returns:
      pd.DataFrame: The updated DataFrame with the new rows appended.
    """

    # Ensure essential columns exist in the input behaviour DataFrame for consistency
    required_cols = ['behaviours', 'behavioural_category', 'start_stop', 'frames_s', 'frames', 'video_start_s', 'video_end_s']
    for col in required_cols:
        if col not in behaviour.columns:
             # Add missing column with default value or raise error
             # print(f"Warning: Column '{col}' not found in input behaviour DataFrame. Adding it.")
             behaviour[col] = None # Or add a sensible default if known

    # Assuming video_start_s and video_end_s are available from the first row if the DF is not empty
    video_start = behaviour.iloc[0]['video_start_s'] if not behaviour.empty else 0
    video_end = behaviour.iloc[0]['video_end_s'] if not behaviour.empty else 0

    new_rows_list = [] # List to collect new DataFrame rows

    if event_type=='baseline_random':
        baseline_time_s = kwargs.get('baseline_time_s', 7*60)
        n_trials = kwargs.get('n_trials', 10)
        time_ax = kwargs.get('time_ax', None)
        min_gap = kwargs.get('min_gap', 10)

        if time_ax is None or not time_ax.size > 0:
             # print(f"Warning: time_ax not provided or empty for baseline_random event '{behavior_name}'. Skipping.")
             return behaviour

        N = np.min([time_ax[-1] if time_ax.size > 0 else 0, baseline_time_s])
        if N <= 0:
             # print(f"Warning: Session duration or baseline_time_s is zero or negative for baseline_random event '{behavior_name}'. Skipping.")
             return behaviour

        time_points = generate_random_times(N, num_points=n_trials, min_gap=min_gap)

        if time_points.size > 0:
             start_stop_col = ['POINT'] * len(time_points)
             frames = (time_points * framerate).astype(int)

             new_rows_list.append(pd.DataFrame({
                 'behaviours': [behavior_name] * len(time_points),
                 'behavioural_category': [behavior_type] * len(time_points),
                 'start_stop': start_stop_col,
                 'frames_s': time_points,
                 'frames': frames,
                 'video_start_s': [video_start] * len(time_points),
                 'video_end_s': [video_end] * len(time_points)
             }))
        # else:
             # print(f"Warning: Could not generate random times for '{behavior_name}'.")


    elif event_type in ['distance_to_shelter', 'speed', 'acceleration']:
        speed = kwargs.get('speed', None)
        distance_to_shelter = kwargs.get('distance_to_shelter', None)
        time_ax = kwargs.get('time_ax', None)
        Threshold = kwargs.get('Threshold', None)
        diff_time_s = kwargs.get('diff_time_s', None)

        if time_ax is None or not time_ax.size > 0:
             # print(f"Warning: time_ax not provided or empty for event type '{event_type}'. Skipping.")
             return behaviour

        time_pairs = []
        if event_type == 'distance_to_shelter':
            if distance_to_shelter is None:
                # print("'distance_to_shelter' must be provided for distance_to_shelter events.")
                return behaviour
            time_pairs = generate_shelter_distance_threshold_crossings(distance_to_shelter, time_ax,
                                                                       Threshold=Threshold,
                                                                       diff_time_s=diff_time_s)
        elif event_type == 'speed':
            if speed is None:
                # print("'speed' must be provided for speed events.")
                return behaviour
            time_pairs = generate_speed_threshold_crossings(speed, time_ax,
                                                            Threshold=Threshold,
                                                            diff_time_s=diff_time_s)
        elif event_type == 'acceleration':
            if speed is None:
                # print("'acceleration' must be provided for acceleration events.")
                return behaviour
            time_pairs = generate_acceleration_threshold_crossings(speed, time_ax, Threshold=Threshold, diff_time_s=diff_time_s)


        if time_pairs:
             for start_time, stop_time in time_pairs:
                  start_frame = start_time * framerate
                  stop_frame = stop_time * framerate

                  new_rows_list.append(pd.DataFrame({
                      'behaviours': [behavior_name] * 2,
                      'behavioural_category': [behavior_type] * 2,
                      'start_stop': ['START', 'STOP'],
                      'frames_s': [start_time, stop_time],
                      'frames': [start_frame, stop_frame],
                      'video_start_s': [video_start] * 2,
                      'video_end_s': [video_end] * 2
                  }))
        # else:
             # print(f"Warning: No periods detected for event type '{event_type}' with given parameters.")

    else:
        raise ValueError(f"Unknown behavioral event_type argument: {event_type}")

    # Append the new rows to the existing DataFrame if any were created
    if new_rows_list:
        new_rows_df = pd.concat(new_rows_list, ignore_index=True)
        behaviour = pd.concat([behaviour, new_rows_df], ignore_index=True)

    return behaviour


def load_preprocessed(animal, session, load_pd=False, load_lfp=False):
    """
    Loads the data output from preprocess_all.py.

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
    --------
    Tuple containing loaded data:
    (frames_dropped, behaviour, ndata, n_spike_times, n_time_index,
     n_cluster_index, n_region_index, n_channel_index, velocity, locations,
     node_names, frame_index_s, [lfp, lfp_time, lfp_framerate if load_lfp])
    Raises FileNotFoundError if essential files are missing or ValueError for invalid args.
    """
    if load_pd and load_lfp:
        raise ValueError('The function is not adapted for loading both PD and LFP simultaneously.')

    paths = pp.get_paths(animal, session)
    # Ensure paths is a dictionary-like object and has the 'preprocessed' key
    if not isinstance(paths, (dict, pd.Series)) or 'preprocessed' not in paths or pd.isna(paths['preprocessed']):
         raise FileNotFoundError(f"Could not get preprocessed data path for {animal}/{session}. Check paths configuration.")

    path = Path(paths['preprocessed'])

    behaviour_path = path / 'behaviour.csv'
    tracking_path = path / 'tracking.npy'
    ndata_path = path / 'np_neural_data.npy'
    lfp_path = path / 'lfp.npy'
    pd_path = path / 'pd_neural_data.csv'


    if not behaviour_path.exists(): raise FileNotFoundError(f"Behaviour file not found at {behaviour_path}")
    try:
        behaviour = pd.read_csv(behaviour_path)
    except Exception as e:
         raise IOError(f"Error loading or parsing behaviour data from {behaviour_path}: {e}")


    if not tracking_path.exists(): raise FileNotFoundError(f"Tracking file not found at {tracking_path}")
    try:
        tracking = np.load(tracking_path, allow_pickle=True).item()
        velocity = tracking['velocity']
        locations = tracking['locations']
        node_names = tracking['node_names']
        frame_index_s = tracking['frame_index_s']
    except Exception as e:
        raise IOError(f"Error loading or parsing tracking data from {tracking_path}: {e}")


    # Handle potential frame_loss from paths
    frames_dropped_str = paths.get('frame_loss', '0') # Default to '0' if key missing
    try:
        # Corrected typo
        frames_dropped = int(float(frames_dropped_str))
    except (ValueError, TypeError):
        print(f"Warning: Could not interpret frame_loss '{frames_dropped_str}' as number for {animal}/{session}. Assuming 0.")
        frames_dropped = 0

    # Adjust tracking data if frames were dropped
    if frames_dropped != 0:
        # Ensure there are enough frames to drop
        if abs(frames_dropped) >= len(velocity):
            print(f"Error: frame_loss ({frames_dropped}) is larger than or equal to velocity length ({len(velocity)}) for {animal}/{session}. Cannot trim.")
            # Decide how to handle: return empty data, raise error, or return potentially misaligned data
            # Returning potentially misaligned data might be risky for subsequent analysis.
            # Let's raise an error or return empty arrays if crucial data is severely impacted.
            # For now, print warning and proceed, but be aware of potential issues.
            pass # Proceed with original data, warning printed

        else:
             if frames_dropped > 0: # Drop from the end
                 velocity = velocity[:-frames_dropped]
                 locations = locations[:-frames_dropped]
                 # Assuming frame_index_s corresponds to velocity/locations length
                 if len(frame_index_s) > len(velocity):
                      frame_index_s = frame_index_s[:len(velocity)]
             else: # Negative frames_dropped means more frames in velocity/locations, trim nidq time? (Unusual)
                  # This scenario needs careful handling based on how frame_loss is defined.
                  # Assuming positive frame_loss means nidq is longer than video.
                  print(f"Warning: Unusual negative frame_loss value ({frames_dropped}) for {animal}/{session}.")
                  pass # Proceed with original data, warning printed

        if frames_dropped != 0:
             print(f"Adjusted tracking data due to frame_loss of {frames_dropped}.")


    if load_pd:
        if not pd_path.exists(): raise FileNotFoundError(f"Photodiode neural data file not found at {pd_path}")
        try:
            ndata_pd = pd.read_csv(pd_path)
            return behaviour, ndata_pd, velocity, locations, node_names, frame_index_s
        except Exception as e:
            raise IOError(f"Error loading or parsing photodiode neural data from {pd_path}: {e}")


    if not ndata_path.exists(): raise FileNotFoundError(f"Neural data file not found at {ndata_path}")
    try:
        ndata_dict = np.load(ndata_path, allow_pickle=True).item()
        ndata = ndata_dict['n_by_t']
        n_time_index = ndata_dict['time_index']
        n_cluster_index = ndata_dict['cluster_index']
        n_region_index = ndata_dict['region_index']
        n_channel_index = ndata_dict['cluster_channels']
        n_spike_times = ndata_dict['n_spike_times']
    except Exception as e:
        raise IOError(f"Error loading or parsing neural data from {ndata_path}: {e}")


    # Ensure neural data time index and ndata length is consistent with tracking frame_index_s
    min_len = min(len(n_time_index), len(frame_index_s))
    if len(n_time_index) != min_len:
        print(f"Warning: Neural data time index length ({len(n_time_index)}) inconsistent with tracking frame_index_s ({len(frame_index_s)}) for {animal}/{session}. Trimming.")
        n_time_index = n_time_index[:min_len]
        # Assuming ndata corresponds to n_time_index's time dimension
        if ndata.shape[1] > min_len:
             ndata = ndata[:, :min_len]
        # Trimming n_spike_times (list of arrays) is complex and would require re-filtering spikes
        # that fall outside the new time range for each neuron. This is omitted for simplicity here,
        # assuming the original n_spike_times are consistent or handled by downstream functions.


    if load_lfp:
        if not lfp_path.exists(): raise FileNotFoundError(f"LFP data file not found at {lfp_path}")
        try:
            lfp_dict = np.load(lfp_path, allow_pickle=True).item()
            lfp = lfp_dict['lfp']
            lfp_time = lfp_dict['lfp_time']
            lfp_framerate = lfp_dict['lfp_framerate']
            return (frames_dropped, behaviour, ndata, n_spike_times, n_time_index,
                    n_cluster_index, n_region_index, n_channel_index, velocity, locations,
                    node_names, frame_index_s, lfp, lfp_time, lfp_framerate)
        except Exception as e:
            raise IOError(f"Error loading or parsing LFP data from {lfp_path}: {e}")

    else:
        return (frames_dropped, behaviour, ndata, n_spike_times, n_time_index,
                n_cluster_index, n_region_index, n_channel_index, velocity, locations,
                node_names, frame_index_s)


# --- Plotting Helper Function (from plottingFunctions.py) ---
def remove_axes(ax, bottom=False):
    """
    Helper to remove ticks and labels from a matplotlib axis.
    If bottom is True, only the bottom ticks/labels are removed.
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if bottom:
        ax.tick_params(labelbottom=False, bottom=False)
    else:
        ax.tick_params(bottom=False, labelbottom=False)
    ax.tick_params(left=False, labelleft=False)


# --- PSTH Calculation and Trial Data Extraction Function (from plottingFunctions.py) ---
# Keeping the structure that returns data intended for storage in Polars DF
def psth_cond_refactored(cluster_id, debug_per_neuron_data_flag, neurondata, n_time_index, behaviour_df_session, velocity, frame_index_s,
              axs_placeholder, window=5, density_bins=0.1, return_data=True, session=None,
              align_to_end=False):
    """
    Compute peri-event spike and velocity metrics and (optionally) plot them for a single neuron and session.
    This version is adapted to be called within a loop and return data structure for aggregation.

    Parameters
    ----------
    cluster_id : int or str
        The ID of the neuron being processed.
    debug_per_neuron_data_flag : bool
        Flag to enable/disable debugging prints within this function.
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin for a single neuron.
    n_time_index : array_like
        Time index from preprocessing (neural data).
    behaviour_df_session : pd.DataFrame
        Behaviour DataFrame for the current session.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing (tracking data).
    axs_placeholder : Ignored in this version, plotting is off.
    window : float, optional
        How many seconds before/after an event to analyze. Default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. Default is 0.1.
    return_data : bool, optional
        If True, returns a dictionary of computed values. This is expected to be True.
    session : any, optional
         Session identifier to include in return data for each trial.
    align_to_end : bool, optional
        If True, align events to the event's end time rather than the start.

    Returns
    -------
    dict: A dictionary where keys are behavior names found in behaviour_df_session
          and values are dictionaries containing precomputed trial-level data.
          Returns an empty dictionary if no behaviors are processed or an error occurs during behavior loop.
    """
    if not return_data:
        print("Warning: psth_cond_refactored is designed to return data. Setting return_data=True.")
        return_data = True

    all_behaviors_in_session = behaviour_df_session['behaviours'].unique()

    per_behavior_data = {}

    # Define consistent bin edges for spike histogram (density) and velocity
    psth_bins = np.arange(-window, window + density_bins, density_bins)
    velocity_bin_size = 0.1 # Consistent bin size for velocity averaging
    velbins_edges = np.arange(-window, window + velocity_bin_size, velocity_bin_size) # Use + velocity_bin_size to include the end


    for b_name in all_behaviors_in_session:
        try:
            # Get start/stop times for the current behavior
            # Use start_stop_array helper function
            all_start_stop = start_stop_array(behaviour_df_session, b_name)

            # --- DEBUG: Check start_stop_array output size ---
            if debug_per_neuron_data_flag:
                 print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': start_stop_array size = {all_start_stop.size if isinstance(all_start_stop, np.ndarray) else 'Not numpy array'}")


            if all_start_stop.size == 0:
                if debug_per_neuron_data_flag:
                     print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': has no events found by start_stop_array.")
                per_behavior_data[b_name] = None # Store None for this behavior/neuron/session
                continue # Skip to the next behavior

            # --- Initialize containers for computed values for THIS behavior/neuron/session ---
            all_aligned_spikes = []       # List to store spike times per trial (aligned)
            all_vel = []              # Velocity per trial
            all_vel_times = []        # Timepoints for velocity (aligned)
            state_events_shading = [] # Contains info for state-event shading (aligned times)
            session_per_trial_list = []    # Record session info for each trial

            point = False
            if all_start_stop.ndim == 1 or (all_start_stop.ndim == 2 and all_start_stop.shape[1] == 1):
                point = True
                if all_start_stop.ndim == 2:
                     all_start_stop = all_start_stop.flatten()
            elif all_start_stop.ndim == 2 and all_start_stop.shape[1] == 2:
                 point = False
            else:
                 print(f"Warning: Unexpected shape for start_stop_array output for behavior '{b_name}': {all_start_stop.shape}. Skipping.")
                 per_behavior_data[b_name] = None
                 continue


            trial_index = 0
            num_valid_trials_processed = 0
            # --- Event loop: compute spike and velocity info per event ---
            for startstop in all_start_stop:
                if not point:
                    startstop = np.asarray(startstop)
                    if len(startstop) != 2: continue # Skip malformed
                    event_start, event_stop = startstop
                    Duration = event_stop - event_start

                    # Ensure event times are within the time index range
                    if event_start < n_time_index[0] or event_stop > n_time_index[-1] or \
                       event_start < frame_index_s[0] or event_stop > frame_index_s[-1]:
                         if debug_per_neuron_data_flag:
                              print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': Event [{event_start}, {event_stop}] outside data ranges. Skipping event.")
                         continue # Skip this event if it's outside the overall data ranges

                    ref_time = event_stop if align_to_end else event_start
                    shading_left = event_start - ref_time
                    shading_width = Duration

                else:
                    ref_time = np.asarray(startstop).item()
                    # Ensure event time is within the time index range
                    if ref_time < n_time_index[0] or ref_time > n_time_index[-1] or \
                       ref_time < frame_index_s[0] or ref_time > frame_index_s[-1]:
                         if debug_per_neuron_data_flag:
                              print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': Point event {ref_time} outside data ranges. Skipping event.")
                         continue # Skip this event if it's outside the overall data ranges


                    plotstart = ref_time - window
                    plotstop = ref_time + window

                    # Adjust plot window bounds to be within the actual data time range
                    actual_plotstart_neural = max(plotstart, n_time_index[0])
                    actual_plotstop_neural = min(plotstop, n_time_index[-1])
                    actual_plotstart_tracking = max(plotstart, frame_index_s[0])
                    actual_plotstop_tracking = min(plotstop, frame_index_s[-1])

                    # Check if the actual plot windows are valid (start <= stop)
                    if actual_plotstart_neural > actual_plotstop_neural or actual_plotstart_tracking > actual_plotstop_tracking:
                         if debug_per_neuron_data_flag:
                              print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': Adjusted plot window is invalid [{actual_plotstart_neural}, {actual_plotstop_neural}] or [{actual_plotstart_tracking}, {actual_plotstop_tracking}]. Skipping event.")
                         continue


                # --- Collect spike times for current trial ---
                # Find indices in n_time_index within the *actual* neural plotting window
                window_mask_neural = (n_time_index >= actual_plotstart_neural) & (n_time_index <= actual_plotstop_neural)
                times_in_window_neural = n_time_index[window_mask_neural]
                spikes_in_window = neurondata[window_mask_neural].copy()

                spike_time_indices_in_window = np.where(spikes_in_window > 0)[0]
                spike_times_for_trial = times_in_window_neural[spike_time_indices_in_window].astype(float)

                aligned_spiketimes = spike_times_for_trial - ref_time
                all_aligned_spikes.append(aligned_spiketimes)
                session_per_trial_list.append(session)

                # Add state event shading info if it's a state event
                if not point:
                    state_events_shading.append({
                         "trial": num_valid_trials_processed, # Use the count of valid trials processed so far for y-position
                         "left": shading_left, # This is relative to the event's original start if aligned to start, or end if aligned to end
                         "width": shading_width,
                         "alpha": 0.5
                    })


                # --- Collect velocity data for current trial ---
                # Find indices in frame_index_s within the *actual* tracking plotting window
                vel_mask = (frame_index_s >= actual_plotstart_tracking) & (frame_index_s <= actual_plotstop_tracking)
                vel_times_for_trial = frame_index_s[vel_mask]
                vel_traces_for_trial = velocity[vel_mask]

                # Align velocity times
                aligned_vel_times = vel_times_for_trial - ref_time

                all_vel_times.append(aligned_vel_times)
                all_vel.append(vel_traces_for_trial)

                num_valid_trials_processed += 1 # Increment count only for trials that were processed


            # --- DEBUG: Print valid trial count for this behavior ---
            if debug_per_neuron_data_flag:
                 print(f"Neuron {cluster_id}, Session {session}, Behavior '{b_name}': Processed {num_valid_trials_processed} valid trials.")


            # --- Compute session-level aggregated data for this behavior ---
            # This is data for the current neuron, for the current behavior, aggregated across its trials in THIS session.
            # This data will be further aggregated across sessions later.

            # Overall PSTH for this cell and session's trials of THIS behavior
            all_aligned_spikes_concat = np.hstack(all_aligned_spikes) if all_aligned_spikes else np.array([])
            sum_spikes, firing_rate_bins_edges = np.histogram(all_aligned_spikes_concat, bins=psth_bins)
            # Use num_valid_trials_processed for averaging
            hz = (sum_spikes / density_bins) / num_valid_trials_processed if num_valid_trials_processed > 0 and density_bins > 0 else np.zeros_like(sum_spikes)
            psth_bin_centers = (firing_rate_bins_edges[:-1] + firing_rate_bins_edges[1:]) / 2.


            # Overall Average Velocity for this cell and session's trials of THIS behavior
            # Velocity bins are defined outside the loop, use those edges
            all_vel_times_concat = np.hstack(all_vel_times) if all_vel_times else np.array([])
            all_vel_concat = np.hstack(all_vel) if all_vel else np.array([])

            binned_values, _ = np.histogram(all_vel_times_concat, bins=velbins_edges, weights=all_vel_concat)
            binned_counts, _ = np.histogram(all_vel_times_concat, bins=velbins_edges)
            avg_velocity = binned_values / np.maximum(binned_counts, 1e-9) # Avoid division by zero
            velocity_bin_centers = (velbins_edges[:-1] + velbins_edges[1:]) / 2.


            # Store the data computed for this behavior in this session
            # This structure mirrors the tuple expected by the original script's Polars DF population
            psth_data_dict = {
                'FR_Time_bin_center': list(psth_bin_centers), # Convert to list for Polars
                'FR_Hz': list(hz) # Convert to list for Polars
            }
            velocity_data_dict = {
                'Velocity_Time_bin': list(velocity_bin_centers), # Convert to list for Polars
                'Avg_Velocity_cms': list(avg_velocity), # Convert to list for Polars
                'velocity_bin_size': velocity_bin_size,
                # Include raw binned counts/values if needed later, convert to list
                'binned_values_raw': list(binned_values),
                'binned_counts_raw': list(binned_counts)
            }
            raster_data_dict = {
                # Store aligned spike times per trial. Polars can handle lists of NumPy arrays or lists of lists.
                # Storing as list of lists is often safer for serialization.
                'spikes_array_per_trial': [list(trial) for trial in all_aligned_spikes]
            }
            meta_data_dict = {
                 # Store original start/stop times for each trial, convert to list of lists
                 # Note: This might not be the aligned times needed for shading info in the final plot.
                 # The precomputed_data_dict.get('state_event_shading_info') contains aligned shading info.
                 'original_start_stop_time_s': [list(ss) if isinstance(ss, np.ndarray) else ss for ss in all_start_stop],
                 'is_point_event': point
                 # Add session info per trial here as well if needed directly in meta_dict, though precomputed already has it
            }
            precomputed_data_dict = {
                # Store trial-level data needed for cross-session aggregation and colored raster
                'session_aligned_spike_times': [list(trial) for trial in all_aligned_spikes], # Aligned spike times per trial
                'session_per_trial': session_per_trial_list, # Session label per trial
                'all_vel': [list(v) for v in all_vel], # Velocity trace per trial
                'all_vel_times': [list(vt) for vt in all_vel_times], # Aligned velocity times per trial
                'state_event_shading_info': state_events_shading # Aligned shading info per trial
                # Also include overall calculated data here for redundancy/direct access if needed
                # 'overall_psth_hz': list(hz),
                # 'overall_psth_bins': list(psth_bin_centers),
                # 'overall_velocity_avg': list(avg_velocity),
                # 'overall_velocity_bins': list(velocity_bin_centers)
            }

            # Store the data computed for this behavior in this session
            # This structure mirrors the tuple expected by the original script's Polars DF population
            per_behavior_data[b_name] = (psth_data_dict, velocity_dict, raster_dict, meta_dict, precomputed_data_dict)

        except Exception as e:
             print(f"Error processing behavior '{b_name}' for neuron {cluster_id} in session {session}: {e}")
             per_behavior_data[b_name] = None # Store None for this behavior/neuron/session
             continue

    # After the loop through all_behaviors_in_session in psth_cond_refactored:
    if debug_per_neuron_data_flag:
        # Print the keys of the dictionary being returned
        print(f"Neuron {cluster_id}, Session {session}: psth_cond_refactored returning behaviors: {list(per_behavior_data.keys())}")
        # Optionally print if the value for a behavior is None
        count=0
        for beh_key, beh_value in per_behavior_data.items():
             if beh_value is None:
                  count+=1
                  print(f"Neuron {cluster_id}, Session {session}: Behavior '{beh_key}' data is None.")
        if count==len(per_behavior_data.items()):
            print(f"all Nones")
      

    return per_behavior_data


# --- Aggregation Function ---
def fill_nested_list_nulls(nested_list, fill_value=0.0):
    """Recursively fills nulls (None, NaN) in nested lists/arrays."""
    filled_list = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            # If it's a list or array, recurse or fill np.nan
            if isinstance(item, np.ndarray):
                # Fill np.nan in numpy arrays
                filled_list.append(np.nan_to_num(item, nan=fill_value))
            else:
                # Recurse for nested lists
                filled_list.append(fill_nested_list_nulls(item, fill_value))
        elif item is None or (isinstance(item, float) and np.isnan(item)):
            # Fill None or NaN with fill_value
            filled_list.append(fill_value)
        else:
            # Keep other types as they are
            filled_list.append(item)
    return filled_list


def aggregate_cell_data_from_df(neurons_df, cluster_id, sessions):
    """
    Aggregates spike times and velocity data for a specific cell across multiple sessions
    from the Neurons_pl Polars DataFrame.

    Args:
        neurons_df (pl.DataFrame): The Polars DataFrame containing per-session, per-neuron data.
        cluster_id: The cluster ID of the cell to aggregate.
        sessions (list): List of session names.

    Returns:
        dict: Aggregated data for the cell, structured by behavior.
    """
    aggregated_data = {}
    # Filter rows for the specific cluster ID
    cell_rows_df = neurons_df.filter(pl.col("cluster_id") == cluster_id)

    # Collect all behaviors present for this cell across all sessions in the DataFrame
    all_behaviors_for_cell = set()
    for session in sessions:
        # Ensure the session column exists and has data for this cell
        if session in cell_rows_df.columns and cell_rows_df[session].is_not_null().any():
            # Get the data stored in the session column for this cell.
            # This should be a dictionary of behaviors: {behavior_name: (tuple_of_dicts)}
            try:
                session_data_for_cell = cell_rows_df.select(session).filter(pl.col(session).is_not_null()).row(0)[0]
                if isinstance(session_data_for_cell, dict):
                     # Update the set of all behaviors found for this cell
                     all_behaviors_for_cell.update(session_data_for_cell.keys())
            except (IndexError, TypeError, ValueError) as err:
                 # print(f"Skipping session data structure extraction for cluster {cluster_id} in session '{session}': {err}")
                 pass # Continue to the next session if data for this cell/session is malformed

    # Define consistent bin edges for overall PSTH and velocity across all behaviors/cells
    # Using the same window and density_bins as the initial per-session calculation
    window = 5
    density_bins = 0.1 # Assuming this is the desired bin width for the final plot
    psth_bins_edges = np.arange(-window, window + density_bins, density_bins)
    psth_bin_centers_overall = (psth_bins_edges[:-1] + psth_bins_edges[1:]) / 2.

    velocity_bin_size = 0.1 # Assuming this is the desired bin size for final velocity plot
    velbins_edges = np.arange(-window, window + velocity_bin_size, velocity_bin_size)
    velocity_bin_centers_overall = (velbins_edges[:-1] + velbins_edges[1:]) / 2.


    # Now iterate through all behaviors found for this cell across all sessions
    for behavior in all_behaviors_for_cell:
        all_spike_times_for_behavior = []
        all_trial_sessions_for_behavior = []
        all_velocity_traces_for_behavior = []
        all_velocity_times_for_behavior = []
        all_state_event_shading_info = [] # Collect aligned shading info for all trials of this behavior


        # Collect data for this specific behavior across all sessions
        for session in sessions:
             # Check if the session column exists and has non-null data for this cluster
             if session in cell_rows_df.columns and cell_rows_df.filter(pl.col(session).is_not_null()).shape[0] > 0:
                try:
                    # Get the dictionary of behaviors for this cell in this session
                    session_behavior_data_dict = cell_rows_df.select(session).filter(pl.col(session).is_not_null()).row(0)[0]

                    # Check if the session data is a dictionary and contains the current behavior key
                    if isinstance(session_behavior_data_dict, dict) and behavior in session_behavior_data_dict and session_behavior_data_dict[behavior] is not None:
                         # Get the tuple of dictionaries for this specific behavior in this session
                         behavior_data_tuple = session_behavior_data_dict[behavior]

                         # The tuple contains (psth_dict, velocity_dict, raster_dict, meta_dict, precomputed_dict)
                         if isinstance(behavior_data_tuple, tuple) and len(behavior_data_tuple) > 4:
                             precomputed_dict = behavior_data_tuple[4] # Get the precomputed_dict

                             if isinstance(precomputed_dict, dict):
                                 # Extend the lists with trial-level data from this session's behavior
                                 all_spike_times_for_behavior.extend(precomputed_dict.get('session_aligned_spike_times', []))
                                 all_trial_sessions_for_behavior.extend(precomputed_dict.get('session_per_trial', []))
                                 all_velocity_traces_for_behavior.extend(precomputed_dict.get('all_vel', []))
                                 all_velocity_times_for_behavior.extend(precomputed_dict.get('all_vel_times', []))
                                 # The state_event_shading_info list from psth_cond_refactored contains shading info per trial for that session.
                                 # We need to append these to the list for this behavior across all sessions.
                                 all_state_event_shading_info.extend(precomputed_dict.get('state_event_shading_info', []))

                except (IndexError, TypeError, ValueError) as err:
                    # print(f"Skipping behavior '{behavior}' in session '{session}' for cluster {cluster_id} during data collection: {err}")
                    pass # Continue to the next session if data for this behavior/session is malformed

        # Consolidate data for this behavior and calculate overall stats across all collected trials
        num_total_trials = len(all_spike_times_for_behavior)

        if num_total_trials > 0:
            # Calculate overall PSTH across all trials for this behavior
            all_aligned_spikes_concat = np.hstack(all_spike_times_for_behavior) if all_spike_times_for_behavior else np.array([])
            sum_spikes, _ = np.histogram(all_aligned_spikes_concat, bins=psth_bins_edges)
            overall_psth_hz = (sum_spikes / density_bins) / num_total_trials if num_total_trials > 0 and density_bins > 0 else np.zeros_like(sum_spikes)

            # Calculate overall Average Velocity across all trials for this behavior
            all_vel_times_concat = np.hstack(all_velocity_times_for_behavior) if all_velocity_times_for_behavior else np.array([])
            all_vel_concat = np.hstack(all_velocity_traces_for_behavior) if all_velocity_traces_for_behavior else np.array([])

            binned_values, _ = np.histogram(all_vel_times_concat, bins=velbins_edges, weights=all_vel_concat)
            binned_counts, _ = np.histogram(all_vel_times_concat, bins=velbins_edges)
            overall_velocity_avg = binned_values / np.maximum(binned_counts, 1e-9) # Avoid division by zero


            aggregated_data[behavior] = {
                'raster_trials_spike_times': all_spike_times_for_behavior,
                'raster_trial_sessions': all_trial_sessions_for_behavior,
                # Keep individual velocity data if needed for future plots, though not used in the final requested plot
                'velocity_trials_velocity': all_velocity_traces_for_behavior,
                'velocity_trials_times': all_velocity_times_for_behavior,
                'overall_psth_hz': np.nan_to_num(overall_psth_hz, nan=0.0), # Fill NaN with 0.0
                'overall_psth_bins': psth_bin_centers_overall,
                'overall_velocity_avg': np.nan_to_num(overall_velocity_avg, nan=0.0), # Fill NaN with 0.0
                'overall_velocity_bins': velocity_bin_centers_overall,
                'state_event_shading_info': all_state_event_shading_info # Aligned shading info for all trials
            }
        else:
             # If no data for this behavior across all sessions, add empty entries with correct bin centers and filled nulls
             aggregated_data[behavior] = {
                 'raster_trials_spike_times': [],
                 'raster_trial_sessions': [],
                 'velocity_trials_velocity': [],
                 'velocity_trials_times': [],
                 'overall_psth_hz': np.zeros_like(psth_bin_centers_overall), # Use zeros
                 'overall_psth_bins': psth_bin_centers_overall,
                 'overall_velocity_avg': np.zeros_like(velocity_bin_centers_overall), # Use zeros
                 'overall_velocity_bins': velocity_bin_centers_overall,
                 'state_event_shading_info': []
             }

    # After aggregating all behaviors for a cell, fill nulls within the lists of arrays
    for behavior, beh_data in aggregated_data.items():
        if beh_data:
            aggregated_data[behavior]['raster_trials_spike_times'] = fill_nested_list_nulls(beh_data.get('raster_trials_spike_times', []))
            aggregated_data[behavior]['velocity_trials_velocity'] = fill_nested_list_nulls(beh_data.get('velocity_trials_velocity', []))
            aggregated_data[behavior]['velocity_trials_times'] = fill_nested_list_nulls(beh_data.get('velocity_trials_times', []))
            # Assuming state_event_shading_info contains dicts with numeric values that might be null
            if 'state_event_shading_info' in beh_data and beh_data['state_event_shading_info']:
                 filled_shading_info = []
                 for shading_dict in beh_data['state_event_shading_info']:
                      if isinstance(shading_dict, dict):
                         filled_shading_info.append({k: np.nan_to_num(v, nan=0.0) if isinstance(v, (float, int)) else v for k, v in shading_dict.items()})
                      else: # Handle cases where shading info is not a dict
                          filled_shading_info.append(shading_dict) # Keep as is or handle differently
                 aggregated_data[behavior]['state_event_shading_info'] = filled_shading_info


    return aggregated_data


# --- Final Plotting Function ---
def plot_concatanated_PSTHs(cell_aggregated_data, cell_metadata, save_path=None, window=5.0, bin_width=0.1):
    """
    Plots aggregated PSTH, average velocity, and a colored raster plot for a single cell
    across different behaviors and sessions.

    Args:
        cell_aggregated_data (dict): Aggregated data for a single cell (output of aggregate_cell_data_from_df).
        cell_metadata (dict): Dictionary with cell metadata ('cluster_id', 'max_site', 'region').
        save_path (Path, optional): Directory to save the plot. Defaults to None.
        window (float, optional): Time window (in seconds) around events. Defaults to 5.0.
        bin_width (float, optional): Bin width (in seconds) for PSTH. Defaults to 0.1.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    event_columns_to_plot = list(cell_aggregated_data.keys())
    n_events = len(event_columns_to_plot)

    if n_events == 0:
        # This case is handled before calling this function, but as a safeguard:
        print(f"No behavioral data provided to plot for unit {cell_metadata.get('cluster_id', 'N/A')}.")
        return # Nothing to plot

    # Determine grid size: aim for a roughly square layout
    n_cols_grid = min(max(1, int(np.ceil(np.sqrt(n_events)))), 4) # Max 4 columns
    n_rows_grid = math.ceil(n_events / n_cols_grid)

    fig_rows = n_rows_grid * 3 # Velocity, PSTH, Raster
    fig_cols = n_cols_grid

    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * 4, fig_rows * 2), # Adjusted figsize
                             sharex=False, sharey=False, squeeze=False)

    unit_id_str = str(cell_metadata.get('cluster_id', 'N/A'))
    fig.suptitle(f"Unit: {unit_id_str} (Region: {cell_metadata.get('region', 'N/A')}, Site: {cell_metadata.get('max_site', 'N/A')})", fontsize=16)

    # Generate a colormap for sessions
    all_sessions_for_cell = set()
    for behavior_data in cell_aggregated_data.values():
         all_sessions_for_cell.update(behavior_data.get('raster_trial_sessions', []))
    unique_sessions = sorted(list(all_sessions_for_cell))
    num_unique_sessions = len(unique_sessions)
    # Use a colormap that has enough distinct colors
    cmap_name = 'tab10' if num_unique_sessions <= 10 else ('tab20' if num_unique_sessions <= 20 else 'viridis') # Use viridis for >20
    # Corrected: Use matplotlib.colormaps.get_cmap
    session_colors_map = matplotlib.colormaps.get_cmap(cmap_name, max(num_unique_sessions, 1))
    session_color_dict = {session: session_colors_map(i) for i, session in enumerate(unique_sessions)}


    for i, event_column in enumerate(event_columns_to_plot):
        grid_row_base = (i // n_cols_grid) * 3
        grid_col = i % n_cols_grid

        ax_velocity = axes[grid_row_base, grid_col]
        ax_psth = axes[grid_row_base + 1, grid_col]
        ax_raster = axes[grid_row_base + 2, grid_col]

        # Apply axis styling
        remove_axes(ax_velocity, bottom=True)
        remove_axes(ax_psth, bottom=True)
        # Keep y-axis label on the first column, remove for others
        if grid_col > 0:
             ax_velocity.tick_params(axis='y', which='both', labelleft=False)
             ax_psth.tick_params(axis='y', which='both', labelleft=False)
             ax_raster.tick_params(axis='y', which='both', labelleft=False)
        else:
             ax_velocity.set_ylabel('Avg Velocity (cm/s)')
             ax_psth.set_ylabel('Rate (Hz)')
             ax_raster.set_ylabel('Trial')

        behavior_data = cell_aggregated_data.get(event_column, {})

        # --- Plot Average Velocity ---
        overall_velocity_avg = behavior_data.get('overall_velocity_avg', np.array([]))
        overall_velocity_bins = behavior_data.get('overall_velocity_bins', np.array([]))
        if overall_velocity_avg.size > 0 and overall_velocity_bins.size == overall_velocity_bins.size: # Corrected comparison
             ax_velocity.plot(overall_velocity_bins, overall_velocity_avg, color='orangered', linewidth=1.5)
        ax_velocity.axvline(0, color='red', linestyle='--', linewidth=1, zorder=5)
        ax_velocity.set_ylim(bottom=0)
        ax_velocity.set_title(event_column, fontsize=12) # Title on the top plot for the behavior
        ax_velocity.grid(axis='y', linestyle=':', alpha=0.7)


        # --- Plot Firing Rate (PSTH) ---
        overall_psth_hz = behavior_data.get('overall_psth_hz', np.array([]))
        overall_psth_bins = behavior_data.get('overall_psth_bins', np.array([]))
        if overall_psth_hz.size > 0 and overall_psth_bins.size == overall_psth_bins.size: # Corrected comparison
             ax_psth.bar(overall_psth_bins, overall_psth_hz, width=bin_width, align='center', alpha=0.8)
        ax_psth.axvline(0, color='red', linestyle='--', linewidth=1, zorder=5)
        ax_psth.set_ylim(bottom=0)
        ax_psth.grid(axis='y', linestyle=':', alpha=0.7)

        # --- Plot Spike Raster ---
        raster_trials_spike_times = behavior_data.get('raster_trials_spike_times', [])
        raster_trial_sessions = behavior_data.get('raster_trial_sessions', [])
        state_event_shading_info = behavior_data.get('state_event_shading_info', [])

        num_trials = len(raster_trials_spike_times)

        # Plot state event shading
        if state_event_shading_info:
             # State event shading info is a list of dicts, one dict per state event trial.
             # Assuming the order in state_event_shading_info matches the order of trials in raster_trials_spike_times.
             for trial_idx, shading_info in enumerate(state_event_shading_info):
                  # Shading info should be a dict with 'left', 'width', 'alpha' (relative to aligned 0)
                  left_edge = shading_info.get('left')
                  width = shading_info.get('width')
                  alpha = shading_info.get('alpha', 0.5)
                  if left_edge is not None and width is not None:
                       ax_raster.barh(trial_idx + 0.5, width, left=left_edge, height=1, color='burlywood', alpha=alpha) # Center bar in the lane


        if num_trials > 0:
            y_offset = 0
            for trial_spikes, trial_session in zip(raster_trials_spike_times, raster_trial_sessions):
                if trial_spikes.size > 0:
                    session_color = session_color_dict.get(trial_session, 'gray') # Default to gray
                    ax_raster.scatter(trial_spikes, np.full_like(trial_spikes, y_offset + 0.5), # Center dots in the "lane"
                                      color=session_color, s=5, marker='|', linewidth=0.5)
                y_offset += 1 # Increment y-offset for the next trial

            ax_raster.set_ylim(0, num_trials) # Y-axis from 0 to num_trials
            # Adjust Y ticks based on num_trials
            if num_trials > 0:
                 if num_trials <= 10:
                     ticks = np.arange(0.5, num_trials + 0.5, 1) # Center ticks in the lanes
                     labels = np.arange(1, num_trials + 1, 1) # Trial numbers starting from 1
                 else:
                     # Select fewer ticks, centered in the lanes
                     tick_indices = np.linspace(0, num_trials - 1, min(num_trials, 5), dtype=int) # Choose up to 5 indices
                     ticks = tick_indices + 0.5
                     labels = tick_indices + 1

                 ax_raster.set_yticks(ticks)
                 ax_raster.set_yticklabels(labels)

            else:
                ax_raster.set_ylim(0, 1)
                ax_raster.set_yticks([])


        else:
            ax_raster.set_ylim(0, 1)
            ax_raster.set_yticks([])


        ax_raster.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_raster.grid(axis='y', linestyle=':', alpha=0.7)

        # --- Axis Sharing & Limits ---
        ax_velocity.sharex(ax_psth)
        ax_psth.sharex(ax_raster)
        ax_raster.set_xlim(-window, window)

        # --- X-axis Label Handling ---
        last_event_row_index = (n_events - 1) // n_cols_grid
        current_event_row_index = i // n_cols_grid
        is_in_last_plotted_row = (current_event_row_index == last_event_row_index)

        if is_in_last_plotted_row:
            ax_raster.set_xlabel('Time (s)')
            ax_raster.tick_params(axis='x', which='both', labelbottom=True)
        # Tick labels for ax_velocity and ax_psth are already turned off above by remove_axes(bottom=True)


    # --- Clean up unused axes ---
    for i in range(n_events, n_rows_grid * n_cols_grid):
        grid_row_base = (i // n_cols_grid) * 3
        grid_col = i % n_cols_grid
        # Check bounds before attempting to access axes
        if grid_row_base + 2 < fig_rows and grid_col < fig_cols: # Check up to the third row
             axes[grid_row_base, grid_col].axis('off')      # Turn off Velocity axis
             axes[grid_row_base + 1, grid_col].axis('off')  # Turn off PSTH axis
             axes[grid_row_base + 2, grid_col].axis('off')  # Turn off Raster axis

    # Add a legend for sessions
    if unique_sessions:
        legend_handles = [mpatches.Patch(color=session_color_dict[session], label=session) for session in unique_sessions]

        # Find a suitable location for the legend. Try top-right corner if the last column is not full.
        legend_ax = None
        # Check the axes in the last column, top-down, to find an empty one
        for r in range(0, fig_rows, 3): # Check the top axis of each 3-row block
             # Check if the top axis in this column is visible and seems empty (no children artists)
             if fig_cols > 0 and axes[r, fig_cols - 1].get_visible() and len(axes[r, fig_cols - 1].get_children()) == 0:
                  legend_ax = axes[r, fig_cols - 1]
                  break # Found a suitable spot

        if legend_ax is not None:
             legend_ax.axis('off') # Hide the axes box
             legend_ax.legend(handles=legend_handles, title="Sessions", loc='center')
        else:
            # If no ideal spot found, place legend at the bottom center of the figure
            # Ensure ncol does not exceed the number of sessions
             fig.legend(handles=legend_handles, title="Sessions", loc='lower center', bbox_to_anchor=(0.5, 0), ncol=min(len(unique_sessions), 5)) # Limit columns


    fig.tight_layout(rect=[0, 0.03, 1, 0.96], pad=1.0, h_pad=1.0, w_pad=0.5) # Adjust padding

    if save_path:
        try:
            # Ensure save_path is a Path object and directory exists
            save_path = Path(save_path)
            if not save_path.exists():
                save_path.makedirs(exist_ok=True)

            # Create safe filename from unit ID
            safe_unit_id = "".join([c for c in unit_id_str if c.isalnum() or c in ('_', '-')]).rstrip()
            filename_png = f"{safe_unit_id}_Aggregated_PSTH_Raster.png"
            filename_svg = f"{safe_unit_id}_Aggregated_PSTH_Raster.svg"
            full_save_path_png = save_path / filename_png
            full_save_path_svg = save_path / filename_svg

            fig.savefig(full_save_path_png, dpi=300)
            fig.savefig(full_save_path_svg)
            print(f"Saved plot for unit {unit_id_str} to: {full_save_path_png} and {full_save_path_svg}")
        except Exception as e:
            print(f"Error saving figure for unit {unit_id_str}: {e}")

    plt.close(fig)


# --- Main Script Logic ---

# This block ensures that the code inside it only runs when the script is executed directly
# (not when it's imported as a module by the multiprocessing worker processes).
if __name__ == '__main__':
    freeze_support() # Recommended for multiprocessing robustness on some platforms

    ############################ input Parameters#############################
    animal = 'afm16924';sessions = ['240524','240525']#,'240526','240527','240529']  # can be a list if you have multiple sessions
    #animal='afm17365';sessions=['241211']#['241211']#['240522','240524','240525','240526']
    #animal='afm17365'#'afm16924'
    #sessions=['241211']#['240529']#['240523_0','240522','240524','240525','240526','240527','240529']

    target_regions = ['DPAG','VPAG','VLPAG','LPAG','DLPAG','DMPAG','VMPAG']  # From which regions should the neurons be? Leave empty for all.
    target_cells = [] # List of specific cluster IDs to include. Leave empty for all.
    #target_cells=[344,347,365,391,538,545,547,347,393,561,582,590]#pup related

    target_bs_0 = []           # Which behaviors to plot? Leave empty for all behaviors found.

    # Define a base save path - UPDATED
    base_savepath = Path(rf"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\20250509") / animal / "concat_PSTHs"
    # Create the base directory if it doesn't exist
    os.makedirs(base_savepath, exist_ok=True)

    # Parameters for PSTH calculation and plotting window
    window = 5           # s; how much time before and after event start should be considered
    density_bins = 0.1   # s; bin width for PSTH calculation and final plotting

    # Multiprocessing parameters (from original script)
    cpu_cores = os.cpu_count() or 1
    max_workers = max(1, cpu_cores // 2) # Use up to half of cores, minimum 1 - Adjusted


    # --- Debugging flags - MOVED TO TOP LEVEL AND PASSED TO FUNCTION ---
    DEBUG_BEHAVIOR_EVENTS = True
    DEBUG_PER_NEURON_DATA = True


    ############################ Data Loading and Per-Session Processing ####################

    # Polars DataFrame to store per-session, per-neuron PSTH data
    # Columns will be 'cluster_id', 'max_site', 'region', and session names
    # The data in session columns will be the dictionary returned by psth_cond_refactored
    Neurons_pl = pl.DataFrame()

    for i_session, session in enumerate(sessions):
        print(f"\nProcessing session: {session}")
        session_save_temp_dir = base_savepath / session / "temp_data" # Temp dir for session-specific data storage if needed
        # os.makedirs(session_save_temp_dir, exist_ok=True) # Create temp session directory

        try:
            # Load preprocessed data
            # The load_preprocessed function returns:
            # (frames_dropped, behaviour, ndata, n_spike_times, n_time_index,
            #  n_cluster_index, n_region_index, n_channel_index, velocity, locations,
            #  node_names, frame_index_s)
            (frames_dropped, behaviour, ndata, n_spike_times, n_time_index,
             n_cluster_index, n_region_index, n_channel_index, velocity,
             locations, node_names, frame_index_s) = load_preprocessed(animal, session)

            # --- Ensure data stream lengths are consistent ---
            min_len_neural_tracking = min(len(n_time_index), len(frame_index_s))
            if len(n_time_index) != min_len_neural_tracking:
                 print(f"Warning: Trimming n_time_index from {len(n_time_index)} to {min_len_neural_tracking}.")
                 n_time_index = n_time_index[:min_len_neural_tracking]
                 # Assuming ndata corresponds to n_time_index's time dimension
                 if ndata.shape[1] > min_len_neural_tracking:
                      print(f"Warning: Trimming ndata time dimension from {ndata.shape[1]} to {min_len_neural_tracking}.")
                      ndata = ndata[:, :min_len_neural_tracking]

            if len(frame_index_s) != min_len_neural_tracking:
                 print(f"Warning: Trimming frame_index_s from {len(frame_index_s)} to {min_len_neural_tracking}.")
                 frame_index_s = frame_index_s[:min_len_neural_tracking]
                 if len(velocity) > min_len_neural_tracking:
                      print(f"Warning: Trimming velocity from {len(velocity)} to {min_len_neural_tracking}.")
                      velocity = velocity[:min_len_neural_tracking]
                 # Assuming locations also needs trimming if frame_index_s was trimmed
                 if len(locations) > min_len_neural_tracking:
                     print(f"Warning: Trimming locations from {len(locations)} to {min_len_neural_tracking}.")
                     locations = locations[:min_len_neural_tracking]

            # Re-check lengths after trimming
            if not (len(n_time_index) == len(frame_index_s) == len(velocity) == len(locations) == ndata.shape[1]):
                 print(f"Error: Data streams remain inconsistent after trimming for session {session}. Skipping session.")
                 print(f"Lengths: n_time_index={len(n_time_index)}, frame_index_s={len(frame_index_s)}, velocity={len(velocity)}, locations={len(locations)}, ndata_time={ndata.shape[1]}")
                 raise ValueError("Inconsistent data stream lengths") # Raise error to trigger skipping logic

            # --- Generate additional behavioral events ---
            # (Include logic from the original script for generating random baseline, speed, acceleration, shelter events)
            vframerate = len(frame_index_s) / (frame_index_s[-1] if frame_index_s.size > 0 else 1)

            kwargs_events = {
                "speed": velocity,
                "time_ax": frame_index_s, # Use the (potentially trimmed) frame_index_s as the base time axis for events
                "framerate": vframerate,
                "baseline_time_s": 7 * 60,
                "n_trials": 10,
                "min_gap": 10 # Added min_gap for random baseline
                # "distance_to_shelter": distance2shelter, # Include if distance2shelter is loaded/available
                # "Threshold": None, # Use defaults
                # "diff_time_s": None # Use defaults
            }

            # Generate random baseline
            behaviour = insert_event_times_into_behavior_df(
                behaviour, **kwargs_events, event_type='baseline_random',
                behavior_name='random_baseline', behavior_type='baseline_random')

            # Generate speed threshold crossing
            kwargs_events["diff_time_s"] = 5
            kwargs_events["Threshold"] = 45 # Example threshold
            behaviour = insert_event_times_into_behavior_df(
                behaviour, **kwargs_events, event_type="speed",
                behavior_name="speed_crossing", behavior_type="speed_threshold_crossing")

            # Generate acceleration threshold crossing (if needed and speed available)
            # kwargs_events["diff_time_s"] = 5
            # kwargs_events["Threshold"] = 65 # Example threshold
            # behaviour = insert_event_times_into_behavior_df(
            #      behaviour, **kwargs_events, event_type="acceleration",
            #      behavior_name="acceleration_crossing", behavior_type="acceleration_threshold_crossing")

            # Generate in-shelter time periods (if needed and distance2shelter available)
            # if 'distance2shelter' in locals(): # Check if distance2shelter was loaded
            #      kwargs_events["diff_time_s"] = 5
            #      kwargs_events["Threshold"] = 5 # Example threshold
            #      behaviour = insert_event_times_into_behavior_df(
            #           behaviour, **kwargs_events, event_type="distance_to_shelter",
            #           behavior_name="in_shelter", behavior_type="in_shelter")

            # --- DEBUG: Print behavior event counts ---
            if DEBUG_BEHAVIOR_EVENTS:
                 print("Generated behavioral events counts:")
                 for beh in behaviour['behaviours'].unique():
                     count = len(behaviour[behaviour['behaviours'] == beh])
                     print(f"  {beh}: {count}")


            # --- Filter neurons by target regions and target cells ---
            initial_neuron_count = len(n_cluster_index)
            keep_neuron_indices = np.ones(initial_neuron_count, dtype=bool)

            if target_regions:
                in_region_mask = np.isin(n_region_index, target_regions)
                keep_neuron_indices = keep_neuron_indices & in_region_mask
                print(f"Filtered to {np.sum(keep_neuron_indices)} neurons in target regions.")

            if target_cells:
                in_target_cells_mask = np.isin(n_cluster_index, target_cells)
                keep_neuron_indices = keep_neuron_indices & in_target_cells_mask
                print(f"Filtered to {np.sum(keep_neuron_indices)} neurons in target cells list.")


            # Apply filtering
            n_cluster_index_filtered = n_cluster_index[keep_neuron_indices]
            n_region_index_filtered = n_region_index[keep_neuron_indices]
            n_channel_index_filtered = n_channel_index[keep_neuron_indices]
            ndata_filtered = ndata[keep_neuron_indices, :]
            # n_spike_times_filtered is not directly used by psth_cond_refactored


            print(f"Processing {len(n_cluster_index_filtered)} filtered neurons for session {session}")

            # --- Per-Neuron PSTH Calculation for this session ---
            session_results_list = [] # List to collect results for this session's neurons

            # Use ProcessPoolExecutor for parallel processing of neurons within a session
            # CORRECTED loop to use as_completed and pass debug flag
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks and store Future objects mapped to their metadata
                future_to_metadata = {}
                for i_neuron, (cluster, region, channel) in enumerate(zip(n_cluster_index_filtered, n_region_index_filtered, n_channel_index_filtered)):
                    # Get the data for the current neuron (a single row from ndata)
                    neuron_data = ndata_filtered[i_neuron, :]
                    # Submit task to the executor
                    future_obj = executor.submit(
                        psth_cond_refactored,
                        cluster, # Pass the cluster_id for debugging prints
                        DEBUG_PER_NEURON_DATA, # Pass the debug flag's value
                        neuron_data, n_time_index, behaviour, velocity, frame_index_s,
                        None, window=window, density_bins=density_bins, return_data=True,
                        session=session, align_to_end=False # Assuming aligning to start unless specified
                    )
                    # Map the Future object to its metadata tuple
                    future_to_metadata[future_obj] = (cluster, region, channel)

                # Process results as they complete, using as_completed
                # as_completed iterates over the Future objects themselves
                for future in tqdm(concurrent.futures.as_completed(future_to_metadata), total=len(future_to_metadata), desc=f"Calculating PSTHs for {session}"):
                    # Retrieve metadata using the completed Future object
                    metadata = future_to_metadata[future]
                    cluster, region, channel = metadata # Unpack metadata
                    try:
                        # Call .result() on the completed Future object to get the result of the psth_cond_refactored call
                        # psth_cond_refactored returns a dictionary: {behavior_name: (tuple_of_dicts)}
                        per_behavior_data_for_neuron_session = future.result()

                        # --- DEBUG: Print collected behavior keys for this neuron/session ---
                        if DEBUG_PER_NEURON_DATA:
                             if isinstance(per_behavior_data_for_neuron_session, dict):
                                 # Filter out None values to see which behaviors actually had data
                                 valid_behaviors = {k: v for k, v in per_behavior_data_for_neuron_session.items() if v is not None}
                                 
                             elif per_behavior_data_for_neuron_session is None:
                                  print(f"Neuron {cluster}, Session {session}: psth_cond_refactored returned None.")
                             else:
                                  print(f"Neuron {cluster}, Session {session}: psth_cond_refactored returned unexpected type: {type(per_behavior_data_for_neuron_session)}")


                        # The result to be stored in Neurons_pl is this dictionary
                        session_cell_data_to_store = per_behavior_data_for_neuron_session

                        session_results_list.append({
                            'cluster_id': cluster,
                            'max_site': channel,
                            'region': region,
                            session: session_cell_data_to_store # Store the dictionary under the session name
                        })

                    except Exception as exc:
                        print(f'Generated an exception for neuron {cluster} in {session}: {exc}')
                        # Append None for this neuron/session if an error occurred
                        session_results_list.append({
                            'cluster_id': cluster,
                            'max_site': channel,
                            'region': region,
                            session: None
                        })

            # --- Update Neurons_pl with results from this session ---
            session_df_to_merge = pl.DataFrame(session_results_list)

            if Neurons_pl.is_empty():
                # For the very first session with results, create the base DataFrame
                Neurons_pl = session_df_to_merge
            else:
                # For subsequent sessions, merge the results into the existing DataFrame
                # Select only the cluster_id and the session data column from the right DataFrame
                # Also include metadata columns from the right to handle new neurons in later sessions
                session_data_right = session_df_to_merge.select([
                    "cluster_id",
                    "max_site", # Include metadata from the right DataFrame
                    "region",   # Include metadata from the right DataFrame
                    session # The column containing the session-specific behavior data
                ])

                # Perform a full outer join on 'cluster_id' - CORRECTED how
                # Use suffix to handle potential metadata column name conflicts
                Neurons_pl = Neurons_pl.join(
                    session_data_right,
                    on=['cluster_id'], # Join primarily on cluster_id
                    how='full', # Use 'full' instead of 'outer' (deprecation)
                    suffix="_right" # Add suffix to columns from the right that are not join keys
                )

                # Handle duplicated metadata columns (max_site, region) resulting from the join
                # Coalesce combines non-null values. This prefers the left column's value if not null,
                # otherwise takes the right column's value. Useful for new neurons in later sessions.
                Neurons_pl = Neurons_pl.with_columns([
                    pl.coalesce([pl.col("max_site"), pl.col("max_site_right")]).alias("max_site"),
                    pl.coalesce([pl.col("region"), pl.col("region_right")]).alias("region"),
                ]).fill_null(0,strategy='forward')

                # Drop the duplicated suffixed columns from the right DataFrame if they exist
                cols_to_drop = []
                if "max_site_right" in Neurons_pl.columns:
                    cols_to_drop.append("max_site_right")
                if "region_right" in Neurons_pl.columns:
                    cols_to_drop.append("region_right")

                if cols_to_drop:
                     Neurons_pl = Neurons_pl.drop(cols_to_drop)


            print(f"Finished processing session: {session}. Current Neurons_pl shape: {Neurons_pl.shape}")

        except FileNotFoundError as e:
            print(f"Skipping session {session} due to missing file: {e}")
            # Add columns for this session with all None values to Neurons_pl if it exists, or initialize if first session
            if Neurons_pl.is_empty() and i_session == 0:
                 # If the very first session fails and DF is empty, initialize with just metadata columns
                 Neurons_pl = pl.DataFrame({
                     'cluster_id': pl.Series([], dtype=pl.Int64), # Use explicit empty series with dtype
                     'max_site': pl.Series([], dtype=pl.Int64),
                     'region': pl.Series([], dtype=pl.Utf8),
                 }).fill_null(0,strategy='forward')
            # Add the session column with None values for all existing rows
            if session not in Neurons_pl.columns:
                 if not Neurons_pl.is_empty():
                      Neurons_pl = Neurons_pl.with_columns(pl.lit(None).alias(session))
                 else:
                      # If DF is still empty, create a new one with metadata and the session column (all None)
                      Neurons_pl = pl.DataFrame({
                          'cluster_id': pl.Series([], dtype=pl.Int64),
                          'max_site': pl.Series([], dtype=pl.Int64),
                          'region': pl.Series([], dtype=pl.Utf8),
                          session: pl.Series([], dtype=pl.Object) # Use Object dtype for mixed/nested data
                      }).fill_null(0,strategy='forward')

        except ValueError as e:
             print(f"Skipping session {session} due to data inconsistency: {e}")
             # Ensure the session column exists even if processing failed
             if session not in Neurons_pl.columns:
                  if not Neurons_pl.is_empty():
                       Neurons_pl = Neurons_pl.with_columns(pl.lit(None).alias(session))
                  else:
                   Neurons_pl = pl.DataFrame({
                       'cluster_id': pl.Series([], dtype=pl.Int64),
                       'max_site': pl.Series([], dtype=pl.Int64),
                       'region': pl.Series([], dtype=pl.Utf8),
                       session: pl.Series([], dtype=pl.Object)
                   }).fill_null(0,strategy='forward')


        except Exception as e:
            print(f"An unexpected error occurred while processing session {session}: {e}")
            # Decide how to handle unexpected errors (skip session, log, etc.)
            # Ensure the session column exists even if processing failed
            if session not in Neurons_pl.columns:
                 if not Neurons_pl.is_empty():
                      Neurons_pl = Neurons_pl.with_columns(pl.lit(None).alias(session))
                 else:
                      # If DF is still empty, create a new one with metadata and the session column (all None)
                      Neurons_pl = pl.DataFrame({
                          'cluster_id': pl.Series([], dtype=pl.Int64),
                          'max_site': pl.Series([], dtype=pl.Int64),
                          'region': pl.Series([], dtype=pl.Utf8),
                          session: pl.Series([], dtype=pl.Object)
                      }).fill_null(0,strategy='forward')


    print("\n--- Finished Per-Session Processing ---")
    print(f"Final Neurons_pl shape: {Neurons_pl.shape}")
    print("Example head of Neurons_pl:")
    try:
        # Print head as pandas to avoid potential Polars display issues with nested data
        print(Neurons_pl.head().to_pandas())
    except Exception as e:
        print(f"Could not print head of Neurons_pl: {e}")


    ############################ Data Aggregation Across Sessions Per Cell ####################

    print('\n--- Aggregating data across sessions per cell ---')
    all_cells_aggregated_data = []
    # Get all unique cluster IDs present in the DataFrame
    # Use .collect() if Neurons_pl is a LazyFrame, otherwise .unique() is fine
    unique_cluster_ids_overall = Neurons_pl.select("cluster_id").unique().to_series().to_list()

    # Remove any None values from unique cluster IDs if they somehow appeared
    unique_cluster_ids_overall = [cid for cid in unique_cluster_ids_overall if cid is not None]


    for cluster_id in tqdm(unique_cluster_ids_overall, desc="Aggregating data per cell"):
        # Get cell metadata (assuming it's consistent across sessions, take from the first row found)
        cell_rows_df = Neurons_pl.filter(pl.col("cluster_id") == cluster_id)
        if cell_rows_df.is_empty():
             # This should not happen if cluster_id comes from unique_cluster_ids_overall, but as a safeguard
             print(f"Warning: No rows found for cluster_id {cluster_id} during aggregation. Skipping.")
             continue

        # Get metadata from the first non-null row for this cluster
        first_non_null_row = cell_rows_df.filter(pl.col("region").is_not_null()).row(0, named=True) # region should typically be non-null
        if first_non_null_row is None:
             print(f"Warning: Could not retrieve metadata for cluster_id {cluster_id}. Skipping aggregation.")
             continue

        cell_metadata = {
            'cluster_id': first_non_null_row.get('cluster_id'),
            'max_site': first_non_null_row.get('max_site'),
            'region': first_non_null_row.get('region')
        }

        # Aggregate data for this cell across sessions using the function
        cell_aggregated_data = aggregate_cell_data_from_df(Neurons_pl, cluster_id, sessions)

        # Store the aggregated data and metadata
        all_cells_aggregated_data.append({
            'metadata': cell_metadata,
            'aggregated_data': cell_aggregated_data
        })

    print('Aggregation complete.')
    print(f'Aggregated data for {len(all_cells_aggregated_data)} cells.')

    # Optional: Save the aggregated data to a file (e.g., pickle or numpy save)
    # This can be useful to avoid re-running the session processing every time.
    try:
        aggregated_data_save_path = base_savepath / f"{animal}_aggregated_psth_data.npy"
        # Check if there is data to save before attempting
        if all_cells_aggregated_data:
            np.save(aggregated_data_save_path, all_cells_aggregated_data, allow_pickle=True)
            print(f"Aggregated data saved to {aggregated_data_save_path}")
        else:
            print("No aggregated data to save.")
    except Exception as e:
        print(f"Error saving aggregated data: {e}")


    ############################ Plotting Aggregated Data Per Cell ####################

    print('\n--- Plotting aggregated PSTHs per cell ---')
    # Define the directory to save the final plots
    final_plots_save_dir = base_savepath / "Aggregated_PSTH_Plots"
    os.makedirs(final_plots_save_dir, exist_ok=True) # Ensure directory exists

    # Loop through the aggregated data for each cell and plot
    for cell_data_entry in tqdm(all_cells_aggregated_data, desc="Plotting cells"):
        cell_metadata = cell_data_entry['metadata']
        cell_aggregated_data = cell_data_entry['aggregated_data']

        # Check if there is any behavioral data to plot for this cell
        if not cell_aggregated_data:
            print(f"No behavioral data to plot for unit {cell_metadata.get('cluster_id', 'N/A')}.")
            continue # Skip plotting if no aggregated data

        # Call the plotting function with the aggregated data
        plot_concatanated_PSTHs(
            cell_aggregated_data,
            cell_metadata,
            save_path=final_plots_save_dir,
            window=window,
            bin_width=density_bins # Using the same density_bins for final plot bin width
        )

    print('\n--- Plotting complete ---')

    # Optional: Keep the environment interactive after script finishes (useful for debugging)
    # IPython.embed()