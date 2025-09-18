# -*- coding: utf-8 -*-
""" 
Analysis/PopulationLibs/read_data_light.py

Created on: 2025-07-23
Author: Dylan Festa

(generated   with copy-paste from preprocessing/helperFunctions.py)

This module provides the `load_preprocessed` function, identitcal to the one 
in `preprocessing/helperFunctions.py`, but with less overhead (e.g. no 
dependencies on movement,spikeinterface, etc)
"""
from __future__ import annotations
from typing import List, Union,Tuple

import os
from pathlib import Path, PureWindowsPath
import re


import pandas as pd, numpy as np, xarray as xr
import pickle
import random

def is_session(s: str) -> bool:
    """
    A session is exactly six digits starting with '2' (e.g., '240522')
    that might be followed by other stuff like: '240622_0_test123_small_bins'.
    """
    return re.fullmatch(r"2\d{5}(?:_[a-z0-9]+)*", s) is not None

def is_mouse(s: str) -> bool:
    """Exactly three lowercase letters followed by five digits."""
    return re.fullmatch(r"[a-z]{3}\d{5}", s) is not None

def check_mouse_session(mouse: str, session: str) -> Union[None, ValueError]:
    if not is_mouse(mouse):
        return ValueError(f"Animal {mouse} is not a valid mouse ID. It should be in the format 'abc12345'.")
    if not is_session(session):
        return ValueError(f"Session {session} is not valid. It should be in the format '201234' or '201234_some_suffix'.")
    return None



def convert_unc_path(win_path_str: str) -> Path:
    """
    Converts a Windows UNC path to a POSIX path. Also accounting for server names and shares.
    Author: chatGPT o3-mini
    """
    # Parse the Windows UNC path.
    p = PureWindowsPath(win_path_str)
    
    # p.parts for a UNC path is something like:
    # ('\\\\gpfs.corp.brain.mpg.de\\stem', 'data', 'project_hierarchy', 'data', 'paths-Copy.csv')
    if not p.parts:
        raise ValueError("Invalid UNC path format.")
    
    # The first part contains both server and share.
    unc_root = p.parts[0].lstrip("\\")  # e.g. "gpfs.corp.brain.mpg.de\\stem"
    try:
        server, share = unc_root.split("\\", 1)
        share = share.rstrip("\\")  # Remove trailing backslash if present
    except ValueError:
        raise ValueError("Invalid UNC path format.")
    
    # Map the UNC share to your local mount.
    if server.lower() == 'gpfs.corp.brain.mpg.de':
        local_root = Path("/gpfs") / share
    else:
        raise ValueError(f"Server '{server}' is not recognized. Please update the mapping.")

    # Use the remaining parts from the UNC path.
    tail_parts = p.parts[1:]
    return local_root.joinpath(*tail_parts)    

def maybe_convert(cell):
    """
    This must be applied to Pandas dataframe as
    my_dataframe.map(maybe_convert).
    Looks for all path strings and converts them to POSIX paths.
    author: chatGPT o3-mini
    
    """
    if isinstance(cell, str) and cell.startswith(r"\\gpfs.corp.brain.mpg.de"):
        try:
            # Convert the UNC path and return its string representation.
            return convert_unc_path(cell)
        except Exception as e:
            # Optionally log the exception e.
            return cell  # If conversion fails, return the original cell.
    else:
        return cell


def get_path_of_paths_file():
    """
    Returns the absolute path to the paths.csv file.
    Accounting for windows and unix differences.
    """
    # Check if the current directory is a Windows path
    if os.name == 'nt':
        # If it's a Windows path, use the Windows-style path
        return Path('\\\\gpfs.corp.brain.mpg.de\\stem\\data\\project_hierarchy\\data\\paths-Copy.csv')
    else:
        # If it's a Unix path, use the Unix-style path
        return Path('/gpfs/stem/data/project_hierarchy/data/paths-Copy.csv')


def convert_to_unix_path(input_path: str,gpfs_root='/gpfs') -> Path:
    """
    Converts a Windows-style path to a Unix-style path. But only if
    the system is not Windows. (i.e. assumes `input_path` is a Windows path, and
    does not require any conversion or change if the system is Windows).

    Args:
        input_path (str): The Windows-style path starting with 'gpfs.corp.brain.mpg.de'.
        gpfs_root (str): The Unix-style root path to replace the domain part, default is '/gpfs'.
    Returns:
        Path: The converted Unix-style path as a Path object, starting with '/gpfs' 
        or the specified root.
    """
    # if windows, return the path as is
    if os.name == 'nt':
        return Path(input_path)
    # Replace the backslashes with forward slashes
    input_path = input_path.replace('\\', '/')
    # Replace the domain part ('gpfs.corp.brain.mpg.de') with '/gpfs'
    input_path = input_path.replace('//gpfs.corp.brain.mpg.de', gpfs_root)
    # Convert the string to a Path object
    return Path(input_path)   
    

def get_good_paths():
    """
    Returns the paths to the data files that are not empty.
    """
    path_of_paths = get_path_of_paths_file()
    assert os.path.exists(path_of_paths), f"File not found: {path_of_paths}"
    df_paths=pd.read_csv(path_of_paths)
    # filter for 'time_start_stop' column that is not NaN
    mask_a = df_paths['time_start_stop'].notna()
    # filter for 'preprocessed' column is not nan and if it exisist is a valid path
    def _is_good_path(p):
        if pd.isna(p):
            return False
        p = convert_to_unix_path(p)
        return os.path.exists(p)
    mask_b = df_paths['preprocessed'].apply(_is_good_path)
    # inside preprocessed folder, check that the file  np_neural_data.npy exists
    def _has_neural_data(p):
        if pd.isna(p):
            return False
        p = convert_to_unix_path(p)
        return os.path.exists(p / 'np_neural_data.npy')
    mask_c = df_paths['preprocessed'].apply(_has_neural_data)
    # merge masks
    mask_all = mask_a & mask_b & mask_c
    # return the paths that are not empty
    df_paths_new = df_paths[mask_all]
    # sort by 'Mouse_ID' and 'session'
    df_paths_new = df_paths_new.sort_values(by=['Mouse_ID', 'session'])
    return df_paths_new.reset_index(drop=True)


def get_good_animals() -> List[str]:
    """
    Returns a list of unique animal IDs from the paths DataFrame.
    """
    df_paths = get_good_paths()
    df_paths.sort_values(by=['Mouse_ID'], inplace=True)
    return df_paths['Mouse_ID'].unique()

def get_good_sessions(the_animal: str) -> List[str]:
    """
    Returns a list of unique session IDs for a given animal from the paths available
    (generated using get_good_paths).
    """
    df_paths = get_good_paths()
    df_paths_animal = df_paths[df_paths['Mouse_ID'] == the_animal]
    # error if empty
    if df_paths_animal.empty:
        raise ValueError(f"No sessions found for animal: {the_animal}")
    return df_paths_animal['session'].unique().tolist()

def get_paths(animal=None, session=None,
        csvfile=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv",
        csvfile_posix="/gpfs/stem/data/project_hierarchy/data/paths-Copy.csv"):
    """
    Retrieves paths from excel
    if no session is specified, then entire pd is returned
    

    Parameters
    ----------
    session : name of session, as given in the .csv
    Animal: if specified, all sessions from tha animal are returned

    Returns
    -------
    pd with paths to data

    """
    
    # Reads the full paths file 
    if os.name == 'nt':#windows
        csvfile=Path(csvfile).as_posix()
        try:
            paths=pd.read_csv(csvfile)
        except:
            raise ValueError(f"could not open {csvfile}. check you have access to the folder (GPFS) if running as administrator")
    elif os.name == 'posix': # Linux
        csvfile = Path(csvfile_posix)
        if not csvfile.exists():
            raise ValueError(f"Path {csvfile_posix} does not exist. Please check the path.")
        paths = pd.read_csv(csvfile)
         
    if animal is not None:
        apaths=paths[paths['Mouse_ID']==animal]
        _ret = apaths
        if session is not None:
            sesline=apaths[apaths['session']==session]
            _ret = sesline.squeeze()
    
    elif animal is None:
        sesline=paths[paths['session']==session]
        _ret = sesline.squeeze().astype(str)
    else:
        _ret = paths
    
    if os.name == 'nt':
        return _ret
    elif os.name == 'posix':
        _ret2 = _ret.map(maybe_convert)
        return _ret2


def load_preprocessed(animal, session):
    """
    loads the data output from preprocess_all.py
    Parameters
    ----------
    
    animal (str) 
        Which animal to load data for
    session (str)
        Which session to load data for

    Returns
    --------
    frames_dropped (int)
        frame difference between frame index and velocity vector (positive numbers mean there is more in nidq frame index)
    behaviour (pd.DataFrame)
        DataFrame with the behaviour data
    ndata (np.ndarray)
        Neural data, shape (n_clusters, n_timepoints)
    n_spike_times (list)
        List of spike times for each cluster
    n_time_index (np.ndarray)
        Time index for neural data
    n_cluster_index (np.ndarray)
        Cluster index for neural data
    n_region_index (np.ndarray)
        Region index for neural data
    n_channel_index (np.ndarray)
        Channel index for neural data
    velocity (np.ndarray)
        Velocity data, shape (n_timepoints,)
    locations (np.ndarray)
        Locations data, shape (n_timepoints, 2)
    node_names (list)
        List of node names for tracking
    frame_index_s (np.ndarray)
        Frame index in seconds, shape (n_timepoints,)
    """

    err=check_mouse_session(animal, session)
    if err is not None:
        raise err

    paths=get_paths(animal, session)
    preprocessed_path=str(paths['preprocessed'])
    if os.name == 'posix':
        path_behaviour = Path(preprocessed_path) / 'behaviour.csv'
        path_tracking = Path(preprocessed_path) / 'tracking.npy'
        if not path_behaviour.exists() or not path_tracking.exists():
            raise FileNotFoundError(f"Required file `behaviour.npy` not found in {preprocessed_path}.")
        if not path_tracking.exists():
            raise FileNotFoundError(f"Tracking file `tracking.npy` not found in {preprocessed_path}.")
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
    
   
    if os.name == 'posix':  # For Mac/Linux
        path_ndata_dict = Path(preprocessed_path) / 'np_neural_data.npy' 
        if not path_ndata_dict.exists():
            raise FileNotFoundError(f"Data file `np_neural_data.npy` not found in {preprocessed_path}.")
    else:  # For Windows
        path_ndata_dict = fr"{preprocessed_path}\np_neural_data.npy"
    ndata_dict=np.load(path_ndata_dict, allow_pickle=True).item()
    ndata=ndata_dict['n_by_t']
    n_time_index=ndata_dict['time_index']
    n_cluster_index=ndata_dict['cluster_index']
    n_region_index=ndata_dict['region_index']
    n_channel_index=ndata_dict['cluster_channels']
    n_spike_times=ndata_dict['n_spike_times']
    
    return frames_dropped, behaviour, ndata,n_spike_times, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s

def load_preprocessed_dict(animal, session):
    """
    Reads the preprocessed data, returning a dictionary with the same keys as load_preprocessed.
    
    Parameters
    ----------
    animal : str
        The animal ID.
    session : str
        The session ID.

    Returns
    -------
    dict
        A dictionary containing the preprocessed data, with keys:
        - 'frames_dropped'
        - 'behaviour'
        - 'ndata'
        - 'spike_times'
        - 'time_index'
        - 'cluster_index'
        - 'region_index'
        - 'channel_index'
        - 'velocity'
        - 'distance_from_shelter'
        - 'locations'
        - 'node_names'
        - 'frame_index_s'
    """

    err=check_mouse_session(animal, session)
    if err is not None:
        raise err

    frames_dropped, behaviour, ndata, n_spike_times, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s = load_preprocessed(animal, session)
    # add distance from shelter too!
    
    paths=get_paths(animal, session)
    preprocessed_path=str(paths['preprocessed'])
    if os.name == 'posix':
        path_behaviour = Path(preprocessed_path) / 'behaviour.csv'
        path_tracking = Path(preprocessed_path) / 'tracking.npy'
        if not path_behaviour.exists() or not path_tracking.exists():
            raise FileNotFoundError(f"Required file `behaviour.npy` not found in {preprocessed_path}.")
        if not path_tracking.exists():
            raise FileNotFoundError(f"Tracking file `tracking.npy` not found in {preprocessed_path}.")
    else:  # For Windows
        path_behaviour = fr"{preprocessed_path}\behaviour.csv"
        path_tracking = fr"{preprocessed_path}\tracking.npy"    
    tracking=np.load(path_tracking, allow_pickle=True).item()
    distance_from_shelter=tracking['distance_to_shelter'] 
    
    return {
        'frames_dropped': frames_dropped,
        'behaviour': behaviour,
        'ndata': ndata,
        'spike_times': n_spike_times,
        'time_index': n_time_index,
        'cluster_index': n_cluster_index,
        'region_index': n_region_index,
        'channel_index': n_channel_index,
        'velocity': velocity,
        'distance_from_shelter': distance_from_shelter,
        'locations': locations,
        'node_names': node_names,
        'frame_index_s': frame_index_s
    }


def save_preprocessed_dict(animal: str, session: str, save_path: Union[str, Path]) -> None:
    """
    Reads the preprocessed data, exactly like in `load_preprocessed_dict`, and then
    saves it as a single pickled file in `save_path`, in the format:
    {animal}_{session}_preprocessed_dict.pkl

    Parameters
    ----------
    animal : str
        The animal ID.
    session : str
        The session ID.
    save_path : str | pathlib.Path
        Destination directory where the pickle will be written.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the animal or session format is invalid.
    FileNotFoundError
        If the destination directory does not exist.
    """
    dest = Path(save_path)
    if not dest.exists():
        raise FileNotFoundError(f"Destination folder {dest} does not exist.")

    data_dict = load_preprocessed_dict(animal, session)
    save_filename = f"{animal}_{session}_preprocessed_dict.pkl"
    save_filepath = dest / save_filename
    pd.to_pickle(data_dict, save_filepath)
    print(f"Preprocessed data for {animal} session {session} saved to {save_filepath}.")
    return None


def load_local_preprocessed_dict(animal: str, session: str, load_path: Union[str, Path]) -> Dict:
    """
    Loads a locally saved preprocessed data dictionary from a pickle file.

    Parameters
    ----------
    animal : str
        The animal ID.
    session : str
        The session ID.
    load_path : str | pathlib.Path
        Directory where the pickle file is located.

    Returns
    -------
    dict
        The preprocessed data dictionary.

    Raises
    ------
    ValueError
        If the animal or session format is invalid.
    FileNotFoundError
        If the pickle file does not exist.
    """
    err = check_mouse_session(animal, session)
    if err is not None:
        raise err

    load_dir = Path(load_path)
    if not load_dir.exists():
        raise FileNotFoundError(f"Load directory {load_dir} does not exist.")

    load_filename = f"{animal}_{session}_preprocessed_dict.pkl"
    load_filepath = load_dir / load_filename
    if not load_filepath.exists():
        raise FileNotFoundError(f"Pickle file {load_filepath} does not exist.")

    data_dict = pd.read_pickle(load_filepath)
    return data_dict

def convert_to_behaviour_timestamps(mouse:str,session:str,behaviour:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the behaviour DataFrame into a new dataframe where each row is a unique behaviour.

    Arguments
    ---------
    mouse : str
        The mouse identifier, added as a new column in the output DataFrame.
    session : str
        The session identifier, added as a new column in the output DataFrame.
    behaviour : pd.DataFrame
        The input DataFrame containing behaviour data with 
        columns 'behaviours', 'start_stop', 'frames_s', and 'behavioural_category'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns:
        - 'mouse': The mouse identifier.
        - 'session': The session identifier.
        - 'behavioural_category': The category of the behaviour.
        - 'behaviour': The unique behaviour names.
        - 'n_trials': How many times the behaviour occurred in the session.
        - 'is_start_stop': True if the behaviour is a start/stop event, 
                    False if the behaviour is a 'POINT' event.
        - 'start_stop_times': A list of tuples containing the start and
                    stop times of the behaviour for start/stop events.
        - 'point_times': A list of times for 'POINT' events.
        
    Throws
    ------
    ValueError
        If the input DataFrame does not contain the required columns.
    ValueError
        If there is no matching 'STOP' event for a 'START' event in the same behaviour.
    """
    df_grouped = behaviour.groupby('behaviours')
    rows_ret = []
    for name, group in df_grouped:
        if 'start_stop' not in group.columns or 'frames_s' not in group.columns:
            raise ValueError("Input DataFrame must contain 'start_stop' and 'frames_s' columns.")

        is_start_stop = group['start_stop'].str.contains('START|STOP').any()
        start_stop_times = []
        point_times = []
        total_duration = 0.0

        if is_start_stop:
            startstopvalues = group['start_stop'].values
            # check that structure alternates as ['START','STOP','START','STOP'...]
            if not (np.all(startstopvalues[::2] == 'START') and np.all(startstopvalues[1::2] == 'STOP')):
                raise ValueError(f"Behaviour '{name}' does not alternate 'START' and 'STOP' correctly.")

            start_times = group[group['start_stop'] == 'START']['frames_s'].values
            stop_times = group[group['start_stop'] == 'STOP']['frames_s'].values
            # Check if following start events are always after previous stop events
            if len(start_times) > 1 and not np.all(start_times[1:] >= stop_times[:-1]):
                raise ValueError(f"Behaviour '{name}' has START events that are not after previous STOP events.")
            n_trials = len(start_times)
            for _start, _stop in zip(start_times, stop_times):
                if _start > _stop:
                    raise ValueError(f"Behaviour '{name}' has START events that are not before STOP events.")
                total_duration += (_stop - _start)
                start_stop_times.append((_start, _stop))

            if len(start_times) != len(stop_times):
                raise ValueError(f"Behaviour '{name}' has mismatched START and STOP events.")

        else:
            point_times = group['frames_s'].values.tolist()
            n_trials = len(point_times)

        behavioural_category = group['behavioural_category'].iloc[0]

        rows_ret.append({
            'mouse': mouse,
            'session': session,
            'behavioural_category': behavioural_category,
            'behaviour': name,
            'n_trials': n_trials,
            'is_start_stop': is_start_stop,
            'start_stop_times': start_stop_times,
            'total_duration': total_duration,
            'point_times': point_times
        })
    df_ret = pd.DataFrame(rows_ret, columns=['mouse', 'session', 'behavioural_category', 'behaviour',
                                         'n_trials', 'is_start_stop', 'total_duration', 
                                         'start_stop_times', 'point_times'])
    # sort by n_trials
    df_ret.sort_values(by='n_trials', ascending=False, inplace=True)
    return df_ret

def intervals_to_nan_vector(intervals_list: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a list of (t1,t2) intervals, returns two np.Array :
    x = (t1,t2,NaN,t3,t4,NaN, t5,t6, NaN...)
    y = ( 1,1, NaN, 1, 1, NaN, 1, 1, NaN ...)

    Arguments
    ----------
    intervals_list : List
        List of (t1,t2) intervals.

    Returns
    -------
    x : np.array
        1D array with start and end times interleaved with NaNs.
    y : np.array
        1D array with 1s for valid intervals and NaNs for gaps.
    """

    if len(intervals_list) == 0:
        return np.array([]), np.array([])

    arr = np.array(intervals_list, dtype=float).reshape(-1)
    n = len(intervals_list)
    if n == 1:
        x = arr
        y = np.ones_like(arr)
    else:
        x = np.full(3 * n - 1, np.nan)
        y = np.full(3 * n - 1, np.nan)
        x[0::3] = arr[0::2]
        x[1::3] = arr[1::2]
        y[0::3] = 1
        y[1::3] = 1
    return x, y    
   
   
def behaviour_startstop_df_to_segments(behaviour_timestamps_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts plottable coordinates from the behaviour start stop DataFrame.

    Arguments
    ----------
    behaviour_timestamps_df : pd.DataFrame
        DataFrame obtained using the function `convert_to_behaviour_timestamps`.

    Returns
    -------
    xy_all : List[Tuple[np.ndarray, np.ndarray]]
        List of tuples, each tuple containing (x,y) numpy arrays for plotting.
        The y values are all the same, the x values are the start and stop times.
        Both x and y have NaN values interleaved to indicate gaps.
    behaviour_dict : dict
        Dictionary with behaviour names and numbers matching the y values for that
        particular behaviour.
    """
    xy_all = []
    idx_dict = 1
    behaviour_dict = {}
    for _, row in behaviour_timestamps_df.iterrows():
        if row['is_start_stop']:
            # If the row is a start-stop row, extract the start and stop times
            intervals_list = row['start_stop_times']
            _x, _y = intervals_to_nan_vector(intervals_list)
            _y *= idx_dict
            xy_all.append((_x, _y))
            # Add the behavior name to the dictionary
            behaviour_dict[row['behaviour']] = int(idx_dict)
            idx_dict += 1
    return xy_all, behaviour_dict
   


def add_padding_to_intervals(intervals_list: List[Tuple[float, float]],
        pad_left: float, pad_right: float) -> List[Tuple[float, float]]:
    """
    Given a list of (start, stop) intervals, adds left and right padding to each interval.
    If padding results in intervals touching or overlapping, they are merged.

    Arguments
    ----------
    intervals_list : List[Tuple[float, float]]
        List of (start, stop) intervals.
    pad_left : float
        Amount of time to subtract from the start of each interval.
    pad_right : float
        Amount of time to add to the end of each interval.

    Returns
    -------
    List[Tuple[float, float]]
        New list of (start, stop) intervals with padding applied and merged if necessary.
    """
    if not intervals_list:
        return []


    # Small tolerance to treat "touching" intervals as overlapping in the presence of FP noise.
    EPS = 1e-12

    # Normalize and pad each interval.
    padded: List[Tuple[float, float]] = []
    for (s, e) in intervals_list:
        s -= pad_left
        e += pad_right
        # Ensure start <= end after padding (handles extreme negative pads)
        if e < s:
            raise ValueError(f"Padding results in invalid interval: ({s}, {e}), did you use negative padding values?")
        padded.append((s, e))

    # Sort by start then end.
    padded.sort(key=lambda iv: (iv[0], iv[1]))

    # Merge overlapping or touching intervals.
    padded_merged: List[Tuple[float, float]] = []

    cur_start, cur_end = padded[0]
    for s, e in padded[1:]:
        if s <= cur_end + EPS:  # overlaps or touches
            if e > cur_end:
                cur_end = e
        else:
            padded_merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    # and last one...
    padded_merged.append((cur_start, cur_end))

    return padded_merged    


def merge_time_interval_list(intervals_list1: List[Tuple[float, float]], 
                             intervals_list2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge two lists of (start, stop) intervals into a single, non-overlapping list.
    Intervals that touch or overlap are merged together.

    Arguments
    ----------
    intervals_list1 : List[Tuple[float, float]]
        First list of (start, stop) intervals.
    
    """
    # first, combine both lists
    all_intervals = intervals_list1 + intervals_list2
    if not all_intervals:
        raise ValueError("Both interval lists are empty!")
    # Sort by start time
    all_intervals.sort(key=lambda x: x[0])
    merged = [all_intervals[0]]
    for current in all_intervals[1:]:
        last = merged[-1]
        # If current starts before or at the end of last, merge
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged
    
def generate_behaviour_startstop_segments(
        behaviour_timestamps_df: pd.DataFrame,
        behaviour_idxs_dict: dict | None = None,
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts plottable coordinates from the behaviour DataFrame, for start-stop behaviours.
    (where `is_start_stop` is True).
    If `behaviour_idxs_dict` is provided, it will use the indices from that dictionary.
    If None, it will generate a dictionary mapping behaviour names to unique indices (starting from 1).
    If multiple behaviours share the same index, their intervals will be merged (touching/overlapping intervals are merged).

    Arguments
    ----------
    behaviour_timestamps_df : pd.DataFrame
        DataFrame obtained using the function `convert_to_behaviour_timestamps`.
    behaviour_idxs_dict : dict | None, default None
        If None, infers the behaviour names and indices from the DataFrame.
        If provided, it should be a dictionary with behaviour names as keys and indices as values.
        The behaviour names should match those in the DataFrame.
        Warning: indexes can be repeated for multiple behaviours, resulting in the
        plot having a merged timestamps for those behaviours. The entry in
        behaviour_dict_plot will also contain both behaviour names.

    Returns
    -------
    xy_all : List[Tuple[np.ndarray, np.ndarray]]
        List of tuples, each tuple containing (x,y) numpy arrays for plotting.
        the x values are the start and stop times, the y values are 
        different for each behaviour, starting from 1 and following the ordering
        in behaviour_idx_dict if provided.
        Both x and y have NaN values interleaved to indicate gaps.
    behaviour_dict_plot : dict
        Dictionary with behaviour names and numbers matching the y values for that
        particular behaviour. In case `behaviour_idxs_dict` contains repeated indices,
        the behavior name will be `behaviour1_and_behaviour2` where `behaviour1` and `behaviour2`
        are the names of the behaviours that share the same index.
    """

    # Step 1: Generate behaviour_idxs_dict if None
    if behaviour_idxs_dict is None:
        _behaviour_idxs_dict = {row['behaviour']: i+1 for i, row in enumerate(behaviour_timestamps_df.itertuples()) if row.is_start_stop}
    else:
        # copy dictionary and remove 'none' entry if present
        _behaviour_idxs_dict = behaviour_idxs_dict.copy()
        if 'none' in _behaviour_idxs_dict:
            del _behaviour_idxs_dict['none']
        # If needed, remap values to consecutive integers starting from 1, preserving the mapping structure
        # (do not modify input dict in place)
        unique_vals = sorted(set(_behaviour_idxs_dict.values()))
        val_map = {old: new+1 for new, old in enumerate(unique_vals)}
        _behaviour_idxs_dict = {k: val_map[v] for k, v in _behaviour_idxs_dict.items()}

    # Step 2: For each unique index, collect all behaviours sharing that index
    idx_to_behaviours = {}
    for beh, idx in _behaviour_idxs_dict.items():
        idx_to_behaviours.setdefault(idx, []).append(beh)

    xy_all = []
    behaviour_dict_plot = {}
    for idx, beh_list in idx_to_behaviours.items():
        # Merge all intervals for all behaviours with this index
        merged_intervals = []
        for beh in beh_list:
            row = behaviour_timestamps_df.loc[behaviour_timestamps_df['behaviour'] == beh]
            if row.empty or not row.iloc[0]['is_start_stop']:
                continue
            intervals = row.iloc[0]['start_stop_times']
            merged_intervals = merge_time_interval_list(merged_intervals, intervals)
        if not merged_intervals:
            continue
        _x, _y = intervals_to_nan_vector(merged_intervals)
        _y *= idx
        xy_all.append((_x, _y))
        # If multiple behaviours, join names
        beh_name = '_and_'.join(sorted(beh_list))
        behaviour_dict_plot[beh_name] = idx
    return xy_all, behaviour_dict_plot
    
def generate_behaviour_labels(behaviour_timestamps: pd.DataFrame, 
                    time_bin_centers: np.ndarray,
                    behaviour_labels_dict: dict) -> xr.DataArray:
    """
    Given the timestamps dataframe (generated from convert_to_behaviour_timestamps), a vector of time bin 
    centers, and a dictionary mapping behaviour names to indices, this function generates an xarray
    label vector, where each time bin is labeled with the corresponding behaviour index. 
    The key 'none' in the dictionary indicates the label for bins where no behaviour is detected.
    If the dictionary does not have a 'none' key, it will use index -1.

    Warning: it does not handle well superimposed behaviours, i.e. if two behaviours overlap in time, the last one will overwrite the previous one.

    Arguments
    ---------
    behaviour_timestamps: pd.DataFrame
        DataFrame generated from `convert_to_behaviour_timestamps`
    time_bin_centers: np.ndarray
        1D array of time bin centers or timestamps in general
    behaviour_labels_dict: dict
        Dictionary mapping behaviour names to indices, e.g. {'pup_grab': 0, 'pup_retrieve': 1, ...}
    
    Returns
    -------
    xr.DataArray
        An xarray with the following structure:
        - Dimensions: ['time_bins']
        - Data: 1D array of behaviour indices corresponding to each time bin
        - Attributes:
            - 'behaviour_labels_dict': The original dictionary mapping behaviour names to indices
            - 'labels_named': A an array of strings of the same length as the data, containing the behaviour names
            
    Throws
    ------
    Warning if any behaviour in `behaviour_labels_dict` is not found in `behaviour_timestamps`.

    """
    # Create an empty array for the labels
    T = len(time_bin_centers)
    # check the 'none' key is in the dictionary, if not, it defaults to -1
    if 'none' not in behaviour_labels_dict:
        # if -1 is present, throw an error
        if -1 in behaviour_labels_dict.values():
            raise ValueError("behaviour labels dictionary already contains -1 as a key. Please remove it, or include a 'none' key.")
        behaviour_labels_dict['none'] = -1
    none_label = behaviour_labels_dict['none']
    # inizialize all labels with all 'none' entries
    labels = np.full(T, none_label, dtype=int)
    # Iterate through each behaviour and its timestamps
    for behaviour, idx in behaviour_labels_dict.items():
        # Skip 'none' since it's the default label used
        if behaviour == 'none':
            continue
        # check if behaviour is present in the 'behaviour' column
        if  behaviour not in behaviour_timestamps['behaviour'].values:
            print(f"Warning: behaviour '{behaviour}' not found in behaviour_timestamps.")
            continue
        # Get the behaviour timestamps
        timestamps_row = behaviour_timestamps.loc[behaviour_timestamps['behaviour'] == behaviour, 'start_stop_times'].values
        # if timestamps is empty, send a warning and continue
        if len(timestamps_row) == 0:
            print(f"Warning: behaviour '{behaviour}' has no timestamps in behaviour_timestamps.")
            continue
        # Get the timestamps for the current behaviour
        timestamps = timestamps_row[0]
        # Iterate through each timestamp pair
        for start, end in timestamps:
            # Find the indices in time_bin_centers that fall within this range
            start_idx = np.searchsorted(time_bin_centers, start, side='left')
            end_idx = np.searchsorted(time_bin_centers, end, side='right')
            # check is the range has been assigned to another behaviour
            if np.any(labels[start_idx:end_idx] != none_label):
                print(f"Warning: Overlapping behaviour detected for '{behaviour}' between {start} and {end}. Previous behaviour will be overwritten.")
            # Assign the index to the corresponding time bins
            labels[start_idx:end_idx] = idx
    # Create an array of behaviour names matching the label indices, and add it as attribute
    index_to_behaviour = {v: k for k, v in behaviour_labels_dict.items()}
    labels_named = np.array([
        index_to_behaviour[label] if label in index_to_behaviour else 'none'
        for label in labels], dtype=object)
    array_ret = xr.DataArray(
        labels,
        dims=['time_bins'],
        coords={'time_bins': time_bin_centers},
        attrs={'behaviour_labels_dict': behaviour_labels_dict,
               'labels_named': labels_named,
               'description': 'behaviour labels for each time bin, where -1 indicates no behaviour detected.',
               'units': 'index value'}
    )
    return array_ret

def generate_behaviour_labels_inclusive(behaviour_timestamps: pd.DataFrame,
                                t_start: float, t_stop: float, dt: float,
                                behaviour_labels_dict: dict,*,
                                discount_none=True) -> xr.DataArray:
    """
    'Inclusive' label generator, where an interval is assigned to a behaviour label if the
    majority of the interval duration is covered by the behaviour. If `discount_none` is True (default),
    then the no action is always considered as minority, even when it covers most of the interval.
    (in other words, if a tiny portion of the interval is covered by a behaviour,
    the label will match the behaviour, not 'none', as long as `discount_none` is True.
    If `discount_none` is False, then the label will be 'none' if the behaviour covers less than 
    50% of the interval.

    N.B.The size of the returned xarray is 1-Ntime_bins, where Nt is the number of time bins obtained from
    `np.arange(t_start, t_stop, dt)`.

    Arguments
    ---------
    behaviour_timestamps: pd.DataFrame
        DataFrame generated from `convert_to_behaviour_timestamps`
    t_start: float
        Start time of the interval to consider for labeling.
    t_stop: float
        Stop time of the interval to consider for labeling.
    dt: float
        Time bin size, used to create the time bins for labeling.
    behaviour_labels_dict: dict
        Dictionary mapping behaviour names to indices, e.g. {'pup_grab': 0, 'pup_retrieve': 1, ...}
        The term 'none' is used to indicate the label for bins where no behaviour is detected.
        If 'none' is not present, it will be added with index -1.
    discount_none: bool, optional, default is True
        If True, the 'none' label is always considered as minority, even when it covers most of 
        the interval. If False, the 'none' label is equivalent to any other label.

    Returns
    -------
    xr.DataArray
        An xarray with the following structure:
        - Dimensions: ['time_bin_centers']
            The centers of the time bins obtained from `np.arange(t_start, t_stop, dt)`.
        - Data: 1D array of behaviour indices corresponding to each time bin
        - Attributes:
            - 'behaviour_labels_dict': The original dictionary mapping behaviour names to indices
            - 'labels_named': A an array of strings of the same length as the data, containing
              the behaviour names
    
    Throws
    ------
    Warning if any behaviour in `behaviour_labels_dict` is not found in `behaviour_timestamps`.
    
    """
    # N.B.: Function generated by chatGPT, not well optimized for performance (multiple loops in bin size)
    # but simple enough to do the job.
    # Create time bins
    time_bin_edges = np.arange(t_start, t_stop, dt)
    time_bin_centers = time_bin_edges[:-1] + dt / 2
    T = len(time_bin_centers)
    # Ensure 'none' is in the dict
    if 'none' not in behaviour_labels_dict:
        if -1 in behaviour_labels_dict.values():
            raise ValueError("behaviour labels dictionary already contains -1 as a key. Please remove it, or include a 'none' key.")
        behaviour_labels_dict['none'] = -1
    none_label = behaviour_labels_dict['none']
    # Prepare per-bin coverage for each behaviour
    # For each bin, store a dict of {behaviour: overlap_duration}
    bin_behaviour_overlap = [dict() for _ in range(T)]
    # For each behaviour, fill in the overlap durations
    for behaviour, idx in behaviour_labels_dict.items():
        if behaviour == 'none':
            continue  # We'll handle 'none' at the end
        if behaviour not in behaviour_timestamps['behaviour'].values:
            print(f"Warning: behaviour '{behaviour}' not found in behaviour_timestamps.")
            continue
        # Get the intervals for this behaviour
        intervals = behaviour_timestamps.loc[behaviour_timestamps['behaviour'] == behaviour, 'start_stop_times'].values[0]
        for start, end in intervals:
            # For each bin, compute overlap with this interval
            for i in range(T):
                bin_start = time_bin_edges[i]
                bin_end = time_bin_edges[i+1]
                overlap = max(0, min(end, bin_end) - max(start, bin_start))
                if overlap > 0:
                    bin_behaviour_overlap[i][behaviour] = bin_behaviour_overlap[i].get(behaviour, 0) + overlap
    # Assign labels to bins
    labels = np.full(T, none_label, dtype=int)
    for i in range(T):
        overlaps = bin_behaviour_overlap[i]
        if not overlaps:
            # No behaviour covers this bin
            labels[i] = none_label
            continue
        # Find the behaviour with the largest overlap
        best_behaviour = None
        best_overlap = 0
        total_overlap = sum(overlaps.values())
        for beh, ov in overlaps.items():
            if ov > best_overlap:
                best_behaviour = beh
                best_overlap = ov
        # If discount_none: any nonzero overlap by a behaviour wins over 'none'
        if discount_none:
            labels[i] = behaviour_labels_dict[best_behaviour]
        else:
            # If the best overlap is less than half the bin, assign 'none'
            if best_overlap < (dt / 2):
                labels[i] = none_label
            else:
                labels[i] = behaviour_labels_dict[best_behaviour]
    # Create an array of behaviour names matching the label indices, and add it as attribute
    index_to_behaviour = {v: k for k, v in behaviour_labels_dict.items()}
    labels_named = np.array([
        index_to_behaviour[label] if label in index_to_behaviour else 'none'
        for label in labels], dtype=object)
    array_ret = xr.DataArray(
        labels,
        dims=['time_bin_centers'],
        coords={'time_bin_centers': time_bin_centers},
        attrs={'behaviour_labels_dict': behaviour_labels_dict,
               'labels_named': labels_named,
               'description': f"Inclusive behaviour labels for each time bin, where {none_label} indicates no behaviour detected.",
               'units': 'index value'}
    )
    return array_ret

def compute_isis(spike_times_list: List[np.ndarray],pad_zero: bool = True) -> List[np.ndarray]:
    """
    Computes inter-spike intervals (ISI) for each neuron in the provided spike times.
    
    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron (assumed to be sorted and in seconds).
    pad_zero : bool, optional, default is True
        If True, pads the spike times with a zero at the beginning to compute first ISI
        If False, it ignores the first spike time, which is not a valid ISI, and returns
        an array that has one less element than the original spike times.

    Returns
    -------
    isi_array : list of np.ndarray
        List containing ISI arrays for each neuron.
    Raises
    ------
    ValueError
        If any spike time is negative, indicating an invalid spike time.
    """
    isi_array = []
    for i in range(len(spike_times_list)):
        # handle cases with too few spikes
        if len(spike_times_list[i]) == 0:
            isi_array.append(np.array([]))
            continue
        if not pad_zero and len(spike_times_list[i]) <= 1:
            isi_array.append(np.array([]))
            continue
        # if pad_zero append a zero at the beginning
        if pad_zero:
            _spike_times = np.append(0, spike_times_list[i])
        else:
            _spike_times = spike_times_list[i]
        if any(_spike_times<0):
            raise ValueError(f"Negative spike times found in neuron {i}.")
        isi_array.append(np.diff(_spike_times))
    return isi_array

def compute_iFRs(spike_times_list: List[np.ndarray],
                 pad_zero: bool = True) -> List[np.ndarray]:
    """
    Computes instantaneous firing rates (iFR) for each neuron based on spike times.
    iFRs are defined as the inverse of the inter-spike interval (ISI).

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron (assumed to be sorted and in seconds).
    pad_zero : bool, optional, default is True
        If True, pads the spike times with a zero at the beginning to compute first ISI
        If False, it ignores the first spike time, which is not a valid ISI,
        and returns an array that has one less element than the original spike times.
        
    Returns
    -------
    iFRs : list of np.ndarray
        List containing instantaneous firing rates for each neuron.
    """
    isis = compute_isis(spike_times_list, pad_zero=pad_zero)
    iFRs = []
    for isi in isis:
        # Handle cases with no spikes or single spike
        if len(isi) == 0:
            iFRs.append(np.array([]))
            continue
        # Compute iFR as the inverse of ISI, avoiding division by zero
        iFR = np.zeros_like(isi, dtype=float)
        iFR[isi > 0] = 1.0 / isi[isi > 0]
        iFRs.append(iFR)
    return iFRs


def filter_train_small_isi_legacy(spike_times_list: List[np.ndarray],
                        isi_threshold: float = (1.0/200.0),
                        pad_zero: bool = False) -> List[np.ndarray]:
    """
    Filters out spike trains with ISI smaller than the specified threshold.
    
    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron.
    isi_threshold : float, optional
        Threshold for ISI filtering (default is 0.001 seconds).
    pad_zero : bool, optional, default is True
        If True, pads the spike times with a zero at the beginning to compute first ISI
        If False, the first ISI will be set to NaN for the first spike time

    Returns
    -------
    filtered_spike_times_list : list of np.ndarray
        List containing filtered spike times for each neuron.
    filtered_isi_list : list of np.ndarray
        List containing ISI arrays for each neuron after filtering.
    """
    isis_all = compute_isis(spike_times_list, pad_zero=pad_zero)
    if len(isis_all) != len(spike_times_list):
        raise ValueError("Mismatch in length of spike times list and ISI list.")
    n_units = len(spike_times_list) 
    filtered_spike_times_list = []
    filtered_isi_list = []
    for i in range(n_units):
        isis = isis_all[i]
        # if empty, append empty arrays
        if (len(isis) == 0):
            filtered_spike_times_list.append(np.array([]))
            filtered_isi_list.append(np.array([]))
            continue
        if pad_zero:
            mask = isis >= isi_threshold
            mask[0] = True  # No point in removing it, since the first ISI is referred to 0.0
        else:
            # now the first ISI is not valid, we fix it so the first spike is always kept
            _isi_check = np.append(1E10, isis[1:])
            mask = _isi_check >= isi_threshold
        # if all good, do nothing
        if np.all(mask):
            filtered_spike_times_list.append(spike_times_list[i])
            filtered_isi_list.append(isis_all[i])
        else: # else apply the mask
            filtered_spike_times_list.append(spike_times_list[i][mask])
            filtered_isi_list.append(isis[mask])
    return filtered_spike_times_list, filtered_isi_list



def filter_onetrain_small_isis(spiketimes: np.ndarray, threshold: float = 0.004) -> np.ndarray:
    """
    Collapse pairs of spikes whose inter‑spike interval (ISI) is below *threshold*.

    Parameters
    ----------
    spiketimes : 1‑D NumPy array, ascending order
        Spike occurrence times (seconds, milliseconds – any consistent unit).
    threshold  : float, default 0.004
        Minimum allowed interval between two retained spikes.

    Returns
    -------
    np.ndarray
        A **new** array in which spikes closer than *threshold* have been
        merged into their neighbours.  The input array is left untouched.
    """
    n = spiketimes.shape[0]
    if n < 2:
        return spiketimes.copy()

    # Initial ISIs
    isis = np.diff(spiketimes)
    to_remove = []          # indices of spikes to drop

    for k in range(len(isis) - 1):
        isi = isis[k]
        if isi < threshold:
            to_remove.append(k + 1)      # delete the *next* spike
            isis[k + 1] += isi           # merge the interval into the next ISI

    # Handle the very last ISI separately
    if isis[-1] < threshold:
        to_remove.append(n - 1)

    # Nothing to drop? return a copy to match “input is unchanged” guarantee
    if not to_remove:
        return spiketimes.copy()

    return np.delete(spiketimes, to_remove)



def filter_train_small_isi(spike_times_list: List[np.ndarray],
                        isi_threshold: float = (1.0/200.0)) -> List[np.ndarray]:
    """
    Filters out spike trains with ISI smaller than the specified threshold.
    Less efficient than the 'legacy' method of simply removing the spikes with ISI smaller than the threshold,
    but more nuanced. Once a small ISI is found, it removes the spike that caused it and recomputes the 
    subsequent ISIs.
    
    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron.
    isi_threshold : float, optional
        Threshold for ISI filtering (default is 0.001 seconds).

    Returns
    -------
    filtered_spike_times_list : list of np.ndarray
        List containing filtered spike times for each neuron.
    """
    filtered_spike_times_list = []
    for _spike_train in spike_times_list:
        # if empty, append empty arrays
        if (len(_spike_train) == 0):
            filtered_spike_times_list.append(np.array([]))
            continue
        # now we filter the spikes
        _filtered_spikes = filter_onetrain_small_isis(_spike_train, threshold=isi_threshold)
        filtered_spike_times_list.append(_filtered_spikes)
    return filtered_spike_times_list





def compute_filtered_spike_xarray(neuron_idxs,
                                spike_times_list: List[np.ndarray],
                                isi_threshold: float = (1.0/200.0),
                                pad_zero: bool = True,
                                ) ->List[xr.DataArray]:
    """
    This function filters out spike trains with ISI smaller than the specified threshold, 
    and then computes ISI and iFRs for the remaining spike times. Returns a list of data arrays, one for each neuron.
    Each data array contains ISI and iFR for that neuron and spike time as coordinate.

    Warning: I recommend using the dataframe version of this function, called `compute_filtered_spike_dataframe`.

    Arguments
    ----------
    neuron_idxs : list or np.ndarray of int
        The neuron indices, in a list or np.array format
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron.
    isi_threshold : float, optional
        Threshold for ISI filtering (default is 0.005 seconds).
    pad_zero : bool, optional, default is True
        If True, pads the spike times with a zero at the beginning to compute first ISI and first iFR.
        If False, the first ISI and iFR will be set to NaN for the first spike time.

    Returns
    -------
    List of xarray.DataArray
        Each DataArray is structured as follows:
        - Data: matrix of spike_times x what_is_computed (isi, iFR)
        - Coordinates:
            - 'spike_time': The spike time 
            - 'wot': Wot is computed (isi, iFR)
        - Attributes:
            - 'neuron_idx': The index of the neuron

    Examples
    --------
    To access the spike time directly for the first neuron:
    >>> filtered_spike_array[0].coords['spike_time'].values
    To access the ISIs:
    >>> filtered_spike_array[0].sel(wot='isi').values
    Let's say I want the ISIs of neuron 123:
    >>> neuron_123 = next(da for da in filtered_spike_array if da.attrs['neuron_idx'] == 123)
    >>> neuron_123.sel(wot='isi').values
    """
    # first, filter the spike times
    filtered_spike_times_list, filtered_isi_list = filter_train_small_isi(spike_times_list,
                                                                        isi_threshold=isi_threshold,
                                                                        pad_zero=pad_zero)
    # then compute iFRs
    filtered_iFRs = compute_iFRs(filtered_spike_times_list, pad_zero=pad_zero)
    # create the xarray DataArray for each neuron
    data_arrays_ret = []
    for i, neuron_idx in enumerate(neuron_idxs):
        _spike_times = filtered_spike_times_list[i]
        # if empty or not pad zero and only one spike, return empty DataArray
        if (len(_spike_times) == 0) or (not pad_zero and len(_spike_times) <= 1):
            _isi_list = np.array([])
            _iFRs = np.array([])
            # create the empty DataArray for this neuron
            _da = xr.DataArray(
                np.empty((0, 2), dtype=float),
                dims=['spike_time', 'wot'],
                coords={'spike_time': np.array([]),
                        'wot': ['isi', 'iFR']},
                attrs={'neuron_idx': neuron_idx})
            data_arrays_ret.append(_da)
            continue
        # here the spike train is not empty, we can compute ISI and iFR
        # if pad_zero is false, must add NaN to the first ISI and iFR to have the same length as spike times
        if not pad_zero:
            _isi_list = np.append(np.nan, filtered_isi_list[i])
            _iFRs = np.append(np.nan, filtered_iFRs[i])
        else:
            _isi_list = filtered_isi_list[i]
            _iFRs = filtered_iFRs[i]
        # check that sizes match
        _n_spi= len(_spike_times)
        if _n_spi != len(_isi_list) or len(_spike_times) != len(_iFRs):
            raise ValueError(f"Mismatch in lengths for neuron {neuron_idx}: "
                             f"spike_times ({len(_spike_times)}), ISI ({len(_isi_list)}), iFRs ({len(_iFRs)})")
        # create the DataArray for this neuron
        _data = np.column_stack((_isi_list, _iFRs))
        _da = xr.DataArray(
            _data,
            dims=['spike_time', 'wot'],
            coords={'spike_time': _spike_times,
                    'wot': ['isi', 'iFR']},
            attrs={'neuron_idx': neuron_idx})
        data_arrays_ret.append(_da)
    return data_arrays_ret

    

def compute_filtered_spike_dataframe(neuron_idxs,
                                spike_times_list: List[np.ndarray],
                                isi_threshold: float = (1.0/200.0),
                                pad_zero: bool = True,
                                ) ->List[xr.DataArray]:
    """
    This function filters out spike trains with ISI smaller than the specified threshold, 
    and then computes ISI and iFRs for the remaining spike times. Returns a dataframe.
    Each row contains spike times, ISIs and IFRs for that neuron.

    Arguments
    ----------
    neuron_idxs : list or np.ndarray of int
        The neuron indices, in a list or np.array format
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron.
    isi_threshold : float, optional
        Threshold for ISI filtering (default is 0.005 seconds).
    pad_zero : bool, optional, default is True
        If True, pads the spike times with a zero at the beginning to compute first ISI and first iFR.
        If False, the first ISI and iFR will be set to NaN for the first spike time.

    Returns
    -------
    pd.DataFrame : Pandas DataFrame with the following columns:
        - 'unit': The index of the neuron
        - 'spike_time': np.Array of spike times for that neuron
        - 'isi': np.Array of ISI for that neuron
        - 'iFR': np.Array of iFR for that neuron
    

    Examples
    --------
    To access the spike time directly for the first neuron:
    >>> spike_times_df.iloc[0]['spike_time'].values
    To access the ISIs:
    >>> spike_times_df.iloc[0]['isi'].values
    Let's say I want the ISIs of neuron 123:
    >>> spike_times_df[spike_times_df['unit'] == 123]['isi'].values
    """
    # first, filter the spike times
    filtered_spike_times_list, filtered_isi_list = filter_train_small_isi(spike_times_list,
                                                                        isi_threshold=isi_threshold,
                                                                        pad_zero=pad_zero)
    # then compute iFRs
    filtered_iFRs = compute_iFRs(filtered_spike_times_list, pad_zero=pad_zero)
    # build a list of dicts, one per neuron
    rows_df_ret = []
    for i, neuron_idx in enumerate(neuron_idxs):
        _spike_times = filtered_spike_times_list[i]
        if (len(_spike_times) == 0) or (not pad_zero and len(_spike_times) <= 1):
            _isi_list = np.array([])
            _iFRs = np.array([])
            rows_df_ret.append({
                'unit': neuron_idx,
                'spike_time': _spike_times,
                'isi': _isi_list,
                'iFR': _iFRs
            })
            continue
        if not pad_zero:
            _isi_list = np.append(np.nan, filtered_isi_list[i])
            _iFRs = np.append(np.nan, filtered_iFRs[i])
        else:
            _isi_list = filtered_isi_list[i]
            _iFRs = filtered_iFRs[i]
        _n_spi = len(_spike_times)
        if _n_spi != len(_isi_list) or len(_spike_times) != len(_iFRs):
            raise ValueError(f"Mismatch in lengths for neuron {neuron_idx}: "
                             f"spike_times ({len(_spike_times)}), ISI ({len(_isi_list)}), iFRs ({len(_iFRs)})")
        rows_df_ret.append({
            'unit': neuron_idx,
            'spike_time': _spike_times,
            'isi': _isi_list,
            'iFR': _iFRs
        })
    df_ret = pd.DataFrame(rows_df_ret, columns=['unit', 'spike_time', 'isi', 'iFR'])
    df_ret.sort_values(by='unit', inplace=True)
    return df_ret
    
    
def generate_lag_dimensions_expansion(array: np.ndarray, lag: int) -> np.ndarray:
    """
    Expands the dimensions of samples in an array by stacking lagged 
    versions of the array along the feature axis.

    Parameters
    ----------
    array : np.ndarray
        1D or 2D array to be expanded, in format (n_samples, n_features).
    lag : int
        The number of lagged samples to include. If lag=0, the function simply
        returns a copy the original array (in float type), if lag=1, it expands 
        the length of the  feature dimension by n_features, doubling it, so that 
        the samples are [x(t), x(t-1)]. In general, samples are [x(t), x(t-1), ..., x(t-lag)].

    Returns
    -------
    np.ndarray
        A 2D array with shape (n_samples, n_features*(lag+1)), where each sample 
        is expanded to include its lagged versions.
        Warning: the first `lag` samples will be entirely NaN, as there is no
        sufficient data in the past to fill them completely.
        Therefore if you have a label vector y, you should pad it NaN accordingly,
        as in y[lag:] = NaN to match the gap in the features.
        
    Raises
    ------
    ValueError
        If the lag is larger than the number of samples in the input array,
        as it is not possible to compute the requested lag.
    """
    array = np.asarray(array, dtype=float)
    if lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag}.")
    if lag == 0:
        # if lag is 0, return a copy of the original array
        return array.copy()
    if array.ndim == 1:
        array = array[:, None]
    n_samples, n_features = array.shape
    # need a float output array to have NaNs
    out = np.full((n_samples, n_features * (lag + 1)), np.nan, dtype=float)
    for l in range(lag + 1):
        # For each lag, fill the appropriate columns
        start_row = l
        end_row = n_samples
        # If the lag is too large for the number of samples, raise an error.
        if end_row - start_row <= 0:
            raise ValueError(f"lag ({lag}) is too large for the number of samples ({n_samples}) in the input array.")
        out[start_row:end_row, n_features * l:n_features * (l + 1)] = array[0:n_samples - l, :]
    # fill with nan the first `lag` rows
    out[:lag, :] = np.nan
    return out


def generate_lag_dimensions_expansion_xr(x: xr.DataArray, lag: int) -> xr.DataArray:
    """
    Xarray wrapper around `generate_lag_dimensions_expansion`.

    - Accepts a 1D or 2D DataArray (samples[, features]).
    - Expands features by stacking lagged copies along the feature axis.
    - Repeats the feature coordinates (the part expanded by stacking).
    - Keeps the sample dimension name and coordinates unchanged.
    - First `lag` rows are NaN, matching NumPy implementation.

    Parameters
    ----------
    x : xr.DataArray
        1D or 2D DataArray. If 1D, a new feature dimension named 'feature' is created.
    lag : int
        Number of lags to include (>=0).

    Returns
    -------
    xr.DataArray
        DataArray with shape (n_samples, n_features*(lag+1)).
    """
    if not isinstance(x, xr.DataArray):
        raise TypeError("x must be an xarray.DataArray")
    if lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag}.")
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got {x.ndim}D.")

    # Determine dims and base feature coordinates
    if x.ndim == 1:
        sample_dim = x.dims[0]
        feature_dim = 'feature'
        base_feature_coords = np.array(
            [x.name if x.name is not None else 'feature0'],
            dtype=object
        )
        arr = x.values[:, None]  # (n_samples, 1)
    else:
        sample_dim, feature_dim = x.dims
        if feature_dim in x.coords:
            base_feature_coords = np.asarray(x.coords[feature_dim].values)
        else:
            base_feature_coords = np.arange(x.sizes[feature_dim])
        arr = x.values  # (n_samples, n_features)

    # Use the NumPy implementation for the heavy lifting
    out_np = generate_lag_dimensions_expansion(arr, lag)

    # Repeat feature coordinates for each lag block
    repeated_feature_coords = np.tile(base_feature_coords, lag + 1)

    # Build coords, preserving sample coordinates
    coords_out = {}
    if sample_dim in x.coords:
        coords_out[sample_dim] = x.coords[sample_dim].values
    else:
        coords_out[sample_dim] = np.arange(out_np.shape[0])
    coords_out[feature_dim] = repeated_feature_coords

    out = xr.DataArray(
        out_np,
        dims=(sample_dim, feature_dim),
        coords=coords_out,
        name=x.name
    )
    out.attrs = dict(getattr(x, "attrs", {}))
    out.attrs.update({
        "lag": lag,
        "description": "Lag-expanded features along feature dim; first `lag` rows are NaN."
    })
    return out

def generate_random_intervals_within_time(
    t_total: float,
    k_intervals: int,
    t_interval_duration: float,
) -> List[Tuple[float, float]]:
    """
    Picks k randomly arranged, non-overlapping intervals of duration t_interval_duration
    within [0, t_total]. Raises an error if there isn't enough space.

    This implementation samples *uniformly* over all feasible placements (w.r.t. Lebesgue
    measure) by distributing the available slack S = t_total - k_intervals * t_interval_duration
    across the (k_intervals + 1) gaps. Concretely, it uses the order-statistics trick:
      - draw k i.i.d. U(0, S) variables and sort them to y_0 <= ... <= y_{k-1}
      - set start_i = y_i + i * t_interval_duration
      - intervals are [start_i, start_i + t_interval_duration]

    Parameters
    ----------
    t_total : float
        Total time range available, starting from 0.
    k_intervals : int
        Number of intervals to pick (>= 0).
    t_interval_duration : float
        Duration of each interval (>= 0).

    Returns
    -------
    list[tuple[float, float]]
        A list of non-overlapping intervals between 0 and t_total, sorted by start time.

    Raises
    ------
    ValueError
        If inputs are invalid or there isn't enough room (k_intervals * t_interval_duration > t_total).

    Notes
    -----
    • For reproducible results, call `random.seed(some_int)` before invoking this function.
    • If the slack S is (numerically) zero, the intervals are packed back-to-back starting at 0.
    """
    # Basic validation and normalization
    total = float(t_total)
    d = float(t_interval_duration)
    try:
        k = int(k_intervals)
    except Exception as e:
        raise ValueError("k_intervals must be an integer") from e

    if k < 0:
        raise ValueError("k_intervals must be >= 0")
    if total < 0 or d < 0:
        raise ValueError("t_total and t_interval_duration must be >= 0")

    tol = 1e-12
    required = k * d
    if required > total + tol:
        raise ValueError(
            f"Not enough space: need {required} but only have {total}."
        )
    if k == 0:
        return []

    # Available slack to distribute among (k + 1) gaps
    slack = total - required

    # If there's (numerically) no slack, pack intervals back-to-back
    if slack <= tol:
        return [(i * d, (i + 1) * d) for i in range(k)]

    # Order-statistics method: sample k points uniformly over [0, slack] and sort
    ys = sorted(slack * random.random() for _ in range(k))

    intervals: List[Tuple[float, float]] = []
    for i, y in enumerate(ys):
        start = y + i * d
        end = start + d
        intervals.append((start, end))

    # Numerical guard (very conservative): ensure the last end isn't beyond t_total due to FP roundoff
    # This should never trigger meaningfully; keep it as a tiny clamp.
    if intervals[-1][1] > total and intervals[-1][1] - total < 1e-10:
        last = list(intervals[-1])
        last[1] = total
        intervals[-1] = (last[0], last[1])

    return intervals