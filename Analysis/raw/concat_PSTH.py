# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:43:21 2025

@author: su-weisss
"""

"""
Created by Tom Kern (modified)
Last modified 05/08/24

Makes a PSTH plot per neuron, with one subplot for each behavior in target_bs.
In addition to plotting, the result of pf.psth (a multidimensional array) is saved per neuron,
organized into a dictionary with keys corresponding to behaviors.
At the end, data is concatenated across sessions (note that not all cells or behaviors
are present in each session).
"""
import IPython
import polars as pl
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import matplotlib; matplotlib.use('Agg')  # Use a non-interactive backend
import helperFunctions as hf
import os
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
#from time import time
#import math
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
import math
#proprietery functions
import preprocessFunctions as pp
import plottingFunctions as pf
import matplotlib.pyplot as plt


plt.ion()
plt.style.use('default')
window = 5           # s; how much time before and after event start should be plotted
density_bins = 0.1   # s; over what time should average activity be computed in axs[1]



# input Parameters
animal = 'afm16924'
sessions = ['240522','240524','240525','240526']  # can be a list if you have multiple sessions
#sessions = ['240525']
target_regions = ['DPAG','VPAG','VLPAG','LPAG','DLPAG','DMPAG','VMPAG']  # From which regions should the neurons be? Leave empty for all.
PAG_flag=True
target_bs_0 = []           # Which behaviors to plot? Leave empty for all.
savepath_0 = Path(rf"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\20250408\{animal}\concat_PSTHs")
#savepath_0 = Path(rf"E:\runfolder\PSTHs\{animal}")

# Global dictionary to store PSTH data for all sessions.
overall_psth = {}
# Determine n_jobs as half the available CPU cores plus a small overhead.
n_jobs = max(1, multiprocessing.cpu_count() // 2) + 2





    
def generate_shelter_distance_threshold_crossings(distance_to_shelter, time_ax, Threshold=5,diff_time_s=1):
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
    if Threshold is None:
        Threshold=5
    if diff_time_s is None:
        diff_time_s=1


    # Iterate over each sample in distance_to_shelter with its corresponding time index.
    for s, t in zip(distance_to_shelter, time_ax):
        # Detect upward crossing: distance_to_shelter goes from below to above or equal to threshold.
        if not in_segment and s < Threshold:
            in_segment = True
            start_time = t
        # Detect downward crossing: distance_to_shelter falls below threshold when already in a segment.
        elif in_segment and s >= Threshold:
            crossings.append([start_time, t])
            in_segment = False
    
    # remove events, less then diff_time_s time apart   
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        # Always include if no interval has been previously kept.
        if last_end is None or start - last_end >= 5:
            filtered.append(interval)
            last_end = end

    return filtered

    
def generate_speed_threshold_crossings(speed, time_ax, Threshold=20,diff_time_s=5):
    """
    Identify segments where speed rises above a threshold and then falls back below it.

    Parameters:
        speed (array-like): Array of speed values.
        time_ax (array-like): Array of corresponding time values.
        Threshold (float): The speed threshold for detecting crossings.

    Returns:
        list: A list of [start, stop] pairs, where 'start' is the time when 
              speed first crosses above the threshold and 'stop' is the time when
              speed falls back below the threshold.
    """
    crossings = []
    in_segment = False
    start_time = None
    if Threshold is None:
        Threshold=20
    if diff_time_s is None:
        diff_time_s=5

    # Iterate over each sample in speed with its corresponding time index.
    for s, t in zip(speed, time_ax):
        # Detect upward crossing: speed goes from below to above or equal to threshold.
        if not in_segment and s >= Threshold:
            in_segment = True
            start_time = t
        # Detect downward crossing: speed falls below threshold when already in a segment.
        elif in_segment and s < Threshold:
            crossings.append([start_time, t])
            in_segment = False
    
    # remove events, less then diff_time_s time apart   
    filtered = []
    last_end = None

    for interval in crossings:
        start, end = interval
        # Always include if no interval has been previously kept.
        if last_end is None or start - last_end >= 5:
            filtered.append(interval)
            last_end = end

    return filtered

def generate_random_times(N, num_points=10, min_gap=10,):
    """Generate sorted random times between 0 and N with a minimum gap between them."""
    while True:
        times = np.sort(np.random.uniform(0, N, num_points))
        if np.all(np.diff(times) >= min_gap):
            return times

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
    
    # Generate N random time points with at least 10 seconds between each.
    if event_type=='baseline_random':    # generate N random times from baseline    
        n_trials =      kwargs.get('n_trials',10)
        baseline_time_s = kwargs.get('baseline_time_s',7*60)
        
        N=np.min([behaviour.iloc[0,3],baseline_time_s])
        time_points = generate_random_times(N, num_points=n_trials, min_gap=10)
        start_stop=['POINT'] * n_trials
        frames=time_points*framerate
        frames=frames.astype(int)
        # Create a new DataFrame with the 10 additional rows.
        new_rows = pd.DataFrame({
            'behaviours': ['random_baseline'] * n_trials,
            'behavioural_category': ['RAND_BASELINE_POINT'] * n_trials,
            'start_stop': start_stop,
             'frames_s' : time_points,
             'frames' :frames,
            'video_start_s': [behaviour.iloc[0,5]] * n_trials,
            'video_end_s': [behaviour.iloc[0,6]] * n_trials  # All rows have the same video_end_s as the first row
        })
    
    elif event_type=='distance_to_shelter' or  event_type=='speed':
        # Extract extra parameters from kwargs.
        speed = kwargs.get('speed', None)
        distance_to_shelter = kwargs.get('distance_to_shelter', None)
        time_ax = kwargs.get('time_ax', None)
        Threshold = kwargs.get('Threshold', None)
        diff_time_s = kwargs.get('diff_time_s', None)

        # Determine which threshold function to call.
        #generate
        if event_type=='distance_to_shelter':
            if event_type == 'distance_to_shelter':
                if distance_to_shelter is None or time_ax is None:
                    raise ValueError("Both 'distance_to_shelter' and 'time_ax' must be provided for distance_to_shelter events.")
                time_pairs = generate_shelter_distance_threshold_crossings(distance_to_shelter, time_ax,
                                                                           Threshold=Threshold,
                                                                           diff_time_s=diff_time_s)
        elif event_type=='speed':    
            if speed is None or time_ax is None:
                raise ValueError("Both 'speed' and 'time_ax' must be provided for speed events.")
            time_pairs = generate_speed_threshold_crossings(speed, time_ax,
                                                            Threshold=Threshold,
                                                            diff_time_s=diff_time_s)
    
            
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
    print(behaviour.tail(15))
    
    return behaviour

def insert_random_times_to_behavior_df(behaviour,baseline_time_s=7*60,framerate=50,n_trials=10):
    N=np.min([behaviour.iloc[0,3],baseline_time_s])
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

#%% Define the function that both plots and returns PSTH data for one cell                    


def duplicate_with_replacement(df, original_value, replacement_value):
    """
    Finds rows in df where the 'behaviours' column matches original_value,
    duplicates them with the 'behaviours' column replaced by replacement_value,
    and returns a new DataFrame with the additional rows appended.
    
    Parameters:
      df (pd.DataFrame): The input DataFrame.
      original_value (str): The value to look for in the 'behaviours' column.
      replacement_value (str): The value to replace with in the duplicated rows.
      
    Returns:
      pd.DataFrame: A new DataFrame with the duplicated rows appended.
    """
    # Filter rows where the 'behaviours' column equals original_value
    to_duplicate = df[df['behaviours'] == original_value].copy()
    
    # Replace the value in the 'behaviours' column with replacement_value
    to_duplicate['behaviours'] = replacement_value
    
    # Concatenate the original DataFrame with the duplicated one
    df_new = pd.concat([df, to_duplicate], ignore_index=True)
    
    return df_new

def create_figure_of_multiple_PSTHs(i_neuron, region, channel, cluster,neurondata):
    
    new_behaviour =behaviour.copy() #repilicate the behaviors dataframe
    #generate random data trials from baseline period
    point_behaviors = np.unique(new_behaviour.loc[new_behaviour['start_stop'] == 'POINT', 'behaviours'])
    non_point_behaviors = np.unique(new_behaviour.loc[new_behaviour['start_stop'] != 'POINT', 'behaviours'])
    total_num_subplots=len(point_behaviors)+2*len(non_point_behaviors)
    
    # Create END aligned entries only for behaviors NOT in point_behaviors    
    target_bs_aligned = np.array([
        f"{item}_end" for item in target_bs if item not in point_behaviors
    ])
    #add these to the behaviours dataframe    
    for item in target_bs:
        if item not in point_behaviors:
            new_behaviour = duplicate_with_replacement(
                df=new_behaviour,
                original_value=item,
                replacement_value=f"{item}_end"
            )
    
    # Concatenate the original target_bs with the new aligned entries
    new_target_bs = np.concatenate((target_bs, target_bs_aligned))
    
    # Set figsize depending on the number of subplots
    figlength = int(6*total_num_subplots/2)#int(4* len(new_target_bs)-len(target_bs)/2)#int(6 * len(new_target_bs) / 2)
    figheight = int(8*total_num_subplots/2)#int(3* len(new_target_bs)-len(target_bs)/2)#int(8 * len(new_target_bs) / 2)
        
    # Create a figure with subplots for all behaviors using gridspec from pf.subplots
    rows, cols, gs, fig = pf.subplots(len(new_target_bs), gridspec=True, figsize=(figlength, figheight))
    
    fig.suptitle(f'{region}, neuron: {cluster}\nBaseline: {base_mean[i_neuron]}Hz site: {channel}', 
                 fontsize=72, fontname='Arial')
    gs.update(hspace=2)  # Adjust vertical space between plots
    plt.rcParams.update({
    'font.size': 12,            # controls default text sizes
    'axes.titlesize': 18,       # fontsize of the axes title
    'axes.labelsize': 12,       # fontsize of the x and y labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 18      # fontsize of the figure title
    })    
    labelfontsize=24
    ticklabelsize=18
    psth_cell = {}  # Dictionary to store PSTH (as dicts) per behavior for this cell
    ys = []
    Hz_axs = []
    
    
    # Loop through each behavior
    align_to_end_lags = np.array(['_end' in item for item in new_target_bs])

    for i, b_name in enumerate(new_target_bs):

        # Create a sub-gridspec for velocity, average firing rate and trial firing plots (3 subplots)
        gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
        axs = [plt.subplot(gs_sub[j]) for j in range(3)]
        
                                   
        # Get behavior frames
        
        start_stop = hf.start_stop_array(new_behaviour, b_name) 
        
        
        if len(start_stop) == 0:
            print(f'Behavior {b_name} not present for unit {cluster}, skipping.')
            psth_cell[b_name] = None
            continue                         

        # Compute PSTH and capture the computed array.
        # (Assumes pf.psth has been modified to return a Polars DataFrame containing the computed PSTH data.)
       
        
        #if start_stop.ndim==2:
            
        #    align_to_end_flags = np.append(align_to_end_flags, True)
        if align_to_end_lags[i]==True:
            
            align_to_end=True
        else:
            align_to_end=False
      #  for i, align_to_end in enumerate(align_to_end_flags):
       #     if align_to_end==True:
        #        b_name=b_name + '_end' 
        
        psth_data = pf.psth_cond(neurondata, 
                            n_time_index, 
                            start_stop, 
                            velocity,
                            frame_index_s,
                            axs, 
                            window=window, 
                            density_bins=density_bins,
                            return_data=True,
                            align_to_end=align_to_end)
        
        # Add behavior column and rearrange order so that it comes first
        
        psth_data = psth_data.with_columns([pl.lit(b_name).alias("behavior")])
        cols_temp = psth_data.columns
        new_order = ["behavior"] + [col for col in cols_temp if col != "behavior"]
        psth_data = psth_data.select(new_order)
      
        
        # Store the result. Before returning, convert the Polars DataFrame to a plain dictionary
        # to ensure serialization when returning from parallel jobs.
        psth_cell[b_name] = psth_data.to_dict(as_series=False)
        
        # Set axis labels and title
        axs[0].set_title(b_name, fontsize=40, fontname='Arial')
        if i in [0, 3]:
            axs[0].set_ylabel('Velocity [cm/s]',fontsize=labelfontsize)            
            axs[0].tick_params(axis='x', labelsize=ticklabelsize)
            axs[0].tick_params(axis='y', labelsize=ticklabelsize)
            
            axs[1].set_ylabel('avg firing [Hz]',fontsize=labelfontsize)
            axs[1].tick_params(axis='x', labelsize=ticklabelsize)
            axs[1].tick_params(axis='y', labelsize=ticklabelsize)
            
            axs[2].set_ylabel('trials',fontsize=labelfontsize)
            axs[2].tick_params(axis='x', labelsize=ticklabelsize)
            axs[2].tick_params(axis='y', labelsize=ticklabelsize)
        axs[2].set_xlabel('time [s]',fontsize=labelfontsize)
        
      
        
        
        
        
        # Collect y_max for uniform scaling of avg firing rate plots
        ys.append(axs[1].get_ylim()[1])
        Hz_axs.append(axs[1])
      
    max_y = np.max(ys) if ys else None
    if max_y is not None:
        for ax in Hz_axs:
            ax.set_ylim((0, max_y))
      
    # Save the figure to disk for this neuron
    #fig.tight_layout(rect=[0, 0.03, 1, 0.96],pad=1)
    
    save_fname = session_savepath / f"{session}_cell{cluster}_ch{channel}_{region}.png"
    plt.savefig(save_fname.as_posix())
    
    save_fname = session_savepath / f"{session}_cell{cluster}_ch{channel}_{region}.svg"
    plt.savefig(save_fname.as_posix())
    print(f'saved to {save_fname}')
    IPython.embed()
    plt.close()
    return psth_cell#,psth_cell_end
    
    ### TO DO : PSTH to end of event###########################
    
   #  # only inf not poinst event
   #  shape=start_stop.shape
   
   #  if len(shape) == 1:
   #      return psth_cell, None
    
   #  psth_cell_end= {}  # Dictionary to store PSTH (as dicts) per behavior for this cell
   #  ys_end = []
   #  Hz_axs_end = []
    
   #  # Loop through each behavior
   #  for i, b_name in enumerate(target_bs):
   #      # Create a sub-gridspec for velocity, average firing rate and trial firing plots (3 subplots)
   #      gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
   #      axs = [plt.subplot(gs_sub[j]) for j in range(3)]
                              
   #      # Get behavior frames
   #      start_stop = hf.start_stop_array(behaviour, b_name)  
   #      if len(start_stop) == 0:
   #          print(f'Behavior {b_name} not present for unit {cluster}, skipping.')
   #          psth_cell_end[b_name] = None
   #          continue                         

   #      # Compute PSTH and capture the computed array.
   #      # (Assumes pf.psth has been modified to return a Polars DataFrame containing the computed PSTH data.)
   #      end_psth_data = pf.psth_end_of_event(neurondata, 
   #                          n_time_index, 
   #                          start_stop, 
   #                          velocity,
   #                          frame_index_s,
   #                          axs, 
   #                          window=window, 
   #                          density_bins=density_bins,
   #                          return_data=True)
   #      # Add behavior column and rearrange order so that it comes first
       
   #      end_psth_data = end_psth_data.with_columns([pl.lit(b_name).alias("behavior")])
   #      cols_temp = end_psth_data.columns
   #      new_order = ["behavior"] + [col for col in cols_temp if col != "behavior"]
   #      end_psth_data = end_psth_data.select(new_order)
      
        
   #      # Store the result. Before returning, convert the Polars DataFrame to a plain dictionary
   #      # to ensure serialization when returning from parallel jobs.
   #      psth_cell_end[b_name] = end_psth_data.to_dict(as_series=False)
        
   #      # Set axis labels and title
   #      axs[0].set_title(b_name, fontsize=50, fontname='Arial')
   #      if i in [0, 3]:
   #          axs[0].set_ylabel('Velocity [cm/s]',fontsize=labelfontsize)            
   #          axs[0].tick_params(axis='x', labelsize=ticklabelsize)
   #          axs[0].tick_params(axis='y', labelsize=ticklabelsize)
            
   #          axs[1].set_ylabel('avg firing [Hz]',fontsize=labelfontsize)
   #          axs[1].tick_params(axis='x', labelsize=ticklabelsize)
   #          axs[1].tick_params(axis='y', labelsize=ticklabelsize)
            
   #          axs[2].set_ylabel('trials',fontsize=labelfontsize)
   #          axs[2].tick_params(axis='x', labelsize=ticklabelsize)
   #          axs[2].tick_params(axis='y', labelsize=ticklabelsize)
   #      axs[2].set_xlabel('time [s]',fontsize=labelfontsize)
        
        
        
        
   #      # Collect y_max for uniform scaling of avg firing rate plots
   #      ys.append(axs[1].get_ylim()[1])
   #      Hz_axs.append(axs[1])
      
   #  max_y = np.max(ys) if ys else None
   #  if max_y is not None:
   #      for ax in Hz_axs:
   #          ax.set_ylim((0, max_y))
      
   #  # Save the figure to disk for this neuron
   #  save_fname = session_savepath / f"{session}_cell{cluster}_ch{channel}_{region}_AlignedToEnd.png"
   # # save_fname = session_savepath / f"{session}_cell{cluster}_ch{channel}_{region}_AlignedToEnd..svg"
   #  plt.savefig(save_fname.as_posix())
    
   #  plt.close()
    
      
    # Return the dictionary of PSTH data for this cell
    # return psth_cell#,psth_cell_end

# Helper function for parallel processing of neurons.
def process_neuron(i_neuron, region, channel,n_cluster, neurondata):
    psth_dict = create_figure_of_multiple_PSTHs(i_neuron, region, channel, n_cluster,neurondata)
    return i_neuron, psth_dict



def resort_data_shortlist(sorted_indices):  
    global n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR

    n_region_index = n_region_index[sorted_indices]
    n_cluster_index = n_cluster_index[sorted_indices]
    n_channel_index=n_channel_index[sorted_indices]
    ndata=ndata[sorted_indices,:]        
    n_spike_times = [n_spike_times[i] for i in sorted_indices]
    
    return n_region_index,n_cluster_index,n_channel_index,ndata,n_spike_times
def reduce_dataset():
    global n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR
    if len(target_cells)!=0: # reduce to target cells
      sorted_indices = [index for index, value in enumerate(n_cluster_index) if value in target_cells]
     
      n_region_index,n_cluster_index,n_channel_index,ndata,n_spike_times=resort_data_shortlist(sorted_indices)       
  
    elif len(target_regions) != 0:# #sort by region
       in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
       n_region_index = n_region_index[in_region_index]    
       sorted_indices = np.argsort(n_region_index,axis=0)        
       n_region_index,n_cluster_index,n_channel_index,ndata,n_spike_times=resort_data_shortlist(sorted_indices)
       
    return n_region_index,n_cluster_index,n_channel_index,ndata,n_spike_times

def resort_data(sorted_indices):  
    global n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR

    n_region_index = n_region_index[sorted_indices]
    n_cluster_index = n_cluster_index[sorted_indices]
    n_channel_index=n_channel_index[sorted_indices]
    ndata=ndata[sorted_indices,:]
    n_spike_times = [n_spike_times[i] for i in sorted_indices]
    
    neurons_by_all_spike_times_binary_array=neurons_by_all_spike_times_binary_array[sorted_indices,:]
    firing_rates=firing_rates[sorted_indices,:]   
    
    iFR_array=iFR_array[sorted_indices,:]
    iFR = [iFR[i] for i in sorted_indices]
    
    return n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR



target_regions = ['DPAG','VPAG','VLPAG','LPAG','DLPAG','DMPAG','VMPAG']  # From which regions should the neurons be? Leave empty for all.



target_cells=[344,365,347,391]# escape active
target_behaviors=['startle','turn','escape']
#target_cells=[344,365,391,538,545,547]#pup positive
#target_cells=[347,393,561,582,590]#pup negative
target_cells=[344,347,365,391,538,545,547,347,393,561,582,590]#pup related
target_behaviors=['pup_run','pup_grab','pup_retrieve','pup_drop']
target_cells=[]
target_behaviors=[]
windowsize_s=10 #how long before+ after the loom should the video playreading is about 10 minutes. but going to get 10 gig card
view_window_s=5#with what window should be plotted around the current time?


#%% Loop through sessions
for session in sessions:
    paths=pp.get_paths(animal, session)    
    print(f"session: {session}\n")
    velocity,locations,node_names,bottom_node_names,frame_index_s,frames_dropped,distance2shelter,bottom_distance_to_shelter=hf.load_specific_preprocessed_data (animal, session, 'tracking',load_pd=False )
    distance2shelter=distance2shelter[:,3]
    
    if np.nanmax(distance2shelter)>300:
         values = paths['Cm2Pixel_xy'].split(' ')
         # Convert each value to np.float16
         Cm2Pixel_xy = [np.float32(value) for value in values]
         distance2shelter=distance2shelter*Cm2Pixel_xy[0]
         distance2shelter=distance2shelter-np.nanmin(distance2shelter)#DEBUG ONLY!!!!!!!!!!!!!
         max_dist=np.nanmax(distance2shelter)
         max_vel=np.nanmax(velocity)
    
    target_bs = target_bs_0.copy()
    session_savepath = savepath_0 / session
    os.makedirs(session_savepath, exist_ok=True)
    print(f'loading data')
    # Load data using helperFunctinons
    [frames_dropped, 
     behaviour, 
     ndata, 
     n_spike_times,
     n_time_index, 
     n_cluster_index, 
     n_region_index, 
     n_channel_index,
     velocity, 
     locations, 
     node_names, 
     frame_index_s,
     ] = hf.load_preprocessed(animal,session)
    
      # Adjust data if necessary per session
    if session == '240522':
            velocity = velocity[0:len(frame_index_s)]
            locations = locations[0:len(frame_index_s)]
    
    
    vframerate=len(frame_index_s)/max(frame_index_s)
    
    ################################################################################
    print('inserting random baseline time periods to behavior dataframe \n')
    behaviour=insert_random_times_to_behavior_df(behaviour=behaviour,baseline_time_s=7*60,framerate=vframerate,n_trials=10)   
    ################################################################################
    
    frame_index_s = frame_index_s[:len(velocity)]
    distance2shelter = distance2shelter[:len(velocity)]
    
   
    kwargs={"speed":velocity,
            "distance_to_shelter": distance2shelter,
                "time_ax":frame_index_s,
                    "Threshold":None,
                    "diff_time_s":None,
                    "baseline_time_s":7*60,
                    "n_trials": 10}
    
    behavior_name=None
    behavior_type=None
    framerate=len(frame_index_s)/max(frame_index_s)
    

    print('inserting random baseline time periods to behavior dataframe \n')
    event_type='baseline_random'
    df=insert_event_times_into_behavior_df(behaviour,framerate,event_type,behavior_name,behavior_type,**kwargs)    
    
    print('inserting speed threshold crossing periods to behavior dataframe \n')
    event_type="speed"
    behavior_name="Speed_threshold_crossing"
    behavior_type="Speed_threshold_crossing"
    df=insert_event_times_into_behavior_df(behaviour,framerate,event_type,behavior_name,behavior_type,**kwargs)
    
    print('inserting in-sheltertime periods to behavior dataframe \n')
    event_type= "distance_to_shelter"
    behavior_name="in_shelter"
    behavior_type="IN_SHELTER"
    df=insert_event_times_into_behavior_df(behaviour,framerate,event_type,behavior_name,behavior_type,**kwargs)
    ################################################################################
    
    
    
    
    if len(target_cells)!=0: # reduce to target cells
        sorted_indices = [index for index, value in enumerate(n_cluster_index) if value in target_cells]
        n_region_index = n_region_index[sorted_indices]
        n_cluster_index = n_cluster_index[sorted_indices]
        n_channel_index=n_channel_index[sorted_indices]
        ndata=ndata[sorted_indices,:]
        n_spike_times = [n_spike_times[i] for i in sorted_indices]
    else:
        if len(target_regions) != 0:# #sort by region          
            in_region_index = np.where(np.isin(n_region_index, target_regions))[0]                    
                
            #sorted_indices = np.argsort(in_region_index,axis=0)  
            sorted_indices=in_region_index
            n_region_index =  n_region_index[sorted_indices]
            n_cluster_index = n_cluster_index[sorted_indices]
            n_channel_index=n_channel_index[sorted_indices]
            ndata=ndata[sorted_indices,:]
            n_spike_times = [n_spike_times[i] for i in sorted_indices]
        
   # print(f"{target_cells=}, {target_regions=} \n keeping only targeted cells: {ndata.shape=} \n")    
#    n_region_index,n_cluster_index,n_channel_index,ndata,n_spike_times=reduce_dataset()
#    print(f'after removing units:{ndata.shape=} \n')
    
    
    
    print(f"calculating intananeous firing rate \n" )
    iFR,iFR_array,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate
    
    
    print(f"recalculating ndata \n")
    # spike_res=0.001
    # FR_res=0.1
    # n_time_index, ndata, firing_rate_bins_time,firing_rates,neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds=hf.recalculate_ndata_firing_rates2(n_spike_times,
    # bin_size=spike_res, firing_rate_bin_size=FR_res)
    
    
    spike_res=0.001
    FR_res=0.02
    res=0.001
#    print(f"before: {len(n_spike_times)=}, {np.shape(ndata)=}")
    n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=res)
#    print(f"after: {len(n_spike_times)=}, {np.shape(ndata)=}")
    #n_time_index, ndata, firing_rate_bins_time,firing_rates,neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds=hf.recalculate_ndata_firing_rates2(n_spike_times,
#    bin_size=spike_res, firing_rate_bin_size=FR_res)
   
    
    print(f"1 {np.shape(n_time_index)=}, {np.shape(ndata)=}")
    # if len(target_cells)!=0: # reduce to target cells
    #   sorted_indices = [index for index, value in enumerate(n_cluster_index) if value in target_cells]
     
    #   n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)       
    # else:
    #     if len(target_regions) != 0:# #sort by region
    #         in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
    #         n_region_index = n_region_index[in_region_index]    
    #         sorted_indices = np.argsort(n_region_index,axis=0)        
    #         n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)

 
    print(f"calculating baseline\n")
    vframerate=len(frame_index_s)/max(frame_index_s)
   
    base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
    #    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
    base_mean = np.round(base_mean, 2)
    base_mean = base_mean[:, np.newaxis]  # now shape is (N, 1)
    base_mean = np.where(base_mean == 0, 1/10000, base_mean)
    base_mean = np.squeeze(base_mean)  # Remove the extra dimension, now shape is (N,)        
    windowsize_f=np.round(windowsize_s*vframerate).astype(int)
    view_window_f=np.round(view_window_s*vframerate).astype(int)
    #target node
    node_ind=np.where(node_names=='b_back')[0][0]#node to use for tracking        
      
       #Target neurons
    target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
       
  
        
#######    # Reduce selection to subsets.
    unique_behaviours = behaviour.behaviours.unique()
    if len(target_bs) == 0:  # If no target behavior is specified, take all present
        target_bs = np.sort(unique_behaviours)

    if len(target_regions) != 0:  # If target brain region(s) specified    
        in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
        n_region_index = n_region_index[in_region_index]    
    else:
        n_region_index = np.arange(len(n_region_index))
        
    n_channel_index = n_channel_index[in_region_index]  # Channel data
    ndata = ndata[in_region_index]    # Neural data
    n_cluster_index = n_cluster_index[in_region_index] 
    
    # Create a container (list) to hold PSTH data per neuron in this session.
    total_neurons = np.shape(ndata)[0]    
    values = [None] * total_neurons  
    
##### Get baseline firing rates.
#     base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
# #    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
#     base_mean = np.round(base_mean, 2)
    
    # Add cluster info to a Polars DataFrame.
    if session == sessions[0]:
        Neurons_pl = pl.DataFrame({session: [None] * total_neurons for session in sessions})
        
        new_column_df = pl.DataFrame({"region": n_region_index})
        Neurons_pl = pl.concat([new_column_df, Neurons_pl], how="horizontal")
            
        new_column_df = pl.DataFrame({"max_site": n_channel_index})
        Neurons_pl = pl.concat([new_column_df, Neurons_pl], how="horizontal")
            
        new_column_df = pl.DataFrame({"cluster_id": n_cluster_index})
        Neurons_pl = pl.concat([new_column_df, Neurons_pl], how="horizontal")
    else:
        # Here, Neurons_pl already exists from the previous sessions.
        # First, check if there are cluster ids in n_cluster_index that are missing from Neurons_pl.
        existing_cluster_ids = Neurons_pl["cluster_id"].to_list()
        new_rows = []
        
        for i, cid in enumerate(n_cluster_index):
            if cid not in existing_cluster_ids:
                # Build a dictionary for the new row
                new_row = {
                    "cluster_id": cid,
                    "max_site": n_channel_index[i],
                    "region": n_region_index[i]
                }
                # For all session columns, set a default value of None.
                for sess in sessions:
                    new_row[sess] = None
                new_rows.append(new_row)
        
        # If there are any new rows, add them to Neurons_pl.
        if new_rows:
            new_rows_df = pl.DataFrame(new_rows)
            Neurons_pl = pl.concat([Neurons_pl, new_rows_df], how="vertical")
    results = []
    for i, (reg, ch, nc, nd) in enumerate(tqdm(zip(n_region_index, n_channel_index, n_cluster_index, ndata), total=len(n_region_index))):
        result = process_neuron(i, reg, ch, nc, nd)
        results.append(result)
    # with parallel_backend("loky", timeout=24*3600):  # timeout in seconds
    #  results,results_end = Parallel(n_jobs=n_jobs)(
    #     delayed(process_neuron)(i, reg, ch, nc, nd)
    #     for i, (reg, ch, nc, nd) in enumerate(
    #         tqdm(zip(n_region_index, n_channel_index, n_cluster_index, ndata),
    #              total=len(n_region_index))
    #     )
    # )
    #%% Process neurons in parallel.
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(process_neuron)(i, reg, ch,nc, nd)
    #     for i, (reg, ch,nc, nd) in enumerate(tqdm(zip(n_region_index, n_channel_index,n_cluster_index, ndata), total=len(n_region_index)))
    # )
    print('populating dataframe...\n')
    # Create a mapping from cluster_id to its corresponding PSTH dictionary.
    psth_mapping = {cid: psth for cid, psth in zip(n_cluster_index, results)}

    # Build the new session column values by iterating over each row.
    session_values = []
    for row in Neurons_pl.iter_rows(named=True):
        cid = row["cluster_id"]
        session_values.append(psth_mapping.get(cid, None))

    # Add/update the session column using with_columns.
    Neurons_pl = Neurons_pl.with_columns([
        pl.Series(session, session_values, strict=False)
    ])

    # Print the updated DataFrame.
    print(Neurons_pl)
IPython.embed()



# # Slice the DataFrame to get a single-row DataFrame


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
#savepath_0 = Path(rf"E:\2025\Figures\PSTH\{animal}")
behaviors=[]

def is_list_like(val):
    # Check if value is iterable but not a string/bytes type.
    return isinstance(val, Iterable) and not isinstance(val, (str, bytes))

def convert_value(val):
    """
    Convert a single value. If it's list-like (except strings), convert each element
    (if possible) into an np.array; otherwise, try to convert the value directly.
    """
    if is_list_like(val):
        converted = []
        for item in val:
            # If the item itself is list-like but not a string, we convert recursively.
            if is_list_like(item):
                converted.append(np.array(item))
            else:
                try:
                    converted.append(np.array(item))
                except Exception:
                    converted.append(item)
        return converted
    else:
        try:
            return np.array(val)
        except Exception:
            return val

def process_cell_data(cell_dict):
    """
    Process one cell's dictionary (from a single DataFrame row). For each behavior's b_data,
    it iterates through every key. If the first element indicates a dictionary structure
    (i.e. subkeys are present), then for each subkey, we convert its value (list-like or scalar)
    using convert_value(). If the b_data item is not a dictionary, we process it directly.
    
    The aggregated_behavior structure is:
      aggregated_behavior[behavior][b_key][subkey] ->
          a list with converted values from each session (if b_data had subkeys)
      aggregated_behavior[behavior][b_key] ->
          a converted result (if b_data item was a scalar or similar structure)
    """
    aggregated_behavior = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Skip the first three metadata keys.
    session_keys = list(cell_dict.keys())[3:]
    
    for session in session_keys:
        session_data = cell_dict[session]
        if not session_data[0]:
            continue
        try:
            # Extract nested dictionary that maps behaviors to their data.
            nested_dict = session_data[0][1]
        except (IndexError, TypeError) as err:
            print(f"Skipping session '{session}' due to error: {err}")
            continue

        for behavior, b_data in nested_dict.items():
            for b_key, item in b_data.items():
               
                if not item:
                    continue
                first_item = item[0]
                # If the first_item is a dictionary, assume there are subkeys to process.
                if isinstance(first_item, dict):
                    for subkey, value in first_item.items():
                        try:
                            converted = (value)
                            #converted = convert_value(value)
                        except Exception as err:
                            print(f"Conversion failed for behavior '{behavior}', key '{b_key}', subkey '{subkey}': {err}")
                            continue
                        aggregated_behavior[behavior][b_key][subkey].append(converted)
                else:
                    # Otherwise, process the value directly.
                    try:
                        converted = convert_value(first_item)
                    except Exception as err:
                        print(f"Direct conversion failed for behavior '{behavior}', key '{b_key}': {err}")
                        aggregated_behavior[behavior][b_key] = None
                        continue
                    # Store it directly (this will be overwritten if there are multiple sessions,
                    # so modify this behavior if you want to aggregate across sessions)
                    aggregated_behavior[behavior][b_key] = converted
                    
                    

    return aggregated_behavior

def recursive_defaultdict_to_dict(dd):
    """
    Recursively convert nested defaultdicts to plain dicts.
    """
    if isinstance(dd, defaultdict):
        dd = {k: recursive_defaultdict_to_dict(v) for k, v in dd.items()}
    elif isinstance(dd, dict):
        dd = {k: recursive_defaultdict_to_dict(v) for k, v in dd.items()}
    return dd

# --------- Aggregation Phase (assume Neurons_pl is defined elsewhere) ----------
cell_list = []  # This list will hold the aggregated behavior data for each cell.

for row_idx in range(Neurons_pl.height):
    row_df = Neurons_pl.slice(row_idx, 1)
    cell_dict = row_df.to_dict(as_series=False)
    aggregated_behavior = process_cell_data(cell_dict)
    cell_list.append(recursive_defaultdict_to_dict(aggregated_behavior))


import numpy as np
def concat_psth(precomputed,  window=5, density_bins=0.5):
    """
    Parameters
    ----------
    precomputed : dict
        Dictionary containing precomputed plotting values with the following keys:
           - "all_spikes": list of np.array, each containing spike times (aligned to the event) 
                           for a trial (for the raster plot).
           - "firing_rate_bins": np.array of bin edges for the PSTH.
           - "hz": np.array containing the firing rates computed for each bin.
           - "all_vel_times": list of np.array, time points (aligned) for each trial's velocity.
           - "all_vel": list of np.array, velocity trace for each trial.
           - "velbins": np.array of bin edges for velocity averaging.
           - "avg_velocity": np.array with average velocity per velocity bin.
           - "state_events" (optional): list of dicts, each with keys:
                   "trial": trial number or y-position for the shading,
                   "left": left edge of the state event (in seconds, relative to the event),
                   "width": duration of the state event,
                   "alpha": (optional) transparency (default 0.5)
    axs : list of matplotlib.axes.Axes
        Three axes on which to plot:
           axs[0] - velocity trace,
           axs[1] - average firing (PSTH),
           axs[2] - raster plot of spikes.
    window : float, optional
        Time window (in seconds) before/after the event (default 5).
    density_bins : float, optional
        Bin width (in seconds) for the PSTH (default 0.5).
    """
    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[b_i])
    axs = [plt.subplot(gs_sub[j]) for j in range(3)]
    
    # --- Prettify axes ---
    remove_axes(axs[2])
    remove_axes(axs[0], bottom=True)
    remove_axes(axs[1], bottom=True)
    
    # --- Plot state-event shading (burlywood) if provided ---
    if "state_events" in precomputed:
        for event in precomputed["state_events"]:
            trial = event[0]["trial"]
            left = event[0]["left"]
            width = event[0]["width"]
            alpha = event[0].get("alpha", 0.5)
            axs[2].barh(trial, width, left=left,
                        height=1, color='burlywood', alpha=alpha)
    
    # --- Plot raster ---
    for i, spiketimes in enumerate(precomputed["all_spikes"]):
        axs[2].scatter(spiketimes, np.full(spiketimes.shape, i + 1), 
                       c='teal', s=0.5)
    axs[2].set_ylim((0, len(precomputed["all_spikes"]) + 4))
    axs[2].set_xlim((-window, window))
    axs[2].set_yticks(np.arange(0, len(precomputed["all_spikes"]) + 1, step=10))
    
    # --- Plot average firing rate (PSTH) ---
    all_spikes=precomputed["all_spikes"]
    all_spikes_flat = np.hstack(all_spikes)
    if all_spikes_flat.size == 0:
        print("Warning: No spike data found. Skipping PSTH plotting.")
    else:
        sum_spikes, firing_rate_bins = np.histogram(all_spikes_flat)
        hz = sum_spikes / density_bins
        print("firing_rate_bins shape: {}, hz shape: {}".format(firing_rate_bins[:-1].shape, hz.shape))
        axs[1].bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
    firing_rate_bins = precomputed["firing_rate_bins"]
    hz = precomputed["hz"]
    axs[1].bar(firing_rate_bins[:-1], hz, align='edge', 
               width=density_bins, color='grey')
    axs[1].set_xlim((-window, window))
    axs[1].set_xticks([])
    
    # --- Plot velocity traces ---
    axs[0].set_xlim((-window, window))
    axs[0].set_ylim((0, 130))
    
    for vel, t in zip(precomputed["all_vel"], precomputed["all_vel_times"]):
        
        axs[0].plot(t, vel, lw=0.5, c='grey')
        velbins=precomputed["velbins"][0]
        
    axs[0].plot(precomputed["velbins"][:-1], precomputed["avg_velocity"], 
                c='orangered')
    axs[0].set_xticks([])
    
    # --- Mark event time ---
    for ax in axs:
        ax.axvline(0, linestyle='--', c='k')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95],pad=0.75)
    fig.canvas.draw()  # Force update to the figure    
    # Save the figure to disk for this neuron
    save_fname = savepath_0 / f"test.png"
    plt.savefig(save_fname.as_posix())
    plt.close()

def remove_axes(ax, bottom=False):
    """
    Helper to remove ticks and labels from a matplotlib axis.
    If bottom is True, only the bottom ticks/labels are removed.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if bottom:
        ax.tick_params(labelbottom=False, bottom=False)
    else:
        ax.tick_params(bottom=False, labelbottom=False)
    ax.tick_params(left=False, labelleft=False)


# --------- Plotting Phase ---------
def plot_behavior(cell_id, behavior, spikes_sessions, velocity_sessions,all_start_stop):
    """
    Create one figure per cell/behavior with a raster in the top subplot and 
    velocity in the bottom subplot.
    
    spikes_sessions: expected as a list over sessions (each session is a list over trials)
      with each trial being a numpy array of spike times.
    velocity_sessions: expected as a list of velocity_bin_size values (scalars or 1-element arrays)
    """
   
    # Create a figure with two subplots (vertical layout)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    i = 0
   
    # ----------------- Raster Plot -----------------

    #print(behavior)
    # Determine if the provided start-stop structure represents point versus state events
    for all_start_stops in all_start_stop:
      for startstop in all_start_stops: 
        startstop=np.array(startstop)
        point = False
        if np.shape(startstop) != (2,):
            point = True
        
        # For the first trial, set the previous trial variables.
        if i == 0:
            previous_plotstop = 0
            previous_zero = 0
        
            
        if  point==False:  # State events
           
            plotzero = startstop[0]
            # If events occur too close to the previous event, just add shading.
            if (plotzero - window) < previous_plotstop:
                axs[0].barh(i - 2, startstop[1] - startstop[0], 
                             left=plotzero - previous_zero,
                             height=1, color='burlywood', alpha=.5)
                continue                
            axs[0].barh(i + 1, startstop[1] - startstop[0], height=1, color='burlywood')
            
        else:  # Point events
           
            plotzero = startstop
        i += 1       
        plotstart = plotzero - window
        plotstop = plotzero + window
       
        
    ax_raster = axs[0]
    trial_count = 0
 
    if spikes_sessions is not None:
       
        # sum_spikes, firing_rate_bins = np.histogram(all_spikes, bins)
        # hz = sum_spikes / time_per_bin
        
        for session in spikes_sessions:
            for trial in session:
                # Ensure trial is a numpy array
                trial = np.array(trial)
                # Create y-values, one per trial.
                y = np.full(trial.shape, trial_count)
                ax_raster.scatter(trial, y, marker='|', color='black')
                trial_count += 1
   
    ax_raster.set_title(f"Cell {cell_id} Behavior '{behavior}' - Raster")
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Trial")


    
    # ----------------- Velocity Plot -----------------
    ax_velocity = axs[1]
    velocities = []
    if velocity_sessions is not None:
        # Flatten the list in case there are multiple sessions.
        for session in velocity_sessions:
            # Each session may be a scalar or an array.
            if isinstance(session, (list, np.ndarray)):
                try:
                    velocities.append(np.array(session).item())
                except Exception:
                    velocities.append(session)
            else:
                velocities.append(session)
        ax_velocity.plot(velocities, marker='o', linestyle='-', color='blue')
    ax_velocity.set_title(f"Cell {cell_id} Behavior '{behavior}' - Velocity")
    ax_velocity.set_xlabel("Session index")
    ax_velocity.set_ylabel("Velocity Bin Size")
    
    
    
    
    fig.tight_layout()
    # Save the combined figure.
    save_behavior = behavior.replace('/', '_')
    save_fname = savepath_0 / f"cell{cell_id}_{save_behavior}.png"
    plt.savefig(save_fname.as_posix())
#   plt.close()
  


# Loop through aggregated cells (from cell_list) and plot available data.
for cell_idx, cell in enumerate(cell_list):
    # Set figsize depending on the number of subplots
    figlength = int(8 * len(target_bs) / 2)
    figheight = int(6 * len(target_bs) / 2)
    rows, cols, gs, fig = pf.subplots(len(target_bs), gridspec=True, figsize=(figlength, figheight))
    
    # fig.suptitle(f'{region}, neuron: {cluster}\nBaseline: {base_mean[i_neuron]}Hz site: {channel}', 
    #              fontsize=72, fontname='Arial')
    gs.update(hspace=0.5)  # Adjust vertical space between plots
    b_i=0
    for  behavior, behavior_data in cell.items():
        b_i+=1
        if behavior not in behaviors:
           behaviors.append(behavior)
        # Extract spikes info (raster) if available.
        spikes_info = None
        if 'raster_dict' in behavior_data:
            spikes_info = behavior_data['raster_dict'].get('spikes_array')

        # Extract velocity info if available.
        velocity_info = None
        if 'velocity_dict' in behavior_data:
           
            velocity_info = behavior_data['velocity_dict'].get('Avg_Velocity_cms')[0]
            
        start_stop_info=behavior_data['meta_dict'].get('behavior_start_stop_time_s')
        
        precomputed_info= behavior_data['precomputed']
        
      
        # Only plot if at least one of the two datasets is available.
        if spikes_info is not None or velocity_info is not None:
            concat_psth(precomputed_info,  window=5, density_bins=0.5)
        #    plot_behavior(cell_idx, behavior, spikes_info[0][0], velocity_info,start_stop_info)
            
            
            
            
        




















def process_cell_data2(cell_dict):
    """
    Given a dictionary for one cell's row (converted from a single-row DataFrame),
    this function iterates over session keys (ignoring the first three metadata keys)
    and accumulates spikes arrays and corresponding meta arrays for each behavior.
    Returns two dictionaries: one for raster data and one for start–stop data.
    """
    session_keys = list(cell_dict.keys())[3:]  # first three keys are metadata
    behavior_rasters = {}
    behavior_start_stops = {}
    behavior_Avg_Velocity_cms={}
    
    for session in session_keys:
        session_data = cell_dict[session]
        
        # Skip if there are no spikes for this session.
        if not session_data[0]:
            continue
        
        # Extract the nested dictionary containing behavior data.
        # Here, session_data is assumed to be a list (with one element)
        # that itself is a list or tuple where the second item is the nested dict.
        try:
            nested_dict = session_data[0][1]
        except (IndexError, TypeError) as err:
            print(f"Error extracting nested dict for session '{session}': {err}")
            continue
        
        for behavior, b_data in nested_dict.items():
            # Extract lists under "raster_dict" and "meta_dict"
            raster_list = b_data.get("raster_dict", [])
            meta_list = b_data.get("meta_dict", [])
            velocity_list=b_data.get("velocity_dict", [])
            
            # If either list is missing or empty, skip this behavior.
            if not raster_list or not meta_list:
                continue
            
            # Get the lists stored under the appropriate keys.
            spikes_array_list = raster_list[0].get("spikes_array", [])
            start_stop_list = meta_list[0].get("behavior_start_stop_time_s", [])
            velocity_time_bin=velocity_list[0].get("Velocity_Time_bin")
            velocity_bin_size=velocity_list[0].get("velocity_bin_size")
            Avg_Velocity_cms_list=velocity_list[0].get("Avg_Velocity_cms")
            
            # Convert entries to numpy arrays.
            spikes_arrays = [np.array(s) for s in spikes_array_list]
            start_stop_arrays = [np.array(s) for s in start_stop_list]
            Avg_Velocity_cms_arrays = [np.array(s) for s in Avg_Velocity_cms_list]
            
            # Append data for this behavior.
            if behavior in behavior_rasters:
                behavior_rasters[behavior].append(spikes_arrays)
                behavior_start_stops[behavior].append(start_stop_arrays)
                behavior_Avg_Velocity_cms[behavior].append(Avg_Velocity_cms_arrays)
            else:
                behavior_rasters[behavior] = [spikes_arrays]
                behavior_start_stops[behavior] = [start_stop_arrays]
                behavior_Avg_Velocity_cms[behavior]= Avg_Velocity_cms_arrays
                
    return behavior_rasters, behavior_start_stops,behavior_Avg_Velocity_cms

# Main accumulation over cells
cell_list = []         # to store the raster data for each cell
start_stop_list = []   # to store the meta data for each cell
Avg_Velocity_cms_list=[]
# Iterate over rows by index.
for row_idx in range(Neurons_pl.height):
    # Get a single-row DataFrame slice.
    row_df = Neurons_pl.slice(row_idx, 1)
    # Convert the row to a dictionary.
    cell_dict = row_df.to_dict(as_series=False)
    # Process the row to get both dictionaries.
    rasters, meta ,Avg_Velocity_cms= process_cell_data2(cell_dict)
    
    cell_list.append(rasters)
    start_stop_list.append(meta)
    Avg_Velocity_cms_list.append(Avg_Velocity_cms)

out_dict={
    'meta':meta,
    'cell_list':cell_list,
    'start_stop_list':start_stop_list,
    'Avg_Velocity_cms_list':Avg_Velocity_cms_list   
    }
np.save( f"{savepath_0}\concat.npy",out_dict)









row_count = Neurons_pl.height #Neurons_pl is a polars dataframe
cell_list= [None] *row_count
start_stop_list= [None] *row_count
for row_i in range(row_count):#loop through cells
    row_df = Neurons_pl.slice(row_i, 1) # get the current row of Neurons_pl    
    row_dict = row_df.to_dict(as_series=False) # Convert that row to a dictionary, which will return columns as keys.  
    keys_list = list(row_dict.keys())    
    # Keep keys from index 3 to the end, as the first 3  are metadata
    desired_session_keys = keys_list[3:]    
    #print(desired_session_keys)
    concatenated_rasters = {} # Initialize the dictionary to accumulate the spikes arrays for each behavior.
    behaviors=[] # initialize the general behaviors list
    concatenated_start_stops={}
    # Loop across all desired keys.
    for session in desired_session_keys:
        if not (row_dict[session][0]): #no spikes for this cell in this session        
            continue#skip            
        current_entry = row_dict[session]
        # current_entry is a list that contains a dictionary; extract the nested dictionary.
        entry_dict = current_entry[0]
        entry_dict = entry_dict[1]  # Now entry_dict holds your behavior keys        
        # Process each behavior
        for behavior, behavior_data in entry_dict.items():            
            if behavior not in behaviors:
                behaviors.append(behavior)
            # Extract the list stored under "raster_dict"
            raster_dict_list = behavior_data.get("raster_dict", [])
            start_stop_dict =  behavior_data.get("meta_dict", [])
            # Create a list to store the spikes_array for this behavior
            spikes_arrays = []            
            start_stop_arrays=[]
            for spikes_array, start_stop_array in zip(raster_dict_list[0]['spikes_array'],
                                                      start_stop_dict[0]['behavior_start_stop_time_s']):
                spikes_array = np.array(spikes_array)
                spikes_arrays.append(spikes_array)

                start_stop_array = np.array(start_stop_array)
                start_stop_arrays.append(start_stop_array)                   
            # Update the concatenated_rasters dictionary.
            # If the behavior already exists from a previous key, extend its list.
            if behavior in concatenated_rasters:
                concatenated_rasters[behavior].append(spikes_arrays)
                concatenated_start_stops[behavior].append(start_stop_arrays)
            else:            
                concatenated_rasters[behavior] = spikes_arrays
                concatenated_start_stops[behavior]=start_stop_arrays
                
    cell_list[row_i]=concatenated_rasters
    start_stop_list[row_i]=concatenated_start_stops
    
    
            
            


# # Create a new polars dataframe with only the first 3 columns
# cell_behavior_rasters_df = Neurons_pl.select(["cluster_id", "max_site", "region"],)
# new_columns = [pl.lit(None).alias(behavior) for behavior in behaviors]
# cell_behavior_rasters_df = cell_behavior_rasters_df.with_columns(new_columns)

# #
# # Populate behavior columns per cell
# for row_i in range(row_count):
#     values = cell_list[row_i]
#     for behavior in behaviors:
#         behavior_value = values[behavior]

#     # Use a when/then/otherwise expression to replace the value at row 9.
#     cell_behavior_rasters_df = cell_behavior_rasters_df.with_column(
#         pl.when(row_i)
#           .then(behavior_value)
#           .otherwise(pl.col(behavior))
#           .alias(behavior)
#     )

# updated_behavior_columns = {}


# Create the base DataFrame and add empty behavior columns.
cell_behavior_rasters_df = Neurons_pl.select(["cluster_id", "max_site", "region"])
new_columns = [pl.lit(None).alias(behavior) for behavior in behaviors]
cell_behavior_rasters_df = cell_behavior_rasters_df.with_columns(new_columns)

# Build new full columns for each behavior using the cell_list.
updated_columns = {}
for behavior in behaviors:
    col_vals = []
    for row_i in range(cell_behavior_rasters_df.height):
        # Retrieve the value; default to an empty list if key is missing.
        behavior_value = cell_list[row_i].get(behavior, [])
        col_vals.append(behavior_value)
    updated_columns[behavior] = col_vals
    
    

# Update the DataFrame columns with the new lists,
# specifying dtype=pl.Object to allow for mixed types.
for behavior, vals in updated_columns.items():
    cell_behavior_rasters_df = cell_behavior_rasters_df.with_columns([
        pl.Series(name=behavior, values=vals, dtype=pl.Object)
    ])

print(cell_behavior_rasters_df)

cell_behavior_rasters_pd=cell_behavior_rasters_df.to_pandas(use_pyarrow_extension_array=True)
cell_behavior_rasters_pd.to_pickle('concatanated_PSTHs.pkl')
# #plot and save raster for each cell
# #plot raster
#plot behavior limits
#save
cell_behavior_rasters_pd.to_
np.save('pandas.npy',cell_behavior_rasters_pd.to_numpy())




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import plottingFunctions as pf
#from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from joblib import Parallel, delayed
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm


# Custom callback for updating tqdm progress bar.
class TqdmBatchCompletionCallback(Parallel.BatchCompletionCallBack):
    def __init__(self, *args, tqdm_instance, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_instance = tqdm_instance

    def __call__(self, *args, **kwargs):
        self.tqdm_instance.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

@contextmanager
def tqdm_joblib(tqdm_instance):
    # Monkey-patch joblib's BatchCompletionCallBack
    old_callback = Parallel.BatchCompletionCallBack
    Parallel.BatchCompletionCallBack = lambda *args, **kwargs: TqdmBatchCompletionCallback(
        *args, tqdm_instance=tqdm_instance, **kwargs
    )
    try:
        yield
    finally:
        Parallel.BatchCompletionCallBack = old_callback
        tqdm_instance.close()
        
def plot_cells_ProcessPoolExecutor(df, savepath_0):
    # Create a process pool with as many processes as available cores.
    with ProcessPoolExecutor() as executor:
        # Submit each task to the process pool.
        futures = {
            executor.submit(plot_concatanated_PSTHs, df, i, savepath_0): i
            for i in range(len(df))
        }
        # Use tqdm to display a progress bar as tasks complete.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cells"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred for index {futures[future]}: {e}")
                
def plot_cells_joblib(df, savepath_0):
    # Run plot_concatanated_PSTHs in parallel using all available CPUs.
    Parallel(n_jobs=-1)(
        delayed(plot_concatanated_PSTHs)(df, i, savepath_0)
        for i in range(len(df))
    )
def plot_cells(df,savepath_0):
    #for i in range(0,len(df)):        
    for i in range(len(df) - 1, -1, -1):
        plot_concatanated_PSTHs(df,i,savepath_0)
        
def plot_concatanated_PSTHs(df,neuron_index=1,save_path=None):
    plt.rcParams.update({
    'font.size': 14,            # controls default text sizes
    'axes.titlesize': 26,       # fontsize of the axes title
    'axes.labelsize': 16,       # fontsize of the x and y labels
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 48      # fontsize of the figure title
    })   
    
    
    # --- Configuration ---
    #neuron_index = 0  # Choose the row index (neuron) to plot
    # Select the event columns to plot from the DataFrame keys
    event_columns_to_plot = [key for key in df.keys() if key not in ['cluster_id', 'max_site', 'region']]
    unit_IDs=df['cluster_id']
    # Filter out columns if they don't exist or don't contain lists of arrays (optional robustness)
    valid_event_columns = []
    if neuron_index < len(df):
        for col in event_columns_to_plot:
            if col in df.columns:
                 cell_data = df.loc[neuron_index, col]
                 # Basic check if it looks like a list of arrays/lists
                 if isinstance(cell_data, list) and (not cell_data or isinstance(cell_data[0], (np.ndarray, list))):
                     valid_event_columns.append(col)
                 else:
                    print(f"Skipping column '{col}' for unit {unit_IDs[neuron_index]}: Data is not a list of arrays/lists.")
            else:
                print(f"Skipping column '{col}': Not found in DataFrame.")
        event_columns_to_plot = valid_event_columns
    else:
        print(f"Error: neuron_index {neuron_index} is out of bounds for DataFrame length {len(df)}.")
        event_columns_to_plot = []
    
    
    if not event_columns_to_plot:
        print("No valid event columns found to plot. Exiting.")
        # exit() # Commented out exit for interactive environments
    else:
        print(f"Plotting for unit: {df.loc[neuron_index, 'cluster_id']} (Index: {neuron_index})")
        print(f"Events to plot: {event_columns_to_plot}")
    
    
    time_start = -5.0 # seconds
    time_end = 5.0   # seconds
    bin_width = 0.02  # seconds
    
    # --- Figure Layout ---
    n_events = len(event_columns_to_plot)
    n_cols_grid = 4  # How many event pairs per row in the figure
    n_rows_grid = math.ceil(n_events / n_cols_grid)
    
    # Each event needs 2 rows (PSTH, Raster)
    fig_rows = n_rows_grid * 2
    if fig_rows <1:
        fig_rows=1
    fig_cols = n_cols_grid
    if fig_cols <1:
        fig_cols=1
    
    # Create the figure and the grid of axes
    # Adjust figsize as needed; this might need to be quite large
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * 4, fig_rows * 2.5), # Adjust size here
                             sharex=False, # We will share X within pairs manually
                             sharey=False, # Y axes will likely have different scales
                             squeeze=False) # Ensure axes is always 2D array
    
    fig.suptitle(f"PSTH & Raster Plots for unit: {unit_IDs[neuron_index]}")#, fontsize=16)
    
    # Define time bins for histogram (same for all plots)
    bins = np.arange(time_start, time_end + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    
    # --- Loop Through Events and Plot ---
    for i, event_column in enumerate(event_columns_to_plot):
        # Calculate the grid position for this event pair
        grid_row_base = (i // n_cols_grid) * 2
        grid_col = i % n_cols_grid
    
        # Ensure calculated indices are within the bounds of the axes array
        if grid_row_base + 1 >= fig_rows or grid_col >= fig_cols:
            print(f"Warning: Calculated axes index out of bounds for event '{event_column}'. Skipping.")
            continue
    
        ax_psth = axes[grid_row_base, grid_col]
        ax_raster = axes[grid_row_base + 1, grid_col]
        pf.remove_axes(ax_raster)
        pf.remove_axes(ax_psth)
        
         
    
        # --- Data Extraction for the current event ---
        try:
            neuron_spike_data_list = df.loc[neuron_index, event_column]
            if not isinstance(neuron_spike_data_list, list):
                print(f"Warning: Data type mismatch for {event_column}. Setting to empty list.")
                neuron_spike_data_list = [] # Treat as empty
    
            # Ensure all elements are numpy arrays and filter out non-numeric if needed
            valid_trials = []
            invalid_trials = []
            for trial in neuron_spike_data_list:
                 try:
                     # Attempt conversion, ensure it's 1D
                     numeric_trial = np.array(trial, dtype=float).flatten()
                     # Optional: Check for NaNs or Infs if data quality is uncertain
                     numeric_trial = numeric_trial[np.isfinite(numeric_trial)]
                     valid_trials.append(numeric_trial)
                 except (ValueError, TypeError):
                     
                     invalid_trials+=1
                     print(f"Warning: Could not convert trial data to numeric for {event_column}. Skipping trial.")
            neuron_spike_data_list = valid_trials
    
    
            if not neuron_spike_data_list:
                all_spikes = np.array([])
                num_trials = 0
            else:
                 # Filter out completely empty trials before concatenating
                non_empty_trials = [trial for trial in neuron_spike_data_list if trial.size > 0]
                if non_empty_trials:
                    all_spikes = np.concatenate(non_empty_trials)
                else:
                    all_spikes = np.array([])
                num_trials = len(neuron_spike_data_list) # Count original number of trials attempted
    
        except Exception as e:
            print(f"Error processing data for event '{event_column}': {e}")
            all_spikes = np.array([])
            num_trials = 0
            neuron_spike_data_list = [] # Ensure it's an empty list for raster
    
    
        # --- Plot 1: Firing Rate (PSTH) ---
        if num_trials > 0 and all_spikes.size > 0: # Ensure there are spikes to calculate rate
            counts, _ = np.histogram(all_spikes, bins=bins)
            firing_rate = counts / (num_trials * bin_width)
            ax_psth.bar(bin_centers, firing_rate, width=bin_width, align='center', alpha=0.8)
            ax_psth.set_ylim(bottom=0) # Ensure y starts at 0
        else:
             # Plot empty PSTH if no trials or no spikes
            ax_psth.bar(bin_centers, np.zeros_like(bin_centers), width=bin_width, align='center')
            ax_psth.set_ylim(bottom=0, top=1) # Give some minimal height
    
        ax_psth.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_psth.set_ylabel('Mean FR (Hz)')
        ax_psth.grid(axis='y', linestyle=':', alpha=0.7)
        # Hide x-axis labels and ticks for PSTH plots (will be shared with raster below)
        ax_psth.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    
        # --- Plot 2: Spike Raster Plot ---
        if num_trials > 0:
             # Filter neuron_spike_data_list for eventplot robustness if needed
            plot_data = [trial for trial in neuron_spike_data_list if len(trial) > 0] # Only plot trials with spikes
            plot_indices = [idx for idx, trial in enumerate(neuron_spike_data_list) if len(trial) > 0]
            if plot_data: # Check if there's anything left to plot
                 ax_raster.eventplot(plot_data, colors='black', lineoffsets=plot_indices, # Use original indices
                                    linelengths=0.8, linewidths=0.5)
            ax_raster.set_ylim(-1, num_trials) # Keep Y limit based on original number of trials
            # Adjust Y ticks based on num_trials
            if num_trials <= 10:
                ticks=np.arange(0, num_trials,2,dtype=int)
                ax_raster.set_yticks(ticks)
                ax_raster.set_yticklabels(ticks)
            else:
                ticks=np.linspace(0, max(0, num_trials), 10, dtype=int)  # Fewer Y ticks for many trials
                ax_raster.set_yticks(ticks)
                ax_raster.set_yticklabels(ticks)
        else:
            ax_raster.set_ylim(-1, 1)
            ax_raster.set_yticks([])
    
    
        ax_raster.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_psth.set_title(event_column)#, fontsize=10) # Title on the raster plot
        ax_raster.set_ylabel('Trial')
        ax_raster.grid(axis='y', linestyle=':', alpha=0.7)
    
        # --- * CORRECTED AXIS SHARING * ---
        ax_psth.sharex(ax_raster)
        # --- * ------------------------ * ---
    
        # Set shared x-limits for the pair (setting on one affects the other now)
        ax_raster.set_xlim(time_start, time_end)
    
        # Determine if this subplot is in the last row *that contains plots*
        # This logic needs to be correct even if the last row isn't full
        last_event_row_index = (n_events - 1) // n_cols_grid
        current_event_row_index = i // n_cols_grid
        is_in_last_plotted_row = (current_event_row_index == last_event_row_index)
    
        if is_in_last_plotted_row:
            ax_raster.set_xlabel('Time (s)')
            # Ensure tick labels are visible if they were turned off by sharing
            ax_raster.tick_params(axis='x', which='both', labelbottom=True)
        # else: # Explicitly turn off if not bottom row (sharex might have turned it on)
        #      ax_raster.tick_params(axis='x', which='both', labelbottom=False) #This is handled by ax_psth tick_params + sharex
    
    
        # Remove y-labels for plots not in the first column to reduce clutter
        if grid_col > 0:
            ax_psth.tick_params(axis='y', which='both', labelleft=False)
            ax_raster.tick_params(axis='y', which='both', labelleft=False)
    
    
    # --- Clean up unused axes ---
    for i in range(n_events, n_rows_grid * n_cols_grid):
        grid_row_base = (i // n_cols_grid) * 2
        grid_col = i % n_cols_grid
        # Check bounds before attempting to access axes
        if grid_row_base + 1 < fig_rows and grid_col < fig_cols:
            axes[grid_row_base, grid_col].axis('off') # Turn off PSTH axis
            axes[grid_row_base + 1, grid_col].axis('off') # Turn off Raster axis
    
    
    # --- Final Adjustments ---
    #plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout (leave space for suptitle)
    #plt.show()
    fig.tight_layout(rect=[0, 0.03, 1, 0.96],pad=1)
    fig.canvas.draw()  # Force update to the figure
    prefix=save_path / f"{unit_IDs[neuron_index]}_concat_PSTH_iFR"
    suffix = "png"
    file_name=f"{prefix}.{suffix}"
    fig.savefig(file_name)
#    fig.savefig(save_path / f"{unit_IDs[neuron_index]}_concat_PSTH.svg")
    
    print(f"saved to {save_path} / {unit_IDs[neuron_index]}_concat_PSTH.png")
    plt.close()





