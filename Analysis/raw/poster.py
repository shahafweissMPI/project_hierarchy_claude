# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:26:32 2025

@author: su-weisss
"""


## Standard libraries
import os
import re
import time
import math
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict
import multiprocessing
import concurrent.futures

# Third-party libraries
import IPython
import polars as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import matplotlib
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch

# Multiprocessing helpers
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ProcessPoolExecutor, as_completed

# Proprietary functions
import preprocessFunctions as pp
import plottingFunctions as pf
import helperFunctions as hf
import spikeinterface.full as si
import spikeinterface.widgets as sw
# Optional: Set up matplotlib defaults (A4 size)
#A0: 841 x 1189 mm A1: 594 x 841 mm A2: 420 x 594 mm A3: 297 x 420 mm A4: 210 x 297 mm A5: 148 x 210 mm
hf.set_page_format('A2')
inchpermm=1/25.4
A3_fig_size=( 297 *inchpermm, 420*inchpermm);A2_fig_size=( 420 *inchpermm, 594*inchpermm);A1_fig_size=( 594 *inchpermm, 841*inchpermm);A0_fig_size=( 841 *inchpermm, 1189*inchpermm)
A0_fig_size_portrait=( 1189 *inchpermm, 841*inchpermm)

#%% Loop through sessions
animal = 'afm16924';sessions =['240522','240523_0','240524','240525','240526','240527','240529']  # can be a list if you have multiple sessions
#animal='afm17365';sessions=['241210_02']
target_bs_0 = []           # Which behaviors to plot? Leave empty for all.
savepath_0 = Path(r"G:\test")
#savepath_0 = Path(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\figures_for_Petros')
target_bs = []
target_cells=[]
target_regions = ['DPAG','VPAG','VLPAG','LPAG','DLPAG','DMPAG','VMPAG','PAG']  # From which regions should the neurons be? Leave empty for all.
plot_waveforms_by_column=True
plot_template_metrics=False
plot_trajectory=True



#%% functions

def plot_waveforms_from_PAG_columns(sorting_analyzer,unit_ids,PAG_column,savepath_0):         
     region = PAG_column 
     savepath_0=Path(savepath_0).as_posix()
     
     try:     
         
         unit_ids = sorting_analyzer.unit_ids[np.isin(sorting_analyzer.unit_ids,unit_ids)]#check units are actually in the sorting analyzer         
         unit_ids=unit_ids[::50]
         sw.plot_unit_templates(sorting_analyzer,unit_ids=unit_ids,scalebar=True,max_spikes_per_unit=1,ncols=5,set_title=False,figsize=A2_fig_size,axis_equal=True,figtitle=f'{animal} {session} {region}')     
         save_str=f'{savepath_0}/templates{region}_{session}.png';     plt.savefig(save_str,dpi=300)
         print(save_str)
         IPython.embed()
         save_str=f'{savepath_0}/templates{region}_{session}.svg';     plt.savefig(save_str)
         plt.close('all')
        # sw.plot_unit_waveforms(sorting_analyzer,unit_ids=unit_ids,scalebar=True,scale = 0.5,plot_legend=False,ncols=10,set_title=False,figsize=A2_fig_size,axis_equal=True)     
       #  save_str=f'{savepath_0}/{region}_{session}_waveforms.png';     plt.savefig(save_str,dpi=300)
         #save_str=f'{savepath_0}/{region}_{session}_waveforms.svg';     plt.savefig(save_str)
         #plt.close('all')
         print(f'{session} \ {region} \ waveforms saved')
     except:
             print(f'something wrong with {session} analyzer. recompute please')
            
         
         
def plottemplate_metrics_from_PAG_columns(sorting_analyzer,savepath_0): 
         if len(sorting_analyzer.get_loaded_extension_names()) == 0 :
             print(f'{session} no extensions computed')
             return
         extensions=sorting_analyzer.extensions
         if len(extensions)<5:
             print(f'{session} not enough extensions computed')
             return
         savepath_0=Path(savepath_0).as_posix()                 
         unit_ids = n_cluster_index
         unit_ids = sorting_analyzer.unit_ids[np.isin(sorting_analyzer.unit_ids,unit_ids)]#check units are actually in the sorting analyzer 
        # Define the target regions and choose a mapping to specific indices in tab20.
        # Here we ensure LPAG is blue (tab20 index 0) and VPAG is red (tab20 index 3)

         region_to_color = {
            'LPAG': 'magenta',   # blue
            'VLPAG': 'springgreen',   # red
            'VPAG': 'green',
            'VMPAG': 'red',
            'DPAG': 'orangered',
            'DLPAG': 'cornflowerblue',
            'DMPAG': 'blue',                        
            'PAG': 'black'
        }
         
         # Find the indices in n_cluster_index for each unit in unit_ids
         indices = [np.where(n_cluster_index == uid)[0][0] for uid in unit_ids]
         
        # Now use these indices to get corresponding region labels.
         region_labels = n_region_index[indices]
         unit_colors = {}
         for unit_id, region_i in zip(unit_ids, region_labels):
            # Use the region's color if available; otherwise, assign a default color.
           unit_colors[unit_id] = region_to_color.get(region_i, 'black')
        
        
         sw.plot_template_metrics(sorting_analyzer,unit_ids=unit_ids,unit_colors=unit_colors,figsize=A0_fig_size,figtitle=f'{animal} {session}')
         
         save_str=f'{savepath_0}/templates_metrics_{session}.png';     plt.savefig(save_str,dpi=300)
         save_str=f'{savepath_0}/templates_metrics_{session}.svg';     plt.savefig(save_str)
         plt.close('all')
        
        # unit_colors is now a dictionary mapping unit id -> color.
     
         return
     
     
     
def plot_neuron_behavioral_tuning_subplots_by_trial_sample(n_spike_times, locations, frame_index_s, velocity,
                                                           n_cluster_index, neuron_ID, savepath_0, behaviour, session):
    """
    Plot mouse trajectories with color coding by relative sample index (0 to 1)
    from the start to stop of each behavioral trial, with overlaid neuron spike locations.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    velocity[np.isnan(velocity)] = 0
    velocity[velocity > 150] = 150

    # Extract spikes for this neuron
    i = np.where(n_cluster_index == neuron_ID)
    if len(i[0]) == 0:
        print(f"Neuron ID {neuron_ID} not found.")
        return
    i = i[0][0]
    spike_timestamps = n_spike_times[i]

    # Positions and time
    x_coords_all = locations[:, 1]
    y_coords_all = locations[:, 0]
    time_coords_all = frame_index_s.astype(float)

    # Equalize lengths
    min_length = min(len(x_coords_all), len(y_coords_all), len(time_coords_all))
    x_coords_all, y_coords_all, time_coords_all = x_coords_all[:min_length], y_coords_all[:min_length], time_coords_all[:min_length]

    # Global limits
    gx_min, gx_max = np.nanmin(x_coords_all), np.nanmax(x_coords_all)
    gy_min, gy_max = np.nanmin(y_coords_all), np.nanmax(y_coords_all)
    pad_x = (gx_max - gx_min) * 0.05
    pad_y = (gy_max - gy_min) * 0.05
    gx_min -= pad_x; gx_max += pad_x
    gy_min -= pad_y; gy_max += pad_y

    plt.rcParams.update({'font.size': 14})

    unique_behaviors = sorted(behaviour['behaviours'].unique().tolist())
    filtered_behaviors = [b for b in unique_behaviors if b != 'random_baseline']

    total_axes_needed = 1 + len(filtered_behaviors)
    cols = 3
    rows = int(np.ceil(total_axes_needed / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axs = axs.flatten()

    sort_idx = np.argsort(time_coords_all)
    sorted_time_coords = time_coords_all[sort_idx]
    sorted_x_coords = x_coords_all[sort_idx]
    sorted_y_coords = y_coords_all[sort_idx]

    # --- Subplot 0: Overall session ---
    ax_overall = axs[0]
    overall_relative_sample = np.linspace(0, 1, len(x_coords_all))
    path = ax_overall.scatter(y_coords_all, x_coords_all,
                              c=overall_relative_sample, cmap='cool',
                              s=5, vmin=0, vmax=1, zorder=1)
    span_mask = (spike_timestamps >= time_coords_all.min()) & (spike_timestamps <= time_coords_all.max())
    spikes_overall = spike_timestamps[span_mask]
    if len(spikes_overall) > 0:
        x_spikes = np.interp(spikes_overall, sorted_time_coords, sorted_x_coords)
        y_spikes = np.interp(spikes_overall, sorted_time_coords, sorted_y_coords)
        #ax_overall.scatter(y_spikes, x_spikes, color='red', s=10, zorder=2, label='Spikes')

    ax_overall.set_title('Overall Session')
    ax_overall.set_xlabel('X (cm)')
    ax_overall.set_ylabel('Y (cm)')
    ax_overall.set_aspect('equal', adjustable='box')
    ax_overall.set_ylim(gy_min, gy_max)
    ax_overall.set_xlim(gx_min, gx_max)
    ax_overall.legend(loc='upper right')

    # --- Per behavior ---
    for idx, beh_name in enumerate(filtered_behaviors, start=1):
        ax = axs[idx]
        beh_group = behaviour[behaviour['behaviours'] == beh_name]
        starts = beh_group[beh_group['start_stop'] == 'START']['frames_s'].values
        stops = beh_group[beh_group['start_stop'] == 'STOP']['frames_s'].values
        points = beh_group[beh_group['start_stop'] == 'POINT']['frames_s'].values
        intervals = list(zip(starts, stops))

        for start, stop in intervals:
            trial_mask = (time_coords_all >= start) & (time_coords_all <= stop)
            trial_x = x_coords_all[trial_mask]
            trial_y = y_coords_all[trial_mask]
            num_points = len(trial_x)
            if num_points < 2:
                continue
            trial_relative_sample = np.linspace(0, 1, num_points)

            ax.scatter(trial_y, trial_x, c=trial_relative_sample, cmap='cool',
                       s=5, vmin=0, vmax=1, zorder=1)

            # Spikes for trial
            spike_mask = (spike_timestamps >= start) & (spike_timestamps <= stop)
            trial_spikes = spike_timestamps[spike_mask]
            if len(trial_spikes) > 0:
                x_spk = np.interp(trial_spikes, sorted_time_coords, sorted_x_coords)
                y_spk = np.interp(trial_spikes, sorted_time_coords, sorted_y_coords)
                ax.scatter(y_spk, x_spk, color='red', s=20, zorder=2)

        for point in points:
            point_mask = (np.abs(spike_timestamps - point) < 0.05)
            point_spikes = spike_timestamps[point_mask]
            if len(point_spikes) > 0:
                x_pt = np.interp(point_spikes, sorted_time_coords, sorted_x_coords)
                y_pt = np.interp(point_spikes, sorted_time_coords, sorted_y_coords)
             #   ax.scatter(y_pt, x_pt, color='red', s=10, zorder=2)

        ax.set_title(f'{beh_name}')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(gy_min, gy_max)
        ax.set_xlim(gx_min, gx_max)

    # Turn off unused subplots
    for ax in axs[total_axes_needed:]:
        fig.delaxes(ax)

    # Colorbar (applies to normalized sample index)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(path, cax=cbar_ax)
    cbar.set_label('Relative Sample Index (0=start, 1=end)', rotation=270, labelpad=20)

    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, wspace=0.3, hspace=0.4)
    IPython.embed()

    save_path_svg = savepath_0 / f"{session}/neuron_{neuron_ID}_session_{session}_behaviors_trajectory_samples.svg"
    save_path_png = savepath_0 / f"{session}/neuron_{neuron_ID}_session_{session}_behaviors_trajectory_samples.png"
    fig.savefig(save_path_svg, bbox_inches='tight')
    fig.savefig(save_path_png, bbox_inches='tight')
    plt.close(fig)

    print(f"Finished plotting behavioral tuning subplots (trial relative sample-coded) for Neuron {neuron_ID} at:\n {save_path_png}")     

def plot_neuron_behavioral_tuning_subplots_speed(n_spike_times, locations, frame_index_s, velocity, n_cluster_index, neuron_ID, savepath_0, behaviour, session):
    """
    Plot mouse trajectories (color-coded by speed) with overlaid neuron spike locations
    for each individual behavior as subplots within a single figure for a given neuron,
    with consistent axis scaling and an additional subplot for the overall trajectory.

    Parameters:
    - n_spike_times (array): Timestamps of neuron spikes for all neurons.
    - locations (array): Mouse's [y, x] coordinates over time.
    - frame_index_s (array): Mouse's time coordinates (timestamps of frames).
    - velocity (array): Mouse's speed at each time point.
    - n_cluster_index (array): Array of neuron IDs corresponding to n_spike_times.
    - neuron_ID (int): The specific neuron ID to plot.
    - savepath_0 (pathlib.Path): Directory to save the plots.
    - behaviour (pd.DataFrame): DataFrame containing behavioral annotations with 'behavioural_category',
                                'behaviours', 'start_stop', and 'frames_s' columns.
    - session (str): Identifier for the current session (e.g., 'session_01').
    """
    velocity[np.where(np.isnan(velocity))]=0
    velocity[np.where(150<(velocity))]=150
    # Extract data for the specific neuron
    n = neuron_ID
    i = np.where(n_cluster_index == n)
    if len(i[0]) == 0:
        print(f"Neuron ID {neuron_ID} not found in n_cluster_index. Skipping plot.")
        return
    i = i[0][0]
    spike_timestamps = n_spike_times[i]

    # Prepare general data
    # Ensure consistency in coordinate order: locations[:, 1] for x, locations[:, 0] for y
    x_coords_all = locations[:, 1]
    y_coords_all = locations[:, 0]
    time_coords_all = frame_index_s
    speed_all = velocity # Use the provided velocity array for speed

    # Ensure all arrays are of the same length by trimming to the minimum
    min_length = min(len(x_coords_all), len(y_coords_all), len(time_coords_all), len(speed_all))
    x_coords_all = x_coords_all[:min_length]
    y_coords_all = y_coords_all[:min_length]
    time_coords_all = time_coords_all[:min_length]
    speed_all = speed_all[:min_length]

    # Calculate global X, Y, and Speed limits for consistent scaling and colormap
    global_x_min, global_x_max = np.nanmin(x_coords_all), np.nanmax(x_coords_all)
    global_y_min, global_y_max = np.nanmin(y_coords_all), np.nanmax(y_coords_all)
    global_speed_min, global_speed_max = np.nanmin(speed_all), np.nanmax(speed_all)
    
    # Add a small padding to the position limits
    padding_x = (global_x_max - global_x_min) * 0.05
    padding_y = (global_y_max - global_y_min) * 0.05
    global_x_min -= padding_x
    global_x_max += padding_x
    global_y_min -= padding_y
    global_y_max += padding_y

    # Set up plot styling
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 20
    })

    unique_behaviors = sorted(behaviour['behaviours'].unique().tolist()) # Sort for consistent order
    num_behaviors = len(unique_behaviors)
    total_subplots = num_behaviors # +1 for the overall trajectory plot

    # Determine grid size for subplots
    cols = 3 # You can adjust this number of columns
    rows = int(np.ceil(total_subplots / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False) # Adjust figsize as needed
    axs = axs.flatten() # Flatten the 2D array of axes for easy iteration

    # To ensure consistent interpolation, time_coords_all must be sorted
    sorted_time_indices = np.argsort(time_coords_all)
    sorted_time_coords_interp = time_coords_all[sorted_time_indices]
    sorted_x_coords_interp = x_coords_all[sorted_time_indices]
    sorted_y_coords_interp = y_coords_all[sorted_time_indices]

    # --- Plot the overall trajectory first (index 0) ---
    ax_overall = axs[0]
    
    # Use scatter for color-coding by speed
    # 'path' object is captured to create the colorbar later
    path = ax_overall.scatter(y_coords_all, x_coords_all, c=speed_all, cmap='viridis', 
                              s=5, vmin=global_speed_min, vmax=global_speed_max, zorder=1)
    
    # Filter spikes that occurred within the entire session time range
    spike_mask_overall = (spike_timestamps >= time_coords_all.min()) & (spike_timestamps <= time_coords_all.max())
    spikes_overall = spike_timestamps[spike_mask_overall]

    if len(spikes_overall) > 0 and len(sorted_time_coords_interp) > 1:
        x_spike_interp_overall = np.interp(spikes_overall, sorted_time_coords_interp, sorted_x_coords_interp)
        y_spike_interp_overall = np.interp(spikes_overall, sorted_time_coords_interp, sorted_y_coords_interp)
   #     ax_overall.scatter(y_spike_interp_overall, x_spike_interp_overall, color='red', s=10, label='Spikes', zorder=2)
    
    ax_overall.set_title('Overall Session')
    ax_overall.set_xlabel('X (cm)')
    ax_overall.set_ylabel('Y (cm)')
    ax_overall.set_aspect('equal', adjustable='box')
    ax_overall.set_ylim(global_y_min, global_y_max) # Apply global limits
    ax_overall.set_xlim(global_x_min, global_x_max) # Apply global limits
    # Legend for spikes (trajectory legend comes from the colorbar)
  #  ax_overall.legend(handles=[plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Spikes')], loc='upper right')


    # --- Loop through each unique behavior and plot its segments in subsequent subplots ---
    for idx, beh_name in enumerate(unique_behaviors):
        ax = axs[idx + 1] # Offset index by 1 because ax[0] is for overall trajectory
        if 'random_baseline'==beh_name:
            random_ind=idx
            continue
       
        
        behavior_group = behaviour[behaviour['behaviours'] == beh_name]
        
        starts = behavior_group[behavior_group['start_stop'] == 'START']['frames_s'].values
        stops = behavior_group[behavior_group['start_stop'] == 'STOP']['frames_s'].values
        points = behavior_group[behavior_group['start_stop'] == 'POINT']['frames_s'].values
        
        intervals = list(zip(starts, stops))
        
        # Plot trajectory segments and spikes for START/STOP intervals
        for start, stop in intervals:
            trajectory_mask = (time_coords_all >= start) & (time_coords_all <= stop)
            current_x = x_coords_all[trajectory_mask]
            current_y = y_coords_all[trajectory_mask]
            current_speed = speed_all[trajectory_mask]

            # Use scatter for color-coding by speed
            ax.scatter(current_y, current_x, c=current_speed, cmap='viridis', 
                       s=5, vmin=global_speed_min, vmax=global_speed_max, zorder=1)

            spike_mask = (spike_timestamps >= start) & (spike_timestamps <= stop)
            current_spike_times = spike_timestamps[spike_mask]

            if len(current_spike_times) > 0 and len(sorted_time_coords_interp) > 1:
                x_spike_interp = np.interp(current_spike_times, sorted_time_coords_interp, sorted_x_coords_interp)
                y_spike_interp = np.interp(current_spike_times, sorted_time_coords_interp, sorted_y_coords_interp)
              #  ax.scatter(y_spike_interp, x_spike_interp, color='red', s=20, zorder=2) # Smaller dots for subplots

        # Plot spikes for POINT behaviors
        for point_time in points:
            # Filter spikes at this point (or very close in time)
            point_spike_mask = (np.abs(spike_timestamps - point_time) < 0.05) 
            point_spike_times = spike_timestamps[point_spike_mask]
            
            if len(point_spike_times) > 0 and len(sorted_time_coords_interp) > 1:
                x_point_spike_interp = np.interp(point_spike_times, sorted_time_coords_interp, sorted_x_coords_interp)
                y_point_spike_interp = np.interp(point_spike_times, sorted_time_coords_interp, sorted_y_coords_interp)
             #   ax.scatter(y_point_spike_interp, x_point_spike_interp, color='red', s=10, zorder=2)


        ax.set_title(f'{beh_name}')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for spatial plots
        ax.set_ylim(global_y_min-20, global_y_max+20) # Apply global limits
        ax.set_xlim(global_x_min-20, global_x_max+20) # Apply global limits

    # Turn off any unused subplots (if total_subplots is not a perfect multiple of cols)
    for i in range(total_subplots, len(axs)):
        fig.delaxes(axs[i])
    fig.delaxes(axs[random_ind])

    # Add an overall title for the figure
    fig.suptitle(f'Neuron {neuron_ID} - Session {session}: Trajectories (Speed) and Spikes per Behavior', y=1.02) # Adjust y to prevent overlap

    # Add a single colorbar for the entire figure based on the 'path' object from the overall plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(path, cax=cbar_ax)
    cbar.set_label('Speed (cm/s)', rotation=270, labelpad=20)


    #plt.tight_layout(rect=[0, 0.03, 0.9, 0.98]) # Adjust rect to make space for suptitle and colorbar
    
    # Save the plot
    save_path_str = savepath_0 / f"neuron_{neuron_ID}_session_{session}_behaviors_trajectory_speed.svg"
    plt.savefig(save_path_str)
    save_path_str = savepath_0 / f"neuron_{neuron_ID}_session_{session}_behaviors_trajectory_speed.png"
    plt.savefig(save_path_str)
    plt.close(fig) # Close the figure to free up memory  
   

    print(f"Finished plotting behavioral tuning subplots (speed-coded) for Neuron {neuron_ID} at:\n {save_path_str}")



def plot_neuron_behavioral_tuning_subplots_by_trial_time(
    n_spike_times, locations, frame_index_s, velocity,
    n_cluster_index, neuron_ID, savepath_0, behaviour, session
):
    """
    Plot mouse trajectories with color coding by relative sample number (0–1)
    for each behavioral trial (START to STOP) as subplots, with overlaid neuron spike locations.
    
    Circle: diameter=100, center = mean(x,y) from all session coordinates.
    Behaviors with n=0 trials are skipped from the plot.
    """

    # Handle NaNs in velocity
    velocity = np.nan_to_num(velocity, nan=0.0)
    velocity[velocity > 150] = 150

    # Extract spike times for specified neuron
    idx = np.where(n_cluster_index == neuron_ID)[0]
    if len(idx) == 0:
        print(f"Neuron ID {neuron_ID} not found in n_cluster_index. Skipping plot.")
        return
    spike_timestamps = n_spike_times[idx[0]]

    # Prepare positions and timestamps
    x_coords_all = locations[:, 1]
    y_coords_all = locations[:, 0]
    time_coords_all = frame_index_s.astype(float)
    min_len = min(len(x_coords_all), len(y_coords_all), len(time_coords_all))
    x_coords_all = x_coords_all[:min_len]
    y_coords_all = y_coords_all[:min_len]
    time_coords_all = time_coords_all[:min_len]

    # Compute circle center from data mean
    circle_center_x = np.nanmin(x_coords_all) + (np.nanmax(x_coords_all) - np.nanmin(x_coords_all)) / 2
    circle_center_y = np.nanmin(y_coords_all) + (np.nanmax(y_coords_all) - np.nanmin(y_coords_all)) / 2
    circle_center = (circle_center_x, circle_center_y)
    circle_diameter = 95
    circle_radius = circle_diameter / 2

    # Axis limits to fit the full circle
    global_x_min, global_x_max = circle_center_x - circle_radius - 2, circle_center_x + circle_radius + 2
    global_y_min, global_y_max = circle_center_y - circle_radius - 2, circle_center_y + circle_radius + 2

    # Normalized relative sample index for overall session
    overall_rel_samples = np.linspace(0, 1, len(time_coords_all))

    # Filter behaviors and skip any with 0 trials
    behaviors = []
    for beh_name in sorted(behaviour['behaviours'].unique().tolist()):
        if beh_name == "random_baseline":
            continue
        beh_group = behaviour[behaviour['behaviours'] == beh_name]
        starts = beh_group.loc[beh_group['start_stop'] == 'START', 'frames_s'].values
        stops = beh_group.loc[beh_group['start_stop'] == 'STOP', 'frames_s'].values
        intervals = list(zip(starts, stops))
        if len(intervals) > 0:
            behaviors.append(beh_name)

    # Calculate subplot grid size
    total_axes = 1 + len(behaviors)
    cols = 3
    rows = int(np.ceil(total_axes / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axs = axs.flatten()

    # Sort arrays for spike interpolation
    sort_idx = np.argsort(time_coords_all)
    sorted_time_coords = time_coords_all[sort_idx]
    sorted_x_coords = x_coords_all[sort_idx]
    sorted_y_coords = y_coords_all[sort_idx]

    # --- Overall session subplot (index 0) ---
    ax0 = axs[0]
    path = ax0.scatter(y_coords_all, x_coords_all, c=overall_rel_samples, cmap='cool', s=5, vmin=0, vmax=1, zorder=1)
    mask_overall = (spike_timestamps >= time_coords_all.min()) & (spike_timestamps <= time_coords_all.max())
    overall_spikes = spike_timestamps[mask_overall]
    if len(overall_spikes) > 0:
        x_spk = np.interp(overall_spikes, sorted_time_coords, sorted_x_coords)
        y_spk = np.interp(overall_spikes, sorted_time_coords, sorted_y_coords)
        #ax0.scatter(y_spk, x_spk, color='red', s=10, zorder=2)
    ax0.add_patch(Circle(circle_center, circle_radius, color='black', fill=False, lw=2))
    ax0.set_title("Entire Session")

    # --- Per behavior subplots ---
    for i, beh_name in enumerate(behaviors, start=1):
        ax = axs[i]
        beh_group = behaviour[behaviour['behaviours'] == beh_name]
        starts = beh_group.loc[beh_group['start_stop'] == 'START', 'frames_s'].values
        stops = beh_group.loc[beh_group['start_stop'] == 'STOP', 'frames_s'].values
        intervals = list(zip(starts, stops))
        trial_count = 0

        for start, stop in intervals:
            mask_trial = (time_coords_all >= start) & (time_coords_all <= stop)
            if not np.any(mask_trial):
                continue
            trial_x = x_coords_all[mask_trial]
            trial_y = y_coords_all[mask_trial]
            n_samples = len(trial_x)
            if n_samples <= 1:
                continue
            trial_count += 1

            # Normalized sample number in trial
            trial_rel_samples = np.linspace(0, 1, n_samples)
            ax.scatter(trial_y, trial_x, c=trial_rel_samples, cmap='cool', s=5, vmin=0, vmax=1, zorder=1)

            # Overlay spikes in trial
            mask_spk = (spike_timestamps >= start) & (spike_timestamps <= stop)
            trial_spikes = spike_timestamps[mask_spk]
            if len(trial_spikes) > 0:
                x_s = np.interp(trial_spikes, sorted_time_coords, sorted_x_coords)
                y_s = np.interp(trial_spikes, sorted_time_coords, sorted_y_coords)
          #      ax.scatter(y_s, x_s, color='red', s=20, zorder=2)

        # Add circle and title
        ax.add_patch(Circle(circle_center, circle_radius, color='black', fill=False, lw=1.5))
        ax.set_title(f"{beh_name} (n={trial_count})")

    # --- Formatting ---
    for ax in axs[:total_axes]:
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(global_y_min-20, global_y_max+20)
        ax.set_xlim(global_x_min-20, global_x_max+20)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Remove unused axes
    for ax in axs[total_axes:]:
        fig.delaxes(ax)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(path, cax=cbar_ax)
    cb.set_label('relative time', rotation=270, labelpad=20)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["start", "stop"])

    # Supertitle
    fig.suptitle(f"{animal} {session} Unit {neuron_ID}", fontsize=18, y=0.98)

    # Layout adjustment
    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)

    # Save
    save_path_svg = Path(savepath_0) / f"{session}" / f"{animal}_{session}_unit_{neuron_ID}_behaviors_trajectory_time.svg"
    save_path_png = Path(savepath_0) / f"{session}" / f"{animal}_{session}_unit_{neuron_ID}_behaviors_trajectory_time.png"
    fig.savefig(save_path_svg)
    fig.savefig(save_path_png)
    plt.close(fig)

    print(f"Finished plotting for {animal} {session} Unit {neuron_ID} → {save_path_png}")
def plot_neuron_behavioral_tuning_subplots_by_trial_time_all(
    n_spike_times, locations, frame_index_s, velocity,
    n_cluster_index, savepath_0, behaviour, session
):
    """
    Plot mouse trajectories with color coding by relative sample number (0–1)
    for each behavioral trial (START to STOP) as subplots, with overlaid neuron spike locations.
    
    Circle: diameter=100, center = mean(x,y) from all session coordinates.
    Behaviors with n=0 trials are skipped from the plot.
    """

    # Handle NaNs in velocity
    velocity = np.nan_to_num(velocity, nan=0.0)
    velocity[velocity > 150] = 150

    # Extract spike times for specified neuron
    
    

    # Prepare positions and timestamps
    x_coords_all = locations[:, 1]
    y_coords_all = locations[:, 0]
    time_coords_all = frame_index_s.astype(float)
    min_len = min(len(x_coords_all), len(y_coords_all), len(time_coords_all))
    x_coords_all = x_coords_all[:min_len]
    y_coords_all = y_coords_all[:min_len]
    time_coords_all = time_coords_all[:min_len]

    # Compute circle center from data mean
    circle_center_x = np.nanmin(x_coords_all) + (np.nanmax(x_coords_all) - np.nanmin(x_coords_all)) / 2
    circle_center_y = np.nanmin(y_coords_all) + (np.nanmax(y_coords_all) - np.nanmin(y_coords_all)) / 2
    circle_center = (circle_center_x, circle_center_y)
    circle_diameter = 95
    circle_radius = circle_diameter / 2

    # Axis limits to fit the full circle
    global_x_min, global_x_max = circle_center_x - circle_radius - 2, circle_center_x + circle_radius + 2
    global_y_min, global_y_max = circle_center_y - circle_radius - 2, circle_center_y + circle_radius + 2

    # Normalized relative sample index for overall session
    overall_rel_samples = np.linspace(0, 1, len(time_coords_all))

    # Filter behaviors and skip any with 0 trials
    behaviors = []
    for beh_name in sorted(behaviour['behaviours'].unique().tolist()):
        if beh_name == "random_baseline":
            continue
        beh_group = behaviour[behaviour['behaviours'] == beh_name]
        starts = beh_group.loc[beh_group['start_stop'] == 'START', 'frames_s'].values
        stops = beh_group.loc[beh_group['start_stop'] == 'STOP', 'frames_s'].values
        intervals = list(zip(starts, stops))
        if len(intervals) > 0:
            behaviors.append(beh_name)

    # Calculate subplot grid size
    total_axes = 1 + len(behaviors)
    cols = 3
    rows = int(np.ceil(total_axes / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axs = axs.flatten()

    # Sort arrays for spike interpolation
    sort_idx = np.argsort(time_coords_all)
    sorted_time_coords = time_coords_all[sort_idx]
    sorted_x_coords = x_coords_all[sort_idx]
    sorted_y_coords = y_coords_all[sort_idx]

    # --- Overall session subplot (index 0) ---
    ax0 = axs[0]
    path = ax0.scatter(y_coords_all, x_coords_all, c=overall_rel_samples, cmap='cool', s=5, vmin=0, vmax=1, zorder=1)
    
    
        #ax0.scatter(y_spk, x_spk, color='red', s=10, zorder=2)
    ax0.add_patch(Circle(circle_center, circle_radius, color='black', fill=False, lw=2))
    ax0.set_title("Entire Session")

    # --- Per behavior subplots ---
    for i, beh_name in enumerate(behaviors, start=1):
        ax = axs[i]
        beh_group = behaviour[behaviour['behaviours'] == beh_name]
        starts = beh_group.loc[beh_group['start_stop'] == 'START', 'frames_s'].values
        stops = beh_group.loc[beh_group['start_stop'] == 'STOP', 'frames_s'].values
        intervals = list(zip(starts, stops))
        trial_count = 0

        for start, stop in intervals:
            mask_trial = (time_coords_all >= start) & (time_coords_all <= stop)
            if not np.any(mask_trial):
                continue
            trial_x = x_coords_all[mask_trial]
            trial_y = y_coords_all[mask_trial]
            n_samples = len(trial_x)
            if n_samples <= 1:
                continue
            trial_count += 1

            # Normalized sample number in trial
            trial_rel_samples = np.linspace(0, 1, n_samples)
            ax.scatter(trial_y, trial_x, c=trial_rel_samples, cmap='cool', s=5, vmin=0, vmax=1, zorder=1)

            # Overlay spikes in trial
    
    
        # Add circle and title
        ax.add_patch(Circle(circle_center, circle_radius, color='black', fill=False, lw=1.5))
        ax.set_title(f"{beh_name} (n={trial_count})")

    # --- Formatting ---
    for ax in axs[:total_axes]:
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(global_y_min-20, global_y_max+20)
        ax.set_xlim(global_x_min-20, global_x_max+20)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Remove unused axes
    for ax in axs[total_axes:]:
        fig.delaxes(ax)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(path, cax=cbar_ax)
    cb.set_label('relative time', rotation=270, labelpad=20)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["start", "stop"])

    # Supertitle
    fig.suptitle(f"{animal} {session}", fontsize=18, y=0.98)

    # Layout adjustment
    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)

    # Save
    save_path_svg = Path(savepath_0) / f"{session}" / f"{animal}_{session}_behaviors_trajectory_time.svg"
    save_path_png = Path(savepath_0) / f"{session}" / f"{animal}_{session}_behaviors_trajectory_time.png"
    fig.savefig(save_path_svg)
    fig.savefig(save_path_png)
    plt.close(fig)

    print(f"Finished plotting for {animal} {session}  → {save_path_png}")    
def generate_unique_colors(n):
    """Return n unique RGBA colors for neurons.
       If using 'hsv' for n > 20, shuffle colors randomly."""
    if n <= 3:
        cmap = plt.get_cmap('brg', n)
        return [cmap(i) for i in range(n)]
    if n <= 10:
        cmap = plt.get_cmap('tab10', n)
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap('tab20', n)
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / n) for i in range(n)]
        np.random.shuffle(colors)
        return colors

def smooth_trace(trace, sigma_bins):
    """Smooths a 1D array with a Gaussian kernel."""
    return gaussian_filter1d(trace, sigma=sigma_bins)

def plot_sta_per_behavior(
    behaviour_df,
    n_spike_times,
    window=(-1, 2),
    bin_size=0.01,
    baseline_window=(-1, 0),
    raster_marker_size=2,
    plot_all_neurons=True,
    output_folder=None,
    show_avg_peak_lines=False,
    raw_alpha=0.3,
    smooth=True,
    smooth_sigma=1.0,
    only_raster_grouped=False,
    n_cluster_index=None,
    n_cluster_index_subset=None
):
    """
    Full spike-triggered average + rasters for each behavior.
    Supports subset filtering, merged raster mode, and ordered legends in raster plots.
    """
    if output_folder is None:
        raise ValueError("You must provide output_folder.")
    os.makedirs(output_folder, exist_ok=True)

    merged_subset_mode = (
        n_cluster_index is not None and
        n_cluster_index_subset is not None and
        len(n_cluster_index_subset) > 0
    )

    if n_cluster_index is not None and n_cluster_index_subset is not None:
        mask = np.isin(n_cluster_index, n_cluster_index_subset)
        n_spike_times_filtered = [st for m, st in zip(mask, n_spike_times) if m]
        n_cluster_index_filtered = n_cluster_index[mask]
    else:
        n_spike_times_filtered = n_spike_times
        n_cluster_index_filtered = (
            n_cluster_index if n_cluster_index is not None 
            else np.arange(len(n_spike_times))
        )

    time_bins = np.arange(window[0], window[1] + bin_size, bin_size)
    t_centers = time_bins[:-1] + bin_size / 2
    neuron_colors = generate_unique_colors(len(n_spike_times_filtered))
    unique_neurons = np.unique(n_cluster_index_filtered)
    behaviours_unique = behaviour_df['behaviours'].unique()

    for beh in behaviours_unique:
        beh_onsets = behaviour_df.loc[
            (behaviour_df['behaviours'] == beh) &
            (behaviour_df['start_stop'].isin(['START', 'POINT'])),
            'frames_s'
        ].values

        if beh_onsets.size == 0:
            print(f"⚠️ No START/POINT events for behavior '{beh}'; skipping.")
            continue

        all_rates_raw = []
        all_rates_norm = []
        trial_rasters = []

        # Build rate arrays and trial spike lists (always)
        for neuron_idx, spikes in enumerate(n_spike_times_filtered):
            neuron_id = n_cluster_index_filtered[neuron_idx]
            if spikes.size == 0:
                continue
            aligned_counts = []
            for onset in beh_onsets:
                mask = (spikes >= onset + window[0]) & (spikes <= onset + window[1])
                rel_spikes = spikes[mask] - onset
                trial_rasters.append((rel_spikes, neuron_id))
                counts, _ = np.histogram(rel_spikes, bins=time_bins)
                aligned_counts.append(counts)
            if aligned_counts:
                mean_counts = np.mean(aligned_counts, axis=0) / bin_size
                if smooth:
                    mean_counts = smooth_trace(mean_counts, smooth_sigma)
                all_rates_raw.append(mean_counts)
                baseline_mask = (
                    (time_bins[:-1] >= baseline_window[0]) &
                    (time_bins[:-1] < baseline_window[1])
                )
                baseline_mean = np.mean(mean_counts[baseline_mask])
                baseline_std = np.std(mean_counts[baseline_mask]) if np.std(mean_counts[baseline_mask]) > 0 else 1e-9
                z_rate = (mean_counts - baseline_mean) / baseline_std
                if smooth:
                    z_rate = smooth_trace(z_rate, smooth_sigma)
                all_rates_norm.append(z_rate)

        n_trials_full = len(trial_rasters)

        # --- Single merged raster mode ---
        if only_raster_grouped and merged_subset_mode:
            trial_spike_data = []
            neuron_order_appearance = []
            for trial_idx, onset in enumerate(beh_onsets):
                merged_times, merged_colors = [], []
                for neuron_idx, spikes in enumerate(n_spike_times):
                    neuron_id = n_cluster_index[neuron_idx]
                    if neuron_id in n_cluster_index_subset:
                        mask = (spikes >= onset + window[0]) & (spikes <= onset + window[1])
                        rel_spikes = spikes[mask] - onset
                        if rel_spikes.size > 0:
                            color_idx = np.where(unique_neurons == neuron_id)[0][0]
                            if neuron_id not in neuron_order_appearance:
                                neuron_order_appearance.append(neuron_id)
                            merged_times.extend(rel_spikes)
                            merged_colors.extend([neuron_colors[color_idx]] * len(rel_spikes))
                earliest = np.min(merged_times) if merged_times else np.inf
                trial_spike_data.append((trial_idx, earliest, merged_times, merged_colors))
            trial_spike_data.sort(key=lambda x: x[1])

            fig, ax = plt.subplots(figsize=(10, 8))
            for new_y, (_, _, spikes_times, spikes_colors) in enumerate(trial_spike_data):
                ax.scatter(spikes_times, [new_y] * len(spikes_times),
                           color=spikes_colors, s=raster_marker_size, marker='.', alpha=0.8)
            if len(n_cluster_index_subset) < 10:
                handles = [Patch(color=neuron_colors[np.where(unique_neurons == nid)[0][0]], label=f"N{nid}")
                           for nid in neuron_order_appearance if nid in unique_neurons]
                ax.legend(handles=handles, title="Neuron IDs", bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.axvline(0, color='k', linestyle='--', alpha=0.7)
            ax.set_ylabel("Trial idx (sorted by earliest spike)")
            ax.set_xlabel("Time from onset (s)")
            ax.set_title(f"{beh} - Merged Subset Raster Sorted (n={len(beh_onsets)})")

            plt.tight_layout()
            safe_name = re.sub(r'[\\/*?:"<>|]', "_", str(beh))
            plt.savefig(os.path.join(output_folder, f"{safe_name}.png"), dpi=300)
            #plt.savefig(os.path.join(output_folder, f"{safe_name}.svg"))
            plt.close(fig)  # ✅ Release memory
            continue

        # --- Full 2×2 plotting mode ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
        fig.suptitle(f"Behavior: {beh}", fontsize=14, fontweight="bold")
        ax0, ax1, ax2, ax3 = axes[0,0], axes[1,0], axes[0,1], axes[1,1]

        # Subplot 1: Raw firing rates
        if plot_all_neurons:
            for neuron_idx, rate in enumerate(all_rates_raw):
                ax0.plot(t_centers, rate, color=neuron_colors[neuron_idx], alpha=raw_alpha)
        else:
            ax0.plot(t_centers, np.mean(all_rates_raw, axis=0), color='b', alpha=raw_alpha)
        ax0.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax0.set_ylabel("Firing rate (Hz)")
        ax0.set_title("Raw Firing Rates")

        # Subplot 2: Normalized rates
        if plot_all_neurons:
            for neuron_idx, rate in enumerate(all_rates_norm):
                ax1.plot(t_centers, rate, color=neuron_colors[neuron_idx], alpha=0.8)
        else:
            ax1.plot(t_centers, np.mean(all_rates_norm, axis=0), color='r', alpha=0.8)
        ax1.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax1.set_ylabel("Norm. Rate (z-score)")
        ax1.set_title("Normalized Firing Rates")
        ax1.set_xlabel("Time from onset (s)")

        # Subplot 3: Peak-time sorted raster
        peak_times_trials = [np.min(times) if len(times) > 0 else np.inf for times, _ in trial_rasters]
        peak_sorted = [trial_rasters[i] for i in np.argsort(peak_times_trials)]
        sx, sy, sc = [], [], []
        peak_sorted_neuron_order = []
        for idx, (times, nid) in enumerate(peak_sorted):
            if nid not in peak_sorted_neuron_order:
                peak_sorted_neuron_order.append(nid)
            color_idx = np.where(unique_neurons == nid)[0][0]
            sx.extend(times)
            sy.extend([idx] * len(times))
            sc.extend([neuron_colors[color_idx]] * len(times))
        ax2.scatter(sx, sy, c=sc, s=raster_marker_size, marker='.', alpha=0.8)
        ax2.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Trial idx")
        ax2.set_title(f"Raster - Peak-Time Sorted (n={n_trials_full})")
        if n_cluster_index_subset is not None and len(n_cluster_index_subset) < 10:
            handles = [Patch(color=neuron_colors[np.where(unique_neurons == nid)[0][0]], label=f"N{nid}")
                       for nid in peak_sorted_neuron_order if nid in unique_neurons]
            ax2.legend(handles=handles, title="Neuron IDs", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Subplot 4: Neuron-grouped raster
        sx, sy, sc = [], [], []
        neuron_group_appearance = []
        trial_index = 0
        for neuron_id in unique_neurons:
            neuron_group_appearance.append(neuron_id)
            neuron_trials = [(times, nid) for times, nid in trial_rasters if nid == neuron_id]
            peak_within = [np.min(times) if len(times) > 0 else np.inf for times,_ in neuron_trials]
            sorted_trials = [neuron_trials[i] for i in np.argsort(peak_within)]
            for times, nid in sorted_trials:
                color_idx = np.where(unique_neurons == nid)[0][0]
                sx.extend(times)
                sy.extend([trial_index] * len(times))
                sc.extend([neuron_colors[color_idx]] * len(times))
                trial_index += 1
        ax3.scatter(sx, sy, c=sc, s=raster_marker_size, marker='.', alpha=0.8)
        ax3.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax3.set_ylabel("Trial idx")
        ax3.set_title(f"Raster - Neuron Grouped (n={n_trials_full})")
        ax3.set_xlabel("Time from onset (s)")
        if n_cluster_index_subset is not None and len(n_cluster_index_subset) < 10:
            handles = [Patch(color=neuron_colors[np.where(unique_neurons == nid)[0][0]], label=f"N{nid}")
                       for nid in neuron_group_appearance if nid in unique_neurons]
            ax3.legend(handles=handles, title="Neuron IDs", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0,0,1,0.96])
        safe_beh = re.sub(r'[\\/*?:"<>|]', "_", str(beh))
        plt.savefig(os.path.join(output_folder, f"{safe_beh}.png"), dpi=300)
        plt.close(fig)  # ✅ Release memory

if __name__ == '__main__':
#    freeze_support() # Recommended for multiprocessing robustness on some platforms

    for session in sessions:
        paths=pp.get_paths(animal, session)   
        print(paths['sorting_spikes'])
        
        
        print(f"session: {session}\n")
        
        velocity,locations,node_names,bottom_node_names,frame_index_s,frames_dropped,distance2shelter,bottom_distance_to_shelter=hf.load_specific_preprocessed_data (animal, session, 'tracking',load_pd=False )
       
        
        distance2shelter=distance2shelter[:,3].data
        if distance2shelter[0]==np.inf:
            distance2shelter[0]=distance2shelter[1]
        
        #except:
         #   distance2shelter=distance2shelter.data.squeeze()
        
        if np.nanmax(distance2shelter)>300:
             values = paths['Cm2Pixel_xy'].split(' ')
             # Convert each value to np.float16
             Cm2Pixel_xy = [np.float32(value) for value in values]
             distance2shelter=distance2shelter*Cm2Pixel_xy[0]
             distance2shelter=distance2shelter-np.nanmin(distance2shelter)#DEBUG ONLY!!!!!!!!!!!!!
             max_dist=np.nanmax(distance2shelter)
             max_vel=np.nanmax(velocity)
        
        #target_bs = target_bs_0.copy()
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
        
        behaviour_counts = (
            behaviour[behaviour["start_stop"].isin(["START", "POINT"])]
            .groupby("behaviours")
            .size()
        )
        print(session)
        print(behaviour_counts)
        #continue 
        
       
        locations=locations.squeeze()
        vframerate=len(frame_index_s)/max(frame_index_s)
        min_length=np.nanmin([len(velocity),len(frame_index_s)])
        
        frame_index_s = frame_index_s[:min_length]
        distance2shelter = distance2shelter[:min_length]
    
        velocity = velocity[0:min_length]
        locations = locations[0:min_length]
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
        
    
        # print('inserting random baseline time periods to behavior dataframe \n')
        # event_type='baseline_random'
        # behaviour=hf.insert_event_times_into_behavior_df(behaviour,framerate,event_type,behavior_name,behavior_type,**kwargs)    
        
        # print('inserting speed threshold crossing periods to behavior dataframe \n')
        # event_type="speed"
        # behavior_name="speed"
        # behavior_type="Speed_threshold_crossing"
        # kwargs['diff_time_s']=5
        # kwargs["Threshold"]=45#np.percentile(velocity,95) #around 20cm/s
        # behaviour=hf.insert_event_times_into_behavior_df(behaviour,framerate,event_type,behavior_name,behavior_type,**kwargs)
        
        
            
        # plot_sta_per_behavior(
        #     behaviour,
        #     n_spike_times,
        #     window=(-2, 3),
        #     bin_size=0.02,
        #     baseline_window=(-1.0, -0.1),
        #     raster_marker_size=4,
        #     plot_all_neurons=True,
        #     output_folder=session_savepath,
        #     show_avg_peak_lines=False,
        #     raw_alpha=0.3,
        #     smooth=True,
        #     smooth_sigma=1.0,
        #     only_raster_grouped=True,
        #     n_cluster_index = n_cluster_index,
        #     n_cluster_index_subset = [344,347]
        # )
        
        
            
        print(f"calculating intananeous firing rate \n" )
        iFR,iFR_array,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate
        
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
            
           
    #######    # Reduce selection to subsets.
        unique_behaviours = behaviour.behaviours.unique()
        if len(target_bs) == 0:  # If no target behavior is specified, take all present
            target_bs = np.sort(unique_behaviours)
       
            
        #Target neurons
        target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
        
        if plot_waveforms_by_column==True:
            
            analyzer_path = Path(paths['preprocessed']) / 'sorting_analyzer_results'
            try:
                sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
            except:
                print(f"session {session} analyzer could not be loaded")
                
            if plot_template_metrics==True:
                plottemplate_metrics_from_PAG_columns(sorting_analyzer,savepath_0)
            hf.set_page_format('A2')
            for  region in target_regions:
                
                region_index = np.where(n_region_index==region)
                if region_index[0].size == 0:
                    continue
                unit_ids = n_cluster_index[region_index]
                unit_idx= region_index
                print(region)
                print(unit_ids)
                plot_waveforms_from_PAG_columns(sorting_analyzer,unit_ids,region,savepath_0)
            
        
        if plot_trajectory==True:
            hf.set_page_format('A4')
            plot_neuron_behavioral_tuning_subplots_by_trial_time_all(n_spike_times, locations, frame_index_s, velocity, n_cluster_index,  savepath_0, behaviour, session)
#            for idx,neuron_ID in tqdm(enumerate(np.sort(n_cluster_index[target_neurons_ind])),desc="vizualiasing spikes",colour="green"):
            
 #             if neuron_ID in([344,347,691, 673, 674, 671, 589, 365, 577]):
  #              print(neuron_ID)                             
                #plot_neuron_behavioral_tuning_subplots_by_trial_time(n_spike_times, locations, frame_index_s, velocity, n_cluster_index, neuron_ID, savepath_0, behaviour, session)
   #             plt.close('all')
               # IPython.embed()
                #plot_neuron_behavioral_tuning_subplots_by_trial_sample(n_spike_times, locations, frame_index_s, velocity, n_cluster_index, neuron_ID, savepath_0, behaviour, session)
               # plot_neuron_behavioral_tuning_subplots_speed(n_spike_times, locations, frame_index_s, velocity, n_cluster_index, neuron_ID, savepath_0, behaviour, session)
                #IPython.embed()
        
        
        
    #     print(f"recalculating ndata \n")
    #     # spike_res=0.001
    #     # FR_res=0.1
    #     # n_time_index, ndata, firing_rate_bins_time,firing_rates,neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds=hf.recalculate_ndata_firing_rates2(n_spike_times,
    #     # bin_size=spike_res, firing_rate_bin_size=FR_res)
        
        
    #     spike_res=0.001
    #     FR_res=0.02
    #     res=0.001
    # #    print(f"before: {len(n_spike_times)=}, {np.shape(ndata)=}")
    #     n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=res)