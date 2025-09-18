# -*- coding: utf-8 -*-
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import plottingFunctions as pf
plt.rcParams.update({
'font.size': 16,            # controls default text sizes
'axes.titlesize': 16,       # fontsize of the axes title
'axes.labelsize': 14,       # fontsize of the x and y labels
'xtick.labelsize': 14,
'ytick.labelsize': 14,
'legend.fontsize': 14,
'figure.titlesize': 20      # fontsize of the figure title
})    

import IPython
import polars as pl
import pandas as pd
import numpy as np
import plottingFunctions as pf
import matplotlib.pyplot as plt
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
        
        
def plot_concatanated_PSTHs(df,neuron_index=0,save_path=None):
    # --- Assume 'df' is your pre-existing DataFrame ---
    # Example DataFrame structure (replace with your actual df)
    # Simulating the data structure based on your description
    #if 'df' not in locals(): # Check if df exists, if not, create a dummy one
        # print("Creating dummy DataFrame 'df' for demonstration.")
        # n_neurons = 5 # Reduced for example
        # event_cols = ['approach', 'attack', 'bed_retrieve', 'escape', 'pup_grab',
        #               'pup_retrieve', 'pup_run', 'pursuit', 'gas_escape', 'loom',
        #               'pup_drop', 'startle', 'turn']
        # data = {}
        # data['cluster_id'] = [f'neuron_{i}' for i in range(n_neurons)]
        # data['max_site'] = np.random.randint(1, 5, n_neurons)
        # data['region'] = np.random.choice(['A', 'B', 'C'], n_neurons)
    
        # for col in event_cols:
        #     col_data = []
        #     for i in range(n_neurons):
        #         # Each neuron has a list of trials for this event
        #         num_trials = np.random.randint(5, 15) # Variable number of trials
        #         trials_list = []
        #         for _ in range(num_trials):
        #             # Spikes centered around 0, from -5 to 5
        #             # Make spike rate dependent on event type and neuron for variety
        #             center_shift = event_cols.index(col) * 0.1 - 0.6 + (i*0.2)
        #             num_spikes = np.random.randint(0, 20 + event_cols.index(col) % 5)
        #             spikes = np.random.normal(loc=center_shift, scale=1.0 + (i*0.1), size=num_spikes)
        #             spikes = spikes[(spikes >= -5) & (spikes <= 5)] # Ensure within range
        #             trials_list.append(np.sort(spikes))
        #         col_data.append(trials_list)
        #     data[col] = col_data
    
        # df = pd.DataFrame(data)
        # # Ensure the index includes the required keys
        # df = df[['cluster_id', 'max_site', 'region'] + event_cols]
        # print(f"Dummy DataFrame created with columns: {df.columns.tolist()}")
        # print(f"Shape: {df.shape}")
    
    
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
                    print(f"Skipping column '{col}' for neuron {unit_IDs[neuron_index]}: Data is not a list of arrays/lists.")
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
        print(f"Plotting for Neuron: {df.loc[neuron_index, 'cluster_id']} (Index: {neuron_index})")
        print(f"Events to plot: {event_columns_to_plot}")
    
    unit_id_str = str({unit_IDs[neuron_index]}) # Get unit ID for titles/saving
    print(f"Plotting for unit: {unit_id_str} (Index: {neuron_index})")
    
    time_start = -5.0 # seconds
    time_end = 5.0   # seconds
    bin_width = 0.1 # seconds
    
    # --- Figure Layout ---
    n_events = len(event_columns_to_plot)
    n_cols_grid = 4  # How many event pairs per row in the figure
    n_rows_grid = math.ceil(n_events / n_cols_grid)
    
    fig_rows = n_rows_grid * 3 # *** UPDATED: 3 rows per event ***
    fig_cols = n_cols_grid

    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * 4.5, fig_rows * 2.3), # *** Adjusted figsize ***
                             sharex=False, sharey=False, squeeze=False)
    # # Each event needs 2 rows (PSTH, Raster)
    # fig_rows = n_rows_grid * 2
    # fig_cols = n_cols_grid
    
    # # Create the figure and the grid of axes
    # # Adjust figsize as needed; this might need to be quite large
    # fig, axes = plt.subplots(fig_rows, fig_cols,
    #                          figsize=(fig_cols * 4, fig_rows * 2.5), # Adjust size here
    #                          sharex=False, # We will share X within pairs manually
    #                          sharey=False, # Y axes will likely have different scales
    #                          squeeze=False) # Ensure axes is always 2D array
    fig.suptitle(f"IFR, PSTH & Raster Plots for Unit: {unit_id_str}", fontsize=16)
#    fig.suptitle(f"PSTH & Raster Plots for Neuron: {unit_IDs[neuron_index]}")
    
    # Define time bins for histogram (same for all plots)
    bins = np.arange(time_start, time_end + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    
    for i, event_column in enumerate(event_columns_to_plot):
        grid_row_base = (i // n_cols_grid) * 3 # *** UPDATED: 3 rows per event ***
        grid_col = i % n_cols_grid

        if grid_row_base + 2 >= fig_rows or grid_col >= fig_cols: # *** UPDATED: Check bounds for 3 rows ***
            print(f"Warning: Calculated axes index out of bounds for event '{event_column}'. Skipping.")
            continue

        # *** Define all 3 axes per event ***
        ax_ifr = axes[grid_row_base, grid_col]
        ax_psth = axes[grid_row_base + 1, grid_col]
        ax_raster = axes[grid_row_base + 2, grid_col]

        # Optional: Apply styling
        pf.remove_axes(ax_ifr)
        pf.remove_axes(ax_psth)
        pf.remove_axes(ax_raster)

        # --- Data Extraction (same as before) ---
        try:
            neuron_spike_data_list = df.loc[neuron_index, event_column]
            if not isinstance(neuron_spike_data_list, list): neuron_spike_data_list = []
            valid_trials = []
            for trial in neuron_spike_data_list:
                 try:
                     numeric_trial = np.array(trial, dtype=float).flatten()
                     numeric_trial = numeric_trial[np.isfinite(numeric_trial)]
                     valid_trials.append(numeric_trial)
                 except (ValueError, TypeError): pass
            neuron_spike_data_list = valid_trials
            if not neuron_spike_data_list:
                all_spikes = np.array([]); num_trials = 0
            else:
                non_empty_trials = [trial for trial in neuron_spike_data_list if trial.size > 0]
                if non_empty_trials: all_spikes = np.concatenate(non_empty_trials)
                else: all_spikes = np.array([])
                num_trials = len(neuron_spike_data_list)
        except Exception as e:
            print(f"Error processing data for event '{event_column}': {e}")
            all_spikes = np.array([]); num_trials = 0; neuron_spike_data_list = []

        # --- Calculate IFR Data --- *** NEW SECTION ***
        all_ifr_times_list = []
        all_ifr_values_list = []
        if num_trials > 0:
            for trial_spikes in neuron_spike_data_list:
                if trial_spikes.size >= 2:
                    isis = np.diff(trial_spikes)
                    # Calculate IFR, handle potential zero ISI
                    ifr_values = 1.0 / np.maximum(isis, 1e-9) # Avoid division by zero
                    # Times associated with IFR values are the times of the second spike in each pair
                    ifr_times = trial_spikes[1:]
                    # Append to lists for global calculation
                    all_ifr_times_list.append(ifr_times)
                    all_ifr_values_list.append(ifr_values)

        average_ifr = np.zeros_like(bin_centers) # Initialize average IFR array
        if all_ifr_times_list: # Check if any IFR data was generated
            # Concatenate lists into single arrays
            all_ifr_times = np.concatenate(all_ifr_times_list)
            all_ifr_values = np.concatenate(all_ifr_values_list)

            if all_ifr_times.size > 0:
                # Calculate sum of IFRs in each bin
                ifr_sum_in_bin, _ = np.histogram(all_ifr_times, bins=bins, weights=all_ifr_values)
                # Calculate count of spikes (with preceding ISI) in each bin
                ifr_spike_counts_in_bin, _ = np.histogram(all_ifr_times, bins=bins)
                # Calculate average IFR, avoiding division by zero
                average_ifr = np.divide(ifr_sum_in_bin, ifr_spike_counts_in_bin,
                                        out=np.zeros_like(ifr_sum_in_bin), where=ifr_spike_counts_in_bin!=0)


        # --- Plot 1: Average Instantaneous Firing Rate (IFR PSTH) --- *** NEW PLOT ***
        ax_ifr.axvline(0, color='red', linestyle='--', linewidth=1, zorder=5)
        ax_ifr.plot(bin_centers, average_ifr, color='steelblue', linewidth=1.5) # Use a different color
        ax_ifr.set_ylim(bottom=0) # Start y-axis at 0
        ax_ifr.set_ylabel('Avg IFR (Hz)')
        ax_ifr.set_title(event_column, fontsize=14) # Title on the top plot
        ax_ifr.grid(axis='y', linestyle=':', alpha=0.7)
        ax_ifr.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-ticks


        # --- Plot 2: Regular Firing Rate (PSTH) --- (Logic mostly same as before)
        ax_psth.axvline(0, color='red', linestyle='--', linewidth=1, zorder=5)
        max_trial_rate = 0 # Reset for this event
        if num_trials > 0:
            # Plot Individual Trial PSTHs (Grey Lines)
            for trial_spikes in neuron_spike_data_list:
                if trial_spikes.size > 0:
                    trial_counts, _ = np.histogram(trial_spikes, bins=bins)
                    trial_firing_rate = trial_counts / bin_width
                    ax_psth.plot(bin_centers, trial_firing_rate, color='grey', alpha=0.3, linewidth=0.5, zorder=1)
                    max_trial_rate = max(max_trial_rate, np.max(trial_firing_rate))

            # Plot Average PSTH (Black Line)
            if all_spikes.size > 0:
                counts, _ = np.histogram(all_spikes, bins=bins)
                firing_rate = counts / (num_trials * bin_width)
                #ax_psth.plot(bin_centers, firing_rate, color='black', linewidth=1.5, zorder=10)
                #ax_psth.bar(bin_centers, firing_rate, width=bin_width, align='center', alpha=0.8)
                current_max_rate = max(max_trial_rate, np.max(firing_rate) if firing_rate.size > 0 else 0)
                ax_psth.set_ylim(bottom=0, top=max(1, current_max_rate * 1.1))
            else:
                ax_psth.plot(bin_centers, np.zeros_like(bin_centers), color='black', linewidth=1.5, zorder=10)
                ax_psth.set_ylim(bottom=0, top=1)
        else:
             ax_psth.plot(bin_centers, np.zeros_like(bin_centers), color='black', linewidth=1.5, zorder=10)
             ax_psth.set_ylim(bottom=0, top=1)

        ax_psth.set_ylabel('Rate (Hz)')
        ax_psth.grid(axis='y', linestyle=':', alpha=0.7)
        ax_psth.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-ticks


        # --- Plot 3: Spike Raster Plot --- (Same as before)
        if num_trials > 0:
            plot_data = [trial for trial in neuron_spike_data_list if len(trial) > 0]
            plot_indices = [idx for idx, trial in enumerate(neuron_spike_data_list) if len(trial) > 0]
            if plot_data:
                 ax_raster.eventplot(plot_data, colors='black', lineoffsets=plot_indices,
                                    linelengths=0.8, linewidths=0.5)
            ax_raster.set_ylim(-1, num_trials)
            if num_trials <= 10: ax_raster.set_yticks(np.arange(0, num_trials))
            else: ax_raster.set_yticks(np.linspace(0, max(0, num_trials-1), 5, dtype=int))
        else:
            ax_raster.set_ylim(-1, 1); ax_raster.set_yticks([])

        ax_raster.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_raster.set_ylabel('Trial')
        ax_raster.grid(axis='y', linestyle=':', alpha=0.7)

        # --- Axis Sharing & Limits --- *** UPDATED for 3 plots ***
        ax_ifr.sharex(ax_psth)
        ax_psth.sharex(ax_raster)
        ax_raster.set_xlim(time_start, time_end) # Set limits (affects all three now)

        # --- X-axis Label Handling --- *** UPDATED for 3 plots ***
        last_event_row_index = (n_events - 1) // n_cols_grid
        current_event_row_index = i // n_cols_grid
        is_in_last_plotted_row = (current_event_row_index == last_event_row_index)

        if is_in_last_plotted_row:
            ax_raster.set_xlabel('Time (s)')
            ax_raster.tick_params(axis='x', which='both', labelbottom=True)
        # Tick labels for ax_ifr and ax_psth are already turned off above

        # --- Y-axis Label Handling (Outer Columns) --- *** UPDATED for 3 plots ***
        if grid_col > 0:
            ax_ifr.tick_params(axis='y', which='both', labelleft=False)
            ax_psth.tick_params(axis='y', which='both', labelleft=False)
            ax_raster.tick_params(axis='y', which='both', labelleft=False)


    # --- Clean up unused axes --- *** UPDATED for 3 plots ***
    for i in range(n_events, n_rows_grid * n_cols_grid):
        grid_row_base = (i // n_cols_grid) * 3
        grid_col = i % n_cols_grid
        # Check bounds before attempting to access axes
        if grid_row_base + 2 < fig_rows and grid_col < fig_cols: # Check up to the third row
            axes[grid_row_base, grid_col].axis('off')      # Turn off IFR axis
            axes[grid_row_base + 1, grid_col].axis('off')  # Turn off PSTH axis
            axes[grid_row_base + 2, grid_col].axis('off')  # Turn off Raster axis

   
    # --- Final Adjustments & Save ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.96], pad=1.0, h_pad=1.5, w_pad=1.0) # Adjust padding

    if save_path:
        try:
            safe_unit_id = "".join([c for c in unit_id_str if c.isalnum() or c in ('_', '-')]).rstrip()
            filename = f"{safe_unit_id}_IFR_PSTH_Raster.png" # Updated filename
            full_save_path = save_path / filename
            fig.savefig(full_save_path, dpi=300) # Optional: increase dpi for higher resolution
            print(f"Saved plot to: {full_save_path}")
        except Exception as e:
            print(f"Error saving figure for unit {unit_id_str}: {e}")
    else:
        print("No save_path provided. Figure not saved.")
        # plt.show() # Optionally show if not saving

    plt.close(fig) # Close figure


# Example Usage (assuming df is loaded and savepath_0 is defined):
# savepath_root = Path("./plots_ifr") # Example save path
# plot_cells(df, savepath_root)
# Or plot a single neuron:
# plot_concatanated_PSTHs(df, neuron_index=5, save_path=savepath_root)
    
#     # --- Loop Through Events and Plot ---
#     for i, event_column in enumerate(event_columns_to_plot):
#         # Calculate the grid position for this event pair
#         grid_row_base = (i // n_cols_grid) * 2
#         grid_col = i % n_cols_grid
    
#         # Ensure calculated indices are within the bounds of the axes array
#         if grid_row_base + 1 >= fig_rows or grid_col >= fig_cols:
#             print(f"Warning: Calculated axes index out of bounds for event '{event_column}'. Skipping.")
#             continue
#         ax_ifr = axes[grid_row_base, grid_col]
#         ax_psth = axes[grid_row_base + 1, grid_col]
#         ax_raster = axes[grid_row_base + 2, grid_col]
        
# #        ax_psth = axes[grid_row_base, grid_col]
# #        ax_raster = axes[grid_row_base + 1, grid_col]
#         pf.remove_axes(ax_ifr)
#         pf.remove_axes(ax_psth)
#         pf.remove_axes(ax_raster)
        
         
    
#         # --- Data Extraction for the current event ---
#         # --- Data Extraction (same as before) ---
#         try:
#             neuron_spike_data_list = df.loc[neuron_index, event_column]
#             if not isinstance(neuron_spike_data_list, list): neuron_spike_data_list = []
#             valid_trials = []
#             for trial in neuron_spike_data_list:
#                  try:
#                      numeric_trial = np.array(trial, dtype=float).flatten()
#                      numeric_trial = numeric_trial[np.isfinite(numeric_trial)]
#                      valid_trials.append(numeric_trial)
#                  except (ValueError, TypeError): pass
#             neuron_spike_data_list = valid_trials
#             if not neuron_spike_data_list:
#                 all_spikes = np.array([]); num_trials = 0
#             else:
#                 non_empty_trials = [trial for trial in neuron_spike_data_list if trial.size > 0]
#                 if non_empty_trials: all_spikes = np.concatenate(non_empty_trials)
#                 else: all_spikes = np.array([])
#                 num_trials = len(neuron_spike_data_list)
#         except Exception as e:
#             print(f"Error processing data for event '{event_column}': {e}")
#             all_spikes = np.array([]); num_trials = 0; neuron_spike_data_list = []
#         # --- Calculate IFR Data --- *** NEW SECTION ***
#         all_ifr_times_list = []
#         all_ifr_values_list = []
#         if num_trials > 0:
#             for trial_spikes in neuron_spike_data_list:
#                 if trial_spikes.size >= 2:
#                     isis = np.diff(trial_spikes)
#                     # Calculate IFR, handle potential zero ISI
#                     ifr_values = 1.0 / np.maximum(isis, 1e-9) # Avoid division by zero
#                     # Times associated with IFR values are the times of the second spike in each pair
#                     ifr_times = trial_spikes[1:]
#                     # Append to lists for global calculation
#                     all_ifr_times_list.append(ifr_times)
#                     all_ifr_values_list.append(ifr_values)

#         average_ifr = np.zeros_like(bin_centers) # Initialize average IFR array
#         if all_ifr_times_list: # Check if any IFR data was generated
#             # Concatenate lists into single arrays
#             all_ifr_times = np.concatenate(all_ifr_times_list)
#             all_ifr_values = np.concatenate(all_ifr_values_list)

#             if all_ifr_times.size > 0:
#                 # Calculate sum of IFRs in each bin
#                 ifr_sum_in_bin, _ = np.histogram(all_ifr_times, bins=bins, weights=all_ifr_values)
#                 # Calculate count of spikes (with preceding ISI) in each bin
#                 ifr_spike_counts_in_bin, _ = np.histogram(all_ifr_times, bins=bins)
#                 # Calculate average IFR, avoiding division by zero
#                 average_ifr = np.divide(ifr_sum_in_bin, ifr_spike_counts_in_bin,
#                                         out=np.zeros_like(ifr_sum_in_bin), where=ifr_spike_counts_in_bin!=0)

    
    
    #     # --- Plot 1: Firing Rate (PSTH) ---
    #     if num_trials > 0 and all_spikes.size > 0: # Ensure there are spikes to calculate rate
    #         counts, _ = np.histogram(all_spikes, bins=bins)
    #         firing_rate = counts / (num_trials * bin_width)
    #         ax_psth.bar(bin_centers, firing_rate, width=bin_width, align='center', alpha=0.8)
    #         ax_psth.set_ylim(bottom=0) # Ensure y starts at 0
    #     else:
    #          # Plot empty PSTH if no trials or no spikes
    #         ax_psth.bar(bin_centers, np.zeros_like(bin_centers), width=bin_width, align='center')
    #         ax_psth.set_ylim(bottom=0, top=1) # Give some minimal height
    
    #     ax_psth.axvline(0, color='red', linestyle='--', linewidth=1)
    #     ax_psth.set_ylabel('Rate (Hz)')
    #     ax_psth.grid(axis='y', linestyle=':', alpha=0.7)
    #     # Hide x-axis labels and ticks for PSTH plots (will be shared with raster below)
    #     ax_psth.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    
    #     # --- Plot 2: Spike Raster Plot ---
    #     if num_trials > 0:
    #          # Filter neuron_spike_data_list for eventplot robustness if needed
    #         plot_data = [trial for trial in neuron_spike_data_list if len(trial) > 0] # Only plot trials with spikes
    #         plot_indices = [idx for idx, trial in enumerate(neuron_spike_data_list) if len(trial) > 0]
    #         if plot_data: # Check if there's anything left to plot
    #              ax_raster.eventplot(plot_data, colors='black', lineoffsets=plot_indices, # Use original indices
    #                                 linelengths=0.8, linewidths=0.5)
    #         ax_raster.set_ylim(-1, num_trials) # Keep Y limit based on original number of trials
    #         # Adjust Y ticks based on num_trials
    #         if num_trials <= 10:
    #              ax_raster.set_yticks(np.arange(0, num_trials))
    #         else:
    #              ax_raster.set_yticks(np.linspace(0, max(0, num_trials-1), 5, dtype=int)) # Fewer Y ticks for many trials
    #     else:
    #         ax_raster.set_ylim(-1, 1)
    #         ax_raster.set_yticks([])
    
    
    #     ax_raster.axvline(0, color='red', linestyle='--', linewidth=1)
    #     ax_raster.set_title(event_column, fontsize=10) # Title on the raster plot
    #     ax_raster.set_ylabel('Trial')
    #     ax_raster.grid(axis='y', linestyle=':', alpha=0.7)
    
    #     # --- * CORRECTED AXIS SHARING * ---
    #     ax_psth.sharex(ax_raster)
    #     # --- * ------------------------ * ---
    
    #     # Set shared x-limits for the pair (setting on one affects the other now)
    #     ax_raster.set_xlim(time_start, time_end)
    
    #     # Determine if this subplot is in the last row *that contains plots*
    #     # This logic needs to be correct even if the last row isn't full
    #     last_event_row_index = (n_events - 1) // n_cols_grid
    #     current_event_row_index = i // n_cols_grid
    #     is_in_last_plotted_row = (current_event_row_index == last_event_row_index)
    
    #     if is_in_last_plotted_row:
    #         ax_raster.set_xlabel('Time (s)')
    #         # Ensure tick labels are visible if they were turned off by sharing
    #         ax_raster.tick_params(axis='x', which='both', labelbottom=True)
    #     # else: # Explicitly turn off if not bottom row (sharex might have turned it on)
    #     #      ax_raster.tick_params(axis='x', which='both', labelbottom=False) #This is handled by ax_psth tick_params + sharex
    
    
    #     # Remove y-labels for plots not in the first column to reduce clutter
    #     if grid_col > 0:
    #         ax_psth.tick_params(axis='y', which='both', labelleft=False)
    #         ax_raster.tick_params(axis='y', which='both', labelleft=False)
    
    
    # # --- Clean up unused axes ---
    # for i in range(n_events, n_rows_grid * n_cols_grid):
    #     grid_row_base = (i // n_cols_grid) * 2
    #     grid_col = i % n_cols_grid
    #     # Check bounds before attempting to access axes
    #     if grid_row_base + 1 < fig_rows and grid_col < fig_cols:
    #         axes[grid_row_base, grid_col].axis('off') # Turn off PSTH axis
    #         axes[grid_row_base + 1, grid_col].axis('off') # Turn off Raster axis
    
    
    # --- Final Adjustments ---
    #plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout (leave space for suptitle)
    #plt.show()
    # fig.tight_layout(rect=[0, 0.03, 1, 0.96],pad=2)
    # fig.canvas.draw()  # Force update to the figure
    
    # fig.savefig(save_path / f"{unit_IDs[neuron_index]}_concat_PSTH.png")
    # plt.close('all')
    
       
    

# plot_concatanated_PSTHs(df=df,neuron_index=0,save_path=savepath_0)    