import IPython
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.gridspec as gridspec
import math
from matplotlib.gridspec import GridSpec
import os
#%%Plotting

def show_loom_in_video_clip(frame_num, stim_frame, vframerate, loom_loc, ax):
    """
    stim frame: when does loom happen?
    frame: the actual pixel data from that frame
    frame_num: what is th enumber of the current frame?
    loom_loc: x, y
    """
    
    # get the radius of the expanding circle
    loom_length= .63*vframerate #seconds --> frames
    loom_pause= .5*vframerate #seconds --> frames
    loom_size=10
    
    i=0 # loom num
    stim=0
    while not stim and i<5:
        
        stim= ((frame_num - stim_frame) < (1+i)*loom_length+i*loom_pause) * (frame_num >= (stim_frame+i*loom_length+i*loom_pause))
            # frame is < than future loomstart                             # frame is > than past loomstart
        i+=1
    
    if stim:
        i-=1
        past_frame=stim_frame+i*loom_length+i*loom_pause
        radius=loom_size * (frame_num - past_frame) +10

        circle=plt.Circle(loom_loc, radius, edgecolor='k', facecolor='k', alpha=.7)
        ax.add_patch(circle)
    

def make_window(image, centre, window_size):
    """
    Create a fixed-size window around a given point in an image using vectorized operations.

    Parameters:
    - image: 2D numpy array representing the image.
    - point: Tuple (x, y) representing the coordinates of the point.
    - window_size: Size of the window (default is 100x100 pixels).

    Returns:
    - window: 2D numpy array representing the fixed-size window around the point.
    """

    # Extract image dimensions
    img_y, img_x = image.shape

    # Calculate window boundaries
    x_ctr, y_ctr = centre
    half_window = window_size // 2
    
    #are you at an edge?
    if (x_ctr-half_window<0) or (x_ctr+half_window>img_x):
        if x_ctr-half_window<0: #left edge
            x_ctr=half_window
        elif  x_ctr+half_window>img_x: #right_edge
            x_ctr=img_x-half_window
        
        
    if y_ctr-half_window<0 or y_ctr+half_window>img_y:
        if y_ctr-half_window<0: #top edge
            y_ctr=half_window
        elif  y_ctr+half_window>img_y: #bottom edge
            y_ctr=img_y-half_window
    
    x_min = x_ctr - half_window
    x_max = x_ctr + half_window
    y_min = y_ctr + half_window #the top of the image is at y=0
    y_max = y_ctr - half_window
    

    return int(x_min), int(x_max), int(y_min), int(y_max), (int(x_ctr),int(y_ctr))



def psth_end_of_event(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, 
         window=5, density_bins=.5, return_data=False,session=None):
    """   
    Parameters
    ----------
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin.
    n_time_index : array_like
        Time index from preprocessing.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing.
    all_start_stop : array_like
        Matrix (or vector) with event start (and stop for state events) times.
    axs : list of matplotlib.axes.Axes
        Contains three axes on which to plot: velocity, average firing, and raster.
    window : float, optional
        How many seconds before/after an event to plot. The default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. The default is 0.5.
    return_data : bool, optional
        If True, the function returns a pandas DataFrame (see details below).
        
    Returns
    -------
    If return_data is True, returns a dictionary with two DataFrames:
       - "Firing": DataFrame with columns ["Time_bin", "Firing_Rate"].
         Here, Time_bin is computed as the center of bins using density_bins.
       - "Velocity": DataFrame with columns ["Time_bin", "Avg_Velocity"].
         Here, Time_bin is computed as the center of 0.1-s bins.
         
    If return_data is False, nothing is returned.
    """
    
    # --- Prettify axes ---
    remove_axes(axs[2])
    remove_axes(axs[0], bottom=True)
    remove_axes(axs[1], bottom=True)
    # Containers for precomputed values:
    all_spikes = []       # List to store spike times per trial (aligned)
    all_vel = []          # List to store velocity trace per trial
    all_vel_times = []    # List to store velocity time points (aligned) per trial
    state_events = []     # List to store state-event shading info
    session_aligned_spike_times=[]
    session_per_trial=[]
   
    bins = np.arange(-window, window + density_bins, density_bins)
    time_per_bin = 0
    i = 0 
    point = False

    # Determine if the provided start-stop structure represents point versus state events
    if all_start_stop.shape == (2,):
        all_start_stop = [all_start_stop]
    elif len(all_start_stop.shape) == 1:
        point = True
    
    #spikes_mat = np.zeros((len_spiketimes, len(all_start_stop))) 
    for startstop in all_start_stop:
        # For the first trial, set the previous trial variables.
        if i == 0:
            previous_plotstop = 0
            previous_zero = 0
        
        if  point==False:  # State events
            plotzero = startstop[1]
            # If events occur too close to the previous event, just add shading.
            if (plotzero - window) < previous_plotstop:
                axs[2].barh(i - 2, startstop[1] - startstop[0], 
                             left=plotzero - previous_zero,
                             height=1, color='burlywood', alpha=.5)
                state_events.append({
                    "trial": i - 2,
                    "left": plotzero - previous_zero,
                    "width": startstop[1] - startstop[0],
                    "alpha": 0.5
                })
                continue                
            axs[2].barh(i + 1, startstop[1] - startstop[0], height=1, color='burlywood')
            state_events.append({
                "trial": i + 1,
                "left": 0,  # zero offset for normally separated events
                "width": startstop[1] - startstop[0],
                "alpha": 1.0
            })
        else:  # Point events
            plotzero = startstop
        
        plotstart = plotzero - window
        plotstop = plotzero + window

        # Collect spikes within the plotting window for this trial.
        window_ind = (n_time_index >= plotstart) & (n_time_index <= plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind = np.where(spikes_around_stimulus > 0)[0]
        spiketimes = n_time_index[window_ind][spikeind]
        session_aligned_spike_times.append(spiketimes)
        session_per_trial.append(session)
        

        # Align spikes relative to the event
        spiketimes -= plotzero

        axs[2].scatter(spiketimes, np.ones_like(spikeind) * (i + 1), c='teal', s=0.5)
        
        previous_plotstop = plotstop
        previous_zero = plotzero
        all_spikes.append(spiketimes)
       
        #spikes_mat[:, i] = spiketimes
        
        # Adjust for multiple spikes in the same bin.
        while np.sum(spikes_around_stimulus > 0):
            spikes_around_stimulus[spikes_around_stimulus > 0] -= 1
            app_ind = np.where(spikes_around_stimulus > 0)[0]
            app_times = n_time_index[window_ind][app_ind]
            all_spikes.append(app_times - plotzero)
           
           
        
        # Sanity check: all trials should cover the same number of time bins.
        if i == 0:
            num_timebins = np.sum(window_ind)
        else:
            if (np.sum(window_ind) - num_timebins) > 1:
                raise ValueError('Not all trials cover the same time window for Hz calculation')
        
        # Get velocity for this trial.
        velind = (frame_index_s > plotstart) & (frame_index_s < plotstop)
        all_vel_times.append(frame_index_s[velind] - plotzero)
        all_vel.append(velocity[velind])
        
        i += 1
        time_per_bin += density_bins

    axs[2].set_ylim((0, np.nanmax((12, i + 4))))
    axs[2].set_yticks(np.hstack((np.arange(0, i, 10), [i])))
    axs[2].set_xlim((-window, window))
    
    # --- Compute and plot average firing rate ---
    all_spikes_list = all_spikes
    all_spikes = np.hstack(all_spikes)
    sum_spikes, firing_rate_bins = np.histogram(all_spikes, bins)
    hz = sum_spikes / time_per_bin
    axs[1].set_xticks([])
    axs[1].bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
    axs[1].set_xlim((-window, window))
    
    # --- Plot velocity ---
    axs[0].set_xticks([])
    axs[0].set_xlim((-window, window))
    axs[0].set_ylim((0, 130))
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        axs[0].plot(plotveltime, plotvel, lw=0.5, c='grey')
      
    # Bin velocity with fixed 0.1-s bins.
    velocity_bin_size=0.1 #seconds
    velbins = np.arange(-window, window, velocity_bin_size)
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    avg_velocity = binned_values / np.maximum(binned_counts, 1)
    axs[0].plot(velbins[:-1], avg_velocity, c='orangered')
        
    for ax in axs:
        ax.axvline(0, linestyle='--', c='k')
    
    # --- Return results as dictionary if requested ---
    if return_data:
        velocity_bin_centers = (velbins[:-1] + velbins[1:]) / 2
        firing_centers= (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2
        # Build dictionary of precomputed variables.
        precomputed = {
        "all_spikes": all_spikes,
        "firing_rate_bins": firing_rate_bins,
        "firing_centers":firing_centers,
        "hz": hz,
        "all_vel_times": all_vel_times,
        "all_vel": all_vel,
        "velbins": velbins,
        "velocity_bin_centers":velocity_bin_centers,
        "avg_velocity": avg_velocity,
        "state_events": state_events,
        'session':session,
        'session_aligned_spike_times':session_aligned_spike_times,
        'session_per_trial':session_per_trial
        }
        # firing_df = pd.DataFrame({
        #     "Time_bin": (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2,
        #     "Firing_Rate": hz
        # })
       
        # velocity_df = pd.DataFrame({
        #     "Time_bin": velocity_bin_centers,
        #     "Avg_Velocity": avg_velocity
        # # })
        # precomputed["Firing"] = firing_df
        # precomputed["Velocity"] = velocity_df
    
        firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        
        psth_dict={
        'FR_Time_bin_center': firing_centers,
        'FR_Hz': hz}
        
        velocity_dict={
        'Velocity_Time_bin': vel_centers,
        'Avg_Velocity_cms': avg_velocity,
        'velocity_bin_size': velocity_bin_size,
        'binned_values': binned_values,
        'binned_counts':binned_counts        
        }
            
        meta_dict = {
            'behavior_start_stop_time_s': all_start_stop,
            'point':point}
            #'behavior_limits':None,
        
        
        raster_dict={
            'spikes_array': all_spikes_list
        }
        
        return_df= pl.DataFrame({
    'psth_dict': psth_dict,
    'velocity_dict':velocity_dict,
    'raster_dict':raster_dict,
    'meta_dict':meta_dict,
    'precomputed':precomputed},
            strict=False)
    
        
        return return_df
    
    
def plot_concatanated_PSTHs(df,neuron_index=0,save_path=None):
   
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
        remove_axes(ax_ifr)
        remove_axes(ax_psth)
        remove_axes(ax_raster)

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
        all_ifr_values_list = [0]
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
    
def psth_cond_iFR(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, 
         window=5, density_bins=.5, return_data=False, session=None, 
         align_to_end=False):
    """   
    Parameters
    ----------
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin.
    n_time_index : array_like
        Time index from preprocessing.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing.
    all_start_stop : array_like
        Matrix (or vector) with event start (and stop for state events) times.
    axs : list of matplotlib.axes.Axes
        Contains three axes on which to plot: velocity, average firing, and raster.
    window : float, optional
        How many seconds before/after an event to plot. The default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. The default is 0.5.
    return_data : bool, optional
        If True, the function returns a pandas DataFrame (see details below).
        
    Returns
    -------
    If return_data is True, returns a dictionary with two DataFrames:
       - "Firing": DataFrame with columns ["Time_bin", "Firing_Rate"].
         Here, Time_bin is computed as the center of bins using density_bins.
       - "Velocity": DataFrame with columns ["Time_bin", "Avg_Velocity"].
         Here, Time_bin is computed as the center of 0.1-s bins.
         
    If return_data is False, nothing is returned.
    """
    
    # --- Prettify axes --- 
    
    ax_velocity=axs[0]    
    ax_iFR = axs[1]
    ax_FR = axs[2]
    ax_raster = axs[3]

    # Optional: Apply styling
    remove_axes(ax_velocity, bottom=True)
    remove_axes(ax_iFR, bottom=True)
    remove_axes(ax_FR, bottom=True)
    remove_axes(ax_raster)

    # Containers for precomputed values:
    all_spikes = []       # List to store spike times per trial (aligned)
    all_vel = []          # List to store velocity trace per trial
    all_vel_times = []    # List to store velocity time points (aligned) per trial
    state_events = []     # List to store state-event shading info
    session_aligned_spike_times=[]
    session_per_trial=[]
    all_ifr_times_list = []
    all_ifr_values_list = []
    bins = np.arange(-window, window + density_bins, density_bins)
    time_per_bin = 0
    i = 0 
    point = False

    # Determine if the provided start-stop structure represents point versus state events
    if all_start_stop.shape == (2,):
        all_start_stop = [all_start_stop]
    elif len(all_start_stop.shape) == 1:
        point = True
       
    debug_counter=-1
    #spikes_mat = np.zeros((len_spiketimes, len(all_start_stop))) 
    for startstop in all_start_stop:
        debug_counter+=1
        if point==True:                    
            ref_time= startstop #doesn't matter where to align if only 1 value
        elif point==False:
            Duration = startstop[1] - startstop[0]
            if align_to_end==True:
                ref_time = startstop[1]  # Align to event end
                Duration= -Duration
            elif align_to_end==False:
                ref_time = startstop[0]  # Align to event start
            
            
        plotzero =ref_time
        
        # For the first trial, set the previous trial variables.
        if i == 0:
            previous_plotstop = 0
            previous_zero = 0
        
       
        if  point==False:  # State events
        # If events occur too close to the previous event, just add shading.
            if (plotzero - window) < previous_plotstop:
                
                ax_raster.barh(i - 2,Duration, 
                             left=plotzero - previous_zero,
                             height=1, color='burlywood', alpha=.5)
                state_events.append({
                    "trial": i - 2,
                    "left": plotzero - previous_zero,
                    "width": Duration,
                    "alpha": 0.5
                })
                continue
           
            ax_raster.barh(i + 1, Duration, height=1, color='burlywood')
            state_events.append({
                "trial": i + 1,
                "left": 0,  # zero offset for normally separated events
                "width": Duration,
                "alpha": 1.0
            })

            
        
        plotstart = plotzero - window
        plotstop = plotzero + window

        # Collect spikes within the plotting window for this trial.
        window_ind = (n_time_index >= plotstart) & (n_time_index <= plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind = np.where(spikes_around_stimulus > 0)[0]
        spiketimes = n_time_index[window_ind][spikeind]
        session_aligned_spike_times.append(spiketimes)
        session_per_trial.append(session)
        
        spiketimes -= ref_time
       
        
        
        ######### iFR ################
    
        if spiketimes.size >= 0:
                    isis = np.diff(spiketimes)
                    # Calculate IFR, handle potential zero ISI
                    ifr_values = 1.0 / np.maximum(isis, 1e-9) # Avoid division by zero
                    # Times associated with IFR values are the times of the second spike in each pair
                    ifr_times = spiketimes
                    # Append to lists for global calculation
                    all_ifr_times_list.append([ifr_times])
                    all_ifr_values_list.append([ifr_values])
        
    
        # --- Plot 1: Average Instantaneous Firing Rate (IFR PSTH) --- *** NEW PLOT ***
        bin_width=density_bins#0.1
        #bins = np.arange(plotstart, plotstop + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2
        average_ifr = np.zeros_like(bin_centers) # Initialize average IFR array
        
        
            # Concatenate lists into single arrays
        all_ifr_times = all_ifr_times_list[0][0]#np.concatenate(all_ifr_times_list[0][0])
        all_ifr_values = all_ifr_values_list[0][0]#np.concatenate(all_ifr_values_list)
        
       

        # Define the bin width in seconds
        
        time_per_bin += density_bins
        bin_width =time_per_bin
        
        # Containers for results
        binned_ifr_values = []
        binned_time_edges = []
        
        # Loop over each cell (or trial)
        for ifr_list, time_list in zip(all_ifr_values_list, all_ifr_times_list):
            trial_binned = []
            trial_bins = []
            for ifr_array, time_array in zip(ifr_list, time_list):
                # Check if lengths match; if not, trim to the minimum length.
                if len(time_array) != len(ifr_array):
                  #  print("Warning: Mismatched lengths. Trimming to minimum.")
                    m = min(len(time_array), len(ifr_array))
                    time_array = time_array[:m]
                    ifr_array = ifr_array[:m]
        
                # Handle the case where the time array is empty.
                if time_array.size == 0:
                  #  print("Encountered empty time array; skipping binning for this entry.")
                    trial_binned.append(np.array([]))  # or use np.nan, as preferred
                    trial_bins.append(np.array([]))
                    continue
        
                # Define bin edges that span the spike times.
                t_min = np.floor(time_array.min() / bin_width) * bin_width
                t_max = np.ceil(time_array.max() / bin_width) * bin_width
                #bins = np.arange(t_min, t_max + bin_width, bin_width)
        
                # Use right=True so that a spike time equal to the bin edge goes to the lower bin.
                inds = np.digitize(time_array, bins, right=True) - 1
        
                # Prepare an array for the average IFR per bin.
                avg_in_bin = np.empty(len(bins) - 1)
                avg_in_bin[:] = np.nan
                
                # Compute the average IFR in each bin.
                for K in range(len(avg_in_bin)):
                    sel = inds == K
                    if np.any(sel):
                        avg_in_bin[K] = ifr_array[sel].mean()
        
                trial_binned.append(avg_in_bin)
                trial_bins.append(bins)
                
            binned_ifr_values.append(trial_binned)
            binned_time_edges.append(trial_bins)
        
        ifr_binned = binned_ifr_values[0][0]  # average IFR values in each bin for example
        #bins = binned_time_edges[0][0]          # corresponding bin edges
        
        # Compute bin centers from the edges.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot the histogram as a bar plot.
        
        
        #plt.xlabel("Time (s)")
#        plt.ylabel("Average iFR")
#        plt.title("Average Instantaneous Firing Rate (iFR) Across Time Bins")
        # # Example: Display binned output for the first instance.
        # print("Binned IFR values for first entry:", binned_ifr_values[0][0])
        # print("Corresponding bin edges:", binned_time_edges[0][0])
        
        # if all_ifr_times.size > 0:
        #         # Calculate sum of IFRs in each bin
                
        #         ifr_sum_in_bin, _ = np.histogram(all_ifr_times[1::], bins=bins, weights=all_ifr_values)
        #         # Calculate count of spikes (with preceding ISI) in each bin
        #         ifr_spike_counts_in_bin, _ = np.histogram(all_ifr_times, bins=bins)
        #         # Calculate average IFR, avoiding division by zero
        #         average_ifr = np.divide(ifr_sum_in_bin, ifr_spike_counts_in_bin,
        #                                 out=np.zeros_like(ifr_sum_in_bin), where=ifr_spike_counts_in_bin!=0)
        ax_iFR.axvline(0, color='red', linestyle='--', linewidth=1, zorder=5)
      
        plt.bar(bin_centers, ifr_binned, width=0.1, align='center', alpha=0.7, edgecolor='k')
        ax_iFR.plot(bin_centers, average_ifr, color='steelblue', linewidth=1.5) # Use a different color
        ax_iFR.set_ylim(bottom=0) # Start y-axis at 0
        ax_iFR.set_ylabel('Avg IFR (Hz)')
        #ax_iFR.set_title(event_column, fontsize=14) # Title on the top plot
        ax_iFR.grid(axis='y', linestyle=':', alpha=0.7)
        ax_iFR.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-ticks





#############################        
            
        
        # if align_to_end and isinstance(startstop, (list, tuple)) and len(startstop) > 1:
        #     spiketimes -= startstop[1]
        # elif isinstance(startstop, (list, tuple)):
        #     spiketimes -= startstop[0] 
        # else:
        
        #     spiketimes -= startstop
            
        ax_raster.scatter(spiketimes, np.ones_like(spikeind) * (i + 1), c='teal', s=0.5)
        
        previous_plotstop = plotstop
        previous_zero = plotzero
        all_spikes.append(spiketimes)
       
        #spikes_mat[:, i] = spiketimes
        
        # Adjust for multiple spikes in the same bin.
        while np.sum(spikes_around_stimulus > 0):
            spikes_around_stimulus[spikes_around_stimulus > 0] -= 1
            app_ind = np.where(spikes_around_stimulus > 0)[0]
            app_times = n_time_index[window_ind][app_ind]
            all_spikes.append(app_times - plotzero)
           
           
        
        # Sanity check: all trials should cover the same number of time bins.
        # if i == 0:
        #     num_timebins = np.sum(window_ind)
        # else:
        #     if (np.sum(window_ind) - num_timebins) > 1:
                
        #         continue
        #         raise ValueError('Not all trials cover the same time window for Hz calculation')
        
        # Get velocity for this trial.
        velind = (frame_index_s > plotstart) & (frame_index_s < plotstop)
    
        all_vel_times.append(frame_index_s[velind] - ref_time)        
        all_vel.append(velocity[velind])
        
        
        i += 1
        time_per_bin += density_bins
    
   
    
    ax_raster.set_ylim((0, np.nanmax((12, i + 4))))
    ax_raster.set_yticks(np.hstack((np.arange(0, i, 10), [i])))
    ax_raster.set_xlim((-window, window))
    
    # --- Compute and plot average firing rate ---
    all_spikes_list = all_spikes
    all_spikes = np.hstack(all_spikes)
    sum_spikes, firing_rate_bins = np.histogram(all_spikes, bins)    
    hz = sum_spikes / time_per_bin
    ax_FR.set_xticks([])
    ax_FR.bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
    ax_FR.set_xlim((-window, window))
    
    # --- Plot velocity ---
    ax_velocity.set_xticks([])
    ax_velocity.set_xlim((-window, window))
    #ax_velocity.set_ylim((0, 130))
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        ax_velocity.plot(plotveltime, plotvel, lw=0.5, c='grey')
      
    # Bin velocity with fixed 0.1-s bins.
    velocity_bin_size=0.1 #seconds
    velbins = np.arange(-window, window, velocity_bin_size)
  
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    avg_velocity = binned_values / np.maximum(binned_counts, 1)
    ax_velocity.plot(velbins[:-1], avg_velocity, c='orangered')
        
    for ax in axs:
        ax.axvline(0, linestyle='--', c='k')
    
    # --- Return results as dictionary if requested ---
    if return_data:
        velocity_bin_centers = (velbins[:-1] + velbins[1:]) / 2
        firing_centers= (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        precomputed = {
        "all_spikes": all_spikes,
        "firing_rate_bins": firing_rate_bins,
        "firing_centers":firing_centers,
        "hz": hz,
        "all_vel_times": all_vel_times,
        "all_vel": all_vel,
        "velbins": velbins,
        "velocity_bin_centers":velocity_bin_centers,
        "avg_velocity": avg_velocity,
        "state_events": state_events,
        'session':session,
        'session_aligned_spike_times':session_aligned_spike_times,
        'session_per_trial':session_per_trial
        }
        firing_df = pd.DataFrame({
            "Time_bin": (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2,
            "Firing_Rate": hz
        })
       
        velocity_df = pd.DataFrame({
            "Time_bin": velocity_bin_centers,
            "Avg_Velocity": avg_velocity
         })
        precomputed["Firing"] = firing_df
        precomputed["Velocity"] = velocity_df
    
        firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        
        psth_dict={
        'FR_Time_bin_center': firing_centers,
        'FR_Hz': hz}
        
        velocity_dict={
        'Velocity_Time_bin': vel_centers,
        'Avg_Velocity_cms': avg_velocity,
        'velocity_bin_size': velocity_bin_size,
        'binned_values': binned_values,
        'binned_counts':binned_counts        
        }
            
        meta_dict = {
            'behavior_start_stop_time_s': all_start_stop,
            'point':point}
            #'behavior_limits':None,
        
        
        raster_dict={
            'spikes_array': all_spikes_list
        }
        
        return_df= pl.DataFrame({
    'psth_dict': psth_dict,
    'velocity_dict':velocity_dict,
    'raster_dict':raster_dict,
    'meta_dict':meta_dict,
    'precomputed':precomputed},
            strict=False)   
  
        
        return return_df
    
def psth_cond(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, 
         window=5, density_bins=.5, return_data=False, session=None, 
         align_to_end=False):
    """   
    Parameters
    ----------
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin.
    n_time_index : array_like
        Time index from preprocessing.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing.
    all_start_stop : array_like
        Matrix (or vector) with event start (and stop for state events) times.
    axs : list of matplotlib.axes.Axes
        Contains three axes on which to plot: velocity, average firing, and raster.
    window : float, optional
        How many seconds before/after an event to plot. The default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. The default is 0.5.
    return_data : bool, optional
        If True, the function returns a pandas DataFrame (see details below).
        
    Returns
    -------
    If return_data is True, returns a dictionary with two DataFrames:
       - "Firing": DataFrame with columns ["Time_bin", "Firing_Rate"].
         Here, Time_bin is computed as the center of bins using density_bins.
       - "Velocity": DataFrame with columns ["Time_bin", "Avg_Velocity"].
         Here, Time_bin is computed as the center of 0.1-s bins.
         
    If return_data is False, nothing is returned.
    """
    
    # --- Prettify axes ---
    
    remove_axes(axs[2])
    remove_axes(axs[0], bottom=True)
    remove_axes(axs[1], bottom=True)
    
    # Containers for precomputed values:
    all_spikes = []       # List to store spike times per trial (aligned)
    all_vel = []          # List to store velocity trace per trial
    all_vel_times = []    # List to store velocity time points (aligned) per trial
    state_events = []     # List to store state-event shading info
    session_aligned_spike_times=[]
    session_per_trial=[]
   
    bins = np.arange(-window, window + density_bins, density_bins)
    time_per_bin = 0
    i = 0 
    point = False

    # Determine if the provided start-stop structure represents point versus state events
    if all_start_stop.shape == (2,):
        all_start_stop = [all_start_stop]
    elif len(all_start_stop.shape) == 1:
        point = True
       
    debug_counter=-1
    #spikes_mat = np.zeros((len_spiketimes, len(all_start_stop))) 
    for startstop in all_start_stop:
        debug_counter+=1
        if point==True:                    
            ref_time= startstop #doesn't matter where to align if only 1 value
        elif point==False:
            Duration = startstop[1] - startstop[0]
            if align_to_end==True:
                ref_time = startstop[1]  # Align to event end
                Duration= -Duration
            elif align_to_end==False:
                ref_time = startstop[0]  # Align to event start
            
            
        plotzero =ref_time
        
        # For the first trial, set the previous trial variables.
        if i == 0:
            previous_plotstop = 0
            previous_zero = 0
        
       
        if  point==False:  # State events
        # If events occur too close to the previous event, just add shading.
            if (plotzero - window) < previous_plotstop:                
                axs[2].barh(i - 2,Duration, 
                             left=plotzero - previous_zero,
                             height=1, color='burlywood', alpha=.5)
                state_events.append({
                    "trial": i - 2,
                    "left": plotzero - previous_zero,
                    "width": Duration,
                    "alpha": 0.5
                })
                continue
           
            axs[2].barh(i + 1, Duration, height=1, color='burlywood')
            state_events.append({
                "trial": i + 1,
                "left": 0,  # zero offset for normally separated events
                "width": Duration,
                "alpha": 1.0
            })

            
        
        plotstart = plotzero - window
        plotstop = plotzero + window

        # Collect spikes within the plotting window for this trial.
        window_ind = (n_time_index >= plotstart) & (n_time_index <= plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind = np.where(spikes_around_stimulus > 0)[0]
        spiketimes = n_time_index[window_ind][spikeind]
        session_aligned_spike_times.append(spiketimes)
        session_per_trial.append(session)
        
        spiketimes -= ref_time
       
                
            
        
        # if align_to_end and isinstance(startstop, (list, tuple)) and len(startstop) > 1:
        #     spiketimes -= startstop[1]
        # elif isinstance(startstop, (list, tuple)):
        #     spiketimes -= startstop[0] 
        # else:
        
        #     spiketimes -= startstop
            
        axs[2].scatter(spiketimes, np.ones_like(spikeind) * (i + 1), c='teal', s=0.5)
        
        previous_plotstop = plotstop
        previous_zero = plotzero
        all_spikes.append(spiketimes)
       
        #spikes_mat[:, i] = spiketimes
        
        # Adjust for multiple spikes in the same bin.
        while np.sum(spikes_around_stimulus > 0):
            spikes_around_stimulus[spikes_around_stimulus > 0] -= 1
            app_ind = np.where(spikes_around_stimulus > 0)[0]
            app_times = n_time_index[window_ind][app_ind]
            all_spikes.append(app_times - plotzero)                      
        # Sanity check: all trials should cover the same number of time bins.
        if i == 0:
            num_timebins = np.sum(window_ind)
        else:
            if (np.sum(window_ind) - num_timebins) > 1:
                
                continue
                raise ValueError('Not all trials cover the same time window for Hz calculation')
        
        # Get velocity for this trial.
        velind = (frame_index_s > plotstart) & (frame_index_s < plotstop)    
        all_vel_times.append(frame_index_s[velind] - ref_time)        
        all_vel.append(velocity[velind])             
        i += 1
        
        #time_per_bin += density_bins        
    time_per_bin = density_bins
    axs[2].set_ylim((0, np.nanmax((12, i + 4))))
    axs[2].set_yticks(np.hstack((np.arange(0, i, 10), [i])))
    axs[2].set_xlim((-window, window))
    
    # --- Compute and plot average firing rate ---
    all_spikes_list = all_spikes
    all_spikes = np.hstack(all_spikes)
    sum_spikes, firing_rate_bins = np.histogram(all_spikes, bins)
    
    hz = (sum_spikes / time_per_bin) / len(all_start_stop) # Convert to Hz, divide by number of trials.
    axs[1].set_xticks([])
    axs[1].bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
    axs[1].set_xlim((-window, window))
    
    # --- Plot velocity ---
    axs[0].set_xticks([])
    axs[0].set_xlim((-window, window))
    #axs[0].set_ylim((0, 130))
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        axs[0].plot(plotveltime, plotvel, lw=0.5, c='grey')
      
    # Bin velocity with fixed 0.1-s bins.
    velocity_bin_size=0.1 #seconds
    velbins = np.arange(-window, window, velocity_bin_size)
  
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    avg_velocity = binned_values / np.maximum(binned_counts, 1)
    axs[0].plot(velbins[:-1], avg_velocity, c='orangered')
        
    for ax in axs:
        ax.axvline(0, linestyle='--', c='k')
    
    # --- Return results as dictionary if requested ---
    if return_data:
        velocity_bin_centers = (velbins[:-1] + velbins[1:]) / 2
        firing_centers= (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        precomputed = {
        "all_spikes": all_spikes,
        "firing_rate_bins": firing_rate_bins,
        "firing_centers":firing_centers,
        "hz": hz,
        "all_vel_times": all_vel_times,
        "all_vel": all_vel,
        "velbins": velbins,
        "velocity_bin_centers":velocity_bin_centers,
        "avg_velocity": avg_velocity,
        "state_events": state_events,
        'session':session,
        'session_aligned_spike_times':session_aligned_spike_times,
        'session_per_trial':session_per_trial
        }
        firing_df = pd.DataFrame({
            "Time_bin": (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2,
            "Firing_Rate": hz
        })
       
        velocity_df = pd.DataFrame({
            "Time_bin": velocity_bin_centers,
            "Avg_Velocity": avg_velocity
         })
        precomputed["Firing"] = firing_df
        precomputed["Velocity"] = velocity_df
    
        firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        
        psth_dict={
        'FR_Time_bin_center': firing_centers,
        'FR_Hz': hz}
        
        velocity_dict={
        'Velocity_Time_bin': vel_centers,
        'Avg_Velocity_cms': avg_velocity,
        'velocity_bin_size': velocity_bin_size,
        'binned_values': binned_values,
        'binned_counts':binned_counts        
        }
            
        meta_dict = {
            'behavior_start_stop_time_s': all_start_stop,
            'point':point}
            #'behavior_limits':None,
        
        
        raster_dict={
            'spikes_array': all_spikes_list
        }
        
        return_df= pl.DataFrame({
    'psth_dict': psth_dict,
    'velocity_dict':velocity_dict,
    'raster_dict':raster_dict,
    'meta_dict':meta_dict,
    'precomputed':precomputed},
            strict=False)   
    #     psth_dict={
    #     'FR_Time_bin_center': firing_centers,
    #     'FR_Hz': hz}
        
    #     velocity_dict={
    #     'Velocity_Time_bin': vel_centers,
    #     'Avg_Velocity_cms': avg_velocity,
    #     'velocity_bin_size': velocity_bin_size,
    #     'binned_values': binned_values,
    #     'binned_counts':binned_counts        
    #     }
            
    #     meta_dict = {
    #         'behavior_start_stop_time_s': all_start_stop,
    #         'point':point}
    #         #'behavior_limits':None,
        
        
    #     raster_dict={
    #         'spikes_array': all_spikes_list
    #     }
        
    #     return_df= pl.DataFrame({
    # 'psth_dict': psth_dict,
    # 'velocity_dict':velocity_dict,
    # 'raster_dict':raster_dict,
    # 'meta_dict':meta_dict},
    # #'precomputed':precomputed},
    #         strict=False)
    
        
        return return_df
def psth_cond_refactored(neurondata, n_time_index, all_start_stop, velocity, frame_index_s,
              axs, window=5, density_bins=0.5, return_data=False, session=None,
              align_to_end=False):
    """
    Compute peri-event spike and velocity metrics and (optionally) plot them.

    Parameters
    ----------
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin.
    n_time_index : array_like
        Time index from preprocessing.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing.
    all_start_stop : array_like
        Matrix (or vector) with event start (and stop for state events) times.
    axs : matplotlib.axes.Axes or None
         If provided, plotting is done on these axes.
         If None, a new figure+axes object is created and plotting takes place.
    window : float, optional
        How many seconds before/after an event to analyze. Default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. Default is 0.5.
    return_data : bool, optional
        If True, returns a dictionary of computed values.
    session : any, optional
         Session identifier to include in return.
    align_to_end : bool, optional
        If True, align events to the event's end time rather than the start.

    Returns
    -------
    If return_data is True, returns a dictionary with the following keys:
      - "Firing": DataFrame with columns ["Time_bin", "Firing_Rate"].
      - "Velocity": DataFrame with columns ["Time_bin", "Avg_Velocity"].
      - Additional computed values in a dict 'precomputed'.
    Otherwise, returns None.
    """

    # --- Setup plotting mode ---
    plot_enabled = False
    if axs is not None:
        plot_enabled = True
        #fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    # --- Initialize containers for computed values ---
    all_spikes_list = []      # For raster (list of arrays per trial)
    trial_raster_info = []    # Each element: tuple(spike_times, y-value for raster)
    all_vel = []              # Velocity per trial
    all_vel_times = []        # Timepoints for velocity (aligned)
    state_events = []         # Contains info for state-event shading
    session_aligned_spike_times = []  # Aligned spike times per trial
    session_per_trial = []    # Record session info for each trial

    # Define bin edges for spike histogram (density)
    bins = np.arange(-window, window + density_bins, density_bins)
    time_per_bin = density_bins

    # Determine whether events are "point" events or "state" events
    point = False
    if all_start_stop.shape == (2,):
        # Provided as two-value array: treat as a single state event.
        all_start_stop = [all_start_stop]
    elif len(all_start_stop.shape) == 1:
        # 1-D array: treat each value as a point event.
        point = True

    # For tracking overlapping state events
    previous_plotstop = -np.inf
    previous_zero = 0
    trial_index = 0

    # --- Event loop: compute spike and velocity info per event ---
    for startstop in all_start_stop:
        if not point:
            Duration = startstop[1] - startstop[0]
            if align_to_end:
                ref_time = startstop[1]  # Align to event end
                Duration = -Duration
            else:
                ref_time = startstop[0]  # Align to event start
        else:
            ref_time = startstop

        plotzero = ref_time
        plotstart = plotzero - window
        plotstop = plotzero + window

        # For state events, check if overlapping with previous event
        if not point:
            if (plotzero - window) < previous_plotstop:
                state_events.append({
                    "trial": trial_index - 2,
                    "left": plotzero - previous_zero,
                    "width": Duration,
                    "alpha": 0.5
                })
                trial_index += 1
                continue
            else:
                state_events.append({
                    "trial": trial_index + 1,
                    "left": 0,
                    "width": Duration,
                    "alpha": 1.0
                })

        # --- Compute spike times for current trial ---
        window_mask = (n_time_index >= plotstart) & (n_time_index <= plotstop)
        spikes_around_event = neurondata[window_mask].copy()
        spike_inds = np.where(spikes_around_event > 0)[0]
        spiketimes = n_time_index[window_mask][spike_inds].astype(float)
        session_aligned_spike_times.append(spiketimes)
        session_per_trial.append(session)
        # Align spikes relative to ref_time:
        spiketimes = spiketimes - ref_time

        # Save raster information.
        trial_raster_info.append((spiketimes, trial_index + 1))
        all_spikes_list.append(spiketimes)

        # Adjust for multiple spike counts per time bin
        spikes_copy = spikes_around_event.copy()
        while np.sum(spikes_copy > 0):
            spikes_copy[spikes_copy > 0] -= 1
            extra_inds = np.where(spikes_copy > 0)[0]
            extra_times = n_time_index[window_mask][extra_inds] - plotzero
            all_spikes_list.append(extra_times)

        # --- Compute velocity data for current trial ---
        vel_mask = (frame_index_s > plotstart) & (frame_index_s < plotstop)
        all_vel_times.append(frame_index_s[vel_mask] - ref_time)
        all_vel.append(velocity[vel_mask])

        previous_plotstop = plotstop
        previous_zero = plotzero
        trial_index += 1

    # --- Compute average firing rate ---
    if all_spikes_list:
        all_spikes_concat = np.hstack(all_spikes_list)
    else:
        all_spikes_concat = np.array([])

    sum_spikes, firing_rate_bins = np.histogram(all_spikes_concat, bins=bins)
    hz = (sum_spikes / time_per_bin) / len(all_start_stop) # Convert to Hz, divide by number of trials.
    firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
    if np.max(hz)>=500:
        print('{np.max(hz)=}')
        IPython.embed()
    # --- Compute average velocity ---
    velocity_bin_size = 0.1
    velbins = np.arange(-window, window, velocity_bin_size)
    if all_vel_times:
        all_vel_times_concat = np.hstack(all_vel_times)
    else:
        all_vel_times_concat = np.array([])
    if all_vel:
        all_vel_concat = np.hstack(all_vel)
    else:
        all_vel_concat = np.array([])

    binned_values, _ = np.histogram(all_vel_times_concat, bins=velbins, weights=all_vel_concat)
    binned_counts, _ = np.histogram(all_vel_times_concat, bins=velbins)
    avg_velocity = binned_values / np.maximum(binned_counts, 1)
    velocity_bin_centers = (velbins[:-1] + velbins[1:]) / 2.
##################################################################
    # --- Plotting: Only if plot_enabled is True ---
    if plot_enabled:
        alpha=.5
        # Prettify axes (example: remove x-ticks on upper plots)
        for ax in axs:
            ax.label_outer()
            ax.axvline(0, linestyle='--', c='k')

        # Plot velocity (axs[0])
        for trial_vel, trial_times in zip(all_vel, all_vel_times):
            axs[0].plot(trial_times, trial_vel, lw=0.5, c='grey')
        axs[0].plot(velbins[:-1], avg_velocity, c='orangered')
        axs[0].set_xlim((-window, window))
        axs[0].set_title("Velocity")

        # Plot average firing rate (axs[1])
        axs[1].bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
        axs[1].set_xlim((-window, window))
        axs[1].set_title("Average Firing Rate")
        axs[1].set_xticks([])

        # Plot raster (axs[2])
        for spiketimes, yval in trial_raster_info:
            axs[2].scatter(spiketimes, np.full_like(spiketimes, yval),
                           c='teal', s=0.5)
        # Plot state event shading (if any)
        for se in state_events:
            axs[2].barh(se["trial"], se["width"], left=se["left"],height=1, color='burlywood', alpha=alpha)
                         #height=1, color='burlywood', alpha=se["alpha"])
        axs[2].set_xlim((-window, window))
        axs[2].set_ylim((0, trial_index + 4))
        axs[2].set_title("Raster")
        axs[2].set_xlabel("Time (s)")
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.tight_layout()
        plt.show()

    # --- Prepare data to return if requested ---
    if return_data:
        # Prepare a dictionary for already computed values
        precomputed = {
            "all_spikes": all_spikes_concat,
            "firing_rate_bins": firing_rate_bins,
            "firing_centers": firing_centers,
            "hz": hz,
            "all_vel_times": all_vel_times,
            "all_vel": all_vel,
            "velbins": velbins,
            "velocity_bin_centers": velocity_bin_centers,
            "avg_velocity": avg_velocity,
            "state_events": state_events,
            "session": session,
            "session_aligned_spike_times": session_aligned_spike_times,
            "session_per_trial": session_per_trial
        }
        
        # Create DataFrames (using pandas)
        firing_df = pd.DataFrame({
            "Time_bin": firing_centers,
            "Firing_Rate": hz
        })
        velocity_df = pd.DataFrame({
            "Time_bin": velocity_bin_centers,
            "Avg_Velocity": avg_velocity
        })
        
        # You may also want to bundle additional dictionaries:
        psth_dict = {
            'FR_Time_bin_center': firing_centers,
            'FR_Hz': hz
        }
        velocity_dict = {
            'Velocity_Time_bin': velocity_bin_centers,
            'Avg_Velocity_cms': avg_velocity,
            'velocity_bin_size': velocity_bin_size,
            'binned_values': binned_values,
            'binned_counts': binned_counts        
        }
        meta_dict = {
            'behavior_start_stop_time_s': all_start_stop,
            'point': point
        }
        raster_dict = {
            'spikes_array': all_spikes_list
        }
        
        # If you are using polars, you can build a polars DataFrame.
        # For now, we return a combined dictionary.
        return_data_dict = {
            'psth_dict': psth_dict,
            'velocity_dict': velocity_dict,
            'raster_dict': raster_dict,
            'meta_dict': meta_dict,
            'precomputed': precomputed,
            'Firing': firing_df,
            'Velocity': velocity_df
        }
        
        # If using polars uncomment the following lines:
        return_df= pl.DataFrame({
         'psth_dict': psth_dict,
         'velocity_dict':velocity_dict,
         'raster_dict':raster_dict,
         'meta_dict':meta_dict,
         'precomputed':precomputed},
                 strict=False)
        return return_df
       
       
        return return_data_dict

    return None    
def psth(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, 
         window=5, density_bins=.5, return_data=False,session=None):
    """   
    Parameters
    ----------
    neurondata : array_like
        Row from ndata, containing number of spikes per timebin.
    n_time_index : array_like
        Time index from preprocessing.
    velocity : array_like
        Velocity from preprocessing.
    frame_index_s : array_like
        Frame index (in seconds) from preprocessing.
    all_start_stop : array_like
        Matrix (or vector) with event start (and stop for state events) times.
    axs : list of matplotlib.axes.Axes
        Contains three axes on which to plot: velocity, average firing, and raster.
    window : float, optional
        How many seconds before/after an event to plot. The default is 5.
    density_bins : float, optional
        Bin width (in seconds) for averaging the spike activity. The default is 0.5.
    return_data : bool, optional
        If True, the function returns a pandas DataFrame (see details below).
        
    Returns
    -------
    If return_data is True, returns a dictionary with two DataFrames:
       - "Firing": DataFrame with columns ["Time_bin", "Firing_Rate"].
         Here, Time_bin is computed as the center of bins using density_bins.
       - "Velocity": DataFrame with columns ["Time_bin", "Avg_Velocity"].
         Here, Time_bin is computed as the center of 0.1-s bins.
         
    If return_data is False, nothing is returned.
    """
    
    # --- Prettify axes ---
    remove_axes(axs[2])
    remove_axes(axs[0], bottom=True)
    remove_axes(axs[1], bottom=True)
    # Containers for precomputed values:
    all_spikes = []       # List to store spike times per trial (aligned)
    all_vel = []          # List to store velocity trace per trial
    all_vel_times = []    # List to store velocity time points (aligned) per trial
    state_events = []     # List to store state-event shading info
    session_aligned_spike_times=[]
    session_per_trial=[]
   
    bins = np.arange(-window, window + density_bins, density_bins)
    time_per_bin = 0
    i = 0 
    point = False

    # Determine if the provided start-stop structure represents point versus state events
    if all_start_stop.shape == (2,):
        all_start_stop = [all_start_stop]
    elif len(all_start_stop.shape) == 1:
        point = True
    
    #spikes_mat = np.zeros((len_spiketimes, len(all_start_stop))) 
    for startstop in all_start_stop:
        # For the first trial, set the previous trial variables.
        if i == 0:
            previous_plotstop = 0
            previous_zero = 0
        
        if  point==False:  # State events
            plotzero = startstop[0]
            # If events occur too close to the previous event, just add shading.
            if (plotzero - window) < previous_plotstop:
                axs[2].barh(i - 2, startstop[1] - startstop[0], 
                             left=plotzero - previous_zero,
                             height=1, color='burlywood', alpha=.5)
                state_events.append({
                    "trial": i - 2,
                    "left": plotzero - previous_zero,
                    "width": startstop[1] - startstop[0],
                    "alpha": 0.5
                })
                continue                
            axs[2].barh(i + 1, startstop[1] - startstop[0], height=1, color='burlywood')
            state_events.append({
                "trial": i + 1,
                "left": 0,  # zero offset for normally separated events
                "width": startstop[1] - startstop[0],
                "alpha": 1.0
            })
        else:  # Point events
            plotzero = startstop
        
        plotstart = plotzero - window
        plotstop = plotzero + window

        # Collect spikes within the plotting window for this trial.
        window_ind = (n_time_index >= plotstart) & (n_time_index <= plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind = np.where(spikes_around_stimulus > 0)[0]
        spiketimes = n_time_index[window_ind][spikeind]
        session_aligned_spike_times.append(spiketimes)
        session_per_trial.append(session)
        

        # Align spikes relative to the event
        spiketimes -= plotzero

        axs[2].scatter(spiketimes, np.ones_like(spikeind) * (i + 1), c='teal', s=0.5)
        if len(spiketimes) > 10:
            axs[2].set_yticklabels([])
        
        previous_plotstop = plotstop
        previous_zero = plotzero
        all_spikes.append(spiketimes)
       
        #spikes_mat[:, i] = spiketimes
        
        # Adjust for multiple spikes in the same bin.
        while np.sum(spikes_around_stimulus > 0):
            spikes_around_stimulus[spikes_around_stimulus > 0] -= 1
            app_ind = np.where(spikes_around_stimulus > 0)[0]
            app_times = n_time_index[window_ind][app_ind]
            all_spikes.append(app_times - plotzero)
           
           
        
        # Sanity check: all trials should cover the same number of time bins.
        if i == 0:
            num_timebins = np.sum(window_ind)
        else:
            if (np.sum(window_ind) - num_timebins) > 1:
                raise ValueError('Not all trials cover the same time window for Hz calculation')
        
        # Get velocity for this trial.
        velind = (frame_index_s > plotstart) & (frame_index_s < plotstop)
        all_vel_times.append(frame_index_s[velind] - plotzero)
        all_vel.append(velocity[velind])
        
        i += 1
        time_per_bin += density_bins

    axs[2].set_ylim((0, np.nanmax((12, i + 4))))
    axs[2].set_yticks(np.hstack((np.arange(0, i, 10), [i])))
    axs[2].set_xlim((-window, window))
    
    # --- Compute and plot average firing rate ---
    all_spikes_list = all_spikes
    all_spikes = np.hstack(all_spikes)
    sum_spikes, firing_rate_bins = np.histogram(all_spikes, bins)
    hz = sum_spikes / time_per_bin
    axs[1].set_xticks([])
    axs[1].bar(firing_rate_bins[:-1], hz, align='edge', width=density_bins, color='grey')
    axs[1].set_xlim((-window, window))
    
    # --- Plot velocity ---
    axs[0].set_xticks([])
    axs[0].set_xlim((-window, window))
    #axs[0].set_ylim((0, 130))
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        axs[0].plot(plotveltime, plotvel, lw=0.5, c='grey')
      
    # Bin velocity with fixed 0.1-s bins.
    velocity_bin_size=0.1 #seconds
    velbins = np.arange(-window, window, velocity_bin_size)
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    avg_velocity = binned_values / np.maximum(binned_counts, 1)
    axs[0].plot(velbins[:-1], avg_velocity, c='orangered')
        
    for ax in axs:
        ax.axvline(0, linestyle='--', c='k')
    
    # --- Return results as dictionary if requested ---
    if return_data:
        velocity_bin_centers = (velbins[:-1] + velbins[1:]) / 2
        firing_centers= (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2
        # Build dictionary of precomputed variables.
        precomputed = {
        "all_spikes": all_spikes,
        "firing_rate_bins": firing_rate_bins,
        "firing_centers":firing_centers,
        "hz": hz,
        "all_vel_times": all_vel_times,
        "all_vel": all_vel,
        "velbins": velbins,
        "velocity_bin_centers":velocity_bin_centers,
        "avg_velocity": avg_velocity,
        "state_events": state_events,
        'session':session,
        'session_aligned_spike_times':session_aligned_spike_times,
        'session_per_trial':session_per_trial
        }
        firing_df = pd.DataFrame({
            "Time_bin": (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2,
            "Firing_Rate": hz
        })
       
        velocity_df = pd.DataFrame({
            "Time_bin": velocity_bin_centers,
            "Avg_Velocity": avg_velocity
         })
        precomputed["Firing"] = firing_df
        precomputed["Velocity"] = velocity_df
    
        firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # Compute bin centers for velocity (using 0.1-s bins):
        vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        
        psth_dict={
        'FR_Time_bin_center': firing_centers,
        'FR_Hz': hz}
        
        velocity_dict={
        'Velocity_Time_bin': vel_centers,
        'Avg_Velocity_cms': avg_velocity,
        'velocity_bin_size': velocity_bin_size,
        'binned_values': binned_values,
        'binned_counts':binned_counts        
        }
            
        meta_dict = {
            'behavior_start_stop_time_s': all_start_stop,
            'point':point}
            #'behavior_limits':None,
        
        
        raster_dict={
            'spikes_array': all_spikes_list
        }
        
        return_df= pl.DataFrame({
    'psth_dict': psth_dict,
    'velocity_dict':velocity_dict,
    'raster_dict':raster_dict,
    'meta_dict':meta_dict,
    'precomputed':precomputed},
            strict=False)
    
        
        return return_df
                
        # # Compute bin centers for firing rate:
        # firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
        # df_firing = pd.DataFrame({
        #     'FR_Time_bin': firing_centers,
        #     'FR_Hz': hz
        # })
        
        # # Compute bin centers for velocity (using 0.1-s bins):
        # vel_centers = (velbins[:-1] + velbins[1:]) / 2.
        # df_velocity = pd.DataFrame({
        #     'Velocity_Time_bin': vel_centers,
        #     'Avg_Velocity_cms': avg_velocity
        # })
        
        # # Optionally, create a DataFrame for start-stop times. 
        # # (If all_start_stop is an array/matrix, you might need to process it further)
        # df_start_stop = pd.DataFrame({'behavior_start_stop': all_start_stop})
        
        # # Option 1: Simply concatenate the three DataFrames side by side.
        # # Note: if they have different numbers of rows, missing entries will be NaN.
        # df_all = pd.concat([df_firing, df_velocity, df_start_stop], axis=1)
        # return df_all
       #  # Compute bin centers for firing rate:
       #  firing_centers = (firing_rate_bins[:-1] + firing_rate_bins[1:]) / 2.
       #  df_firing = pd.DataFrame({
       #      'FR_Time_bin': firing_centers,
       #      'FR_Hz': hz
       #  })
        
       #  # Compute bin centers for velocity (using 0.1-s bins):
       #  vel_centers = (velbins[:-1] + velbins[1:]) / 2.
       #  df_velocity = pd.DataFrame({
       #      'celocity_Time_bin': vel_centers,
       #      'Avg_Velocity_cms': avg_velocity
       #  })
       # # df_start_stop=pd.DataFrame({'behavior_start_stop':all_start_stop})
        
       #  # Returning a dictionary of two DataFrames.
       #  #return {'Firing': df_firing, 'Velocity': df_velocity,'behavior_start_stop':df_start_stop}
       #  return {'Firing': df_firing, 'Velocity': df_velocity}
    
    # If return_data is False, nothing is returned.


def display_concatenated_psth(cell_id, concatenated_psth_data, window=5, density_bins=0.5, behavior_list=None):
    """
    Display the concatenated PSTHs for a given cell.

    Parameters
    ----------
    cell_id : int or str
        Identifier for the cell (matching the "Cell" column in concatenated_psth_data).
    concatenated_psth_data : pd.DataFrame
        DataFrame with one row per cell and one column per behavior.
        Columns like "Session" or "Cell" are ignored.
        Each behavior cell is assumed to store either a dict with keys "Firing" and "Velocity" or a DataFrame.
    window : float, optional
        The time window used for plotting (default is 5 seconds; -window to window will be plotted).
    density_bins : float, optional
        The bin width used for the firing rate histogram (default is 0.5 s).
    behavior_list : list of str, optional
        List of behavior column names to display. If None, all columns except "Session" and "Cell"
        (and optionally "Sessions") are used.

    Returns
    -------
    None

    The function creates a figure with three rows and one column per behavior:
        - Top row: Average Velocity plot (if available)
        - Middle row: Firing rate histogram (bar plot)
        - Bottom row: Dummy raster area (since individual trial raster data are not available in the aggregated data)
    """
    # Determine which behavior columns to use.
    if behavior_list is None:
        behavior_list = [col for col in concatenated_psth_data.columns if col not in ['Session', 'Cell', 'Sessions']]
    
    # Get the row corresponding to the cell.
    cell_rows = concatenated_psth_data[concatenated_psth_data['Cell'] == cell_id]
    if cell_rows.empty:
        print(f"No data for cell {cell_id}")
        return
    # Assume cell_id is unique.
    cell_data = cell_rows.iloc[0]
    
    num_behaviors = len(behavior_list)
    # Create a figure with 3 rows (for velocity, firing, raster) and one column per behavior.
    fig = plt.figure(figsize=(6 * num_behaviors, 12))
    gs = gridspec.GridSpec(3, num_behaviors, hspace=0.5, wspace=0.4)
    
    for i, beh in enumerate(behavior_list):
        psth_entry = cell_data[beh]
        if psth_entry is None:
            # Skip or leave blank if no data.
            continue
        # Check if the stored object is a dictionary (with keys "Firing" and "Velocity")
        if isinstance(psth_entry, dict):
            df_firing = psth_entry.get('Firing', None)
            df_velocity = psth_entry.get('Velocity', None)
        else:
            # Assume the stored object is the firing PSTH DataFrame.
            df_firing = psth_entry
            df_velocity = None
        
        # ---------------------------
        # Top subplot: Velocity plot
        ax_velocity = plt.subplot(gs[0, i])
        if df_velocity is not None:
            # Plot average velocity versus time.
            ax_velocity.plot(df_velocity['Time_bin'], df_velocity['Avg_Velocity'], color='orangered')
        ax_velocity.set_title(beh, fontsize=14)
        ax_velocity.set_xlim([-window, window])
        ax_velocity.set_xticks([])
        ax_velocity.set_ylabel("Velocity")
        ax_velocity.axvline(0, linestyle='--', color='k')
        
        # ---------------------------
        # Middle subplot: Firing Rate
        ax_firing = plt.subplot(gs[1, i])
        if df_firing is not None:
            # Plot the firing rate as a bar plot. Here, we assume df_firing contains:
            #   "Time_bin": bin centers and "Firing_Rate": firing rate in Hz.
            ax_firing.bar(df_firing['Time_bin'], df_firing['Firing_Rate'],
                          width=density_bins, align='center', color='grey')
        ax_firing.set_xlim([-window, window])
        ax_firing.set_xticks([])
        ax_firing.set_ylabel("Firing Rate")
        ax_firing.axvline(0, linestyle='--', color='k')
        
        # ---------------------------
        # Bottom subplot: Raster (not available in aggregated data)
        ax_raster = plt.subplot(gs[2, i])
        # Display a textual message in place of the raster plot.
        ax_raster.text(0.5, 0.5, "Aggregated\nRaster", ha='center', va='center',
                       fontsize=12, color='gray')
        ax_raster.set_xlim([-window, window])
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])
        ax_raster.axvline(0, linestyle='--', color='k')
        
    plt.suptitle(f"Concatenated PSTHs for Cell {cell_id}", fontsize=16)
    plt.show()    

def plot_and_save_concatenated_psth(cell_id, concatenated_psth_data, save_path,
                                    window=5, density_bins=0.5, behavior_list=None):
    """
    Plot and save the concatenated PSTH for a given cell.
    
    Parameters
    ----------
    cell_id : int or str
        The unique identifier for the cell (as found in the 'Cell' column).
    concatenated_psth_data : pd.DataFrame
        DataFrame with one row per cell and at least the columns "Cell" and one column
        per behavior. Each behavior column should store a PSTH result (either a DataFrame
        or a dictionary with keys "Firing" and "Velocity").
    save_path : str
        Full path (including filename, e.g., "path/to/fig_cell5.png") where the figure will be saved.
    window : float, optional
        Time window for the x-axis (from -window to +window seconds). Default is 5.
    density_bins : float, optional
        Bin width for the firing rate histogram. Default is 0.5.
    behavior_list : list of str, optional
        List of behavior column names to display. If None, all columns aside from "Session" and "Cell"
        (and "Sessions" if present) will be displayed.
        
    Returns
    -------
    None
    """
    # Determine behavior columns if not provided.
    if behavior_list is None:
        behavior_list = [col for col in concatenated_psth_data.columns if col not in ['Session', 'Cell', 'Sessions']]
    
    # Extract the row for this cell.
    cell_rows = concatenated_psth_data[concatenated_psth_data['Cell'] == cell_id]
    if cell_rows.empty:
        print(f"No data for cell {cell_id}")
        return
    # Assume cell_id is unique.
    cell_data = cell_rows.iloc[0]
    
    num_behaviors = len(behavior_list)
    # Create a figure with 3 rows (velocity, firing rate, and raster) and one column per behavior.
    fig = plt.figure(figsize=(6 * num_behaviors, 12))
    gs = gridspec.GridSpec(3, num_behaviors, hspace=0.5, wspace=0.4)
    
    for i, beh in enumerate(behavior_list):
        psth_entry = cell_data[beh]
        if psth_entry is None:
            continue

        # If the stored object is a dictionary, extract both firing and velocity data.
        if isinstance(psth_entry, dict):
            df_firing = psth_entry.get('Firing', None)
            df_velocity = psth_entry.get('Velocity', None)
        else:
            # Otherwise, assume it's the firing DataFrame.
            df_firing = psth_entry
            df_velocity = None
        
        # ---------------------------
        # Top subplot: Average Velocity
        ax_velocity = plt.subplot(gs[0, i])
        if df_velocity is not None:
            ax_velocity.plot(df_velocity['Time_bin'], df_velocity['Avg_Velocity'], color='orangered')
        ax_velocity.set_title(beh, fontsize=14)
        ax_velocity.set_xlim([-window, window])
        ax_velocity.set_xticks([])
        ax_velocity.set_ylabel("Velocity")
        ax_velocity.axvline(0, linestyle='--', color='k')
        
        # ---------------------------
        # Middle subplot: Firing Rate
        ax_firing = plt.subplot(gs[1, i])
        if df_firing is not None:
            ax_firing.bar(df_firing['Time_bin'], df_firing['Firing_Rate'], 
                          width=density_bins, align='center', color='grey')
        ax_firing.set_xlim([-window, window])
        ax_firing.set_xticks([])
        ax_firing.set_ylabel("Firing Rate")
        ax_firing.axvline(0, linestyle='--', color='k')
        
        # ---------------------------
        # Bottom subplot: Raster placeholder
        ax_raster = plt.subplot(gs[2, i])
        ax_raster.text(0.5, 0.5, "Aggregated\nRaster", ha='center', va='center',
                       fontsize=12, color='gray')
        ax_raster.set_xlim([-window, window])
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])
        ax_raster.axvline(0, linestyle='--', color='k')
    
    plt.suptitle(f"Concatenated PSTHs for Cell {cell_id}", fontsize=16)
    # Save the figure.
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PSTH plot for cell {cell_id} at {save_path}")


def plot_and_save_all_concatenated_psths(concatenated_psth_data, out_dir,
                                         window=5, density_bins=0.5, behavior_list=None):
    """
    Loop over all cells in concatenated_psth_data, plot, and save each concatenated PSTH.
    
    Parameters
    ----------
    concatenated_psth_data : pd.DataFrame
        DataFrame with one row per cell and one column per behavior.
    out_dir : str
        The output directory where figures will be saved.
    window : float, optional (default: 5)
        Time window for plotting.
    density_bins : float, optional (default: 0.5)
        Bin width for firing rate histogram.
    behavior_list : list of str, optional
        List of behaviors to display. If None, all available columns (aside from "Session" and "Cell")
        are used.
        
    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Loop through each row in the DataFrame.
    for index, row in concatenated_psth_data.iterrows():
        cell_id = row['Cell']
        save_filename = f"cell_{cell_id}_concatenated_PSTH.png"
        save_path = os.path.join(out_dir, save_filename)
        plot_and_save_concatenated_psth(cell_id, concatenated_psth_data, 
                                        save_path, window, density_bins, behavior_list)

# ==============================================================================
# Example Usage:
# ------------------------------------------------------------------------------
# Assume concatenated_psth_data is your DataFrame with one row per cell.
#
# Set the output directory where the figures will be saved:
# out_dir = "path/to/save/figures"
#
# Then call:
# plot_and_save_all_concatenated_psths(concatenated_psth_data, out_dir, window=5, density_bins=0.5)







# --- Helper function for prettifying axes (if not already defined) ---
def remove_axes(ax, bottom=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False)

def old_psth(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, window=5, density_bins=.5,return_data=False):
    """   
    Parameters
    ----------
    neurondata : int vector
        row from ndata, containing nuber of spikes per timebin.
    n_time_index : from preprocessing.
    velocity : from preprocessing.
    frame_index_s : from preprocessing
        .
    start_stop : matrix, left are start frames, right are stop frames, for one behaviour
        from hf.start_stop_array.    
    axs : List
        contains 3 axes, on which to plot the figure.
    window : float, optional
        how many sec before/ after an event to start/stop plotting.
        The default is 5.
    density_bins : float, optional
        how big should the bins be that average the activity in PSTH.
        The default is .5.
    """
    
    point=False
    
    #Prettify axes
    remove_axes(axs[2])
    remove_axes(axs[0],bottom=True)
    remove_axes(axs[1],bottom=True)
    
    all_spikes=[]
    all_vel=[]
    all_vel_times=[]
    bins=np.arange(-window,window+density_bins, density_bins)

    time_per_bin=0
    i=0 
    
    
    #Test if b is point or state behaviour
    if all_start_stop.shape == (2,):
        all_start_stop=[all_start_stop]
    elif len(all_start_stop.shape) ==1:
        point=True
        
        
    # Go through each trial
    for  startstop in all_start_stop:
       # print(startstop)
        # mark where the trial is happening
        if not i: #Exception for first trial
            previous_plotstop=0
            previous_zero=0
        
        # plot centred at starts
        if not point: #State events
            plotzero=startstop[0]
            
            # shade additional trials (i.e. if distance to next trial is too short, don't give them a separate line but just shade them in the previous trial)
            if (plotzero-window)<previous_plotstop:
                axs[2].barh(i-2, startstop[1]-startstop[0], left = plotzero-previous_zero ,height=1 , color='burlywood', alpha=.5)                
                continue            
            # shade trial time
            axs[2].barh(i+1, startstop[1]-startstop[0],height=1 , color='burlywood')
            
        # # plot centred at stops
        # elif plotref==1:
        #     plotzero=startstop[1]
            
        #     # shade trial time
        #     plt.barh(i+1, startstop[0]-startstop[1],height=1 , color='burlywood')
            
        #     # shade additional trials 
        #     if (plotzero-window)<previous_plotstop:
        #         plt.barh(i+1, startstop[0]-startstop[1], left = plotzero-previous_zero ,height=1 , color='burlywood')
        #         continue
        else: #point events
            plotzero=startstop
       
        #Determine plotting window
        plotstart=plotzero-window
        plotstop=plotzero+window
        
        # Collect spikes in plotting window for this trial
        window_ind=(n_time_index>=plotstart) & (n_time_index<=plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind=np.where(spikes_around_stimulus>0)[0]
        spiketimes=n_time_index[window_ind][spikeind]
        
        # if  np.size(startstop)==1:
        #     if startstop>4365:

        # align to eventtime
        spiketimes-=plotzero

        # Plot the dots for one trial
        axs[2].scatter(spiketimes, np.ones_like(spikeind)*(i+1), c='teal', s=.5)
        
        # Save trial stop, to know if the next trial should get its own line, or just be plotted in the same line
        previous_plotstop=plotstop
        previous_zero=plotzero
        all_spikes.append(spiketimes)
        
        # For calculating firing in Hz, account for multiple spikes in the same 10ms bin
        while np.sum(spikes_around_stimulus>0): 
            spikes_around_stimulus[spikes_around_stimulus>0]-=1
            app_ind=np.where(spikes_around_stimulus>0)[0]
            app_times=n_time_index[window_ind][app_ind]
            all_spikes.append(app_times-plotzero)
        
        # sanity check
        if not i:
            num_timebins=np.sum(window_ind)
        else:
            if (np.sum(window_ind) - num_timebins)>1:
                raise ValueError('not in all trials you cover the same time. You assume this though in Hz calculation')
        
        # get veloity for trial
        velind=(frame_index_s>plotstart) & (frame_index_s< plotstop)
        all_vel_times.append(frame_index_s[velind]-plotzero)
        all_vel.append(velocity[velind])
        
        i+=1
        time_per_bin+=density_bins

    axs[2].set_ylim((0,np.nanmax((12,i+4))))
    axs[2].set_yticks(np.hstack((np.arange(0,i,10),[i])))
    axs[2].set_xlim((-window,window))
    
    #plot avg Hz on top    
    all_spikes=np.hstack(all_spikes)    
    sum_spikes, return_bins=np.histogram  (all_spikes, bins)  
    
    hz=sum_spikes/time_per_bin
    
    axs[1].set_xticks([])   
    axs[1].bar(bins[:-1],hz, align='edge',width=density_bins, color='grey')
    axs[1].set_xlim((-window,window))
     
    # plot velocity
    axs[0].set_xticks([])
    axs[0].set_xlim((-window,window))
    axs[0].set_ylim((0,130))
   
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        axs[0].plot(plotveltime, plotvel, lw=.5, c='grey')
      
    # get/ plot mean velocity
    velbins=np.arange(-window, window, .1)
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    axs[0].plot( velbins[:-1], binned_values/binned_counts, c='orangered')
        
    # make dashed line at 0 
    for ax in axs:
        ax.axvline(0,linestyle='--', c='k')#, ymax=len(stimulus_times))
    
   # if return_data:
      
#       return 

def box_plotter(behaviour,event, c,p, plotmax=None, ax=None):
    target_b=behaviour[behaviour['behaviours']==event]
    if not len(target_b):
        return
    if p=='axvline':
        for entry in target_b['frames_s']:
            if ax is None:
                plt.axvline(entry, label=event, color=c, lw=1.5)
            else:
                ax.axvline(entry, label=event, color=c, lw=1.5)
                
    elif p=='box':
        starts=target_b[target_b['start_stop']=='START']['frames_s'].to_numpy()
        stops=target_b[target_b['start_stop']=='STOP']['frames_s'].to_numpy()
        #replace starts/stops outside of plotting window with 0/inf
        if target_b['start_stop'].iloc[0]!='START':
            starts=np.insert(starts,0,0)
        if target_b['start_stop'].iloc[-1]!='STOP':
            stops=np.insert(stops,-1,plotmax)
            
        for start, stop in zip(starts, stops):
            if ax is None:
                plt.axvspan(start, stop, label=event, color=c)
            else:
                ax.axvspan(start, stop, label=event, color=c)
        # plt.plot()
        
    else:
        raise ValueError('p variable is unvalid')

def plot_events_shahaf(behaviour, plotmin=None, plotmax=None, ax=None):
    """
    creates shading in figure according to behaviour. If behaviour is point behaviour, 
    axvline is created
    
    parameters
    -----------
    behaviour: The behaviour output file from preprocessing. It should have following colomns:
        behaviours, frames_s, start_stop
    plotmin: in s, from what time onwards  should plotting of behaviours start
    plotmax: in s, until what time should behaviour be plotted
    --> only works if both plotmin AND plotmax are given
    ax: ax object from plt.subplots()
    """
    
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Assuming `behaviour` is a DataFrame or similar structure with a 'behaviours' column
    unique_behaviours = behaviour.behaviours.unique()
    
    # Create a colormap with the same number of colors as unique behaviors
    cmap = plt.get_cmap('tab20_r', len(unique_behaviours))
    
    # Store plot objects for the legend
    plot_objects = []
    plot_obj=[]
    
    # Loop through the unique behaviors and assign a color and label
    for i, behav in enumerate(unique_behaviours):
        print(behav)
        color = cmap(i)  # Get a color from the colormap
        #if behav in ['approach', 'pursuit', 'attack', 'pullback', 'nesting', 'escape', 'freeze', 'startle', 'switch', 'eat']:
         
        if behav in ['loom', 'introduction', 'turn','pup_drop']:
            plot_obj=    box_plotter(behaviour, behav, color, 'axvline', ax=ax)
            plot_objects.append((plot_obj, behav))
        else:
            
                plot_obj=    box_plotter(behaviour, behav, color, 'box', plotmax, ax)
                plot_objects.append((plot_obj, behav))
            #except:
               
        
        # Append the plot object and label to the list
    
    
    
# Create the legend
   
    proxy_artists = []
    labels = []
 
    for _, label in plot_objects:
         # Create a proxy artist (e.g., a colored rectangle)
         proxy = mpatches.Patch(color=cmap(unique_behaviours.tolist().index(label)), label=label)
         proxy_artists.append(proxy)
         labels.append(label)
     
    #ax.legend(proxy_artists, labels, title="Behaviors")
     
    

    # Show the plot
    plt.show()
 

    return proxy_artists,labels

def plot_events(behaviour, plotmin=None, plotmax=None, ax=None):
    """
    creates shading in figure according to behaviour. If behaviour is point behaviour, 
    axvline is created
    
    parameters
    -----------
    behaviour: The behaviour output file from preprocessing. It should have following colomns:
        behaviours, frames_s, start_stop
    plotmin: in s, from what time onwards  should plotting of behaviours start
    plotmax: in s, until what time should behaviour be plotted
    --> only works if both plotmin AND plotmax are given
    ax: ax object from plt.subplots()
    """
    
    if plotmin is not None:
        behaviour = behaviour[(behaviour['frames_s'] >= plotmin) & (behaviour['frames_s'] <= plotmax)]
        
    
    
    #state behaviours
    box_plotter(behaviour, 'approach', 'tan','box', plotmax, ax)
    box_plotter(behaviour, 'pursuit', 'coral','box', plotmax, ax)
    box_plotter(behaviour, 'attack', 'firebrick','box', plotmax, ax)
    box_plotter(behaviour, 'pullback', 'cadetblue','box', plotmax, ax)
    box_plotter(behaviour, 'nesting', 'green','box', plotmax, ax)
    
    box_plotter(behaviour, 'escape', 'steelblue','box', plotmax, ax)
    box_plotter(behaviour, 'freeze', 'aquamarine','box', plotmax, ax)
    box_plotter(behaviour, 'startle', 'mediumaquamarine','box', plotmax, ax)
    box_plotter(behaviour, 'switch', 'violet','box', plotmax, ax)
    
    box_plotter(behaviour, 'eat', 'slategray','box', plotmax, ax)
    
    
    
    # point behaviours
    box_plotter(behaviour, 'loom', 'seagreen','axvline', ax=ax)    
    box_plotter(behaviour, 'introduction', 'm','axvline', ax=ax)
    
    box_plotter(behaviour, 'turn', 'darkblue','axvline', ax=ax) 
    



def get_start_stop(b_names, frames_s, behaviours, start_stop):
    out=[]
    for b in b_names:
        starts=frames_s[(behaviours==b) & (start_stop=='START')]
        stops=frames_s[(behaviours==b) & (start_stop=='STOP')]
        both=np.array((starts,stops)).T # this is now trials * start, stop 
        out.append(both)
    return out


def get_cmap_colors(cmap_name,num_colors):
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors

def region_ticks(n_region_index, ycoords=None, ax=None, yaxis=True, xaxis=False):
    """
    makes ticks on y axis to mark the regions

    Parameters
    ----------
    n_region_index : in order of plotting, which region does each cluster belong to
        
    ycoords :  optional; if yticks shoulld be on different location than just their index
        

    Returns
    -------
    None.

    """
    yticks = [(i, region) for i, region in enumerate(n_region_index) if i == 0 or n_region_index[i-1] != region]
    
    indices, labels = zip(*yticks)
    indices=np.array(indices)
    if ax is None:
        ax=plt.gca()
    if ycoords is not None:
        
        tick_locations=ycoords[indices]
    else:
        tick_locations=indices
    
    if yaxis:
        ax.set_yticks(tick_locations, labels)
    
    if xaxis:
        ax.set_xticks(tick_locations, labels)
    
    return tick_locations







def remove_axes(axis=None, top=True, right=True, bottom=False, left=False, ticks=True, rem_all=False):
    if axis is None:
        axis = plt.gca()  
    if rem_all:
        ticks=False
    
    if isinstance(axis, matplotlib.axes.Axes):
        axs=[axis]
    else:
        axs=axis
        
    for ax in axs:
        if top or rem_all:
            ax.spines['top'].set_visible(False)
        if right or rem_all:
            ax.spines['right'].set_visible(False)
        if bottom or rem_all:
            ax.spines['bottom'].set_visible(False)
            if not ticks:
                ax.set_xticks([])
        if left or rem_all:
            ax.spines['left'].set_visible(False)
            if not ticks:
                ax.set_yticks([])





def make_cmap(colors, values):

    norm = mcolors.Normalize(min(values), max(values))
    tuples = list(zip(map(norm, values), colors))
    cmap = mcolors.LinearSegmentedColormap.from_list("", tuples)
    return cmap





def plot_tuning(tuning_matrix, target_bs,n_region_index, cmap='viridis', vmin=None, vmax=None, lines=True, area_labels=True):    
    plt.figure(figsize=(13,20))
    plt.imshow(tuning_matrix,
               vmin=vmin,
               vmax=vmax,
               aspect='auto',
               cmap=cmap)
    remove_axes()
    plt.xticks(np.arange(len(target_bs)),target_bs)
    cbar=plt.colorbar()
    if lines:
        plt.axvline(2.5, c='k')
        plt.axvline(3.5, c='k',ls='--')
        plt.axvline(6.5, ls='--',c='k')
        plt.axvline(7.5, c='k')
    if area_labels:
        region_ticks(n_region_index)
    return cbar




def logplot(Y):
    plt.plot(np.arange(1,len(Y)+1),Y)
    plt.xscale('log')
    
def subplots(n, rem_axes=True, figsize=None, gridspec=False):
    """
    creates subplots objet that has optimal layout
    also removes top and right axes 

    Parameters
    ----------
    n : int
        total number of subplots.
    rem_axes : bool, optional
        whether to remove top and right axis. The default is True.
    gridspec: whether instead of a new figure, a gridspec object should be returned
        This is useful for subdividing subplots, necessary e.g. for the PSTH function
    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    # Calculate the grid size: find two factors of n that are as close together as possible
    rows = math.floor(math.sqrt(n))
    while n % rows != 0:
      rows -= 1
    cols = n // rows

    # If n is a prime number, make the number of rows and columns as equal as possible
    if rows == 1:
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / rows)

   # Create the subplots
    if gridspec:
       fig=plt.figure(figsize=figsize)
       gs=GridSpec(rows, cols)
       return rows, cols, gs, fig
   
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs=axs.flatten()
    if rem_axes:
        remove_axes(axs)
    
    #remove unused axes
    if rows * cols > n:
        for i in range(n, rows * cols):
            fig.delaxes(axs[i])
   

    return fig, axs