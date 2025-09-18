"""
Created by Tom Kern
Last modified on 05/08/24

Plots all neural activity across a session + all the behaviours
- Lineplot: neural activity as lines, zscored
- dotplot: each spike as dot, 10ms resolution
- imshow_plot: Neural activity, either zscored or not, colored

"""
import cupy as cp
from tqdm import tqdm
import IPython
import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import preprocessFunctions as pp
import os
import matplotlib
# Parameters
animal='afm16924'
sessions=['240525']
target_regions =['DPAG','VPAG','VLPAG','LPAG','DLPAG','DMPAG','MPAG']  # From which regions should the neurons be? Leave empty for all.
target_cells=[34,347]
lineplot=True
rolling_avg=True
dotplot=False
imshowplot=False
plot_by_base=False
close_plots=False
plt.style.use('default')
plt.close('all')

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
       
def get_inst_FR(n_spike_times):    
    max_value = 250
    threshold = 200
    
    while max_value > threshold:        
       
        isi_array = []
        firing_rates = []
        
        #1st pass
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
        max_value = np.max(np.concatenate(firing_rates))
    firing_rates_array = np.array(iFR, dtype=object)  # Convert to numpy array
    return firing_rates,firing_rates_array,n_spike_times
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




    
def binned_iFR_vec(x_vals, y_vals, bin_width=0.01, rolling_avg=False):
    """
    Bin instantaneous firing rates (y_vals) using bins of width `bin_width`
    determined by the x_vals timestamps.
    """
    # Create bin edges and compute bin centers.
    bins = np.arange(x_vals.min(), x_vals.max() + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    # Compute sum of y values per bin and count of values per bin.
    sum_y, _ = np.histogram(x_vals, bins=bins, weights=y_vals)
    count, _ = np.histogram(x_vals, bins=bins)

    # Compute the mean firing rate in each bin.
    with np.errstate(divide='ignore', invalid='ignore'):
        binned_y = sum_y / count
    binned_y[count == 0] = np.nan  # Handle empty bins

    # Optionally apply a rolling average via convolution that ignores NaNs.
    if rolling_avg:
        window_length = 3  # Example: rolling average over 5 bins.
        window = np.ones(window_length)

        # Create a mask: 1.0 for valid values, 0.0 for NaN.
        valid_mask = np.where(np.isnan(binned_y), 0, 1)

        # Replace NaNs with 0 in binned_y for convolution.
        binned_y_zeroed = np.where(np.isnan(binned_y), 0, binned_y)

        # Compute weighted sums and counts in the rolling window.
        smoothed_sum = np.convolve(binned_y_zeroed, window, mode='same')
        valid_counts = np.convolve(valid_mask, window, mode='same')

        # Avoid division by 0: where valid_counts==0, the result will be set to NaN.
        with np.errstate(invalid='ignore', divide='ignore'):
            binned_y = smoothed_sum / valid_counts
        binned_y[valid_counts == 0] = np.nan
        
    return binned_y, bin_centers

def binned_iFR(x_vals,y_vals,bin_width=0.01,rolling_avg=False):
    """
    x_vals = np.array([...])  # timestamps in seconds
    y_vals = np.array([...])  # instantaneous firing rates
    
    bin iFR into "bin_width" (s)  bins
    """
    #
    # Define 10 ms bin width (in seconds)
    #bin_width = 0.01
    
    # Create bin edges from the minimum to maximum of x_vals
    bins = np.arange(x_vals.min(), x_vals.max() + bin_width, bin_width)
    window_length = 5   # Example: rolling average over 5 bins
    
    # Digitize x_vals to determine bin indices (bins are 1-indexed)
    bin_indices = np.digitize(x_vals, bins)
    bin_centers = bins[:-1] + bin_width/2
    binned_y = []
    if rolling_avg: # Option to apply a rolling average (moving average) filter
        for i in range(1, len(bins)):
            cur_bin = np.where(bin_indices == i)[0]
            if cur_bin.size > 0:
                binned_y.append(y_vals[cur_bin].mean())
            else:
                binned_y.append(np.nan)
        binned_y = np.array(binned_y)
        # Create a window for moving average (uniform weights) and normalize:
        window = np.ones(window_length) / window_length
        
        # Use 'same' mode to get an array that has the same length as binned_y.
        # Note: The edges may be less reliable because of the effect of the convolution.
        smoothed_y = np.convolve(binned_y, window, mode='same')
        # smoothed_y now contains the binned_y values after applying the rolling average.
        binned_y=smoothed_y
    else: # Compute the average y value in each bin
        
       
        for i in range(1, len(bins)):
            # Find indices for this bin (if no values fall in the bin, we use np.nan)
            cur_bin = np.where(bin_indices == i)[0]
            if cur_bin.size > 0:
                binned_y.append(y_vals[cur_bin].mean())
            else:
                binned_y.append(np.nan)
        # binned_y now contains the average instantaneous firing rate for each 10 ms bin.
    
    return binned_y,bin_centers
            
    
def fullscreen():
    
    mng = plt.get_current_fig_manager()
    if matplotlib.get_backend()=='tkagg':        
        mng.window.state('zoomed') 
    else:
        mng.full_screen_toggle()  # Toggle full screen mode
    
for session in sessions:
    print(f"loading preprocessed data for {session}" )
    paths=pp.get_paths(animal, session)    
    output_path=paths['preprocessed']
    # load data
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
    
    print(f"calculating intananeous firing rate" )
    iFR,iFR_array,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate
    
    
    # # Get the overall maximum value
       
    
    spike_res=0.001
    FR_res=0.02
    print(f"recalculating ndata")
    #n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=spike_res)
    n_time_index, ndata, firing_rate_bins_time,firing_rates,neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds=hf.recalculate_ndata_firing_rates2(n_spike_times,
    bin_size=spike_res, firing_rate_bin_size=FR_res)
    
    
 #   print(f"1 {np.shape(n_time_index)=}, {np.shape(ndata)=}")
    if len(target_cells)!=0: # reduce to target cells
       sorted_indices = [index for index, value in enumerate(n_cluster_index) if value in target_cells]
      
       n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)       
    elif len(target_regions) != 0:# #sort by region
        in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
        n_region_index = n_region_index[in_region_index]    
        sorted_indices = np.argsort(n_region_index,axis=0)        
        n_region_index,n_cluster_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)
 
  
    # iFR,firing_rates_array,n_spike_times=get_inst_FR(n_spike_times)#instananous firing rate


    # # Get the overall maximum value
   
    # IPython.embed()
    # res=0.001
    # print(f"recalculating ndata")
    # n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=res)
    
    
    
    
    # if len(target_regions) != 0:  # If target brain region(s) specified    
    #     in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
    #     n_region_index = n_region_index[in_region_index]    
    # else:
    #     pass
    #     #n_region_index = np.arange(len(n_region_index))
    
    base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)# baseline to first 7 minutes or before first behavior
#    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
    base_mean = np.round(base_mean, 2)    
    base_mean = base_mean[:, np.newaxis]  # now shape is (N, 1)
    base_mean = np.where(base_mean == 0, 1/10000, base_mean)
    
    
    
    

    
    if plot_by_base:
    #def plot_firing_rate_sorted_by_base_mean(ndata, n_time_index, behaviour, n_cluster_index, target_cells, base_mean, n_region_index, output_path):
          
            resolution = 0.1
            resampled_ndata, resampled_timestamps = hf.resample_ndata(ndata, n_time_index, resolution)
            frng_rate = resampled_ndata / resolution
        
            # Sort neurons by base_mean
            sorted_indices = np.argsort(base_mean,axis=0)
            sorted_frng_rate = frng_rate[sorted_indices]
            sorted_cluster_index = n_cluster_index[sorted_indices]
        
            #fig = plt.figure(figsize=(20, 10))
            
            fig, axes = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 5]}, sharex=True) #sharex=True
            ax_vel = axes[0]  # Velocity plot on top
            ax_fr = axes[1]  # Firing rate plot below

#            fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [5, 1]})
          

            ## Plot Velocity            
            proxy_artists, labels = pf.plot_events_shahaf(behaviour,ax=ax_vel)
            v_time = np.linspace(0, frame_index_s[-1], len(velocity))
            ax_vel.plot(v_time, velocity, lw=1, color='k')
            ax_vel.set_xlabel('Time (s)')
            ax_vel.set_ylabel('Speed cm/s')
            
            

#            ax = fig.add_subplot(111)
          
            ax_fr.set_title(f'{animal} {session} zscored firing; Units sorted by base mean, {int(resolution*1000)} ms bins')

        
            
        
            ycoords = np.linspace(0, len(sorted_frng_rate) * 10, len(sorted_frng_rate)) * -1

         
            
            #proxy_artists, labels = pf.plot_events_shahaf(behaviour)

        
            for i, n in enumerate(sorted_frng_rate):
               
                n = n.reshape(-1)
                FR = base_mean[sorted_indices[i]].item()
                cluster=sorted_cluster_index[i].item()
                
                n = n / FR           
                n = zscore(n, nan_policy='omit') 
                n = n / np.nanmax(n) 
        
                if cluster in target_cells:
                    cell_color = 'r'
                else:
                    cell_color = 'k'
        
                ax_fr.plot(resampled_timestamps, n+ ycoords[i], lw=0.5, color=cell_color)
                fstring = f"{cluster}, {FR:.1f}Hz"
                ax_fr.text(-50, ycoords[i], fstring, va='top', color='r', fontsize=10)
        
            #pf.region_ticks(n_region_index, ycoords=ycoords)
            #plt.show()
            
            #plt.xlabel('time [s]')
            ax_vel.legend(
                proxy_artists, labels,  loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=len(labels)
            )        
            
            ## plot firing rates
            ax_fr.set_xlabel('Time (s)')
            ax_fr.set_ylabel('firing rate (zscored)')            
            ax_fr.set_ylim(np.nanmin(ycoords), np.nanmax(ycoords))
            ax_fr.set_yticklabels([])
            ax_fr.set_yticks([])

          #  ax_vel.set_xticks(ax_fr.get_xticks())
            #ax_vel.set_xticklabels(ax_fr.get_xticklabels())
            
            

            
            pf.remove_axes(axis=ax_fr,top=True,right=True,left=True,bottom=False) # Apply remove_axes to the velocity subplot as well        
            pf.remove_axes(axis=ax_vel,top=True,right=True,left=False,bottom=False)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            plt.show()
        
           
        
            figfilename = f"entire_session_line_plot_sorted_by_base_mean.svg"
            full_path = os.path.join(output_path, figfilename)
            plt.savefig(full_path)
            print(f'saved line plot sorted by base mean to:\n {full_path}')    
            
    
       # plot_firing_rate_sorted_by_base_mean(ndata, n_time_index, behaviour, n_cluster_index, target_cells, base_mean, n_region_index, output_path)
        #%% Line plot
        # plt.close('all')
        
    if lineplot:#sorted by brain region
        #resolution=.1
        #resampled_ndata, resampled_timestamps =hf.resample_ndata(ndata, n_time_index, resolution)
        #frng_rate = resampled_ndata / resolution
        frng_rate = iFR
        
        
        
        sorted_frng_rate=[]
        xtimes=[]
        sorted_indices = np.argsort(n_region_index,axis=0)
        sorted_cluster_index = n_cluster_index[sorted_indices]
        ifr_count=-1
      
        for ifr_i in sorted_indices:
            sorted_frng_rate.append(frng_rate[ifr_i])
            ifr_count+=1
            xtimes.append(n_spike_times[ifr_i])
       
    
        
    
        fig, axes = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 5]}, sharex=True) #sharex=True
        ax_vel = axes[0]  # Velocity plot on top
        ax_fr = axes[1]  # Firing rate plot belowr

#            fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [5, 1]})
      

        ## Plot Velocity            
        proxy_artists, labels= pf.plot_events_shahaf(behaviour,ax=ax_vel)
        v_time = np.linspace(0, frame_index_s[-1], len(velocity))
        ax_vel.plot(v_time, velocity, lw=1, color='k')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Speed cm/s')
        ax_vel.legend(
            proxy_artists, labels,  loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=len(labels)
        )   
        
        

#            ax = fig.add_subplot(111)
        bin_width=0.1
        ax_fr.set_title(f'{animal} {session} Instantaneous Firing rate:\n Units sorted by region\n zscored, time bin={bin_width*1000:0.0f} ms ')
        
        ycoords = np.linspace(0, len(sorted_frng_rate) * 10, len(sorted_frng_rate)) * -1
    
        
    
        for i, n in enumerate(tqdm(sorted_frng_rate,desc='plotting cells')):
          # n = np.insert(n, 0, 0)  # Insert 0 first timestamp(cuz it's based on a diff)
           x_vals = xtimes[i] #timestamps
           cluster=sorted_cluster_index[i].item()
           if len(n)==0:
               continue#skip line if no firing
           
           #n = n.reshape(-1)#why?
           spike_times=n_spike_times[sorted_indices[i]][1:]
           start_time=np.nanmin(n_spike_times[sorted_indices[i]])#seconds
           stop_time=60*7 # seconds#np.nanmax(n_spike_times)
           valid_indices = np.where((spike_times >= start_time) & (spike_times < stop_time))
           #FR = base_mean[sorted_indices[i]].item()
           
           
           #n=n[valid_indices]
           
           #n = n / FR           
          # IPython.embed()
           n,x_vals= binned_iFR_vec(x_vals=x_vals,y_vals=n, bin_width=bin_width, rolling_avg=rolling_avg)
       
          
           n = zscore(n, nan_policy='omit') 
          # n[np.isnan(n)] = 0
          # n = n / np.nanmax(n) 
         #  IPython.embed()
           if cluster in target_cells:
               cell_color = 'r'
           else:
               cell_color = 'k'
   
           ax_fr.plot(x_vals, n+ ycoords[i], lw=0.5,color=cell_color, linestyle='--',marker = ".", markersize=2)
           fstring = f"{cluster}, {np.mean(n):.1f}Hz"
           ax_fr.text(-50, ycoords[i], fstring, va='top', color='b', fontsize=10)
           y_vals = n + ycoords[i]
           
           # ### Create boolean masks
           # pos_mask = n > 1
           # neg_mask = n < -1
           # # Plot negative n values in red
           # ax_fr.plot(
           # x_vals[neg_mask],
           # y_vals[neg_mask],
           # color='red',
           # linestyle='--',
           # marker=".",
           # markersize=2,
           # lw=0.01
           # )
           
            
           # # Plot positive n values in blue
           # ax_fr.plot(
           # x_vals[pos_mask],
           # y_vals[pos_mask],
           # color='green',
           # linestyle='--',
           # marker=".",
           # markersize=2,
           # lw=0.01
           # )
            
         
            
            
           
            
          
           
           
        
          
   
        pf.region_ticks(n_region_index[sorted_indices], ycoords=ycoords)        # add brain region Yticklabels
        
        #plt.xlabel('time [s]')             
        
        ## plot firing rates
        ax_fr.set_xlabel('Time (s)')
        ax_fr.set_ylabel('firing rate (zscored)')            
        ax_fr.set_ylim(np.nanmin(ycoords), np.nanmax(ycoords))
       # ax_fr.set_yticklabels([])
        #ax_fr.set_yticks([])

        
        pf.remove_axes(axis=ax_fr,top=True,right=True,left=False,bottom=False) # Apply remove_axes to the velocity subplot as well        
        pf.remove_axes(axis=ax_vel,top=True,right=True,left=False,bottom=False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])        
        plt.show()
       # IPython.embed()
      
        
        
           
    
        print('saving...')
        figfilename=f"entire_session_line_plot_instFR_binned.png"
        if rolling_avg:
            figfilename=f"entire_session_line_plot_instFR_binned_rlavg.png"
        #figfilename=f"entire_session_line_plot_instFR_binned.pdf"
        full_path = os.path.join(output_path, figfilename)
        #plt.savefig(full_path,dpi=100)
        plt.savefig(full_path)
        print(f'saved line plot to: {full_path}')
 #       plt.show();IPython.embed()
        
       
        
####################################      
    
    #% dot plot
    if dotplot==True:
        
        plot_n=ndata.copy()
        ax=plt.figure(figsize=(30,15))
        #fullscreen()
        plt.title('dot=spike')
       # pf.plot_events(behaviour)
      
        proxy_artists, labels=pf.plot_events_shahaf(behaviour) 
         # Create the legend using proxy artists
        #ax.legend(proxy_artists, labels, title="Behaviors")
        ax.legend(
    proxy_artists, labels,
    title="Behaviors",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.05),frameon=False,
    ncol=len(labels)  # Adjust the number of columns as needed
    )
    
        # get dot indices 
        
        ycoords=np.linspace(0,len(plot_n)*4,len(plot_n))*-1
        for i, n in enumerate(plot_n):
            spikeind=n.astype(bool)
            spiketime=n_time_index[spikeind]
         
            plt.scatter(spiketime, np.zeros_like(spiketime)+ycoords[i], color='k', s=1)
    
        pf.region_ticks(n_region_index, ycoords=ycoords)
        
       
       
        pf.remove_axes()
        plt.show()
        figfilename=f"entire_session_dot_plot.svg"

        full_path = os.path.join(output_path, figfilename)
        plt.savefig(full_path)
        print('saved dot plot')
    
    
    #%% imshow plot
    
    if imshowplot:
        cmap= pf.make_cmap(('b','w','w','r'), 
                           (-3,-.9,.9,3))
        zscoring=True
        
        resolution=.1
        resampled_ndata, resampled_timestamps =hf.resample_ndata(ndata, n_time_index, resolution)
        
        
        
        # plt.title('s are wrong!!')
        fig, axs=plt.subplots(2,1, sharex=True,figsize=(20, 10))
       # fullscreen()
        
        if zscoring:
            firing=zscore(resampled_ndata/resolution, axis=1)
            vmin=-3
            vmax=3
            cmap='seismic'
            fig.suptitle(f'{session}\n resolution: {resolution}s')
            cmap= pf.make_cmap(('b','w','w','r'), 
                               (-3,-.0,.0,3))
            
        elif not zscoring:
            firing = resampled_ndata/resolution
            vmin=0
            vmax=np.percentile(firing,95)
            fig.suptitle(f'{session} resolution: {resolution}s\nfiring')
            cmap='binary'
        
        
        pf.plot_events_shahaf(behaviour, ax=axs[0])
        proxy_artists, labels=pf.plot_events_shahaf(behaviour) 
        #axs[0].legend(proxy_artists, labels, title="Behaviors")
        axs[0].legend(
    proxy_artists, labels,
    title="Behaviors",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.05),frameon=False,
    ncol=len(labels)  # Adjust the number of columns as needed
    )
        
        axs[0].plot(frame_index_s[:-frames_dropped],velocity[0:len(frame_index_s[:-frames_dropped])], c='k', lw=.5, label='velocity')
        axs[0].set_ylabel('velocity')
        pf.remove_axes(axs)
    #    hf.unique_legend(axs[0])
        
        figcolors=axs[1].imshow(firing,
                      aspect='auto',
                      origin='upper', # index to n_region_index is reverse to channel number
                      extent=[resampled_timestamps.min(),
                              resampled_timestamps.max(),
                              len(firing),
                              0],
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmap)
        
        pf.region_ticks(n_region_index)
        cbar=plt.colorbar(figcolors,ax=[axs[0],axs[1]])
        
        if zscoring:
            cbar.set_label('zscore')
        else:
            cbar.set_label('firing [Hz]')
        
        plt.xlabel('time [s]')
        plt.ylabel('firing rate (zscored)')
       # ax.tight_layout()
        plt.show()
        figfilename=f"entire_session_cmap.svg"
        full_path = os.path.join(output_path, figfilename)
        plt.savefig(full_path,bbox_inches='tight')
        print('saved cmap')
        
    if close_plots:
        plt.close('all')
        
        
