# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:15:04 2025

@author: su-weisss
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:47:53 2025

@author: su-weisss
"""
animal = 'afm16924';sessions =['240523_0']
from pathlib import Path 
import matplotlib
import spikeinterface
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
from spikeinterface.exporters import export_report
from probeinterface import Probe, get_probe
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from pathlib import Path
import IPython
job_kwargs=si.get_best_job_kwargs();
from typing import Union
import time
import spikeinterface_helper_functions as sf
import helperFunctions as hf                
import preprocessFunctions as pp               
#%matplotlib inline


def find_problematic_periods_in_LFP(spikeglx_folder):
    
    # import matplotlib
    # import spikeinterface
    # import spikeinterface.full as si
    # import spikeinterface.extractors as se
    # import spikeinterface.sorters as ss
    # import spikeinterface.comparison as sc
    # import spikeinterface.widgets as sw
    # from spikeinterface.exporters import export_report
    # from probeinterface import Probe, get_probe
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.ion()
    # from pathlib import Path
    # import IPython
    # job_kwargs=si.get_best_job_kwargs();
    # from typing import Union
    # import time
    # import spikeinterface_helper_functions as sf
    # import helperFunctions as hf
    
    spikeglx_folder=Path(spikeglx_folder).as_posix()
    try:
        recording = si.read_spikeglx(spikeglx_folder,stream_id='imec0.lf',load_sync_channel=False)
        
    except:
        print(f"\ndidnt find LF stream, switiching to ap\n")
        recording = si.read_spikeglx(spikeglx_folder,stream_id='imec0.ap',load_sync_channel=False)
        
    print(recording)
    
    t_start=recording.get_time_info()['t_start']
    recording.shift_times(shift=-t_start)
    t_start=recording.get_time_info()['t_start']
    
    #recording=recording.time_slice(t_start+3150, t_start+3200)
    
    t_stop=recording.get_duration()
    channel_ids = recording.get_channel_ids()
    selected_channels = channel_ids[::50]
    #selected_channels = [channel_ids[0]] #take only 1 channel for debugging
    print(F"using N={len(selected_channels)} channels")
    

    
    #recording = sf.preprocess_NP_rec(spikeglx_folder,kwargs,stream_id='')
    recording = si.phase_shift(recording)
    recording = si.highpass_filter(recording,300)
    # recording = si.bandpass_filter(
    #     recording,
    #     freq_min=300,
    #     freq_max=1250,
    
    #     margin_ms=1500.,
    #     filter_order=3,
    #     dtype="float32",
    #     add_reflect_padding=True,
    # )
    
    
    resample_rate=1250
    print(f"downsampling to {resample_rate}Hz...\n")
    recording = si.resample(recording, resample_rate=resample_rate, margin_ms=1000)
    from spikeinterface.preprocessing import scale_to_uV
    recording = scale_to_uV(recording)
    #physical_value = raw_value * gain + offset
    
    # Convert to physical units (whatever they may be)
    #recording_physical = scale_to_physical_units(recording)
    
    #sw.plot_traces(recording,time_range=(t_start+3175,t_start+3195),mode='line',channel_ids=selected_channels)
    fs= recording.get_sampling_frequency()
    print(f"getting samples...")
    all_traces = recording.get_traces(channel_ids=selected_channels, return_scaled=True)
    print(f"scanning for zeros and saturations...")
    
    #
    
    
    def find_zeros_for(all_traces,saturation_threshold = 2000,N_samples=10):
        from itertools import groupby
        from operator import itemgetter
        import numpy as np
        
        # Compute the mean across channels for each sample (axis=1)
        mean_traces = all_traces.mean(axis=1)
        
        # Find indices where the mean is exactly zero
        zero_or_above2000_inds = np.where((mean_traces == 0) | (mean_traces > saturation_threshold))[0]
       # zero_mean_inds = np.where(mean_traces == 0)[0]
        
        # Find runs of at least 10 consecutive zeros
        groups = []
        for k, g in groupby(enumerate(zero_or_above2000_inds), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            if len(group) >= N_samples:
                groups.append(group)
        
        # Find indices where after a run of zeros, the next sample is non-zero
        after_zero_runs = []
        for group in groups:
            last_idx = group[-1]
            if last_idx + 1 < len(mean_traces) and mean_traces[last_idx + 1] != 0:
                after_zero_runs.append(last_idx + 1)
        
        # Find margins: 3000 samples before the first zero and after the last zero in each group
        margin_in_samples=3000
        margins = []
        n_samples = len(mean_traces)
        for group in groups:
            first = group[0]
            last = group[-1]
            start_margin = max(0, first - margin_in_samples)
            end_margin = min(n_samples - 1, last + margin_in_samples)
            margins.append((start_margin, end_margin))
            margins_seconds = [(start / fs, end / fs) for start, end in margins]
           
        return margins_seconds
    
    def find_zeros_vectorized_GPT41(all_traces, fs=recording.get_sampling_frequency(),saturation_threshold = 2000,N_samples=30,margin_in_samples=0):
        
        mean_traces = all_traces.mean(axis=1)
        zero_or_above2000 = (mean_traces == 0) 
        idx = np.flatnonzero(zero_or_above2000)
    
        # Find runs of at least 10 consecutive indices
        if idx.size == 0:
            return []
    
        # Find the breaks between runs
        breaks = np.where(np.diff(idx) != 1)[0]
        run_starts = np.insert(idx[breaks + 1], 0, idx[0])
        run_ends = np.append(idx[breaks], idx[-1])
    
        # Filter runs by length >= 10
        valid = (run_ends - run_starts + 1) >= N_samples
        run_starts = run_starts[valid]
        run_ends = run_ends[valid]
    
       
        n_samples = len(mean_traces)
        margins = []
        for start, end in zip(run_starts, run_ends):
            start_margin = max(0, start - margin_in_samples)
            end_margin = min(n_samples - 1, end + margin_in_samples)
            margins.append((start_margin, end_margin))
    
        margins_seconds = [(start / fs, end / fs) for start, end in margins]
        return margins_seconds
    
    def find_zeros_and_saturations_vectorized_GPT41(
        all_traces, 
        fs, 
        saturation_threshold=2000, 
        return_threshold=50, 
        N_samples=30, 
        margin_in_samples=10
    ):
        """
        Find margins (in seconds) around:
        1. Runs of zeros in mean_traces (length >= N_samples)
        2. Runs where abs(mean_traces) exceeds saturation_threshold, until it returns below return_threshold
    
        Returns:
            margins_seconds_zeros: list of (start, end) in seconds for zero runs
            margins_seconds_saturation: list of (start, end) in seconds for saturation runs
        """
        
        mean_traces = all_traces.mean(axis=1)
        zero_or_above2000 = (mean_traces == 0) 
        idx = np.flatnonzero(zero_or_above2000)
    
        # Find runs of at least 10 consecutive indices
        if idx.size == 0:
            return []
    
        # Find the breaks between runs
        breaks = np.where(np.diff(idx) != 1)[0]
        run_starts = np.insert(idx[breaks + 1], 0, idx[0])
        run_ends = np.append(idx[breaks], idx[-1])
    
        # Filter runs by length >= 10
        valid = (run_ends - run_starts + 1) >= N_samples
        run_starts = run_starts[valid]
        run_ends = run_ends[valid]
    
       
        n_samples = len(mean_traces)
        margins = []
        for start, end in zip(run_starts, run_ends):
            start_margin = max(0, start - margin_in_samples)
            end_margin = min(n_samples - 1, end + margin_in_samples)
            margins.append((start_margin, end_margin))
    
        margins_seconds = [(start / fs, end / fs) for start, end in margins]
    
        # --- Saturation ---
        abs_trace = np.abs(mean_traces)
        above = abs_trace > saturation_threshold
        margins_saturation = []
        n_samples = len(mean_traces)
        i = 0
        while i < n_samples:
            if above[i]:
                start = i
                # Move forward until abs(mean_traces) < return_threshold
                while i < n_samples and abs_trace[i] > return_threshold:
                    i += 1
                end = i - 1
                # Add margin
                start_margin = max(0, start - margin_in_samples)
                end_margin = min(n_samples - 1, end + margin_in_samples)
                margins_saturation.append((start_margin, end_margin))
            else:
                i += 1
        margins_seconds_saturation = [(start / fs, end / fs) for start, end in margins_saturation]
    
        return margins_seconds, margins_seconds_saturation
    
    def check_overlap(margins_second):
        # First, sort the intervals by their start times.
        sorted_intervals = sorted(margins_second, key=lambda x: x[0])
        
        merged_intervals = []
        for interval in sorted_intervals:
            # If there are no intervals in merged_intervals, simply add the interval.
            if not merged_intervals:
                merged_intervals.append(interval)
                continue
        
            prev_start, prev_end = merged_intervals[-1]
            curr_start, curr_end = interval
            
            # Check for overlap: if current start is less than or equal to the previous end,
            # they overlap and we need to merge.
            if curr_start <= prev_end:
                # Merge by updating the end to the maximum of both intervals' ends.
                merged_intervals[-1] = (prev_start, max(prev_end, curr_end))
            else:
                # No overlap, so append the current interval as is.
                merged_intervals.append(interval)
        
        #print("Merged intervals:", merged_intervals)
        return merged_intervals
    
    #start_time = time.time()
    #margins_second= find_zeros_vectorized_o3(all_traces,fs)
    #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n{margins_second[0]}")
    #start_time = time.time()
    margins_second= find_zeros_vectorized_GPT41(all_traces,fs)
    margins_seconds_zeros, margins_seconds_saturation = find_zeros_and_saturations_vectorized_GPT41(
        all_traces, 
        fs, 
        saturation_threshold=800, 
        return_threshold=50, 
        N_samples=10, 
        margin_in_samples=10
    )
    
    all_margins=np.sort(margins_seconds_saturation+margins_second)#merge to a single list, sort by time
    all_margins=check_overlap(all_margins)#remove overlaps
    
    #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n")
    #start_time = time.time()
    #margins_second= find_zeros_for(all_traces)
    #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n{margins_second[0]}")
    
    
      
    
    # Create arrays for each field
    
    times = np.array([float(pair[0]) for pair in margins_seconds_zeros])#only start times
    zero_all_times = np.array([[float(pair[0]), float(pair[1])] for pair in margins_seconds_zeros])#also stop times
    zeros_duration = np.array([float(pair[1] - pair[0]) for pair in margins_second])
    label = np.array(["zeros"] * len(margins_seconds_zeros))
    zeros_duration_sum=np.sum(zeros_duration)
    # Assemble the dictionary
    zero_dict = {"time": time, "duration": zeros_duration, "label": label}
    
    
    
    sat_times = np.array([float(pair[0]) for pair in margins_seconds_saturation])#only start times
    sat__all_times = np.array([[float(pair[0]), float(pair[1])] for pair in margins_seconds_saturation])#also stop times
    sat_duration = np.array([float(pair[1] - pair[0]) for pair in margins_seconds_saturation])
    label = np.array(["saturations"] * len(margins_seconds_saturation))
    
    sat_dict= {"time": sat_times, "duration": sat_duration, "label": label}
    #all_values = [value+offset for pair in margins_second for value in pair]
    
    
    
    
    
    duration = recording.get_duration()
    
    # all_values = np.array([value for pair in margins_second for value in pair])
    # sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]],events=all_values,events_color='red',events_alpha=.5,return_scaled=True)
    
    # all_values = np.array([value for pair in margins_seconds_zeros for value in pair])
    # sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]],events=all_values,events_color='green',events_alpha=.5,return_scaled=True)
    
    all_values = np.array([value for pair in all_margins for value in pair])
    
    zeros_duration = np.array([float(pair[1] - pair[0]) for pair in all_margins])
    zeros_duration_sum=np.sum(zeros_duration)
    zeros_duration_sum/duration
    
    
    #plot behaviors on traces timeline
    sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]]
                    ,events=all_values,events_color='blue',events_alpha=.8,return_scaled=True)
    plt.savefig(Path.joinpath(out_path, 'bad_times.png')) 
    plt.show()
    
    #plot behaviors on traces timeline
    try:
     plt.close('all')
     frames_dropped, behaviour, ndata,n_spike_times, n_time_index, n_cluster_index,n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s=hf.load_preprocessed(animal, session)
     behaviour.iloc[0].video_start_s 
     behavior_times=np.array(behaviour.iloc[:].frames_s)   
     sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]]
                   ,events=np.array(behavior_times),events_color='red',events_alpha=.8,return_scaled=True)
     plt.savefig(Path.joinpath(out_path, 'behavior_times.png')) 
     plt.show()
    except:
        pass
    
    plt.close('all')
    step = 21
    T = np.arange(0, duration + step, step)
    
    # Suppose all_margins might have 1D arrays
    
    shared_bin_indices = []
    
    for margins in all_margins:
        # Ensure margins is at least 2D.
        margins2d = np.atleast_2d(margins)
        
        # Check that the second dimension is even so we can reshape it into pairs.
        if margins2d.shape[1] % 2 != 0:
            raise ValueError("The number of elements isn't even; cannot form start-stop pairs.")
        
        # Reshape each array so that each row represents an event [start, stop]
        events = margins2d.reshape(-1, 2)
        
        for event in events:
            start, stop = event
            
            # Determine bin indices for start and stop times.
            bin_index_start = np.digitize(start, T) - 1
            bin_index_stop  = np.digitize(stop, T) - 1
            
            # Only if both times fall in the same bin do we record the bin index.
            if bin_index_start == bin_index_stop:
                shared_bin_indices.append(bin_index_start)
    shared_bin_indices=np.unique(shared_bin_indices)
    #print("Bin indices for events contained within a single bin:", shared_bin_indices)
    print({spikeglx_folder})
    print(f"\n duration:{duration:.2f}\n ")
    print(f"\n lost seconds:{zeros_duration_sum:.2f}\n ")
    print(f"\n lost seconds/duration :{zeros_duration_sum/duration:.5f}\n ")
    print(f"\n {len(shared_bin_indices)} chunks, {len(shared_bin_indices)*step} seconds of data will be lost \n")
    plt.show(block=False)
    
    return all_margins



###############################################################################

for session in sessions:
    paths=pp.get_paths(session=session,animal=animal)
    spikeglx_folder=Path(paths['lf']).parent
    out_path =Path(paths['preprocessed'])

    print(spikeglx_folder)
    all_margins = find_problematic_periods_in_LFP(spikeglx_folder)
    
    np.save(Path.joinpath(out_path, 'bad_times.npy'),all_margins)
    
    
    
    
    
    
    
  
# #sw.plot_traces(recording,time_range=(t_start+3150,t_start+3200),mode='map',return_scaled=True,order_channel_by_depth=True)
# # noise_levels_microV = si.get_noise_levels(recording, return_scaled=True,**job_kwargs)
# # fig, ax = plt.subplots()
# # A = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
# # ax.set_xlabel('noise  [microV]')
# import IPython; IPython.embed()
# channel_ids = recording.get_channel_ids()
# selected_channels = channel_ids[::50]
# selected_channels
# # Get sampling frequency
# fs = recording.get_sampling_frequency()
# fs
# # Duration in seconds



# import numpy as np
# import cupy as cp
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from scipy import signal  # optional, if you want to compare with scipy.signal.spectrogram

# print('FFTing')
# # -------------------------------------------------------------------
# # User-defined parameters
# t0=t_start+3175
# tend=t_start+3195
# bin_size = 0.05        # window length (in seconds) for FFT computation
# freq_band = (0.01, 4)  # frequency band of interest (in Hz)
# chunk_duration = 1     # each chunk is 2 seconds long

# # Derived parameters
# chunk_length = int(fs * chunk_duration)  # number of samples per chunk
# NFFT = int(fs * bin_size)                # number of samples per FFT window
# pad_to = 1500#1500                            # pad FFT to this length
# channel_downsample_factor=150
# selected_channels = channel_ids[::channel_downsample_factor]
# selected_channels = [channel_ids[0]]

# # -------------------------------------------------------------------
# # Example: Fetch all traces at once from recording using selected_channels.
# # Replace this with your recording fetching code.
# # For demonstration, let's generate dummy data:

# # Generate random data for each channel

# short_rec=  recording.time_slice(t0, tend)
# print(short_rec)
# all_traces = short_rec.get_traces(channel_ids=selected_channels, return_scaled=True)
# # Expect all_traces to have shape (n_channels, n_samples)
# n_samples,n_channels = all_traces.shape

# # -------------------------------------------------------------------
# # Determine the number of chunks per channel.
# n_chunks = int(np.ceil(n_samples / chunk_length))
# pad_amt = n_chunks * chunk_length - n_samples
# if pad_amt > 0:
#     all_traces = np.pad(all_traces, ((0, 0), (0, pad_amt)), mode='constant')

# # Reshape the signal into chunks: shape -> (n_channels, n_chunks, chunk_length)
# traces_chopped = all_traces.reshape(n_channels, n_chunks, chunk_length)

# # -------------------------------------------------------------------
# # In each chunk, segment data into non-overlapping FFT windows.
# n_segments = chunk_length // NFFT  # extra samples at the end are ignored
# traces_cropped = traces_chopped[:, :, :n_segments * NFFT]  # crop out incomplete segment portions

# # Reshape into windows: shape -> (n_channels, n_chunks, n_segments, NFFT)
# traces_windows = traces_cropped.reshape(n_channels, n_chunks, n_segments, NFFT)

# # Detrend each window by subtracting its mean:
# traces_windows = traces_windows - traces_windows.mean(axis=-1, keepdims=True)

# # -------------------------------------------------------------------
# # Compute the FFT on each window with zero-padding to pad_to.
# # FFT is computed along the last axis.
# traces_windows_cp = cp.asarray(traces_windows)
# fft_result_cp = cp.fft.rfft(traces_windows_cp, n=pad_to, axis=-1)

# #fft_result_np = np.fft.rfft(traces_windows, n=pad_to, axis=-1)#slower

# # Calculate power spectral density (PSD) = squared magnitude of FFT result.
# Pxx = np.abs(fft_result_cp) ** 2

# # Compute the corresponding frequency vector (Hz)
# freqs = np.fft.rfftfreq(pad_to, d=1/fs)

# # -------------------------------------------------------------------
# # Select the frequency bins within the desired frequency band.
# freq_idx = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]

# # Instead of averaging over the frequency bins, we keep the entire frequency axis
# # so that our final matrix is channels x time x frequencies.
# # Pxx shape: (n_channels, n_chunks, n_segments, n_frequency_bins)
# # Select frequency bins within your frequency band:
# Pxx_band = Pxx[..., freq_idx]  # shape -> (n_channels, n_chunks, n_segments, len(freq_idx))

# # Reshape the time axes (chunks and segments) into a single time dimension:
# # Final shape will be (n_channels, total_time_windows, n_freqs)
# total_time_windows = n_chunks * n_segments
# spectrogram_matrix = Pxx_band.reshape(n_channels, total_time_windows, len(freq_idx))

# # Optionally, convert power to logarithmic scale for visualization:
# spectrogram_matrix_log = 10 * np.log10(spectrogram_matrix + 1e-12)
# spectrogram_matrix_log=spectrogram_matrix_log.T#transpose
# spectrogram_matrix_log=spectrogram_matrix_log.get()
# # -------------------------------------------------------------------
# # Plot the spectrogram for each channel
# time_per_window = bin_size  # each FFT window covers bin_size seconds
# total_time = total_time_windows * time_per_window

# # Create a time vector in seconds for the x-axis of the image
# time_vec = np.linspace(0, total_time, total_time_windows)
# freq_vec = freqs[freq_idx]

# # # Plot all channels in a multi-panel figure
# # fig, axs = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)

# # if n_channels == 1:
# #     axs = [axs]

# # for ch in range(n_channels):
# #     im = axs[ch].imshow(
# #         spectrogram_matrix_log[ch].T,
# #         extent=[time_vec[0], time_vec[-1], freq_vec[0], freq_vec[-1]],
# #         aspect='auto',
# #         origin='lower',
# #         cmap='turbo'
# #     )
# #     axs[ch].set_title(f"Channel {ch} Spectrogram")
# #     axs[ch].set_ylabel("Frequency (Hz)")
# #     axs[ch].set_xlabel("Time (s)")
# #     fig.colorbar(im, ax=axs[ch], label="Power (dB)")

# # plt.tight_layout()
# # plt.show()

# # Plot the spectrogram and raw trace for each channel side-by-side.
# # Create a time vector for the raw traces using t0 and tend.
# t_trace = np.linspace(t0, tend, all_traces.shape[0])

# # Adjust the spectrogram time vector to absolute time by adding t0.
# time_vec_abs = t0 + np.linspace(0, total_time, total_time_windows)

# # Create subplots with two columns per channel (left: raw trace, right: spectrogram)
# fig, axs = plt.subplots(n_channels, 2, figsize=(16, 3 * n_channels), sharex='row')

# # If only one channel exists, make sure axs is 2D.
# if n_channels == 1:
#     axs = np.array([axs])

# for ch in range(n_channels):
#     # Left subplot: Plot raw trace for the channel.
#     ax_trace = axs[ch, 0]
#     ax_trace.plot(t_trace.T, all_traces[:, ch], color='k')
#     ax_trace.set_title(f"Channel {ch} - Trace")
#     ax_trace.set_ylabel("uV")
    

#     # Hide x-axis labels on the upper plots if not the bottom row.
#     if ch < n_channels - 1:
#         ax_trace.tick_params(labelbottom=False)
    
#     # Right subplot: Plot spectrogram.
#     ax_spec = axs[ch, 1]
#     im = ax_spec.imshow(
#         spectrogram_matrix_log[:,:,ch],
#         extent=[time_vec_abs[0], time_vec_abs[-1], freq_vec[0], freq_vec[-1]],
#         aspect='auto',
#         origin='lower',
#         cmap='turbo'
#     )
#     ax_spec.set_title(f"Channel {selected_channels[ch]} - Spectrogram")
#     ax_spec.set_ylabel("Frequency (Hz)")
    

#     # Only set the xlabel for the bottom subplot.
#     if ch == n_channels - 1:
#         ax_spec.set_xlabel("Time (s)")
#     else:

#        ax_spec.tick_params(labelbottom=False)
#     # Link x-axis of the trace and spectrogram plots for the same channel.
#     #ax_spec.get_shared_x_axes().join(ax_spec, ax_trace)
    
#     # Add a colorbar for the spectrogram subplot.
#     fig.colorbar(im, ax=ax_spec, label="Power (dB)")

# plt.tight_layout()
# plt.show()

# der_short=spikeinterface.preprocessing.directional_derivative(recording= short_rec, direction = 'x', order = 1, edge_order= 1, dtype='float32')
