import preprocessFunctions as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os
import subprocess
from tqdm import tqdm
import io
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import IPython
# Setup logging
logging.basicConfig(level=logging.INFO)
import multiprocessing
multiprocessing.set_start_method('spawn')
# Constants
CACHE_PATH = r"E:\2025\Figures\PSTH\afm16924\videos"
animal = 'afm16924'
session = '240526'
WINDOW_SIZE_S = 7  # how long before+ after the loom should the video play
VIEW_WINDOW_S = 3  # with what window should be plotted around the current time?
ZOOM_ON_MOUSE = False
TARGET_REGIONS = [ 'DMPAG', 'DLPAG', 'LPAG']
PLOT_ALL_NEURONS = True  # whether to plot activity in the whole probe or zoom in on a certain region
FFMPEG_FRAMERATE = 50
FRAME_CHUNK_SIZE = FFMPEG_FRAMERATE*VIEW_WINDOW_S

def run_forest_run():
    # Style
    plt.style.use('dark_background')
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
    iFR,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate
    
    
    # Get the overall maximum value
    
    
    res=0.001
    print(f"recalculating ndata")
    n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=res)
    
    
    if len(TARGET_REGIONS) != 0:  # If target brain region(s) specified    
        in_region_index = np.where(np.isin(n_region_index, TARGET_REGIONS))[0]
        n_region_index = n_region_index[in_region_index]    
    else:
        pass
        #n_region_index = np.arange(len(n_region_index))
    
    base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
    #    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
    base_mean = np.round(base_mean, 2)    
    base_mean = base_mean[:, np.newaxis]  # now shape is (404, 1)
    base_mean = np.where(base_mean == 0, 1/10000, base_mean)
        
    
    vframerate = len(frame_index_s) / max(frame_index_s) # or 1/(np.median(np.diff(frame_index_s)))
    paths = pp.get_paths(animal,session)
    
    # Windows
    windowsize_f = np.round(WINDOW_SIZE_S * vframerate).astype(int)
    view_window_f = np.round(VIEW_WINDOW_S * vframerate).astype(int)
    # Target node
    node_ind = np.where(node_names == 'f_back')[0][0]
    
    # Distance to shelter
    
    video_path=paths['video']
    vframerate=50
    Shelter_xy=paths['Shelter_xy'].split(' ')
    distance2shelter, pixel2cm, shelter_point = hf.get_shelterdist(paths,locations, vframerate,video_path)
    max_dist = max(distance2shelter)
    max_vel = max(velocity)
    
    # Target neurons
    target_neurons_ind = np.where(np.isin(n_region_index, TARGET_REGIONS))[0]
    n_ybottom_ind = np.max(target_neurons_ind) + 2
    n_ytop_ind = np.min(target_neurons_ind) - 2
    
    if PLOT_ALL_NEURONS:
        fileend = 'all'
        ndot = .1  # size of raster dots
    else:
        fileend = 'zoom'
        ndot = .6
    fileend
    # Change directory
    os.chdir(CACHE_PATH)
    
    loc_all_looms = np.where([behaviour['behaviours'] == 'loom'])[1]
    time_all_looms = behaviour['frames_s'][loc_all_looms]
    frame_all_looms = time_all_looms * vframerate
    
    def process_loom(lframe, ltime):
        output_video_path = os.path.join(CACHE_PATH, f'loom_at_{np.round(ltime / 60, 2)}_{fileend}.mp4')
        ffmpeg_command = [
            'ffmpeg', '-y', '-f', 'image2pipe', '-framerate', str(FFMPEG_FRAMERATE), '-i', '-', 
            '-c:v', 'h264_nvenc','-b:v', '5M' ,'-maxrate:v', '5M' ,'-bufsize:v', '10M', '-profile:v', 'main', '-an', '-gpu 0', '-pix_fmt', 'yuv420p', output_video_path
        ]
    
        around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # Step by 2 to plot every other frame
        if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
            logging.error('Frame calculation error')
            raise ValueError('Something is wrong in the calculation of frames (either here or in the preprocessing script)')
        around_lframe = np.round(around_lframe).astype(int)
        frames = hf.read_frames(paths['video'], desired_frames=around_lframe)
    
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE) as ffmpeg_proc:
            for chunk_start in range(0, len(around_lframe), FRAME_CHUNK_SIZE):
                chunk_end = min(chunk_start + FRAME_CHUNK_SIZE, len(around_lframe))
                chunk_frames = around_lframe[chunk_start:chunk_end]
    
                for i, window_frame in enumerate(chunk_frames):
                    window_time = window_frame / vframerate
    
                    # Make figure
                    gs = gridspec.GridSpec(2, 2)
                    fig = plt.figure(figsize=(20, 12))
    
                    # Add subplots
                    ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
                    ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
                    ax2 = fig.add_subplot(gs[:, 1])  # All rows, second column
    
                    windowtime = frame_index_s[window_frame]
                    plt.suptitle(hf.convert_s(windowtime))
    
                    # Plot frame + sleap labels
                    ax0.imshow(frames[i + chunk_start], cmap='binary_r')
                    pf.remove_axes(ax0, rem_all=True)
    
                    # Zoom in on mouse
                    if ZOOM_ON_MOUSE:
                        x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i + chunk_start], locations[window_frame, node_ind, :], 200)
                        ax0.set_xlim((x_min, x_max))
                        ax0.set_ylim((y_min, y_max))
                    else:
                        x_min = 650
                        y_max = 300
    
                    # Show loom
                    if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
                        pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)
    
                    # Plot distance and velocity
                    plot_start = window_frame - view_window_f  # this is in units of video frames
                    plot_end = window_frame + view_window_f
                    x_v = np.linspace(-5, 5, plot_end - plot_start)
    
                    # Velocity
                    line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity')
                    ax1.set_ylabel('velocity (cm/s)')
                    ax1.set_ylim((0, max_vel))
    
                    # Distance to shelter
                    ax1_1 = ax1.twinx()
                    line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter')
                    ax1_1.set_ylabel('distance to shelter (cm)')
                    ax1_1.set_ylim((0, max_dist))
    
                    # Add loom line, legend, xlabel, remove top axis
                    ax1.set_xlim(x_v[0], x_v[-1])
                    ax1.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
                    ax1.axvline(0, linestyle='--', color='Gray')  # current time
                    ax1.spines['top'].set_visible(False)
                    ax1_1.spines['top'].set_visible(False)
                    ax1.legend(handles=[line1, line2])
                    ax1.set_xlabel('time (s)')
    
                    plot_start = window_time - VIEW_WINDOW_S
                    plot_end = window_time + VIEW_WINDOW_S
    
                    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
                    for i, n in enumerate(ndata):
                        spikeind = n.astype(bool)
                        all_spiketimes = n_time_index[spikeind]
                        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
                        spiketime = all_spiketimes[window_ind] - window_time
                        ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[i], color='w', s=ndot)
    
                    pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
                    pf.remove_axes(ax2)
                    ax2.set_xlabel('time (s)')
                    ax2.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
                    ax2.axvline(0, linestyle='--', color='Gray')  # current time
                    ax2.set_xlim((-5, 5))
    
                    if not PLOT_ALL_NEURONS:  # zoom in to target areas
                        bottom = ycoords[n_ybottom_ind]
                        top = ycoords[n_ytop_ind]
                        ax2.set_ylim((bottom, top))
    
                    # Save this in a buffer
                    with io.BytesIO() as buf:
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        img = Image.open(buf)
                        img.save(ffmpeg_proc.stdin, format='PNG')
    
                    plt.close(fig)
        result=f'done with loom at {hf.convert_s(ltime)}'
        return result
    
    # Process looms concurrently
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_loom, frame_all_looms, time_all_looms))
      
    return results

    # with ProcessPoolExecutor(2) as executor:
    #     results = list(executor.map(process_loom, frame_all_looms, time_all_looms))
    #     for result in results:
    #         print(result)


if __name__ == '__main__':    
    CACHE_PATH = r"E:\2025\Figures\PSTH\afm16924\videos"
    animal = 'afm16924'
    sessions = '240526'
    WINDOW_SIZE_S = 7  # how long before+ after the loom should the video play
    VIEW_WINDOW_S = 3  # with what window should be plotted around the current time?
    ZOOM_ON_MOUSE = False
    TARGET_REGIONS = [ 'DMPAG', 'DLPAG', 'LPAG']
    PLOT_ALL_NEURONS = True  # whether to plot activity in the whole probe or zoom in on a certain region
    FFMPEG_FRAMERATE = 25
    FRAME_CHUNK_SIZE = 25*VIEW_WINDOW_S
    for session in sessions:
       results= run_forest_run()
       for result in results:
           print(result)
