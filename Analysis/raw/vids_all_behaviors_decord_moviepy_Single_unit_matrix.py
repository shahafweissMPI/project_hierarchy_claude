#import torch
#import torchvision
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed

from joblib import Parallel, delayed
from multiprocessing import Pool
from scipy.stats import zscore
import numpy as np
import time
from tqdm import tqdm
import time
import preprocessFunctions as pp
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import helperFunctions as hf
import plottingFunctions as pf
import os
from moviepy import VideoFileClip,ImageSequenceClip
from decord import VideoReader,cpu,gpu
from PIL import Image
import IPython
from pathlib import Path
import shutil

############# USER PARMAETERS #######################
# Custom colormap: blue for positive, white for zero, red for negative
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'black','red'], N=256)

#plotting params
plt.style.use('dark_background')
matplotlib.use('Agg')  #Use the Agg backend for non-interactive plotting

#User defined Parameters
savepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\20250402\afm16924\videos\subset\pups"#Where should the video be stored? (best an SSD)
video_cache=r"E:\videos"
if not os.path.exists(savepath):
    os.makedirs(savepath)
#savepath=r"E:\videos" #Where should the video be stored? (best an SSD)
animal = 'afm16924'
sessions = ['240525','240526','240522','240524']
target_regions = ['DPAG','DMPAG', 'DLPAG','LPAG','VLPAG','VMPAG']


target_cells=[344,365,347,391]# escape active
target_behaviors=['startle','turn','escape']
#target_cells=[344,365,391,538,545,547]#pup positive
#target_cells=[347,393,561,582,590]#pup negative
target_cells=[344,365,391,538,545,547,347,393,561,582,590]#pup related
target_behaviors=['pup_run','pup_grab','pup_retrieve','pup_drop']


windowsize_s=10 #how long before+ after the loom should the video playreading is about 10 minutes. but going to get 10 gig card
view_window_s=5#with what window should be plotted around the current time?

spike_res=0.001 #n_spikes_time matrix resolution (1ms~=1 spike)
FR_res=0.02 # firing rates matrix resolution
ndot = 1.0 # dot size
#performance tweaking
looptype='multithreaded' # can be 'serial' / 'multithreaded' ~10 min to plot frames / 'multiprocess'. currently multiprocess doesn't work
num_workers = int(os.cpu_count()/2) # Or set manually

zoom_on_mouse=False



############# FUNCTIONS #######################
def delete_cached_file(local_file_name):    
    """
    Deletes the cached video file.

    Args:
        local_file_name (str): The local file name of the cached video.

    Returns:
        bool: True if the file was successfully deleted, False otherwise.
    """
    try:
        if os.path.exists(local_file_name):
            os.remove(local_file_name)
            return True
        return False
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False
def copy_video_to_local_cache(server_path, cache_path):
    """
    Copies a video file from the server to a local cache and returns the local file name.

    Args:
        server_path (str): The path to the video file on the server.
        cache_path (str): The local cache directory path.

    Returns:
        str: The local file name of the cached video.
    """
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    local_file_name = os.path.join(cache_path, os.path.basename(server_path))
    shutil.copy2(server_path, local_file_name)
    return local_file_name
    
   
def save_debugfig(fig):
    fig.savefig('debug.png')
    
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
    
def read_frames_decord(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))
    #vr.get_avg_fps()
            
    frames = vr.get_batch(frame_indices)
    return frames.asnumpy()

def write_video_moviepy(images, output_path, fps=25):
    ffmpeg_command = [
       '-y', '-f', 'image2pipe', '-framerate', str(vframerate), '-i', '-', 
       '-c:v', 'h264_nvenc','-b:v', '10M' ,'-maxrate:v', '10M' ,'-bufsize:v', '10M', 
       '-profile:v', 'main', '-an',  '-pix_fmt', 'yuv420p', output_path]#'-gpu 0',
    #ffmpeg_params=['-y', '-f', 'image2pipe','-i', '-','-c:v','-vcodec', 'h264_nvenc','-b:v', '10M','-maxrate:v', '10M' ,'-bufsize:v', '10M','-profile:v', 'main', '-an',  '-pix_fmt', 'yuv420p']
    ffmpeg_params = ['-c:v', 'h264_nvenc']
    clip = ImageSequenceClip(list(images), fps=fps)
    
    # start_time = time.time()
    # clip.write_videofile(output_path, codec='libx264', ffmpeg_params=ffmpeg_command)
    # end_time = time.time()
    # print(f"ffmpeg_command took {end_time - start_time:.4f} seconds")
    # start_time = time.time()
    
    
    #start_time = time.time()
    clip.write_videofile(output_path, codec='libx264', audio=False,threads=16,ffmpeg_params=ffmpeg_params)
    #end_time = time.time()
    #print(f"ffmpeg_params took {end_time - start_time:.4f} seconds")

    

def process_frame(i, window_frame, lframe, frames):
    vframerate= 50
    plt.rcParams.update({
    'font.size': 16,            # controls default text sizes
    'axes.titlesize': 18,       # fontsize of the axes title
    'axes.labelsize': 16,       # fontsize of the x and y labels
    'xtick.labelsize': 20,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'figure.titlesize': 30,      # fontsize of the figure title
    'figure.dpi': 100
    })    
    window_time = window_frame / vframerate

    # Make figure
    #gs = gridspec.GridSpec(2, 2)
    #fig = plt.figure(figsize=(20, 12))
    

    fig, ax = plt.subplots(2, 2, figsize=(20, 15), gridspec_kw={'height_ratios': [2, 2]})
    
    # Flatten the 2x2 array of axes for easier indexing
    ax = ax.flatten()
    
    # Assign subplots
    vid_ax = ax[0]
    speed_ax = ax[2]
    spike_ax = ax[3]
    FR_ax1 = ax[1]
    
    # Make all but vid_ax share the x-axis
    
    spike_ax.sharex(speed_ax)
    FR_ax1.sharex(speed_ax)
    
    # Create twin axes
    dist_ax = speed_ax.twinx()
   # FR_ax2 = FR_ax1.twinx()
   
    axes = [speed_ax, spike_ax, FR_ax1]

    for fax in axes:
       #fax.set_xlim(common_xlim)
       fax.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # event
       fax.axvline(0, linestyle='--', color='Gray')  # current time
       
    

    windowtime = frame_index_s[window_frame]
    fig.suptitle(f"{animal} {session} {target_behavior} {hf.convert_s(windowtime)} FR binsize={FR_res*1000:0.0f} ms")
   # IPython.embed()
    # plot frame + sleap labels
    vid_ax.imshow(frames[i], cmap='binary_r')
    pf.remove_axes(vid_ax, rem_all=True)
    
   
    #xlim = vid_ax.get_xlim()
    #ylim = vid_ax.get_ylim()

    #vid_ax.set_xlim(xlim[0] + 0.05 * (xlim[1] - xlim[0]), xlim[1] - 0.05 * (xlim[1] - xlim[0]))
#    vid_ax.set_ylim(ylim[0] + 0.05 * (ylim[1] - ylim[0]), ylim[1] - 0.05 * (ylim[1] - ylim[0]))

    # zoom in on mouse
    # if zoom_on_mouse==True:
    #     x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
    #     vid_ax.set_xlim((x_min, x_max))
    #     vid_ax.set_ylim((y_min, y_max))
    # else:
    #     pass
    #     #x_min = 90#650
    #    # y_max = 90#300
    
    # if target_behavior=='loom': # show loom
    #     if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
    #         x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
    #         pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), vid_ax)

    # plot distance and velocity
    plot_start = window_frame - view_window_f  # this is in units of video frames
    plot_end = window_frame + view_window_f
    
    duration=abs(plot_end - plot_start)
    offset=3
    x_v = np.linspace(-view_window_s, view_window_s, plot_end - plot_start)
    #x_v = np.linspace(-1 * (duration+offset), duration+offset, duration)

    # velocity
    #line1, = speed_ax.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity',lw=4)
    line1, = speed_ax.plot(x_v, velocity[plot_start:plot_end], color='m', label='velocity',lw=4)
    speed_ax.set_ylabel('Speed (cm/s)')
    speed_ax.set_ylim((0, max_vel))
    speed_ax.yaxis.label.set_color("m")

    # distance to shelter
    #dist_ax = speed_ax.twinx()
   
    #line2, = dist_ax.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter',lw=4)
    line2, = dist_ax.plot(x_v, distance2shelter[plot_start:plot_end], color='c', label='distance to shelter',lw=4)
    dist_ax.set_ylabel('Distance to shelter (cm)')
    dist_ax.yaxis.label.set_color("c")
    dist_ax.set_ylim((0, max_dist))

    # Add loom line, legend, xlabel, remove top axis
    #speed_ax.set_xlim(x_v[0], x_v[-1])
    
    #speed_ax.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # loom
 #   speed_ax.axvline(0, linestyle='--', color='Gray')  # current time
    speed_ax.spines['top'].set_visible(False)
    dist_ax.spines['top'].set_visible(False)
    #speed_ax.legend(handles=[line1, line2])
    #speed_ax.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    speed_ax.set_xlabel('Time (s)')
    

    # plot raster
    plot_start = window_time - view_window_s
    plot_end = window_time + view_window_s
   
    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
    cell_color = 'w'
    #### ####plot spikes
   # spike_ax.axvline(0, linestyle='--', color='Gray')  # current time
#    spike_ax.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # event
    
    
    
    # clusters=[]
    # regions=[]
    yticklabels=[]
    yticklabels_short=[]
    for j, n in enumerate(ndata):
        # clusters.append(n_cluster_index[j])
        # regions.n_region_index[j]
      
        
        spikeind = n.astype(bool)
        all_spiketimes = n_time_index[spikeind]
        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
        spiketime = all_spiketimes[window_ind] - window_time
        #cluster=n_cluster_index[j]     
        spike_ax.scatter(spiketime, np.zeros_like(spiketime) + ycoords[j], color=cell_color, s=10)
        yticklabels.append(f"{n_region_index[j]} #{n_cluster_index[j]}")
        yticklabels_short.append(f"{n_cluster_index[j]}")

    pf.region_ticks(n_region_index, ycoords=ycoords, ax=spike_ax)
    
    pf.remove_axes(spike_ax)
    spike_ax.set_xlabel('Time (s)')
    #spike_ax.tick_params(axis='x', colors='white')
    spike_ax.yaxis.label.set_color("white")    
    #spike_ax.set_xlim((-view_window_s, view_window_s))   
    #FR_ax1.set_xlim((-view_window_s, view_window_s))    
    
    
   
    
    minval=0.0
    ##plot firing rate    
    cell_color = 'white'
    
    
   # FR_ax1.axvline(0, linestyle='--', color='Gray')  # current time
    
    #FR_ax1.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # event
    
    minval = float(10000)
    maxval = float(-10000)
    
    # for j, firing_rate in enumerate(firing_rates):
    #     firing_rate_time = firing_rate_bins_time[:-1]                       
    #     window_ind = (firing_rate_time > plot_start) & (firing_rate_time < plot_end)
        
    #     firing_rate_time = firing_rate_time[window_ind] - window_time
    #     current_FR = firing_rate[window_ind]
        
    #     # First plot_FR assignment
    #     plot_FR = current_FR
    #     minval = min(minval, np.nanmin(plot_FR))
    #     maxval = max(maxval, np.nanmax(plot_FR))
    #     FR1 = FR_ax1.plot(firing_rate_time, plot_FR + j, color='green', lw=4, label='Firing rate', alpha=0.4)
        
    #     # Second plot_FR assignment
    #     plot_FR = current_FR - base_mean[j]
    #     minval = min(minval, np.nanmin(plot_FR))
    #     maxval = max(maxval, np.nanmax(plot_FR))
    #     FR2 = FR_ax1.plot(firing_rate_time, plot_FR + j, color='red', lw=4, label='Firing rate - baseline avg', alpha=0.4)
        
    #     # Third plot_FR assignment
    #     plot_FR = zscore(current_FR)
    #     minval = min(minval, np.nanmin(plot_FR))
    #     maxval = max(maxval, np.nanmax(plot_FR))
    #     FR3 = FR_ax2.plot(firing_rate_time, plot_FR + j, color='blue', lw=4, label='zscored Firing rate', alpha=0.4)
    multiple_cells= firing_rates.shape[0]>1
   # ycoords=ycoords*3 # separate cells, and allow for -/+ 1 values
    linewidth=2
    firing_rate_time = firing_rate_bins_time[:-1]
    window_ind = (firing_rate_time > plot_start) & (firing_rate_time < plot_end)
    # Compute z-scored firing rates for all cells
    if multiple_cells==True:
       zscored_matrix = (firing_rates - base_mean[:,None]) / np.std(firing_rates, axis=1, keepdims=True)
      

    else:#single cell
      
        zscored_matrix = (firing_rates - base_mean) / np.std(firing_rates, axis=1, keepdims=True)
        # Ensure window_ind matches the second dimension of zscored_matrix    
       

    # Ensure window_ind matches the second dimension of zscored_matrix    
    zscored_matrix_window = zscored_matrix[:,  window_ind]
    # Subset the zscored_matrix to the window
    zscored_matrix_window = zscored_matrix[:, window_ind]

       
        
              
   # window_ind = window_ind[:zscored_matrix.shape[1]]
  
    
    firing_rate_time = firing_rate_time[window_ind] - window_time
    
   

    
    # Plot the matrix as an image
    im = FR_ax1.imshow(
        zscored_matrix_window,
        aspect='auto',
        cmap=cmap,
        extent=[firing_rate_time[0], firing_rate_time[-1], len(zscored_matrix) - 0.5, -0.5],
        vmin=-3,  # Set limits for z-score values
        vmax=3
    )
    
    # Add colorbar
    # Move the colorbar to the top of the subplot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(FR_ax1)
    cax = divider.append_axes("bottom", size="5%", pad=1.25)
    cbar= fig.colorbar(im, cax=cax,orientation='horizontal')
#        cbar = plt.colorbar(im, ax=cax, orientation='horizontal')
    cbar.set_label('Firing Rate Z-scored to baseline mean')
    
    # Set the ticks and label to appear on the top of the colorbar
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    
    
    # Optionally, adjust the colorbar's position manually if further fine-tuning is needed.
    # Get the current position and subtract a larger offset from the y0 coordinate.
   
   # cbar.ax.set_position([pos.x0, pos.y0 - 0.3, pos.width, pos.height])
    
    cbar.outline.set_visible(False)

    # cbar = plt.colorbar(im, ax=FR_ax1, orientation='horizontal', pad=0.05)
    # cbar.set_label('Firing Rate Z-scored to baseline mean')
    # cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.xaxis.set_label_position('top')

    # cbar.ax.set_position([cbar.ax.get_position().y0, cbar.ax.get_position().y0 - 0.5, 
    #               cbar.ax.get_position().width, cbar.ax.get_position().height])
    cbar.outline.set_visible(False)
    
    # Set y-axis ticks and labels
    FR_ax1.set_yticks(np.arange(len(yticklabels)))                   
    if len(yticklabels)<15:
        spike_ax.set_yticks(ycoords)
        spike_ax.set_yticklabels(yticklabels_short) 
        FR_ax1.set_yticklabels(yticklabels)
    else:        
        pf.region_ticks(n_region_index, ycoords=ycoords, ax=spike_ax)
        pf.region_ticks(n_region_index, ycoords=FR_ax1.get_yticks(), ax=FR_ax1)        
    FR_ax1.set_ylabel('Units')        
    FR_ax1.spines['top'].set_visible(False)        
    pf.remove_axes(FR_ax1)
    FR_ax1.set_xlabel('Time (s)')                
    FR_ax1.yaxis.label.set_color("white")
    FR_ax1.tick_params(axis='y', colors='white')
    
    common_xlim = (-view_window_s, view_window_s)
    FR_ax1.set_xlim((-view_window_s, view_window_s))    
    fig.subplots_adjust(wspace=0.3)  # Increase wspace to add more horizontal padding
    fig.tight_layout(rect=[0, 0, 1, 0.95],pad=2)
    fig.canvas.draw()  # Force update to the figure    
   # save_debugfig(fig)
#    IPython.embed()
   
    w, h = fig.canvas.get_width_height()
    # Get an array of RGBA values and drop the alpha channel
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)[..., :3]
#    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)

    return i, image

def process_behavior(args):    
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # plot every other frame (25 instead of 60 FPS)
   # if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
      #  print(np.abs(np.mean(around_lframe - np.round(around_lframe))) )
      #  raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print(f"collecting frames from original video\n this stage depends mostly the ethernet connection speed")
    frames = read_frames_decord(paths['video'], around_lframe)# read original frame
    images = [None] * len(around_lframe)#initialize new frames   
    
    if looptype=='serial':        
        print(f"serially looping through frames...")
        # serial for loop   
        for i, window_frame in enumerate(tqdm(around_lframe, desc="Processing new frames serially")):
            i, image = process_frame(i, window_frame, lframe, frames)
            images[i] = image
    
    #multithreaded    
    if looptype=='multithreaded': 
        print(f"Using ThreadPoolExecutor with {num_workers} workers...")
        start_time = time.time()  
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="creating new frames  (multithreaded), this stage depends on the local CPU core count"):
                i, image = future.result()
                images[i] = image
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    #     for future in as_completed(futures):
    #         i, image = future.result()
    #         images[i] = image
        end_time = time.time()
        print(f"ThreadPoolExecutor took {end_time - start_time:.4f} seconds")
        
    if looptype=='multiprocess':
        print(f"Using ProcessPoolExecutor with {num_workers} workers...")
        start_time = time.time()  
        # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # # Submit tasks: Pass arguments needed by process_frame
        # # Ensure all arguments (lframe, frames, and any other globals process_frame uses)
        # # are picklable and passed explicitly if possible.
        #  futures = [executor.submit(process_frame, i, window_frame, lframe, frames)
        #            for i, window_frame in enumerate(around_lframe)]
    
        # # Collect results as they complete
        # for future in tqdm(as_completed(futures), total=len(futures), desc="Creating frames (ProcessPool)"):
        #     try:
        #         i, image = future.result() # Get result from the completed future
        #         images[i] = image         # Place it in the correct spot in the list
        #     except Exception as exc:
        #         print(f'Frame generation failed with exception: {exc}')
        # end_time = time.time()
        # print(f"ProcessPoolExecutor took {end_time - start_time:.4f} seconds")
    
   #multiprocss
    start_time = time.time()
    with ProcessPoolExecutor(2) as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
        for future in as_completed(futures):
            i, image = future.result()
            images[i] = image
    end_time = time.time()
    print(f"ProcessPoolExecutor took {end_time - start_time:.4f} seconds")
    plt.close('all')
    return images
    
    
def save_vid_from_images_array(images,ltime):       
    if len(target_cells)==1:
        str=rf"{savepath}\{animal}_{session}_{target_behavior}_{np.round(ltime/60,2)}_{FR_res}_unit{target_cells[0]}.mp4"
    else:
        str=rf"{savepath}\{animal}_{session}_{target_behavior}_{np.round(ltime/60,2)}_{FR_res}.mp4"
    start_time = time.time()
    write_video_moviepy(images,str)
    end_time = time.time()
    print(f"moviepy: saving video took {end_time - start_time:.4f} seconds")
       

def process_behavior_ProcessPoolExecutor(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)
    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")          
    start_time = time.time()
    with ProcessPoolExecutor(4) as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        i, image = future.result()
        images[i] = image
         
    end_time = time.time()

    print(f"ProcessPoolExecutor took {end_time - start_time:.4f} seconds")
    
    return images  


def process_frame_wrapper(args):
    return process_frame(*args)

def process_behavior_Pool(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")  
        
    start_time = time.time()
    with Pool(4) as pool:
        results = list(tqdm(pool.imap(process_frame_wrapper, [(i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]), total=len(around_lframe)))
    
    for i, image in results:
        images[i] = image
         
    end_time = time.time()

    print(f"Pool took {end_time - start_time:.4f} seconds")
    
    return images



def process_behavior_joblib(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")  
        
    start_time = time.time()
    results = Parallel(n_jobs=8)(delayed(process_frame)(i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe))
    
    for i, image in tqdm(results, total=len(results)):
        images[i] = image
         
    end_time = time.time()

    print(f"joblib took {end_time - start_time:.4f} seconds")
    
    return images
def vid_by_behav(target_behavior):
    #%%Precompute

    #Windows

    loc_all_looms = np.where([behaviour['behaviours'] == target_behavior])[1]
    time_all_looms = behaviour['frames_s'][loc_all_looms]
    frame_all_looms = time_all_looms * vframerate
    # args = [(lframe, ltime, target_behavior) for lframe, ltime in zip(frame_all_looms, time_all_looms)] # loop through events in target behavior
    # with Pool() as pool:
    #    pool.map(process_behavior, args)
    
    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        
        images = process_behavior([lframe, ltime, target_behavior]) #get array of frames
        print(f"saving video")
        save_vid_from_images_array(images,ltime) # save as video
        
        
    
#############END OF FUNCTIONS #######################
if __name__ == '__main__':    
    
    for session in sessions:
        print(f'working on : {session}')
        #%% load data
        paths=pp.get_paths(animal, session)    
        output_path=paths['preprocessed']
        
        vpath=paths['video']
        print(f"loading tracking")        
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
        
        print(f"loading neural data")
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
        
        global vframerate
        vframerate=len(frame_index_s)/max(frame_index_s)
        
        
   
        #distance2shelter, pixel2cm, shelter_point=hf.get_shelterdist(paths,locations, vframerate ,vpath)
        # velocity, all_locations, node_names, ds_movement, Cm2Pixel, distance2shelter, shelterpoint=pp.extract_sleap(session, animal, mouse_tracking_path = paths['mouse_tracking'], 
        #                                                                                                      camera_video_path = paths['video'], vframerate=vframerate, 
        #                                                                                                      Cm2Pixel_from_paths=paths['Cm2Pixel_xy'], 
        #                                                                                                      Shelter_xy_from_paths=paths['Shelter_xy'],
        #                                                                                                      node = 'b_back')
        #print(f"0 {np.shape(n_time_index)=}, {np.shape(ndata)=}")

        print(f"calculating intananeous firing rate" )
        iFR,iFR_array,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate
        
        
        # # Get the overall maximum value
           
        
        
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
     
      
        print(f"calculating baseline")
        base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
        #    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
        base_mean = np.round(base_mean, 2)
        base_mean = base_mean[:, np.newaxis]  # now shape is (404, 1)
        base_mean = np.where(base_mean == 0, 1/10000, base_mean)
        base_mean = np.squeeze(base_mean)  # Remove the extra dimension, now shape is (404,)        
        windowsize_f=np.round(windowsize_s*vframerate).astype(int)
        view_window_f=np.round(view_window_s*vframerate).astype(int)
        #target node
        node_ind=np.where(node_names=='f_back')[0][0]#node to use for tracking        
      
        #Target neurons
        target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
        n_ybottom_ind = np.max(target_neurons_ind) + 2
        n_ytop_ind = np.min(target_neurons_ind) - 2
        
        
        

        os.chdir(savepath)
        print("creating local copy of video file")
        tempfile=copy_video_to_local_cache(paths['video'],video_cache)   
        paths['video']=tempfile
        unique_behaviours = behaviour.behaviours.unique()
        if len(target_behaviors)!=0:#reduce to target behaviors
            unique_behaviours = [behavior for behavior in target_behaviors if behavior in unique_behaviours]
            
        #unique_behaviours = [ 'escape'] #debug
        for target_behavior in unique_behaviours:
            print(target_behavior)
            vid_by_behav(target_behavior)
        print("removing local copy of video file")  
        delete_cached_file(tempfile)    
