#import torch
#import torchvision

from concurrent.futures import ThreadPoolExecutor, as_completed
import preprocessFunctions as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os
from tqdm import tqdm
from moviepy import VideoFileClip,ImageSequenceClip

from PIL import Image
import IPython
import preprocessFunctions as pp
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import matplotlib.gridspec as gridspec

import time



from decord import VideoReader, cpu


#plotting params
plt.style.use('dark_background')
import matplotlib

#User defined Parameters
cachepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\PSTH\videos" #Where should the video be stored? (best an SSD)
#cachepath=r"E:\2025\Figures\PSTH\afm16924\videos" #Where should the video be stored? (best an SSD)
animal = 'afm16924'
session = '240525'
#   target_behavior='pup_run'
target_regions = [ 'DMPAG', 'DLPAG','LPAG']


plt.style.use('dark_background')

all_neurons=False #whether to plot activity in the whole probe or zoom in on a certain region
zoom_on_mouse=False



windowsize_s=7 #how long before+ after the loom should the video play
view_window_s=5#with what window should be plotted around the current time?
    
    
#%% load data
paths=pp.get_paths(animal, session)    
output_path=paths['preprocessed']

vpath=paths['video']

velocity,locations,node_names,bottom_node_names,frame_index_s,frames_dropped,distance2shelter,bottom_distance_to_shelter=hf.load_specific_preprocessed_data (animal, session, 'tracking',load_pd=False )
distance2shelter=distance2shelter[:,3]
distance2shelter=distance2shelter*0.050896642




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

vframerate=len(frame_index_s)/max(frame_index_s)

#distance2shelter, pixel2cm, shelter_point=hf.get_shelterdist(paths,locations, vframerate ,vpath)
# velocity, all_locations, node_names, ds_movement, Cm2Pixel, distance2shelter, shelterpoint=pp.extract_sleap(session, animal, mouse_tracking_path = paths['mouse_tracking'], 
#                                                                                                      camera_video_path = paths['video'], vframerate=vframerate, 
#                                                                                                      Cm2Pixel_from_paths=paths['Cm2Pixel_xy'], 
#                                                                                                      Shelter_xy_from_paths=paths['Shelter_xy'],
#                                                                                                      node = 'b_back')

# print(f"calculating intananeous firing rate" )
# iFR,n_spike_times=hf.get_inst_FR(n_spike_times)#instananous firing rate


# # Get the overall maximum value
   

# res=0.001
# print(f"recalculating ndata")
# n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=res)




if len(target_regions) != 0:  # If target brain region(s) specified    
    in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
    n_region_index = n_region_index[in_region_index]    
    sorted_indices = np.argsort(n_region_index,axis=0)
    n_cluster_index = n_cluster_index[sorted_indices]
    n_channel_index=n_channel_index[sorted_indices]
    ndata=ndata[sorted_indices,:]
    n_spike_times = [n_spike_times[i] for i in sorted_indices]
   
    
else:
    pass
    #n_region_index = np.arange(len(n_region_index))





base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
#    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
base_mean = np.round(base_mean, 2)    
base_mean = base_mean[:, np.newaxis]  # now shape is (404, 1)
base_mean = np.where(base_mean == 0, 1/10000, base_mean)


windowsize_f=np.round(windowsize_s*vframerate).astype(int)
view_window_f=np.round(view_window_s*vframerate).astype(int)
#target node
node_ind=np.where(node_names=='f_back')[0][0]#node to use for tracking


#Distance to shelter


#distance2shelter, pixel2cm, shelter_point=hf.get_shelterdist(paths,locations, vframerate ,vpath)
max_dist=np.nanmax(distance2shelter)
max_vel=np.nanmax(velocity)

#Target neurons
target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
n_ybottom_ind = np.max(target_neurons_ind) + 2
n_ytop_ind = np.min(target_neurons_ind) - 2

if all_neurons:
    fileend = 'all'
    ndot = .1  #size of raster dots
else:
    fileend = 'zoom'
    ndot = .6
matplotlib.use('Agg')  #Use the Agg backend for non-interactive plotting
os.chdir(cachepath)

#%% functions


# def read_video_torch(video_path, frame_indices):
#     video, _, _ = torchvision.io.read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
#     frames = video[frame_indices]
#     return frames

# def process_frame_torch(i, frame, lframe, frames):
#     # Example processing: convert to grayscale
#     gray_frame = torch.mean(frame, dim=0, keepdim=True)
#     return i, gray_frame

def read_frames_decord(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices)
    return frames.asnumpy()

def write_video_moviepy(images, output_path, fps=25):
    clip = ImageSequenceClip(list(images), fps=fps)
    clip.write_videofile(output_path, codec='libx264', ffmpeg_params=['-vcodec', 'h264_nvenc'])
    

def process_frame(i, window_frame, lframe, frames):
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

    # plot frame + sleap labels
    ax0.imshow(frames[i], cmap='binary_r')
    pf.remove_axes(ax0, rem_all=True)

    # zoom in on mouse
    if zoom_on_mouse:
        x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
        ax0.set_xlim((x_min, x_max))
        ax0.set_ylim((y_min, y_max))
    else:
        x_min = 650
        y_max = 300

    
    if target_behavior=='loom': # show loom
        if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)

    # plot distance and velocity
    plot_start = window_frame - view_window_f  # this is in units of video frames
    plot_end = window_frame + view_window_f
    x_v = np.linspace(-5, 5, plot_end - plot_start)

    # velocity
    line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity',lw=4)
    ax1.set_ylabel('velocity (cm/s)')
    ax1.set_ylim((0, max_vel))

    # distance to shelter
    ax1_1 = ax1.twinx()
   
    line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter',lw=4)
    ax1_1.set_ylabel('distance to shelter (cm)')
    ax1_1.set_ylim((0, max_dist))

    # Add loom line, legend, xlabel, remove top axis
    ax1.set_xlim(x_v[0], x_v[-1])
    ax1.axvline((lframe - window_frame) / vframerate, color='y', lw=2)  # loom
    ax1.axvline(0, linestyle='--', color='Gray')  # current time
    ax1.spines['top'].set_visible(False)
    ax1_1.spines['top'].set_visible(False)
    ax1.legend(handles=[line1, line2])
    ax1.set_xlabel('time (s)')

    # plot raster
    plot_start = window_time - view_window_s
    plot_end = window_time + view_window_s

    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
    for j, n in enumerate(ndata):
        spikeind = n.astype(bool)
        all_spiketimes = n_time_index[spikeind]
        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
        spiketime = all_spiketimes[window_ind] - window_time
        ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[j], color='y', s=ndot)

    pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2,color='w')
    pf.remove_axes(ax2)
    ax2.set_xlabel('time (s)')
    ax2.axvline((lframe - window_frame) / vframerate, color='y', lw=2)  # loom
    ax2.axvline(0, linestyle='--', color='Gray')  # current time
    ax2.set_xlim((-5, 5))

    # if not all_neurons:  # zoom in to target areas
    #     bottom = ycoords[n_ybottom_ind]
    #     top = ycoords[n_ytop_ind]
    #     ax2.set_ylim((bottom, top))

    # save figure to array
    fig.canvas.draw()  # Force update to the figure
    
    w, h = fig.canvas.get_width_height()
    # Get an array of RGBA values and drop the alpha channel
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)[..., :3]
#    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return i, image

def process_loom(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # plot every other frame (25 instead of 60 FPS)
   # if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
      #  print(np.abs(np.mean(around_lframe - np.round(around_lframe))) )
      #  raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
    start_time = time.time()
    
    # for i, window_frame in enumerate(around_lframe):
    #     i, image = process_frame(i, window_frame, lframe, frames)
    #     images[i] = image
    with ThreadPoolExecutor(6) as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
        for future in as_completed(futures):
            i, image = future.result()
            images[i] = image
    end_time = time.time()
    print(f"ThreadPoolExecutor took {end_time - start_time:.4f} seconds")
    return images
    
    
    
def save_vid_from_images_array(images,ltime):
    print('saving movie file')
    start_time = time.time()
    write_video_moviepy(images, fr'{cachepath}\{target_behavior}_at_{np.round(ltime/60,2)}_{fileend}.mp4')
    end_time = time.time()
    print(f"moviepy: saving video took {end_time - start_time:.4f} seconds")
    
  
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time
from tqdm import tqdm

def process_loom_ProcessPoolExecutor(args):
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
from multiprocessing import Pool
import numpy as np
import time
from tqdm import tqdm

def process_frame_wrapper(args):
    return process_frame(*args)

def process_loom_Pool(args):
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

from joblib import Parallel, delayed
import numpy as np
import time
from tqdm import tqdm

def process_loom_joblib(args):
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
    results = Parallel(n_jobs=4)(delayed(process_frame)(i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe))
    
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
    #    pool.map(process_loom, args)
    
    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):

        images = process_loom([lframe, ltime, target_behavior]) #get array of frames
        
        save_vid_from_images_array(images,ltime) # save as video
    
    
#IPython.embed()
unique_behaviours = behaviour.behaviours.unique()
#unique_behaviours = [ 'escape'] #debug
for target_behavior in unique_behaviours:
    print(target_behavior)
    vid_by_behav(target_behavior)