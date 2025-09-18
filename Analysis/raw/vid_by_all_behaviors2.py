
from concurrent.futures import ThreadPoolExecutor, as_completed
import preprocessFunctions as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from PIL import Image
import IPython
from multiprocessing import Pool
from tqdm import tqdm

import time



from decord import VideoReader, cpu


#plotting params
plt.style.use('dark_background')
import matplotlib

#User defined Parameters
#cachepath=r"F:\stempel\data\vids" #Where should the video be stored? (best an SSD)
cachepath=r"E:\test\vids" #Where should the video be stored? (best an SSD)
animal = 'afm16924'
session = '240522'
#   target_behavior='pup_run'
target_regions = [ 'DpWh','DMPAG', 'DLPAG','LPAG', 'Su3']




all_neurons=True #whether to plot activity in the whole probe or zoom in on a certain region
zoom_on_mouse=False



windowsize_s=10 #how long before+ after the loom should the video play
view_window_s=5 #with what window should be plotted around the current time?
    
    
#%% load data
[dropped_frames,
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
 locations, 
 node_names, 
 frame_index_s, 
] = hf.load_preprocessed(session, load_lfp=False)

vframerate=len(frame_index_s)/max(frame_index_s)
paths=pp.get_paths(session)
windowsize_f=np.round(windowsize_s*vframerate).astype(int)
view_window_f=np.round(view_window_s*vframerate).astype(int)
#target node
node_ind=np.where(node_names=='f_back')[0][0]


#Distance to shelter


distance2shelter,loc,vector=hf.get_shelterdist(locations, node_ind )

# #debug
# dx= 0.04
# x_coords_50 = loc[:,1]
# y_coords_50 = loc[:,0]
# distance2shelter_50 = distance2shelter
# times = np.arange(0, len(loc) / 50, 1/50)

# plt.plot(x_coords_50,y_coords_50)
# plt.plot(times,distance2shelter_50)
# plt.scatter(x_coords_50, y_coords_50, c=distance2shelter_50, cmap='plasma')
# plt.colorbar(label='velocity to Shelter')

# plt.xlabel('X Coordinates')
# plt.ylabel('Y Coordinates')

# distance2shelter[0]
#
max_dist=max(distance2shelter)
max_vel=max(velocity)

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

#%% Plot!!!
matplotlib.use('Agg')  #Use the Agg backend for non-interactive plotting
os.chdir(cachepath)

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

    if target_behavior=='loom': # show loom# show loom
        if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)

    # plot distance and velocity
    plot_start = window_frame - view_window_f  # this is in units of video frames
    plot_end = window_frame + view_window_f
    x_v = np.linspace(-5, 5, plot_end - plot_start)

    # velocity
    line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity')
    ax1.set_ylabel('velocity (cm/s)')
    ax1.set_ylim((0, max_vel))

    # distance to shelter
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

    # plot raster
    plot_start = window_time - view_window_s
    plot_end = window_time + view_window_s

    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
    for j, n in enumerate(ndata):
        spikeind = n.astype(bool)
        all_spiketimes = n_time_index[spikeind]
        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
        spiketime = all_spiketimes[window_ind] - window_time
        ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[j], color='w', s=ndot)

    pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
    pf.remove_axes(ax2)
    ax2.set_xlabel('time (s)')
    ax2.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
    ax2.axvline(0, linestyle='--', color='Gray')  # current time
    ax2.set_xlim((-5, 5))

    if not all_neurons:  # zoom in to target areas
        bottom = ycoords[n_ybottom_ind]
        top = ycoords[n_ytop_ind]
        ax2.set_ylim((bottom, top))

    # save figure to array
    fig.canvas.draw()  # Force update to the figure
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return i, image
# async def process_frame_async(i, window_frame, lframe, frames):
#     # Assuming process_frame is an async function
#     return await =process_frame(i, window_frame, lframe, frames)

# async def process_loom_async(args):
#     lframe, ltime, target_behavior = args
#     around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # plot every other frame (25 instead of 60 FPS)
#     if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
#         raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
#     around_lframe = np.round(around_lframe).astype(int)
#     frames = read_frames_decord(paths['video'], around_lframe)

#     images = [None] * len(around_lframe)
#     print(f' loading frames')
    
#     tasks = [process_frame(i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    
#     for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing frames"):
#         result = await future
#         i, image = result
#         images[i] = image
        
#     print(f'done loading frames')
    
#     return images
# async def run_async(lframe, ltime, target_behavior):
#         args = (lframe, ltime, target_behavior)  # Replace with actual arguments
#         images= asyncio.run(process_loom_async(args))              
#         return images
   

def process_loom(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # plot every other frame (25 instead of 60 FPS)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    # To run the async function

    print("plotting frames to figure array")  
        
    start_time = time.time()
    with ThreadPoolExecutor(44) as executor:
     futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    
    # Wrap the futures with tqdm to display a progress bar
    for future in tqdm(as_completed(futures), total=len(futures)):
        i, image = future.result()
        images[i] = image
         
    end_time = time.time()

    print(f"it took {end_time - start_time:.4f} seconds")
    
    return images
    
  
  
def write_images_to_video(images,ltime):
    # Create video using ffmpeg-python
    print('saving movie file')
    start_time = time.time()
    write_video_moviepy(images, fr'{cachepath}\{target_behavior}_at_{np.round(ltime/60,2)}_{fileend}.mp4')
    end_time = time.time()
    print(f"moviepy: saving video took {end_time - start_time:.4f} seconds")
    
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
        
         print(f"processing event {iframe}/{len(frame_all_looms)}")
         start_time = time.time()
         images=process_loom([lframe, ltime, target_behavior])
         end_time = time.time()
         print(f"ThreadPoolExecutor took {end_time - start_time:.4f} seconds")
         
         write_images_to_video(images,ltime)
        
        # start_time = time.time()
        # images = process_frame_async(iframe,lframe, ltime, target_behavior)
        # end_time = time.time()
        # print(f"asyncio took {end_time - start_time:.4f} seconds")
        
   
    

unique_behaviours = behaviour.behaviours.unique()
#unique_behaviours = [ 'loom' ] #debug
#unique_behaviours = ['pup_drop'] #debug
for target_behavior in unique_behaviours:
    print(target_behavior)
    vid_by_behav(target_behavior)