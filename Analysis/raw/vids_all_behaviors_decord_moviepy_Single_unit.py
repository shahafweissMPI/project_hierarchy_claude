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
import helperFunctions as hf
import plottingFunctions as pf
import os
from moviepy import VideoFileClip,ImageSequenceClip
from decord import VideoReader,cpu,gpu
from PIL import Image
import IPython



############# USER PARMAETERS #######################

#plotting params
plt.style.use('dark_background')
matplotlib.use('Agg')  #Use the Agg backend for non-interactive plotting

#User defined Parameters
cachepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\PSTH\videos\subset" #Where should the video be stored? (best an SSD)
if not os.path.exists(cachepath):
    os.makedirs(cachepath)
#cachepath=r"E:\videos" #Where should the video be stored? (best an SSD)
animal = 'afm16924'
sessions = ['240522','240524']

target_regions = ['DMPAG', 'DLPAG','LPAG','VLPAG','VMPAG']
target_cells=[202,204,212,213,222,234,237,247,386,404,458]
target_behaviors=['escape','loom','turn','startle','pup_run','pup_grab','pup_retrieve','pup_drop']


zoom_on_mouse=False
windowsize_s=7 #how long before+ after the loom should the video play
view_window_s=5#with what window should be plotted around the current time?



############# FUNCTIONS #######################
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
    
    return n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR
    
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
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(20, 12))

    # Add subplots
    ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
    ax2 = fig.add_subplot(gs[0, 1])  # All rows, second column
    ax3 = fig.add_subplot(gs[1, 1])  # All rows, second column
    

    windowtime = frame_index_s[window_frame]
    plt.suptitle(f"{animal} {session} {target_behavior} {hf.convert_s(windowtime)} FR binsize={FR_res*1000:0.0f} ms")

    # plot frame + sleap labels
    ax0.imshow(frames[i], cmap='binary_r')
    pf.remove_axes(ax0, rem_all=True)

    # zoom in on mouse
    if zoom_on_mouse:
        x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
        ax0.set_xlim((x_min, x_max))
        ax0.set_ylim((y_min, y_max))
    else:
        x_min = 90#650
        y_max = 90#300
    
    if target_behavior=='loom': # show loom
        if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)

    # plot distance and velocity
    plot_start = window_frame - view_window_f  # this is in units of video frames
    plot_end = window_frame + view_window_f
    
    duration=abs(plot_end - plot_start)
    offset=3
    x_v = np.linspace(-view_window_s, view_window_s, plot_end - plot_start)
    #x_v = np.linspace(-1 * (duration+offset), duration+offset, duration)

    # velocity
    #line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity',lw=4)
    line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='m', label='velocity',lw=4)
    ax1.set_ylabel('Speed (cm/s)')
    ax1.set_ylim((0, max_vel))
    ax1.yaxis.label.set_color("m")

    # distance to shelter
    ax1_1 = ax1.twinx()
   
    #line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter',lw=4)
    line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='c', label='distance to shelter',lw=4)
    ax1_1.set_ylabel('Distance to shelter (cm)')
    ax1_1.yaxis.label.set_color("c")
    ax1_1.set_ylim((0, max_dist))

    # Add loom line, legend, xlabel, remove top axis
    ax1.set_xlim(x_v[0], x_v[-1])
    ax1.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # loom
    ax1.axvline(0, linestyle='--', color='Gray')  # current time
    ax1.spines['top'].set_visible(False)
    ax1_1.spines['top'].set_visible(False)
    #ax1.legend(handles=[line1, line2])
    #ax1.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    ax1.set_xlabel('time (s)')

    # plot raster
    plot_start = window_time - view_window_s
    plot_end = window_time + view_window_s
   
    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
    cell_color = 'w'
    #### ####plot spikes
    ax2.axvline(0, linestyle='--', color='Gray')  # current time
    ax2.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # event
    
    # clusters=[]
    # regions=[]
    yticklabels=[]
    for j, n in enumerate(ndata):
        # clusters.append(n_cluster_index[j])
        # regions.n_region_index[j]
        yticklabels.append(f"{n_region_index[j]} #{n_cluster_index[j]}")
        spikeind = n.astype(bool)
        all_spiketimes = n_time_index[spikeind]
        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
        spiketime = all_spiketimes[window_ind] - window_time
        #cluster=n_cluster_index[j]     
        ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[j], color=cell_color, s=10)

    pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
    
    pf.remove_axes(ax2,rem_all=True)
   # ax2.set_xlabel('Time (s)')
    #ax2.tick_params(axis='y', colors='white')
    ax2.yaxis.label.set_color("white")    
    ax2.set_xlim((-view_window_s, view_window_s))       
    
    ax2.set_yticks(ycoords)
    ax2.set_yticklabels(yticklabels)
    
    minval=0.0
    ##plot firing rate    
    cell_color = 'white'
    ax3.axvline((lframe - window_frame) / vframerate, color='y', lw=4)  # event
    
   # ax3.axvline(0, linestyle='--', color='Gray')  # current time
    ax3_1 = ax3.twinx()
    
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
    #     FR1 = ax3.plot(firing_rate_time, plot_FR + j, color='green', lw=4, label='Firing rate', alpha=0.4)
        
    #     # Second plot_FR assignment
    #     plot_FR = current_FR - base_mean[j]
    #     minval = min(minval, np.nanmin(plot_FR))
    #     maxval = max(maxval, np.nanmax(plot_FR))
    #     FR2 = ax3.plot(firing_rate_time, plot_FR + j, color='red', lw=4, label='Firing rate - baseline avg', alpha=0.4)
        
    #     # Third plot_FR assignment
    #     plot_FR = zscore(current_FR)
    #     minval = min(minval, np.nanmin(plot_FR))
    #     maxval = max(maxval, np.nanmax(plot_FR))
    #     FR3 = ax3_1.plot(firing_rate_time, plot_FR + j, color='blue', lw=4, label='zscored Firing rate', alpha=0.4)
    multiple_cells= firing_rates.shape[0]==2

    for j, firing_rate in enumerate(firing_rates):                  
           
            if np.nanmax(firing_rate)>0:#don't plot cells that didn't fire
                #time axis
                firing_rate_time = firing_rate_bins_time[:-1]                       
                window_ind = (firing_rate_time > plot_start) & (firing_rate_time < plot_end)            
                firing_rate_time=firing_rate_time[window_ind]-window_time
                #firing rate
                current_FR=firing_rate[window_ind]            
                
                
                
                
                if multiple_cells==True:
                    plot_FR=current_FR
                    plot_FR=plot_FR/np.nanmax(plot_FR)
                    linewidth=2
                    #FR1= ax3.plot(firing_rate_time, plot_FR+ycoords[j], color='b',lw=2, linestyle='--',marker = ".", markersize=10,label='Firing rate Hz')
                    
                    current_FR=(current_FR-base_mean[j]) / np.std(firing_rate)
                    plot_FR=plot_FR/np.nanmax(plot_FR)
                    FR2= ax3.plot(firing_rate_time, plot_FR+ycoords[j], color='w',lw=2, linestyle='--',marker = ".", markersize=10 ,label='Firing rate Z-scored')
                    
                    
                else:                    
                    #linewidth=4
                    ax3_1.axhline(y=0, color='gray', linestyle='--', linewidth=2)  # event
                    plot_FR=current_FR                    
                    FR1= ax3.plot(firing_rate_time, plot_FR, lw=2,color='w', linestyle='--',marker = ".", markersize=10,label='Firing rate Hz')
                    current_FR=(current_FR-base_mean[j]) / np.std(firing_rate)
                    plot_FR=current_FR
                    FR2= ax3_1.plot(firing_rate_time, plot_FR+ycoords[j], color='r',lw=2, linestyle='--',marker = ".", markersize=10,label='Firing rate Z-scored')
                    minval = np.nanmin([minval, np.nanmin(firing_rate)])
                    maxval = np.nanmax([maxval, np.nanmax(firing_rate)])


                
                
                
        #plot_FR=plot_FR/np.nanmax(plot_FR)
        #minval = np.nanmin(minval, np.nanmin(plot_FR))
        
#        current_FR=zscore(current_FR)        
        
       # ax2.bar(firing_rate_time, current_FR, width=0.1,color='white', bottom=ycoords[j],  align='edge', edgecolor='y')
        #ax3.bar(firing_rate_time, current_FR, width=np.nanmedian(np.diff(firing_rate_bins_time)),color='white', bottom=j,  align='edge', edgecolor='y')
     
#        FR1= ax3.plot(firing_rate_time, current_FR+j, color='w',lw=4,label='Firing rate zscored')
        
    pf.remove_axes(ax3)
    ax3.set_xlabel('Time (s)')    
        
        #ax3_1.set_ylabel('Z-scored Firing Rate')    
    ax3.yaxis.label.set_color("white")
    ax3.tick_params(axis='y', colors='white')        
    
    #    ax3_1.yaxis.label.set_color("blue")
        #ax3.legend(handles=[FR1[0], FR2[0],FR3[0]], loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)    
        
    ax3.set_xlim((-view_window_s, view_window_s))    
        
        
    if multiple_cells==True:
        ax3.set_yticks(ycoords)
        ax3.set_yticklabels(yticklabels)
        ax3.set_ylim((np.nanmin(ycoords)-1, np.nanmax(ycoords))+1)               
        #bottom = ycoords[n_ybottom_ind]
#            top = ycoords[n_ytop_ind]
      #  ax3.set_ylim((bottom, top))
        ax3.set_ylabel('Units')
    else:
        ax3.set_ylabel('Firing Rate Hz - baseline mean')
        ax3_1.set_ylabel('Firing Rate, zscored to baseline mean')
        ax3.set_ylim((minval, maxval))   
        ax3_1.set_ylim((-4, 10))
        ax3.yaxis.label.set_color("white")
        ax3.tick_params(axis='y', colors='white')        
        ax3_1.yaxis.label.set_color("r")
        ax3_1.tick_params(axis='y', colors='r')        

    ax3.spines['top'].set_visible(False)
    ax3_1.spines['top'].set_visible(False)

    
        #ax2.plot(firing_rate_time, firing_rate + ycoords[j], color=cell_color)
    #plt.show()
    #pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
    #ax3.set_xlim((0, 1))
  
    
    

    # save figure to array
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()  # Force update to the figure
   
    
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
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
    start_time = time.time()
    
    # serial for loop
    # for i, window_frame in enumerate(around_lframe):
    #     i, image = process_frame(i, window_frame, lframe, frames)
    #     images[i] = image
    
    #multithreaded
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="creating new frames"):
            i, image = future.result()
            images[i] = image
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    #     for future in as_completed(futures):
    #         i, image = future.result()
    #         images[i] = image
    end_time = time.time()
    print(f"ThreadPoolExecutor took {end_time - start_time:.4f} seconds")
    #multiprocss
    # start_time = time.time()
    # with ProcessPoolExecutor(2) as executor:
    #     futures = [executor.submit(process_frame, i, window_frame, lframe, frames) for i, window_frame in enumerate(around_lframe)]
    #     for future in as_completed(futures):
    #         i, image = future.result()
    #         images[i] = image
    # end_time = time.time()
    # print(f"ProcessPoolExecutor took {end_time - start_time:.4f} seconds")
    plt.close('all')

    return images
    
    
    
def save_vid_from_images_array(images,ltime):
    
    start_time = time.time()
    write_video_moviepy(images, fr'{cachepath}\{animal}_{session}_{target_behavior}_at_{np.round(ltime/60,2)}.mp4')
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
        print(f"collecting frames")
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
           
        
        spike_res=0.001
        FR_res=0.1
        print(f"recalculating ndata")
        #n_time_index, ndata, firing_rates, neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds = hf.recalculate_ndata_firing_rates(n_spike_times, bin_size=spike_res)
        n_time_index, ndata, firing_rate_bins_time,firing_rates,neurons_by_all_spike_times_binary_array,neurons_by_all_spike_times_t_seconds=hf.recalculate_ndata_firing_rates2(n_spike_times,
        bin_size=spike_res, firing_rate_bin_size=FR_res)
     #   print(f"1 {np.shape(n_time_index)=}, {np.shape(ndata)=}")
       
        #sort by region
        if len(target_regions) != 0:
            in_region_index = np.where(np.isin(n_region_index, target_regions))[0]
            n_region_index = n_region_index[in_region_index]    
            sorted_indices = np.argsort(n_region_index,axis=0)        
            n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)
        
        # reduce to target cells
        if len(target_cells)!=0:
            sorted_indices = [index for index, value in enumerate(n_cluster_index) if value in target_cells]
            n_region_index,n_cluster_index,n_channel_index,n_channel_index,ndata,neurons_by_all_spike_times_binary_array,firing_rates,n_spike_times,iFR_array,iFR=resort_data(sorted_indices)       
    #    print(f"2{np.shape(n_time_index)=}, {np.shape(ndata)=}")
          
      
        print(f"calculating baseline")
        base_mean            =   hf.baseline_firing_initial_period(behaviour, n_time_index, ndata, initial_period=7)
        #    base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
        base_mean = np.round(base_mean, 2)    
        base_mean = base_mean[:, np.newaxis]  # now shape is (404, 1)
        base_mean = np.where(base_mean == 0, 1/10000, base_mean)
        
        
        windowsize_f=np.round(windowsize_s*vframerate).astype(int)
        view_window_f=np.round(view_window_s*vframerate).astype(int)
        #target node
        node_ind=np.where(node_names=='f_back')[0][0]#node to use for tracking        


        
        #Target neurons
        target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
        n_ybottom_ind = np.max(target_neurons_ind) + 2
        n_ytop_ind = np.min(target_neurons_ind) - 2
        
        
        ndot = 1.0

        os.chdir(cachepath)
        
            
        #IPython.embed()
        unique_behaviours = behaviour.behaviours.unique()
        if len(target_behaviors)!=0:#reduce to target behaviors
            unique_behaviours = [behavior for behavior in target_behaviors if behavior in unique_behaviours]
            
        #unique_behaviours = [ 'escape'] #debug
        for target_behavior in unique_behaviours:
            print(target_behavior)
            vid_by_behav(target_behavior)
            
