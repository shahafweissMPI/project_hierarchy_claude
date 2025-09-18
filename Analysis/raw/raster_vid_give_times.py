"""
Created by Tom Kern
Last modified 05/08/24

Puts video of mouse next to neural firing. You can specify which parts of the video 
you want in the parameter 'vidtimes'
- Goes through each video frame, 
    -makes a plots video frame, neural activity centred at that time,
        and shelter distance and velocity. 
    -Saves this as a picture on SSD
    -loads picture, and saves into video
    --> this seems a bit unnecessarily complicated, but I didn't find another way to make it work'

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os
import cv2
cv2.setUseOptimized(True)
from time import time
from joblib import Parallel, delayed

#Parameters



vidtimes=[[0, 4860]]
#           [640, 690],
#           [1170, 1300]]
plt.ioff()
plt.style.use('dark_background')
cachepath=r"E:\test" # Where should the video be stored? (best an SSD)
animal='afm16924'
session='240524'
target_behavior='pup_grab'

all_neurons=False #whether to plot activity in the whole probe or zoom in on a certain region
target_regions=['DMPAG','LPAG','DLPAG','LPAG','Su3'] # only relevant if all_neurons==False
zoomin=False #Whether to show only mouse or whole arena
plot_markers=True #Whether to show sleap tracking 
step=100 #How many frames to 


view_window_s=5 # with what window should be plotted around the current time?


#%% load data
[frames_dropped, 
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
] = hf.load_preprocessed(session) 

vframerate=len(frame_index_s)/max(frame_index_s)
paths=hf.get_paths(session)

def pair_start_stop(group):# get video times by behaviors
    starts = group[group['start_stop'] == 'START']['frames_s'].tolist()
    stops = group[group['start_stop'] == 'STOP']['frames_s'].tolist()
    points = group[group['start_stop'] == 'POINT']['frames_s'].tolist()
    # Pair START and STOP
    start_stop_pairs = [[start, stop] for start, stop in zip(starts, stops)]
    
    # Combine pairs and points
    return start_stop_pairs + points

# Group by 'behaviours' and apply the pairing function
behaviour_timepoints = behaviour.groupby('behaviours').apply(pair_start_stop).to_dict()


vidtimes= behaviour_timepoints[target_behavior]
#%%Precompute

#Windows

view_window_f=np.round(view_window_s*vframerate).astype(int)
#target node
node_ind=np.where(node_names=='f_back')[0][0]


#Distance to shelter
shelterpoint=np.array([650,910])
pixel2cm=88/1005
distance2shelter=hf.euclidean_distance(locations[:,node_ind,:],shelterpoint,axis=1)
distance2shelter*=pixel2cm
max_dist=max(distance2shelter)
max_vel=max(velocity)

#Target neurons
target_neurons_ind=np.where(np.isin(n_region_index, target_regions))[0]
n_ybottom_ind= np.max(target_neurons_ind)+2
n_ytop_ind= np.min(target_neurons_ind)-2

if all_neurons:
    fileend='all'
    ndot=.1 # size of raster dots
else:
    fileend='zoom'
    ndot=.6



os.chdir(cachepath)


#%% define function
def write_vid(start_s, stop_s):
    
    
    plt.style.use('dark_background')
    start_frame=np.min(np.where(frame_index_s>start_s))
    stop_frame=np.max(np.where(frame_index_s<stop_s))
    
  
    #get loom frames
    loomframes=behaviour['frames'][behaviour['behaviours']==target_behavior].to_numpy()  
    

    # set parameters for video saving    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    file_name=fr'{target_behavior}_{fileend}_{start_s}s.avi'
    Vid_resolution = (1920, 1200)
    out = cv2.VideoWriter(file_name,
                          fourcc, 
                          vframerate, 
                          Vid_resolution,
                          )
    
    
    
    
    #Get loomframes
    all_frames_ind=np.arange(start_frame, stop_frame)
    chunk_iterator= range (0, stop_frame-start_frame, step)
    for  chunk_start in chunk_iterator:#load frames in chunks of 100 to save memory
    
        frame_chunk=all_frames_ind[chunk_start:chunk_start+100]
        im_frames=hf.read_frames(paths['video'], frame_chunk ) 
        if len(im_frames.shape)<3: #if there is only one frame in a loop left
            continue
        
        for chunk_i, (window_frame, window_time) in enumerate(zip(frame_chunk, frame_index_s[frame_chunk])):
            
            
            # Make figure 
            gs = gridspec.GridSpec(3, 2)
            fig = plt.figure(figsize=(20,12))
            
            # Add subplots
            ax0 = fig.add_subplot(gs[:2, 0])  # vid
            ax1 = fig.add_subplot(gs[2, 0])  # graph
            ax2 = fig.add_subplot(gs[:, 1])  # raster
            
            windowtime=frame_index_s[window_frame]
            plt.suptitle(hf.convert_s(windowtime))
            
        #%plot frame + sleap labels
        
            ax0.imshow(im_frames[chunk_i],cmap='binary_r')
            
            pf.remove_axes(axis=ax0,rem_all=True)
    
    
        # zoom in on mouse
            if zoomin:
                x_min,x_max,y_min,y_max,new_centre = pf.make_window(im_frames[chunk_i], locations[window_frame, node_ind,:], 200)
                ax0.set_xlim((x_min,x_max))
                ax0.set_ylim((y_min,y_max))
                loom_loc=(x_min, y_max)
            else:
                loom_loc=(645, 200)
        #show loom
            loom_dist=loomframes-window_frame
            try:
                lframe=loomframes[loom_dist<=0][-1]
            except IndexError:
                lframe=0
        
            if (window_frame>=lframe) and (window_frame<lframe+5*vframerate):
                pf.show_loom_in_video_clip(window_frame, lframe, vframerate, loom_loc, ax0)
            
        #Plot markers
            if plot_markers:
                ax0.plot(locations[window_frame,:,0],locations[window_frame,:,1],'.', markersize=8, color='salmon')
        
        
        #% plot distance and velocity
            plot_start=window_frame-view_window_f     # this is in units of video frames
            plot_end=window_frame+view_window_f
            x_v=np.linspace(-5,5, plot_end-plot_start)
        
        
            # avg distance between points
            # plt.plot(x_v, avg_point_distances[plot_start:plot_end], color='silver')
            #velocity
            line1,=ax1.plot(x_v,velocity[plot_start:plot_end], color='firebrick', label='velocity')
            ax1.set_ylabel('velocity (cm/s)')
            ax1.set_ylim((0,max_vel))
            
            
            #distance to shelter
            ax1_1=ax1.twinx()
            line2,= ax1_1.plot(x_v,distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter')
            ax1_1.set_ylabel('distance to shelter (cm)')
            ax1_1.set_ylim((0,max_dist))
            
            #Add loom line, legend, xlabel, remove top axis
            ax1.set_xlim(x_v[0],x_v[-1])
            ax1.axvline((lframe-window_frame)/vframerate, color='k', lw=1.5)#loom
            ax1.axvline(0,linestyle='--',color='Gray') # current time
            ax1.spines['top'].set_visible(False)
            ax1_1.spines['top'].set_visible(False)
            ax1.legend(handles=[line1,line2])
            ax1.set_xlabel('time (s)')
        
        
            
            
        #%plot raster
        
            plot_start=window_time-view_window_s
            plot_end=window_time+view_window_s
            
            
                    
                    
            ycoords=np.linspace(0,len(ndata)*4,len(ndata))*-1
            for i, n in enumerate(ndata):
                spikeind=n.astype(bool)
                all_spiketimes=n_time_index[spikeind]
                window_ind=(all_spiketimes>plot_start) & (all_spiketimes< plot_end)
                spiketime=all_spiketimes[window_ind] - window_time
                ax2.scatter(spiketime, np.zeros_like(spiketime)+ycoords[i], color='w', s=ndot)
            
            pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
            pf.remove_axes(axis=ax2)
            ax2.set_xlabel('time (s)')
            ax2.axvline((lframe-window_frame)/vframerate, color='k', lw=1.5)#loom
            ax2.axvline(0,linestyle='--',color='Gray') # current time
            ax2.set_xlim((-5,5))
            
            if not all_neurons: # zoom in to target areas
                bottom=ycoords[n_ybottom_ind]
                top=ycoords[n_ytop_ind]
                ax2.set_ylim((bottom,top))
                
            
            #save this in a video
            fig.savefig(fr'temp_plot{window_frame%2}_{start_s}.png') # the % thing is to avoid permission errors when overwriting old files
            plot_frame = cv2.imread(f'temp_plot{window_frame%2}_{start_s}.png')
            out.write(plot_frame)
        
            plt.close()
    out.release()





#%% run function in parallel

import math
num_cores = os.cpu_count()
half_cores = math.ceil(num_cores / 2)


t0=time()
for start_s, stop_s in vidtimes:
    write_vid((start_s, stop_s)
              
#Parallel(n_jobs=half_cores)(delayed(write_vid)(start_s, stop_s) for start_s, stop_s in vidtimes)
t1=time()

print('the whole thing took {convert_s(t1-t0)}')

hf.endsound()
    
    
    
print("""
      Making this script faster:
      -------------------------
      
      -Do parallel computing on each chunk. Make separate videos for
      each chunk and then concatenate the videos in the end
      
      -decrease figsize/ resolution
""")