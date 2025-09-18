"""
Created by Tom Kern
Last modified 05/08/24

Puts video of mouse next to neural firing. 
Autmoatically makes a video around each loom
- Goes through each video frame, 
    -makes a plots video frame, neural activity centred at that time,
        and shelter distance and velocity. 
    -Saves this as a picture on SSD
    -loads picture, and saves into video
    --> this seems a bit unnecessarily complicated, but I didn't find another way to make it work'

"""
import os
import numpy as np
import cv2
cv2.setUseOptimized(True)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess


import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os


from tqdm import tqdm
import preprocessFunctions as pp
#Parameters

cachepath=r"E:\test" # Where should the video be stored? (best an SSD)
animal='afm16924'
session='240524'
plt.style.use('dark_background')

all_neurons=True #whether to plot activity in the whole probe or zoom in on a certain region
zoom_on_mouse=False
target_regions=['LPAG','DMPAG', 'DLPAG'] # only relevant if all_neurons==False


windowsize_s=7 # how long before+ after the loom should the video play
view_window_s=5 # with what window should be plotted around the current time?

#frames_dropped, behaviour, ndata, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s
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
] = hf.load_preprocessed(session=session, load_lfp=False) # 

vframerate=len(frame_index_s)/max(frame_index_s)
paths=pp.get_paths(session)


#%%Precompute

#Windows
windowsize_f=np.round(windowsize_s*vframerate).astype(int)
view_window_f=np.round(view_window_s*vframerate).astype(int)
#target node
node_ind=np.where(node_names=='f_back')[0][0]


#Distance to shelter


distance2shelter=hf.get_shelterdist(locations, node_ind )
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


#%%Plot!!!
## new 

os.chdir(cachepath)

loc_all_looms = np.where([behaviour['behaviours'] == 'loom'])[1]
time_all_looms = behaviour['frames_s'][loc_all_looms]
frame_all_looms = time_all_looms * vframerate

# Open pipe to FFmpeg

# Create figure and axes once
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2)
ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
ax2 = fig.add_subplot(gs[:, 1])  # All rows, second column

# Loop through each loom
for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)): 
        
    ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',  # Input format
            '-vcodec', 'rawvideo',  # Input codec
            '-s', '1024x1024',  # Frame size
            '-pix_fmt', 'gray',  # Pixel format
            '-r', '50',  # Frame rate
            '-i', '-',  # Input from pipe
            '-c:v', 'h264',  # Use NVENC for H.264 encoding
            '-pix_fmt', 'gray',  # Output pixel format
            fr'{cachepath}\loom_at_{np.round(ltime/60,2)}_{fileend}.mp4',
            '-an',
        ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)    

    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 1)
    #if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
#        raise ValueError('Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    frames = hf.read_frames(paths['video'], desired_frames=around_lframe) 

    for i, window_frame in enumerate(around_lframe):

        window_time = window_frame / vframerate
        
        # Clear axes
        ax0.clear()
        ax1.clear()
        ax2.clear()
        
        # Plot frame + sleap labels
        ax0.imshow(frames[i], cmap='binary_r')
        pf.remove_axes(ax0, rem_all=True)

        # Zoom in on mouse
        if zoom_on_mouse:
            x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
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

        # Plot raster
        plot_start = window_time - view_window_s
        plot_end = window_time + view_window_s
        
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
        
        if not all_neurons:  # zoom in to target areas
            bottom = ycoords[n_ybottom_ind]
            top = ycoords[n_ytop_ind]
            ax2.set_ylim((bottom, top))
        
        # Save this in a video
        fig.canvas.draw()
        plot_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_frame = plot_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ffmpeg_process.stdin.write(plot_frame.tobytes())

    print(f'done with loom at {hf.convert_s(ltime)}')

# Close the FFmpeg process
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import subprocess



# os.chdir(cachepath)

# loc_all_looms = np.where([behaviour['behaviours'] == 'loom'])[1]
# time_all_looms = behaviour['frames_s'][loc_all_looms]
# frame_all_looms = time_all_looms * vframerate

# # Loop through each loom
# for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)): 
#     ffmpeg_cmd = [
#         'ffmpeg',
#         '-y',  # Overwrite output file if it exists
#         '-f', 'rawvideo',  # Input format
#         '-vcodec', 'rawvideo',  # Input codec
#         '-s', '124x1024',  # Frame size
#         '-pix_fmt', 'rgb24',  # Pixel format
#         '-r', '50',  # Frame rate
#         '-i', '-',  # Input from pipe
#         '-c:v', 'h264_nvenc',  # Use NVENC for H.264 encoding
#         '-pix_fmt', 'yuv420p',  # Output pixel format
#         fr'{cachepath}\loom_at_{np.round(ltime/60,2)}_{fileend}.mp4'
#     ]
    
#     ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#     around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 1)
#     #if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
#     #    raise ValueError('Something is wrong in the calculation of frames (either here or in the preprocessing script)')
#     around_lframe = np.round(around_lframe).astype(int)
#     frames = hf.read_frames(paths['video'], desired_frames=around_lframe) 

#     for i, window_frame in enumerate(around_lframe):

#         window_time = window_frame / vframerate
        
#         # Make figure 
#         gs = gridspec.GridSpec(2, 2)
#         fig = plt.figure(figsize=(20, 12))
        
#         # Add subplots
#         ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
#         ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
#         ax2 = fig.add_subplot(gs[:, 1])  # All rows, second column 
        
#         windowtime = frame_index_s[window_frame]
#         plt.suptitle(hf.convert_s(windowtime))
        
#         # Plot frame + sleap labels
#         ax0.imshow(frames[i], cmap='binary_r')
#         pf.remove_axes(ax0, rem_all=True)

#         # Zoom in on mouse
#         if zoom_on_mouse:
#             x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
#             ax0.set_xlim((x_min, x_max))
#             ax0.set_ylim((y_min, y_max))
#         else:
#             x_min = 650 
#             y_max = 300
    
#         # Show loom
#         if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
#             pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)
        
#         # Plot distance and velocity
#         plot_start = window_frame - view_window_f  # this is in units of video frames
#         plot_end = window_frame + view_window_f
#         x_v = np.linspace(-5, 5, plot_end - plot_start)

#         # Velocity
#         line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity')
#         ax1.set_ylabel('velocity (cm/s)')
#         ax1.set_ylim((0, max_vel))
        
#         # Distance to shelter
#         ax1_1 = ax1.twinx()
#         line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter')
#         ax1_1.set_ylabel('distance to shelter (cm)')
#         ax1_1.set_ylim((0, max_dist))
        
#         # Add loom line, legend, xlabel, remove top axis
#         ax1.set_xlim(x_v[0], x_v[-1])
#         ax1.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
#         ax1.axvline(0, linestyle='--', color='Gray')  # current time
#         ax1.spines['top'].set_visible(False)
#         ax1_1.spines['top'].set_visible(False)
#         ax1.legend(handles=[line1, line2])
#         ax1.set_xlabel('time (s)')

#         # Plot raster
#         plot_start = window_time - view_window_s
#         plot_end = window_time + view_window_s
        
#         ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
#         for i, n in enumerate(ndata):
#             spikeind = n.astype(bool)
#             all_spiketimes = n_time_index[spikeind]
#             window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
#             spiketime = all_spiketimes[window_ind] - window_time
#             ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[i], color='w', s=ndot)
        
#         pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
#         pf.remove_axes(ax2)
#         ax2.set_xlabel('time (s)')
#         ax2.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
#         ax2.axvline(0, linestyle='--', color='Gray')  # current time
#         ax2.set_xlim((-5, 5))
        
#         if not all_neurons:  # zoom in to target areas
#             bottom = ycoords[n_ybottom_ind]
#             top = ycoords[n_ytop_ind]
#             ax2.set_ylim((bottom, top))
        
#         # Save this in a video
#         fig.canvas.draw()
#         plot_frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         plot_frame = plot_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         ffmpeg_process.stdin.write(plot_frame.tobytes())
#         plt.close()

#     print(f'done with loom at {hf.convert_s(ltime)}')

# # Close the FFmpeg process
# ffmpeg_process.stdin.close()
# ffmpeg_process.wait()




## old

os.chdir(cachepath)

loc_all_looms=np.where([behaviour['behaviours']=='loom'])[1]
time_all_looms=behaviour['frames_s'][loc_all_looms]
frame_all_looms=time_all_looms*vframerate


#loop through each loom 
for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)): 

    # set parameters for video saving
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Changed from 'MJPG' to 'H264'
  #   fr'{cachepath}\loom_at_{np.round(ltime/60,2)}_{fileend}.mp4'  # Changed from .avi to .mp4
#    out = cv2.VideoWriter(fr'{cachepath}\loom_at_{np.round(ltime/60,2)}_{fileend}.avi',
    out = cv2.VideoWriter(fr'{cachepath}\loom_at_{np.round(ltime/60,2)}_{fileend}.avi',
                          fourcc, 
                          50.0, 
                          (2000, 1200)
                          )

    
    around_lframe=np.arange(lframe-windowsize_f,lframe+windowsize_f, 1)
#    if np.abs(np.mean(around_lframe- np.round(around_lframe)))> .001: # if frame numbers are not close to integers
 #       raise ValueError(' Something is wrong in the calculation of frames (either here or in the poreprocessing script)')
    around_lframe=np.round(around_lframe).astype(int)
    frames=hf.read_frames(paths['video'], desired_frames=around_lframe) 

    for i,window_frame in enumerate(around_lframe):

        window_time=window_frame/vframerate
        
        
        # Make figure 
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(20,12))
        
        # Add subplots
        ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
        ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
        ax2 = fig.add_subplot(gs[:, 1])  # All rows, second column 
        
        windowtime=frame_index_s[window_frame]
        plt.suptitle(hf.convert_s(windowtime))
        
#%plot frame + sleap labels

        ax0.imshow(frames[i],cmap='binary_r')
        pf.remove_axes(ax0, rem_all=True)

        
    # zoom in on mouse
        if zoom_on_mouse:
            x_min,x_max,y_min,y_max,new_centre = pf.make_window(frames[i], locations[window_frame, node_ind,:], 200)
            ax0.set_xlim((x_min,x_max))
            ax0.set_ylim((y_min,y_max))
        else:
            x_min=650 
            y_max=300
    
    #show loom
        if (window_frame>=lframe) and (window_frame<lframe+5*vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)
        
    #Plot markers
        # ax0.plot(locations[window_frame,:,0],locations[window_frame,:,1],'.', markersize=1, color='salmon')


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
        ax1.axvline((lframe-window_frame)/vframerate, color='w', lw=1.5)#loom
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
        pf.remove_axes(ax2)
        ax2.set_xlabel('time (s)')
        ax2.axvline((lframe-window_frame)/vframerate, color='w', lw=1.5)#loom
        ax2.axvline(0,linestyle='--',color='Gray') # current time
        ax2.set_xlim((-5,5))
        
        if not all_neurons: # zoom in to target areas
            bottom=ycoords[n_ybottom_ind]
            top=ycoords[n_ytop_ind]
            ax2.set_ylim((bottom,top))
            
        
        
        
        
        # plt.show()
        
        #save this in a video
        fig.savefig(fr'temp_plot{window_frame%2}_{fileend}.png') # the % thing is to avoid permission errors when overwriting old files
        plot_frame = cv2.imread(f'temp_plot{window_frame%2}_{fileend}.png')
        out.write(plot_frame)

        plt.close()
    print(f'done with loom at {hf.convert_s(ltime)}')
    out.release()


print("""
      Making this script faster
      -------------------------
      
      -Have separate processes for each loom 
      (thath should be really easy actually)
      
      -Divide each video into chunks of 100 frames. Make separate videos for
      each chunk and then concatenate the videos in the end
      
      -decrease figsize/ resolution
      
      - use a different codec?? .avi consumes a lot of space""")

