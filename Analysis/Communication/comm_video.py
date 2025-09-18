"""
Created by Tom Kern
Last modified 04.08.2024

Makes video of correlation between all neurons across time. 
-Takes 50s (time_interval) of recording, correlates activity between all neurons
- then moves forward 5s (step), takes again the next 50s and correlates activity
- at each step, makes a figure of correlation matrix, and writes that as a new frame into video
- white lines in behaviour plot (axs[0]) indicate period in which correlation is obtained

Important details
-Sorts out neurons with firing below .4 (min_firing), because they too often have 
    no spikes at all in the 50s periods
- data is downsampled to .5s bins, to incrase correlations in low-firing neurons
-framerate of resulting video you can determine with 'interval', which gives 
    the time in ms between frames

""" 


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# Set parameters
session='231213_0'

res=.5 # binsize
time_interval=50#s
step=5#s
min_firing=.4 #Hz; neurons below that will be sorted out
target_regions = ['DpWh','LPAG','VLPAG'] # which regions should be shown in plot
interval=40 # how many ms should be between each frame in the video (40 is equivalent to framerate of 25)
b_window=12 *60#s; what timeperiod should be shown in the behaviour plot above correlation matrix
savepath=r'F:\scratch\comm_video'

plt.style.use('dark_background')
     

# Load data      
[dropped, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index, 
 velocity, 
 _, 
 _, 
 frame_index_s] = hf.load_preprocessed(session)


resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, res)
n_Hz=resampled_n/res
mean_frg_Hz, base_ind=hf.baseline_firing(behaviour, resampled_t, n_Hz, velocity, frame_index_s[:-dropped])




#%% Correlation of single neurons
plt.close('all')
target_n_ind=np.isin(n_region_index, target_regions )

#filter neurons with too low firing out
meanHz=np.mean(n_Hz, axis=1)
high_ind=(meanHz>min_firing) & target_n_ind
high_neurons=n_Hz[high_ind]





#%% Create a figure
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
box1 = ax1.get_position()
ax1.set_position([0, box1.y0, 1, box1.height])

pf.remove_axes(ax2)
pf.remove_axes(ax1, left=True, ticks=False)
ax1.set_xlabel('time (s)')


# Update function for the animation
def update(t_ind):

    ax1.clear()
    ax2.clear()
    
    
    end=t_ind+int(time_interval/res)
    
    #Show where we are in terms of behaviour
    
    ax1.axvline(resampled_t[t_ind])
    ax1.axvline(resampled_t[end])
    
    corr=np.corrcoef(high_neurons[:,t_ind:end])
    ax2.imshow(corr, vmin=-.4,vmax=.4)

    
    plotmin=resampled_t[t_ind]-b_window
    plotmax=resampled_t[end]+b_window
    pf.plot_events(behaviour,ax =ax1)
    ax1.set_xlim((plotmin, plotmax))
    
    ticks=pf.region_ticks(n_region_index[high_ind], xaxis=True)
    plt.title(f'single neuron correlation\nresolution: {res}s')
    plt.suptitle(hf.convert_s(resampled_t[t_ind]))
    ticks=np.append(ticks,len(ndata))
    for i in range(len(ticks)-1):
        rect = patches.Rectangle((ticks[i], ticks[i]), 
                                 ticks[i+1]-ticks[i], 
                                 ticks[i+1]-ticks[i], 
                                 linewidth=1, 
                                 edgecolor='w', 
                                 facecolor='none')
        ax2.add_patch(rect)

# Create the animation
tpoints=np.arange(0,len(resampled_t), int(step/res))
ani = animation.FuncAnimation(fig, update, frames=tpoints, interval=interval)

# Save the animation as a video file
ani.save(rf'{savepath}\{session}_zoom.mp4', writer='ffmpeg')

plt.show()

plt.close('all')
