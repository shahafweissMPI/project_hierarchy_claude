"""
Created by Tom Kern
Last modified 04.08.2024

trials*time plots of escape velocity, like vanessa has in her papers
In the plots, each line is an escape, color is the velocity. Sorted by peak velocity.

pre_time: escapes with no loom within pre_time s before them are excluded
pre/post: For plotting, how many frames before/after loom should the plot show
velocity
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
import seaborn as sns
from scipy.ndimage import convolve



# animal='afm16618'
plt.style.use('default')
animals=['afm16924']

pre_time=7.5 #how many s before an escape should a loom have occured (note this is running onset, not turn)

# plotting window
pre=1* 50 # frames
post=10* 50 # frames

#%% Collect all behaviour in one big v


bigB, all_frame_index, all_vel, animal_borders=hf.bigB_multiple_animals(animals)
vframerate=all_frame_index[-1]/len(all_frame_index)

def sort_by_peak (loom_vels):
    post_loom=loom_vels[:,pre+10:]
    kernel=np.ones((1,5))/5
    smoothed = convolve(post_loom, kernel, mode='constant', cval=0.0)
    
    peak_indices = np.argmax(smoothed, axis=1)
    peak_values=smoothed[range(len(smoothed)),peak_indices]
    peak_indices[peak_values<30]=post
    sorted_indices = np.argsort(peak_indices)

    return sorted_indices


#%% all looms
looms=bigB['frames_s'][bigB['behaviours']=='loom']

loom_vels=[]

last_loom=0
for loom in looms:
    if loom-last_loom >10:
        ind=np.min(np.where(all_frame_index> loom)[0]) 
        
        loom_vels.append(all_vel[ind-pre:ind+post])
        last_loom=loom
loom_vels=np.array(loom_vels)


#sort by peak velocity
sorted_indices=sort_by_peak (loom_vels)
sorted_loom_vels = loom_vels[sorted_indices]



# plot
cmap=pf.make_cmap(['steelblue','w','darkorange'], [0,20,80])
plt.figure()
plt.imshow(sorted_loom_vels,cmap=cmap, aspect='auto', vmax=80)
plt.colorbar()
plt.axvline(pre, ls='--',color='w')
plt.title('all looms')


#%% Escapes and switches

for b_name in ['escape','switch']:
    b_times=hf.start_stop_array(bigB,b_name, pre_time=pre_time)
    b_vel=[]
    for b in b_times:
        pre_bs=bigB[(bigB['frames_s']>(b[0]-pre_time)) & (bigB['frames_s']<b[0])]
        
        if not 'loom' in pre_bs['behaviours'].values:
            continue
        
        loom=pre_bs['frames_s'][pre_bs['behaviours']=='loom'].iloc[0]
        
        loom_ind=np.min(np.where(all_frame_index>loom))
       
        b_vel.append(all_vel[loom_ind-pre:loom_ind+post])
    b_vel=np.array(b_vel)
    
    #sort by peak 
    sort_ind=sort_by_peak(b_vel)
    
    plt.figure()
    plt.imshow(b_vel[sort_ind],cmap=cmap, aspect='auto', vmax=100)
    cbar=plt.colorbar()
    cbar.set_label('velocity (cm/s)')
    plt.axvline(pre, ls='--',color='w')
    plt.title(b_name) 
    plt.ylabel('trials')
    tics=np.arange(0,11,2)
    plt.xticks(tics/vframerate, tics)
    plt.xlabel('time (s)')
        
        
        
        
        
        
        