"""
Created by Tom Kern
Last modified 05/08/24

Makes a PSTH plot for one behaviour from many neurons, one subplot per neuron
-If the same behaviour happens twice in close sequence (e.g. two attacks),
    the second behaviour will not be plotted in a new line, but will just be
    shaded in a less intense color
    
""" 

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import os
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
from time import time



# parameters
session='240524'
b_name='pup_run' # Which behaviour should be plotted (can only be 1)
target_regions=['LPAG'] # From which regions should the neurons be?



# load data
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

frame_index_s=frame_index_s[:len(velocity)]


# get baseline firing 
base_mean, _= hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
base_mean=np.round(base_mean, 2)


# Get neurons from target regions
target_n=np.where(np.isin(n_region_index, target_regions))[0]

# initialise figure
rows, cols, gs, fig=pf.subplots(len(target_n), gridspec=True)
gs.update(hspace=0.5) #vertical distance between plots

ys=[]
Hz_axs=[]
# for i, n in enumerate(target_n):
    
#     #divide one subplot into three (velocity, avg Hz, firing)
#     gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
#     axs=[]
#     for j in range(3):
#         axs.append(plt.subplot(gs_sub[j]))

#     #get behaviour frames
#     start_stop=hf.start_stop_array(behaviour, b_name)                
   

#     #Make PSTH
#     pf.psth(ndata[n], 
#             n_time_index, 
#             start_stop, 
#             velocity,
#             frame_index_s,
#             axs, 
#             window=5, 
#             density_bins=.5)
    
#     # make x/y labels
#     axs[0].set_title(n_region_index[n])
#     if i==0:
        
#         axs[0].set_ylabel('Velocity [cm/s]')            
#         axs[1].set_ylabel('avg firing [Hz]')
#         axs[2].set_ylabel('trials')
#         axs[2].set_xlabel('time [s]')
    
#     ys.append(axs[1].get_ylim()[1])
#     Hz_axs.append(axs[1])
    
# # set ylim for for all subplots
# max_y=max(ys)
# [ax.set_ylim((0,10)) for ax in Hz_axs]

# # plt.savefig(rf"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Tom_Msc\Thesis_SVGs\Communication\all_PSTH.svg")
# plt.savefig(rf"E:\test\DpG.svg")
# Break up the figure generation into chunks of 25 items from target_n at a time
chunk_size = 10
num_chunks = len(target_n) // chunk_size + (1 if len(target_n) % chunk_size != 0 else 0)

for chunk in range(num_chunks):
    start_idx = chunk * chunk_size
    end_idx = min((chunk + 1) * chunk_size, len(target_n))
    current_target_n = target_n[start_idx:end_idx]

    # initialise figure
    rows, cols, gs, fig = pf.subplots(len(current_target_n), gridspec=True)
    gs.update(hspace=0.5)  # vertical distance between plots

    ys = []
    Hz_axs = []
    for i, n in enumerate(current_target_n):
        # divide one subplot into three (velocity, avg Hz, firing)
        gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
        axs = []
        for j in range(3):
            axs.append(plt.subplot(gs_sub[j]))

        # get behaviour frames
        start_stop = hf.start_stop_array(behaviour, b_name)

        # Make PSTH
        pf.psth(ndata[n],
                n_time_index,
                start_stop,
                velocity,
                frame_index_s,
                axs,
                window=5,
                density_bins=.5)

        # make x/y labels
        axs[0].set_title(f'#{n} {base_mean[n]} Hz\n {n_region_index[n]}')
        if i == 0:
            axs[0].set_ylabel('Velocity [cm/s]')
            axs[1].set_ylabel('avg firing [Hz]')
            axs[2].set_ylabel('trials')
            axs[2].set_xlabel('time [s]')

        ys.append(axs[1].get_ylim()[1])
        Hz_axs.append(axs[1])
       # plt.get_current_fig_manager().full_screen_toggle()

    # set ylim for all subplots
    max_y = max(ys)
    [ax.set_ylim((0, 10)) for ax in Hz_axs]

    # Save the figure
    plt.savefig(rf"E:\test\LPAG{chunk}.png")
    plt.close(fig)