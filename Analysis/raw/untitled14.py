# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:58:12 2024

@author: su-weisss
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:45:50 2024

@author: su-weisss
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import os
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
from time import time

from pathlib import Path

# parameters
animal = 'afm16924'
session='240524'
target_regions = [ 'DMPAG', 'DLPAG','LPAG']
save_path=Path(rf"E:\test\Figures\PSTH\{animal}")

# load data
[frames_dropped, 
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

if session=='240522':
    velocity=velocity[0:len(frame_index_s)]

frame_index_s=frame_index_s[:len(velocity)]

# get baseline firing 
base_mean, _= hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
base_mean=np.round(base_mean, 2)

# Get neurons from target regions
target_n=np.where(np.isin(n_region_index, target_regions))[0]

# Function to plot a chunk of neurons
def plot_neurons(neurons_chunk, chunk_num):
    rows, cols, gs, fig = pf.subplots(len(neurons_chunk), gridspec=True)
    fig.set_size_inches(19.2, 12)  # Set figure size to 1920 by 1200 pixels
    gs.update(hspace=0.5)  # vertical distance between plots

    ys = []
    Hz_axs = []
    for i, n in enumerate(neurons_chunk):
        # divide one subplot into three (velocity, avg Hz, firing)
        gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
        axs = [plt.subplot(gs_sub[j]) for j in range(3)]

        for b_name in unique_behaviours:
            # get behaviour frames
            start_stop = hf.start_stop_array(behaviour, b_name)
            time_window = 5
            if b_name == 'eat':
                time_window = 10

            # Make PSTH plot
            pf.psth(ndata[n], 
                    n_time_index, 
                    start_stop, 
                    velocity,
                    frame_index_s,
                    axs, 
                    window=time_window, 
                    density_bins=.5)

        # make x/y labels
        axs[0].set_title(f"#{neurons_chunk[i]} {n_region_index[n]} {base_mean[i]}Hz")
        if i == 0:
            axs[0].set_ylabel('Velocity [cm/s]')
            axs[1].set_ylabel('avg firing [Hz]')
            axs[2].set_ylabel('trials')
            axs[2].set_xlabel('time [s]')

        ys.append(axs[1].get_ylim()[1])
        Hz_axs.append(axs[1])

    # set ylim for all subplots
    max_y = max(ys)
    [ax.set_ylim((0, 10)) for ax in Hz_axs]
   
    if not Path.is_dir(save_path):
        Path.mkdir(save_path, exist_ok=True)
    plt.savefig(f'{fig_path}_{chunk_num}.png')
    plt.close(fig)  # Close the figure after saving

# Split target_n into chunks of 25 and plot each chunk
chunk_size = 25

unique_behaviours = behaviour.behaviours.unique()

fig_path = Path(f'{save_path}\{animal}_{session}_all_behaviours')

for i in range(0, len(target_n), chunk_size):
    plot_neurons(target_n[i:i + chunk_size], i)
    # Close all open plots
    plt.close('all')
