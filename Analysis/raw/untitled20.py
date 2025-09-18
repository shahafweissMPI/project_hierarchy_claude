# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:21:09 2024

@author: su-weisss
"""

"""
Created by Tom Kern
Last modified 05/08/24

Makes a PSTH plot for all behaviours from many neurons, one subplot per neuron
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
target_regions=['LPAG']  # From which regions should the neurons be?

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

frame_index_s = frame_index_s[:len(velocity)]

# get baseline firing 
base_mean, _ = hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
base_mean = np.round(base_mean, 2)

# Get neurons from target regions
target_n = np.where(np.isin(n_region_index, target_regions))[0]

# Get all unique behaviours
unique_behaviours = behaviour.behaviours.unique()
behaviours=unique_behaviours
 

# Break up the figure generation into chunks of 10 neurons at a time
chunk_size = len(behaviours)
num_chunks = 1#len(target_n) // chunk_size + (1 if len(target_n) % chunk_size != 0 else 0)

for chunk in range(num_chunks):
    start_idx = chunk * chunk_size
    end_idx = min((chunk + 1) * chunk_size, len(target_n))
    current_target_n = target_n[start_idx:end_idx]
    
    # initialise figure
    rows, cols, gs, fig = pf.subplots(len(current_target_n), gridspec=True)
    gs.update(hspace=0.5)  # vertical distance between plots
    
    # Set figure size
    fig.set_size_inches(19.20, 10.80)  # width, height in inches at 100 DPI

    ys = []
    Hz_axs = []
    for i, n in enumerate(current_target_n):
        # Create subplots for each behaviour
        gs_sub = gridspec.GridSpecFromSubplotSpec(len(behaviours), 1, subplot_spec=gs[i])
        axs = [plt.subplot(gs_sub[j]) for j in range(len(behaviours))]

        for idx, b_name in enumerate(behaviours):
            # get behaviour frames
            start_stop = hf.start_stop_array(behaviour, b_name)
            if len(start_stop) == 0:
                continue

            # Make PSTH
            pf.psth(ndata[n],
                    n_time_index,
                    start_stop,
                    velocity,
                    frame_index_s,
                    axs,  # Provide the corresponding axis
                    window=5,
                    density_bins=.5)

            # make x/y labels
            axs[idx].set_title(f'Neuron #{n} ({n_region_index[n]}) - Behaviour: {b_name}')
            axs[idx].set_ylabel('Firing rate [Hz]')
            if idx == len(behaviours) - 1:
                axs[idx].set_xlabel('Time [s]')

            ys.append(axs[idx].get_ylim()[1])
            Hz_axs.append(axs[idx])

    # set consistent y-limits for all subplots
    max_y = max(ys)
    [ax.set_ylim((0, max_y)) for ax in Hz_axs]

    # Save the figure without displaying it
    plt.savefig(rf"E:\test\LPAG{chunk}.png", bbox_inches='tight')
    plt.close(fig)
