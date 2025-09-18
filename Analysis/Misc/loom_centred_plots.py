"""
Created by Tom Kern
Last modified 04.08.2024

Plot neural activity in 'target_regions' at each loom,
eitehr zscore or not
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import pandas as pd


# Parameters
session='231213_0'
plt.style.use('default')

resolution=.1 # in s
before=10 # in s; relative to loomframe
after= 20# in s; relative to loomframe
zscoring=True
target_regions=None # either None or list with areas, e.g. ['VLPAG','LPAG']

# load data
[_, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
 locations, 
 node_names, 
 frame_index_s] = hf.load_preprocessed(session)



resampled_ndata, resampled_timestamps =hf.resample_ndata(ndata, n_time_index, resolution)
frng_rate=resampled_ndata/resolution



if target_regions is not None:
    frng_rate=frng_rate[np.isin(n_region_index, target_regions)]

#%%
plt.close('all')
for l in behaviour[behaviour['behaviours']=='loom']['frames_s'].to_numpy():
    
    
    plotstart=np.min(np.where(resampled_timestamps>(l-before)))
    plotstop=np.min(np.where(resampled_timestamps>(l+after)))

    
    plt.figure(figsize=(6,12))
    plt.subplot(2,1,2)


    if zscoring:
        frng=zscore(frng_rate, axis=1)
        vmin=-1.5
        vmax=1.5
        plt.title(f'zscored activity\n resolution: {resolution}')
    else:
        frng=frng_rate
        vmin=0
        vmax=30
        plt.title(f'neural activity\n resolution: {resolution}')
    
   
    
    plt.imshow(frng[:,plotstart:plotstop], 
               cmap='viridis', 
               aspect='auto',
               vmin=vmin,
               vmax=vmax)
    
    ticks=[i for i in range(plotstart, plotstop+1) if i % 100== 0]
    labels=resampled_timestamps[ticks]
    
    plt.xticks(ticks-plotstart, labels)
    plt.xlabel('time [s]')
    plt.ylabel('clusters')
    pf.region_ticks(n_region_index)
   
    
    plt.subplot(2,1,1)
    pf.plot_events(behaviour,resampled_timestamps[plotstart], resampled_timestamps[plotstop])
    
    plt.plot(frame_index_s[:-3], velocity, c='k', label='velocity') # the [:-3]is because of misalignment between video and nidq, not an optimal solution!!!
    plt.xlim(resampled_timestamps[plotstart], resampled_timestamps[plotstop])
    plt. ylabel( 'cm/ s')
    plt.ylim((0,70))
    # plt.xticks([])
    
    
    # plt.yticks([])
    # plt.ylim()
    
    hf.unique_legend()
    # plt.colorbar()


    # figpath= r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\switch_plots"
    # if not zscoring:
    #     plt.savefig(f'{figpath}\{res}_switch{s}_no_zscore.svg')
    # else:
    #     plt.savefig(f'{figpath}\{res}_switch{s}.svg')



