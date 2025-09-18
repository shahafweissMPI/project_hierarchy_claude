"""
Created by Tom Kern
Last modified 04.08.2024

Plot activity of a single neuron over time
with 'b' you can set which behaviours should be marked in plot

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf



session='231213_0'
plt.style.use('dark_background')
b=['approach','pursuit', 'pullback','escape'] 

[_, 
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

b_names=behaviour['behaviours']

#%% plot
n=117
 
    
plt.figure();
plt.title(f'neuron {n}');
plt.plot(n_time_index, ndata[n], c='w');
pf.plot_events(behaviour[np.isin(b_names, b)]);
hf.unique_legend();
plt.xlabel('time (s)');
plt.ylabel('num spikes');
pf.remove_axes()