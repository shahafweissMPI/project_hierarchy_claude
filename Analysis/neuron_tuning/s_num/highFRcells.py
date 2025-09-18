# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:09:06 2024

@author: su-weisss
"""

from IPython import embed as stop
import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from numpy.random import choice
from time import time
from joblib import Parallel, delayed
import numpy as np
# Set parameters
#sessions= ['231215_2','240212']
cachepath=r"E:\test\vids" #Where should the video be stored? (best an SSD)
animal = 'afm16505'
sessions= ['231215_2']
	

n_perms=1000 # How often permute (should be at least 1000 times for stable results, more is better)
target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline'])
hunting_bs= ['approach','pursuit','attack','eat'] # Which behaviours are considered hunting? Important for 
resolution=.01 # In case you want to run things quick, otherwise put to None
sig_threshold=.05 #Significace threshold. Has to be larger then 1/n_perms

pre_window=1.5 # How much time before the behaviour should serve as reference for neuron's firing rate?
plt.style.use('seaborn-v0_8-talk')

savepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\plotdata\test"
animals = 'afm16505'
all_behaviors = np.array([...])  
for animal_i in animals:
    animal_avg_FR=np.array(len(animals))
    for session in sessions:
        #Load data
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
         frame_index_s] = hf.load_preprocessed(animal=animal,session=session)
        
        
        unique_behaviours = behaviour.behaviours.unique()
        
        target_bs = np.append(unique_behaviours, 'baseline')
    
        
        array1 = np.array([x for x in all_behaviors if isinstance(x, str)])
    
        # Ensure that array2 does not contain any ellipsis or other non-string objects
        array2 = np.array([x for x in target_bs if isinstance(x, str)])
        
        # Find the unique values of the intersection
        all_behaviors = np.union1d(array1, array2)
        all_behaviors=np.intersect1d(all_behaviors,all_behaviors)   
    
        #%% PRECOMPUTE SOME THINGS    
        if session=='240522':
            velocity=velocity[0:len(frame_index_s)]
        
        frame_index_s=frame_index_s[:len(velocity)]
    
        
        
        # In case you want to run things quick: downsample data
        if resolution is not None:
            ndata, n_time_index=hf.resample_ndata(ndata, n_time_index, resolution)
        bintime=n_time_index[1] # What is the width of one bin?
        
        #get baseline period
        mean_frg_hz , base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:len(velocity)])
        #zscore n
        n_Hz=ndata/n_time_index[1]
        bintime=n_time_index[1]
        base_mean=np.mean(n_Hz[:, base_ind], axis=1)
        base_std=np.std(n_Hz[:, base_ind], axis=1)
        z_ndata= (ndata - base_mean[:, None])/ base_std[:, None]  
        all_mean_frg_hz
  
# Assuming mean_frg_hz and n_region_index are already defined arrays
# Get the indices of top 10% neurons based on firing rate
ptile=75
threshold = np.percentile(mean_frg_hz, ptile)
top_indices = np.where(mean_frg_hz >= threshold)[0]
bottom_indices = np.where(mean_frg_hz < threshold)[0]

# Get the corresponding regions and rates for top neurons
top_regions = n_region_index[top_indices]
top_rates = mean_frg_hz[top_indices]

all_regions = n_region_index
all_rates = mean_frg_hz


# Count frequency of each region in top 10%
unique_regions, counts = np.unique(top_regions, return_counts=True)
all_regions, all_counts = np.unique(all_regions, return_counts=True)



all_counts_dict = dict(zip(all_regions, all_counts))

# Calculate the proportion for each unique_region
proportions = np.array([counts[i] / all_counts_dict[region] for i, region in enumerate(unique_regions) if region in all_counts_dict])

proportions
# Create bar plot
plt.figure(figsize=(12, 6))
bars = plt.bar(unique_regions, proportions)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Brain Region')
plt.ylabel('% of High Firing-rate Neurons')
plt.title(f'Distribution of Top {100-ptile}% Highest-Firing Neurons by Brain Region (>{threshold:.2f} Hz)')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{(height) :.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create histogram with threshold line
plt.figure(figsize=(12, 6))
plt.hist(mean_frg_hz, 100)
plt.axvline(x=threshold, color='r', linestyle='--', label=f'{ptile}th percentile: {threshold:.2f} Hz')
plt.xlabel('Base FR (HZ)')
plt.ylabel('Number of Neurons')
plt.title('Distribution of Base Firing Rates (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

# Select neurons that meet both criteria
specific_region = 'LPAG'
selected_neurons = np.logical_and(mean_frg_hz >= threshold, n_region_index == specific_region)
selected_indices = np.where(selected_neurons)[0]

# Print results
print(f"Number of neurons in {specific_region} above {threshold:.2f} Hz: {len(selected_indices)}")
if len(selected_indices) > 0:
    print(f"Their firing rates: {mean_frg_hz[selected_indices]}")
    
    
# Extract data for selected neurons
    selected_ndata = ndata[:, selected_indices]

    # Function to get mean activity during specific behaviors
    def get_behavior_activity(behavior_name, data, behavior_obj, time_index):
        behavior_times = behavior_obj[behavior_obj.behaviours == behavior_name].index
        behavior_activity = []
        for time in behavior_times:
            time_idx = np.abs(time_index - time).argmin()
            if time_idx < len(data):
                behavior_activity.append(data[time_idx])
        return np.mean(behavior_activity, axis=0) if behavior_activity else np.zeros(data.shape[1])

    # Calculate mean activity for escape and hunting behaviors
    escape_activity = get_behavior_activity('escape', selected_ndata, behaviour, n_time_index)
    hunting_activity = np.mean([get_behavior_activity(b, selected_ndata, behaviour, n_time_index) 
                              for b in hunting_bs], axis=0)

    # Find defensive and hunting neurons
    defensive_neurons = np.where(np.logical_and(
        escape_activity > np.mean(escape_activity),
        hunting_activity < np.mean(hunting_activity)
    ))[0]

    hunting_neurons = np.where(np.logical_and(
        escape_activity < np.mean(escape_activity),
        hunting_activity > np.mean(hunting_activity)
    ))[0]

    print(f"\nDefensive neurons (high escape, low hunting): {len(defensive_neurons)}")
    print(f"Hunting neurons (low escape, high hunting): {len(hunting_neurons)}")

    # Map back to original indices
    original_defensive_neurons = selected_indices[defensive_neurons]
    original_hunting_neurons = selected_indices[hunting_neurons]