"""
Created by Tom Kern
Last modified 04.08.2024

How much is each neuron's firing correlated to velocity/ distance to shelter

Sheterdistance is computed with reference to a manually set point in the shelter
- This is a bit inaccurate, the shelter should better be defined as an areas
- for the new sessions, this should be updated, because of the new camera format


Correlation
- DO properly correlate, you need the same datapoints in both vectors that you correlate.
    Because of this, I interpolate velocity/ shelterdistance to have more datapounts
- Doesn't work so welll for low-firing neurons, as they have an almost binary signal

To do this better, have a look into this paper:
    https://www.sciencedirect.com/science/article/pii/S0143416021000440
    --> Thats also how Vanessa does it in her postdoc paper (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
from scipy import interpolate



animal='afm16505'
session='231213_0'
colors=['firebrick','peru']

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

resampled_ndata, resampled_timestamps =hf.resample_ndata(ndata, n_time_index, .02)


#Compute distance to shelter
node_ind=np.where(node_names=='f_back')[0][0]
shelterpoint=np.array([650,910])
dist2shelter=hf.euclidean_distance(locations[:,node_ind,:],shelterpoint,axis=1)

#%%
plt.close('all')
for i, (name, parameter) in enumerate(zip(['velocity','shelter_distance'],
                           [velocity, dist2shelter])):
    # interpolate velocity/ shelterdistance
    # This is necessary because framerate is not exactly at 50, but a little bit above. resampling can only be done to even number though
    interp = interpolate.interp1d(frame_index_s[:-3], parameter)
    new_t= np.arange(0,max(frame_index_s[:-3]),.02)
    new_par=interp(new_t)

    
    
    corr=[]
    for n in resampled_ndata[:,:-3]:
        corr.append(np.corrcoef(n,new_par)[1,0])
        
    plt.figure('corr')    
    plt.subplot(1,2,i+1)
    plt.title(name)
    plt.plot(corr, '.', c=colors[i])
    plt.xlabel('clusters (oredere by depth)')
    plt.ylabel(f'{name} correlation')
    plt.ylim((-.3,.2))
    pf.remove_axes()
    
    
    plt.figure('validation')
    plt.plot(new_t, zscore(new_par), label=name)
plt.plot(n_time_index, zscore(ndata[205]), label='n205')
plt.legend()

print('did you update shelter point to new camera?')