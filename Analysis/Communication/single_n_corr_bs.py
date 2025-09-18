"""
Created by Tom Kern
Last modified 04.08.2024

Makes correlation matrix for of firing for each behaviour  specified in 'target_bs'

- The shorter the period over which correlation is computed, the larger the 
    correlation tends to be. To account for this, you can control the maximum time
    included in plot for one behaviour (time_thr)
- Neural data is downsampled to binsize .5s to increase correlation of low-firing neurons
- low firing neurons (<.4 Hz; min_frg) are excluded, because they too often
     don't fire at all during relevant periods and then produce these black bars

"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import matplotlib.patches as patches

session='231213_0'
plt.style.use('dark_background')
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline']
time_thr=20 #s
res=.5
min_frg=.4
           
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
mean_frg_Hz, base_ind=hf.baseline_firing(behaviour, resampled_t, n_Hz, velocity, frame_index_s[:-3])


#%% Correlation of single neurons



#Correlation during base
base_corr=np.corrcoef(n_Hz[:,base_ind])
rest_corr=np.corrcoef(n_Hz[:,~base_ind])

for name, corr in zip(['base', 'rest'], [base_corr, rest_corr]):
    plt.figure()
    plt.imshow(corr, vmin=-.3,vmax=.3)
    plt.colorbar()
    ticks=pf.region_ticks(n_region_index, xaxis=True)
    plt.title(f'single neuron correlation\nresolution: {res}s')
    plt.suptitle(name)
    ticks=np.append(ticks,len(ndata))
    for i in range(len(ticks)-1):
        rect = patches.Rectangle((ticks[i], ticks[i]), 
                                 ticks[i+1]-ticks[i], 
                                 ticks[i+1]-ticks[i], 
                                 linewidth=1, 
                                 edgecolor='w', 
                                 facecolor='none')
        plt.gca().add_patch(rect)
    
#%% single behaviours
# plt.close('all')

#filter neurons with too low firing out
meanHz=np.mean(n_Hz, axis=1)
high_ind=meanHz>min_frg
high_neurons=n_Hz[high_ind]




for b_name in target_bs:
    
    if b_name=='baseline':
        b_ind=base_ind.copy()
    else:
        b=behaviour[behaviour['behaviours']==b_name]
        frames_s=b['frames_s'].to_numpy()
        
        
        start_stops=hf.start_stop_array(behaviour, b_name, frame=False)
        
        
        #Make boolean index for when behaviour happens
        b_ind=np.zeros(high_neurons.shape[1])
        for stst in start_stops:  
            if stst[1]-stst[0]<2:
                stst[1]+=1
                stst[0]-=1
            b_ind+=(resampled_t>stst[0]) & (resampled_t<stst[1])
        # if sum(b_ind>1)>0:
        #     raise ValueError('sth is wrong with the computation of your b_ind')
        b_ind=b_ind.astype(bool)
    

    short_b_ind=np.where(b_ind)[0][-int(time_thr/res):]
    
    b_corr=np.corrcoef(high_neurons[:,short_b_ind])
    
#PLOT
    plt.figure(figsize=(10,10))
    plt.imshow(b_corr, vmin=-.5,vmax=.5)
    plt.colorbar()
    ticks=pf.region_ticks(n_region_index[high_ind], xaxis=True)
    plt.title(f'single neuron correlation\nresolution: {res}s')
    plt.suptitle(f'{b_name}\ncovers {hf.convert_s(len(short_b_ind)*res)}')
    ticks=np.append(ticks,len(ndata))
    for i in range(len(ticks)-1):
        rect = patches.Rectangle((ticks[i], ticks[i]), 
                                 ticks[i+1]-ticks[i], 
                                 ticks[i+1]-ticks[i], 
                                 linewidth=1, 
                                 edgecolor='w', 
                                 facecolor='none')
        plt.gca().add_patch(rect)