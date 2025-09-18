"""
Created by Tom Kern
Last modified 04.08.2024

Does lfp signal contain information relevant to instinctive behaviours?

axs[0]
-behaviour + velocity trace

axs[1]
-raw lfp traces, taken from the most central channel of each region

axs[2]
-lfp power across time

Outcome:
    -not really anything
    
Important:
    -you need to have lfp signal preprocessed in the preprocess_all.py script
    - You can modify the result a bit by
        \\Changing vmin/ vmax in the imshow function
        \\changing frequency cutoff
        \\ changing windwo size and overlap (basically determines in which intervals 
                                             lfp power is calculatd)
    - this analysis is inspired by Lu et al (2023) 
        https://www.nature.com/articles/s41378-023-00546-8/figures/5
        
        But we don't find what they find'

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
from scipy.signal import stft
from scipy.stats import zscore

animal='afm16505'
session='231213_0'
target_regions=['LPAG', 'VLPAG','DpWh']
plt.style.use('default')

window=5 # s; In what window to compute lfp power
overlap=2.5 # s; what overlap should each window have
freq_cutoff=100 #til when to plot frquency



# load data
[behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
 locations, 
 node_names, 
 frame_index_s, 
 lfp, 
 lfp_time, 
 lfp_framerate] = hf.load_preprocessed(session, load_lfp=True)



#%% get one channel per region
regions_pd = pd.Series(n_region_index)
#where does a new region start?
changes = regions_pd.ne(regions_pd.shift())
#assign number to each area complex
groups = changes.cumsum()

# Find the middle channel of each region
target_chans = regions_pd.groupby(groups).apply(lambda x: x.index[len(x)//2]).values


#Get y values for lfp traces
ylfp=np.linspace ( 0,-50,len(target_regions)) #needs to be - , so that they are ordered in the same way as the frequency plots

#%%
plt.close('all')
window*=lfp_framerate
overlap*=lfp_framerate

# Plot velocity and behaviour
fig, axs=plt.subplots(5,1, sharex=True)
pf.plot_events(behaviour, ax=axs[0])
axs[0].plot(frame_index_s[:-3],velocity, lw=.3, c='k', label='velocity')
axs[0].set_ylabel('velocity (cm/s)')
pf.remove_axes(axs[0],bottom=True)
hf.unique_legend(axs[0])

#Frequency plot +LFP traces

i=0
rnames=[] # This is to make sure that the labels are in the right order
for chanind in target_chans:
    rname=n_region_index[chanind]
    if rname not in target_regions:
        continue
    axs[1].plot(lfp_time,zscore(lfp[chanind])+ylfp[i], lw=.2, c='teal')
    #get power spectrum
    frequencies, times, Zxx = stft(lfp[chanind], lfp_framerate, nperseg=window, noverlap=overlap)
    
    # get resolution
    res=hf.unique_float(np.diff(times))
    plt.suptitle(f'resolution: {window/lfp_framerate} s')
    
    # Compute the power spectrum
    power_spectrum = np.abs(Zxx)**2
    

    freq_ind=frequencies<freq_cutoff
    db=10 * np.log10(power_spectrum[freq_ind])
    a=power_spectrum[freq_ind]
    im=axs[i+2].imshow(db, 
               origin='lower', 
               extent=[times.min(),
                       times.max(), 
                       frequencies[freq_ind].min(), 
                       frequencies[freq_ind].max()],
               vmin=-15,
               vmax=25,
               aspect='auto',
               cmap='nipy_spectral')
    
    axs[i+2].set_title(rname)
    pf.remove_axes(axs[i+2])
    
    rnames.append(rname)
    i+=1
    
axs[4].set_xlabel('time[s]')

for i in range(2,5):
    axs[i].set_ylabel('frequency')
cbar_ax = fig.add_axes([0.93, 0.1, 0.01, 0.6])
cbar_ax.set_title('power [db]')
fig.colorbar(im,cax=cbar_ax)

axs[0].set_title('behaviour')

axs[1].set_title('LFP traces')
# axs[1].set_yticks([])
pf.remove_axes(axs[1], bottom=True, left=True)

axs[1].set_yticks(ylfp, rnames)







