"""
Created by Tom Kern
Last modified 04.08.2024

Compares correlation between neurons globally (all timepoints), during hunt, 
and during baseline

- You can set behaviours that should be considered as hunting (e.g. debatable 
    whetehr eating counts as hunting) in 'hunt_bs'. You can also put maternal 
    behaviours in there if you want
-Downsampled to .5s bins to increase correlations between low-firing neurons
"""





import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import matplotlib.patches as patches

session='231213_0'
hunt_bs=['approach','pursuit', 'attack','pullback']
res=.5

plt.style.use('dark_background')

           
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

def make_corr_matrix(corr, vmin, vmax, n_region_index):
    plt.figure()
    plt.imshow(corr, vmin=vmin,vmax=vmax)
    cbar=plt.colorbar()
    cbar.set_label('correlation')
    ticks=pf.region_ticks(n_region_index, xaxis=True)
    
    ticks=np.append(ticks,len(corr))
    for i in range(len(ticks)-1):
        rect = patches.Rectangle((ticks[i], ticks[i]), 
                                 ticks[i+1]-ticks[i], 
                                 ticks[i+1]-ticks[i], 
                                 linewidth=1, 
                                 edgecolor='w', 
                                 facecolor='none')
        plt.gca().add_patch(rect)


#%% Correlation of single neurons

corr=np.corrcoef(n_Hz)

make_corr_matrix(corr, -.3, .3, n_region_index)
plt.title('global correlation')

#%% rrawwww example

n_ind=np.array([104, 154])



#Get index for when hunting is happening

hunt_ind=np.zeros_like(resampled_t)
for i, b_name in enumerate(hunt_bs):

    b_start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
      
    for b in b_start_stop:
        hunt_ind+=(resampled_t>=(b[0]-.5)) & (resampled_t<=(b[1]+.5))
hunt_ind=hunt_ind.astype(bool)



#get base index
_, base_ind=hf.baseline_firing(behaviour, resampled_t, resampled_n, velocity, frame_index_s[:-3])

n_hunt=n_Hz[:,hunt_ind]
n_base=n_Hz[:,base_ind]

hunt_corr=np.corrcoef(n_hunt[n_ind])[0,1]
base_corr=np.corrcoef(n_base[n_ind])[0,1]

plt.figure()
plt.plot(resampled_t, n_Hz[n_ind[0]],c='w', label=f'{n_region_index[n_ind[0]]} neuron')
plt.plot(resampled_t, n_Hz[n_ind[1]],c='lightslategray', label=f'{n_region_index[n_ind[1]]} neuron')
plt.title(f'base correlation: {np.round(base_corr,3)}\nhuting correlation: {np.round(hunt_corr,3)}\nresolution {res}')
pf.remove_axes()
plt.xlabel('time(s)')
plt.ylabel('spikes/s')
plt.legend()
pf.plot_events(behaviour[np.isin(behaviour['behaviours'],hunt_bs)])



#%% all corr hunt vs base


make_corr_matrix(np.corrcoef(n_hunt),
                 -.4, .4, n_region_index)
plt.title('hunting correlation')

make_corr_matrix(np.corrcoef(n_base),
                 -.4, .4, n_region_index)
plt.title('Baseline correlation')



