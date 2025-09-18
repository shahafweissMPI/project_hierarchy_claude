"""
Created by Tom Kern
Last modified 04.08.2024

Averaging each cell-s response across multiple trials, by stretching/ compressing
each trial to the same length. Using this to look at sequence from attack -- switch -- escape
--> Adapted from Vanessas code

- Interpolates each trial to the average length of the respective behaviour
    --> you can warp all to the same length (e.g. 1 s by hardcoding avg_dur in 
                                             line 89)
- baseline is just mean firing in baseline period

Limitations
-Some switches are extremely short. I exclude the ones that are just 1 videoframe long,
    But still I feel like that makes the stretching a bit unfair
- If I do exclude a switch, I do not exclude the escape/ attack that happened
    before/ after. Maybe not a major problem, but doesn't feel right as well


"""
import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.interpolate import interp1d

plt.style.use('dark_background')



session='231215_2'
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

nres=n_time_index[1]
ndata_Hz=ndata/nres

#%% get time periods of hunt -- switch -- escape 

frame_dur=np.max(np.diff(frame_index_s))
escapes=hf.start_stop_array(behaviour, 'escape') #escape period
_,e_mask=hf.exclude_switch_trials(escapes, behaviour, return_mask=True)
s_escapes=escapes[~e_mask]

# get pre hunt
hunting_bs=['approach','pursuit','eat','attack']

pre_hunts=np.zeros_like(s_escapes)
for i,s in enumerate(s_escapes):
    pre_period=behaviour[behaviour['frames_s']<s[0]+0.1]
    last_hstart= pre_period[ 
        np.isin(pre_period['behaviours'], hunting_bs) & ( 
        pre_period['start_stop'] == 'START')].iloc[-1]
    last_hstop=pre_period[ 
        np.isin(pre_period['behaviours'], hunting_bs) & ( 
        pre_period['start_stop'] == 'STOP')].iloc[-1]
    pre_hunts[i,0]=last_hstart['frames_s']
    pre_hunts[i,1]=last_hstop['frames_s']

# get period between hunt stop and escape start 
in_between=hf.start_stop_array(behaviour, 'switch')
    
# Exclude trials where less than 1 frame between hunt and escape
no_dist_ind=(in_between[:,1]- in_between[:,0])<=frame_dur
in_between[no_dist_ind]=np.nan
print(f'{sum(no_dist_ind)} switches excluded out of{len(no_dist_ind)} switches in total')

#get baseline as control
base_frg, base_ind=hf.baseline_firing(behaviour, 
                n_time_index, 
                ndata, 
                velocity, 
                frame_index_s[:-3])


#%% Get neural data and time-warp

def time_warp(b_time):
    avg_dur=np.nanmean(b_time[:,1]-b_time[:,0])
    # avg_dur=min((avg_dur, 1))
    interp_len=int(avg_dur/nres)
    
    all_warped=[]
    for b in b_time:
                    
        n_ind=(n_time_index>b[0]) & (n_time_index<b[1])
        n=ndata_Hz[:,n_ind]
        
    # interpolation
        xnew=np.linspace(0,n.shape[1]-1,interp_len)
        interp_matrix=np.zeros((len(n),len(xnew)))
        
        if np.sum(n_ind)==0:
            interp_matrix[:,:]=np.nan
        else:
            for i, row in enumerate(n):
                set_interp = interp1d(np.arange(len(row)), row, kind='linear') 
                interp_matrix[i]=set_interp(xnew)
    
        all_warped.append(interp_matrix)
         
    return np.nanmean(all_warped, axis=0)
    
  
n_pre_hunt=time_warp(pre_hunts)
n_in_between=time_warp(in_between)    
n_switch=time_warp(s_escapes)    
n_base=np.tile(base_frg, (100, 1)).T


#%%
fig, axs= plt.subplots(1,4, sharey=True)
pf.remove_axes(axs, top=True, ticks=False)

for ax, data, name in zip(axs, 
                          [n_base,n_pre_hunt, n_in_between, n_switch],
                          ['Baseline','hunting','switch','escape']):
    ax.set_xticks([])
    ax.set_xlabel(f'warped to {data.shape[1]*nres} s')
    ax.imshow(data, aspect='auto', vmin=0, vmax=80)
    ax.set_title(name)
pf.region_ticks(n_region_index, ax=axs[0])
