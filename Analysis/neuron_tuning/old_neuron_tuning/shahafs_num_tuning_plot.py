import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore

session='231213_0'
pre_window=5#s
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


#%% Get behaviour periods
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat']




shahafs_nums=[]
for b_name in target_bs:
    b=behaviour[behaviour['behaviours']==b_name]
    frames_s=b['frames_s'].to_numpy()

    
    b_pd=behaviour[behaviour['behaviours']==b_name]
    frames_s=b_pd['frames_s'].to_numpy()
    
    # # Exclude_switches
    # if b_name in ['startle', 'escape', 'freeze']:
    #     frames_s, mask=hf.exclude_switch_trials(frames_s, behaviour, return_mask=True)
    #     b_pd=b_pd[mask]
    
    start_stop_s=hf.start_stop_array(behaviour, None, frame=False, b_pd=b_pd)
    
    
    # Calculate shahafs number
    all_b_min_pre=[]
    for b in start_stop_s:  
        
    #Make boolean index for when behaviour happens
        pre_ind=(n_time_index>(b[0]-pre_window)) & (n_time_index<b[0])
        b_ind=(n_time_index>(b[0])) & (n_time_index<b[1])
        if np.sum(b_ind)==0:
            continue
        both_ind=pre_ind+b_ind

        
        # get b minus b 
        pre_fir=np.mean(ndata[:, pre_ind], axis=1)
        b_fir=np.mean(ndata[:, b_ind], axis=1)
        all_b_min_pre.append(b_fir-pre_fir)
    all_b_min_pre=np.array(all_b_min_pre).T
    
    #get P+/ p-
    p_plus=np.sum(all_b_min_pre>0, axis=1)/all_b_min_pre.shape[1]
    p_minus=np.sum(all_b_min_pre<0, axis=1)/all_b_min_pre.shape[1]
    sum_plusminus=(p_plus+p_minus)
    sum_plusminus[sum_plusminus==0]=np.nan
    
    shahafs_nums.append((p_plus-p_minus)/sum_plusminus)
shahafs_nums=np.array(shahafs_nums).T

#%Make imshow plot
# plt.close('all')
#MEAN
values=[-1, -.1,.1,  1]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)

plt.figure()
plt.imshow(shahafs_nums,
           vmin=values[0],
           vmax=values[-1],
           aspect='auto',
           cmap=cmap)
pf.remove_axes()
plt.xticks(np.arange(len(target_bs)),target_bs)
pf.region_ticks(n_region_index)
plt.suptitle('neural tuning to behaviour')
cbar=plt.colorbar()
cbar.set_label("Shahaf's Number")
plt.axvline(2.5, c='k')
plt.axvline(3.5, c='k',ls='--')
plt.axvline(6.5, ls='--',c='k')
plt.axvline(7.5, c='k')

    
    
print("""
      Maybe you only wanna include changes if they are larger than e.g. .1""")
    
    