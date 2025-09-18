"""
Created by Tom Kern
Last modified 04.08.2024

!!!Somewhat outdated, you probably need to update the way it computes s_num !!!


-It was just for visualising how the computation works. makes a plot per neuron
    that you specify in target_n, for the behaviours you specify in target_bs
    
- it's a good script to try out how new computations of s_num affect the end-result
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import seaborn as sns

session='240524'
# target_bs=['attack', 'escape']
target_bs=[ 'attack', 'escape','pullback']
target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline'])

# b_colors=['brown','slategray']
b_colors=['coral','lightblue','dimgrey']
target_n=np.arange(120,125)
pre_window=.5 #s
n_binsize=.1
plt.style.use('dark_background')


cachepath=r"F:\scratch\shahafs_number"

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
unique_behaviours = behaviour.behaviours.unique()
target_bs = np.append(unique_behaviours, 'baseline')
if session=='240522':
    velocity=velocity[0:len(frame_index_s)]
frame_index_s=frame_index_s[:len(velocity)]
#%% Get behaviour periods
plt.close('all')
# resample ndata 

# ndata, n_time_index= hf.resample_ndata(ndata, n_time_index, n_binsize)
n_Hz=ndata/n_time_index[1]

bintime=n_time_index[1]
_, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:len(velocity)])

#Determine threshold per neuron

def get_cutoff(n=1000):

    
    base_start_stop_s=hf.rand_base_periods (base_ind, 
                        n_time_index, 
                        num_base_samples=n,  # how many baseline samples should be taken?
                        base_period_length=1,
                        overlapping=True) #how long should each baseline period be?
    
    
    base_diff=[]
    for b in base_start_stop_s:            
        pre_ind=(n_time_index>(b[0]-pre_window)) & (n_time_index<b[0])
        b_ind=(n_time_index>(b[0])) & (n_time_index<b[1])
        if np.sum(b_ind)==0:
            continue
        
        # get b minus b     
        pre_fir=np.sum(ndata[:, pre_ind], axis=1)/(np.sum(pre_ind)*bintime)
        b_fir=np.sum(ndata[:, b_ind], axis=1)/(np.sum(b_ind)*bintime)
        base_diff.append(b_fir-pre_fir)
    cutoff_mean=np.mean(base_diff, axis=0)
    cutoff_spread=np.std(base_diff, axis=0)
    
    return cutoff_spread


cutoff_spread=get_cutoff()





#zscore n

#mean_frg_Hz, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:-3])
mean_frg_Hz, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity,frame_index_s[:len(velocity)])
base_mean=np.mean(n_Hz[:, base_ind], axis=1)
base_std=np.std(n_Hz[:, base_ind], axis=1)
z_ndata= (ndata - base_mean[:, None])/ base_std[:, None]

looms=hf.start_stop_array(behaviour, 'loom')

for n_ind in target_n:
    
    fig, axs=plt.subplots(2,3, figsize=(10,7))
    axs=axs.flatten()
    for beh_i, (b_name, c) in enumerate(zip(target_bs, b_colors)):
        
        b_pd=behaviour[behaviour['behaviours']==b_name]
        frames_s=b_pd['frames_s'].to_numpy()
        
       
        
        start_stop_s=hf.start_stop_array(behaviour, b_name)
        
        #Collect for all behaviours
        all_n_firing=[]
        all_n_z=[]
        all_b_min_pre=[]
        
        for b in start_stop_s:        
        #Make boolean index for when behaviour happens
            if b_name in ['switch','escape','startle','freeze']:
                loom = looms[(looms>(b[0]-8)) & (looms<b[0])][-1]                    
                pre_ind=(n_time_index>(loom-pre_window)) & (n_time_index<loom)                
            
            else:            
                pre_ind=(n_time_index>(b[0]-pre_window)) & (n_time_index<b[0])
            b_ind=(n_time_index>(b[0])) & (n_time_index<b[1])
            if np.sum(b_ind)==0:
                continue
            both_ind=pre_ind+b_ind
            n_time=n_time_index[both_ind] - b[0]
                        
            # Plot n firing rate 
            n=n_Hz[n_ind, both_ind]            
            axs[0].plot(n_time, n, c=c, label=b_name)
            all_n_firing.append(n)
            
            #Plot zscored n firing
            n_z=z_ndata[n_ind, both_ind]
            axs[1].plot(n_time, n_z, c=c, label=b_name)
            all_n_z.append(n_z)
            
            # get b minus b 
            
            pre_fir=np.sum(ndata[n_ind, pre_ind])/(np.sum(pre_ind)*bintime)
            b_fir=np.sum(ndata[n_ind, b_ind])/(np.sum(b_ind)*bintime)
            all_b_min_pre.append(b_fir-pre_fir)
        all_b_min_pre=np.array(all_b_min_pre)
        
        #get P+/ p-
        cutoff=cutoff_spread[n_ind]
        p_plus=np.sum(all_b_min_pre>cutoff)/len(all_b_min_pre)
        p_minus=np.sum(all_b_min_pre<-cutoff)/len(all_b_min_pre)
        shahafs_num=(p_plus-p_minus)
        
        
        #% Start plotting
        # plot b - pre
        vparts=axs[2].violinplot(all_b_min_pre, [beh_i], showmeans=True, showextrema=False)
        vparts['bodies'][0].set_facecolor(c)
        vparts['bodies'][0].set_alpha(.7)
        vparts['cmeans'].set_color(c)
        axs[2].scatter(np.ones_like(all_b_min_pre)*beh_i,all_b_min_pre, c=c)

        #plot p+
        axs[3].bar(beh_i, p_plus, color=c)
        
        #plot p-
        axs[4].bar(beh_i, p_minus, color=c)
        
        #plot shhafs number
        axs[5].bar(beh_i, shahafs_num, color=c)

    
    #format figure
    fig.suptitle(f'neuron: {n_ind}\nbehaviour: {b_name}\nsampling: {bintime}s\nfiring rate: {mean_frg_Hz[n_ind]}Hz')    
    
    #firing subplot 1,1
    axs[0].set_title('firing')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('firing (Hz)')
    axs[0].set_xlim((-pre_window, pre_window))
    axs[0].axvline(0,c='w')
    hf.unique_legend(axs[0])
    
    
    #zscore subplot 1,2
    axs[1].set_title('zscored firing')
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('zscore')
    axs[1].set_xlim((-pre_window, pre_window))
    axs[1].axvline(0,c='w')
    hf.unique_legend(axs[1])
        # 'also plot baseline firing as axhline'
    
    #Violin b- pre
    axs[2].set_title('firing durin_b-pre')
    axs[2].set_ylabel('b - pre')
    axs[2].set_xticks(np.arange(len(target_bs)), target_bs)
    axs[2].axhline(0, c='w')
    axs[2].axhline(cutoff, c='w', ls='--')
    axs[2].axhline(-cutoff, c='w', ls='--')
    axs[2].set_xlim((-1,len(target_bs)+1))
    
    #barplots
    axs[3].set_ylabel('chance of firing increasing')
    axs[3].set_title('p+')
    axs[3].set_xticks((np.arange(len(target_bs))),target_bs) 
    axs[3].set_ylim((0,1))
    # axs[3].axhline(.5,c='w',ls='--')
    
    axs[4].set_ylabel('chance of firing decreasing')
    axs[4].set_title('p-')
    axs[4].set_xticks((np.arange(len(target_bs))),target_bs) 
    axs[4].set_ylim((0,1))
    # axs[4].axhline(.5,c='w',ls='--')
    
    axs[5].set_ylabel('chance of firing changing positively, if it does change')
    axs[5].set_title("Shahaf's Number")
    axs[5].set_xticks((np.arange(len(target_bs))),target_bs)    
    axs[5].set_ylim((-1,1))
    axs[5].axhline(0,c='w',ls='--')
    
    
    pf.remove_axes(axs)
    
    # plt.savefig(fr'{cachepath}\n{n_ind}.png')
    # plt.close()