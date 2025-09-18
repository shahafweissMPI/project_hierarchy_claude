import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf


session='231213_0'
exclude_switches=False
pre_window=5 #s

plt.style.use('dark_background')

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
 frame_index_s] = hf.load_preprocessed(session, load_lfp=False)


#%% Get behaviour periods

base_frg_hz, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
base_times=n_time_index[base_ind]

"""WHAT AM I DOING HERE???"""



n_srate=1/n_time_index[1]
num_pre_entries=int(n_srate*pre_window)

target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat']

b_tuning_mean=[]
# b_tuning_std=[]

for i, b_name in enumerate(target_bs):
   
    b_loop_means=[]
    # b_loop_std=[]
    start_stop_s=hf.start_stop_array(behaviour, b_name, frame=False)
    
    for b in start_stop_s:     
        
        #exclude bs that are too short
        # if (b[1]-b[0]) < .5:
        #     continue
        
        b_ind=(n_time_index>b[0]) & (n_time_index<b[1])
        b_activity=ndata[:,b_ind]
        
        pre_times=base_times[base_times<b[0]][-num_pre_entries:]
        pre_ind=np.isin(n_time_index, pre_times)

        
#         #Testplot
#         a=n_time_index[b_ind]
#         plt.plot(a,np.ones_like(a)*i, c='w',label='b')
#         a=n_time_index[pre_ind]
#         plt.plot(a,np.ones_like(a)*i, c='skyblue',label='pre')
# hf.unique_legend()
# plt.yticks(range(len(target_bs)),target_bs)
# pf.plot_events(behaviour)
        
    
        # Average neural activity during b
        mean_b=np.mean(ndata[:,b_ind], axis=1)
        # std_b=np.std(ndata[:,b_ind], axis=1)
        
        # Average neural activity during pre
        mean_pre=np.mean(ndata[:,pre_ind], axis=1)
        std_pre=np.std(ndata[:,pre_ind], axis=1)
        
        # replace 0s with nan 
        mean_pre[mean_pre==0]=np.nan
        std_pre[std_pre==0]=np.nan
        mean_b[mean_b==0]=np.nan
        # std_b[std_b==0]=np.nan
        
        #Normalise  to pre period        
        # b_loop_std.append(std_b/std_pre)
        b_loop_means.append((mean_b -mean_pre)/std_pre)
        
    b_tuning_mean.append(b_loop_means)
    # b_tuning_std.append(b_loop_std)
    
b_tuning_mean=[np.array(l).T for l in b_tuning_mean]
# b_tuning_std=[np.array(l).T for l in b_tuning_std]

b_tot_means=np.array([hf.nanmean(l,fill=0, axis=1) for l in b_tuning_mean]).T
# b_tot_stds=np.array([hf.nanmean(l,fill=1, axis=1) for l in b_tuning_std]).T


            
#%%Make imshow plot
plt.close('all')
#MEAN
values=[-1,-.1 ,.1, 1]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)

plt.figure()
plt.imshow(b_tot_means,
           vmin=values[0],
           vmax=values[-1],
           aspect='auto',
           cmap=cmap)
pf.remove_axes()
plt.xticks(np.arange(len(target_bs)),target_bs)
# pf.region_ticks(n_region_index)
plt.title('neural tuning to behaviour')
cbar=plt.colorbar()
cbar.set_label('mean zscore')
plt.axvline(2.5, c='k')
plt.axvline(3.5, c='k',ls='--')
plt.axvline(6.5, ls='--',c='k')
plt.axvline(7.5, c='k')


# # STD
# # values=[0, .9,1.1, 5]
# cmap=pf.make_cmap(['teal', 'w', 'w', 'brown'], values)

# plt.figure()
# plt.imshow(b_tot_stds,
#            vmin=values[0],
#            vmax=values[-1],
#            aspect='auto',
#            cmap=cmap)
# pf.remove_axes()
# plt.xticks(np.arange(len(target_bs)),target_bs)
# pf.region_ticks(n_region_index)
# plt.title('neural std during behaviour')
# cbar=plt.colorbar()
# # cbar.set_label('std of zscore')
# plt.axvline(2.5, c='k')
# plt.axvline(3.5, c='k',ls='--')
# plt.axvline(6.5, ls='--',c='k')
# plt.axvline(7.5, c='k')  
    


print('WHAT CHANGES THIS PLOT A LOT IS TO NOT NORMALISE TO std')

print("""
      right now I replace all behaviour/ control periods without
      spikes with nans. This might bias the result quite a bit!!
      """)
      


#%% get 



print("""
      Problems here
      -------------
      -I feel like it doesn't make so much sense to look at the change with respect
      to the period right before, since all the behaviours overlap so much,
      and a neuron might be affected by a combination of attack, pursuit for example
      
      -Copilot suggested to do a logistic regression analysis instead, trying 
      to predict neural activity based on behaviour
      """)