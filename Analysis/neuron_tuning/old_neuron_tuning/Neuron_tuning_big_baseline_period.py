import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore


session='231213_0'
resolution=.1 #s


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
 frame_index_s] = hf.load_preprocessed(session)



#%% Get behaviour periods
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat']

#Downsample n
ndata, n_time_index= hf. resample_ndata(ndata, n_time_index, resolution)


#zscore n
mean_frg_Hz, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:-3])
base_mean=np.mean(ndata[:, base_ind], axis=1)
base_std=np.std(ndata[:, base_ind], axis=1)
z_ndata= (ndata - base_mean[:, None])/ base_std[:, None]

print('is it correct to work with the binary data here, instead of frg rate???')


b_tuning=[]
b_std=[]
for b_name in target_bs:
   
    
    start_stops=hf.start_stop_array(behaviour, b_name, frame=False)
    
    
    #Make boolean index for when behaviour happens
    b_ind=np.zeros(ndata.shape[1])
    for stst in start_stops:        
        b_ind+=(n_time_index>stst[0]) & (n_time_index<stst[1])
    if sum(b_ind>1)>0:
        raise ValueError('sth is wrong with the computation of your b_ind')
    b_ind=b_ind.astype(bool)
    
    # Average neural activity during that period
    mean_n_for_b=np.mean(z_ndata[:,b_ind], axis=1)
    std_n_for_b=np.std(z_ndata[:,b_ind], axis=1)

    #normalise to avg firing rate
    b_std.append(std_n_for_b)
    b_tuning.append(mean_n_for_b)

b_tuning=np.array(b_tuning).T
b_std=np.array(b_std).T


#%%Make imshow plot



values=[-5, -.1,.1,  5]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)

cbar= pf.plot_tuning(b_tuning, 
               target_bs, 
               n_region_index,
               cmap=cmap, 
               vmin=values[0], 
               vmax=values[-1])
cbar.set_label('zscore')
plt.title(f'{session}\nbaseline comparison\n resolution: {resolution}s')
print("""
      Neural tuning
      -------------
      -zscore each neurons activity (mean=0, std=1)
      -get periods where a certain behaviour is happening
      -calculate mean/ std for those periods (from the zscored data)
      - 'tuning' of a neuron is difference to 0, 1 for mean, std, respectively
      
      """)

    
print("""
      This plot might have problem with duration of trials:
          -slow escapes have more weight, since they cover more time
          -behaviours that are really short, e.g. startle/ switch might not be 
          represented well
      Additionally, is it correct to use the binary neural data??
      """)
    
    
    
