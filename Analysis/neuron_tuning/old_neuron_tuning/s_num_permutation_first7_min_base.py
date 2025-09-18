"""

firing during each behaviour is compared to 
avg in first 7 minutes of recording
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from numpy.random import choice
from tqdm import tqdm
import concurrent.futures
from time import time
import functools 
from joblib import Parallel, delayed


session='231213_0'
n_perms=1000
plt.style.use('dark_background')

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
n_srate=len(n_time_index)/n_time_index[-1]


target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline'])


################ get ind for random baseline periods ##########################
res=.1
ndata, n_time_index= hf.resample_ndata(ndata, n_time_index, res)


#get paseline samples
_, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:-3])

base_start_stop_s=hf.rand_base_periods (base_ind, 
                    n_time_index, 
                    num_base_samples=10,  # how many baseline samples should be taken?
                    base_period_length=2) #how long should each baseline period be?



############################Define function#####################################


def shahafs_number_array(ndata, target_bs, behaviour, n_time_index, base_start_stop_s,base_ind, pre_window=5, shift=None):
    
    nstd=np.std(ndata[:,base_ind], axis=1)
    
    #Shift ndata if called by permutations
    if shift is not None:
        ndata=hf.shift_data(ndata, shift, column_direction=True)
    
    all_b_min_pre=[]
    shahafs_nums=[]
    for b_name in target_bs:
        
        #get baseline
        if b_name=='baseline':
            start_stop_s=base_start_stop_s.copy()
        
        else:
            # Get behaviour periods            
            start_stop_s=hf.start_stop_array(behaviour, b_name, frame=False)
        
        
        # Calculate shahafs number
        b_b_min_pre=[]
        for b in start_stop_s:  
            
        #Make boolean index for when behaviour happens
            pre_ind=(n_time_index>0) & (n_time_index<7*60)
            b_ind=(n_time_index>(b[0])) & (n_time_index<b[1])
            if np.sum(b_ind)==0:
                continue
    
            
            # get b minus b 
            pre_fir=np.mean(ndata[:, pre_ind], axis=1)
            b_fir=np.mean(ndata[:, b_ind], axis=1)
            b_b_min_pre.append(b_fir-pre_fir)
        b_b_min_pre=np.array(b_b_min_pre).T
        
        #get P+/ p-
        p_plus=np.sum(b_b_min_pre>nstd[:,None], axis=1)/b_b_min_pre.shape[1]
        p_minus=np.sum(b_b_min_pre<(-nstd[:,None]), axis=1)/b_b_min_pre.shape[1]
        
        shahafs_nums.append(p_plus-p_minus)
        all_b_min_pre.append(b_b_min_pre)
    shahafs_nums=np.array(shahafs_nums).T
    
    return shahafs_nums, all_b_min_pre

#get data
s, b_min_pre=shahafs_number_array(ndata, target_bs, behaviour, n_time_index,base_start_stop_s,base_ind, pre_window=5)

values=[-1, -.1,.1,  1]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)

cbar= pf.plot_tuning(s, 
               target_bs, 
               n_region_index,
               cmap=cmap, 
               vmin=values[0], 
               vmax=values[-1])
cbar.set_label('shahfs number')

plt.title(f'{session}\nshahafs number\nresolution: {res}s')


#%% Run permutations
################################Get random shift values##########################
max_shift=int((45   *60) /res) #min-->frames
min_shift=int((10   *60)/res) #min-->frames
# this is so large, so that the eating period is shifted as well
if (max_shift* res)> (n_time_index[-1]/2):
    raise ValueError ('shift is larger than session is long')

# create matrix with randomly distributed shifts
shift_values=np.concatenate(( 
    np.arange(min_shift,max_shift), 
    np.arange(min_shift,max_shift)*-1))

random_shifts=choice(shift_values, 
                               n_perms, 
                               replace=False)


################# PARALLElised permutation

start_time=time()
partial=functools.partial(shahafs_number_array,
                          ndata,
                          target_bs, 
                          behaviour, 
                          n_time_index, 
                          base_start_stop_s, 
                          base_ind,
                          pre_window=5)

double_unpack = Parallel(n_jobs=15)(delayed(partial)(shift=shift) for shift in random_shifts)
perm_s, perm_b_min_pre = zip(*double_unpack)
perm_s=np.transpose(perm_s, (1,2,0))


end_time=time()
print(f'permutation took {hf.convert_s(end_time-start_time)}')


#%Get percentile of each value in s

sorted_perm=np.sort(perm_s, axis=2)
more_than = (sorted_perm < s[:,:, None]).sum(axis=2)
equal = (sorted_perm == s[:,:, None]).sum(axis=2)

ranks = more_than + (equal/2).astype(int)
percentiles = (ranks / n_perms) *100

#Adjust to minimum resolution
percentiles[percentiles==0]=1/n_perms *100
percentiles[percentiles==100]=100-1/n_perms*100

# Convert to p
p=percentiles.copy() /100
p[p>.5]=1-p[p>.5]



#%Plot



c=['darkslategray','w','w']
values=[0,.05,.1]
cmap=pf.make_cmap(c, values)
cbar=pf.plot_tuning(p, target_bs,n_region_index, cmap=cmap, vmin=np.min(p), vmax=values[-1])
cbar.set_label('p-value')
cbar.set_ticks((np.min(p),.01,.05,.1))
plt.title(f'num permutations: {n_perms}')


#Show whether significant values change positively or negatatively
sig_threshold=.002
change=np.zeros_like(percentiles)
change[percentiles>(100-(sig_threshold*100))]=1
change[percentiles<(sig_threshold*100)]=-1

cbar=pf.plot_tuning(change, target_bs,n_region_index, cmap='coolwarm', vmin=-1, vmax=1)
cbar.set_label('BINARY change value')

#expected false positive per column
f_p=len(ndata)*sig_threshold

plt.title(f'significant at p<{sig_threshold}\n{int(np.round(f_p))} false positives expected per column')
plt.suptitle('take first 7 min as baseline')






