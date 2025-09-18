"""
here take a period before start of pursuit as pre
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




beh_ind=np.ones(ndata.shape[1]).astype(int)

#Cut out periods of behviour
for b_name in behaviour['behaviours'].unique():
    b=behaviour[behaviour['behaviours']==b_name]
    
    if b['start_stop'].iloc[0]=='START':
        start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
        for start_stop_s in start_stop:
            ind=(n_time_index>(start_stop_s[0])) & (n_time_index<(start_stop_s[1]))
            beh_ind-=ind
beh_ind=(beh_ind>0).astype(int)
    




############################Define function#####################################


def shahafs_number_array(ndata, target_bs, behaviour, n_time_index, base_start_stop_s,base_ind,beh_ind, pre_window=.5, shift=None):
    
    nstd=np.std(ndata[:,base_ind], axis=1)
    
    num_pre_entries=int(pre_window/n_time_index[1])
    
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
            pre_ind=np.where(((n_time_index>(b[0])) & beh_ind))[0][-num_pre_entries:]
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
s, b_min_pre=shahafs_number_array(ndata, target_bs, behaviour, n_time_index,base_start_stop_s,base_ind,beh_ind, pre_window=1)

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
                          beh_ind,
                          pre_window=1)

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
sig_threshold=.001
change=np.zeros_like(percentiles)
change[percentiles>(100-(sig_threshold*100))]=1
change[percentiles<(sig_threshold*100)]=-1

cbar=pf.plot_tuning(change, target_bs,n_region_index, cmap='coolwarm', vmin=-1, vmax=1)
cbar.set_label('BINARY change value')

#expected false positive per column
f_p=len(ndata)*sig_threshold

plt.title(f'significant at p<{sig_threshold}\n{int(np.round(f_p))} false positives expected per column')













# #%%plot b-pre vs perm
# # plt.close('all')
# """
# its gonna be a pain in the bum to index into this fcking list.
# it is structured n_perms[behaviour][neurons*b_trials]
# """

# sig_base=np.where(p[:,-1]<.02)[0]
# nstd=np.std(ndata, axis=1)
# for n in sig_base:
#     plt.figure()
#     for i_b, b_name in enumerate(target_bs):
        
#         #perm stuff
#         pdata=[]
#         for perm in perm_b_min_pre:
#             pdata.append(perm[i_b][n,:])
#         pdata=np.hstack(pdata)
#         plt.scatter(np.ones_like(pdata)*i_b, pdata, color='gray')
        
        
#         #data
#         data=b_min_pre[i_b][n,:]
#         plt.scatter(np.ones_like(data)*i_b, data, color='r')
    
#     plt.axhline(0,c='w')
#     plt.title(f'neuron {n}')
#     plt.ylabel('b - pre')
#     plt.xticks(range(len(target_bs)),target_bs)
#     plt.axhline(nstd[n],ls='--',c='w')
#     plt.axhline(-nstd[n],ls='--',c='w')







#%%%OTHER CODE





#%%plot perm distribution



# nstd=np.std(ndata[:,base_ind], axis=1)

# n=1
# b_ind=np.where(target_bs=='attack')[0][0]
# diff=[]
# for b_values in perm_b_min_pre:
#     b=b_values[b_ind]
#     diff.append(b[n])
# diff=np.array(diff)

# p_plus=np.sum(diff>nstd[n], axis=1)/diff.shape[1]
# p_minus=np.sum(diff<-nstd[n], axis=1)/diff.shape[1]

# s_values=
# plt.figure()
# plt.hist(s_values)
# plt.axvline(s[n,b_ind])
# plt.title(f'neuron {n}\n{target_bs[b_ind]}')
# pf.remove_axes()














# #%%########################TEST SECTION#####################################################

# #significant in baseline
# from scipy.stats import zscore

# plt.figure();plt.imshow(change[:,-1,None], aspect='auto')

# sig_base=np.where(change[:,-1])[0]

# n_sig_base=ndata[sig_base]
# #%% baseline periods + neural activity
# fig, ax=plt.subplots(2,1,sharex=True)
# ax[1].imshow(zscore(n_sig_base, axis=1),
#            aspect='auto',
#            extent=[n_time_index[0],
#                    n_time_index[-1],
#                    0,
#                    len(n_sig_base)],
#            vmin=-2,
#            vmax=2)

# for [start, stop] in base_start_stop_s:
#     ax[0].axvspan(start, stop, label='random baseline events', color='darkslategray')

# #%% histograms of perms
# # plt.close('all')
# for i in sig_base:
#     plt.figure() 
#     plt.hist(perm_s[i,-1,:],bins=50) 
#     plt.axvline(s[i,-1], c='r')
#     plt.xlim((-1,1))
    
# #%% Test trial length and trial num of behaviours
# trial_lengths=[]
# trial_num=[]
# for b_name in target_bs[:-1]:
#     start_stop_s=hf.start_stop_array(behaviour, b_name, frame=False)
#     trial_num.append(len(start_stop_s))
#     trial_lengths.append(start_stop_s[:,1]-start_stop_s[:,0])
    
# plt.figure()
# plt.title('trial number')
# plt.bar(range(len(target_bs[:-1])),trial_num)
# plt.ylabel('number of trials')
# plt.xticks(range(len(target_bs[:-1])),target_bs[:-1])



# plt.figure()
# plt.title('trial length')


# for i, t in enumerate(trial_lengths):
#     plt.bar(i,np.mean(t), color='teal')
#     plt.scatter(np.ones_like(t)*i,t, c='aliceblue')
    
    
# plt.ylabel('trial lengths (s)')
# plt.xticks(range(len(target_bs[:-1])),target_bs[:-1])


