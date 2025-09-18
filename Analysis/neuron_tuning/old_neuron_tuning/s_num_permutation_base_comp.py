import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from numpy.random import choice
from time import time
from joblib import Parallel, delayed


n_perms=5000
pre_window=.5
plt.style.use('dark_background')
all_regions=['CIC', 'DCIC', 'DpWh', 'ECIC', 'VLPAG', 'LPAG', 'DRL','PDR', 'isRt']

hunting_bs= ['approach','pursuit','attack','eat']
all_change=[]
all_region_change=[]
sessions=['231213_0','231215_2','240212']
# sessions=['231213_0']
for session in sessions:
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
    
    

    
    frame_index_s=frame_index_s[:len(velocity)]
    target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline'])
    
    
    ################ get ind for random baseline periods ##########################
    
    ndata, n_time_index=hf.resample_ndata(ndata, n_time_index, .1)
    bintime=n_time_index[1]

    #get baseline samples
    _, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:len(velocity)])
    
    base_mean=np.sum(ndata[:,base_ind], axis=1)/(np.sum(base_ind)*bintime)
    
    
    base_start_stop_s=hf.rand_base_periods (base_ind, 
                        n_time_index, 
                        num_base_samples=10,  # how many baseline samples should be taken?
                        base_period_length=2) #how long should each baseline period be?
    

   
    ############################Define function#####################################
    
    #%%
    
    def get_cutoff(n=10000):
        _, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:len(velocity)])
        
        base_start_stop_s=hf.rand_base_periods (base_ind, 
                            n_time_index, 
                            num_base_samples=n,  # how many baseline samples should be taken?
                            base_period_length=1.5) #how long should each baseline period be?
        
        
        base_diff=[]
        for b in base_start_stop_s:            
            
            b_ind=(n_time_index>(b[0])) & (n_time_index<b[1])
            if np.sum(b_ind)==0:
                continue
            
            # get b minus b     
            pre_fir=base_mean.copy()
            b_fir=np.sum(ndata[:, b_ind], axis=1)/(np.sum(b_ind)*bintime)
            base_diff.append(b_fir-pre_fir)
        cutoff_mean=np.mean(base_diff, axis=0)
        cutoff_spread=np.std(base_diff, axis=0)
        
        return cutoff_spread


    def shahafs_number_array(shuffle=False):
        
        shahafs_nums=np.zeros((len(ndata), len(target_bs))) #preallocate to speed up computaion!!
     
        #Shift ndata if called by permutations
        if shuffle:
            index=np.random.permutation(ndata.shape[1])
            n=ndata[:, index]
        else:
            n=ndata #DELETED .COPY()!!!
        all_b_min_pre=[]

        for i_bname, b_name in enumerate(target_bs):
            
            #get baseline
            if b_name=='baseline':
                start_stop_s=base_start_stop_s.copy()
            
            else:
                # Get behaviour periods   
                try:
                    start_stop_s=hf.start_stop_array(behaviour, b_name, merge_attacks=pre_window)
                except IndexError:
                    shahafs_nums[:,i_bname]=np.full(len(n), np.nan)
                    all_b_min_pre.append(np.full(len(n), np.nan))
                    continue            
           
            
                
            # Calculate shahafs number
            b_b_min_pre=np.zeros((len(n), len(start_stop_s)))
            for ib, b in enumerate(start_stop_s):  
                
            #Make boolean index for when behaviour happens
            
                #Get pre ind, without the previous behavour
                
               
                    
                    
                # get b_ind 
                b_ind=np.where((n_time_index>(b[0])) & (n_time_index<b[1]))[0]
                if np.sum(b_ind)==0:
                    continue
    

                # get b minus b 
                pre_fir=base_mean.copy()
                b_fir=np.sum(n[:, b_ind], axis=1)/ (len(b_ind)*bintime)
                b_b_min_pre[:,ib] = b_fir-pre_fir

            #get P+/ p-
            p_plus=np.sum(b_b_min_pre>cutoff[:,None], axis=1)/b_b_min_pre.shape[1]
            p_minus=np.sum(b_b_min_pre<(-cutoff[:,None]), axis=1)/b_b_min_pre.shape[1]
            
            shahafs_nums[:,i_bname]=p_plus-p_minus
            all_b_min_pre.append(b_b_min_pre)
        # shahafs_nums=np.array(shahafs_nums).T
        
        return shahafs_nums, all_b_min_pre
    
    #get data
    cutoff=get_cutoff()*2
    s, b_min_pre=shahafs_number_array()
    
   
    
    #%% Run permutations
    
 
    
    start_time=time()
    
    double_unpack = Parallel(n_jobs=20)(delayed(shahafs_number_array)(True) for i in range(n_perms))
    perm_s, perm_b_min_pre = zip(*double_unpack)
    perm_s=np.transpose(perm_s, (1,2,0))
    
    
    end_time=time()
    print(f'permutation took {hf.convert_s(end_time-start_time)}\n')
    
    
    #%%Get percentile of each value in s
    
    sorted_perm=np.sort(perm_s, axis=2)
    more_than = (sorted_perm < s[:,:, None]).sum(axis=2)
    equal = (sorted_perm == s[:,:, None]).sum(axis=2)
    less_than=(sorted_perm > s[:,:, None]).sum(axis=2)
    
    ranks = more_than + (equal/2).astype(int).astype(float)
    ranks[(more_than==less_than) & (more_than==0)]=np.nan
    percentiles = (ranks / n_perms) *100
    
    #Adjust to minimum resolution
    percentiles[percentiles==0]=1/n_perms *100
    percentiles[percentiles==100]=100-1/n_perms*100
    
    # Convert to p
    p=percentiles.copy() /100
    p[p>.5]=1-p[p>.5]
    
    
    
    #%%Plot tuning matrix
    plot_random_bullshit=False
    
    if plot_random_bullshit:
        # s values
        values=[-1, -.1,.1,  1]
        cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)
        
        cbar= pf.plot_tuning(s, 
                       target_bs, 
                       n_region_index,
                       cmap='viridis', 
                       vmin=-.75, 
                       vmax=.75)
        cbar.set_label('shahfs number')
        plt.title(f'{session}\nshahafs number\nresolution: {n_time_index[0]}s')
        
        # p-Values
        
        c=['darkslategray','w','w']
        values=[0,.05,.1]
        cmap=pf.make_cmap(c, values)
        cbar=pf.plot_tuning(p, target_bs,n_region_index, cmap=cmap, vmin=np.min(p), vmax=values[-1])
        cbar.set_label('p-value')
        cbar.set_ticks((np.min(p),.01,.05,.1))
        plt.title(f'num permutations: {n_perms}')
    
    
    # significant values
    sig_threshold=.001
    change=np.zeros_like(percentiles)
    change[percentiles>(100-(sig_threshold*100))]=1
    change[percentiles<(sig_threshold*100)]=-1
    change[np.isnan(percentiles)]=np.nan
    
    cbar=pf.plot_tuning(change, target_bs,n_region_index, cmap='PuOr_r', vmin=-1.5, vmax=1.5)
    cbar.set_label('BINARY change value')
    
    f_p=len(ndata)*sig_threshold #expected false positive per column
    
    plt.title(f'{session}\nsignificant at p<{sig_threshold}\n{int(np.round(f_p))} false positives expected per column')
    plt.suptitle('pre_window=.5s')
    
    plt.savefig(fr"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\area_tuning\tuning_per_neuron\{session}.svg")



# for quantifyng tuning
    all_change.append(change)
    region_change=[]
    for region in all_regions:
        region_change.append(change[n_region_index==region])
    all_region_change.append(region_change)

savedict={'all_region_change': all_region_change,
          'all_regions': all_regions,
          'target_bs': target_bs}

np.save(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\plotdata\s_num_perm.npy",
        savedict)

hf.endsound()

