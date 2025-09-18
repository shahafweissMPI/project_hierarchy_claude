"""
Created by Tom Kern
Last modified on 04/08/24

Computes responsiveness of each neuron to each behaviour, based on shahafs number (s_num) 
-firing change: For each neuron and each behaviour, looks in what proportion of trials of the 
    behaviour the neurons changes its firing above/below threshold as == P+/P-

-threshold: 
    -Takes n random timepoints in baseline period
    -calculates change in firing before and after timepoint
    -std of these changes is threshold of firing

-s_num: P+ - P-, is caclulated per neuron and behaviour

-Permutation: recaluclates s_num x times, each time with newly shuffled neural data

-significance: if the actual s_num is not really found when you shuffle the neural 
    data, (i.e. in a very low or very high percentile), then we think it is 
    likely that the s_num is not just due to random fluctuations
    
-reference period: 
    -escape behaviours: Before the loom (unless there are hunting behaviours 
                                         before the loom, then before hunting)
    - hunting behaviours: before hunting behaviour, no matter what came before
    - switches/ pullbacks: before hunting behaviour
    
- Results are saved, and can be summarised across sessions with 
    total_tuning_per_area.py and venn3


Important
-When wanting to compare output from different sessions later, it is important 
    that target_bs stays the same across differnet sesions (at least in the current
                                                            version of the code)

"""
from IPython import embed as stop
import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from numpy.random import choice
from time import time
from joblib import Parallel, delayed

# Set parameters
#sessions= ['231215_2','240212']
sessions= ['240524']
n_perms=1000 # How often permute (should be at least 1000 times for stable results, more is better)
target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat', 'baseline'])
hunting_bs= ['approach','pursuit','attack','eat'] # Which behaviours are considered hunting? Important for 
resolution=.01 # In case you want to run things quick, otherwise put to None
sig_threshold=.05 #Significace threshold. Has to be larger then 1/n_perms

pre_window=1.5 # How much time before the behaviour should serve as reference for neuron's firing rate?
plt.style.use('seaborn-v0_8-talk')

savepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\plotdata\test"

all_behaviors = np.array([...])  
for session in sessions:
    #Load data
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

    
    array1 = np.array([x for x in all_behaviors if isinstance(x, str)])

    # Ensure that array2 does not contain any ellipsis or other non-string objects
    array2 = np.array([x for x in target_bs if isinstance(x, str)])
    
    # Find the unique values of the intersection
    all_behaviors = np.union1d(array1, array2)
    all_behaviors=np.intersect1d(all_behaviors,all_behaviors)   

    #%% PRECOMPUTE SOME THINGS    
    if session=='240522':
        velocity=velocity[0:len(frame_index_s)]
    
    frame_index_s=frame_index_s[:len(velocity)]

    
    
    # In case you want to run things quick: downsample data
    if resolution is not None:
        ndata, n_time_index=hf.resample_ndata(ndata, n_time_index, resolution)
    bintime=n_time_index[1] # What is the width of one bin?
    
    #get baseline period
    mean_frg_hz , base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:len(velocity)])
    
    # Get baseline sample for comparisons with behaviours
    base_start_stop_s=hf.rand_base_periods (base_ind, 
                        n_time_index, 
                        num_base_samples=10,  # how many baseline samples should be taken?
                        base_period_length=2) #how long should each baseline period be?
    
    
    
    
    # Get looms, in case there are any
    try:
        looms=hf.start_stop_array(behaviour, 'loom')
    except IndexError:
        print(' no looms found, continuing without')
        looms=None
    
    
    #Get index of when hunting behaviours are happening (for reference periods for pullbacks and switches )
    if np.isin(behaviour['behaviours'], hunting_bs).sum()!=0:
        hunt_ind=np.zeros_like(n_time_index)
        start_stop=[]
        
        # go through each behaviour that is defined as hunting behaviour
        for i, b_name in enumerate(hunting_bs):
            
            #If it exists, get start_stop array
            try:
                start_stop.append(hf.start_stop_array(behaviour, b_name, frame=False))
            except IndexError:
                continue      
        
        #Stack all the start_stops on top of each other and sort them
        start_stop=np.vstack(start_stop)
        start_stop=start_stop[np.argsort(start_stop[:,0])]
        
        #Exclude overlapping periods
        final_startstop=[]
        last_stop=0
        for i, b in enumerate(start_stop):
            
            if b[0]>(last_stop+pre_window):
                final_startstop.append(b)
            
            # Concatenate periods that are closer together than pre_window
            elif (b[0]<=(last_stop+pre_window)) & (b[1]>= (last_stop+pre_window)):
                final_startstop[-1][1]=b[1]        
            
            last_stop=final_startstop[-1][1]
    
        #Get times where hunting happens
        for b in final_startstop:
            hunt_ind+=(n_time_index>(b[0])) & (n_time_index<b[1])
        if sum(hunt_ind>1)!=0:
            raise ValueError('hunt ind computation wrong')
        
        hunt_times=n_time_index[hunt_ind.astype(bool)]
    else:
        hunt_times=None
    
    #%% S_NUM CALCULATION FUNCTIONS    
    def get_cutoff(n=1000):
        """
        Cutoff for determining what is considered a 'large change'
        -Takes n random timepoints in baseline period
        -calculates change in firing before and after timepoint
        -std of these changes is threshold of firing
        """
        
        # get n samples 
        base_start_stop_s=hf.rand_base_periods (base_ind, 
                            n_time_index, 
                            num_base_samples=n,  # how many baseline samples should be taken?
                            base_period_length=1,
                            overlapping=True) #how long should each baseline period be?
        
        # Calculate firing change 
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
        
        return cutoff_spread, cutoff_mean
    
    
    def shahafs_number_array(shuffle=False):
        
        #preallocate to speed up computaion!!
        shahafs_nums=np.zeros((len(ndata), len(target_bs))) 
     
        #Only for permutations: shuffle ndata
        if shuffle:
            index=np.random.permutation(ndata.shape[1])
            n=ndata[:, index]
        else:
            n=ndata 
            
        # calculate s_num for each behaviour
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
           
            
            
            # Get index for BEFORE and DURING the behaviour
            b_b_min_pre=np.zeros((len(n), len(start_stop_s)))
            for ib, b in enumerate(start_stop_s):  
                
            #Make boolean index for reference period BEFORE the behvaiour
            
                #In case of escape behaviour, take reference from before loom         
                if (b_name in ['escape','startle','freeze']) and (looms is not None):
                    loom = looms[(looms>(b[0]-8)) & (looms<b[0])][-1] 
                    before_loom=behaviour['behaviours'][(behaviour['frames_s']>loom-3) & (
                                                         behaviour['frames_s']<loom+.01)]
                    
                    # In case of escapes after hunting, take reference from befor hunting
                    if np.isin(hunting_bs, before_loom).sum() != 0:
                        pre_ind=np.where(hunt_times<b[0])[0][-int(pre_window/bintime):]
                    else:
                        pre_ind=np.where((n_time_index>(loom-pre_window)) & (n_time_index<loom))[0]                                   
                
                #In case of swicthes/ pullbacks, take reference from before hunting
                elif b_name in ['pullback','switch'] :                     
                    pre_ind=np.where(hunt_times<b[0])[0][-int(pre_window/bintime):]                  
                    
                # Otherwise, just take the period right before the behaviour
                else: 
                    try:
                        pre_ind=np.where((n_time_index>(b[0]-pre_window)) & (n_time_index<b[0]))[0]
                    except:
                        pre_ind = np.where((n_time_index > (b - pre_window)) & (n_time_index < b))[0]                                                         
                    
            # get index of DURING behvaioir 
                try:
                    b_ind=np.where((n_time_index>(b[0])) & (n_time_index<b[1]))[0]
                except:
                                       
 
                    b_array = np.array([b])  # As a numpy array
                    if isinstance(b_array, (list, np.ndarray)) and len(b_array) == 1:
                        print(f'single timepoint event for {b_name} at {b}')

#                        continue
                        start_value = pre_ind[-1] + 1
                        end_value = start_value + int(pre_window / 2 / bintime)
                        b_ind = np.arange(start_value, end_value + 1)
                        print(f'taking half the pre_window as duration')

                    
                if np.sum(b_ind)==0:
                    continue
    
    
    
    
            # Get firing change
                pre_fir=np.sum(n[:, pre_ind], axis=1)/ (len(pre_ind)*bintime)
                b_fir=np.sum(n[:, b_ind], axis=1)/ (len(b_ind)*bintime)
                b_b_min_pre[:,ib] = b_fir-pre_fir
    
            #get P+/ p-
            print(f"{i_bname}, {b_name}")

            p_plus=np.sum(b_b_min_pre>cutoff_std[:,None], axis=1)/b_b_min_pre.shape[1]
            p_minus=np.sum(b_b_min_pre<(-cutoff_std[:,None]), axis=1)/b_b_min_pre.shape[1]
            
           
            
            shahafs_nums[:,i_bname]=p_plus-p_minus
            
            #Save in big matrix for all the behaviours
            all_b_min_pre.append(b_b_min_pre)
        # shahafs_nums=np.array(shahafs_nums).T
        
        return shahafs_nums, all_b_min_pre
    
    
    
    
    
    #%% Run on data
    cutoff_std, cutoff_mean=get_cutoff(1000)
    
    s, b_min_pre=shahafs_number_array()
    
    
    
       
    
    #%% Run for permutations
    
    start_time=time()
    
    double_unpack = Parallel(n_jobs=5)(delayed(shahafs_number_array)(True) for i in range(n_perms))
    perm_s, perm_b_min_pre = zip(*double_unpack)
    perm_s=np.transpose(perm_s, (1,2,0)) 
    
    
    end_time=time()
    print(f'permutation took {hf.convert_s(end_time-start_time)}\n')
    
    
    
    
    #%% Get p-values from permutations
    
    # How many of the permuted s_num values are more/less/equal to the actual one?
    sorted_perm=np.sort(perm_s, axis=2)
    more_than = (sorted_perm < s[:,:, None]).sum(axis=2)
    equal = (sorted_perm == s[:,:, None]).sum(axis=2)
    less_than=(sorted_perm > s[:,:, None]).sum(axis=2)
    
    # In which percentile of the permuted dataset is the real s_num?
    ranks = more_than + (equal/2).astype(int).astype(float)
    ranks[(more_than==less_than) & (more_than==0)]=np.nan
    percentiles = (ranks / n_perms) *100
    
    #Adjust to minimum resolution
    percentiles[percentiles==0]=1/n_perms *100
    percentiles[percentiles==100]=100-1/n_perms*100
    
    # Convert to p-values
    p=percentiles.copy() /100
    p[p>.5]=1-p[p>.5]
    
    
    
    
 
    #%% Get significant positive and negative changes
    if sig_threshold < 1/n_perms:
        raise ValueError("""
                         sig_threshold smaller than p-value resolution. 
                         Either increase threshold or number of permutations
                         """)
    change=np.zeros_like(percentiles)
    change[percentiles>(100-(sig_threshold*100))]=1
    change[percentiles<(sig_threshold*100)]=-1
    change[np.isnan(percentiles)]=np.nan

    import IPython
    IPython.embed()   
    
    if np.sum(change)==None:
        print('change is nan')
        stop()
    
    
    #%%Plot tuning matrix
    
    cbar=pf.plot_tuning(change,
                        target_bs,
                        n_region_index, 
                        cmap='PuOr_r', 
                        vmin=-1.5, 
                        vmax=1.5,
                        lines=False,
                        area_labels=True)
    
    cbar.set_label('BINARY change value')
    
    #expected false positive per column
    f_p=len(ndata)*sig_threshold 
    
    plt.title(f'{session}\nsignificant at p<{sig_threshold}\n{int(np.round(f_p))} false positives expected per column')
    plt.suptitle(f'pre_window={pre_window}s')
    
    
    
    #%% Save results
    
    savedict={'region_change': change,
              'regions': n_region_index,
              'target_bs': target_bs}
    
    np.save(rf"{savepath}\s_num_perm_{session}.npy",
            savedict)

hf.endsound()


# change=P.item()['region_change']
# n_region_index=P.item()['regions']
# target_bs=P.item()['target_bs']