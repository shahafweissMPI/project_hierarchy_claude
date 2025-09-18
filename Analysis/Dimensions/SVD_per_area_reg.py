"""
Created by Tom Kern
Last modified 04.08.2024

Using SVD + regression to understand how much neurons in each area respond to the 
instinctive behvaiours that we marked

For each area:
- Compute SVD on neural data
- Predict each SVD factor with all the behaviour information from boris (binary
     indices of when behaviour happens)
    --> Fig 2 shows how well each factor is predicted by instinctive behaviours
- to know which behaviorus contribute how much to the prediction of each factor,
    we can have a look at the beta weights from the regression. By multiplying
    beta weights with prediction performance (r2), we have a measure of how much
    each behaviour is reflected in each behaviour
    --> fig1 shows this for the 4 (num_factors) best predicted factors 

NOtes
- considerable downsampling necessary
- 'eating' has a major influence on this

"""



import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore


session='231213_0'
plt.style.use('dark_background')
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','escape','pullback']
SVD_analysis=True
           
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

resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, 1)






#%%RUN REGRESSION
# plt.close('all')
target_areas=[['DpWh'],['LPAG'],['VLPAG']]
target_factors=range(4)

for anames in target_areas:
    
    
    
    #GEt X
    X=np.zeros((len(resampled_t), len(target_bs)))
    
    
    for i, b_name in enumerate(target_bs):
        if b_name =='velocity':
            vel=hf.interp(velocity, frame_index_s[:-dropped], resampled_t)
            X[:,-1]=zscore(vel)
            continue
            
        start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
        for b in start_stop:
            b_ind=(resampled_t>(b[0])) & (resampled_t<b[1])
            X[:,i]+=b_ind
    if np.sum(X[:,:-1]>1)!=0:
        raise ValueError('check your calculation of X')

#%SVD
    region_ind=np.isin(n_region_index, anames)
    n=zscore(resampled_n[region_ind,:], axis=1)
    U, s, VT = np.linalg.svd(n.T[:,:20], full_matrices=False)
    
    
    # regression
    Y=U[:,:100]
    perf_r2, bs=hf.OLS_regression(X,Y,nfolds=5, normalise=False)
    
    # scale weights to 1 and then scale by r2   
    avg_weight=np.mean(np.array(bs), axis=0)
    scaled_weight=avg_weight/np.sum(np.abs(avg_weight), axis=0)
    weighted_weight=scaled_weight*perf_r2
    
    
#plot
    best_f=np.where(perf_r2==np.max(perf_r2))[0][0]

    #fACTOR WEIGHTING
    
    fig, axs=plt.subplots(1,len(target_factors), figsize=(20,10))
    fig.suptitle(anames)
    for i, ax in zip(target_factors, axs.flatten()):
        ax.set_title(f'Factor: {i+1}\nr2: {np.round(perf_r2[i],3)}')
        if i == best_f:
            ax.bar(range(len(weighted_weight)),weighted_weight[:,i], color='lightsteelblue')
        else:
            ax.bar(range(len(weighted_weight)),weighted_weight[:,i], color='lightslategray')
        ax.set_xticks(range(len(target_bs)), target_bs, rotation=45)
        ymin, ymax=ax.get_ylim()
        ax.set_ylim(min(ymin,-.1), max(ymax,.1))
    pf.remove_axes(axs)
    
    # perf
    plt.figure()
    plt.title(anames)
    pf.logplot(perf_r2)
    plt.xlabel('factors')
    plt.ylabel('r2')
    pf.remove_axes()
    
    # #raw factor (the one with highest r2)
    # plt.figure()
    # pf.plot_events(behaviour)
    # plt.xlabel('time (s)')
    # plt.ylabel('Factor activity')
    # plt.title(f'{anames}\nfactor {best_f+1}\nr2: {np.round(perf_r2[best_f],3)}')
    
    # plt.plot(resampled_t, U[:,best_f], c='w')


