"""
Created by Tom Kern
Last modified 04.08.2024

How well can neurons from one area predict activity of neurons in otehr areas?
-Takes a sample of 15 (nmb_n) neurons from each area 
    --> Each area needs to have same number of neurons for comparability of predictions
- extracts 4 (nmb_f) SVD factors from them 
    --> Get only the major dimensions of variablility
- For each area, use it's SVD factors to predict the SVD factors of each other area
- How well this prediction works (r2) is the measure for 'communication' between these areas
- avg firing rate for samples are  .9 +/- .3 (target_frg_rate +/- tolerance), to control for
    influence of firing rate on r2. Still, communication prediction is not very stable

Limitations
- The prediction works a lot better when the neural signal is not binary, 
    as it is in very low-firing neurons
- thus, you need to downsample the signal quite a bit (2s bins at the moment),
    and still the prediction works a lot better for high-firing neurons. 
- The final r2 between areas reflects more how many high-firing neurons are in
    the sample, than actual synchrony. This would be less of a problem if you have
    many neurons (e.g. > 50) so that you can draw a sample where firing rates 
    of neurons are matched/ controlled for
-This script is probably more useful ones you have many sessions, so that you can averages across sessions

Plots
-fig1 shows r2 of predictions
-fig2-4 show source, target, and predictd activity for one exampel area
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import matplotlib.patches as patches

session='231213_0'
plt.style.use('dark_background')

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

#Set parameters
res=2
nmb_n=15 # how many neurons should there be per region
nmb_f=4 # How many factors should be used?
target_frg_rate=.9 #mean firing rate of semi-random sample
tolerance=.3 # tolerance of deviation from target_frg_rate



resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, res)
example_a='VLPAG'



#%%Population prediction

#
n_factors, f_areas, s_factors =hf.sample_neurons(resampled_n,
                                             n_region_index,
                                             res, 
                                             nmb_n, 
                                             target_frg_rate, 
                                             tolerance, 
                                             return_factors=nmb_f)



un_plot_areas=np.unique(f_areas)
fig, axs= pf.subplots(len(un_plot_areas))


#Do regression
for ax, aname in zip(axs, un_plot_areas):
    
    f_region_ind=f_areas==aname
    
    X=n_factors[:,f_region_ind]
    Y=n_factors[:,~f_region_ind]

    perf_r2, bs=hf.OLS_regression(X,Y,nfolds=5, normalise=False)
    if aname==example_a:
        Y_OLS, B_OLS, r2=hf.regression_no_kfolds(X,Y)
    
    
    #weight r2 by s^2
    plot_areas=f_areas[~f_region_ind]
    plot_s=s_factors[~f_region_ind]
    
    mean_r2=[]

#Collect mean r2 per factor
    for  ar in  np.unique(plot_areas):
        ar_r2=perf_r2[plot_areas==ar]
        ar_s=plot_s[plot_areas==ar]
        
        weighted_r2=np.sum(ar_r2*ar_s)/np.sum(ar_s)
        mean_r2.append(weighted_r2)

#Start plotting
    ax.bar(range(len(mean_r2)),mean_r2)
   
    
    ax.set_xticks(range(len(np.unique(plot_areas))),np.unique(plot_areas))    
    ax.set_title(f'{aname} predictor', y=.9)
    ax.set_ylim((0,.4))#np.max((.5,np.max(all_points)))))
    

axs[0].set_ylabel('r2', rotation=0)
axs[3].set_ylabel('r2', rotation=0)

axs[3].set_xlabel(f'predicted areas')
axs[4].set_xlabel(f'predicted areas')
axs[5].set_xlabel(f'predicted areas')


fig.suptitle(f'each area has {Y.shape[1]/(len(un_plot_areas)-1)} factors and is predicted by {X.shape[1]} factors from another area\nneurons per area: {nmb_n}')
  
 

#%% Plot raawwww

#raw predictor
aind= f_areas==example_a

plt.figure()
plt.title(f'X\nresolution: {res}s')
plt.imshow(zscore(n_factors[:,aind].T, axis=1),
           aspect='auto',
           vmin=-3,
           vmax=3)
cbar=plt.colorbar()
cbar.set_label('zscore')
plt.xlabel('time')
pf.region_ticks(f_areas[aind])

# raw predicted (Y)
plt.figure()
plt.title(f'Y\nresolution: {res}s')
plt.imshow(zscore(n_factors[:,~aind].T, axis=1),
           aspect='auto',
           vmin=-3,
           vmax=3)
cbar=plt.colorbar()
cbar.set_label('zscore')
plt.xlabel('time')
tics=pf.region_ticks(f_areas[~aind])
for tic in tics:
    plt.axhline(tic)

# raw predicted (Yhat)
plt.figure()
plt.title(f'Y hat\nresolution: {res}s')
plt.imshow(Y_OLS.T,
           aspect='auto',
           vmin=-3,
           vmax=3)
cbar=plt.colorbar()
cbar.set_label('zscore')
plt.xlabel('time')
tics=pf.region_ticks(f_areas[~aind])
for tic in tics:
    plt.axhline(tic)







    
