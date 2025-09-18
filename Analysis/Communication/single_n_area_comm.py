"""
Created by Tom Kern
Last modified 04.08.2024

Takes each area and predicts neural activity in each other area (same as factor_area_comm.py, but with single neurons)

How well can neurons from one area predict activity of neurons in otehr areas?
-Takes a sample of 15 (nmb_n) neurons from each area 
    --> Each area needs to have same number of neurons for comparability of predictions
- For each area, use the 15 neurons to predict the 15 neurons of each other area
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
-This script is probably  more useful ones you have many sessions,  so that you can averages across sessions


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
import scipy

session='231213_0'
plt.style.use('default')

           
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

res=2
nmb_n=15 # how many neurons should there be per region
target_frg_rate=.9 #what should be the mean firing rate for the samples of neurons from each area
tolerance=.3 # how much above or below can the population avg be?
example_a='VLPAG' #Which area should be taken as example for plotting X, Y, and Y_hat

resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, res)



#%%Population prediction

# Get sample of neurons
n_filtered, filtered_areas=hf. sample_neurons(resampled_n,
                                             n_region_index,
                                             res, 
                                             nmb_n, 
                                             target_frg_rate, 
                                             tolerance, 
                                             return_factors=None)



# Prepare figure 
fig, axs= pf.subplots(len(np.unique(filtered_areas)))



for ax, aname in zip(axs, np.unique(filtered_areas)):
        
#Do regression
    source_ind=filtered_areas==aname
    
    X=zscore(n_filtered[source_ind,:], axis=1).T
    Y=zscore(n_filtered[~source_ind,:], axis=1).T
    

    perf_r2, bs=hf.OLS_regression(X,Y,nfolds=5, normalise=False)
    
    if aname==example_a:
        Y_OLS, B_OLS, r2=hf.regression_no_kfolds(X,Y)
    
    
    
#get mean r2 per area
    plot_areas=filtered_areas[~source_ind]
    mean_r2=[]
    all_points=[]
    for  ar in  np.unique(plot_areas):
        ar_r2=perf_r2[plot_areas==ar]
        mean_r2.append(np.mean(ar_r2))
        all_points.append(ar_r2)

        
        
    #get mean firing for each neuron
    target_mean_frg=np.mean(n_filtered[~source_ind,:], axis=1)/res
    target_mean_frg=target_mean_frg.reshape((-1,nmb_n))
    
    #Make barplot
    ax.bar(range(len(mean_r2)),mean_r2, color='gray')
    for i, p in enumerate(all_points):
        ax.scatter(np.ones_like(p)*i,p, c=target_mean_frg[i], cmap='plasma', vmin=0, vmax=10, s=7)
    
    ax.set_xticks(range(len(np.unique(plot_areas))),np.unique(plot_areas))    
    ax.set_title(f'{aname} predictor', y=.9)
    ax.set_ylim((0,np.max((.4,np.max(all_points)+.05))))
    

    

axs[0].set_ylabel('r2', rotation=0)
axs[3].set_ylabel('r2', rotation=0)

axs[3].set_xlabel(f'predicted areas')
axs[4].set_xlabel(f'predicted areas')
axs[5].set_xlabel(f'predicted areas')

fig.suptitle(f'{nmb_n} neurons per area with ~{target_frg_rate}Hz avg\nresolution: {res}')

    
#%% Plot raawwww

#raw predictor
aind=filtered_areas==example_a

plt.figure()
plt.title(f'X\nresolution: {res}s')
plt.imshow(zscore(n_filtered[aind],axis=1),
           aspect='auto',
           vmin=-3,
           vmax=3)
cbar=plt.colorbar()
cbar.set_label('zscore')
plt.xlabel('time')
pf.region_ticks(filtered_areas[aind])

# raw predicted (Y)
plt.figure()
plt.title(f'Y\nresolution: {res}s')
plt.imshow(zscore(n_filtered[~aind],axis=1),
           aspect='auto',
           vmin=-3,
           vmax=3)
cbar=plt.colorbar()
cbar.set_label('zscore')
plt.xlabel('time')
tics=pf.region_ticks(filtered_areas[~aind])
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
tics=pf.region_ticks(filtered_areas[~aind])
for tic in tics:
    plt.axhline(tic)










