"""
Created by Tom Kern
Last modified 04.08.2024

Shows average correlation between neurons before, during, and after pullback
-Concatenates all pullback periods and make correlation matrix from that
- Concatenates all pre- and post- pullback periods and makes correlationmatrix from that
    --> Time that is taken before/ after is the average pullback duration.
        This is important, to have correlation values comparable
- The result of this is shown in fig5. fig 2-4 are averages of the different fields
    in fig5

Limitations
-periods before and after pullback are taken blindly, i.e. it is possible that 
    the 'after pullback' times already contain new hunting, or that the 'before'
    timepoints contain previous pullbacks
- resamples data to binsize of .2s to increase correlations in low-firing neurons

Extra info
-you can run this script also on other behaviors by changing target_b. I focussed
    on pullbacks because they showed the most striking change
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import matplotlib.patches as patches
import pandas as pd

# Set parameters
session='231213_0'
res=.2
target_regions=['DpWh','LPAG','VLPAG']
target_b=['pullback']
vmin=-.5
vmax=.5
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


# target_regions=np.unique(n_region_index)

#%%

resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, res)


# filter for target regions
n_ind=np.isin(n_region_index, target_regions)

n_ind[[110,116,145,161]]=False # only for cosmetic reasons


n_Hz=resampled_n[n_ind]/res

# For later, convert the start_stop format into a boolean index into nural data
def get_ind(start_stop):
        ind=np.zeros_like(resampled_t)
        for b in start_stop:
            ind+=(resampled_t>=(b[0])) & (resampled_t<=(b[1]))
        return ind.astype(bool)

#
for b_name in target_b:
    
    #get behaviour periods
    b_start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
    
    # extend period to have more data to work with 
    # if b_name=='switch':
    #     diff=np.squeeze(np.diff(b_start_stop, axis=1))
    #     b_start_stop[diff<1,0]-=.5
    #     b_start_stop[diff>1,1]+=.5
    
    # Get start_stop of periods before and after target_b
    time=np.mean(np.diff(b_start_stop, axis=1))    
    pre_start_stop=np.vstack((b_start_stop[:,0]-(time),b_start_stop[:,0])).T
    post_start_stop=np.vstack((b_start_stop[:,1],b_start_stop[:,1]+time)).T
    
    # make start_stop into boolean index
    hunt_ind=get_ind(b_start_stop)
    pre_hunt_ind=get_ind(pre_start_stop)
    post_hunt_ind=get_ind(post_start_stop)
    
    
    
    
    #Get average correlation within and between regions
    def region_corrs(matrix, regions):
        unique_regions = np.unique(regions)
        within_region_corrs = {}
        between_region_corrs = {region: {} for region in unique_regions}
    
        for region in unique_regions:
            # Get neurons in the current region
            region_neurons = matrix[regions == region]
    
            # Calculate average correlation within the region
            within_region_corr = np.corrcoef(region_neurons)
            
            # within_region_corr[within_region_corr>0]=np.nan
            
            np.fill_diagonal(within_region_corr, np.nan)
            within_region_corrs[region] = np.nanmean(within_region_corr)
    
            # Calculate average correlation with each other region
            for other_region in unique_regions:
                if other_region != region:
                    other_region_neurons = matrix[regions == other_region]
                    between_region_corr = np.corrcoef(region_neurons, other_region_neurons)[
                        :len(region_neurons), len(region_neurons):]
                    # between_region_corr[between_region_corr>0]=np.nan
                    between_region_corrs[region][other_region] = np.nanmean(between_region_corr)
    
        return within_region_corrs, between_region_corrs
    
    
    
    pre_corr=region_corrs(n_Hz[:,pre_hunt_ind], n_region_index[n_ind])
    dur_corr=region_corrs(n_Hz[:,hunt_ind], n_region_index[n_ind])
    post_corr=region_corrs(n_Hz[:,post_hunt_ind], n_region_index[n_ind])
    
    
    
    #%% Plot average correlations    
    all_bars=[]
    all_bar_names=[]
    
    phase_names=['pre','pullback','after']
    for t_corr, name in zip([pre_corr, dur_corr,post_corr],
                              phase_names):
        bars=[]
        bar_names=[]
        
        for region in np.unique(n_region_index[n_ind]):
            bars.append(t_corr[0][region])
            bar_names.append(region)
            
            for subregion in t_corr[1][region].keys():
                bars.append(t_corr[1][region][subregion])
                bar_names.append(f'{region}-{subregion}')
        
        
        all_bars.append(bars)
        all_bar_names.append(bar_names)
    all_bars=np.array(all_bars).T
    
    plt.figure()
    for i,(entry, name) in enumerate(zip(all_bars,all_bar_names[0])):
        if (i)%len(target_regions) == 0:
            plt.xticks(range(len(phase_names)),phase_names)
            plt.ylim((-.05,.23))
            plt.legend()
            pf.remove_axes()
            plt.title(b_name)
            plt.figure()
        if len(name)<6:
            c='w'
        else:
            c='grey'
        plt.plot(range(len(entry)),entry, label=name)
    plt.xticks(range(len(phase_names)),phase_names)
    plt.ylim((-.05,.23))
    plt.legend()
    pf.remove_axes()
    plt.title(b_name)




    
    #%% Make correlation matrices
    
    def make_corr_matrix(corr, vmin, vmax, n_region_index, ax=None):
        
        # get only lower half of matrix
        mask = np.triu(np.ones_like(corr)).astype(bool)
        corr[mask]=np.nan
        
        if ax is None:
            fig, ax=plt.subplots()
        cax=ax.imshow(corr, vmin=vmin,vmax=vmax)
        # cbar=ax.set_colorbar()
        # cbar.set_label('correlation')
        
        
        ticks=pf.region_ticks(n_region_index, xaxis=True, ax=ax)
        
        ticks=np.append(ticks,len(corr))
        for i in range(len(ticks)-1):
            rect = patches.Rectangle((ticks[i], ticks[i]), 
                                      ticks[i+1]-ticks[i], 
                                      ticks[i+1]-ticks[i], 
                                      linewidth=1, 
                                      edgecolor='w', 
                                      facecolor='none')
            ax.add_patch(rect)
        return cax
    
    fig, axs=plt.subplots(1,3)
    fig.suptitle(b_name)
    make_corr_matrix(np.corrcoef(n_Hz[:, pre_hunt_ind]),
                     vmin, vmax, n_region_index[n_ind], axs[0])
    axs[0].set_title('before')
    
    make_corr_matrix(np.corrcoef(n_Hz[:, hunt_ind]),
                      vmin, vmax, n_region_index[n_ind], axs[1])
    axs[1].set_title('during')
    
    
    cax=make_corr_matrix(np.corrcoef(n_Hz[:, post_hunt_ind]),
                      vmin, vmax, n_region_index[n_ind], axs[2])
    axs[2].set_title('after')
    
    fig.colorbar(cax)
pf.remove_axes(axs)
