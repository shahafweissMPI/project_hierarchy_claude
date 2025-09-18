import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf
from scipy.stats import zscore

#%% Set Things
# animal='afm16605'
session='231213_0'

#Paths
paths=pp.get_paths(session)
spike_source='ironclust'

filelength_s=5000


#%% load neural data
def load_IRC(spikes_path, t1_s, tend_s, quality_path ):
    """
    loads clustered data from ironclust Only returns units marked as 'single!
    It also cuts the data, resmaples it to s
    
    Parameters
    ----------
    spikes_path: path to csv with spikes, channels, clusters columns
    t1_s: when does the video start, relative to recording start (output from cut_rawData2vframes)
    tend_s: when does the video end, relative to recording start (output from cut_rawData2vframes)
    quality_path: path to the _quality.csv that is output from ironclust
    
    
    Returns
    -------
    good_spike_times: When do spikes happen? (shape: n_spikes)
    
    good_spike_clusters: Whch cluster do they belong to?? (shape: n_spikes)
    good_clusters: Which clusters are good? (shape: n_clusters)
    good_clusterchannels: On which channels are these clusters? (shape: n_clusters)

    """

    
    #Assign variables
    spike_file=pd.read_csv(spikes_path, header=None).to_numpy()
    spike_times=spike_file[:,0]
    spike_clusters=spike_file[:,1]
    spike_channel=spike_file[:,2]
    
    
    cluster_info=pd.read_csv(quality_path)
    cluster_KSlabel=cluster_info['note'].to_numpy()
    cluster_id=cluster_info['unit_id'].to_numpy()
    cluster_channels=cluster_info['center_site'].to_numpy()

    
    # Exclude spikes after the end of video 
    include_ind=(spike_times<tend_s) & (spike_times>t1_s)
    
    spike_times=spike_times[include_ind]
    spike_channel=spike_channel[include_ind]
    spike_clusters=spike_clusters[include_ind]
    
    # give video start time 0
    spike_times -= t1_s


    
    #get good clusters/spikes
    good_ind=cluster_KSlabel=='single'
    good_clusters=cluster_id[good_ind]
    good_cluster_channels=cluster_channels[good_ind]
    
    spike_mask = np.isin(spike_clusters, good_clusters)
    good_spike_times=spike_times[spike_mask]        
    good_spike_clusters=spike_clusters[spike_mask]
    
    
    if np.nanmin(good_spike_clusters)<1:
        print('you included wrong clusters in spike_times')

    
    return good_spike_times, good_spike_clusters, good_clusters, good_cluster_channels


(spike_times, 
 spike_clusters, 
 clusters, 
 cluster_channels)= load_IRC(paths['sorting_spikes'], 0, filelength_s, paths['sorting_quality'])
    
    





#sort by depth, or at least channel 
sort_ind=np.flip(np.argsort(cluster_channels))
cluster_channels=cluster_channels[sort_ind]
clusters=clusters[sort_ind]



#Make neurons*time matrix
res=.01
n_by_t, time_index, cluster_index=pp.neurons_by_time(spike_times, spike_clusters, clusters, bin_size=res)

# exclude  neurons (low firing/ too small ISI)
n_by_t, cluster_index, cluster_channels=pp.exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels)


plt.figure()
plt.imshow(n_by_t, aspect='auto', vmax=1)
plt.xlabel('time (s)')
plt.ylabel('neurons')

plt.figure()
plt.imshow(zscore(n_by_t, axis=1), aspect='auto', vmin=-2, vmax=2)
plt.colorbar()
plt.title('zscored')
plt.xlabel('time (s)')
plt.ylabel('neurons')