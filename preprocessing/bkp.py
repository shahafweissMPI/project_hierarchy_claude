import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf

animal='afm16963'
session='240526_sw'

if session in ['231215_2']:
    split_sessions=2 #False, if no split; Otherwise should be which part of the split you want, numbering starting from 0
    split_values=[0, 866.6939557 , 2381.22186649, 5056.96703951] #where should the sessions be split??, in s
    #This gives session starts, the last value is end of file
else:
    split_sessions=False

lfp=False


#Paths
	

paths=pp.get_paths(session='240526_sw',animal='afm16963')
spike_source='ironclust'


#%%Get nidq data and cut it to sessionstart

(frame_index_s, 
 t1_s, 
 tend_s, 
 vframerate, 
 cut_rawData, 
 nidq_frame_index, 
 nidq_meta)= pp.cut_rawData2vframes(paths)

# sanity checks
frames_in_video=pp.count_frames(paths['video'])
if frames_in_video!= len(frame_index_s):
    print('\nATTENTION!!! VIDEO AND NIDQ MISALIGNED\n\n')
    print(f'{frames_in_video-len(frame_index_s)} frames difference!!\n\n')
    
unique_diffs=pp.unique_float(np.diff(frame_index_s))
if (len(unique_diffs)>2) or np.abs((np.nanmax(unique_diffs)-.02)>.005):
    raise ValueError('check the diffs of your frame index, something is off there')
    
if np.abs(vframerate-50.02)>.11:
    raise ValueError ('video frame rate looks suspicious, check that')


#%% Get diode signal

diode_th=float(paths['diode_threshold'])
(diode_s_index, 
 diode_frame_index, 
 diode_index,
 corrected_diode)=pp.get_loom_times(cut_rawData, 
                                nidq_meta, 
                                nidq_frame_index,  
                                threshold=diode_th,
                                detrend=True, 
                                gain_correction=True,
                                min_delay=.3)

#sanity checks
csv_loom_s=pp.convert2s(paths['csv_timestamp'], paths['csv_loom'])

loom_mismatch=len(csv_loom_s)!=len(diode_s_index)
if loom_mismatch:
    raise ValueError ('Error: number of looms in nidq doesnt match number of looms in csv file')

delay=np.abs(csv_loom_s-diode_s_index)     
target_delay=.5
if np.nanmax(delay)>target_delay:
   
    # plot diode signal, in case of misalignment
    tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
    x=np.linspace(0,tot_s,len(corrected_diode))
    plt.figure(figsize=(16,4))
    plt.plot(x, corrected_diode)
    plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
    plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
    plt.xlabel('time (s)')
    plt.axhline(diode_th, label='threshold', c='salmon')
    plt.legend(loc='upper right')
    
    raise ValueError('csv timestamps dont agree with nidq stamps')




#%%get spiketimes
if split_sessions:
    seslength=split_values[split_sessions+1] - split_values[split_sessions]
    if (seslength - float(nidq_meta['fileTimeSecs'])) > .01:
        raise ValueError ('split values dont agree with nidq session length')
        
    split_t1_s=split_values[split_sessions]+t1_s
    
    t_end_diff=seslength-tend_s
    split_tend_s=split_values[split_sessions+1] -t_end_diff
else:
    split_t1_s=t1_s.copy()
    split_tend_s=tend_s.copy()

if spike_source=='phy':
    
    (spike_times, 
      spike_clusters, 
      clusters, 
      cluster_channels)= pp.load_phy(paths['sorting_directory'], paths['ap'],split_t1_s, split_tend_s)    
elif spike_source=='ironclust':

    (spike_times, 
     spike_clusters, 
     clusters, 
     cluster_channels)= pp.load_IRC(paths['sorting_spikes'], split_t1_s, split_tend_s, paths['sorting_quality'])
    
    pd_neural_data=pp.load_IRC_in_pd(paths['sorting_quality'], paths['sorting_spikes'], split_t1_s, split_tend_s)
    


#sort by depth, or at least channel 
sort_ind=np.flip(np.argsort(cluster_channels))
cluster_channels=cluster_channels[sort_ind]
clusters=clusters[sort_ind]


#sanity checks
vid_length=tend_s-t1_s
if int(vid_length)!=int(max(spike_times)):
    raise ValueError('check session length, something is off there')
if (np.abs(spike_times[0]-frame_index_s[0])>.01) or (np.abs(spike_times[-1]-frame_index_s[-1])>.01):
    raise ValueError('spike times misaligned to frames')
if vid_length-np.nanmax(spike_times)>.01:
    raise ValueError ('Check alignment of nidq with neural data')

if not np.array_equal(pp.unique_float(spike_clusters), np.sort(clusters)):
    raise ValueError('something wrong with spike_clusters or clusters')

#Make neurons*time matrix
res=.01
n_by_t, time_index, cluster_index=pp.neurons_by_time(spike_times, spike_clusters, clusters, bin_size=res)

if not np.array_equal(clusters, cluster_index):
    raise ValueError ('some calculation is wrong probably')
    


# exclude  neurons (low firing/ too small ISI)
n_by_t, cluster_index, cluster_channels=pp.exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels)



#%% get channel brain region

channelregions, channelnums=pp.get_probe_tracking(paths['probe_tracking'], paths['last_channel_manually'])

cluster_regions=channelregions[cluster_channels] #this is an index for which region each neuron in the neuron*time matrix belongs to




# sanity check
if channelnums[-1] < max(cluster_channels):
    print('there are clusters outside the brain. Maybe alignement to atlas isnt great?')
    
    

#%% get LFPs

if lfp:
    [lfp, 
     lfp_time, 
     lfp_framerate]=pp.load_lfp(paths['lf'],
                                t1_s, 
                                tend_s, 
                                max_channel=channelnums[-1], 
                                rshp_factor=10,
                                njobs=25)


#%% get boris data

behaviours, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
boris_frames_s=boris_frames/vframerate

#sanity checks
boris_loomtimes=boris_frames_s[behaviours=='loom']
if boris_loomtimes.size==0:
    raise ValueError('no looms annotated in borris')
if np.nanmax(diode_s_index-boris_loomtimes)>.02:
    raise ValueError('You did a mistake when labelling looms in boris')
   
if np.sum(boris_frames=='NA')!= 0:
    raise ValueError ('conflicting annotations in Boris, Go over that again')

# if sum(behaviours=='turn') != sum(behaviours=='escape')/2:
#     raise ValueError('make sure that each escape starts with one turn')



#%% get velocity

velocity, locations, node_names=pp.extract_sleap(paths['mouse_tracking'], 'f_back', vframerate)


#sanity checks
w,h=pp.video_width_height(paths['video'])

if (w!=1280) or (h!=1024):
    raise ValueError('Video has different format than expected. This F***s up your cm/s values')

if not len(velocity)==frames_in_video:
    raise ValueError('something is wrong with the number of frames in the sleap file')


#%%SAVE
np_neural_data = {
    'n_by_t': float(n_by_t),
    'time_index': float(time_index),
    'cluster_index': int(cluster_index),
    'region_index': str(cluster_regions),
    'spike_source': str(spike_source),
    'cluster_channels': int(cluster_channels)
}

behaviour=pd.DataFrame({
    'behaviours' : str(behaviours),
    'start_stop' : str(start_stop),
    'frames_s': float(boris_frames_s),
    'frames': int(boris_frames)
    })

#Sanity check
escapes=hf.start_stop_array(behaviour, 'escape')
if escapes.size==0:
    raise ValueError('there are no escapes')
if np.nanmin(np.diff(escapes[:,0]))< 15:
    raise ValueError('there are two escapes which should be merged')
#Sanity check over

tracking={'velocity': velocity, 
                    'locations': locations, 
                    'node_names': node_names,
                    'frame_index_s': frame_index_s}
if lfp:
    lfp_dict={'lfp': lfp,
         'lfp_time': lfp_time,
         'lfp_framerate': lfp_framerate}

from pathlib import Path
savepath=paths['preprocessed']
savepath=Path(savepath).as_posix()
if not(savepath.is_dir()):
    savepath.mkdir(parents=True, exist_ok=True)
    
np.save(fr'{savepath}\np_neural_data.npy', np_neural_data)
pd_neural_data.to_csv(fr'{savepath}\pd_neural_data.csv', index=False)
behaviour.to_csv(fr'{savepath}\behaviour.csv', index=False)
np.save(fr'{savepath}\tracking.npy', tracking)

if lfp:
    np.save(fr'{savepath}\lfp.npy', lfp_dict)

import polars as pl
#pyarrow, polars required

