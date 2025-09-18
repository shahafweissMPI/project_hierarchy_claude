import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf

animal='afm16924'
session='240524'

if session in ['231215_2']:
    split_sessions=2 #False, if no split; Otherwise should be which part of the split you want, numbering starting from 0
    split_values=[0, 866.6939557 , 2381.22186649, 5056.96703951] #where should the sessions be split??, in s
    #This gives session starts, the last value is end of file
else:
    split_sessions=False

lfp=False


#Paths
	

paths=pp.get_paths(session=session,animal=animal)
spike_source='ironclust'


#%%Get nidq data and cut it to sessionstart

(frame_index_s, 
 t1_s, 
 tend_s, 
 vframerate, 
 cut_rawData, 
 nidq_frame_index, 
 nidq_meta)= pp.cut_rawData2vframes(paths)

if paths['MINI2P_channel'] is not None and paths['MINI2P_channel']>-1:
    (mini2P_frame_index_s, 
     mini2P_t1_s, 
     mini2P_tend_s, 
     mini2P_vframerate, 
     mini2P_cut_rawData, 
     mini2P_nidq_frame_index, 
     mini2P_nidq_meta)= pp.cut_rawData2vframes(paths,int(paths['MINI2P_channel']))




# sanity checks
frames_in_video=pp.count_frames(paths['video'])
if frames_in_video!= len(frame_index_s):
    print('\nATTENTION!!! VIDEO AND NIDQ MISALIGNED\n\n')
    print(f'{frames_in_video-len(frame_index_s)} frames difference!!\n\n')
    
unique_diffs=pp.unique_float(np.diff(frame_index_s))
if np.abs((np.nanmax(unique_diffs)-.02)>.005):
    plt.hist(unique_diffs,11)
    raise ValueError('check the diffs of your frame index, something is off there')
    
if np.abs(vframerate-50.02)>.11:
    raise ValueError ('video frame rate looks suspicious, check that')


#%% Get diode signal
diode_channel_num = int(paths['diodechannel'])
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
                                min_delay=.3,
                                )


#sanity checks
#csv_loom_s=pp.convert2s(paths['csv_timestamp'], paths['csv_loom'])

try:
    #if not np.isnan(paths['boris_labels']):
    behaviours, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
    boris_frames_s=boris_frames/vframerate
    
    #sanity checks
    boris_loomtimes=boris_frames_s[behaviours=='loom']
    csv_loom_s=boris_loomtimes
except Exception as e:
    print('no boris file found. check the .csv')
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()   
    
    

def check_and_convert_variable(x):
    
    if  isinstance(x, np.ndarray):
        return x
    if isinstance(x, str): #x is a string

        if '.' in x:
            x = list(map(float, x.split(',')))
        else:            
            x = list(map(int, x.split(',')))
            
    elif isinstance(x, int):
        if np.isnan(x): #x is NaN
            x=[]            # Do nothing if x is NaN
        else:#x is a float
            x = list(map(int, str(x).split(',')))
    elif isinstance(x, list): # x is a list
        # No conversion needed for lists
        return x
    elif x is None: #x is None
        x=[]
    else: #x is of an unknown type
        x=[]

    return x

def add_and_sort(array1, array2):
    # manualy add values to array1 (diode or boris looms)
    # Check if array2 is not empty, None, or contains NaN values
    array1=check_and_convert_variable(array1)
    array2=check_and_convert_variable(array2)


    if (array2 is None or 
        len(array2)==0 or 
        np.isnan(array2).any()):
        return array1
    
    # Add values from diode_times_to_add to diode_s_index
    combined_list = np.append(array1, array2)
    
    # Sort the combined list in ascending order
    sorted_list = np.sort(combined_list)
    return sorted_list


indices_to_remove=check_and_convert_variable(paths.diode_indeces_to_remove)
if indices_to_remove:
    diode_s_index = np.delete(diode_s_index, indices_to_remove)
    print(f"removing diode indeces {indices_to_remove}")
    
indices_to_remove=check_and_convert_variable(paths.CSV_indeces_to_remove)
if indices_to_remove:
        print(f"removing csv indeces {indices_to_remove}")
        csv_loom_s = np.delete(csv_loom_s, indices_to_remove)

diode_s_index = add_and_sort(diode_s_index, paths.diode_times_to_add) # check and add any  manually noted events
csv_loom_s = add_and_sort(csv_loom_s, paths.boris_times_to_add) # check and add any  manually noted events



loom_mismatch=len(csv_loom_s)!=len(diode_s_index)
if loom_mismatch:
    # plot diode signal, in case of misalignment
    tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
    x=np.linspace(0,tot_s,len(corrected_diode))
    plt.figure(figsize=(16,4))
    plt.plot(x, corrected_diode)
    plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
    plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
    # Draw red lines between each point in diode_s_index and the corresponding value in csv_loom_s
    for i in range(len(diode_s_index)):
        plt.plot([diode_s_index[i], csv_loom_s[i]], [diode_th + .5, diode_th - .5], 'r-')
    plt.xlabel('time (s)')
    plt.axhline(diode_th, label='threshold', c='salmon')
    plt.legend(loc='lower right')
    plt.show()  # Add this line to display the plot
    raise ValueError ('Number of looms in nidq doesnt match number of looms in csv file')

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
    plt.legend(loc='lower right')
    plt.show()  # Add this line to display the plot
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


# Save region information in df
pd_regions=channelregions[pd_neural_data['center_site']]
pd_neural_data=pd_neural_data.assign(region=pd_regions)


# sanity check
if channelnums[-1] < max(cluster_channels):
    print('there are clusters outside the brain. Maybe alignement to atlas isnt great?')
    
    




#%% get boris data

#behaviours, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
#boris_frames_s=boris_frames/vframerate

#sanity checks
#boris_loomtimes=boris_frames_s[behaviours=='loom']
boris_loomtimes=csv_loom_s
if boris_loomtimes.size==0:
    raise ValueError('no looms annotated in borris')
if np.nanmax(diode_s_index-boris_loomtimes)>.02:
    raise ValueError(f'np.nanmax(diode_s_index-boris_loomtimes)>.02 -> You made a mistake when labelling looms in boris')
   
if np.sum(boris_frames=='NA')!= 0:
    raise ValueError ('conflicting annotations in Boris, Go over that again')
# if sum(behaviours=='turn') != sum(behaviours=='escape')/2:
#     raise ValueError('make sure that each escape starts with one turn')

#%% get velocity

[frames_dropped,
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
 locations, #in pixels. to get cm locations[:,:,1 or 2]*Cm2Pixel
 node_names, 
 frame_index_s,
] = hf.load_preprocessed(session=session, load_lfp=False) # 
node_ind = np.where(node_names == 'f_back')[0][0]


 

vframerate=len(frame_index_s)/max(frame_index_s)
paths=pp.get_paths(session)
##Distance to shelter
if paths['Cm2Pixel']=='nan' or paths['Shelter_xy']=='nan' or paths is None:
    print('click on shelter location in pop up plot')
    distance2shelter,loc,vector=hf.get_shelterdist(locations, node_ind )
else:
    Cm2Pixel=hf.check_and_convert_variable(paths['Cm2Pixel'])[0]
    distance2shelter=hf.check_and_convert_variable(paths['Shelter_xy'])
    distance2shelter=(distance2shelter[0],distance2shelter[1])
    loc=np.squeeze(locations[:,node_ind,:])
    loc=loc*Cm2Pixel
    
   

velocity2, locations2, node_names2=pp.extract_sleap(paths['mouse_tracking'], 
                                                 paths['video'],
                                                 'f_back',
                                                 vframerate,Cm2Pixel,locations,node_names)
  

#sanity checks
if not len(velocity)==frames_in_video:
    raise ValueError('something is wrong with the number of frames in the sleap file')


#%%SAVE
np_neural_data = {
    'n_by_t': np.float64(n_by_t),
    'time_index': np.float64(time_index),
    'cluster_index': np.int64(cluster_index),
    'region_index': (cluster_regions),
    'spike_source': (spike_source),
    'cluster_channels': np.int64(cluster_channels)
}

behaviour=pd.DataFrame({
    'behaviours' : (behaviours),
    'start_stop' : (start_stop),
    'frames_s': np.float32(boris_frames_s),
    'frames': np.int64(boris_frames)
    })

#Sanity check
escapes=hf.start_stop_array(behaviour, 'escape')
if escapes.size==0:
    raise ValueError('there are no escapes')
elif np.nanmin(np.diff(escapes[:,0]))< 15:
    raise ValueError('there are two escapes which should be merged')

#Sanity check over
tracking={'velocity': velocity, 
                    'locations': locations, 
                    'node_names': node_names,
                    'frame_index_s': frame_index_s}

## check no behavior after recording ends. no recording before video starts
#t1_s
# exclude  neurons (low firing/ too small ISI)
# make pp.exclude_by_time_threshold 'over' or 'under'
#n_by_t, cluster_index, cluster_channels=pp.exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels)





#make save folder
from pathlib import Path
import math
savepath=paths['preprocessed']
if savepath is None or savepath == "" or (isinstance(savepath, float) and math.isnan(savepath)):
    raise ValueError ('no savepath specified')
savepath=Path(savepath)
if not(savepath.is_dir()):
    savepath=Path(savepath).as_posix()
    savepath.mkdir(parents=True, exist_ok=True)

#   save to save folder
np.save(Path.joinpath(savepath,'np_neural_data.npy'),np_neural_data)
pd_neural_data.to_csv(Path.joinpath(savepath,'pd_neural_data.csv'), index=False)
behaviour.to_csv(Path.joinpath(savepath,'behaviour.csv'), index=False)
np.save(Path.joinpath(savepath,'tracking.npy'),tracking)


if lfp: #if flagged to save LFP
    #%% get LFPs   
    [lfp, 
     lfp_time, 
     lfp_framerate]=pp.load_lfp(paths['lf'],
                            t1_s, 
                            tend_s, 
                            max_channel=channelnums[-1], 
                            rshp_factor=10,
                            njobs=24)
    
    lfp_dict={'lfp': lfp,
         'lfp_time': lfp_time,
         'lfp_framerate': lfp_framerate}
    #save lfp
    np.save(Path.joinpath(savepath,'lfp.npy'), lfp_dict)

#import polars as pl
#pyarrow, polars required

