import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf
import math
from pathlib import Path
import warnings
animal='afm16924'
sessions=['240522'] #['240522','240523_0','240524','240525','240526','240527','240529']

csvfile=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv"
spike_source='ironclust'
lfp=False     #extract LFP?

for session in sessions:
    print(f"working on session {session}")

    #patch for a bad recording that was concatanted 
    if session in ['231215_2']:
        split_sessions=2 #False, if no split; Otherwise should be which part of the split you want, numbering starting from 0
        split_values=[0, 866.6939557 , 2381.22186649, 5056.96703951] #where should the sessions be split??, in s
        #This gives session starts, the last value is end of file
    else:
        split_sessions=False
    

    
    # get meta data from csv
    paths=pp.get_paths(session=session,animal=animal)
    
    def check_and_convert_variable(x, format_type=None):
        if type(value)==format_type:
            return 
        def convert_to_format(value, format_type):
            if type(value)==format_type:
                return value        
            if type(x)== str and  format_type == float:
                return list(map(np.float64, value.replace(',', ' ').split()))
            elif type(x)== str and format_type == int:
                return list(map(lambda v: np.int16(np.float64(v)), value.replace(',', ' ').split()))
            elif format_type == 'str':
                return ' '.join(map(str, value))
            else:
                return value
    
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, str):  # x is a string
            if format_type:
                x = convert_to_format(x, format_type)
            elif '.' in x:
                x = list(map(float, x.replace(',', ' ').split()))
            else:
                x = list(map(int, x.replace(',', ' ').split()))
        elif isinstance(x, int):
            if np.isnan(x):  # x is NaN
                x = []  # Do nothing if x is NaN
            else:  # x is an integer
                x = [x]
        elif isinstance(x, list):  # x is a list
            if format_type == 'str':
                x = convert_to_format(x, format_type)
            else:
                return x
        elif x is None:  # x is None
            x = []
        else:  # x is of an unknown type
            x = []
    
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
    
    
    #%% save Cm2Pixel in paths.csv
    def save_csv(csvfile):
        csv_path=pd.read_csv(csvfile, index_col=0, header=[0])
       
        csv_path.loc[animal, 'Cm2Pixel_xy'] =  check_and_convert_variable(Cm2Pixel,'str')
        csv_path.loc[animal, 'Shelter_xy'] = csv_path.loc[animal, 'Shelter_xy'].apply(lambda x: shelterpoint[:])
        if bottom_Cm2Pixel is not None:
            csv_path.loc[animal, 'bottom_Cm2Pixel_xy'] = check_and_convert_variable(bottom_Cm2Pixel,'str')
            csv_path.loc[animal, 'bottom_Shelter_xy'] = csv_path.loc[animal, 'bottom_Shelter_xy'].apply(lambda x: shelterpoint[:])
        #if paths['MINI2P_channel'] is not None and paths['MINI2P_channel']>-1:         
        
        try:
            csv_path.to_csv(csvfile)
        except Exception as e:
            raise ValueError ('cannot write to CSV file check it is not open somewhere')
    
    
    
    def is_string_nan(value):
        try:
            # Try to convert the string to a float
            number_value = float(value)
            # Check if the converted value is NaN
            return math.isnan(number_value)
        except ValueError:
            # If conversion fails, it's not a number, hence not NaN
            return False
    def reformat_paths(paths, field_types):
        """
        Reformats path strings based on field types, handling commas, spaces, and bracketed lists.
        Returns the reformatted paths as a pandas Series.
    
        Args:
            paths (dict): Dictionary of paths.
            field_types (dict): Dictionary of field types.
    
        Returns:
            pandas.Series: Reformatted paths as a pandas Series.
        """
    
        reformatted_paths_dict = {}
        for field, path_string in paths.items():
            
            field_type = field_types.get(field)
    
            if field_type is None:
                reformatted_paths_dict[field] = path_string
                continue
            
            if isinstance(path_string, str):
                if path_string.startswith('[') and path_string.endswith(']'): # handling bracketed strings
                    try:
                        path_string = path_string[1:-1]  # Remove brackets
                        delimiter = ',' if ',' in path_string else ' '
                        values_str = path_string.strip().split(delimiter)
                        values = [field_type(v.strip()) for v in values_str if v.strip()]
                        reformatted_paths_dict[field] = values
                    except (ValueError, SyntaxError): # Added SyntaxError for cases like "[1, 2, 3]"
                        print(f"Warning: Could not convert bracketed string '{path_string}' to {field_type} for field '{field}'. Keeping as string.")
                        reformatted_paths_dict[field] = path_string
                elif field_type in (float, int):
                    try:
                        delimiter = ',' if ',' in path_string else ' '
                        values_str = path_string.strip().split(delimiter)
                        values = []
                        for v_str in values_str:
                            try:
                                values.append(field_type(v_str.strip())) #Handles correct type conversion
                            except ValueError: #Handles conversion errors for individual values gracefully
                                print(f"Warning: Could not convert '{v_str}' to {field_type} for field '{field}'. Skipping this value.")
                        reformatted_paths_dict[field] = values
                    except ValueError:
                        print(f"Warning: Could not convert '{path_string}' to {field_type} for field '{field}'. Keeping as string.")
                        reformatted_paths_dict[field] = path_string
                elif field_type == 'str':
                    reformatted_paths_dict[field] = path_string
                else:
                    reformatted_paths_dict[field] = path_string #Handles other types the same as string.
            else:
                reformatted_paths_dict[field] = path_string
                
            
                if isinstance(path_string,float) and field_type==int and (not np.isnan(path_string)):                
                        reformatted_paths_dict[field] = paths[field].astype(int).tolist()
                if isinstance(path_string,int) and field_type==float and (not np.isnan(path_string)):
                    reformatted_paths_dict[field]=paths[field].astype(float).tolist()
    
                
           
    
        return pd.Series(reformatted_paths_dict)
    
    ###
    
    def correct_types():
            
        # paths['Cm2Pixel_xy']=check_and_convert_variable(paths['Cm2Pixel_xy'],float)
        # paths['video']=check_and_convert_variable(paths['video'],str)
        # paths['bottom_video']=check_and_convert_variable(paths['bottom_video'],str)
        # paths['MINI2P_channel']=check_and_convert_variable(paths['MINI2P_channel'],int)
        
        paths=pp.get_paths(session=session,animal=animal)
        field_types = {
        "Mouse_ID": str,
        "session": str,
        "time_start_stop": float,
        "csv_timestamp": str,
        "csv_loom": str,
        "nidq": str,
        "lf": str,
        "ap": str,
        "sorting_spikes": str,
        "sorting_quality": str,
        "boris_labels": str,
        "probe_tracking": str,
        "last_channel_manually": int,
        "preprocessed": str,
        "video": 'str',
        "framechannel": int,
        "Cm2Pixel_xy": float,
        "Shelter_xy": float,
        "bottom_video": str,
        "bottom_framechannel": int,
        "bottom_Cm2Pixel_xy": float,
        "bottom_Shelter_xy": float,
        "diode_threshold": float,
        "MINI2P_channel": int,
        "diodechannel": int,
        "frame_loss": int,
        "loom": str,
        "cricket": str,
        "pups": str,
        "mouse_tracking": str,
        "cricket_tracking": str,
        "pup_tracking": str,
        "CSV_indeces_to_remove": int,
        "diode_indeces_to_remove": int,
        "boris_times_to_add": float,
        "diode_times_to_add": float,
        "Sorting": str,
        "notes": str}
        reformatted_paths = reformat_paths(paths, field_types)
        return reformatted_paths
    
    paths=correct_types()
    # Apply the conversion function to the 'time_start_stop' column
    time_start_stop_floats = paths['time_start_stop']
           
    value=paths["diode_threshold"]
    
    
    #%% Get nidq data and cut it to sessionstart
    # main camera
    if paths['video'] is  not None  and  (not (paths['video']))==False: 
        
        (frame_index_s, 
         t1_s, 
         tend_s, 
         vframerate, 
         cut_rawData, 
         nidq_frame_index, 
         nidq_meta)= pp.cut_rawData2vframes(paths,int(paths['framechannel']))    
        velocity, all_locations, node_names, ds_movement, Cm2Pixel, distance2shelter, shelterpoint=pp.extract_sleap(session, animal, mouse_tracking_path = paths['mouse_tracking'], 
                                                                                                            camera_video_path = paths['video'], vframerate=vframerate, 
                                                                                                            Cm2Pixel_from_paths=paths['Cm2Pixel_xy'], 
                                                                                                            Shelter_xy_from_paths=paths['Shelter_xy'],
                                                                                                            node = 'b_back')
    else: 
        velocity = all_locations = node_names = ds_movement = Cm2Pixel = distance2shelter = video_frame_index_s = shelterpoint = None   
        
    #bottom camera
    if paths['bottom_video'] is  not None and  (not paths['bottom_video'])==False and not is_string_nan(paths['bottom_video']) : 
        (bottom_video_frame_index_s, 
         bottom_video_t1_s, 
         bottom_video_tend_s, 
         bottom_video_vframerate, 
         bottom_video_cut_rawData, 
         bottom_video_nidq_frame_index,      
         bottom_video_nidq_meta)= pp.cut_rawData2vframes(paths,int(paths['bottom_framechannel']))    
        bottom_velocity, bottom_all_locations, bottom_node_names, bottom_ds_movement, bottom_Cm2Pixel, bottom_distance2shelter, bottom_shelterpoint=pp.extract_sleap(session, animal, mouse_tracking_path = paths['mouse_tracking'], 
                                                                                                        camera_video_path = paths['video'], vframerate=vframerate, 
                                                                                                        Cm2Pixel_from_paths=[paths['Cm2Pixel_x'], paths['Cm2Pixel_y']], 
                                                                                                        Shelter_xy_from_paths=paths['Shelter_xy'],
                                                                                                        node = 'b_back')
    else: 
        bottom_velocity = bottom_all_locations = bottom_node_names = bottom_ds_movement = bottom_Cm2Pixel = bottom_distance2shelter = bottom_video_frame_index_s = bottom_shelterpoint = None    
        
    #mini2P camera
    if paths['MINI2P_channel'] is not None and (not (paths['MINI2P_channel']))==False and not is_string_nan(paths['MINI2P_channel']):
        (mini2P_frame_index_s, 
         mini2P_t1_s, 
         mini2P_tend_s, 
         mini2P_vframerate, 
         mini2P_cut_rawData, 
         mini2P_nidq_frame_index, 
         mini2P_nidq_meta)= pp.cut_rawData2vframes(paths,int(paths['MINI2P_channel']))
    
    
    
    #TO DO: move this to a function and check also for bottom camera and MINI2P
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
    
    
    save_csv(csvfile)
           
    
    
    #%% Get diode signal
    diode_channel_num = int(paths['diodechannel'])
    diode_th=float(paths['diode_threshold'])
    (diode_s_index, 
     diode_frame_index, 
     diode_index,
     corrected_diode)=pp.get_loom_times(cut_rawData, 
                                    nidq_meta, 
                                    nidq_frame_index,
                                    diode_channel_num,
                                    threshold=diode_th,
                                    detrend=True, 
                                    gain_correction=True,
                                    min_delay=.3,
                                    )
    csv_loom_s=[]
    try:                                        
        csv_loom_s=pp.convert2s(paths['csv_timestamp'], paths['csv_loom'])
    except Exception as e:
         print(f' \n\n\n No csv_timestamp or csv_loom file found or problem reading it. check the .csv \n\n\n')
         
         
    #sanity checks        
    try:
        #if not np.isnan(paths['boris_labels']):
        file_stats=Path.stat(Path(paths['boris_labels']))
        if file_stats.st_size!=0:
         behaviours, behavioural_category, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
         boris_frames_s=boris_frames/vframerate
        
            #sanity checks
         boris_loomtimes=boris_frames_s[behaviours=='loom']
         if len(csv_loom_s)==0:
             csv_loom_s=boris_loomtimes
    except Exception as e:
        print(f' \n\n\n No boris file found or problem reading it. check the .csv \n\n\n')
        
        import traceback
        traceback.print_exc()   
        raise ValueError(f"An error occurred: {e}")
       
        
    
    #make an index for csv indices that correspond to flashes
    file_stats=Path.stat(Path(paths['csv_loom']))
    if file_stats.st_size!=0:    # check file is not empty
     csv_indices_to_remove = np.where((pd.read_csv((paths['csv_loom']), header=None) == 'Keypad7') | (pd.read_csv(paths['csv_loom'], header=None) == 'Keypad8'))[0]
    
    
    #remove csv indices either through code check or manual input in the paths excel file
    csv_indices_to_remove_paths=check_and_convert_variable(paths.CSV_indeces_to_remove,int)
    if csv_indices_to_remove.size>0 or csv_indices_to_remove_paths.size>0:
        csv = np.hstack((csv_indices_to_remove_paths, csv_indices_to_remove)).astype(int)
        csv_loom_s = np.delete(csv_loom_s, csv)
        print(f"removed csv indeces {csv}")
            
    
    
    #make an index for diode indices that do not correspond to bonsai input(csv_indices)
    nidq_indices_to_remove = []
    s_diff = 0.5 #threshold for difference between diode and csv recorded looms. default value = 0.4
    for i in range(len(diode_s_index)):
        if (abs(diode_s_index[i] - csv_loom_s) < s_diff).any() == False:
            nidq_indices_to_remove.append(i)
    
    #remove nidq indices either through code check or manual input in the paths excel file
    nidq_indices_to_remove_paths=check_and_convert_variable(paths.diode_indeces_to_remove,int)
    if nidq_indices_to_remove or nidq_indices_to_remove_paths.size>0:
        nidq = np.hstack((nidq_indices_to_remove, nidq_indices_to_remove_paths)).astype(int)
        diode_s_index = np.delete(diode_s_index, nidq)
        print(f"removed diode indeces {np.unique(nidq)}")
        
    
    diode_s_index = add_and_sort(diode_s_index, paths.diode_times_to_add) # check and add any  manually noted events
    csv_loom_s = add_and_sort(csv_loom_s, paths.boris_times_to_add) # check and add any  manually noted events
    
    
    if len(csv_loom_s) > len(diode_s_index):
        # plot diode signal, in case of misalignment
        tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
        x=np.linspace(0,tot_s,len(corrected_diode))
        plt.figure(figsize=(16,4))
        plt.plot(x, corrected_diode)
        plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
        plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
        plt.plot(boris_loomtimes, np.ones_like(boris_loomtimes)*(diode_th+.5), '.', label='boris looms')
        
        # Draw red lines between each point in diode_s_index and the corresponding value in csv_loom_s
        #this throws error when len(diode_s_index) > (csv_loom_s)
        for i in range(len(diode_s_index)):
            plt.plot([diode_s_index[i], csv_loom_s[i]], [diode_th + .5, diode_th - .5], 'r-')
        plt.xlabel('time (s)')
        plt.axhline(diode_th, label='threshold', c='salmon')
        plt.legend(loc='lower right')
        plt.show()  # Add this line to display the plot
        raise ValueError ('Number of looms in nidq doesnt match number of looms in csv file. Check Values')
    if len(csv_loom_s) < len(diode_s_index):
        # plot diode signal, in case of misalignment
        tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
        x=np.linspace(0,tot_s,len(corrected_diode))
        plt.figure(figsize=(16,4))
        plt.plot(x, corrected_diode)
        plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
        plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
        # Draw red lines between each point in diode_s_index and the corresponding value in csv_loom_s
        #this throws error when len(diode_s_index) > (csv_loom_s)
        for i in range(len(csv_loom_s)):
            plt.plot([csv_loom_s[i], diode_s_index[i]], [diode_th + .5, diode_th - .5], 'r-')
        plt.xlabel('time (s)')
        plt.axhline(diode_th, label='threshold', c='salmon')
        plt.legend(loc='lower right')
        plt.show()  # Add this line to display the plot
        raise ValueError ('Number of looms in nidq doesnt match number of looms in csv file. Check values')
                         
    #Remove excess indices after you have double-checked values
    nidq_indices_to_remove = [] #add indices to remove
    diode_s_index = np.delete(diode_s_index, nidq_indices_to_remove)
    
    delay=np.abs(csv_loom_s-diode_s_index)     
    target_delay=.5
    # if np.nanmax(delay)>target_delay:
       
        # # plot diode signal, in case of misalignment
        # tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
        # x=np.linspace(0,tot_s,len(corrected_diode))
        # plt.figure(figsize=(16,4))
        # plt.plot(x, corrected_diode)
        # plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
        # plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
        # plt.xlabel('time (s)')
        # plt.axhline(diode_th, label='threshold', c='salmon')
        # plt.legend(loc='lower right')
        # plt.show()  # Add this line to display the plot
        # raise ValueError('csv timestamps dont agree with nidq stamps')
    
    
    
    
    #%%get spiketimes
  
    if split_sessions:
        seslength=split_values[split_sessions+1] - split_values[split_sessions]
        if (seslength - float(nidq_meta['fileTimeSecs'])) > .01:
            raise ValueError ('split values dont agree with nidq session length')            
        split_t1_s=split_values[split_sessions]+t1_s        
        t_end_diff=seslength-tend_s
        split_tend_s=split_values[split_sessions+1] -t_end_diff
    else: # if split_sessions is flase take video start times from NIDQ
        if paths['time_start_stop'] is None:     
            split_t1_s=t1_s.copy()
            split_tend_s=tend_s.copy()
        else:            # take from CSV
            if len(paths['time_start_stop']) != 2: #if no start / stop rais error
                error_str=paths['time_start_stop']
                raise ValueError (f'time_start_stop should contain 2 values {error_str}')            
            split_t1_s=paths['time_start_stop'][0]+t1_s # add t1 
            if tend_s<paths['time_start_stop'][1]: # video is shorter than recording
                split_tend_s=(paths['time_start_stop'][0]+tend_s) # take only until t0 +video duration
            else:#video didn't shut down before recording ended
                split_tend_s=(paths['time_start_stop'][1]) # end withuration of recording
            
    
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
    rec_duration=split_tend_s-split_t1_s
    #if int(vid_length)!=int(max(spike_times)):
    #    print(f'{int(vid_length)=},{int(max(spike_times))=}')
    #    raise ValueError('check session length, something is off there')
    #check if spikeglx time and video time align in start. and that recording isn't longer than the video
    if (np.abs(spike_times[0]-frame_index_s[0])>.01) or (np.abs(spike_times[-1]-frame_index_s[-1])<0):
        raise ValueError('spike times misaligned to frames')
    if vid_length-rec_duration>.01:        
        warnings.warn(f'Check alignment of nidq with neural data. video is longer than recording duration by {vid_length-rec_duration} seconds')
        
    
    if not np.array_equal(pp.unique_float(spike_clusters), np.sort(clusters)):
        #raise ValueError('something wrong with spike_clusters or clusters')
        warnings.warn('something wrong with spike_clusters or clusters', UserWarning)
    
        
    
    #Make neurons*time matrix
    res=.01
    n_by_t, time_index, cluster_index=pp.neurons_by_time(spike_times, spike_clusters, clusters, bin_size=res)
    
    if not np.array_equal(clusters, cluster_index):
        raise ValueError ('some calculation is wrong probably')
    # exclude  neurons (low firing/ too small ISI)
    n_by_t, cluster_index, cluster_channels=pp.exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels)
    
    
    
    
    
    #%% get channel brain region
    print('aligning histology to recording sites')
    channelregions, channelnums=pp.get_probe_tracking(paths['ap'], paths['probe_tracking'], paths['last_channel_manually'])
    
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
        raise ValueError(f'np.nanmax(diode_s_index - boris_loomtimes) > 0.02 -> You made a mistake when labelling looms in boris: {np.nanmax(diode_s_index - boris_loomtimes)}')
       
    if np.sum(boris_frames=='NA')!= 0:
        raise ValueError ('conflicting annotations in Boris, Go over that again')
    # if sum(behaviours=='turn') != sum(behaviours=='escape')/2:
    #     raise ValueError('make sure that each escape starts with one turn')
    
    
    
    #sanity checks
    if not len(velocity)==frames_in_video:
        raise ValueError('something is wrong with the number of frames in the sleap file')
    
    
    #%%SAVE
    print('saving...')
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
        'behavioural_category' : (behavioural_category),
        'start_stop' : (start_stop),
        'frames_s': np.float32(boris_frames_s),
        'frames': np.int64(boris_frames),        
       # 'diode_loom_s': np.float32(diode_s_index)
        })
    
    #Sanity check
    escapes=[]
    escapes.append(hf.start_stop_array(behaviour, 'escape'))
    try:
        escapes.append(hf.start_stop_array(behaviour, 'gas_escape'))
    except Exception as e:
         print('')
               
    escapes=np.sort(np.vstack(escapes), axis=0)
    if escapes.size==0:
        raise ValueError('there are no escapes')
    elif np.nanmin(np.diff(escapes))< 15:
        raise ValueError('there are two escapes which should be merged')
    
    #Sanity check over
    tracking={'velocity': velocity,
                  'distance_to_shelter': distance2shelter,
                            'locations': all_locations, 
                            'node_names': node_names,
                            'frame_index_s': frame_index_s,
                            'bottom_distance_to_shelter': bottom_distance2shelter,
                            'bottom_locations': bottom_all_locations, 
                            'bottom_node_names': bottom_node_names,
                            'bottom_frame_index_s': bottom_video_frame_index_s}
    
    
    ## check no behavior after recording ends. no recording before video starts
    #t1_s
    # exclude  neurons (low firing/ too small ISI)
    # make pp.exclude_by_time_threshold 'over' or 'under'
    #n_by_t, cluster_index, cluster_channels=pp.exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels) 
    
    #make save folder
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
    #%% get LFPs   
    if lfp: #if flagged to save LFP
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
        
    print(f"completed session {session}")
    
    #import polars as pl
    #pyarrow, polars required
    