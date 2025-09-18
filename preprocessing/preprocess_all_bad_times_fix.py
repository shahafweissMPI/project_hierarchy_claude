#%reset -sf# Clear memory in spyder
import matplotlib
#%matplotlib inline
import IPython
#%autoreload 0
import ast
import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('default')
plt.ion()
import helperFunctions as hf
import math
from pathlib import Path
import warnings
#animal='afm17365'; sessions=['241211']
animal = 'afm16924';sessions =['240522']
#240523_hard_coded
#240523_2sec_1overlap

# ['240523_0','240524','240525','240526','240527','240529']#['240522','240523_0','240524','240525','240526','240527','240529']  # can be a list if you have multiple sessions
#animal='afm17365';sessions=['241211','241203_01']
csvfile=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv"
spike_source='ironclust'
lfp=False     #extract LFP?
check_LFP_for_zeros=True

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
            csv_path.loc[animal, 'arena_center_xy'] =  check_and_convert_variable([ds_movement.centre_Px.data[0], ds_movement.centre_Px.data[1]],'str')
            csv_path.loc[animal, 'Shelter_xy'] = csv_path.loc[animal, 'Shelter_xy'].apply(lambda x: shelterpoint[:])
            csv_path.loc[animal,'arena_width_arena_height'] = check_and_convert_variable([ds_movement.arena_width.data.item(), ds_movement.arena_height.data.item()],'str')
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
            'arena_center_xy': float,
            'arena_width_arena_height':float,
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
        if type(paths['Shelter_xy'])==str and paths['Shelter_xy'] is not None:
            string = paths['Shelter_xy']
            list_of_tuples = ast.literal_eval(string)               
            paths['Shelter_xy']=tuple(list_of_tuples)
        
       # IPython.embed()
        # Apply the conversion function to the 'time_start_stop' column
        time_start_stop_floats = paths['time_start_stop']
               
        value=paths["diode_threshold"]
        
        
        #make sure downstream formats are kept
        paths['arena_center_xy']=check_and_convert_variable(paths['arena_center_xy'])
        paths['Cm2Pixel_xy']=check_and_convert_variable(paths['Cm2Pixel_xy'])
        
        
        
        
        
        #%% Get nidq data and cut it to sessionstart
        # main camera
        
    
        firstS=0
        lastS=0
        if len(paths['time_start_stop'])==2:
            if Path(paths['nidq']).parts[-2]==Path(paths['ap']).parts[-3]:            
                firstS=paths['time_start_stop'][0]
                lastS=paths['time_start_stop'][1]
            else:
                #user_input = input(f"\n\n csv file contains time_start_stop for this session but a non concatanated nidq file! \n\n to use the time_start_stop values for video time as well, press 'yes'. to ignore and use the local nidq times press 'no':\n  ")
                print(f"")
                user_input="2"
                if user_input.lower() == "1":
                    firstS=paths['time_start_stop'][0]
                    lastS=paths['time_start_stop'][1]
                    
                elif user_input.lower() == "2":
                    firstS=0
                    lastS=0
                   
            
        if paths['video'] is  not None  and  (not (paths['video']))==False: 
            
            (frame_index_s, 
             t1_s, 
             tend_s, 
             vframerate, 
             cut_rawData, 
             nidq_frame_index, 
             nidq_meta)= pp.cut_rawData2vframes(paths,int(paths['framechannel']),firstS=firstS,lastS=lastS)    
            
            
            
            
            
            velocity, all_locations, node_names, ds_movement, Cm2Pixel, distance2shelter, shelterpoint=pp.extract_sleap(paths,vframerate=vframerate,node = 'b_back',view='top')
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
            bottom_velocity, bottom_all_locations, bottom_node_names, bottom_ds_movement, bottom_Cm2Pixel, bottom_distance2shelter, bottom_shelterpoint=pp.extract_sleap(paths,vframerate=vframerate,node = 'b_back',view='bottom')
    
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
        
        
        plt.close('all')
        #TO DO: move this to a function and check also for bottom camera and MINI2P
        # sanity checks
        frames_in_video=pp.count_frames(paths['video'])
        if frames_in_video!= len(frame_index_s):
            print('\nATTENTION!!! VIDEO AND NIDQ MISALIGNED\n\n')
            print(f'{frames_in_video-len(frame_index_s)} frames difference!!\n\n')
            
        unique_diffs=pp.unique_float(np.diff(frame_index_s))
        if np.abs((np.nanmax(unique_diffs)-.02)>.005):
            pass
            #plt.hist(unique_diffs,21)
            #pass
            
            
            #raise ValueError('check the diffs of your frame index, something is off there')
            
        if np.abs(vframerate-50.02)>.11:
            raise ValueError (f'video framerate looks suspicious (  {vframerate} ), check that')
        
        
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
        except Exception:
             print(f'\n\n\n No csv_timestamp or csv_loom file found or problem reading it. check the .csv  \n\n\n')
             print(f"{paths['csv_loom']}")
             
             
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
            print(' \n\n\n No boris file found or problem reading it. check the .csv \n\n\n')
            import traceback
            traceback.print_exc()
            raise ValueError(f"An error occurred: {e}")                                    
                                                 
        if len(csv_loom_s) > 0:   
        
            #make an index for csv indices that correspond to flashes
            # check file is not empty
            file_stats=Path.stat(Path(paths['csv_loom']))
            if file_stats.st_size!=0:
                csv_indices_to_remove = np.where((pd.read_csv((paths['csv_loom']), header=None) == 'Keypad7') | (pd.read_csv(paths['csv_loom'], header=None) == 'Keypad8'))[0] #remove looms that are flashes
                close_csv_indices_to_remove = np.where(np.diff(csv_loom_s) < 0.3)[0] + 1 #remove csv indices very close to each other
            
            
            #remove csv indices either through code check or manual input in the paths excel file
            csv_indices_to_remove_paths=check_and_convert_variable(paths.CSV_indeces_to_remove,int)
            if 'csv_indices_to_remove' in globals() or csv_indices_to_remove_paths:
                csv = np.hstack((csv_indices_to_remove_paths, csv_indices_to_remove, close_csv_indices_to_remove)).astype(int)
                csv_loom_s = np.delete(csv_loom_s, csv)
                print(f"removed csv indeces {csv}")
                
            
            
            #make an index for diode indices that do not correspond to bonsai input(csv_indices)
            if csv_loom_s.size>0:
                nidq_indices_to_remove = []
                for i in range(len(csv_loom_s)):
                    s=np.where((csv_loom_s[i]-diode_s_index>-0.02) & (csv_loom_s[i]-diode_s_index<0.8))[0]
                    nidq_indices_to_remove.append(s)
           
            else:
                nidq_indices_to_remove = np.arange(len(diode_s_index)) 
        
            diode_s_index=diode_s_index[np.unique(np.hstack(nidq_indices_to_remove))]    
                
                
            #remove nidq indices either through code check or manual input in the paths excel file
            # nidq_indices_to_remove_paths=check_and_convert_variable(paths.diode_indeces_to_remove,int)
            # if nidq_indices_to_remove or len(nidq_indices_to_remove_paths)>0:
            #     nidq = np.hstack((nidq_indices_to_remove, nidq_indices_to_remove_paths)).astype(int)
            #     diode_s_index = np.delete(diode_s_index, nidq)
            #     print(f"removed diode indeces {np.unique(nidq)}")
                
            
            #diode_s_index = add_and_sort(diode_s_index, paths.diode_times_to_add) # check and add any  manually noted events
            #csv_loom_s = add_and_sort(csv_loom_s, paths.boris_times_to_add) # check and add any  manually noted events
            
            if len(csv_loom_s) != len(diode_s_index):
                #import user_input as user_input
                
                user_input = input(f"Number of looms in nidq doesnt match number of looms in csv file. Check Values, do you want to proceed only with csv recorded looms? (yes/no) \n ")
                if user_input.lower() == "yes":
                    diode_s_index = csv_loom_s
                elif user_input.lower() == "no":
                    
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
                        
            
                        
                        
                        raise ValueError("Number of looms in nidq doesnt match number of looms in csv file. Check Values")
                        
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
                else:
                    print("Invalid response. Please answer 'yes' or 'no'.")
                    raise ValueError("Invalid response received. Aborting process.")
            #Remove excess indices after you have double-checked values
            nidq_indices_to_remove = [] #add indices to remove
            diode_s_index = np.delete(diode_s_index, nidq_indices_to_remove)
            
            delay=np.abs(csv_loom_s-diode_s_index)     
            target_delay=.55
            if delay.size>0 and np.nanmax(delay)>target_delay:
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
                
        if paths['sorting_format'] is not None:
           spike_source= paths['sorting_format']
        
        
        
        if spike_source=='phy' or spike_source== 'kilosort':
            
            (spike_times, 
              spike_clusters, 
              clusters, 
              cluster_channels,ks_dict)= pp.load_Kilosort4(paths['sorting_spikes'], paths['ap'],split_t1_s, split_tend_s)    
            
              #cluster_channels)= pp.load_phy(paths['sorting_directory'], paths['ap'],split_t1_s, split_tend_s)    
              
        elif spike_source=='ironclust':      
            
            (spike_times, 
             spike_clusters, 
             clusters, 
             cluster_channels)= pp.load_IRC(paths['sorting_spikes'], split_t1_s, split_tend_s, paths['sorting_quality'])
            
            pd_neural_data=pp.load_IRC_in_pd(paths['sorting_quality'], paths['sorting_spikes'], split_t1_s, split_tend_s)
            
        
        
        #sort by depth, or at least channel 
        #sort_ind=np.flip(np.argsort(cluster_channels))
        #cluster_channels=cluster_channels[sort_ind]
        sort_ind = np.argsort(cluster_channels)
        clusters= [clusters[i] for i in sort_ind]
        cluster_channels=cluster_channels[sort_ind]
        
        
        
        #sanity checks
        vid_length=tend_s-t1_s
        rec_duration=split_tend_s-split_t1_s
        #if int(vid_length)!=int(max(spike_times)):
        #    print(f'{int(vid_length)=},{int(max(spike_times))=}')
        #    raise ValueError('check session length, something is off there')
        #check if spikeglx time and video time align in start. and that recording isn't longer than the video
        # if video ends before recording: remove excess spikes
        
        #if recording ended before video: fine
       # if (np.abs(spike_times[0]-t1_s)>.01) or (np.abs(spike_times[-1]-tend_s)<0):
       #     raise ValueError('spike times misaligned to frames')
        if vid_length-rec_duration>.01:        
            warnings.warn(f'Check alignment of nidq with neural data. video is longer than recording duration by {vid_length-rec_duration} seconds')
            
        
        if not np.array_equal(pp.unique_float(spike_clusters), np.sort(clusters)):
            #raise ValueError('something wrong with spike_clusters or clusters')
            warnings.warn('something wrong with spike_clusters or clusters', UserWarning)
        
            
        
        #Make neurons*time matrix
        res=.01
        n_by_t, time_index, cluster_index,n_spike_times=pp.neurons_by_time(spike_times, spike_clusters, clusters, bin_size=res)
        
        if not np.array_equal(clusters, cluster_index):
            raise ValueError ('some calculation is wrong probably')
        
        # exclude  neurons (low firing/ ISI violations)
        #%% quality control for units
        plt.close('all')
        if spike_source=='ironclust':
            print('loading ironclust data with spikeinterface')
            units_id_to_remove,analyzer,sorting,recording,df_irc=pp.load_IRC_in_SPI(paths['preprocessed'],paths['sorting_quality'], paths['sorting_spikes'], t1_s, tend_s,spikeglx_folder=Path(paths['ap']).parent,overwrite=True,analyzer_mode='binary')
            print('excluding units...')
            n_by_t, n_spike_times, cluster_index, cluster_channels,removed_clusters,df_irc=pp.exclude_neurons(t1_s,tend_s,n_by_t, spike_times, n_spike_times, spike_clusters, cluster_index, cluster_channels,units_id_to_remove=units_id_to_remove,df_irc=df_irc)
            
           #%% check for zeros and saturation in raw data 
        print(f"checking for zero values in data")
        bad_times=None
        if check_LFP_for_zeros==True:
         try:
            bad_times = pp.check_for_bad_times(animal,session)
         except:
            print ('check_for_bad_times failed...skiping')
        
        #%% get channel brain region
        print('aligning histology to recording sites')
        
        if spike_source=='ironclust': 
            channelregions, channelnums=pp.get_probe_tracking(paths['ap'], paths['probe_tracking'],paths['last_channel_manually'], df_irc)
        elif spike_source=='phy' or spike_source=='kilosort':
            import pandas as pd
            #df_irc = pd.DataFrame(list(ks_dict.items()), columns=['Key', 'Value'])
            channelregions, channelnums=pp.get_probe_tracking(paths['ap'], paths['probe_tracking'],paths['last_channel_manually'], df_ks=None)
            probe = ks_dict['ops']['probe']
            sites=ks_dict['ops']['probe']['chanMap']
            channels = np.arange(0,len(sites),1,)
            #channels_for_clusters = cluster_channels
            channel_map = dict(zip(np.array(clusters), np.array(sites)))
            channels_for_clusters = [channel_map[unit] for unit in cluster_index]
            cluster_regions=channelregions[channels_for_clusters] #this is an index for which region each neuron in the neuron*time matrix belongs to
        if spike_source=='ironclust':
            Channels=np.array(df_irc['viSite2Chan'])
            sites = np.array(df_irc['max_amp_site'])
            unit_ID=np.array(df_irc['unit_id'])
            
            channel_map = dict(zip(df_irc['unit_id'], df_irc['viSite2Chan']))
        
            # For each cluster (unit) in your cluster_index list, look up the channel:
            channels_for_clusters = [channel_map[unit] for unit in np.array(cluster_index)]
            
            cluster_regions=channelregions[channels_for_clusters] #this is an index for which region each neuron in the neuron*time matrix belongs to
            # Save region information in df
            pd_regions=channelregions[pd_neural_data['center_site']]
            pd_neural_data=pd_neural_data.assign(region=pd_regions)
        
            #print("Channel numbers for each cluster in cluster_index:")
            #print(channels_for_clusters)
        else:
            pass
    #        raise ValueError('channel site/num for non IRONCLUST not implemented yet')
        
        
        if channelnums[-1] < max(cluster_channels):#sanity check
           print('there are clusters outside the brain. Maybe alignement to atlas isnt great?')
        
       
        
        #%% get boris data
        
        #behaviours, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
        #boris_frames_s=boris_frames/vframerate
        
        #sanity checks
        #boris_loomtimes=boris_frames_s[behaviours=='loom']
        boris_loomtimes=csv_loom_s
        if boris_loomtimes.size==0:
            print('no looms annotated in borris')
        elif np.nanmax(diode_s_index-boris_loomtimes)>.02:
            raise ValueError(f'np.nanmax(diode_s_index - boris_loomtimes) > 0.02 -> You made a mistake when labelling looms in boris: {np.nanmax(diode_s_index - boris_loomtimes)}')
           
        if np.sum(boris_frames=='NA')!= 0:
            raise ValueError ('conflicting annotations in Boris, Go over that again')
        # if sum(behaviours=='turn') != sum(behaviours=='escape')/2:
        #     raise ValueError('make sure that each escape starts with one turn')
        
        
        
        #sanity checks
        if not len(velocity)==frames_in_video:
            raise ValueError('something is wrong with the number of frames in the sleap file')
            
        #%% Remove the zero times from n_by_t, time_index, n_spike_times, boris_frames_s
        
        behaviours_f, start_stop_f, boris_frames_s_f, boris_frames_f, behavioural_category_f = pp.label_bad_behaviours(behaviours, boris_frames_s, boris_frames, behavioural_category, bad_times)
        
        #%%SAVE
        print('saving...')
        
        if spike_source=='phy' or spike_source=='kilosort':
            avg_waveforms = ks_dict['templates']    
            df_irc = pd.DataFrame(list(ks_dict.items()), columns=['Key', 'Value'])
            
            
        np_neural_data = {
            'n_by_t': np.float64(n_by_t),
            'time_index': np.float64(time_index),
            'cluster_index': np.int64(cluster_index),
            'region_index': cluster_regions,
            'spike_source': spike_source,
            'cluster_channels': np.int64(cluster_channels),
            'n_spike_times': n_spike_times,
            #'avg_waveforms':avg_waveforms,
            'df_irc': df_irc,#to do : substract t0, remove units
            'neural_data_start_stop':paths['time_start_stop'],
            'bad_times':bad_times              
        }
        
        
        if not diode_s_index:
            diode_s_index=None
        behaviour=pd.DataFrame({
            'behaviours' : (behaviours_f),
            'behavioural_category' : (behavioural_category_f),
            'start_stop' : start_stop_f,
            'frames_s': np.float32(boris_frames_s_f),
            'frames': np.int64(boris_frames_f),    
            'video_start_s': np.float32(t1_s),
            'video_end_s': np.float32(tend_s),
            'diode_loom_s': np.float32(diode_s_index)
            })
        
        #Sanity check
        B_list=list(behaviour['behaviours'])
        found_escape = any(item == "escape" for item in B_list)
        if found_escape==True:
            
            
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
                print('there are two escapes which maybe should be merged')
        
        #Sanity check over
        distance2shelter=distance2shelter.to_numpy()
        ds_movement['position_cm']=ds_movement['position_cm'].squeeze()
        #ds_movement.to_netcdf(path=Path(paths['preprocessed']).joinpath(f'{animal}_{session}movement_data.h5'),format='NETCDF4')
      #  df_movement=ds_movement.to_dataframe()
        
        if bottom_distance2shelter is not None:
            bottom_distance2shelter=bottom_distance2shelter.to_numpy()
        
        tracking=               {'velocity': velocity,
                                'distance_to_shelter': distance2shelter,
                                'Cm2Pixel':Cm2Pixel,
                                'locations': all_locations, 
                                'node_names': node_names,
                                'frame_index_s': frame_index_s,
                                'bottom_distance_to_shelter': bottom_distance2shelter,
                                'bottom_locations': bottom_all_locations, 
                                'bottom_node_names': bottom_node_names,
                                'bottom_frame_index_s': bottom_video_frame_index_s
                               # 'df_movement': df_movement
                                }
        
        
        
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
        np.save(str(Path.joinpath(savepath,'np_neural_data.npy')),np_neural_data)
        if spike_source=='ironclust':
            pd_neural_data.to_csv(Path.joinpath(savepath,'pd_neural_data.csv'), index=False)
        behaviour.to_csv(Path.joinpath(savepath,'behaviour.csv'), index=False)
        np.save(str(Path.joinpath(savepath,'tracking.npy')),tracking)
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
        plt.close('all')
    
    
    
