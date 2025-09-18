import numpy as np
import preprocessFunctions as pp
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf

# animal='afm16605'
sessionlist=['240212']

for session in sessionlist:
        
    split_sessions=False
    
    lfp=False
    
    
    #Paths
    paths=pp.get_paths(session)
    spike_source='ironclust'
    
    
    #%%Get nidq data and cut it to sessionstart
    
    (frame_index_s, 
     t1_s, 
     tend_s, 
     vframerate, 
     cut_rawData, 
     nidq_frame_index, 
     nidq_meta)= pp.cut_rawData2vframes(paths['nidq'])
    
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
                                    gain_correction=True)
    
    #sanity checks
    csv_loom_s=pp.convert2s(paths['csv_timestamp'], paths['csv_loom'])
    error=np.abs(csv_loom_s-diode_s_index)
    if np.nanmax(error)>.5:
        
        # plot diode signal, in case of misalignment
        tot_s=cut_rawData.shape[1]/float(nidq_meta['niSampRate'])
        x=np.linspace(0,tot_s,len(corrected_diode))
        plt.figure()
        plt.plot(x, corrected_diode)
        plt.plot(csv_loom_s, np.ones_like(csv_loom_s)*(diode_th-.5), '.', label='csv looms')
        plt.plot(diode_s_index, np.ones_like(diode_s_index)*(diode_th+.5), '.', label='diode looms')
        plt.xlabel('time (s)')
        plt.axhline(diode_th, label='threshold', c='salmon')
        plt.legend(loc='upper right')
        
        raise ValueError('csv timestamps dont agree with nidq stamps')
    
    
    
    
   
    
    #%% get boris data
    
    behaviours, boris_frames, start_stop, modifier=pp.load_boris(paths['boris_labels'])
    boris_frames_s=boris_frames/vframerate
    
    #sanity checks
    boris_loomtimes=boris_frames_s[behaviours=='loom']
    # if np.nanmax(diode_s_index-boris_loomtimes)>.02:
    #     raise ValueError('You did a mistake when labelling looms in boris')
        
    
    if np.sum(boris_frames=='NA')!= 0:
        raise ValueError ('conflicting annotations in Boris, Go over that again')
    
    # if sum(behaviours=='turn') != sum(behaviours=='escape')/2:
    #     raise ValueError('make sure that each escape starts with one turn')
    
    
    
    #%% get velocity
    
    velocity, locations, node_names=pp.extract_sleap(paths['mouse_tracking'], 'f_back', vframerate)
    
    
    #sanity checks
    if np.nanmax(velocity)>130:
        raise ValueError ('velocity too high, check this!')
    w,h=pp.video_width_height(paths['video'])
    
    if (w!=1280) or (h!=1024):
        raise ValueError('Video has different format than expected. This F***s up your cm/s values')
    
    if not len(velocity)==frames_in_video:
        raise ValueError('something is wrong with the number of frames in the sleap file')
    
    
    #%%SAVE
    
   
    
    behaviour=pd.DataFrame({
        'behaviours' : behaviours,
        'start_stop' : start_stop,
        'frames_s': boris_frames_s,
        'frames': boris_frames
        })
    
    #Sanity check
    escapes=hf.start_stop_array(behaviour, 'escape')
    if np.nanmin(np.diff(escapes[:,0]))< (50*15):
        raise ValueError('there are two escapes which should be merged')
    #Sanity check over
    
    tracking={'velocity': velocity[:], 
                        'locations': locations, 
                        'node_names': node_names,
                        'frame_index_s': frame_index_s}
   
    
    savepath=paths['preprocessed']
    

    behaviour.to_csv(fr'{savepath}\behaviour.csv', index=False)
    np.save(fr'{savepath}\tracking.npy', tracking)
    
    print(f'DONE preprocessing session {session}')
    
    
    
