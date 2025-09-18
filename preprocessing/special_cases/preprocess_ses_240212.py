import numpy as np
import readSGLX as glx
import helperFunctions as hf
import preprocessFunctions as pp
import pandas as pd





def run_script():
    session='240212'


    #Paths
    paths=pp.get_paths(session)

    #%% GET FRAMETIMES
    
    nidq_meta = glx.readMeta(paths['nidq'])
    nidq_srate=float(nidq_meta['niSampRate'])
    # if digital:
    #     rawData=glx.ExtractDigital(rawData, firstSamp, lastSamp, dwReq, dLineList, meta)
    rawData = glx.makeMemMapRaw(paths['nidq'], nidq_meta)
    
    
    digital=True
    frame_channel_index=6
     
    
    # frame_channel=rawData[frame_channel_index].copy()
    frame_channel=np.squeeze(
        glx.ExtractDigital(
            rawData, 
            firstSamp=0, 
            lastSamp= rawData.shape[1], 
            dwReq=0, 
            dLineList=[frame_channel_index], 
            meta=nidq_meta)
        ).astype(int)
     
    
    #get frame indices
    framechannel_diff=np.hstack([0,np.diff(frame_channel)])
    rec_nidq_frame_index=np.where(framechannel_diff>0)[0]
    
    # cut rawData for getting loomtimes in next function
    t1=rec_nidq_frame_index[0]
    cut_rawData=rawData[:,t1:]
    
    # interpolate frame index from framerate
    
    step=np.mean(np.diff(rec_nidq_frame_index))
    csv_frames=pd.read_csv(paths['csv_timestamp'])
    nidq_frame_index=np.arange(0,
                                step*len(csv_frames), 
                                step)
    
    
    
    #convert to s
    frame_index_s=nidq_frame_index/nidq_srate
    
    
    # calculate video framerate
    t1_s=t1/nidq_srate
    tend_s=frame_index_s[-1]+t1_s
    
    tot_time=tend_s-t1_s
    tot_frames=len(frame_index_s)
    vframerate=tot_frames/tot_time  
    
    
        
        
    #%% get diode
    
    threshold=10
    diode_channel_num=0
    diode_channel = cut_rawData[[diode_channel_num], :]
       
    
    #Do gain correction
    
    diode = np.squeeze( 1e3*glx.GainCorrectNI(diode_channel, [1], nidq_meta))# gain correction
    
    #Detrend
    degree=2
    fit=np.polyfit(np.arange(len(diode)), diode, degree)
    y=np.polyval(fit, np.arange(len(diode)))
    d_diode=diode-y
    
    # Make signal binary
    binary_diode=(np.array(d_diode) > threshold).astype(int)
    
    
    # get peaks indices
    srate=float(nidq_meta['niSampRate'])
    diode_diff=hf.diff(binary_diode)
    all_peaks=np.where(diode_diff>0)[0]
    diode_index=all_peaks[hf.diff(all_peaks, fill=4*srate)>1.5*srate] # Take only those peaks where nothing else happens in the next 1.5 secs
                                            # The fill is to make sure that the first index i included
    
    #Save indices in s and in frames format
    # convert to s
    diode_s_index=diode_index/srate
    
    # convert to frames
    diode_frame_index=[]
    for i, loom in enumerate(diode_index):
        framenumber=np.nanmin(np.where(nidq_frame_index>loom))
        diode_frame_index.append(framenumber)
    diode_frame_index=np.array(diode_frame_index)
    
    csv_loom_s=pp.convert2s(paths['csv_timestamp'], paths['csv_loom'])
    error=np.abs(csv_loom_s-diode_s_index)
    if np.nanmax(error)>.5:
        
        
        raise ValueError('csv timestamps dont agree with nidq stamps')
    
    return paths, diode_s_index, diode_frame_index, binary_diode, t1_s, tend_s, frame_index_s, frame_channel, vframerate

(paths, diode_s_index, 
 diode_frame_index,
 binary_diode,
 t1_s,
 tend_s,
 frame_index_s,
 frame_channel, 
 vframerate)= run_script()





print("now continue from section 'get spiketimes' from preprocess_all.py")
