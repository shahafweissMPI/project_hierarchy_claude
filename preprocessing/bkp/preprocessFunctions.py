import numpy as np
import readSGLX as glx
from pathlib import Path
import helperFunctions as hf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import os
import h5py
import cv2
# from moviepy.editor import VideoFileClip
# import imageio
from datetime import datetime as d
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import math
import IPython
#%% NIDQ preprocessing


def ChannelCountsNI(meta):
    chanCountList = meta['snsMnMaXaDw'].split(sep=',')
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return(MN, MA, XA, DW)

def cut_rawData2vframes(paths, frame_channel_index=None):
    """
    Cuts raw data to beginning of video  recording 
    creates index for video frames in s

    Parameters
    ----------
    path : path to nidq.bin containing glx raw data
    
    frame_channel_index: in which channel is the signal for the video frames???
    
    digital: is the signal in frame channel digital or analog. If false, signal gets digitised
    
    
    Returns
    -------
    srate : sampling rate of raw data
        
    cut_rawData : raw data cut to the start of the video frames
        
    nidq_frame_index : in sampling rate of nidq, when does each frame happen?
    
    nidq_meta: meta file for nidq
        
    frame_index_s : in s, when does each frame happen?
    
    t1_s: how many s after recording start does the video start?
    
    tend_s: how many s after recording start does the video stop?
    
    vframerate: Caluclated framerate of video
        
    """
    
    
    #import IPython; IPython.embed()
    nidq_meta = glx.readMeta(paths['nidq'])
    nidq_srate=float(nidq_meta['niSampRate'])
    print(f'ChannelCountsNI (MN, MA, XA, DW ) {ChannelCountsNI(nidq_meta)}')
    rawData = glx.makeMemMapRaw(Path(paths['nidq']).as_posix(), nidq_meta)
    
    # #channel plot
    # plt.figure()
    # plt.plot(rawData[0])
    # for i, n in enumerate(rawData):
    #     plt.subplot(len(rawData),1,i+1)
    #     plt.plot(n)
    #     plt.title(f'channel: {i}')
    
    
    #Is the signal digital??
    # if len (rawData)==4:
    #     digital=True
    #     frame_channel_index=6
    #     print(f'assuming digital signal and framechannel={frame_channel_index}')
    # elif len(rawData)==9:
    #     digital=False
    #     frame_channel_index=0
    #     print(f'assuming analog signal and framechannel={frame_channel_index}')
    # else:
    #     raise ValueError('check which channel is the frame channel, and if signal is digital or analog')
     
   
    # if digital:
    digital=True
    if frame_channel_index is None:
        frame_channel_index=int(float(paths['framechannel']))
    
    # frame_channel=rawData[frame_channel_index].copy()
    frame_channel=np.squeeze(
        glx.ExtractDigital(
            rawData, 
            firstSamp=0, 
            lastSamp= int(rawData.shape[1]/16)*16,#rawData.shape[1], 
            dwReq=0, 
            dLineList=[frame_channel_index], 
            meta=nidq_meta)
        ).astype(int)

        
        
        
    # else: #Analogue signal

    #     # get raw frame_channel signal 
    #     raw_frame_channel = np.squeeze(rawData[[frame_channel_index], :])
        
    #     #Give frames positive peaks        
    #     raw_frame_channel*= -1
    #     raw_frame_channel+=raw_frame_channel[0]
        
    #     if np.sum(raw_frame_channel<0):
    #         raise ValueError('assumption about nidq channel unmet, check if frame peaks are negative')
    
    #     # make signal binary
    #     bin_cutoff=.7*np.nanmax(raw_frame_channel)
    #     frame_channel = (np.array(raw_frame_channel) > bin_cutoff).astype(int)

    #get frame indices
    #
    framechannel_diff=np.hstack([0,np.diff(frame_channel)])
    
    nidq_frame_index=np.where(framechannel_diff>0)[0]
   
    if nidq_frame_index.size<1:
        raise ValueError(f'framechannel_diff is all zeroes \n check channel')


    # cut rawData for getting loomtimes in next function
    t1=nidq_frame_index[0]
    t1_s=t1/nidq_srate
    
    tend=nidq_frame_index[-1]
    tend_s=tend/nidq_srate
    
    cut_rawData=rawData[:,t1:tend]
    
    # cut frame index to start of video (then it has the same length as rawData as well)
    nidq_frame_index-=t1

    #convert to s
    frame_index_s=nidq_frame_index/nidq_srate
    

    # calculate video framerate
    tot_time=tend_s-t1_s
    tot_frames=len(frame_index_s)
    vframerate=tot_frames/tot_time  
    
    
    return frame_index_s, t1_s, tend_s, vframerate, cut_rawData, nidq_frame_index, nidq_meta



def get_meta(metaPath):
    """ adapted from readSGLX. metapath= path to either ap.meta, lf.meta, or nidq.meta"""
    metaPath=Path(metaPath)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)

def get_loom_times(cut_rawData, meta, frame_index, threshold, detrend=True, gain_correction=True, min_delay=.3):
    """
    takes digital signal of loomtimes, thresholds the signa, and outputs the times when looms happen

    Parameters
    ----------
    diode_channel : rawdata, cut to start of video, only the channel on which the diode is
        
    meta : metafile, stored next to ndiq.bin file
        
    frame_index : in srate times, when does each frame happen? (created by cut_rawData2vframes)
        
    threshold : from when is activity considered signal for loom?
        . The default is 9.6.
        
    detrend : BOOLEAN, should detrending be applied to diode data?
        The default is True
        
    gain_correction : BOOLEAN, should detrending be applied to diode data?
        The default is True.
    min_delay: in s; how long after one loom is detected should no more looms be detected?

    Returns
    -------
    diode_s_index : when do the looms happen in s time
        
    diode_frame_index : when do the looms happen in frames
        .
    diode_index : when do the looms happen in srate time
       

    """
    
    if math.isnan(threshold):
        raise ValueError('no threshold specified in paths file')
    
    if  not 'diode_channel_num' in globals():
        if len(cut_rawData)==4:
            diode_channel_num=0
            print('assuming diode channel=0')
        elif len(cut_rawData)==9:
            diode_channel_num=1
            print('assuming diode channel=1')
        elif len(cut_rawData)==3:
            diode_channel_num=1
            print('assuming diode channel=1')
        else:
            raise ValueError ('not sure which is the diode channel')
    else:
        print(f'using diode channel = {diode_channel_num} from csv file')
    
    diode_channel = cut_rawData[[diode_channel_num], :]
   
    
    #Do gain correction
    if gain_correction:
        diode = np.squeeze( 1e3*glx.GainCorrectNI(diode_channel, [1], meta))# gain correction
    else:
        diode=diode_channel

    #Detrend signal
    if detrend:
        degree=2
        fit=np.polyfit(np.arange(len(diode)), diode, degree)
        y=np.polyval(fit, np.arange(len(diode)))
        d_diode=diode-y
    else:
        d_diode=diode

    # Make signal binary
    binary_diode=(np.array(d_diode) > threshold).astype(int)

    # get peaks indices
    srate=float(meta['niSampRate'])
    diode_diff=hf.diff(binary_diode)
    all_peaks=np.where(diode_diff>0)[0]
    diode_index=all_peaks[hf.diff(all_peaks, fill=4*srate)>min_delay*srate] # Take only those peaks where nothing else happens in the next 1.5 secs
                                            # The fill is to make sure that the first index i included

    #Save indices in s and in frames format
    # convert to s
    diode_s_index=diode_index/srate

    # convert to frames
    diode_frame_index=[]
    for i, loom in enumerate(diode_index):
        framenumber=np.nanmin(np.where(frame_index>loom))
        diode_frame_index.append(framenumber)
    diode_frame_index=np.array(diode_frame_index)
    
    return diode_s_index, diode_frame_index, diode_index, d_diode



def convert2s(all_stamps_path,loom_stamp_path, framerate=1):
    """This code converts timestamp in format 2023-09-01T15:24:12.4856064+02:00 
    into timestamps. 
    all_stamps= list of timestamps you want to be converted
    first_stamp= The first timestamp t0
    
    returns time in s, unless framerate is given"""
    
    
    first_stamp=pd.read_csv(all_stamps_path)['Timestamp'][0]
    loom_stamps=pd.read_csv(loom_stamp_path, header=None)[1].to_numpy()
    
    
    timestamp_format = "%Y-%m-%dT%H:%M:%S.%f"
    first_stamp = first_stamp.split('+')[0][:-1]
    
    converted_first_stamp = d.strptime(first_stamp, timestamp_format)
    
    
    out_times=[]
    for stamp in loom_stamps:
        stamp= stamp.split('+')[0][:-1]
        converted_stamp = d.strptime(stamp, timestamp_format)
        frame_number = (converted_stamp - converted_first_stamp).total_seconds() * framerate
        out_times.append(frame_number)
    
    return np.array(out_times)




#%% neural data

def process_ch(channel, cutind, rshp_factor, i, lfp_meta):
    """
    This function is for paralleisation of load_lfp()
    it cuts, resamples, and converts to uV dataRaw separately for each channel
    """
    # cut to video time
    
    cut=channel[cutind]

    # resample rawData 
    rsmpld=hf.padded_reshape(cut, rshp_factor).mean(axis=1)

    # convert to uV 
    conv = 1e6*glx.GainCorrectIM( rsmpld[None, :], [i], lfp_meta)

    return conv



def load_lfp(path, t1_s, tend_s, max_channel=None, rshp_factor=5, njobs=20):
    """
    loads lfp from lf file and downsamples it

    Parameters
    ----------
    path : path to lf file

    t1_s : start of video, output from cut_rawData2vframes

    tend_s : end of video, output from cut_rawData2vframes
      
    max_channel : last channel that is still in the brain, use output from get_probe_tracking for this

    rshp_factor : by how much should be downsampled (how many consecutive values should be averaged for one new value?)
    
    njobs : how many processors to use simultaneously

    Returns
    -------
    dataRsmpld : resampled lfp data, channel*time
    
    tstampsRsmpld : time index for each column of dataRsmpld; in s

    framerate : new framerate after resampling

    """
    lfp_meta=glx.readMeta(path)
    tot_s=float(lfp_meta['fileTimeSecs'])
    
    dataRaw = glx.makeMemMapRaw(path, lfp_meta)
    tstampsRaw=np.linspace(0,tot_s,dataRaw.shape[1])
    
    if max_channel is None:
        max_channel=len(dataRaw)
    
    cutind=(tstampsRaw>=t1_s) & (tstampsRaw<=tend_s)
    tstampsCut=tstampsRaw[cutind]- t1_s
    tstampsRsmpld= hf.padded_reshape(tstampsCut, rshp_factor)[:,0]
    framerate= len(tstampsRsmpld)/tstampsRsmpld[-1]
    
    if len(hf.unique_float(np.diff(tstampsRsmpld)))!=1:
        raise ValueError ('sth wrong with timestamps')

    # cut, resample, convert to uV
    dataRsmpld = np.array(np.squeeze( 
        Parallel(n_jobs=njobs)(
        delayed(process_ch)( 
            channel, cutind, rshp_factor, i, lfp_meta
            ) for i, channel in enumerate(dataRaw[:max_channel+1]))))


    if dataRsmpld.shape[1] != len(tstampsRsmpld):
        raise ValueError ('Timestamps dont match data')
    
    
    return dataRsmpld, tstampsRsmpld, framerate



def load_phy(sortpath, ap_path, t1_s, tend_s, man_clustered=True):
    """
    loads clustered data Only returns units marked as 'good!'
    It also cuts the data to video start, resmaples it to s
    
    ap_path: path to the ap.bin file; needed to retrieve the meta file 

    Parameters
    ----------
    sespath: directory to where sorted data is stored+ the meta file
    t1: When does the video start?
    man_clustered: if True, the 'good' labels from manual clustering will be used, otherwise, the good from kilosort is used

    Returns
    -------
    good_spike_times: When do spikes happen? (shape: n_spikes)
    
    good_spike_clusters: Whch cluster do they belong to?? (shape: n_spikes)
    good_clusters: Which clusters are good? (shape: n_clusters)
    good_clusterchannels: On which channels are these clusters? (shape: n_clusters)

    """
    ap_dir=os.path.dirname(ap_path)
    files=os.listdir(ap_dir)
    meta_name=[file for file in files if 'ap.meta' in file]
    nsignal_meta=get_meta(Path(rf"{ap_dir}\{meta_name[0]}"))
    nsignal_srate=float(nsignal_meta['imSampRate'])
    
    #Assign variables
    spike_times=np.squeeze(np.load(rf'{sortpath}\spike_times.npy')) #when do spikes happen?
    spike_clusters=np.load(rf'{sortpath}\spike_clusters.npy') #to which cluster do spikes belong?
    
    cluster_info=pd.read_csv(f'{sortpath}\cluster_info.tsv',sep='\t')
    loc=cluster_info['tom'].to_numpy().astype(str)# whether spike is axonal or not
    cluster_id=cluster_info['cluster_id'].to_numpy()
    clusterchannel=cluster_info['ch'].to_numpy()
    depth=cluster_info['depth'].to_numpy()
    if man_clustered:
        KSlabel=cluster_info['group'].to_numpy()  
        print('\nusing manually clustered KS labels\n')
    else:
        KSlabel=cluster_info['KSLabel'].to_numpy()
        print('\nusing auto generated KS labels\n')
    
    # convert spike times to s 
    spike_times=spike_times/nsignal_srate
    
    
    if t1_s<=0 or np.nanmax(spike_times) <= tend_s:
        raise ValueError('there is something wrong with t1/tend')
    
    #cut spike times to length of video
    include_ind=(spike_times<tend_s) & (spike_times>t1_s)
    spike_times=spike_times[include_ind]
    spike_clusters=spike_clusters[include_ind]
    
    # make time 0 to video start
    spike_times-=t1_s
    
    #get good clusters/spikes
    print('axonal spikes excluded...')
    good_ind=(KSlabel=='good') & (loc=='nan') #the 'nan' sorts out axonal spikes
    good_clusters=cluster_id[good_ind] # chosen clusters
    good_clusterchannels=clusterchannel[good_ind] #channel nums for chosen clusters
    good_depth=depth[good_ind]
    
    spike_mask = np.squeeze(np.isin(spike_clusters, good_clusters))
    
    good_spike_times = spike_times[spike_mask]
    good_spike_clusters=  spike_clusters[spike_mask]
    
    
    return good_spike_times, good_spike_clusters, good_clusters, good_clusterchannels


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
    
    
    if t1_s<=0 or np.nanmax(spike_times) <= tend_s:
        raise ValueError('there is something wrong with t1/tend')
    
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

def load_IRC_in_pd(quality_file, timestamps_file, t1_s, tend_s):
    # Read the CSV files
    df_a = pd.read_csv(quality_file)
    df_b = pd.read_csv(timestamps_file, header=None)  # Assuming file B doesn't have headers
    
    include_ind=(df_b[0]>t1_s )&(df_b[0]<tend_s)
    df_b=df_b[include_ind]
    df_b.loc[:,0]-=t1_s
    # Filter rows in file A where 'note' column contains 'single'
    filtered_df_a = df_a[df_a['note'].str.contains('single', na=False)]
    # Extract required columns
    result_df = filtered_df_a[['unit_id', 'center_site', 'nSpikes']].copy()
    # Initialize an empty list to store timestamps
    result_df.loc[:, 'timestamps'] = pd.Series([[] for _ in range(len(result_df))], index=result_df.index)
    # For each unit_id in the filtered DataFrame, find matching values in file B and collect timestamps
    for index, row in result_df.iterrows():
        unit_id = row['unit_id']
        # Find rows in df_b where column 2 matches the unit_id
        matching_rows = df_b[df_b[1] == unit_id]
        # Collect the values from column 1 of the matching rows
        timestamps = matching_rows[0].tolist()
        # Assign the collected timestamps to the 'timestamps' column
        result_df.at[index, 'timestamps'] = timestamps
    return result_df


def exclude_neurons(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels):
    """
    exclude neurons that...
    have less than 500 spikes in total
    have more than 2% of spikes with ISI violations (ISI < 2ms)

    Parameters
    ----------
    (all output from previous computations in preprocess_all.py script)
    n_by_t : ndata matrix
        
    spike_times : when do spikes happem, from any neuron
    
    spike_clusters : which cluster do those spikes belong to?
       .
    cluster_index : what is the clusternumber in n_by_t


    Returns
    -------
    n_by_t : neurons that fail criteria are sorted out

    cluster_index : same


    """
    
    exclude_ind=np.ones(len(n_by_t))
    num_spikes=np.sum(n_by_t, axis=1)
    min_spikes=500        
    exclude_ind-= (num_spikes<=min_spikes)
    too_few_spikes=len(n_by_t)-np.sum(exclude_ind)
    print(f'{too_few_spikes} neurons exclude < {min_spikes} spikes')
    
    # Exclude too short ISI
    for n in cluster_index:
        n_spikes=spike_times[spike_clusters==n]
        isi=np.diff(n_spikes)
        violations=np.sum(isi<.002)/len(isi)
        if violations>.02:
            exclude_ind[cluster_index==n] -= 1
    
    exclude_ind[exclude_ind<1]=0
    exclude_ind=exclude_ind.astype(bool)
    n_by_t=n_by_t[exclude_ind]
    cluster_index=cluster_index[exclude_ind]
    cluster_channels=cluster_channels[exclude_ind]
    
    np.count_nonzero(exclude_ind)

    print(f'{len(exclude_ind)-sum(exclude_ind)-too_few_spikes} neurons excluded because too many ISI violations')
    
    return n_by_t, cluster_index, cluster_channels





def get_channels(lfp_path, channel_list,first_frame, last_frame,first_channel, last_channel,step=1):
    
    wanted_channels=np.arange(first_channel, last_channel, step)
    all_channels=[]
    for num in wanted_channels:

        channel=np.load(fr'{lfp_path}\imec0.lf#LF{num}.npy')
        all_channels.append(channel)

    all_channels=np.array(all_channels)
    return all_channels[:,first_frame:last_frame], wanted_channels



def get_probe_tracking(path, manual_last_channel):
    """" 
    
    takes the output from brainreg to assign a region to each recording channel
    
    It assumes that the number of rows in the brainreg csv == (num_channels_in_the_brain + 8_channels_at_the_tip)
    
    It also assumes that the numbering of points is inverted, i.e. 0 in the excel corresponds to the 
    most dorsal point of the probe (whereas 0 in spikeGLX is the most ventral channel)
    
    Parameters
    ----------
    Path: to the track.csv file that is output from brainreg. In the paths.csv it is under 'probe_tracking'
    manual_last_channel: If there is a shift with respect to atlas alignment, this 
        is taken for calculation of depth
    
    """
    regions_csv=pd.read_csv(path)
    
    #assign depth to each channel (also you need to determine num channels)
    depth=regions_csv['Distance from first position [um]'].to_numpy()
    
    if pd.isna(manual_last_channel):        
        tot_depth=depth[-1] - 175 # 175 is the length of the tip
    else:
        tot_depth= (float(manual_last_channel)/2)* 20 #um distance between channels
    
    
    num_rows=int(tot_depth/20) # number of equally spaced rows, each having 2 channels
    row_depth=np.linspace(tot_depth,0,num_rows)
    channel_depth=np.repeat(row_depth, 2, axis=0)
    
    # assign depth to region
    regions=regions_csv['Region acronym'].to_numpy()
    
    diff = np.abs(channel_depth[:, None] - depth)
    indices = np.argmin(diff, axis=1)
    channelregions=regions[indices]
    
    
    # get channel numbers corresponding to regions
    channelnums=np.arange(len(channel_depth))
    
    
    print(f'based on atlas alignment there is {channelnums[-1]} channels in the brain and \ndepth: {tot_depth}um.')
    print('CHECK THIS!!\n\n')
    return channelregions, channelnums


def neurons_by_time(spike_times,spike_clusters, clusters, bin_size):
    """
   Makes neural activity into neurons*time matrix. For each timebin, it counts how many spikes 
   happen for a neuron in that timeperiod

   Parameters
   ----------
   spike_times : when does each spike happen, in s (output from load_phy/ load_IRC)
   spike_clusters: What cluster does each spike belong to (output from load_phy/ load_IRC)
   binsize : in s; in what binsize should the spikes be summed together/ unit of timedimension
   clusters: cluster numbers, should be sorted by depth
   
   


   Returns
   -------
   n_by_t : the neuron*time matrix
   bins: An index for what time in s each timebin corresponds to
   clusters: an index for what cluster each neuron belongs to

    -->to turn this into firing in Hz, you just need to divide it by the bins paramater
    """

    bins = np.arange(0, spike_times.max() + bin_size, bin_size)
    
    # Get the number of neurons
    n_neurons = len(clusters)
    
    # Initialize the firing rate matrix
    n_by_t = np.zeros((n_neurons, len(bins)))
    
    # Calculate the firing rate for each neuron and each time bin
    for i, n in enumerate(clusters):
        neuron_spike_times = spike_times[spike_clusters == n]
        spike_nums, _ = np.histogram(neuron_spike_times, bins=bins)
        n_by_t[i][:len(spike_nums)] = spike_nums   
        
    return n_by_t, bins, clusters


def spike_ind_region(regions, clusterchannels, clusters, spike_clusters, target_regions):
    """
    Takes output from the  pp.load_sorted_good() function and returns  an index 
    into spike_times/ spike_clusters so that only units of that region are left
    
    target_regions can be multiple regions, returns as many arguments as there 
    are target_regions, in the same order as target_regions

    """
    cluster_regions=regions[clusterchannels] #which regions does each cluster have?
    inds=[]
    for region in target_regions:
         
        region_clusters=clusters[cluster_regions==region] # Which clusters belong to the target region?
        region_ind=[]
        for c in region_clusters:
            region_ind.append(np.where(spike_clusters==c)[0])
        inds.append(np.hstack(region_ind))
    return inds



#%%SLEAP
def fill_missing(Y, kind="linear"):
    """interpolates missing values in the sleap data. This happens independently along each dimension after the first.
    Y=???"""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y



def extract_sleap(path, vpath, node, vframerate,Cm2Pixel=None,locations=None,node_names=None):
    """
    path: path to sleap file
    vpath: path to video
    node: which node to take for calculating velocity
    calculates velocity in cm per second (assuming the camera angle/ pixel number didn't change!!')
    # this function assumes x,y pixel to cm ratios are the same !!!!!!!!!!! - TODO
    """
    if node_names is None or locations is None:
        with h5py.File(path, "r") as f:
            dset_names = list(f.keys())
            locations = np.squeeze(f["tracks"][:].T )#frames * nodes * x,y * animals
            node_names = np.array([n.decode() for n in f["node_names"][:]])
    
    #replace outliers with nans
   
   
    mean_loc=np.nanmean(locations, axis=1)
    dist=hf.euclidean_distance_old(locations, mean_loc[:,None,:], axis=2)    
    outlierframes=np.where(dist>75)[0]
    locations[outlierframes]=np.nan
    
    

    # fill in nans via interpolation 
    locations = fill_missing(locations)
    
    # Choose head as central location 
    node_index=np.where(node_names==node)[0]
    node_loc= np.squeeze(locations[:, node_index, :]) #frames *x,y
    
    
    # Convert pixels to cm
    # old camera
    w,h=video_width_height(vpath)
    
    # if (w==1280) and (h==1024):
    #     xpixel2cm=88/1005
    #     ypixel2cm=88/999
    #             #cm/pixel
                
    # # new camera
    # elif w==h==1920:
    #     xpixel2cm=88/1721
    #     ypixel2cm=88/1715
    #             # cm/pixel
    # else:
    #     raise ValueError ('unkown frame size, update pixel to cm conversion')
    
    # node_loc[:,0]*=xpixel2cm
    # node_loc[:,1]*=ypixel2cm
    
    x_coords = locations[:, 2, 0]
    y_coords = locations[:, 2, 1]
    plt.plot(x_coords,y_coords)
    
    #Get velocity out
    # velocity= ep.get_velocity(head[:,:])
    # return velocity
    ## assuming x,y are more or less the same ratio
    node_loc[:,0]*=Cm2Pixel
    node_loc[:,1]*=Cm2Pixel
    
    smooth_node_loc = np.zeros_like((node_loc[:,:]))

    for c in range(node_loc.shape[-1]):
        smooth_node_loc[:, c] = savgol_filter(node_loc[:, c], window_length=25, polyorder=3, deriv=1)
    
    node_vel = np.linalg.norm(smooth_node_loc,axis=1)  
    
    # convert to s 
    node_vel*= vframerate #this is now cm/s
    
    

    return node_vel, np.squeeze(locations), node_names

def video_width_height(video_path):
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height




#%% Boris

def load_boris(path):
    #Load data
    data=pd.read_csv(path)
    
    #Get relevant columns/ data
    behaviours=data['Behavior'].to_numpy()
    frames=data['Image index'].to_numpy()
    start_stop=data['Behavior type'].to_numpy()
    try:
        modifier=data['Modifier #1'].to_numpy()
    except:
        modifier=[]
        
    if np.sum(np.isnan(frames)): # sum nan values>0
        raise ValueError('There is nan values in boris frame numbers, go over the annotations again')

    return behaviours, frames, start_stop, modifier





#%% general

def get_paths(session=None, animal=None):
    """
    Retrieves paths from excel
    if no session is specified, then entire pd is returned
    

    Parameters
    ----------
    session : name of session, as given in the .csv
    Animal: if specified, all sessions from tha animal are returned

    Returns
    -------
    pd with paths to data

    """
    paths=pd.read_csv(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv")
    if animal is not None:
        apaths=paths[paths['Mouse_ID']==animal]
        if session is not None:
            sesline=apaths[apaths['session']==session]
            return sesline.squeeze()
        return apaths
    
    elif session is not None:
        sesline=paths[paths['session']==session]
        return sesline.squeeze().astype(str)
    else:
        return paths


def unique_float(vector, precision=10):
    """ calculates unique values in vector. It ignores differences that are lower than 'precision' decimals """
    
    rounded_vector = np.round(vector, precision)
    unique_values = np.unique(rounded_vector)
    return unique_values

# def count_frames(video_path):
#     clip = VideoFileClip(video_path)
#     num_frames = clip.reader.nframes -1
#     return num_frames


def count_frames(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Could not open video")
        return
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return frame_count


def get_power_spectrum (data, framerate, axis=1):
    
    fft_vals = np.fft.rfft(data, axis)

    # Compute the power spectral density (PSD), which is the square of the absolute value of the FFT
    psd_vals = np.abs(fft_vals) ** 2 # magitude of the respective frequency
    
    # Compute the frequencies corresponding to the PSD values
    # If `dt` is the time step between your data points:
    dt = 1/framerate  # for example
    freqs = np.fft.rfftfreq(data.shape[axis], dt)
    
    return freqs, psd_vals
