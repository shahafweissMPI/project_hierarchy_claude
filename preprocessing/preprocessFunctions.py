import numpy as np
import readSGLX as glx
from pathlib import Path,PureWindowsPath
import helperFunctions as hf
from scipy.interpolate import interp1d
#from scipy.signal import savgol_filter
import pandas as pd
import os
import h5py
try:
    import cv2
except ModuleNotFoundError:
    print("cv2 couldn't be imported, video processing won't work")
# from moviepy.editor import VideoFileClip
# import imageio
from datetime import datetime as d
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from movement.io import load_poses
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.kinematics import compute_velocity
import math
import IPython
import builtins

import numba
#%% NIDQ preprocessing

import time
import logging
import datetime
import xarray as xr

import spikeinterface_helper_functions as sf
import preprocessFunctions as pp
import plottingFunctions as pf
import helperFunctions as hf


from movement.plots import plot_centroid_trajectory
from movement.roi import PolygonOfInterest
# %%
# Load sample dataset
# -------------------
# In this example, we will use the ``SLEAP_three-mice_Aeon_proofread`` example
# dataset. We only need the ``position`` data array, so we store it in a
# separate variable.
from movement.io import load_poses
from matplotlib import pyplot as plt
from scipy.signal import welch

import movement.kinematics as kin
from movement.filtering import (
    interpolate_over_time,
    savgol_filter)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTimer:
    def __init__(self, logger, message="Total elapsed time"):
        self.logger = logger
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()
        self.last_split = self.start
        return self  # Allow calling the split method

    def split(self, label: str):
        """Record an intermediate time and log time in HH:MM:SS format."""
        now = time.perf_counter()
        elapsed_since_split = now - self.last_split
        elapsed_td = datetime.timedelta(seconds=elapsed_since_split)
        self.logger.info(f"{label}: {elapsed_td} since last split")
        self.last_split = now

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_elapsed = time.perf_counter() - self.start
        total_elapsed_td = datetime.timedelta(seconds=total_elapsed)
        self.logger.info(f"{self.message}: {total_elapsed_td}")



def ChannelCountsNI(meta):
    chanCountList = meta['snsMnMaXaDw'].split(sep=',')
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return(MN, MA, XA, DW)

def cut_rawData2vframes(paths, frame_channel_index=None,firstS=0,lastS=0,digital=True):
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
    
  
    nidq_meta = glx.readMeta(paths['nidq'])
    nidq_srate=float(nidq_meta['niSampRate'])
    samplerate=nidq_srate
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
    # if len(rawData)==4:
    #     digital=True
    #     frame_channel_index=6
    #     print(f'assuming digital signal and framechannel={frame_channel_index}')
    # elif len(rawData)==9:
    #     digital=False
    #     frame_channel_index=0
    #     print(f'assuming analog signal and framechannel={frame_channel_index}')
    # else:
    #     raise ValueError('check which channel is the frame channel, and if signal is digital or analog')
     

    if digital:
        
        if frame_channel_index is None:
            frame_channel_index=int(float(paths['framechannel']))
        print('getting cammera TTLs')
        
        if lastS!=0:
            firstSamp=int(firstS*samplerate)
            lastSamp= int(lastS*samplerate)
            
        else:
            print("lastS is 0, computing from session recording data")

            firstSamp=int(0.0)
            
            #lastSamp= int(float(nidq_meta['fileTimeSecs'])*float(nidq_meta['niSampRate']))
            
            try:
                lastSamp= int(rawData.shape[1]/16)*16
        
        # frame_channel=rawData[frame_channel_index].copy()
                frame_channel=np.squeeze(glx.ExtractDigital(rawData, firstSamp=firstSamp, lastSamp= lastSamp,#rawData.shape[1], 
                                dwReq=0,                 dLineList=[frame_channel_index],                 meta=nidq_meta)            ).astype(int)
        
            except:
                lastSamp= int(int(rawData.shape[1]/16)*float(nidq_meta['nSavedChans']))
            frame_channel=np.squeeze(
                glx.ExtractDigital(
                    rawData, 
                    firstSamp=firstSamp, 
                    lastSamp= lastSamp,#rawData.shape[1], 
                    dwReq=0, 
                    dLineList=[frame_channel_index], 
                    meta=nidq_meta)
                ).astype(int)
        #       raise ValueError("bad framchannel ID")
               
    
            
            
            
    else: #Analogue signal
         digital=False
         # get raw frame_channel signal 
         raw_frame_channel = np.squeeze(rawData[[frame_channel_index], :])
        
         #Give frames positive peaks        
         raw_frame_channel*= -1
         raw_frame_channel+=raw_frame_channel[0]
        
         if np.sum(raw_frame_channel<0):
             raise ValueError('assumption about nidq channel unmet, check if frame peaks are negative')
    
         # make signal binary
         bin_cutoff=.7*np.nanmax(raw_frame_channel)
         frame_channel = (np.array(raw_frame_channel) > bin_cutoff).astype(int)

    #get frame indices
    #
    
    framechannel_diff=np.hstack([0,np.diff(frame_channel)])
    #print(f"{np.std(framechannel_diff)}=")
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

def get_loom_times(cut_rawData, meta, frame_index, diode_channel_num, threshold, detrend=True, gain_correction=True, min_delay=.3):
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
    
    if  (math.isnan(diode_channel_num)) or (diode_channel_num is None):
        if len(cut_rawData)==4 or len(cut_rawData)==2:
            diode_channel_num=0
            print('assuming diode channel=0')
        elif len(cut_rawData)==9:
            diode_channel_num=1
            print('assuming diode channel=1')
        elif len(cut_rawData)==3:
            diode_channel_num=1
            print('assuming diode channel=1')
        else:
            print(f"{len(cut_rawData)=}")
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
    if np.all(all_peaks==0):
        diode_index=[]
        diode_s_index=[]
        diode_frame_index=[]
        #raise Warning(f'csv_loom file specified, but diode channel is always zero!, check channel number')
        print(f'csv_loom file specified, but diode channel is always zero!, check channel number')
    else:
        #Save indices in s and in frames format
        # convert to s
        diode_index=all_peaks[hf.diff(all_peaks, fill=4*srate)>min_delay*srate] # Take only those peaks where nothing else happens in the next 1.5 secs
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


def load_ops(ops_path, device=None):
    """Load a saved `ops` dictionary and convert some arrays to tensors."""
    import torch
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    ops = np.load(ops_path, allow_pickle=True).item()
    for k, v in ops.items():
        if k in ops['is_tensor']:
            ops[k] = torch.from_numpy(v).to(device)
    # TODO: Why do we have one copy of this saved as numpy, one as tensor,
    #       at different levels?
    ops['preprocessing'] = {k: torch.from_numpy(v).to(device)
                            for k,v in ops['preprocessing'].items()}

    return ops
def load_Kilosort4(sortpath, ap_path, t1_s, tend_s, man_clustered=True):
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
    fs=nsignal_srate
    
    # outputs saved to results_dir
    results_dir = Path(sortpath)
    ops = load_ops(results_dir / 'ops.npy')
    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
    chan_map =  np.load(results_dir / 'channel_map.npy')
    templates =  np.load(results_dir / 'templates.npy')
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
    chan_inds=chan_best
    chan_best = chan_map[chan_best]
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    st = np.load(results_dir / 'spike_times.npy')
    clu = np.load(results_dir / 'spike_clusters.npy')
    firing_rates = np.unique(clu, return_counts=True)[1] * fs / st.max()
    dshift = ops['dshift']
    chan_positions =  np.load(results_dir / 'channel_positions.npy')
    chan_positions=chan_positions[chan_inds]
    chan_shanks=  np.load(results_dir / 'channel_shanks.npy')
    chan_shanks=chan_shanks[chan_inds]
    
    #from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
#    from npyx.gl import get_units
 #   units = get_units(dp, quality='good')
#  #  if Path(results_dir / 'cluster_group.tsv').is_file():
    if True:
        file='cluster_group.tsv'
        cluster_info=pd.read_csv(results_dir / 'cluster_group.tsv',sep='\t')        
        print('\nusing manually clustered KS labels\n')                   
        KSlabel=cluster_info['group'].to_numpy()          
        cluster_id=cluster_info['cluster_id'].to_numpy()          
    else:
        file = 'cluster_KSLabel.tsv'
        cluster_info=pd.read_csv(results_dir / 'cluster_KSLabel.tsv',sep='\t')
        KSlabel=cluster_info['KSLabel'].to_numpy()          
        cluster_id=cluster_info['cluster_id'].to_numpy()          
        print('\nusing auto generated KS labels\n')
    
    
    
    def get_labels(results_dir,file):
        """Load good/mua labels as a list of ['cluster', 'label'] pairs."""
        results_dir = Path(results_dir)
        filename = results_dir / file
        with open(filename) as f:
            text = f.read()
        rows = text.split('\n')
        labels = [r.split('\t') for r in rows[1:]][:-1]
    
        return labels
    def get_good_clusters(results_dir, n=1):
        """Pick `n` random cluster ids with a label of 'good' and return both the cluster ids and their indices."""
        labels = get_labels(results_dir, file)
        # Filter labels to only those that are 'good'
        good_labels = [x for x in labels if x[1] == 'good']
        
        # Randomly select indices from the filtered good labels
        indices = np.random.choice(
            np.arange(len(good_labels)), size=min(n, len(good_labels)), replace=False
        )
        
        # Get the cluster ids corresponding to the selected indices
        cluster_ids = [int(good_labels[r][0]) for r in indices]
        
        # If only one element is requested, return single values instead of list
        if n == 1:
            cluster_ids = cluster_ids[0]
            indices = indices[0]
        
        return cluster_ids, indices
    
    # def get_good_clusters(results_dir, n=1):
    #     """Pick `n` random cluster ids with a label of 'good.'"""
    #     labels = get_labels(results_dir,file)
    #     labels = [x for x in labels if x[1] == 'good']
    #     rows = np.random.choice(
    #         np.arange(len(labels)), size=min(n, len(labels)), replace=False
    #         )
    #     cluster_ids = [int(labels[r][0]) for r in rows]
    #     if n == 1:
    #         cluster_ids = cluster_ids[0]
    
    #     return cluster_ids
    # Find indices where the second element is "good"
    
    
   
   
    cluster_id=get_labels(results_dir,file)
    # good_clusters,indices=get_good_clusters(results_dir,n=len(cluster_id))
    
    # good_ind = [i for i, pair in enumerate(cluster_id) if pair[1] == "good"]
    
    # Keep only the pairs with "good" as the second element
    #cluster_id2 = [cluster_id[i] for i in good_ind]
   
    # Display the results
    #print("Indices of 'good':", good_ind)
#    print("Cluster IDs with 'good':", cluster_id)
   
    spike_times=st
    spike_clusters=clu
    #rest should be the same as load_phy
    #convert spike times to s     
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
    #print('axonal spikes excluded...')
    cluster_ids = [item[0] for item in cluster_id]
    

    # 2) Extract the second value of each item (the words):
    labels = [item[1] for item in cluster_id]
   
    

    # 3) Get the indices where the second value equals "good":
    #good_ind = [index for index, item in enumerate(cluster_ids) if item[1] == "good"]
    good_ind=np.where(labels=='good')# & (loc=='nan') #the 'nan' sorts out axonal spikes
    good_indx=np.where(good_ind)  
    #good_clusters=cluster_id[good_ind] # chosen clusters
    clusterchannel=chan_best
    
    
    good_clusterchannels=clusterchannel[good_indx] #channel nums for chosen clusters
    adjusted_chan_map = chan_map - 1


    # If chan_best is one-indexed, adjust as well:
    adjusted_chan_best = chan_best 

    # ----------------------------------------------------------------------------------
    # 1) For each element in adjusted_chan_map, get the corresponding Y position
    #    The second column (index 1) of chan_positions is the Y coordinate.
    y_positions_for_chan_map = chan_positions[adjusted_chan_map, 1]
    
    # ----------------------------------------------------------------------------------
    # 2) For each element in adjusted_chan_best, first use the adjusted_chan_best value
    #    to index into adjusted_chan_map, then use that to index into chan_positions
    #    to get the corresponding Y coordinate.
    y_positions_for_chan_best = chan_positions[ adjusted_chan_map[adjusted_chan_best] , 1 ]
    x_positions_for_chan_best = chan_positions[ adjusted_chan_map[adjusted_chan_best] , 0 ]

    
    depth=y_positions_for_chan_best
    good_depth=depth[good_ind]
    depth=good_depth
    x_coord=x_positions_for_chan_best[good_ind]
    y_coord=y_positions_for_chan_best[good_ind]
    
    
    
    spike_mask = np.squeeze(np.isin(spike_clusters, good_clusters))
    
    good_spike_times = spike_times[spike_mask]
    good_spike_clusters=  spike_clusters[spike_mask]
    
    good_templates=templates[good_ind]
   
      
    good_chan_positions=chan_positions[good_indx]
    KS_results={
        'spike_times': good_spike_times,
        'spike_cluster':good_spike_clusters,
        'clusters':good_clusters, 
        'clusterchannels':good_clusterchannels,
        'templates':good_templates,
        'chan_positions':good_chan_positions,
        'chan_shanks':chan_shanks[good_indx],
        'firing_rates':firing_rates[good_ind],
        'ops':ops,
        'results_dir':results_dir,
        }
    
    return good_spike_times, good_spike_clusters, good_clusters, good_clusterchannels,KS_results
   
    
        
    
    
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
   
    if t1_s<0 or np.nanmax(spike_times) <= tend_s:
        
        print(f"{t1_s=} {np.nanmax(spike_times)=}  {tend_s=}")
        IPython.embed()
        #raise ValueError('there is something wrong with t1/tend')
        #maybe use the paths start time?
        #spike_times=spike_times+7271.942530
        include_ind=(spike_times<tend_s) & (spike_times>t1_s)
#        include_ind=spike_times<tend_s
 #       spike_times = spike_times[spike_times < tend_s]
    else:
        # Exclude spikes after the end of video 
        include_ind=(spike_times<tend_s) & (spike_times>t1_s)
        
    spike_times=spike_times[include_ind]
    spike_channel=spike_channel[include_ind]
    spike_clusters=spike_clusters[include_ind]
    #####    #####    #####    #####    #####    #####    
    ## give video start time 0
    spike_times -= t1_s
    #####    #####    #####    #####    #####    #####    

    
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

def load_IRC_in_SPI(output_folder,quality_file, timestamps_file, t1_s, tend_s,spikeglx_folder,analyzer_mode='memory',recompute_anlayzers=False):
    """
    output_folder
    quality_file (if ironclust curation was used)
    timestamps_file  (if ironclust curation was used)
    t1_s: time of session start
    tend_s: time of session end
    spikeglx_folder
    analyzer_mod:=format of analyzer : "memory"/  "binary_folder"
    recompute_anlayzers: if analyzer exists, do you want to recompute it? True/False 
    """
    #good_spike_times, good_spike_clusters, good_clusters, good_cluster_channels=load_IRC(timestamps_file, t1_s, tend_s, quality_file )
#    good_spike_times+=t1_s
    
    #os.environ['NUMEXPR_MAX_THREADS'] = '1'
    #os.environ["OPENBLAS_NUM_THREADS"] = "1"
    ## convert to spikeinterface sorting
    import numpy as np        
    import matplotlib.pyplot as plt
    import spikeinterface.full as si
    from spikeinterface import create_sorting_analyzer, load_sorting_analyzer
    import spikeinterface.extractors as se
    from spikeinterface.exporters import export_report
    from spikeinterface.curation import remove_duplicated_spikes,remove_redundant_units,get_potential_auto_merge    
    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms
    from spikeinterface.qualitymetrics import compute_quality_metrics
    import spikeinterface.qualitymetrics as sqm
    import spikeinterface.postprocessing as spost
    import spikeinterface.core as score 
    import matplotlib.pyplot as plt
    import spikeinterface.widgets as sw 
    import spikeinterface.curation as scur      
    plt.ion()  # Turn on interactive mode
    plt.rcParams.update({
    'font.size': 12,            # controls default text sizes
    'axes.titlesize': 12,       # fontsize of the axes title
    'axes.labelsize': 12,       # fontsize of the x and y labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 18      # fontsize of the figure title
    })                     
    
 
    print('loading ironclust to spikeinterface pipeline to get quality metrics')
    print('loading CSV files')
    #read quality file
    df_quality=pd.read_csv(quality_file)
    single_units = df_quality[df_quality['note'] == 'single']['unit_id']
    #df_quality = df_quality[df_quality['unit_id'].isin(single_units)]
#    ISI_ratio = df_quality[df_quality['unit_id'] == single_units]['unit_id']
#    L_ratio = df_quality[df_quality['note'] == 'single']['unit_id']
    
    #read timestamps file
    df=pd.read_csv(timestamps_file, header=None)
    df.columns = ['spike_times', 'unit_id', 'max_amp_site']

    # Remove rows with negative cluster numbers
    df_filtered = df[df['unit_id'] > 0]#remove deleted and non annotated clusters

    # Filter df_filtered to keep only rows with unit_ids in the single_units list
    df_filtered_single = df_filtered[df_filtered['unit_id'].isin(single_units)]

    # Display the filtered DataFrame
#    print(df_filtered_single)

    #spiketimes group and sort by unit id 
    df_grouped = df_filtered_single.groupby('unit_id')['spike_times'].apply(list).reset_index()
    # Group by 'cluster_number' and 'max_amp_site', and aggregate spike times into lists
    #df_grouped = df_filtered.groupby(['unit_id', 'max_amp_site'])['spike_times'].apply(list).reset_index()

    # Select the first max_amp_site for each cluster_number
    df_grouped['max_amp_site'] = df_filtered_single.groupby('unit_id')['max_amp_site'].first().values

    #merge to single dataframe
    df_irc = pd.merge(df_grouped, df_quality, on='unit_id')
    df_irc = df_irc[df_irc['note'] == 'single'] # keep only single units
    
    df_irc

    #labels = df_irc['note'].to_list()
    unit_ids=  np.array(df_filtered_single.unit_id.to_list())
    #sampling_frequency=
    spike_times = np.array(df_filtered_single.spike_times.tolist())
    
   
    # Assuming spike_times is a NumPy array and t_start, t_end are defined:
    filtered_spike_times_indeces =((spike_times >= t1_s) & (spike_times <= tend_s))
    spike_times=spike_times[filtered_spike_times_indeces]
    unit_ids=unit_ids[filtered_spike_times_indeces]
   
     
    # OR
     # Assuming spike_times is a list and t_start, t_end are defined:

    print(f'loading recording from: \n {spikeglx_folder}')
    #folder = "analyzer_folder"
    #spikeglx_folder = r'G:\scratch\afm16924\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0'
    #output_folder=r'G:\scratch\afm16924\supercat_afm16924_240522_g0\results_multi'
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    recording = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)
    print(f"preprocessing .ap stream")
    t_start=recording.get_time_info()['t_start']
    recording.shift_times(shift=-t_start)
    recording = si.bandpass_filter(recording, freq_min=400., freq_max=3000.0,filter_order=3)
    recording = si.common_reference(recording, operator="median", reference="local",local_radius=(40,160))
    #bandpass and stuff:
    if 'cat' in str(spikeglx_folder):
         pass
    else:
        recording=si.phase_shift(recording)
    
    
    
    
    fs=recording.sampling_frequency
    recording_full_concat=recording.clone()
    recording_full= recording.frame_slice(np.int64(t1_s*fs), np.int64(tend_s*fs))
    
    recording= recording.frame_slice(np.int64(t1_s*fs), np.int64((t1_s+60*7)*fs))
    #recording=recording.time_slice(t1_s,t1_s+60*7)

    first_frame=np.int64(t1_s*fs)
    last_fram=recording.get_num_frames()
   
#    recording.get_start_time()
#    t=recording.get_times()
#    recording.get_total_samples()
#    recording.get_time_info()
   # recording= recording.time_slice(np.float64(t1_s), np.float64(tend_s))
    
    
    

    #recording = si.whiten(recording,dtype=float,mode='local',radius_um=160)
  
# #    bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
#  #   rec2 = rec1.remove_channels(bad_channel_ids)
#   #  print('bad_channel_ids', bad_channel_ids)
    
#     recording = si.phase_shift(recording)
#     recording = si.common_reference(recording, operator="median", reference="local",local_radius=(40,160))
    
    print(recording)
   
    #session = [t1_s, tend_s]
    #session_multiplied = [np.int64(value * recording.sampling_frequency) for value in session]
   
#    t_start=session[0]
 #   t_end=session[1]
    #noise_levels_microV = si.get_noise_levels(catgt_rec, return_scaled=True)
 #   filtered_spike_times = [time for time in spike_times if t_start <= time <= t_end]                          
    # Using a list comprehension to get both indices and values:
  #  filtered_indices_values = [(i, time) for i, time in enumerate(spike_times) if t_start <= time <= t_end]

    # If you want to separate the indices and values into two lists:
   # indices, values = zip(*filtered_indices_values) if filtered_indices_values else ([], [])
   # params={
   #     'sampleRate': fs}
   # from metrics import slidingRP_all
   # RP=slidingRP_all(spikeTimes=spike_times, spikeClusters=unit_ids, **params)
    #spike_times=np.array(spike_times)
    #fs=np.float16(fs)
    #spike_times_in_frames = (spike_times * Fs).astype(np.int64)
     #si.plot_traces(recording,
    print(f"creating a spikeinterface sorting object")
    spike_times_in_frames=np.int64(spike_times*fs)    
    sampling_frequency=fs
    start_frame=int(t1_s*fs) 
    end_frame=int((t1_s+60*7)*fs)
    sorting = se.NumpySorting.from_times_and_labels(times_list=spike_times, labels_list=unit_ids, sampling_frequency=fs) 
    #sorting = se.NumpySorting.from_times_labels(times_list=spike_times_in_frames,labels_list=unit_ids,sampling_frequency=fs)
    sorting.register_recording(recording_full_concat)
    sorting=sorting.frame_slice(start_frame=start_frame,end_frame=end_frame)
    num_units = sorting.get_num_units();    sorting.set_property(key="quality", values=["good"] * num_units)
    
    # segment_unit_spike_times_list =[]
    # segment_dict={}
    # for _ in range(segment_count):
    #     segment_dict = {}
    #     for index, row in df_irc.iterrows():
    #         unit_id = row['unit_id']
    #         spike_times = row['spike_times']
    #         segment_dict[unit_id] = spike_times  # Store all spike times for now
    #     segment_unit_spike_times_list.append(segment_dict)
   
   #n_by_spike_times=sorting.to_spike_vector()    
    sorting=sorting.to_shared_memory_sorting()
    # sorting_slice_frames = sorting.frame_slice(start_frame=0,
    #                                        end_frame=int(10*sampling_frequency))
    print(f"removing excess_spikes")
    sorting = scur.remove_excess_spikes(sorting, recording)#removes spikes out of recording time
    print(f"removing duplicated_spikes")
    sorting = remove_duplicated_spikes(sorting, method = "keep_first_iterative",censored_period_ms=2) #removes duplicated spikes close to each other
    sorting = sorting.remove_empty_units()#some units may lose all spikes
    print(sorting)
    
    
    
    ## get waveforms#####################################
    
    
    
    milisec_in_samples=int(fs/1000)
    sorting.precompute_spike_trains(False)
    sorting_unit_ids = sorting.unit_ids
    unit_ids=sorting_unit_ids
    print(f"about to create analyzer using {analyzer_mode} mode")
   
    if analyzer_mode=='memory':
        job_kwargs=si.get_best_job_kwargs();
        
        si.set_global_job_kwargs(**job_kwargs)
        #analyzer,sorting=compute_extensions(analyzer,sorting)
        analyzer = create_sorting_analyzer(sorting=sorting, recording=recording,
                                       format="memory", 
                                       sparse=True ,num_spikes_for_sparsity=100,
                                       method='radius',radius_um=40, ms_before= 1.0, ms_after= 1.5,
                                       verbose=True,   **job_kwargs)
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,seed=2205,verbose=True,**job_kwargs)        
        analyzer.compute("noise_levels",**job_kwargs)       
        analyzer.compute("waveforms", ms_before=1, ms_after=1.5, **job_kwargs)
        analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
        analyzer.compute(
                        input="correlograms",
                        window_ms=100.0,
                        bin_ms=2.0,
                        method="numba",
                        verbose=True,
                        **job_kwargs)
        
        analyzer.compute(
                        input="isi_histograms",
                        window_ms=100.0,
                        bin_ms=1.0,
                        method="numba",
                        verbose=True,
                        **job_kwargs) 
 #       unit_ids = [unit_id for unit_id in sorting.unit_ids if 190 <= unit_id <= 250]
        #unit_ids=sorting.unit_ids[::50]
#        unit_ids=sorting.unit_ids[sorting.unit_ids==347]
        #plots
        #import spikeinterface.widgets as sw
        # sw.plot_unit_waveforms(analyzer,unit_ids=unit_ids,same_axis=False,plot_templates=True,ncols=5, max_spikes_per_unit=50,figsize=(40, 40)); plt.savefig('waveforms1.png')             #         # 
        # sw.plot_unit_templates(analyzer, unit_ids=unit_ids, ncols=10,plot_waveforms=True, figsize=(15, 40));plt.savefig('templates2.png')                
        # sw.plot_unit_waveforms_density_map(analyzer, unit_ids=unit_ids, figsize=(14, 8));plt.savefig('waveform_density.png');plt.close('all')
         # not 
    else: #save analyzer to folder
        print("preprocessFuncitons,analyzer_mode!='Binary' ")      
        
        analyzer_folder= Path(f"r{output_folder}")
        print(f"checking if {analyzer_folder} exists")
        print(Path(analyzer_folder).is_dir()==True)
        temp_analyzer_path=analyzer_folder / 'sorting_analyzer_results'
        yes_no='yes'
        if Path(analyzer_folder).is_dir()==True:
            if recompute_anlayzers==True:
                pass
            else:
                yes_no=input(f'an analyzer was previously created for this session at\n {analyzer_folder} \n do you want to recompute and overwrite?  \n (press yes if anything changed with  the sorting)      ').lower()
                
        
        print(f"checking if {analyzer_folder} exists")
        print(Path(analyzer_folder).is_dir()==True)
      
        #check if already exists
        
            
      
            
            
        if yes_no=='yes':
            print(f'creating new anlyzer at {analyzer_folder}')                        
            analyzer,temp_analyzer_path = sf.analyze_results(analyzer_folder,sorting,recording,export_to_phy=False,export_report=False,save_rasters=True)
        else:
            print('loading existing analyzer')
            analyzer_folder.mkdir(parents=True, exist_ok=True)
            temp_analyzer_path= analyzer_folder / 'sorting_analyzer_results'
            analyzer = si.load_sorting_analyzer(temp_analyzer_path)
            
        
    
    
    
    
    #from spikeinterface.curation import remove_redundant_units
#    clean_sorting = remove_redundant_units(
#    sorting,
#    duplicate_threshold=0.9,
   # output = sorting.frame_slice(session_multiplied[0], session_multiplied[1])
    
    #file_path = spikeglx_folder
    #output_folder=paths['preprocessed']
      
    sorting_file_path=Path(fr"{output_folder}\numpy_sorting.npz")
    se.NpzSortingExtractor.write_sorting(sorting, sorting_file_path) 
    print(f'saved sorting as {sorting_file_path}')
    sorting = se.NpzSortingExtractor(sorting_file_path)
   

    
    #w_ts = sw.plot_traces(recording, time_range=(spike_times[0]-0.25,spike_times[0]+0.25),order_channel_by_depth=True,return_scaled=True,clim=(-50,50));plt.savefig('traces.png');plt.close()
    
    # clean_sorting = remove_duplicated_spikes(sorting2, censored_period_ms=0.1)
    

    threads_job_kwargs = dict(pool_engine='thread',n_jobs=1, progress_bar=True,mp_context='spawn',max_threads_per_worker=4)
    threads_job_kwargs=score.fix_job_kwargs(threads_job_kwargs)
    job_kwargs=threads_job_kwargs
    # process_job_kwargs = dict(pool_engine='process',n_jobs=8, progress_bar=True,mp_context='spawn',max_threads_per_worker=1, chunk_memory= '0.5G')
    # process_job_kwargs=score.fix_job_kwargs(process_job_kwargs)
    # si.set_global_job_kwargs(**threads_job_kwargs)
    #job_kwargs=process_job_kwargs;  
    
    job_kwargs=si.get_best_job_kwargs();
    
    si.set_global_job_kwargs(**job_kwargs)
   # si.ensure_chunk_size(recording, other_kwargs)
    print(si.get_global_job_kwargs())
    
    # if analyzer_mode=='binary':
    #     analyzer_folder= Path(fr"{output_folder}\Analyzer_ext")
    #     print(f'creating anlyzer at {analyzer_folder}')
    #     analyzer_folder.mkdir(parents=True, exist_ok=True)
    #     print(f'computing extensions...')
    #    # with MultiTimer(logger, "Total computation time") as mt:#run with timing 
    #     analyzer = create_sorting_analyzer(sorting=sorting, recording=recording,
    #       format="binary_folder", folder=analyzer_folder,overwrite=True,
    #       sparse=False ,num_spikes_for_sparsity=100,
    #       method='radius',radius_um=40, ms_before= 0.5, ms_after= 1,
    #       verbose=True,   **job_kwargs)
    #  #  mt.split("0- analyzer created")
    #     analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,seed=2205,verbose=True,**job_kwargs)
    #   # mt.split("0- random_spikes computed")
    #     analyzer.compute("waveforms", ms_before=1.5, ms_after=1.5, **job_kwargs);             
    #  #  mt.split("1- waveforms computed")    
    #     analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
    #   # mt.split("1- templates computed")            
    #     unit_peak_shifts= score.get_template_extremum_channel_peak_shift(analyzer, peak_sign= 'neg')
    #     sorting=spost.align_sorting(sorting, unit_peak_shifts)
    #  #  mt.split("2- sorting realigned")  
       
    #     analyzer.compute("waveforms", ms_before=0.5, ms_after=1.0, **job_kwargs); 
    #   # mt.split("3- waveforms recomputed")
    #     analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
    #  #  mt.split("4- templates recomputed")  
     
       
    #     analyzer.compute("noise_levels",**job_kwargs)
    #  # mt.split("5- noise_levels computed")      
    #     analyzer.compute("unit_locations", method="monopolar_triangulation",**job_kwargs)
    #  #  mt.split("6- nit_locations computed")      
    #     analyzer.compute("isi_histograms",window_ms=100.0,    bin_ms=1.0,    method="numba",    verbose=True,    **job_kwargs)
    #  #  mt.split("7- isi_histograms computed")      
    #     analyzer.compute("correlograms", window_ms=100.0,        bin_ms=5.0,        method="numba",        verbose=True,        **job_kwargs)
    #   # mt.split("8- correlograms computed")      
    #     analyzer.compute("spike_amplitudes", peak_sign="neg", **job_kwargs)
    #  #  mt.split("12- spike_amplitudes computed")
       
    #     analyzer.compute("principal_components", n_components=3, mode='by_channel_local', whiten=True, **job_kwargs)
    # #   mt.split("9- PCA computed")      
    #     analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
    #  #  mt.split("10- quality_metrics computed")      
    #     analyzer.compute("template_similarity",**job_kwargs)
    #   # mt.split("11- template_similarity computed")
    #     analyzer.save_as(format='binary_folder',folder=analyzer_folder / "saved_analyzer")
            
            
    
    #         # analyzer.compute( input="spike_locations",
    #         # ms_before=0.5,            ms_after=0.5,
    #         # spike_retriever_kwargs=dict(
    #         #     channel_from_template=True,                radius_um=50,                
    #         #     peak_sign="neg"            ),   method="grid_convolution",**job_kwargs)
    #         # mt.split("14- spike_locations computed")           
            
    #     analyzer.compute(input="template_metrics", include_multi_channel_metrics=False,**job_kwargs) ; 
    #  #   mt.split("13- template_metrics computed")           
    #     analyzer.compute(input="template_similarity", method='cosine_similarity',**job_kwargs); 
     #   mt.split("template_similarity computed")            
            #
    
    
    FR = sqm.compute_firing_rates(analyzer)
    rp_contamination, rp_violations = sqm.compute_refrac_period_violations(analyzer)
    keys_over_0_9 = [key for key, value in rp_contamination.items() if value > 0.9]
    import warnings
    warning_message = f"Check units for potential splitting: {keys_over_0_9}"
    warnings.warn(warning_message, category=UserWarning)

    
    
    def process_rp_excluded_units(rp_excluded_unit_ids):
        """
        Returns RP_excluded_unit_ids or None if it's empty.
    
        Args:
            rp_excluded_unit_ids: A list or set of unit IDs, or None.
    
        Returns:
            The original rp_excluded_unit_ids, or None if it was empty.
        """
        if rp_excluded_unit_ids is None:
            return None  # No change if already None
    
        if isinstance(rp_excluded_unit_ids, (list, set)):
            if not rp_excluded_unit_ids:  # Check if empty
                return None
            else:
                return rp_excluded_unit_ids
        else:
          return rp_excluded_unit_ids # if it is not a set or list, return it.
      
    RP_excluded_unit_ids=keys_over_0_9
    RP_excluded_unit_ids=process_rp_excluded_units(RP_excluded_unit_ids)   
    
    return RP_excluded_unit_ids,analyzer,sorting,recording,df_irc#,avg_waveform
#######################################################################################

    def plot_sorting_stuff(analyzer):
         import matplotlib.pyplot as plt
         plt.ion()
         unit_ids = sorting.unit_ids[::10]
         filtered_viSite2Chan = df_irc[df_irc['unit_id'].isin(unit_ids)]['viSite2Chan']
         filtered_viSite2Chan=filtered_viSite2Chan.tolist()
       
         sw.plot_unit_locations(analyzer, figsize=(4, 8))
         sw.plot_amplitudes(analyzer,unit_ids=unit_ids, plot_histograms=True, figsize=(24, 16),max_spikes_per_unit=100);plt.savefig('t.png')
         sw.plot_unit_depths(analyzer,figsize=(4, 8));plt.savefig('t.png')
         sw.plot_all_amplitudes_distributions(analyzer, figsize=(10, 10),unit_ids=unit_ids);plt.savefig('t.png')
         sw.plot_unit_waveforms_density_map(analyzer, unit_ids=unit_ids, figsize=(80, 40));plt.savefig('density.png')
         sw.plot_template_metrics(analyzer,unit_ids);plt.savefig('t.png')
         sw.plot_rasters(analyzer,figsize=(40, 80),);plt.savefig('rasters.png')
        
    
         sw.plot_unit_waveforms(analyzer,   unit_ids=unit_ids, plot_waveforms=True, plot_templates=True, plot_channels=False,
                                ncols=10, scale=1, widen_narrow_scale=0.5, axis_equal=False, max_spikes_per_unit=5, set_title=True,
                                same_axis=False,  scalebar=True,  plot_legend=False, alpha_templates=0.5,);plt.savefig('t.png')
         
         sw.plot_spikes_on_traces(analyzer,unit_ids=unit_ids,order_channel_by_depth=True,figsize=(40, 80),mode='line',show_channel_ids=True,return_scaled=True,seconds_per_row=0.05);plt.savefig('t.png')
         plt.close('all')
 
def plot_analyzer_stats(keep_unit_ids,analyzer,sorting,recording,output_path):
    import numpy as np        
    import matplotlib.pyplot as plt
    import spikeinterface.full as si
    from spikeinterface import create_sorting_analyzer, load_sorting_analyzer
    import spikeinterface.extractors as se
    from spikeinterface.exporters import export_report
    from spikeinterface.curation import remove_duplicated_spikes,remove_redundant_units,get_potential_auto_merge    
    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms
    from spikeinterface.qualitymetrics import compute_quality_metrics
    import spikeinterface.qualitymetrics as sqm
    import spikeinterface.postprocessing as spost
    import spikeinterface.core as score 
    import matplotlib.pyplot as plt
    import spikeinterface.widgets as sw 
    import spikeinterface.curation as scur
    job_kwargs=si.get_best_job_kwargs();
    
    
    si.set_global_job_kwargs(**job_kwargs)
    analyzer = analyzer.select_units(keep_unit_ids)
    sorting = sorting.select_units(keep_unit_ids)
    
    FR = sqm.compute_firing_rates(analyzer)
    rp_contamination, rp_violations = sqm.compute_refrac_period_violations(analyzer)      
    
    with MultiTimer(logger, "Total computation time") as mt:
        analyzer.compute(
                input="isi_histograms",
                window_ms=500.0,
                bin_ms=1.0,
                method="numba",
                verbose=True,
                **job_kwargs) 
        mt.split("isi_histograms")           
    print('computing cross-correlograms')
    with MultiTimer(logger, "Total computation time") as mt:
        analyzer.compute(
                input="correlograms",
                window_ms=50.0,
                bin_ms=1.0,
                method="numba",
                verbose=True,
                **job_kwargs)
        mt.split("correlograms") 
    ISI_ext = analyzer.get_extension("isi_histograms")
    ISIdata = ISI_ext.get_data()  
    
    CC_ext = analyzer.get_extension("correlograms")
    CCdata = CC_ext.get_data()
    analyzer.compute(input="unit_locations", method="monopolar_triangulation",**job_kwargs); 
    
        
    
       
    def plot_ISI_and_autocorr_sorted_FR(sorting, analyzer, FR, rp_contamination, ISIdata, CCdata, output_path):
        import matplotlib
        #matplotlib.use("TkAgg")  # Keep using a GUI backend
        import matplotlib.pyplot as plt
        import spikeinterface.qualitymetrics as sqm
        import spikeinterface.widgets as sw 
        import cupy as cp
        import numpy as np
        import matplotlib.pyplot as plt
        import numpy as np
        import spikeinterface.qualitymetrics as sqm
        from tqdm import tqdm  # Import tqdm for progress bar
        import os
    
        num_units = len(sorting.unit_ids)
        num_segments = sorting.get_num_segments()
        fs = sorting.sampling_frequency
        window_ms = float(1000.0)
        bin_ms = float(1.0)
        autocorr_window_ms = 500.0
        autocorr_bin_ms = 1.0
        FR_binsize = 0.02  # 20 ms
    
        histogram_data = {}
        autocorr_data = {}
        FR_all_units = {}
        i = -1
    
        def handle_zero_div_bins(bin_counts):
            if np.sum(bin_counts) == 0:
                bin_counts = np.zeros_like(bin_counts)
            else:
                bin_counts = np.nan_to_num(bin_counts)
    
            if np.all(np.isnan(bin_counts)):
                bin_counts = np.zeros_like(bin_counts)
            else:
                max_bin_counts = np.nanmax(bin_counts)
                if max_bin_counts == 0:
                    bin_counts = np.zeros_like(bin_counts)
                else:
                    bin_counts = bin_counts / max_bin_counts
            return bin_counts
    
        def calculate_firing_rate(st, bin_size=0.1):
            if len(st) == 0:
                return np.array()
            max_time = np.nanmax(st)
            bins = np.arange(0, max_time + bin_size, bin_size)
            spike_counts, _ = np.histogram(st, bins=bins)
            firing_rate = spike_counts / (bin_size)
            return np.array(firing_rate)
    
        for unit_id in tqdm(sorting.unit_ids, desc="Calculating ISI and Autocorrelation for all units"):
            i += 1
            bin_counts = ISIdata[0][i]
            bin_counts = handle_zero_div_bins(bin_counts)
            bin_edges = ISIdata[1]
            histogram_data[unit_id] = (bin_edges[:-1], bin_counts)
    
            autocorr_bins = CCdata[1]
            autocorr = CCdata[0][i, i, :]
            autocorr = handle_zero_div_bins(autocorr)
            autocorr_data[unit_id] = (autocorr_bins[:-1], autocorr)
    
            st = np.array(sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0) / fs)
            FR_vector = calculate_firing_rate(st, bin_size=FR_binsize)
            FR_all_units[unit_id] = FR_vector / np.nanmax(FR_vector)
    
        def get_max_bin_index(unit_id):
            return np.argmax(histogram_data[unit_id][1])
    
        def plot_sorted_data(sorted_unit_ids, histogram_data, autocorr_data, FR, rp_contamination, FR_all_units, sort_type):
            plt.ion()
            plt.rcParams.update({
                'font.size': 36,
                'axes.titlesize': 18,
                'axes.labelsize': 26,
                'xtick.labelsize': 24,
                'ytick.labelsize': 14,
                'legend.fontsize': 10,
                'figure.titlesize': 32
            })
            
            figure, axes = plt.subplots(1, 3, figsize=(50, 30))  # Add one more subplot for FR
            ax_isi, ax_autocorr, ax_fr = axes
    
            num_units = len(sorted_unit_ids)
    
            max_isi_bins = max(len(histogram_data[unit_id][0]) for unit_id in sorted_unit_ids)
            max_autocorr_bins = max(len(autocorr_data[unit_id][0]) for unit_id in sorted_unit_ids)
            max_fr_bins = max(len(FR_all_units[unit_id]) for unit_id in sorted_unit_ids)
    
            isi_bin_edges = np.full((num_units, max_isi_bins), np.nan)
            isi_bin_counts = np.full((num_units, max_isi_bins), np.nan)
    
            autocorr_bin_edges = np.full((num_units, max_autocorr_bins), np.nan)
            autocorr_bin_counts = np.full((num_units, max_autocorr_bins), np.nan)
    
            fr_bin_edges = np.full((num_units, max_fr_bins), np.nan)
            fr_bin_counts = np.full((num_units, max_fr_bins), np.nan)
    
            for i, unit_id in enumerate(tqdm(sorted_unit_ids, desc='plotting units')):
                bin_edges, bin_counts = histogram_data[unit_id]
                autocorr_bins, autocorr = autocorr_data[unit_id]
                fr_bins = np.arange(0, len(FR_all_units[unit_id]) * FR_binsize, FR_binsize)
                fr_counts = FR_all_units[unit_id]
    
                isi_bin_edges[i, :len(bin_edges)] = bin_edges
                isi_bin_counts[i, :len(bin_counts)] = bin_counts
    
                autocorr_bin_edges[i, :len(autocorr_bins)] = autocorr_bins
                autocorr_bin_counts[i, :len(autocorr)] = autocorr
    
                fr_bin_edges[i, :len(fr_bins)] = fr_bins
                fr_bin_counts[i, :len(fr_counts)] = fr_counts
    
            for i in range(num_units):
                valid_isi = ~np.isnan(isi_bin_edges[i])
                ax_isi.plot(isi_bin_edges[i, valid_isi], isi_bin_counts[i, valid_isi] + i, color='black', linewidth=0.5)
    
                valid_autocorr = ~np.isnan(autocorr_bin_edges[i])
                ax_autocorr.plot(autocorr_bin_edges[i, valid_autocorr], autocorr_bin_counts[i, valid_autocorr] + i, color='black', linewidth=0.5)
    
                valid_fr = ~np.isnan(fr_bin_edges[i])
                ax_fr.plot(fr_bin_edges[i, valid_fr], fr_bin_counts[i, valid_fr] + i, color='black', linewidth=0.5)
    
            ax_isi.axvline(x=2.0, color='red', linewidth=2)
    
            for i, unit_id in enumerate(sorted_unit_ids):
                fstring = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
                fstring_color = 'red' if rp_contamination[unit_id] > 0.9 else 'black'
                ax_isi.text(0.1, i, fstring, va='top', color=fstring_color, fontsize=8)
                ax_fr.text(0.1, i, fstring, va='top', color=fstring_color, fontsize=8)
    
            ax_isi.set_xscale('log')
            ax_isi.set_xlabel("ISI (log ms)")
            ax_isi.set_xlim(left=1)
            ax_isi.set_ylim(0, num_units)
            ax_isi.set_yticks([]); ax_isi.set_ylabel("")
    
            ax_autocorr.set_xlabel("Time (ms)")
            ax_autocorr.set_xlim(-100, 100)
            ax_autocorr.set_ylim(0, num_units)
            ax_autocorr.set_yticks([]);ax_autocorr.set_ylabel("")

            ax_fr.set_xlabel("Time (s)")  # Changed to seconds
            ax_fr.set_ylim(0, num_units)
            ax_fr.set_yticks([]); ax_fr.set_ylabel("")

    
            ax_isi.spines['top'].set_visible(False)
            ax_isi.spines['right'].set_visible(False)
            ax_isi.spines['left'].set_visible(False)
    
            ax_autocorr.spines['top'].set_visible(False)
            ax_autocorr.spines['right'].set_visible(False)
            ax_autocorr.spines['left'].set_visible(False)
    
            ax_fr.spines['top'].set_visible(False)
            ax_fr.spines['right'].set_visible(False)
            ax_fr.spines['left'].set_visible(False)
    
            figure.tight_layout()
            figure.suptitle(f"ISI, Autocorrelation, and Firing Rate (Sorted by {sort_type})")
            plt.show()
            figfilename = f'ISIs_autocorr_FR_sorted_by_{sort_type.replace(" ", "_")}.png'
            full_path = os.path.join(output_path, figfilename)
            plt.savefig(full_path)
            figfilename = f'ISIs_autocorr_FR_sorted_by_{sort_type.replace(" ", "_")}.svg'
            full_path = os.path.join(output_path, figfilename)
            plt.savefig(full_path)
    
            plt.close('all')
            print(f'saved sorted by {sort_type}')
    
        sorted_unit_ids_fr = sorted(sorting.unit_ids, key=lambda unit_id: FR[unit_id])
        sorted_unit_ids_isi = sorted(sorting.unit_ids, key=get_max_bin_index)
    
      
        plot_sorted_data(sorted_unit_ids_fr, histogram_data, autocorr_data, FR, rp_contamination, FR_all_units, "FR")      
        plot_sorted_data(sorted_unit_ids_isi, histogram_data, autocorr_data, FR, rp_contamination, FR_all_units, "Max ISI Bin")
    plot_ISI_and_autocorr_sorted_FR(sorting, analyzer, FR, rp_contamination, ISIdata, CCdata, output_path)
    print(f'saved to {output_path}')
    
      
    def plot_ISI_and_autocorr_sorted(sorting, analyzer, FR, rp_contamination,ISIdata,CCdata,output_path):
            import matplotlib
            #matplotlib.use("TkAgg")  # Keep using a GUI backend
            import matplotlib.pyplot as plt
            import spikeinterface.qualitymetrics as sqm
            import spikeinterface.widgets as sw 
            import cupy as cp
            import numpy as np
            import matplotlib.pyplot as plt
            import numpy as np
            import spikeinterface.qualitymetrics as sqm
            from tqdm import tqdm  # Import tqdm for progress bar
            import os
        
            num_units = len(sorting.unit_ids)
            num_segments = sorting.get_num_segments()
            fs = sorting.sampling_frequency
            window_ms = float(1000.0)
            bin_ms = float(1.0)
            autocorr_window_ms = 500.0
            autocorr_bin_ms = 1.0
            FR_binsize= 0.02 #20 ms
        
            histogram_data = {}
            autocorr_data = {}
            FR_all_units = {}
            i=-1
            def handle_zero_div_bins(bin_counts):
                #handle numpy histogram warnings
                if np.sum(bin_counts) == 0:
                    bin_counts = np.zeros_like(bin_counts)
                else:
                    bin_counts = np.nan_to_num(bin_counts)
                
                #handle nanmax warnings.
                if np.all(np.isnan(bin_counts)):
                    bin_counts = np.zeros_like(bin_counts)
                else:
                    max_bin_counts = np.nanmax(bin_counts)
                    if max_bin_counts == 0:
                        bin_counts = np.zeros_like(bin_counts)
                    else:
                        bin_counts = bin_counts / max_bin_counts  
                return bin_counts
            def calculate_firing_rate(st, bin_size=0.1):
                """
                Calculates the firing rate of a neuron over time in bins.
            
                Parameters:
                st (ndarray):  Spike times in seconds.
                bin_size (float, optional): Bin size in seconds. Default is 0.1 (100 ms).
            
                Returns:
                ndarray: Firing rate over time in bins (Hz).
                """
            
                if len(st) == 0:
                    return np.array()  # Handle case with no spikes
            
                max_time = np.nanmax(st)
                bins = np.arange(0, max_time + bin_size, bin_size)  # Create bin edges
                spike_counts, _ = np.histogram(st, bins=bins)  # Count spikes in each bin
                firing_rate = spike_counts / (bin_size)  # Calculate firing rate (Hz)
                return np.array(firing_rate)

            
            for unit_id in tqdm(sorting.unit_ids, desc="Calculating ISI and Autocorrelation for all units"):  
                i+=1
                bin_counts=ISIdata[0][i]
                bin_counts=handle_zero_div_bins(bin_counts)
                bin_edges=ISIdata[1]
                histogram_data[unit_id] = (bin_edges[:-1], bin_counts)
                
                autocorr_bins=CCdata[1]
                autocorr=CCdata[0][i, i, :]  
                autocorr=handle_zero_div_bins(autocorr)
                autocorr_data[unit_id] = (autocorr_bins[:-1], autocorr)
                
                st = np.array(sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0) / fs)
                FR_vector=calculate_firing_rate(st, bin_size=FR_binsize)
                FR_all_units[unit_id]=FR_vector
          
            # Function to get max ISI bin index
            def get_max_bin_index(unit_id):
                return np.argmax(histogram_data[unit_id][1])
            
            
            def plot_sorted_data(sorted_unit_ids, histogram_data, autocorr_data, FR, rp_contamination, sort_type):
                plt.ion()
                plt.rcParams.update({
                    'font.size': 36,
                    'axes.titlesize': 18,
                    'axes.labelsize': 26,
                    'xtick.labelsize': 24,
                    'ytick.labelsize': 14,
                    'legend.fontsize': 10,
                    'figure.titlesize': 32
                })
                figure, axes = plt.subplots(1, 2, figsize=(15, 50))  # Adjust figsize as needed
                ax_isi, ax_autocorr = axes
            
                num_units = len(sorted_unit_ids)
            
                # Pre-allocate NumPy arrays for line plotting
                max_isi_bins = max(len(histogram_data[unit_id][0]) for unit_id in sorted_unit_ids)
                max_autocorr_bins = max(len(autocorr_data[unit_id][0]) for unit_id in sorted_unit_ids)
            
                isi_bin_edges = np.full((num_units, max_isi_bins), np.nan)
                isi_bin_counts = np.full((num_units, max_isi_bins), np.nan)
            
                autocorr_bin_edges = np.full((num_units, max_autocorr_bins), np.nan)
                autocorr_bin_counts = np.full((num_units, max_autocorr_bins), np.nan)
            
                # Populate data
                for i, unit_id in enumerate(tqdm(sorted_unit_ids, desc='plotting units')):
                  
                    
                    bin_edges, bin_counts = histogram_data[unit_id]
                    autocorr_bins, autocorr = autocorr_data[unit_id]
            
                    # Populate line data
                    isi_bin_edges[i, :len(bin_edges)] = bin_edges
                    isi_bin_counts[i, :len(bin_counts)] = bin_counts
            
                    autocorr_bin_edges[i, :len(autocorr_bins)] = autocorr_bins
                    autocorr_bin_counts[i, :len(autocorr)] = autocorr
            
                # Plot lines 
                for i in range(num_units):
                    valid_isi = ~np.isnan(isi_bin_edges[i])
                    ax_isi.plot(isi_bin_edges[i, valid_isi], isi_bin_counts[i, valid_isi] + i, color='black', linewidth=0.5)  # Offset by unit index
            
                    valid_autocorr = ~np.isnan(autocorr_bin_edges[i])
                    ax_autocorr.plot(autocorr_bin_edges[i, valid_autocorr], autocorr_bin_counts[i, valid_autocorr] + i, color='black', linewidth=0.5) # Offset by unit index
            
                # Plot vertical red line
                ax_isi.axvline(x=2.0, color='red', linewidth=2)
            
                # Plot text annotations
                for i, unit_id in enumerate(sorted_unit_ids):
                    fstring = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
                    fstring_color = 'red' if rp_contamination[unit_id] > 0.9 else 'black'
                    ax_isi.text(0.1, i, fstring, va='top', color=fstring_color, fontsize=8)
                   
            
                # Set axis limits to accommodate all plots and text
                ax_isi.set_xscale('log')
                ax_isi.set_xlabel("ISI (log ms)")
                ax_isi.set_xlim(left=1)
                ax_isi.set_ylim(0, num_units)  # Set y-axis limits
                ax_isi.set_yticks([]);ax_isi.set_ylabel("")
                #ax_isi.set_ylabel("Units")  # Label y-axis
            
                ax_autocorr.set_xlabel("Time (ms)")
                ax_autocorr.set_xlim(-autocorr_window_ms / 2, autocorr_window_ms / 2) #set x limit for autocorr
                ax_autocorr.set_ylim(0, num_units) # Set y-axis limits
                ax_autocorr.set_yticks([]);ax_autocorr.set_ylabel("")
                ax_autocorr.set_xlim(-500, 500)
                #ax_autocorr.set_ylabel("Units")
            
                ax_isi.spines['top'].set_visible(False)
                ax_isi.spines['right'].set_visible(False)
                ax_isi.spines['left'].set_visible(False)
            
                ax_autocorr.spines['top'].set_visible(False)
                ax_autocorr.spines['right'].set_visible(False)
                ax_autocorr.spines['left'].set_visible(False)
            
                figure.tight_layout()
                figure.suptitle(f"ISI and Autocorrelation (Sorted by {sort_type})")
                plt.show()
                figfilename= f'ISIs_and_autocorr_sorted_by_{sort_type.replace(" ", "_")}.png'                
                full_path = os.path.join(output_path, figfilename)
                plt.savefig(full_path)
                figfilename= f'ISIs_and_autocorr_sorted_by_{sort_type.replace(" ", "_")}.svg'
                full_path = os.path.join(output_path, figfilename)
                plt.savefig(full_path)
                
                plt.close('all')
                print(f'saved sorted by {sort_type}')

            
         
            
            sorted_unit_ids_fr = sorted(sorting.unit_ids, key=lambda unit_id: FR[unit_id])            
            sorted_unit_ids_isi = sorted(sorting.unit_ids, key=get_max_bin_index)
                
            with MultiTimer(logger, "Total computation time") as mt:
               # plot_plotly_data(sorted_unit_ids_fr, histogram_data, autocorr_data, FR, rp_contamination, "FR")
#                mt.split('plot by firing rate with plotly')
                plot_sorted_data(sorted_unit_ids_fr, histogram_data, autocorr_data, FR, rp_contamination, "FR")
                mt.split('plot by max FR')
                plot_sorted_data(sorted_unit_ids_isi, histogram_data, autocorr_data, FR, rp_contamination, "Max ISI Bin")
                mt.split('plot by max bin')
        
    #plot_ISI_and_autocorr_sorted(sorting, analyzer, FR, rp_contamination,ISIdata,CCdata,output_path)        
    
    # import spikeinterface.widgets as sw  
    # sw.plot_isi_distribution(sorting,window_ms=1000,figsize=(20, 50))
    
    # figfilename= f'ISIs.png'
    # full_path = os.path.join(output_path, figfilename)
    # plt.savefig(full_path)
     

    
    
     

    
   # sorting=sorting.threads_job_kwargs(n_jobs=job_kwargs['n_jobs'])
   
    def compute_extensions(analyzer,sorting):
      with MultiTimer(logger, "Total computation time") as mt:#run with timing  
            #threads_job_kwargs = dict(pool_engine='thread',n_jobs=1, progress_bar=False,mp_context='spawn',max_threads_per_worker=6)
           # job_kwargs=score.fix_job_kwargs(threads_job_kwargs)     

            analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=100, seed=2205) 
            mt.split("random_spikes complete")           
            analyzer.compute("noise_levels",verbose=True,**job_kwargs)           
            mt.split("noise_levels computed")
            analyzer.compute("waveforms", ms_before=0.5, ms_after=1.0, **job_kwargs);             
            mt.split("waveforms computed")    
            analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
            mt.split("templates computed")            
            unit_peak_shifts= score.get_template_extremum_channel_peak_shift(analyzer, peak_sign= 'neg')
            sorting=spost.align_sorting(sorting, unit_peak_shifts)
            mt.split("waveforms recomputed")
            analyzer.compute("waveforms", ms_before=0.5, ms_after=1.0, **job_kwargs); 
            analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
            mt.split("templates recomputed")            
            
            #analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True,**job_kwargs); 
#            mt.split("PCA complete")
            analyzer.compute(input="spike_amplitudes", peak_sign="neg",**job_kwargs); 
            mt.split("spike_amplitudes computed")
            # analyzer.compute( input="spike_locations",
            # ms_before=0.5,            ms_after=0.5,
            # spike_retriever_kwargs=dict(
            #     channel_from_template=True,                radius_um=50,                
            #     peak_sign="neg"            ),   method="grid_convolution",**job_kwargs)
            # mt.split("spike_locations computed")           
            
            analyzer.compute(input="template_metrics", include_multi_channel_metrics=False,**job_kwargs) ; 
            mt.split("template_metrics computed")           
            analyzer.compute(input="template_similarity", method='cosine_similarity',**job_kwargs); 
            mt.split("template_similarity computed")            
            #
            analyzer.compute(input="unit_locations", method="monopolar_triangulation",**job_kwargs); 
            mt.split("unit_locations computed")
            
            print(analyzer)
            analyzer.compute(
                input="isi_histograms",
                window_ms=500.0,
                bin_ms=1.0,
                method="numba",
                **job_kwargs)            
            mt.split("isi_histograms computed")
         
            analyzer.compute(
                input="correlograms",
                window_ms=50.0,
                bin_ms=1.0,
                method="numba",
                **job_kwargs)      
            print(analyzer)
            mt.split("cross-correlograms computed")
            return analyzer,sorting
           # metric_names=['num_spikes','firing_rate', 'snr', 'isi_violation', 'rp_violation','amplitude_cutoff']#,'sd_ratio','amplitude_cv']
            #metric_names=sqm.get_quality_metric_list()
            #sqm.get_quality_pca_metric_list()
            #analyzer.compute("amplitude_scalings",**job_kwargs)
           # analyzer.compute("quality_metrics",metric_names=metric_namesskip_pc_metrics=True,**job_kwargs); 

            
            analyzer.compute(input={"principal_components": dict(n_components=3, mode="by_channel_global", whiten=True),
                                "quality_metrics": dict(skip_pc_metrics=False)})
            mt.split("quality_metrics computed")    
            
    # if (overwrite==True) or  (sum(f.stat().st_size for f in analyzer_folder.rglob('*') if f.is_file())==0):      
    #     print(f' \n\n\n creating_sorting analyzer at {analyzer_folder}  \n\n\n')
    #    # with MultiTimer(logger, "Total computation time") as mt:
        
    #     analyzer = create_sorting_analyzer(sorting=sorting,
    #                                     recording=recording,
    #                                     format="binary_folder",
    #                                     #format="memory",                                       
    #                                     folder=analyzer_folder,
    #                                     return_scaled=True, # this is the default to attempt to return scaled
    #                                     sparse=True ,
    #                                     overwrite=overwrite,                                       
    #                                     verbose=True,
    #                                     num_spikes_for_sparsity=100,
    #                                     ms_before= 0.5,
    #                                     ms_after= 1,
    #                                     method='radius',
    #                                     radius_um=40,
    #                                     #noise_levels= si.get_noise_levels(recording, return_scaled=True,**job_kwargs),
    #                                    **job_kwargs
    #                                     )
    #     # job_kwargs = dict(
        #     n_jobs=1,
        #     progress_bar=False,
        #     verbose=True,
        #     chunk_duration=1.0,
        # )
      
      #  ext=analyzer.get_computable_extensions()
        
       # analyzer.compute(['random_spikes'],verbose=True,**job_kwargs)
       #analyzer= compute_extensions(analyzer)
        #     mt.split('analyzer created!')
    #else:
       
    #     print(f' \n\n\n loading existing analyzer from {analyzer_folder}  \n\n\n')
    # analyzer = load_sorting_analyzer(analyzer_folder,format='binary_folder')
   
      
    #  # sliding_rp_violations=sqm.compute_sliding_rp_violations(analyzer)
   
      
    
    # print(f'created analyzer object at {analyzer_folder}')
    # print(f'computing anlyzer extensions')
    
   
        #si.export_report(analyzer, f"{analyzer_folder}\\report2", format='png',remove_if_exists=True,**job_kwargs)
        
   # def compute_analyzer_extentions(folder):
      # ext=analyzer.get_computable_extensions()
       
     # analyzer.compute_several_extensions(ext, **job_kwargs)
       
#       #compute only extensions needed for sparsity using memory mode
    
#            analyzer.compute("quality_metrics",metric_names=sqm.get_quality_metric_list(),skip_pc_metrics=True,**job_kwargs); 
    #mt.split("quality_metrics computed")
#            metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names,**job_kwargs);print(metrics)
            #analyzer.compute(input="quality_metrics", metric_names=["firing_rate", "snr", "amplitude_cutoff",  "isolation_distance","d_prime"], skip_pc_metrics=True)
           
            #analyzer.save_as(format="binary_folder",folder=analyzer_folder / 'computed')
            #mt.split("analyzer saved")
            
            
            
            
   
            
          
            
            #save_sorting_as_phy(analyzer,analyzer_folder,sparsity)
            
            #recompute extensions
            
    def plot_widgets(analyzer,sorting,unit_ids_subset=None):
        import matplotlib
       # matplotlib.use("TkAgg")  # Keep using a GUI backend
        import matplotlib.pyplot as plt
        import spikeinterface.widgets as sw       
        plt.ion()  # Turn on interactive mode
        plt.rcParams.update({
        'font.size': 12,            # controls default text sizes
        'axes.titlesize': 12,       # fontsize of the axes title
        'axes.labelsize': 12,       # fontsize of the x and y labels
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18      # fontsize of the figure title
        })                           
        
        if unit_ids_subset==None:
            unit_ids = [unit_id for unit_id in sorting.unit_ids if 190 <= unit_id <= 250]
            #unit_ids=sorting.unit_ids[::50]
        else:
            unit_ids=sorting.unit_ids[unit_ids_subset]
        #plots
        sw.plot_unit_waveforms(analyzer,unit_ids=unit_ids,same_axis=False,plot_templates=True,ncols=10, max_spikes_per_unit=50,figsize=(40, 40)); plt.savefig('waveforms.png')     
        sw.plot_unit_templates(analyzer, unit_ids=unit_ids, ncols=10,plot_waveforms=False, figsize=(15, 40));plt.savefig('templates.png')                
        sw.plot_unit_waveforms_density_map(analyzer, unit_ids=unit_ids, figsize=(14, 8));plt.savefig('waveform_density.png');plt.close('all')
        #sw.plot_unit_summary(analyzer,unit_id=unit_ids,figsize=(16, 16));plt.savefig('unit_summary.png')
        #sw.plot_unit_probe_map(analyzer, unit_ids=unit_ids,colorbar=False, figsize=(20, 8));            plt.show(); plt.savefig('probe_map.png')
        sw.plot_isi_distribution(sorting,window_ms=1000,figsize=(20, 50));plt.savefig('ISIs.png')
        sw.plot_crosscorrelograms(sorting,window_ms=500,figsize=(40, 40));plt.savefig('CCs.png')
        sw.plot_amplitudes(analyzer,unit_ids=unit_ids, plot_histograms=True, plot_legend=False ,figsize=(15, 20));plt.savefig('amplitudes.png')
        sw.plot_all_amplitudes_distributions(analyzer, figsize=(20, 20));plt.savefig('plot_all_amplitudes_distributions.png')
        sw.plot_unit_locations(analyzer,unit_ids=unit_ids, figsize=(16, 16)); plt.show();plt.savefig('unit_locations.png')                 
        sw.plot_unit_depths(analyzer, figsize=(10, 10));plt.tight_layout(rect=[0, 0, 1, 0.97]);plt.savefig('unitdepths.png')
        sw.plot_quality_metrics(analyzer,figsize=(20, 20)); plt.tight_layout(rect=[0, 0, 1, 0.97]);plt.show();plt.savefig('waveform_metrics.png')
        sw.plot_template_metrics(analyzer,figsize=(20, 20)); plt.tight_layout(rect=[0, 0, 1, 0.97]);plt.show();plt.savefig('template_metrics.png')
        sw.plot_template_similarity(analyzer,figsize=(16, 16)); plt.tight_layout(rect=[0, 0, 1, 0.97]);plt.show();plt.savefig('template_similiarity.png')        
        w =sw. plot_traces(recording=recording, backend="matplotlib")
        #sw.plot_spikes_on_traces(analyzer,order_channel_by_depth=True,show_channel_ids=True,time_range=time_range,channel_ids=['imec0.ap#AP81'],scale=1,with_colorbar=False,mode='line',figsize=(10, 1000));plt.show();plt.savefig('traces.png') ;plt.close('all')
        w =sw. plot_traces(recording=recording, backend="matplotlib")
        #sw.plot_peak_activity(recording,peaks)
       # sw.plot_sorting_summary(analyzer,figsize=(16, 16));plt.savefig('sorting_summary.png')
        plt.close('all')
       # sw.plot_widgets(analyzer,sorting,unit_ids_subset=None);;plt.show();plt.savefig('plot_widgets.png')        
        
    # def plot_waveforms(self, analyzer):
    #     analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
    #     waveform_ext = analyzer.get_extension("waveforms")
    #     data = waveform_ext.get_data()  # (timeseries,spike,cluster)
    #     nbefore = waveform_ext.nbefore
    #     clusters = data.shape[-1]
    #     #cols, rows = calculate_subplot_layout(clusters)
    #     cols=int(len(sorting.unit_ids)/4)
    #     rows=int(len(sorting.unit_ids)/4)
    #     fig, axs = plt.subplots(cols, rows, figsize=(cols *1, rows * 1) ,layout="constrained")

    #     try:
    #         axs = axs.flatten()
    #     except AttributeError:
    #         axs = [axs]

    #     for cluster in np.arange(clusters):
    #         ax = axs[cluster]
    #         waveform = data[:, :, cluster] * DELTA_Y
    #         X = (
    #             DELTA_X
    #             * (np.arange(waveform.shape[1]) - nbefore)
    #             / waveform.shape[1]
    #             * 0.7
    #         )
    #         ax.plot(X, waveform.T, linewidth=1, color="indigo", alpha=0.05)
    #         ax.plot(
    #             X,
    #             np.mean(waveform.T, axis=1),
    #             linewidth=1.2,
    #             color="magenta",
    #             alpha=0.9,
    #         )
    #         ax.set_title(f"Putative neuron {cluster + 1}", fontweight="bold")
    #         ax.spines["right"].set_visible(False)
    #         ax.spines["top"].set_visible(False)

    #     fig.tight_layout()
    #     plt.show()
    
   

    # def plot_ISI_single_sorted_one_column(sorting, analyzer,FR,rp_contamination):
    #     import matplotlib
    #     #matplotlib.use("TkAgg")  # Keep using a GUI backend
    #     import matplotlib.pyplot as plt
    #     import spikeinterface.qualitymetrics as sqm
    #     import spikeinterface.widgets as sw 
    #     import cupy as cp
    #     import numpy as np
    #     plt.ion()  # Turn on interactive mode
    #     plt.rcParams.update({
    #     'font.size': 36,            # controls default text sizes
    #     'axes.titlesize': 18,       # fontsize of the axes title
    #     'axes.labelsize': 26,       # fontsize of the x and y labels
    #     'xtick.labelsize': 24,
    #     'ytick.labelsize': 14,
    #     'legend.fontsize': 10,
    #     'figure.titlesize': 32      # fontsize of the figure title
    #     })
    #     def cupy_autocorr(all_times_ms, autocorr_window_ms=500,autocorr_bins=2):
    #         import cupy as cp
    #         import numpy as np
           
       
    #         all_times_ms_gpu = cp.array(all_times_ms)
         
    #         diffs = all_times_ms_gpu[:, None] - all_times_ms_gpu[None, :]
    #         diffs = diffs.flatten()
    #         diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
    #         autocorr_gpu = cp.histogram(diffs, bins=len(diffs))[0]        
           
    #         autocorr_gpu= autocorr_gpu/ cp.nanmax(autocorr_gpu) if cp.nanmax(autocorr_gpu) > 0 else autocorr_gpu #normalize
    #         autocorr_cpu = autocorr_gpu.get()  # Copy from GPU to CPU
    #         return autocorr_cpu
       
    #     num_units = len(sorting.unit_ids)
    #     figure, axes = plt.subplots(num_units, 2, figsize=(40, 60))  # 2 columns

    #     num_segments = sorting.get_num_segments()
    #     fs = sorting.sampling_frequency
    #     window_ms = float(1000.0)
    #     bin_ms = float(1.0)
    #     autocorr_window_ms = 500.0
    #     autocorr_bin_ms = 2.0
    #     #orrelograms = analyzer.get_extension("correlograms").get_data()
    #     #isi_histograms = analyzer.get_extension("isi_histograms").get_data()
        
    #     histogram_data = {}

    #     for i, unit_id in enumerate(sorting.unit_ids):
    #        print(f"{len(sorting.unit_ids)-i}")
    #        #spiketimes = sorting.get_unit_spike_train(unit_id=int(unit_id)) / sorting.sampling_frequency
    
    #        bins = np.arange(0, window_ms, bin_ms)
    #        bin_counts = None
    
    #        all_times_ms = []
    
    #        for segment_index in range(num_segments):
    #            times_ms = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index) / fs * 1000.0
    #            isi = np.diff(times_ms)
    #            all_times_ms.extend(times_ms)
    
    #            bin_counts_, bin_edges = np.histogram(isi, bins=bins, density=True)
    #            if segment_index == 0:
    #                bin_counts = bin_counts_
    #            else:
    #                bin_counts += bin_counts_
    
    #        bin_counts = np.nan_to_num(bin_counts)
    #        bin_counts = bin_counts / np.nanmax(bin_counts)
    
    #        histogram_data[unit_id] = (bin_edges[:-1], bin_counts)
    
    #        # Optimized Autocorrelation calculation
    #        #autocorr = np.zeros_like(autocorr_bins[:-1])
    
    #        #diffs = all_times_ms[:, None] - all_times_ms[None, :]
    #        #diffs = diffs.flatten()
    #        #diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
    #        #autocorr = np.histogram(diffs, bins=autocorr_bins)[0]
    
    #        #autocorr = autocorr / np.nanmax(autocorr) if np.nanmax(autocorr) > 0 else autocorr #normalize
           
    #        # Optimized Autocorrelation calculation with chunking
    #        all_times_ms_gpu = cp.array(all_times_ms)
    #        autocorr_bins = np.arange(-autocorr_window_ms / 2, autocorr_window_ms / 2, autocorr_bin_ms)
    #        autocorr = cp.zeros_like(cp.array(autocorr_bins[:-1]))
        
    #        chunk_size = 2000  # Adjust chunk size based on available GPU memory
    #        num_chunks = int(cp.ceil(len(all_times_ms_gpu) / chunk_size))
    #        #Clear the memory pool
    #        cp._default_memory_pool.free_all_blocks()

    #     # Optionally, you can also clear the pinned memory pool
    #        cp._default_pinned_memory_pool.free_all_blocks()
    #        for chunk_idx in range(num_chunks):
    #           start_idx = chunk_idx * chunk_size
    #           end_idx = min((chunk_idx + 1) * chunk_size, len(all_times_ms_gpu))
    #           chunk = all_times_ms_gpu[start_idx:end_idx]
        
    #           diffs = all_times_ms_gpu[:, None] - chunk[None, :]
    #           diffs = diffs.flatten()
    #           diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
    #           autocorr += cp.histogram(diffs, bins=autocorr_bins)[0]
    #        autocorr[int(len(autocorr_bins) / 2)]=0
    #        autocorr = autocorr / cp.nanmax(autocorr) if cp.nanmax(autocorr) > 0 else autocorr

    #        # autocorr_bins = np.arange(-autocorr_window_ms / 2, autocorr_window_ms / 2, autocorr_bin_ms)
    #        # autocorr = cp.zeros_like(autocorr_bins[:-1])
    #        # all_times_ms_gpu = cp.array(all_times_ms)
    #        # autocorr_window_ms=500
          
    #        # diffs = all_times_ms_gpu[:, None] - all_times_ms_gpu[None, :]
    #        # diffs = diffs.flatten()
    #        # diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
    #        # autocorr_gpu = cp.histogram(diffs, bins=autocorr_bins)[0]            
          
    #        # autocorr_gpu= autocorr_gpu/ cp.nanmax(autocorr_gpu) if cp.nanmax(autocorr_gpu) > 0 else autocorr_gpu #normalize
    #        autocorr = autocorr.get()  # Copy from GPU to CPU
           
    #        #autocorr=cupy_autocorr(all_times_ms, autocorr_window_ms=500,autocorr_bins=2)
           
    
    #        # Plotting
    #        ax_isi = axes[i, 0]
    #        ax_autocorr = axes[i, 1]
    
    #        ax_isi.bar(x=bin_edges[:-1], height=bin_counts, width=bin_ms, color='black', align="edge")
    #        ax_isi.vlines(x=2.0, ymin=0, ymax=1, color='red', linewidth=2)
        
    
    #        ax_autocorr.bar(x=autocorr_bins[:-1], height=autocorr, width=autocorr_bin_ms, color='black', align='edge')
           
        
    
    #        fstring = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
    #        fstring_color = 'black'
    #        if rp_contamination[unit_id] > 0.9:
    #            fstring_color = 'red'
    #        ax_isi.text(0.1, 0.9, fstring, va='top', color=fstring_color, fontsize=10)
           
    #     ax_isi.set_xscale('log')
    #     ax_isi.set_xlabel("ISI (log ms)")
    #     ax_isi.set_xlim(left=1)
    #     ax_isi.set_yticklabels([])
    #     ax_isi.set_ylabel("")
    #     ax_isi.spines['top'].set_visible(False)
    #     ax_isi.spines['right'].set_visible(False)
    #     ax_isi.spines['left'].set_visible(False)
    #     ax_autocorr.set_xlabel("Time (ms)")
    #     ax_autocorr.set_yticklabels([])
    #     ax_autocorr.set_ylabel("")
    #     ax_autocorr.spines['top'].set_visible(False)
    #     ax_autocorr.spines['right'].set_visible(False)
    #     ax_autocorr.spines['left'].set_visible(False)
    #     figure.tight_layout()
    #     figure.suptitle(f"ISI and Autocorrelation normalized Probability Density Function\n (Sorted by Max Bin)")
    #     plt.show()
    #     plt.savefig('ISIs_and_autocorr_prob_density_function_sorted_single_subplot.png')
    #     plt.close('all')
    #     print('saved sorted single subplot')
    # plot_ISI_single_sorted_one_column(sorting, analyzer,FR,rp_contamination)
        
    
        #     # autocorr = autocorr / np.nanmax(autocorr) if np.nanmax(autocorr) > 0 else autocorr #normalize
        #     histogram_data[unit_id] = isi_histograms[0][i]/np.nanmax(isi_histograms[0][i])
            
        #     autocorr=correlograms[0][i]/np.nanmax(correlograms[0][i])
        #     # # Autocorrelation calculation
        #     # all_times_ms = np.array(all_times_ms)
        #     # autocorr_bins = np.arange(-autocorr_window_ms / 2, autocorr_window_ms / 2, autocorr_bin_ms)
        #     # autocorr = np.zeros_like(autocorr_bins[:-1])
    
        #     # for spike_time in all_times_ms:
        #     #     diffs = all_times_ms - spike_time
        #     #     diffs = diffs[(diffs >= -autocorr_window_ms / 2) & (diffs < autocorr_window_ms / 2)]
        #     #     autocorr += np.histogram(diffs, bins=autocorr_bins)[0]
    
        #     # autocorr = autocorr / np.nanmax(autocorr) if np.nanmax(autocorr) > 0 else autocorr #normalize
    
        #     # Plotting
        #     ax_isi = axes[i, 0]
        #     ax_autocorr = axes[i, 1]
        #     bin_edges= isi_histograms[1]
            
        #     ax_isi.bar(x=bin_edges[:-1], height=histogram_data[unit_id] , width=bin_ms, color='black', align="edge")
        #     ax_isi.vlines(x=2.0, ymin=0, ymax=1, color='red', linewidth=2)
        #     ax_isi.set_xscale('log')
        #     ax_isi.set_xlabel("ISI (log ms)")
        #     ax_isi.set_xlim(left=1)
        #     ax_isi.set_yticklabels([])
        #     ax_isi.set_ylabel("")
        #     ax_isi.spines['top'].set_visible(False)
        #     ax_isi.spines['right'].set_visible(False)
        #     ax_isi.spines['left'].set_visible(False)
            
        #     autocorr_bins=correlograms[1]
            
        #     ax_autocorr=sw.plot_autocorrelograms(analyzer,unit_ids=[unit_id])
        #     #ax_autocorr.bar(x=autocorr_bins[:-1], height=autocorr, width=2, color='black', align='edge')
        #     ax_autocorr.bar(x=autocorr_bins[:-1], height=autocorr_cpu, width=autocorr_bin_ms, color='black', align='edge')
        #     ax_autocorr.set_xlabel("Time Lag (ms)")
        #     ax_autocorr.set_yticklabels([])
        #     ax_autocorr.set_ylabel("")
        #     ax_autocorr.spines['top'].set_visible(False)
        #     ax_autocorr.spines['right'].set_visible(False)
        #     ax_autocorr.spines['left'].set_visible(False)
    
        #     fstring = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
        #     fstring_color = 'black'
        #     if rp_contamination[unit_id] > 0.9:
        #         fstring_color = 'red'
        #     ax_isi.text(0.1, 0.9, fstring, va='top', color=fstring_color, fontsize=10)
    
        # figure.tight_layout()
        # figure.suptitle(f"ISI and Autocorrelation normalized Probability Density Function\n (Sorted by Max Bin)")
        # plt.show()
        # plt.savefig('ISIs_and_autocorr_prob_density_function_sorted_single_subplot.png')
        # plt.close('all')
        # print('saved sorted single subplot')
    
        # Reorder by FR value and save as a separate figure
        # sorted_unit_ids_fr = sorted(sorting.unit_ids, key=lambda unit_id: FR[unit_id])
    
        # figure_fr, ax_fr = plt.subplots(figsize=(20, 60))
        # y_offset_fr = 0
    
        # for i, unit_id in enumerate(sorted_unit_ids_fr):
        #     bin_edges, bin_counts = histogram_data[unit_id]
        #     ax_fr.bar(x=bin_edges, height=bin_counts, width=bin_ms, bottom=y_offset_fr, color='black', align="edge")
    
        #     ax_fr.vlines(x=2.0, ymin=y_offset_fr, ymax=y_offset_fr + 1, color='red', linewidth=2)
    
        #     y_offset_fr += 2.0
        #     fstring = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
        #     fstring_color = 'black'
        #     if rp_contamination[unit_id] > 0.9:
        #         fstring_color = 'red'
        #     ax_fr.text(0.1, y_offset_fr - np.nanmax(bin_counts) * 0.5, fstring, va='center', color=fstring_color, fontsize=10)
    
        # ax_fr.set_xscale('log')
        # ax_fr.set_xlabel("ISI (log ms)")
        # ax_fr.set_xlim(left=1)
        # ax_fr.set_yticklabels([])
        # ax_fr.set_ylabel("")
        # ax_fr.spines['top'].set_visible(False)
        # ax_fr.spines['right'].set_visible(False)
        # ax_fr.spines['left'].set_visible(False)
    
        # figure_fr.tight_layout()
        # figure_fr.suptitle(f"ISI normalized Probability Density\n Function (Sorted by FR)")
        # plt.show()
        # plt.savefig('ISIs_prob_density_function_sorted_single_subplot_fr.png')
        # plt.close('all')
        # print('saved sorted single subplot FR')  
        
    
        
    
    def plot_ISI_single_sorted(sorting, analyzer):
        import matplotlib
        matplotlib.use("TkAgg")  # Keep using a GUI backend
        import matplotlib.pyplot as plt
        import spikeinterface.qualitymetrics as sqm
        plt.ion()  # Turn on interactive mode
        plt.rcParams.update({
        'font.size': 36,            # controls default text sizes
        'axes.titlesize': 18,       # fontsize of the axes title
        'axes.labelsize': 26,       # fontsize of the x and y labels
        'xtick.labelsize': 24,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 62      # fontsize of the figure title
        })
        FR = sqm.compute_firing_rates(analyzer)
        rp_contamination, rp_violations = sqm.compute_refrac_period_violations(analyzer)
        keys_over_0_9 = [key for key, value in rp_contamination.items() if value > 0.9]
        print(f"split units: {keys_over_0_9}")
    
        num_units = len(sorting.unit_ids)
        fig_height = max(6, num_units * 1.5)
        rows = 1
        cols = 5
        figure, axes = plt.subplots(rows, cols, figsize=(50, fig_height / cols))
    
        num_segments = sorting.get_num_segments()
        fs = sorting.sampling_frequency
        window_ms = float(500.0)
        bin_ms = float(1.0)
    
        histogram_data = {}  # Store histogram data for each unit
    
        for i, unit_id in enumerate(sorting.unit_ids):
            bins = np.arange(0, window_ms, bin_ms)
            bin_counts = None
    
            for segment_index in range(num_segments):
                times_ms = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index) / fs * 1000.0
                isi = np.diff(times_ms)
    
                bin_counts_, bin_edges = np.histogram(isi, bins=bins, density=True)
                if segment_index == 0:
                    bin_counts = bin_counts_
                else:
                    bin_counts += bin_counts_
    
            bin_counts = np.nan_to_num(bin_counts)
            bin_counts = bin_counts / np.nanmax(bin_counts)
    
            histogram_data[unit_id] = (bin_edges[:-1], bin_counts)  # Store bin edges and counts
    
        # Sort histograms by the bin with the highest count
        def get_max_bin_index(unit_id):
            bin_counts = histogram_data[unit_id][1]
            return np.argmax(bin_counts)
    
        sorted_unit_ids = sorted(sorting.unit_ids, key=get_max_bin_index)
    
        y_offset = 0
    
        for i, unit_id in enumerate(sorted_unit_ids):
            ax_i = np.mod(i, 5)
            ax = axes[ax_i]
            bin_edges, bin_counts = histogram_data[unit_id]
    
            ax.bar(x=bin_edges, height=bin_counts, width=bin_ms, bottom=y_offset, color="black", align="edge")
            ax.set_xscale('log')
            y_offset += 1.5
    
            string = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
            ax.text(0.1, y_offset - np.nanmax(bin_counts) * 0.5, string, va='center', fontsize=10, color='red')
    
        for ax in axes:
            ax.set_xlabel("ISI (log ms)")
            ax.set_xlim(left=0.1)
            ax.set_yticklabels([])
            ax.set_ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
        figure.tight_layout()
        figure.suptitle("ISI normalized Probability Density Function (Sorted by Max Bin)")
    
        plt.show()
        plt.savefig('ISIs_prob_density_function_sorted.png')
        plt.close('all')
        print('saved sorted')   
        
        def plot_ISI_single(sorting):
            import matplotlib
            matplotlib.use("TkAgg")  # Keep using a GUI backend
            import matplotlib.pyplot as plt
            plt.ion()  # Turn on interactive mode
            plt.rcParams.update({
            'font.size': 36,            # controls default text sizes
            'axes.titlesize': 18,       # fontsize of the axes title
            'axes.labelsize': 26,       # fontsize of the x and y labels
            'xtick.labelsize': 24,
            'ytick.labelsize': 14,
            'legend.fontsize': 10,
            'figure.titlesize': 62      # fontsize of the figure title
        })
            FR=sqm.compute_firing_rates(analyzer)  
            rp_contamination,rp_violations=sqm.compute_refrac_period_violations(analyzer)
            keys_over_0_9 = [key for key, value in rp_contamination.items() if value > 0.9]
            print(f"split units: {keys_over_0_9}")
            
                        
            num_units = len(sorting.unit_ids)
            
            fig_height = max(6, num_units * 1.5)  # Adjust height based on number of units
    #        figure, ax = plt.subplots(figsize=(50, fig_height))
            rows=1
            cols=5
            figure, axes = plt.subplots(rows, cols, figsize=(50, fig_height/cols))  
    
    
    
            num_segments = sorting.get_num_segments()
            fs = sorting.sampling_frequency
            window_ms=float(500.0)
            bin_ms=float(1.0)
            
            y_offset = 0  # Initial offset for the first unit
            
            for i, unit_id in enumerate(sorting.unit_ids):
                ax_i=np.mod(i,5)
    
                print(f"{num_units-i}")
                bins = np.arange(0, window_ms, bin_ms)
                bin_counts = None
                
                ax = axes[ax_i]
                for segment_index in range(num_segments):
                    times_ms = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index) / fs * 1000.0
                    isi = np.diff(times_ms)
            
                    bin_counts_, bin_edges = np.histogram(isi, bins=bins, density=True)
                    if segment_index == 0:
                        bin_counts = bin_counts_
                    else:
                        bin_counts += bin_counts_
                
                #Replace NaN values with zeros
                bin_counts = np.nan_to_num(bin_counts)
                bin_counts=bin_counts/np.nanmax(bin_counts)
                     
            
                # Plot each unit's ISI distribution with a vertical offset
                ax.bar(x=bin_edges[:-1], height=bin_counts, width=bin_ms, bottom=y_offset, color="black", align="edge")
                ax.set_xscale('log')  # Set x-axis to log scale
                #y_offset += np.nanmax(bin_counts) * 1.1  # Increase offset for the next unit
                y_offset += 1.5
                # Annotate the unit id
                # Format the firing rate to one decimal point and adjust the fontsize
                string = f"Unit {unit_id}, {FR[unit_id]:.1f}Hz, RPC={rp_contamination[unit_id]:.1f}"
                ax.text(0.1, y_offset - np.nanmax(bin_counts) * 0.5, string, va='center', fontsize=10, color='red')
    #            string=f"Unit {unit_id} {FR[unit_id]}Hz"
     #           ax.text(1, y_offset - np.nanmax(bin_counts) * 0.5, string, va='center',color='red')
            for ax in axes:
             ax.set_xlabel("ISI (log ms)")
             ax.set_xlim(left=0.1)  # Ensure x-axis starts at a positive value for log scale
             ax.set_yticklabels([])
             ax.set_ylabel("")
             ax.spines['top'].set_visible(False)
             ax.spines['right'].set_visible(False)
             ax.spines['left'].set_visible(False)
            figure.tight_layout()
    
            
         
    
            #ax.set_ylabel("Units")
            figure.suptitle("ISI normalized Probability Density Function")
        
            plt.show();plt.savefig('ISIs_prob_density_function.png');plt.close('all');print('saved')        
            
            
        
    def plot_analyzer_extensions(analyzer,sorting):
        import matplotlib
        #matplotlib.use("TkAgg")  # Keep using a GUI backend
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode
        plt.rcParams.update({
        'font.size': 12,            # controls default text sizes
        'axes.titlesize': 12,       # fontsize of the axes title
        'axes.labelsize': 12,       # fontsize of the x and y labels
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18      # fontsize of the figure title
    })
        
        
        #A good clustering with well separated and compact clusters will have a silhouette score close to 1. 
        #A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error).
        #simple_sil_score = sqm.simplified_silhouette_score(all_pcs=all_pcs, all_labels=all_labels)#, this_unit_id=0)
        
        
        #The amplitude CV median is expected to be relatively low for well-isolated units, indicating a “stereotypical” spike shape.
        #The amplitude CV range can be high in the presence of noise contamination, due to amplitude outliers like in the example below.
        amplitude_cv_median, amplitude_cv_range = sqm.compute_amplitude_cv_metrics(analyzer)
        # amplitude_cv_median and  amplitude_cv_range are dicts containing the unit ids as keys,
        # and their amplitude_cv metrics as values.

        # this metric identifies unit separation, a high value indicates a highly contaminated unit (type I error) ([Schmitzer-Torbert] et al.). [Jackson] et al. suggests that this measure is also correlated with type II errors (although more strongly with type I errors).
        #A well separated unit should have a low L-ratio ([Schmitzer-Torbert] et al.).
        
        #_, l_ratio = sqm.isolation_distance(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)
        #SD ratio many times over 1 is bad
        sd_ratio = sqm.compute_sd_ratio(sorting_analyzer=analyzer, censored_period_ms=4.0)
        #D-prime is a measure of cluster separation, and will be larger in well separated clusters.
      #  d_prime = sqm.lda_metrics(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)
        
       
        rp_contaminations,rp_violations=sqm.compute_refrac_period_violations(analyzer)
        rp_violations=list(rp_violations.values())
        rp_contamination=list(rp_contaminations.values())
        rp_contaminations=rp_contaminations.items()
        items =  rp_contaminations
        units,rp_contaminations = zip(*items)  
        
       
       

        isi_violations_ratio, isi_violations_count = sqm.compute_isi_violations(sorting_analyzer=analyzer, isi_threshold_ms=1.0)
        
        ISI_violations=sqm.compute_isi_violations(analyzer)
        
        isi_violations_ratio=list(ISI_violations.isi_violations_ratio.values())
        
        isi_violations_count=list(ISI_violations.isi_violations_count.values())
        
        
        FR=sqm.compute_firing_rates(analyzer)        
        FR=list(FR.values())
        
        
        nspikes=sqm.compute_num_spikes(analyzer)
        nspikes=list(nspikes.values())
        
        compute_presence_ratios=sqm.compute_presence_ratios(analyzer,bin_duration_s=60*2)
        presence_ratios=list(compute_presence_ratios.values())
        
        firing_ranges=sqm.compute_firing_ranges(analyzer)
        firing_ranges=list(firing_ranges.values())
        
             
        sliding_rp_violations_contamination = sqm.compute_sliding_rp_violations(sorting_analyzer=analyzer, bin_size_ms=0.25)
        sliding_rp_violations_contamination=list(sliding_rp_violations_contamination.values())
        
        
        snr=sqm.compute_snrs(analyzer)
        snr=list(snr.values())
        
        
        
        #sliding_rp_violations=sqm.compute_sliding_rp_violations(analyzer, min_spikes=0, bin_size_ms=0.25, window_size_s=1, exclude_ref_period_below_ms=0.5, max_ref_period_ms=10)
        # compute_num_spikes,
        # compute_firing_rates,
        # compute_presence_ratios,
        # compute_snrs,
        # compute_isi_violations,
        # compute_refrac_period_violations,
        # compute_sliding_rp_violations,
        # compute_amplitude_cutoffs,
        # compute_amplitude_medians,
        # compute_drift_metrics,
        # compute_synchrony_metrics,
        # compute_firing_ranges,
        # compute_amplitude_cv_metrics,
        # compute_sd_ratio,
        
        #snrs=sqm.compute_snrs(analyzer)
        
        #compute_amplitude_cutoffs= sqm.compute_amplitude_cutoffs(analyzer)
        
        
        template_metrics_df=analyzer.get_extension("template_metrics").get_data()
   
        peak_to_valley=template_metrics_df['peak_to_valley'].values
        peak_trough_ratio=template_metrics_df['peak_trough_ratio'].values
        half_width=template_metrics_df['half_width'].values
        repolarization_slope=template_metrics_df['repolarization_slope'].values
        recovery_slope=template_metrics_df['recovery_slope'].values
        num_positive_peaks=template_metrics_df['num_positive_peaks'].values
        num_negative_peaks=template_metrics_df['num_negative_peaks'].values
        
        
        
        ['peak_to_valley', 'peak_trough_ratio', 'half_width',
               'repolarization_slope', 'recovery_slope', 'num_positive_peaks',
               'num_negative_peaks']
        
        fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(12, 20))
        fig.suptitle('Scatter Plots and Histograms for Computed Variables', fontsize=16)
        axs[0, 0].scatter(FR, isi_violations_ratio, marker='o', linestyle='-', color='b')
        axs[0, 0].set_xlabel('Firing rate (Hz)')    
        axs[0, 0].set_ylabel('ISI Violation Ratio')
        
        axs[0, 1].hist( isi_violations_ratio,bins=60,color='black')
        axs[0, 1].set_xlabel('isi violations ratio')    
        axs[0, 1].set_ylabel('unit count')      
        
        axs[1, 0].scatter(FR, isi_violations_count, marker='o', linestyle='-', color='r')
        axs[1, 0].set_xlabel('Firing rate (Hz)')    
        axs[1, 0].set_ylabel('ISI Violation count')    
        
        axs[1, 1].hist( isi_violations_count, bins=60,color='black')
        axs[1, 1].set_xlabel('isi violations count')    
        axs[1, 1].set_ylabel('unit count')
        
        axs[2, 0].scatter(FR, rp_contamination, marker='o', linestyle='-', color='g' )
        axs[2, 0].set_xlabel('Firing rate (Hz)')    
        axs[2, 0].set_ylabel('refractory period contamination')    
        
        axs[2, 1].hist( rp_contamination, bins=60,color='black')
        axs[2, 1].set_xlabel('refractory period contamination count')    
        axs[2, 1].set_ylabel('unit count')
        
        axs[3, 0].scatter(FR, rp_violations, marker='o', linestyle='-', color='g' )
        axs[3, 0].set_xlabel('Firing rate (Hz)')    
        axs[3, 0].set_ylabel('rp_violations count')    
        
        axs[3, 1].hist(rp_violations, bins=60, color='black')
        axs[3, 1].set_xlabel('rp_violations count')    
        axs[3, 1].set_ylabel('unit count')    
        
        
        axs[4, 0].hist( FR, bins=60,color='black')
        axs[4, 0].set_xlabel('Firing rate (Hz)')  
        axs[4, 0].set_ylabel('unit count')
           
        axs[4, 1].hist( np.log10(FR), bins=60,color='black')
        axs[4, 1].set_xlabel('log Firing rate')  
        axs[4, 1].set_ylabel('unit count')
        
        axs[5, 0].scatter(FR,presence_ratios, marker='o', linestyle='-', color='k')
        axs[5, 0].set_xlabel('Firing rate (Hz)')
        axs[5, 0].set_ylabel('presence ratio')
           
        axs[5, 1].hist( presence_ratios, bins=60,color='black')
        axs[5, 1].set_xlabel('presence_ratios')  
        axs[5, 1].set_ylabel('unit count')
        
        
       
        def save_sorting_as_phy(analyzer,analyzer_folder,sparsity):
            from spikeinterface.exporters import export_report,export_to_phy
            
            #save to Phy
            phy_folder=Path(f"{analyzer_folder}\Phy")
            export_to_phy(sorting_analyzer=analyzer,output_folder=phy_folder,sparsity=sparsity,remove_if_exists=True,**job_kwargs)
            
        def remove_axes(axis=None, top=True, right=True, bottom=False, left=False, ticks=True, rem_all=False):
            import matplotlib
            if axis is None:
                axis = plt.gca()  
            if rem_all:
                ticks=False
            
            if isinstance(axis, matplotlib.axes.Axes):
                axs=[axis]
            else:
                axs=axis
                
            for ax in axs:
                if top or rem_all:
                    ax.spines['top'].set_visible(False)
                if right or rem_all:
                    ax.spines['right'].set_visible(False)
                if bottom or rem_all:
                    ax.spines['bottom'].set_visible(False)
                    if not ticks:
                        ax.set_xticks([])
                if left or rem_all:
                    ax.spines['left'].set_visible(False)
                    if not ticks:
                        ax.set_yticks([])
                        
                        
        
        remove_axes(axs[0,0])
        remove_axes(axs[0,1])
        remove_axes(axs[1,0])
        remove_axes(axs[1,1])
        remove_axes(axs[2,0])
        remove_axes(axs[2,1])
        remove_axes(axs[3,0])
        remove_axes(axs[3,1])
        remove_axes(axs[4,0])
        remove_axes(axs[4,1])
        remove_axes(axs[5,0])
        remove_axes(axs[5,1])

        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        


        plt.savefig('a.png')
        plt.close('all')
        
        
        

        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 20))
        ax = fig.add_subplot(111, projection='3d',)
        
        ax.scatter(FR, rp_violations, isi_violations_ratio, c='r', marker='o')
        # Label the axes
        ax.set_xlabel('Firing rate (Hz)')
        ax.set_ylabel('RP violations')
        ax.set_zlabel('ISI violations_ratio')
        plt.show()
        
        
        plt.savefig('b.png')
        plt.close('all')

        
        metric_headers=['rp_violations','rp_contamination','sliding_rp_violations','isi_violations_ratio','isi_violations_count','snr','FR','nspikes','presence_ratios']
        #[, 'peak_to_valley', 'peak_trough_ratio', 'half_width',                'repolarization_slope', 'recovery_slope', 'num_positive_peaks','num_negative_peaks']
        metric_data={
            'rp_violations':rp_violations,
            'rp_contamination':rp_contamination,
            'sliding_rp_violations':sliding_rp_violations_contamination,
            'isi_violations_ratio':isi_violations_ratio,
            'isi_violations_count':isi_violations_count,
            'snr':snr,
            'FR':FR,
            'nspikes':nspikes,
            'presence_ratios':presence_ratios}
            # 'peak_to_valley':peak_to_valley,
            # 'peak_trough_ratio':peak_trough_ratio,
            # 'half_width':half_width,
            # 'repolarization_slope':repolarization_slope,
            # 'recovery_slope':recovery_slope,
            # 'num_positive_peaks':num_positive_peaks,
            # 'num_negative_peaks':num_negative_peaks
            
            # }
      
        ranges = [[np.nanmin(rp_violations),np.nanmax(rp_violations)],[np.nanmin(rp_contamination),np.nanmax(rp_contamination)],[np.nanmin(sliding_rp_violations_contamination),np.nanmax(sliding_rp_violations_contamination)],
                  [np.nanmin(isi_violations_ratio),np.nanmax(isi_violations_ratio)],[np.nanmin(isi_violations_count),np.nanmax(isi_violations_count)],[np.nanmin(snr),np.nanmax(snr)],
                  [np.nanmin(FR),np.nanmax(FR)],[np.nanmin(nspikes),np.nanmax(nspikes)],[np.nanmin(presence_ratios),np.nanmax(presence_ratios)]]
       
        
        fig, axs = plt.subplots(nrows=9, ncols=1, figsize=(40, 40))
        for idx, metric in enumerate(metric_headers):
            data = metric_data[metric]
            # remove NaNs
            data = np.array(data)[np.invert(np.isnan(data))]
        
            #plt.subplot(len(metrics),1,idx+1)
            axs[idx].boxplot(data, showfliers=False, showcaps=False, vert=False)
        
            axs[idx].set_ylim([0.8,1.2])
            axs[idx].set_xlim(ranges[idx])
            axs[idx].set_yticks([])
            axs[idx].set_xlabel(metric)  
            remove_axes(axs[idx],top=True, right=True, bottom=False, left=True)
            

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
            
        plt.savefig('boxes.png')      
        plt.close('all')
        
        
        


        
        fig, axs = plt.subplots(nrows=len(metric_headers), ncols=len(metric_headers), figsize=(20 ,20))
        for idx, metric_i in enumerate(metric_headers):
            for jdx, metric_j in enumerate(metric_headers):                
                if idx==jdx:
                    axs[idx,jdx].hist(metric_data[metric_i],25,color='gray')
                else:                    
                    axs[idx,jdx].scatter(metric_data[metric_j], metric_data[metric_i], c='gray', s=5, marker="o")                                
                axs[idx, jdx].spines["top"].set_visible(False)
                axs[idx, jdx].spines["right"].set_visible(False)
                if jdx==0:#first col
                    axs[idx,jdx].set_ylabel(metric_i)          
                    #axs[idx, jdx].set_xticklabels([])
                if idx==len(metric_headers)-1:#last row
                    axs[idx,jdx].set_xlabel(metric_j)
                    #axs[idx, jdx].set_yticklabels([])
                if jdx==0 or idx==len(metric_headers)-1:
                    pass
                else:
                   axs[idx, jdx].set_xticklabels([])
                   axs[idx, jdx].set_yticklabels([])
                  # axs[idx, jdx].spines["top"].set_visible(False)
                  #  axs[idx, jdx].spines["right"].set_visible(False)  
                remove_axes(axs[idx,jdx])
        
        plt.rcParams.update({
        'font.size': 14,            # controls default text sizes
        'axes.titlesize': 14,       # fontsize of the axes title
        'axes.labelsize': 14,       # fontsize of the x and y labels
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18      # fontsize of the figure title
    })
        plt.tight_layout(rect=[0, 0 ,0.97,0.97])
        plt.show()
        #fig.subplots_adjust(top=0.1,bottom=0.1, wspace=0.1, hspace=0.1)
        plt.savefig('quality_i_j.png')      
        plt.close('all')
        
        
        
           
                
        def select_units(analyzer,base_folder):
            metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']


            # metrics = analyzer.compute("quality_metrics").get_data()
            # equivalent to
            metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)
            
            
            amplitude_cutoff_thresh = 0.1
            isi_violations_ratio_thresh = 1
            presence_ratio_thresh = 0.9

            our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
            print(our_query)
            keep_units = metrics.query(our_query)
            keep_unit_ids = keep_units.index.values
            keep_unit_ids
            analyzer_clean = analyzer.select_units(keep_unit_ids, folder=base_folder / 'analyzer_clean', format='binary_folder')
            si.export_report(analyzer_clean, base_folder / 'report', format='png')
            analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')
            return analyzer_clean
            

        
    
       
        
       # print('computing quality_metrics')
        
        #analyzer_saved = analyzer.save_as(folder=analyzer_sparse_folder / "computed", format="binary_folder",overwrite=True)
        
        #analyzer.compute("quality_metrics",,**job_kwargs);print(analyzer)

        
        
    
   # analyzer = compute_analyzer_extentions(analyzer_folder)
    
    return analyzer
    
    
    def compute_quality_metrics_extentions(analyzer):
        """
        quality metrics documentation
        https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
        """
                
        # # without PC (depends on "waveforms", "templates", and "noise_levels")
        # qm_ext = analyzer.compute(input="quality_metrics", metric_names=['snr'], skip_pc_metrics=True)
        # metrics = qm_ext.get_data()
        # assert 'snr' in metrics.columns
        
        # # with PCs (depends on "pca" in addition to the above metrics)
        
        # qm_ext = sorting_analyzer.compute(input={"principal_components": dict(n_components=3, mode="by_channel_local"),
        #                                 "quality_metrics": dict(skip_pc_metrics=False)})
        #  metrics = qm_ext.get_data()
        #  assert 'isolation_distance' in metrics.columns
        
        analyzer.compute("quality_metrics",**job_kwargs)
        #analyzer.compute(input="quality_metrics", metric_names=["firing_rate", "snr", "amplitude_cutoff",  "isolation_distance","d_prime"], skip_pc_metrics=True)
        #sliding_rp_violations=sqm.compute_sliding_rp_violations(analyzer, min_spikes=0, bin_size_ms=0.25, window_size_s=1, exclude_ref_period_below_ms=0.5, max_ref_period_ms=10)
        # compute_num_spikes,
        # compute_firing_rates,
        # compute_presence_ratios,
        # compute_snrs,
        # compute_isi_violations,
        # compute_refrac_period_violations,
        # compute_sliding_rp_violations,
        # compute_amplitude_cutoffs,
        # compute_amplitude_medians,
        # compute_drift_metrics,
        # compute_synchrony_metrics,
        # compute_firing_ranges,
        # compute_amplitude_cv_metrics,
        # compute_sd_ratio,
        # spost.get_template_metric_names()
        # Out[24]: 
        # ['peak_to_valley',
        #  'peak_trough_ratio',
        #  'half_width',
        #  'repolarization_slope',
        #  'recovery_slope',
        #  'num_positive_peaks',
        #  'num_negative_peaks',
        #  'velocity_above',
        #  'velocity_below',
        #  'exp_decay',
        #  'spread']
        return analyzer
    
   

        
      
    
    # fig, ax = plt.subplots()
    # _ = ax.hist(firing_rates, bins=np.arange(0, 50, 2))
    # ax.set_xlabel('FR (Hz)')
    # noise_levels_microV = si.get_noise_levels(catgt_rec, return_scaled=True)
    # fig, ax = plt.subplots()
    # _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
    # ax.set_xlabel('noise  [microV]')
    # Text(0.5, 0, 'noise  [microV]')
    return analyzer
    
    

     
 
     
        
    
    


def load_IRC_in_Pynapple(quality_file, timestamps_file,df_irc, t1_s, tend_s):
        
    # spike_times = df_irc.spike_times[::]
    # df_irc['xpos'][ 'ypos']
    # irc_dict=df_irc.to_dict()
    
    
    # import pynapple as nap
    # import numpy as np
    
    # my_ts = spike_times.to_dict()
    
    # import numpy as np
    # import pandas as pd
    # import pynapple as nap
    # import scipy.ndimage
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import requests, math, os
    # import tqdm
    
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
    
    
    # # Changed the backend from 'numba' to 'jax'
    # nap.nap_config.set_backend("numba")     
    # spikes = nap.TsGroup(spike_times, time_units='s')
    # spikes['clustID'] =  np.array(np.int32(df_irc['unit_id']))
    # spikes.set_info(unit_type=irc_dict['note'])
    
    # spikes['amplitudes'] = np.array(np.int16(df_irc['uV_pp']))
    # spikes['probe_Xpos']=df_irc['xpos']
    # spikes['probe_Ypos']=df_irc['ypos']
    
    
    # pf = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)
    
    # tuning_curves = nap.compute_1d_tuning_curves(
    # group=spikes, 
    # feature=angle, 
    # nb_bins=61, 
    # minmax=(0, 2 * np.pi)
    # )
    
        
     # crosscorr=nap.compute_crosscorrelogram(spikes,
     #                               binsize=1,
     #                               windowsize=10,
     #                               norm=True)
    
    
    
    
    
    
    # ##############
    # my_ts = {}
    # chanPosX = []
    # chanPosY = []
    
    # for index, n in enumerate(clusterID):
    #     mask = spike_clusters == n
    #     spikeTimesN = spike_times[mask] / fs
    #     chanN = spike_positions[mask, :]
        
    #     my_ts[index] = np.array(spikeTimesN)
        
    # import pynapple as nap
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import numpy as np
    # import os
    # from scipy.io import loadmat
    
    # spikes = nap.TsGroup(my_ts)
    # spikes['clustID'] = np.array(clusterID)
    # spikes['posX'] = np.array(chanPosX)
    # spikes['posY'] = np.array(chanPosY)
    # spikes.save(os.path.join(datapath, 'clusters.npz'))
    # plt.figure()
    # plt.subplot(211)
    # for n in range(len(spikes)):
    #  plt.eventplot(spikes[n].t,lineoffsets = n,linelengths = 0.3)
    # plt.xlabel('time (s)')
    # plt.ylabel('unit #')
    pass

def save_to_NWB():
    pass
    

def Steinmetz_ISI(spikeTrain,duration, minISI=1/30000, refDur=0.002):
    """
    based on matlab code from:
        https://github.com/cortex-lab/sortingQuality/blob/master/core/ISIViolations.m
                computes an estimated false positive rate of the spikes in the given
     spike train. You probably don't want this to be greater than a few
     percent of the spikes. 
    
     - minISI [sec] = minimum *possible* ISI (based on your cutoff window); likely 1ms
     or so. It's important that you have this number right.
     
     - refDur [sec] = estimate of the duration of the refractory period; suggested value = 0.002.
     It's also important that you have this number right, but there's no way
     to know... also, if it's a fast spiking cell, you could use a smaller
     value than for regular spiking.
    
     This function was updated on 2023-09-05 by NS to reflect a correction.
     There were two problems: 
     1) The approximation previously used, which was chosen to avoid getting
     imaginary results, wasn't accurate to the Hill et al paper on which this
     method was based, nor was it accurate to the correct solution to the
     problem
     2) Hill et al did not have the correct solution to the problem. The Hill
     paper used an expression derived from an earlier work (Meunier et al
     2003) which had assumed a special case: the "contamination" was itself
     only generated by a single neuron and therefore the contaminating spikes
     themselves had a refractory period. If instead the contaminating spikes
     are generated from a real poisson process (as in the case of electrical
     noise or many nearby neurons generating the contamination), then the
     correct expression is different, as now calculated here. This expression
     is given in Llobet et al. bioRxiv 2022
     (https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1.full.pdf)
    
     In practice, the three methods (the real Hill equation, the method
     previously implemented here, and the correct equation as now implemented)
     return almost identical values for contamination less than ~20%. They
     diverge strongly for 30% or more. 
    
    % Bonus: the correct expression no longer returns imaginary values!
    """
    #T is the duration of the recording in seconds.
    #Nt is the number of spikes in the unit’s spike train.
    # refDur the duration of the unit’s refractory period in seconds
    # Ensure spikeTrain is a numpy array
    #UMS_C= (refDur*T)/(2*Nt^2*minISI)
    # Llobet_C= 1- sqrt(1- refDur*T/Nt^2*minISI)  / 1/2* (1- (sqrt(1- 2*refDur*T/Nt^2*minISI))
   
    spikeTrain = np.array(spikeTrain)
    
    # Total spike count
    Nt = len(spikeTrain)
    
    # Duration of recording
    T=spikeTrain[-1]
#    D = spikeTrain.max()
    
    # Compute inter-spike intervals (ISIs)
    isis = np.diff(spikeTrain)
    
    # Number of observed violations
    numViolations = np.sum((isis <= refDur) & (isis > minISI))
    
    # ## Compute false positive rate
    # fpRate = 1 - np.sqrt(1 - numViolations * D / (Nt**2 * (refDur - minISI)))
    
        # Number of observed violations
    numViolations = np.sum((isis <= refDur) & (isis > minISI))
    
    # Compute expression inside sqrt and clip it to be non-negative to avoid invalid sqrt
    inside_sqrt = 1 - numViolations * T / (Nt**2 * (refDur - minISI))
    inside_sqrt = np.clip(inside_sqrt, 0, None)  # Prevent negative values
    
    # Compute false positive rate
    fpRate = 1 - np.sqrt(inside_sqrt)
    
  #   isis = np.diff(spikeTrain)
 	# nSpikes = len(spikeTrain)
 	# numViolations = sum(isis<refDur) 
 	# violationTime = 2*nSpikes*(refDur-minISI)
 	# totalRate = nSpikes/spikeTrain[-1]
 	# violationRate = numViolations/violationTime			
 	# fpRate = violationRate/totalRate
    return fpRate


import numpy as np

def exclude_neurons(t1_s,tend_s,n_by_t, spike_times, n_spike_times, spike_clusters, cluster_index, cluster_channels,
                     refractory_period=0.002, ISI_violations_percentage=0.025,
                     min_spikes=20, units_id_to_remove=None,df_irc=None,avg_waveform=None):
    """
    Excludes neurons based on spike count, ISI violations, and a list of units to remove.

    Parameters
    ----------
    n_by_t : ndarray
        Neuron-by-time matrix.
    spike_times : ndarray
        Spike times for all spikes.
    spike_clusters : ndarray
        Cluster IDs for each spike.
    cluster_index : ndarray
        Cluster number in n_by_t.
    cluster_channels : ndarray
        Channel information for each cluster.
    refractory_period : float, optional
        Refractory period in seconds. Default is 0.002.
    ISI_violations_percentage : float, optional
        Maximum allowed percentage of ISI violations. Default is 0.025.
    min_spikes : int, optional
        Minimum number of spikes required for a neuron. Default is 20.
    units_id_to_remove : list or set, optional
        List or set of unit IDs to remove. Default is None.

    Returns
    -------
    n_by_t : ndarray
        Neuron-by-time matrix with excluded neurons removed.
    cluster_index : ndarray
        Cluster index with excluded neurons removed.
    cluster_channels : ndarray
        Cluster channels with excluded neurons removed.
    removed_clusters : ndarray
        Array of cluster IDs (from the original cluster_index) that were removed.
    """

    include_ind = np.ones(len(n_by_t), dtype=bool)
    num_spikes = np.sum(n_by_t, axis=1)

    include_ind = include_ind & (num_spikes > min_spikes)

    too_few_spikes = len(n_by_t) - np.sum(include_ind)
    print(f'{too_few_spikes} neurons excluded < {min_spikes} spikes')

    removed_clusters_list =[]# Initialize a list to store removed cluster IDs

    # Exclude neurons with too many ISI violations
    for i, n in enumerate(cluster_index):
        n_spikes = spike_times[spike_clusters == n]
        isi = np.diff(n_spikes)
        violations = np.sum(isi < refractory_period) / len(isi) if len(isi) > 0 else 0

        if violations > ISI_violations_percentage:
            include_ind[i] = False
            removed_clusters_list.append(n)
            print(f'unit {n} too many ISI violations ratio: {violations}')

    if units_id_to_remove is not None:
        units_to_remove_set = set(units_id_to_remove)
        cluster_index_array = np.array(cluster_index)
        remove_indices_bool = np.isin(cluster_index_array, list(units_to_remove_set))
        include_ind = include_ind & ~remove_indices_bool
        removed_clusters_list.extend(cluster_index_array[remove_indices_bool])  # Append removed cluster IDs
    
    include_logical_ind=include_ind
    n_by_t = n_by_t[include_ind]
    cluster_index = np.array(cluster_index)
    cluster_index = cluster_index[include_logical_ind]
    cluster_channels = cluster_channels[include_ind]
    nn_spike_times = [n_spike_times[i] for i in range(len(include_ind)) if include_ind[i]]

    removed_clusters = np.array(removed_clusters_list)  # Convert to NumPy array
    
    df_irc = df_irc[df_irc['unit_id'].isin(cluster_index)].copy()
   
    #if avg_waveform is a list
    #avg_waveform=avg_waveform[include_ind]
    #if avg_waveform is a dict
   #  filtered_avg_waveform={}
   #  include_units=np.where(include_ind)[0]
   # # plt.close('all')
   #  for unit_i in include_units:
   #      unit_id=str(int(unit_i))
   #      try:
   #          filtered_avg_waveform[unit_id]=avg_waveform[unit_id].squeeze()
   #        #  plt.plot(filtered_avg_waveform[unit_id])
            
   #      except:
   #          pass
        
    #filtered_avg_waveform = {unit_id: avg_waveform[str(unit_id)] for unit_id in cluster_index if unit_id in avg_waveform}

    
    # For each row in df_filtered, subtract t0 from spike_times and then discard values over t_end_s.
    # For each row in df_filtered, keep spike_times between t1_s and t_end_s, then subtract t1_s from them.
    # Ensure each spike_times is a numpy array before filtering
    df_irc['spike_times'] = df_irc['spike_times'].apply(
        lambda spikes: (
            # Convert to numpy array
            lambda arr: arr[(arr >= t1_s) & (arr <= tend_s)] - t1_s
        )(np.array(spikes))
    )

    print(f'{len(removed_clusters)} neurons excluded because too many ISI violations or in units_id_to_remove')

    return n_by_t, nn_spike_times, cluster_index, cluster_channels, removed_clusters,df_irc#,filtered_avg_waveform


def exclude_neurons_old(n_by_t, spike_times, spike_clusters, cluster_index, cluster_channels,refractory_period=0.002,ISI_violations_percentage=0.025,min_spikes=20,units_id_to_remove=None):
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
    # refractoey_period=0.002 #in seconds
    # ISI_violations_percentage=0.025 #range 0-1
    # min_spikes=5      
    

        
    include_ind=np.ones(len(n_by_t))
    num_spikes=np.sum(n_by_t, axis=1)  
    
    include_ind-= (num_spikes<=min_spikes)

    
    too_few_spikes=len(n_by_t)-np.sum(include_ind)
    print(f'{too_few_spikes} neurons exclude < {min_spikes} spikes')
    
    # Exclude too short ISI
    for n in cluster_index:
        n_spikes=spike_times[spike_clusters==n]
      
      #  ISI_ratio=Steinmetz_ISI(spikeTrain=n_spikes, duration=spike_times.max(),minISI=1/30_000, refDur=0.002)
        isi=np.diff(n_spikes)
        violations=np.sum(isi<refractory_period)/len(isi)
        if violations>ISI_violations_percentage:
            include_ind[cluster_index==n] -= 1
            print(f'unit {n} too many ISI violations ratio: {violations}')
            
   
    
    include_ind[include_ind<1]=0
    include_ind=include_ind.astype(bool)
    if units_id_to_remove!=None:     #add refracotry period excluded unit ids 
       #remove_indices = [i for i, val in enumerate(cluster_index) if val in units_id_to_remove]
       units_to_remove_set = set(units_id_to_remove)
       cluster_index_array = np.array(cluster_index)
       keep_indices_bool= ~np.isin(cluster_index_array, list(units_to_remove_set))           
       include_ind = np.logical_or(include_ind, keep_indices_bool)
    
    
    n_by_t=n_by_t[include_ind]
    cluster_index=cluster_index[include_ind]
    cluster_channels=cluster_channels[include_ind]
    
   # np.count_nonzero(include_ind)

    print(f'{len(include_ind)-sum(include_ind)-too_few_spikes} neurons excluded because too many ISI violations')
    
    return n_by_t, cluster_index, cluster_channels





def get_channels(lfp_path, channel_list,first_frame, last_frame,first_channel, last_channel,step=1):
    
    wanted_channels=np.arange(first_channel, last_channel, step)
    all_channels=[]
    for num in wanted_channels:

        channel=np.load(fr'{lfp_path}\imec0.lf#LF{num}.npy')
        all_channels.append(channel)

    all_channels=np.array(all_channels)
    return all_channels[:,first_frame:last_frame], wanted_channels


def parse_imroTbl(probe_type, imroTbl):
    """
    Parses the imroTbl data based on the probe_type string and returns
    a list of tuples (channel_ID, electrode_ID) if electrode_ID is present;
    otherwise returns None for electrode_ID.
    
    Parameters:
      probe_type (str): a string representing the probe type (e.g., "21", "1020").
      imroTbl (list): a list of lists containing channel parameters.
    
    Returns:
      List[Tuple]: A list of tuples (channel_ID, electrode_ID or None)
    

    # === Example usage ===
    if __name__ == "__main__":
        # Example for a NP 2.0 single multiplexed shank type (probe_type "21")
        example_tbl_type21 = [
            [1, 3, 2, 127],  # channel 1, electrode 127
            [2, 3, 1, 507],
            [3, 3, 4, 887]
        ]
        print("Probe type '21':")
        for ch, elec in parse_imroTbl("21", example_tbl_type21):
            print("Channel: {}, Electrode: {}".format(ch, elec))

        # Example for NP 1.0-like type (no electrode id) with probe_type "1020"
        example_tbl_np10 = [
            [10, 1, 0, 500, 250, 1],
            [11, 1, 1, 500, 250, 1]
        ]
        print("\nProbe type '1020':")
        for ch, elec in parse_imroTbl("1020", example_tbl_np10):
            print("Channel: {}, Electrode: {}".format(ch, elec))

        # Example for type 1110, UHD programmable probe with probe_type "1110"
        example_tbl_1110 = [
            [1110, 2, 1, 100, 100, 1],  # header
            [101, 5, 6],
            [102, 5, 6],
            # ... assume there are 24 total group entries
        ]
        print("\nProbe type '1110':")
        for ch, elec in parse_imroTbl("1110", example_tbl_1110):
            print("Channel: {}, Electrode: {}".format(ch, elec))


    # === Example usage ===
    if __name__ == "__main__":
        # Example for a NP 2.0 single multiplexed shank type (probe_type 21)
        example_tbl_type21 = [
            [1, 3, 2, 127],  # channel 1, electrode 127
            [2, 3, 1, 507],
            [3, 3, 4, 887]
        ]

        print("Probe type 21:")
        for ch, elec in parse_imroTbl(21, example_tbl_type21):
            print("Channel: {}, Electrode: {}".format(ch, elec))

        # Example for NP 1.0-like type (no electrode id)
        example_tbl_np10 = [
            [10, 1, 0, 500, 250, 1],
            [11, 1, 1, 500, 250, 1]
        ]
        print("\nNP 1.0-like probe:")
        for ch, elec in parse_imroTbl(1020, example_tbl_np10):
            print("Channel: {}, Electrode: {}".format(ch, elec))

        # Example for type 1110, UHD programmable
        example_tbl_1110 = [
            [1110, 2, 1, 100, 100, 1],  # header
            [101, 5, 6],
            [102, 5, 6],
            # ... assume there are 24 total group entries
        ]
        print("\nProbe type 1110:")
        for ch, elec in parse_imroTbl(1110, example_tbl_1110):
            print("Channel: {}, Electrode: {}".format(ch, elec))
    """
    # Convert probe_type from string to int for further comparisons
    try:
        p_type = int(probe_type)
    except ValueError:
        raise ValueError("probe_type should be a string representing an integer value.")

    results = []

    # NP 1.0-like types (no electrode ID)
    if p_type in {0,1010, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300}:
        # Expected columns: [Channel ID, Bank number, Ref. ID index, AP band gain, LF band gain, AP hipass filter]
        for row in imroTbl:
            channel_id = row[0]
            results.append((channel_id, None))

    # NP 2.0 single multiplexed shank types
    elif p_type in {21, 2003, 2004}:
        # Expected columns: [Channel ID, Bank mask, Ref. ID index, Electrode ID]
        for row in imroTbl:
            channel_id = row[0]
            electrode_id = row[3]  # 4th element
            results.append((channel_id, electrode_id))

    # NP 2.0 4-shank and Quad-probe types
    elif p_type in {24, 2013, 2014, 2020, 2021}:
        # Expected columns: [Channel ID, Shank ID, Bank ID, Ref. ID index, Electrode ID]
        for row in imroTbl:
            channel_id = row[0]
            electrode_id = row[4]  # 5th element
            results.append((channel_id, electrode_id))

    # Type 1110: UHD programmable
    elif p_type == 1110:
        # Header: [Type, ColumnMode, Ref. ID index, AP band gain, LF band gain, AP hipass filter]
        # Followed by 24 entries: [Group ID, Bank-A, Bank-B]
        if not imroTbl:
            return results
        
        # Skip the header
        for row in imroTbl[1:]:
            channel_id = row[0]  # Using Group ID as channel id
            results.append((channel_id, None))
    
    else:
        raise ValueError("Unknown probe_type: {}".format(probe_type))

    return results

import numpy as np
import pandas as pd

def interpolate_region_acronym2(csv_data: pd.DataFrame, all_probe_depths: list) -> list:
    """
    For each shank (each Series in all_probe_depths), interpolate the Region acronym by finding the closest
    'Distance from first position [um]' value from csv_data to each provided z_coord_um value.
    
    Parameters:
      csv_data: pd.DataFrame with columns including:
                - 'Distance from first position [um]'
                - 'Region acronym'
      all_probe_depths: list of pd.Series; each series contains the z_coord_um values for that shank.
                        Assume the order of this list corresponds to the shank order.
    
    Returns:
      A list (one per shank) of lists of region acronyms corresponding to each z_coord_um.
    """
    
    # Get the array of positions and the region acronyms from the CSV data.
    positions = csv_data['Distance from first position [um]'].values
    region_acronyms = csv_data['Region acronym'].values
    
    # For each shank, map each z coordinate to the region acronym of the track point with the nearest position.
    shank_region_assignment = []
    for shank_depths in all_probe_depths:
        # Ensure the shank depths are treated as a numpy array.
        z_values = np.array(shank_depths)
        # For each z coordinate, find index of the nearest position.
        assigned = []
        for z in z_values:
            # Compute absolute differences, then take the argmin to find the nearest index.
            idx = np.abs(positions - z).argmin()
            assigned.append(region_acronyms[idx])
        shank_region_assignment.append(assigned)
    
    return shank_region_assignment

# Example usage:
# result = interpolate_region_acronym(csv_data, all_probe_depths)
# Now, result is a list where each element is a list of 'Region acronym' values for that shank.
        
def get_probe_tracking(ap_path, probe_tracking, manual_last_channel=None,df_irc=None,df_ks=None):

    """
    takes the output from brainreg to assign a region to each recording channel
    
    It assumes that the number of rows in the brainreg csv == (num_channels_in_the_brain + 8_channels_at_the_tip)
    
    It also assumes that the numbering of points is inverted, i.e. 0 in the excel corresponds to the 
    most dorsal point of the probe (whereas 0 in spikeGLX is the most ventral channel)
    
    Parameters
    ----------
    Path: to the track.csv file that is output from brainreg. In the paths.csv it is under 'probe_tracking'
    manual_last_channel: If there is a shift with respect to atlas alignment, this 
        is taken for calculation of depth
    Channel Entries By Type
    Type {0,1020,1030,1100,1120,1121,1122,1123,1200,1300}, NP 1.0-like:
    
    Channel ID,
    Bank number of the connected electrode,
    Reference ID index,
    AP band gain,
    LF band gain,
    AP hipass filter applied (1=ON)
    The reference ID values are {0=ext, 1=tip, [2..4]=on-shnk-ref}. The on-shnk ref electrodes are {192,576,960}.
    
    Type {21,2003,2004} (NP 2.0, single multiplexed shank):
    
    Channel ID,
    Bank mask (logical OR of {1=bnk-0, 2=bnk-1, 4=bnk-2, 8=bnk-3}),
    Reference ID index,
    Electrode ID (range [0,1279])
    Type-21 reference ID values are {0=ext, 1=tip, [2..5]=on-shnk-ref}. The on-shnk ref electrodes are {127,507,887,1251}.
    
    Type-2003,2004 reference ID values are {0=ext, 1=gnd, 2=tip}. On-shank reference electrodes are removed from commercial 2B probes.
    
    Type {24,2013,2014} (NP 2.0, 4-shank):
    
    Channel ID,
    Shank ID (with tips pointing down, shank-0 is left-most),
    Bank ID,
    Reference ID index,
    Electrode ID (range [0,1279] on each shank)
    Type-24 reference ID values are {0=ext, [1..4]=tip[0..3], [5..8]=on-shnk-0, [9..12]=on-shnk-1, [13..16]=on-shnk-2, [17..20]=on-shnk-3}. The on-shnk ref electrodes of any shank are {127,511,895,1279}.
    
    Type-2013,2014 reference ID values are {0=ext, 1=gnd, [2..5]=tip[0..3]}. On-shank reference electrodes are removed from commercial 2B probes.
    
    Type {2020,2021} (Quad-probe):
    
    Channel ID,
    Shank ID (with tips pointing down, shank-0 is left-most),
    Bank ID,
    Reference ID index,
    Electrode ID (range [0,1279] on each shank)
    Quad-base reference ID values are {0=ext, 1=gnd, 2=tip on same shank as electode}.
    
    Type 1110 (UHD programmable):
    
    This has a unique header:
    
    Type (1110),
    ColumnMode {0=INNER, 1=OUTER, 2=ALL},
    Reference ID index {0=ext, 1=tip},
    AP band gain,
    LF band gain,
    AP hipass filter applied (1=ON)
    
    """
    
    if df_ks is None and df_irc is None:
        pass
        #raise ValueError("both df_ks and df_irc are empty")
    if df_ks is not None and df_irc is not None:                
        pass
       # raise ValueError("both df_ks and df_irc given. pass only one")
    if df_ks is not None and df_irc is None:        
        pass
    
    if df_ks is None and df_irc is not None:                
        
        pass
    def get_row_depth(depth,manual_last_channel,probe_type):
        if (pd.isna(manual_last_channel)) or (manual_last_channel is None) :        
            if probe_type in ['0']:#not 1.0 like
                tot_depth=depth[-1] - 175 # 175 is the length of the tip
                num_rows=int(tot_depth/20) # number of equally spaced rows, each having 2 channels
            elif probe_type in ['24','2013','2014','2020','2021']:
                tot_depth=depth[-1] - 175 # 175 is the length of the tip
                num_rows=int(tot_depth/15) # number of equally spaced rows, each having 2 channels
        #    manual_last_channel is not None
        elif probe_type in ['24','2013','2014','2020','2021']:
            tot_depth= (float(manual_last_channel)/2)* 15 #um distance between channels for np2.0
            num_rows=int(tot_depth/15) # number of equally spaced rows, each having 2 channels
        else:
            tot_depth= (float(manual_last_channel)/2)* 20 #um distance between channels for np1.0
            num_rows=int(tot_depth/20) # number of equally spaced rows, each having 2 channels
        
        row_depth=np.linspace(tot_depth,0,num_rows)
        channel_depth=np.repeat(row_depth, 2, axis=0)
        return row_depth,channel_depth,tot_depth
    ap_path = ap_path
    ap_meta_path = os.path.splitext(ap_path)[0]+'.meta'
    probe_type =glx.readMeta(ap_meta_path)['imDatPrb_type']
    
    # Use a regex to extract text inside parentheses
    def get_probe_site_info(ap_meta_path,Field):    
        import re
        
        meta_field=glx.readMeta(ap_meta_path)[Field]
        matches = re.findall(r'\((.*?)\)', meta_field)
        if glx.readMeta(ap_meta_path)['imDatPrb_type'] == '0':
            imDatPrb_type='NP1010'
            # Assume matches[0] holds the string e.g. 'PRB_1_4_0480_1,1,0,70'
            match_str = matches[0]
            
            # Split the string by commas
            fields = match_str.split(',')
            
            # Replace the first item with imDatPrb_type
            fields[0] = imDatPrb_type
            
            # Recombine the fields into a new string
            matches[0] = ",".join(fields)
            
            
            
        imDatPrb_type = re.findall(r'\((.*?)\)', meta_field)
        # if 'NP' in matches[0]: #in 'snsGeomMap'
        #          #        
        # Process each match: split by whitespace or comma, convert to integers, then wrap in a NumPy array
        arrays = []
        
        meta=matches[0]
        matches=matches[0::]
        for match in matches:    
            # Split using any whitespace or comma delimiter            
            parts = re.split(r'[\s,;:]+', match.strip())
            if 'AP' in parts[0] or 'SY' in parts[0]:      #'snsChanMap'        
               # parts=parts[1:-1]
               parts=[parts[0][2:len(parts[0])],parts[1], parts[2]]
            # Convert parts to integers
            if parts[0][0]=='N':
                parts[0]=parts[0][2::]
            try:
                numbers = list(map(int, parts))
            except:                
                print(f"numbers = list(map(int, parts)) failed in preprocessFunctions.py")
                IPython.embed()
            
            arrays.append(np.array(numbers))
        
            
        return meta,arrays
    
    def assign_electrode_regions_deepseek(positions, regions, z_positions_um):
        from scipy.interpolate import interp1d
        """
        Interpolate the positions of electrodes into physical space and assign corresponding brain regions.

        Parameters:
        - positions: array of float values representing depths in micrometers (physical space).
        - regions: list of strings representing brain regions corresponding to the depths in 'positions'.
        - z_positions_um: array of float values representing depths of electrodes on a probe (independent coordinate space).

        Returns:
        - A list of tuples, where each tuple contains:
          (electrode_depth_in_physical_space, corresponding_brain_region)
          
          example use: result = assign_electrode_regions_deepseek(positions, regions, z_positions_um)
        """
        # Ensure the positions are sorted for interpolation
        sorted_indices = np.argsort(positions)
        sorted_positions = np.array(positions)[sorted_indices]
        sorted_regions = np.array(regions)[sorted_indices]

        # Create an interpolation function for regions
        # We'll map positions to indices and interpolate the indices to find the closest region
        # This is a workaround since regions are categorical
        region_indices = np.arange(len(sorted_regions))
        f_region = interp1d(sorted_positions, region_indices, kind='nearest', fill_value='extrapolate')

        # Interpolate the physical positions for the electrodes
        # For simplicity, assume z_positions_um maps linearly to physical space
        # If the mapping is non-linear, you'll need additional transformation logic
        min_pos = np.min(positions)
        max_pos = np.max(positions)
        f_position = interp1d([np.min(z_positions_um), np.max(z_positions_um)], [min_pos, max_pos])

        electrode_positions_physical = f_position(z_positions_um)

        # Assign regions to each electrode
        electrode_regions = []
        for z_phys in electrode_positions_physical:
            if z_phys < min_pos:
                region = "out of brain"
            else:
                idx = int(np.round(f_region(z_phys)))
                region = sorted_regions[idx]
            electrode_regions.append((z_phys, region))

        return electrode_regions
    
    def assign_electrode_regions_gemini(positions, regions, z_positions_um):
        """
        Interpolates electrode positions from probe space into physical space
        and assigns brain regions to each electrode.

        Parameters:
        - positions (np.ndarray): Array of floats, histological depths (µm), increasing.
        - regions (list of str): List of brain regions for each position.
        - z_positions_um (np.ndarray): Array of electrode depths (µm) in probe space.

        Returns:
        - interp_positions (np.ndarray): Interpolated physical depths for each electrode.
        - assigned_regions (list of str): Brain regions assigned to each electrode.
        
        example use: 
            interp_positions, assigned_regions=assign_electrode_regions_gemini(positions, regions, z_positions_um)
        """
        positions = np.asarray(positions)
        z_positions_um = np.asarray(z_positions_um)

        # Ensure input sizes match
        if len(positions) != len(regions):
            raise ValueError("Length of positions must match length of regions.")

        # Sort positions and associated regions
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        #sorted_regions = [regions[i] for i in sorted_indices]
        sorted_regions = [regions.iloc[i] for i in sorted_indices]

        # Define interpolation from probe space to physical space
        probe_depth_range = [z_positions_um.min(), z_positions_um.max()]
        physical_depth_range = [sorted_positions.min(), sorted_positions.max()]

        # Normalize and interpolate electrode positions into physical space
        interp_positions = np.interp(
            z_positions_um,
            np.linspace(probe_depth_range[0], probe_depth_range[1], len(sorted_positions)),
            sorted_positions
        )

        # Assign regions based on interpolated physical positions
        assigned_regions = []
        for depth in interp_positions:
            if depth < 0:
                assigned_regions.append("out of brain")
            else:
                idx = np.searchsorted(sorted_positions, depth, side='right') - 1
                if idx < 0:
                    assigned_regions.append("out of brain")
                else:
                    assigned_regions.append(sorted_regions[idx])

        return interp_positions, assigned_regions
    
    def assign_electrode_regions_v2(positions, regions, z_positions_um):
        """
        Interpolates electrode positions from probe space into physical space
        and assigns brain regions to each electrode.
    
        Parameters:
        - positions (np.ndarray): Array of floats, histological depths (µm), increasing.
        - regions (list of str): List of brain regions for each position.
        - z_positions_um (np.ndarray): Electrode depths (µm) in probe coordinate space.
    
        Returns:
        - interp_positions (np.ndarray): Interpolated physical depths for each electrode.
        - assigned_regions (list of str): Brain regions assigned to each electrode.
        example:
             interp_positions, assigned_regions = assign_electrode_regions_v2(positions, regions, z_positions_um)
        """
        positions = np.asarray(positions)
        z_positions_um = np.asarray(z_positions_um)
    
        # Validate input sizes
        if len(positions) != len(regions):
            raise ValueError("Length of positions must match length of regions.")
    
        # Sort by depth in case not sorted
        sort_idx = np.argsort(positions)
        sorted_positions = positions[sort_idx]
        sorted_regions = np.array(regions)[sort_idx]
        
        # Clip z_positions_um to exclude the final 175 µm of the probe
        max_z = np.max(z_positions_um)
        clip_threshold = max_z - 175
        valid_idx = z_positions_um <= clip_threshold
        z_positions_um_clipped = z_positions_um[valid_idx]
    
        # Normalize z_positions_um to match physical space scale
        probe_depth_range = [z_positions_um.min(), z_positions_um.max()]
        position_range = [sorted_positions[0], sorted_positions[-1]]
    
        # Map z_positions_um to physical space (linear interpolation)
        interp_positions = np.interp(
            z_positions_um,
            np.linspace(probe_depth_range[0], probe_depth_range[1], len(sorted_positions)),
            sorted_positions
        )
    
        # Assign regions
        assigned_regions = []
        for z, interp_depth in zip(z_positions_um, interp_positions):
            if z < 0 or interp_depth < 0:
                assigned_regions.append("out of brain")
            else:
                idx = np.searchsorted(sorted_positions, interp_depth, side='right') - 1
                if idx < 0:
                    assigned_regions.append("out of brain")
                else:
                    assigned_regions.append(sorted_regions[idx])
    
        return interp_positions, assigned_regions

    def interpolate_region_acronym(csv_data: pd.DataFrame, snsGeomMap_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in snsGeomMap_df, find the nearest 'Distance from first position [um]' value from
        csv_data based on the row's 'z_coord_um', and assign the corresponding 'Region acronym'.
        
        Parameters:
          csv_data: pd.DataFrame with columns including:
                    - 'Distance from first position [um]'
                    - 'Region acronym'
          snsGeomMap_df: pd.DataFrame with a column 'z_coord_um' among other columns.
        
        Returns:
          snsGeomMap_df with an additional column 'Region acronym'
        """
        # Extract the positions and region acronyms from csv_data as numpy arrays.
        positions = csv_data['Distance from first position [um]'].values
        region_acronyms = csv_data['Region acronym'].values
    
        # Define a helper function to find the region acronym for a given z value.
        def get_region_acronym(z):
            idx = np.abs(positions - z).argmin()
            return region_acronyms[idx]
        
        # Apply the helper function to each row's 'z_coord_um'
        snsGeomMap_df = snsGeomMap_df.copy()  # Do not change original
        snsGeomMap_df['Region acronym'] = snsGeomMap_df['z_coord_um'].apply(get_region_acronym)
        
        return snsGeomMap_df
    #get imro table (sites)        
    imroTbl_meta,imroTbl = get_probe_site_info(ap_meta_path,'imroTbl')
    #probe_type=imroTbl[0][0]
    num_channels=imroTbl[0][1]
    imroTbl=imroTbl[1:-1]
    channels_sites=parse_imroTbl(probe_type, imroTbl)#a list of (channel,site) pairs
    channels_sites = [x for x, _ in channels_sites]
    
    #get channel mapping
    snsChanMap_meta, snsChanMap = get_probe_site_info(ap_meta_path,'snsChanMap')
    
    #get site geometry    
    """
    zero-based shank # (with tips pointing down, shank-0 is left-most),    x-coordinate (um) of elecrode center,    z-coordinate (um) of elecrode center,    
    0/1 "used," or, (u-flag), indicating if the electrode should be drawn in the FileViewer and ShankViewer windows, and if it should be included in spatial average <S> calculations.    
    (X,Z) Coordinates: Looking face-on at a probe with tip pointing downward, X is measured along the width of the probe, from left to right. The X-origin is the left edge of the shank. 
    Z is measured along the shank of the probe, from tip, upward, toward the probe base. The Z-origin is the center of the bottom-most elecrode row (closest to the tip).
    Note that for a multi-shank probe, each shank has its own (X,Z) origin and coordinate system. 
    That is, the (X,Z) location of an electrode is given relative to its own shank (with tips pointing down, shank-0 is left-most).
    """
    tip_offset=175#um
    snsGeomMap_meta,snsGeomMap = get_probe_site_info(ap_meta_path,'snsGeomMap')
    snsGeomMap_df = pd.DataFrame(snsGeomMap, columns=["shank", "x_coord_um", "z_coord_um", "used_flag"])#get channel geometry
    snsGeomMap_df['Region acronym'] = np.nan# add a region column
    if snsGeomMap_df['shank'][0]>8:
        snsGeomMap_df = snsGeomMap_df.drop(snsGeomMap_df.index[0])
    snsGeomMap_df["z_coord_um"]=snsGeomMap_df["z_coord_um"] + tip_offset#distance from real tip
    
    snsGeomMap_df["z_coord_um"]=np.max(snsGeomMap_df["z_coord_um"])-snsGeomMap_df["z_coord_um"]# now distances are depth on the probe
    snsGeomMap_df = snsGeomMap_df.sort_values(by='shank')    # Sort the dataframe by the 'z_coord_um' column in ascending order
    snsGeomMap_df["z_coord_um"] = snsGeomMap_df["z_coord_um"].astype(np.float64)     
   
    regions_csv = pd.DataFrame()   
    all_probe_depths=[]
    channel_shanks=[]
    site_counter=0
    
    if type(probe_tracking)!=str and np.isnan(probe_tracking)==True:
        raise ValueError(f"probe_tracking file is not specified in csv file ")
    if probe_type in ['24','2013','2014','2020','2021']:#4 shank     #else probe_type in ['21','2003','2004','0','1020','1030','1100','1120','1121','1122','1123','1200','1300']:#1 shank                  
        if not ',' in probe_tracking: #chck input
            raise Warning(f'NP 2.0 4 shank probe detected, expected multiple histological track files seperated by: , ')
            IPython.embed()
    track_list = [track.strip() for track in probe_tracking.split(',')]                           
    
    for i, track in enumerate(track_list):# Loop through each index and path in track_list
        if len(track_list)>1:
            snsGeomMap_df_shank_i=snsGeomMap_df[snsGeomMap_df['shank'] == i]
            df_idx=np.where(snsGeomMap_df['shank'] ==i)
            
        else:
            snsGeomMap_df_shank_i=snsGeomMap_df
            df_idx=np.arange(len(snsGeomMap_df_shank_i))
        #brainglobe csv file data
        csv_data = pd.read_csv(track)# Read the CSV file from the current path        
        # Add a column 'shank_num' with the order/index of the file
        csv_data['shank_num'] = i#asign shank id        
        csv_data = csv_data[csv_data['Region acronym'] != "Not found in brain"]# Filter out rows where 'Region acronym' equals 'Not found in brain'
        
        csv_data['Distance from first position [um]']=csv_data['Distance from first position [um]']-csv_data['Distance from first position [um]'].iloc[0] #put zero depth on first track point in brain                
        track_depth=csv_data['Distance from first position [um]'].to_numpy()# Append the read data (with the new column) as rows to regions_csv        
        
        #interpolate site depths and regions
        positions =track_depth#histological tract
        regions = csv_data["Region acronym"] # region names
        z_positions_um=snsGeomMap_df["z_coord_um"]#site depths on the probe
       
        
        interp_positions, assigned_regions=assign_electrode_regions_gemini(positions, regions, z_positions_um)
        #interp_positions_v2, assigned_regions_v2 = assign_electrode_regions_v2(positions, regions, z_positions_um)               
        #result = assign_electrode_regions_deepseek(positions, regions, z_positions_um)
        # 1) Update the "z_coord_um" column with the values from interp_positions
       
        snsGeomMap_df_shank_i["z_coord_um"] = interp_positions[df_idx]

        # 2) Add a new column "brain_regions" with values from the array assigned_regions
        
            
        if len(df_idx)==len(channels_sites) or len(df_idx)==len(channels_sites)+1:
            regions_for_df = [assigned_regions[i] for i in df_idx]
            snsGeomMap_df.iloc[df_idx] = snsGeomMap_df_shank_i
        else:
            regions_for_df = [assigned_regions[i] for i in df_idx[0]]
            snsGeomMap_df.iloc[df_idx[0]] = snsGeomMap_df_shank_i.values
       # print(f"\nstopped in regions_for_df=.... in preprocessFunctions\n")
        #IPython.embed()
        snsGeomMap_df_shank_i["Region acronym"] = regions_for_df#assigned_regions[df_idx[0]]
       
        # 3) Replace the rows in snsGeomMap_df at indices specified by df_idx with the updated shank_i dataframe
        
        #snsGeomMap_df.loc[snsGeomMap_df_shank_i.index] = snsGeomMap_df_shank_i
        #
        
        
        
    #     tot_depth=np.max(track_depth)#total depth from csv 
    #     tot_depth-=tip_offset         
    #     probe_depth=tot_depth-snsGeomMap_df_shank_i["z_coord_um"] 
        
        
    #     #get site span on current shank from meta file
    #     sites_span=np.max(snsGeomMap_df_shank_i['z_coord_um'])-np.min(snsGeomMap_df_shank_i['z_coord_um'])
    #     if len(track_list)==1:
    #         pass
    #     else:#multi shank -   need to add shank spacing to x coordinates          
    #         shank_pitch=250#NP2.0
    #         xcoords = snsGeomMap_df_shank_i[snsGeomMap_df_shank_i['shank'] == i]['x_coord_um'] +shank_pitch*(i)
    #     channel_shanks.append(snsGeomMap_df_shank_i['shank'].to_numpy())
         
       
        
        
    #     depth_to_keep=tot_depth-tip_offset  #remove tip disctance not covered by sites
    #     csv_data = csv_data[csv_data['Distance from first position [um]'] <= depth_to_keep]# keep only the part of the track the is covered by probe recorded sites
    #     track_depth=csv_data['Distance from first position [um]'].to_numpy()# Append the read data (with the new column) as rows to regions_csv        
    #     tot_depth=np.max(track_depth)#total depth from csv 
    #     # get probe vertical span in atlas coordinates
    #     depth_to_keep=tot_depth-sites_span# drop coordinates that aren't coverd by any sites
    #     csv_data = csv_data[csv_data['Distance from first position [um]'] >= depth_to_keep]# keep only the part of the track the is covered by probe recorded sites
    #     track_depth=csv_data['Distance from first position [um]'].to_numpy()# Append the read data (with the new column) as rows to regions_csv        
        
    #     track_span=np.max(csv_data['Distance from first position [um]'])-np.min(csv_data['Distance from first position [um]'])
    #     track_n=len(csv_data)
        
    #     regions_csv = pd.concat([regions_csv, csv_data], ignore_index=True)# add to overview dataframe
    #     all_probe_depths.append(probe_depth)
        
    #     #update snsGeomMap_df["z_coord_um"]
        
    #     snsGeomMap_df.iloc[site_counter:site_counter+len(probe_depth), snsGeomMap_df.columns.get_loc('z_coord_um')] = probe_depth.values
    #     site_counter+=len(probe_depth)
        
    # #shank_region_assignment=interpolate_region_acronym(csv_data, all_probe_depths)
    
    # snsGeomMap_df=interpolate_region_acronym(csv_data, snsGeomMap_df)
    #resort snsGeomMap_df by channels
    snsGeomMap_df.sort_index(inplace=True)
#    snsGeomMap_df = snsGeomMap_df.sort_index()
    
    channelregions=snsGeomMap_df['Region acronym'].values
    channel_depth=snsGeomMap_df['z_coord_um'].values
   
    
    

    #assign depth to each channel (also you need to determine num channels)            
    
    
    # get channel numbers corresponding to regions
    channelnums=np.arange(len(channel_depth))
    
    channel_shankNum = np.arange(len(channel_depth))#placeholder
    tot_depth=np.max(channel_depth)
    
    filtered_channelnums = [
    chan for chan, region in zip(channelnums, regions_for_df)
    if 'PAG' in region
    ]
    filtered_channelnums_int = [int(x) for x in filtered_channelnums]

# Print the integers separated by commas
    
    #print(f'estimated channels in PAG\n {filtered_channelnums}')
    print(", ".join(map(str, filtered_channelnums_int)))
    print(f'based on atlas alignment there is {channelnums[-1]} channels in the brain and \ndepth: {tot_depth}um.')
    print('CHECK THIS!!\n\n')
  
    return channelregions, channelnums


def readMeta(metaPath):
    #metaName = binFullPath.stem + ".meta"
    #metaPath = Path(binFullPath.parent / metaName)
    metaPath = Path(metaPath)
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
    n_spike_times=[]
    
    # Calculate the firing rate for each neuron and each time bin
    for i, n in enumerate(clusters):
        neuron_spike_times = spike_times[spike_clusters == n]
        spike_nums, _ = np.histogram(neuron_spike_times, bins=bins)
        n_by_t[i][:len(spike_nums)] = spike_nums   
        n_spike_times.append(neuron_spike_times)
        
    return n_by_t, bins, clusters, n_spike_times


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



def check_and_convert_variable(x, format_type=None):
    def convert_to_format(value, format_type):
        if format_type == float:
            return list(map(float, value.replace(',', ' ').split()))
        elif format_type == int:
            return list(map(lambda v: int(float(v)), value.replace(',', ' ').split()))
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

# def check_and_convert_variable(x, format_type=None):
#     def convert_to_format(value, format_type):
#         if format_type == float:
#             return list(map(float, value.split(',')))
#         elif format_type == int:
#             return list(map(int, value.split(',')))
#         else:
#             return value

#     if isinstance(x, np.ndarray):
#         return x
#     if isinstance(x, str):  # x is a string
#         if format_type:
#             x = convert_to_format(x, format_type)
#         elif '.' in x:
#             x = list(map(float, x.split(',')))
#         else:
#             x = list(map(int, x.split(',')))
#     elif isinstance(x, int):
#         if np.isnan(x):  # x is NaN
#             x = []  # Do nothing if x is NaN
#         else:  # x is a float
#             x = list(map(int, str(x).split(',')))
#     elif isinstance(x, list):  # x is a list
#         # No conversion needed for lists
#         return x
#     elif x is None:  # x is None
#         x = []
#     else:  # x is of an unknown type
#         x = []

#     return x

# %%  Head direction
def get_head_direction(ds,frame,positions_cm,nest_region_CM):
    import numpy as np
    from matplotlib import pyplot as plt
    import IPython
#    from movement import sample_data
    from movement.kinematics import (
    compute_forward_vector,
    compute_forward_vector_angle,
    )
    from movement.plots import plot_centroid_trajectory
    from movement.utils.vector import cart2pol, pol2cart
    position = positions_cm.squeeze()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(frame)

    # Plot the trajectory of the head centre
    plot_centroid_trajectory(
    ds.position,
    keypoints=["l_ear", "r_ear"],
    ax=ax,
    # arguments forwarded to plt.scatter
    s=10,
    cmap="viridis",
    marker="o",
    alpha=0.05,
    )
    # Adjust title
    ax.set_title("Head trajectory")
    ax.set_ylim(frame.shape[0], 0)  # match y-axis limits to image coordinates
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.collections[0].colorbar.set_label("Time (seconds)")
    fig.show()
    # %%
    # We can see that most of the head trajectory data is within a
    # cruciform shape, because the mouse is moving on an
    # `Elevated Plus Maze <https://en.wikipedia.org/wiki/Elevated_plus_maze>`_.
    # The plot suggests the mouse spends most of its time in the
    # covered arms of the maze (the vertical arms).
    
    # %%
    # Compute the head-to-nose vector
    # --------------------------------
    # We can define the head direction as the vector from the midpoint between
    # the ears to the nose.
    
    # Compute the head centre as the midpoint between the ears
    midpoint_ears = position.sel(keypoints=["l_ear", "r_ear"]).mean(
        dim="keypoints"
    )
    # nose position
    # (`drop=True` removes the keypoints dimension, which is now redundant)
    nose = position.sel(keypoints="nose", drop=True)
    
    # Compute the head vector as the difference vector between the nose position
    # and the head-centre position.
    head_to_nose = nose - midpoint_ears
    # Let's validate our computation by plotting the head-to-nose vector
    # alongside the midpoint between the ears and the nose position.
    # We will do this for a small time window to make the plot more readable.
    
    # Time window to restrict the plot
    time_window = slice(54.9, 55.1)  # seconds
    
    fig, ax = plt.subplots()
    
    # Plot the computed head-to-nose vector originating from the ears midpoint
    ax.quiver(
        midpoint_ears.sel(space="x", time=time_window),
        midpoint_ears.sel(space="y", time=time_window),
        head_to_nose.sel(space="x", time=time_window),
        head_to_nose.sel(space="y", time=time_window),
        color="gray",
        angles="xy",
        scale=1,
        scale_units="xy",
        headwidth=4,
        headlength=5,
        headaxislength=5,
        label="Head-to-nose vector",
    )
    
    # Plot midpoint between the ears within the time window
    plot_centroid_trajectory(
        midpoint_ears.sel(time=time_window),
        ax=ax,
        s=60,
        label="ears midpoint",
    )
    
    # Plot the nose position within the time window
    plot_centroid_trajectory(
        nose.sel(time=time_window),
        ax=ax,
        s=60,
        marker="*",
        label="nose",
    )
    
    # Calling plot_centroid_trajectory twice will add 2 identical colorbars
    # so we remove 1
    ax.collections[2].colorbar.remove()
    
    ax.set_title("Zoomed in head-to-nose vectors")
    ax.invert_yaxis()  # invert y-axis to match image coordinates
    ax.legend(loc="upper left")
    
    
    # %%
    # Head-to-nose vector in polar coordinates
    # -----------------------------------------
    # Now that we have the head-to-nose vector, we can compute its
    # angle in 2D space. A convenient way to achieve that is to convert the
    # vector from cartesian to polar coordinates using the
    # :func:`cart2pol()<movement.utils.vector.cart2pol>` function.
    
    head_to_nose_polar = cart2pol(head_to_nose)
    ds.attrs["head_to_nose_polar"] = head_to_nose_polar
    print(head_to_nose_polar)
    
    # %%
    # Notice how the resulting array has a ``space_pol`` dimension with two
    # coordinates: ``rho`` and ``phi``. These are the polar coordinates of the
    # head vector.
    #
    # .. admonition:: Polar coordinates
    #   :class: note
    #
    #   The coordinate ``rho`` is the norm (i.e., magnitude, length) of the vector.
    #   In our case, the distance from the midpoint between the ears to the nose.
    #
    #   The coordinate ``phi`` is the shortest angle (in radians) between the
    #   positive x-axis and the  vector, and ranges from :math:`-\pi` to
    #   :math:`\pi` (following the
    #   `atan2 <https://en.wikipedia.org/wiki/Atan2>`_ convention).
    #   The ``phi`` angle is positive if the rotation
    #   from the positive x-axis to the vector is in the same direction as
    #   the rotation from the positive x-axis to the positive y-axis.
    #   In the default image coordinate system, this means  ``phi`` will be
    #   positive if the rotation is clockwise, and negative if the rotation
    #   is anti-clockwise.
    #
    #   .. image:: ../_static/Cartesian-vs-Polar.png
    #     :width: 600
    #     :alt: Schematic comparing cartesian and polar coordinates
    
    # %%
    # ``movement`` also provides a ``pol2cart`` function to transform
    # data in polar coordinates to cartesian.
    # Note that the resulting ``head_to_nose_cart`` array has a ``space``
    # dimension with two coordinates: ``x`` and ``y``.
    
    head_to_nose_cart = pol2cart(head_to_nose_polar)
    print(head_to_nose_cart)
    ds.attrs["head_to_nose_cart"] = head_to_nose_cart
    # %%
    # Compute the "forward" vector
    # ----------------------------
    # We can also estimate the head direction using the
    # :func:`compute_forward_vector()<movement.kinematics.compute_forward_vector>`
    # function, which takes a different approach to the one we used above:
    # it accepts a pair of bilaterally symmetric keypoints and
    # computes the vector that originates at the midpoint between the keypoints
    # and is perpendicular to the line connecting them.
    #
    # Here we will use the two ears to find the head direction vector.
    # We may prefer this method if we expect the nose detection to be
    # unreliable (e.g., because it's often occluded in a top-down camera view).
    
    forward_vector = compute_forward_vector(
        position,
        left_keypoint="l_ear",
        right_keypoint="r_ear",
        camera_view="top_down",
    )
    print(forward_vector)
    ds.attrs["forward_vector"] = forward_vector
    
    # %%
    # .. admonition:: Why do we need to specify the camera view?
    #   :class: note
    #
    #   You can think about it in this way: in order to uniquely determine which
    #   way is forward for an animal, we need to know the orientation of the other
    #   two body axes: left-right and up-down. The left-right axis is specified
    #   by the left and right keypoints passed to the function, while we use the
    #   ``camera_view`` parameter to determine the upward direction (see image).
    #   The default view is ``"top_down"``, but it can also be ``"bottom_up"``.
    #   Other camera views are not supported at the moment.
    #
    #   .. image:: ../_static/Forward-Vector.png
    #     :width: 600
    #     :alt: Schematic showing forward vector in top-down and bottom-up views
    
    
    # %%
    # You can use ``compute_forward_vector`` to compute the perpendicular
    # vector to any line connecting two bilaterally symmetric keypoints.
    # For example, you could estimate the forward direction for the pelvis given
    # two keypoints at the hips.
    #
    # Specifically for the head direction vector, you may also use the alias
    # :func:`compute_head_direction_vector()\
    # <movement.kinematics.compute_head_direction_vector>`,
    # which makes the intent of the function clearer.
    
    # %%
    # Compute head direction angle
    # ----------------------------
    # We may want to explicitly compute the orientation of the animal's head
    # as an angle, rather than as a vector.
    # We can compute this angle from the forward vector as
    # we did with the head-to-nose vector, i.e., by converting the vector to
    # polar coordinates and extracting the ``phi`` coordinate. However, it's
    # more convenient to use the :func:`compute_forward_vector_angle()\
    # <movement.kinematics.compute_forward_vector_angle>` function, which
    # by default would return the same ``phi`` angle.
    
    forward_vector_angle = compute_forward_vector_angle(
        position,
        left_keypoint="l_ear",
        right_keypoint="r_ear",
        # Optional parameters:
        reference_vector=(1, 0),  # positive x-axis
        camera_view="top_down",
        in_degrees=False,  # set to True for degrees
    )
    print(forward_vector_angle)
    ds.attrs["forward_vector_angle"] = forward_vector_angle
    
    # %%
    # The resulting ``forward_vector_angle`` array contains the head direction
    # angle in radians, with respect to the positive x-axis. This means that
    # the angle is zero when the head vector is pointing to the right of the frame.
    # We could have also used an alternative reference vector, such as the
    # negative y-axis (pointing to the top edge of the frame) by setting
    # ``reference_vector=(0, -1)``.
    
    # %%
    # Visualise head direction angles
    # -------------------------------
    # We can compare the head direction angles computed from the two methods,
    # i.e. the polar angle ``phi`` of the head-to-nose vector and the polar angle
    # of the forward vector, by plotting their histograms in polar coordinates.
    # First, let's define a custom plotting function that will help us with this.


    def plot_polar_histogram(da, bin_width_deg=15, ax=None):
        """Plot a polar histogram of the data in the given DataArray.
    
        Parameters
        ----------
        da : xarray.DataArray
            A DataArray containing angle data in radians.
        bin_width_deg : int, optional
            Width of the bins in degrees.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the histogram.
    
        """
        n_bins = int(360 / bin_width_deg)
    
        if ax is None:
            fig, ax = plt.subplots(  # initialise figure with polar projection
                1, 1, figsize=(5, 5), subplot_kw={"projection": "polar"}
            )
        else:
            fig = ax.figure  # or use the provided axes
    
        # plot histogram using xarray's built-in histogram function
        da.plot.hist(
            bins=np.linspace(-np.pi, np.pi, n_bins + 1), ax=ax, density=True
        )
    
        # axes settings
        ax.set_theta_direction(-1)  # theta increases in clockwise direction
        ax.set_theta_offset(0)  # set zero at the right
        ax.set_xlabel("")  # remove default x-label from xarray's plot.hist()
    
        # set xticks to match the phi values in degrees
        n_xtick_edges = 9
        ax.set_xticks(np.linspace(0, 2 * np.pi, n_xtick_edges)[:-1])
        xticks_in_deg = (
            list(range(0, 180 + 45, 45)) + list(range(0, -180, -45))[-1:0:-1]
        )
        ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])
    
        return fig, ax
    
    
    # %%
    # Now we can visualise the polar ``phi`` angles of the ``head_to_nose_polar``
    # array alongside the values of the ``forward_vector_angle`` array.
    
    # sphinx_gallery_thumbnail_number = 3
    
    head_to_nose_angle = head_to_nose_polar.sel(space_pol="phi")
    ds.attrs["head_direction_phi"] = head_to_nose_angle
    angle_arrays = [head_to_nose_angle, forward_vector_angle]
    angle_titles = ["Head-to-nose", "Forward"]
    
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"}
    )
    for i, angles in enumerate(angle_arrays):
        title = angle_titles[i]
        ax = axes[i]
        plot_polar_histogram(angles, bin_width_deg=10, ax=ax)
        ax.set_ylim(0, 0.25)  # force same y-scale (density) for both plots
        ax.set_title(title, pad=25)
    
    # %%
    # We see that the angle histograms are not identical,
    # i.e. the two methods of computing head angle do not always yield
    # the same results.
    # How large are the differences between the two methods?
    # We could check that by plotting a histogram of the differences.
    
    angles_diff = forward_vector_angle - head_to_nose_angle
    fig, ax = plot_polar_histogram(angles_diff, bin_width_deg=10)
    ax.set_title("Forward vector angle - head-to-nose angle", pad=25)

# %%
# For the majority of the time, the two methods
# differ less than 20 degrees (2 histogram bins).
    
    
    # %%
    # Boundary angles
    # ---------------
    # Having observed the individuals' behaviour as they pass one another in the
    # ``ring_region``, we can begin to ask questions about their orientation with
    # respect to the nest. ``movement`` currently supports the computation of two
    # such "boundary angles";
    #
    # - The **allocentric boundary angle**. Given a region of interest :math:`R`,
    #   reference vector :math:`\vec{r}` (such as global north), and a position
    #   :math:`p` in space (e.g. the position of an individual), the allocentric
    #   boundary angle is the signed angle between the approach vector from
    #   :math:`p` to :math:`R`, and :math:`\vec{r}`.
    # - The **egocentric boundary angle**. Given a region of interest :math:`R`, a
    #   forward vector :math:`\vec{f}` (e.g. the direction of travel of an
    #   individual), and the point of origin of :math:`\vec{f}` denoted :math:`p`
    #   (e.g. the individual's current position), the egocentric boundary angle is
    #   the signed angle between the approach vector from :math:`p` to :math:`R`,
    #   and :math:`\vec{f}`.
    #
    # Note that egocentric angles are generally computed with changing frames of
    # reference in mind - the forward vector may be varying in time as the
    # individual moves around the arena. By contrast, allocentric angles are always
    # computed with respect to some fixed reference frame.
    #
    # For the purposes of our example, we will define our "forward vector" as the
    # velocity vector between successive time-points, for each individual - we can
    # compute this from ``positions`` using the ``compute_velocity`` function in
    # ``movement.kinematics``.
    # We will also define our reference frame, or "global north" direction, to be
    # the direction of the positive x-axis.
    
    # forward_vector = compute_velocity(positions_cm)
    # global_north = np.array([1.0, 0.0])
   
    
    # allocentric_angles = nest_region_CM.compute_allocentric_angle_to_nearest_point(
    #     ds.position,
    #     reference_vector=global_north,
    #     in_degrees=True,
    # )
    # egocentric_angles = nest_region_CM.compute_egocentric_angle_to_nearest_point(
    #     forward_vector,
    #     positions_cm[:-1],
    #     in_degrees=True,
    # )
    
    # angle_plot, angle_ax = plt.subplots(2, 1, sharex=True)
    # allo_ax, ego_ax = angle_ax
    # # Plot trajectories of the individuals
    # mouse_names_and_colours = list(
    #     zip(positions_cm.individuals.values, ["k","r", "g", "b"], strict=False)
    # )
    # for mouse_name, col in mouse_names_and_colours:
    #     allo_ax.plot(
    #         allocentric_angles.sel(individuals=mouse_name),
    #         c=col,
    #         label=mouse_name,
    #     )
    #     ego_ax.plot(
    #         egocentric_angles.sel(individuals=mouse_name),
    #         c=col,
    #         label=mouse_name,
    #     )
    
    # ego_ax.set_xlabel("Time (frames)")
    
    # ego_ax.set_ylabel("Egocentric angle (degrees)")
    # allo_ax.set_ylabel("Allocentric angle (degrees)")
    # allo_ax.legend()
    
    # angle_plot.tight_layout()
    # angle_plot.show()
    print("end of HD function")
    
    
    

# %%
# Allocentric angles show step-like behaviour because they only depend on an
# individual's position relative to the RoI, not their forward vector. This
# makes the allocentric angle graph resemble the distance-to-nest graph
# (inverted on the y-axis), with a "plateau" during frames 200-400 when the
# individuals are (largely) stationary while passing each other.
#
# Egocentric angles, on the other hand, fluctuate more due to their sensitivity
# to changes in the forward vector. Outside frames 200-400, we see trends:
#
# - ``AEON3B_TP2`` moves counter-clockwise around the ring, so its egocentric
#   angle decreases ever so slightly with time - almost hitting an angle of 0
#   degrees as it moves along the direction of closest approach after passing
#   the other individuals.
# - The other two individuals move clockwise, so their angles show a
#   gradual increase with time. Because the two individuals occasionally get in
#   each others' way, we see frequent "spikes" in their egocentric angles as
#   their forward vectors rapidly change.
#
# During frames 200-400, rapid changes in direction cause large fluctuations in
# the egocentric angles of all the individuals, reflecting the individuals'
# attempts to avoid colliding with (and to make space to move passed) each
# other.
    return ds
def get_arena_dim_from_movement(ds,session,animal,vframerate,**kwargs):
# %%
# Imports
# -------
    import IPython
    plt.close('all')
# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following lines in your notebook
# %matplotlib widget
    
    
    
    # Function to capture clicks
    def capture_points(title,arena_image):
        points = []
        def get_screen_size():
            import tkinter as tk
            # Create a temporary tkinter root to get screen dimensions
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width,screen_height

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'ro')  # Mark the point
                fig.canvas.draw()
    
        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(arena_image)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.title(title)
        px2inch=0.0104166667
        len(arena_image)
        # Resize the figure to full screen
        manager = plt.get_current_fig_manager()
        screen_width,screen_height=get_screen_size()
        image_height, image_width, _ = arena_image.shape
        
                # Calculate 66% of the full screen width
        width = screen_width * 0.33
        
        # Get the original aspect ratio of the image
        image_height, image_width, _ = arena_image.shape
        aspect_ratio = image_height / image_width
        
        # Calculate the height based on the original aspect ratio
        height = width * aspect_ratio
        
        # Set the figure size to 66% of the full screen width while maintaining the aspect ratio
        dpi = 100  # Dots per inch
        fig.set_size_inches(width / dpi, height / dpi)
        
        
        
    
        # Connect the click event to the function
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
        # Show the plot and wait for user interaction
        plt.show(block=True)
    
        # Disconnect the click event
        fig.canvas.mpl_disconnect(cid)
    
        return points
    
    
    paths=pp.get_paths(session=session,animal=animal)
    camera_video_path = paths['video']
    
    # Load the video
    video_path = camera_video_path
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if ret:
        # Save the frame as an image
        Filename=Path(paths['preprocessed']).joinpath('first_frame.jpg')
        cv2.imwrite(Filename, frame)
    
    # Release the video capture object
    cap.release()
    
    
    vpath= camera_video_path
    vframerate=vframerate
    
    #load sleap data using movement
    #ds = load_poses.from_sleap_file(paths['mouse_tracking'], vframerate)
    ds['frame_path']=Filename
    positions: xr.DataArray = ds.position
    
    # %%
    # The data we have loaded used an arena setup that we have plotted below.
    
    # arena_fig, arena_ax = plt.subplots(1, 1)
    # # Overlay an image of the experimental arena
    arena_image = frame
    #don't render anymore figures on screen
    # import matplotlib
    # plt.style.use('default')
    # matplotlib.use('Agg')  # Use a non-interactive backend

   
    
    # arena_ax.imshow(frame)
    
    # arena_ax.set_xlabel("x (pixels)")
    # arena_ax.set_ylabel("y (pixels)")
    
    # arena_fig.show()
    
    if kwargs:
  
       pixel_to_cm_x=kwargs['pixel_to_cm_x']
       pixel_to_cm_y=kwargs['pixel_to_cm_y']
       center_of_arena=kwargs['center_of_arena']
       nest_area_points = kwargs['nest_area_points']
       arena_width= kwargs['arena_width']
       arena_height=kwargs['arena_height']
        
    else:
        # Instructions for the user
        print("Click on the center of the arena, then close the plot window.")
        center_points = capture_points("Click on the center of the arena, then close the plot window",frame)
        
        # Ensure at least one point was clicked
        if len(center_points) == 0:
            raise ValueError("No points were clicked. Please run the script again and click on the image.")
        
        # Center of the arena
        center_of_arena = center_points[0]
        print(f"Center of Arena: {center_of_arena}")
        
        # Instructions for the user
        print("Click to define the vertices of the nest area polygon. Close the plot window when done.")
        nest_area_points = capture_points("Select nest of Arena edge points (minimum 3 points), then close the plot window",frame)
        
        # Ensure at least three points were clicked for the polygon
        if len(nest_area_points) < 3:
            raise ValueError("A polygon requires at least 3 points. Please run the script again and click on the image.")
        
      
        
        xy_edge_points = capture_points("click on horizontal,vertical edges of arena, then close the plot window",frame)
        
        # Prompt user for arena dimensions
        arena_width = float(input("Enter arena width  in cm: "))
        
        arena_height = float(input("Enter arena height cm: "))
        
        
        print(f"Arena Width: {arena_width}, Arena Height: {arena_height}")
    
    
    
        # Calculate pixel distances
        width_pixels = np.linalg.norm(np.array(xy_edge_points[0]) - np.array(xy_edge_points[1]))
        height_pixels = np.linalg.norm(np.array(xy_edge_points[2]) - np.array(xy_edge_points[3]))
        
        # Calculate conversion constants
        pixel_to_cm_x = arena_width / width_pixels
        pixel_to_cm_y = arena_height / height_pixels
    
    #Nest area polygon
    ds['arena_width'] = arena_width
    ds['arena_height'] = arena_height
    nest_area_polygon = nest_area_points
    print(f"Nest Area Polygon: {nest_area_polygon}")
   
    
    print(f"Pixel to cm conversion - X: {pixel_to_cm_x}, Y: {pixel_to_cm_y}")
    
    # Function to convert points from pixels to cm
    def convert_to_cm(points, pixel_to_cm_x, pixel_to_cm_y):
        return [(x * pixel_to_cm_x, y * pixel_to_cm_y) for x, y in points]
    
    # Convert center of arena
    center_of_arena_cm = (center_of_arena[0] * pixel_to_cm_x, center_of_arena[1] * pixel_to_cm_y)
    print(f"Center of Arena in cm: {center_of_arena_cm}")
    
    # Convert nest area polygon
    nest_area_polygon_cm = convert_to_cm(nest_area_polygon, pixel_to_cm_x, pixel_to_cm_y)
    print(f"Nest Area Polygon in cm: {nest_area_polygon_cm}")
    
    # Convert positions (assuming positions is a 2D array of shape (n, 2) for n points)
 #   from movement.transforms import scale
 #    scale(ds, factor=1.0, space_unit='centimeter')
    positions_cm = positions.copy()
    
    positions_cm[:, 0,:,:] *= pixel_to_cm_x  # Convert x-coordinates
    positions_cm[:, 1,:,:] *= pixel_to_cm_y  # Convert y-coordinates
    ds.attrs["position_cm"] = positions_cm.copy()
    
    center_mass=positions_cm.sel(keypoints='b_back')
    
    ds.attrs['center_mass'] = center_mass
    print(f"Positions in cm: {positions_cm}")
    
    #show arena in cm
    
    # Display the image
    plt.close('all') 
    fig, ax = plt.subplots()
    ax.imshow(frame)
    title='arena dimensions in centimeters'
    plt.title(title)
    ax.set_xlabel("X in CM")
    ax.set_ylabel("Y in CM")
    
    
    
    xticks = ax.get_xticks()
    xticks=xticks-xticks[0]
    xtickslabels=ax.get_xticklabels()
    yticks = ax.get_yticks()
    yticks=yticks-yticks[0]
    ytickslabels=ax.get_yticklabels()
    
    
    # Multiply each tick position by 3
    
    
    for i in range(len(xticks)):                  
        # Get the current text, convert it to a float, multiply by Px, and set it back
        value = float(xticks[i])
        xtickslabels[i].set_text(str(int(value * pixel_to_cm_x)))
    #    xtickslabels.append((str(int(xticks[i] * pixel_to_cm_x))))
    
    ax.set_xticklabels(xtickslabels)
    
    ytickslabels=[]
    for i in range(len(yticks)):                  
        # Get the current text, convert it to a float, multiply by Px, and set it back
        # value = float(xticks[i].get_text().replace('−', '-'))  # Handle negative sign
        ytickslabels.append((str(int(yticks[i] * pixel_to_cm_y))))
     
    ax.set_yticklabels(ytickslabels)
    
    
    
    plt.show()
    figfilename='arenaCM.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    
    #ax.figure.canvas.draw()
     
    # %%
    # The
    # `arena <https://sainsburywellcomecentre.github.io/aeon_docs/reference/hardware/habitat.html>`_
    # is divided up into three main sub-regions. The cuboidal structure
    # on the right-hand-side of the arena is the nest of the three individuals
    # taking part in the experiment. The majority of the arena is an open
    # octadecagonal (18-sided) shape, which is the bright central region that
    # encompasses most of the image. This central region is surrounded by a
    # (comparatively thin) "ring", which links the central region to the nest.
    # In this example, we will look at how we can use the functionality for regions
    # of interest (RoIs) provided by ``movement`` to analyse our sample dataset.
    
    # %%
    # Define regions of interest
    # --------------------------
    # In order to ask questions about the behaviour of our individuals with respect
    # to the arena, we first need to define the RoIs to represent the separate
    # pieces of our arena programmatically. Since each part of our arena is
    # two-dimensional, we will use a ``PolygonOfInterest`` to describe each of
    # them.
    #
    # In the future, the
    # `movement plugin for napari <../user_guide/gui.md>`_
    # will support creating regions of interest by clicking points and drawing
    # shapes in the napari GUI. For the time being, we can still define our RoIs
    # by specifying the points that make up the interior and exterior boundaries.
    # So first, let's define the boundary vertices of our various regions.
    
    
    
    # The centre of the arena is located roughly here
    centre = np.array(center_of_arena)
    # The "width" (distance between the inner and outer octadecagonal rings) is 40
    # pixels wide
    ring_width = 40.0
    # The distance between opposite vertices of the outer ring is 1090 pixels
    ring_extent = ((arena_width/pixel_to_cm_x+arena_height/pixel_to_cm_y)/2)
    ds.attrs["centre_Px"] = centre
    ds.attrs["ring_width_Px"] = ring_width
    ds.attrs["ring_extent_Px"] = ring_extent
    
   
    
    # Create the vertices of a "unit" octadecagon, centred on (0,0)
    n_pts = 18
    unit_shape = np.array(
        [
            np.exp((np.pi / 2.0 + (2.0 * i * np.pi) / n_pts) * 1.0j)
            for i in range(n_pts)
        ],
        dtype=complex,
    )
    # Then stretch and translate the reference to match our arena
    ring_outer_boundary = ring_extent / 2.0 * unit_shape.copy()
    ring_outer_boundary = (
        np.array([ring_outer_boundary.real, ring_outer_boundary.imag]).transpose()
        + centre
    )
    core_boundary = (ring_extent - ring_width) / 2.0 * unit_shape.copy()
    core_boundary = (
        np.array([core_boundary.real, core_boundary.imag]).transpose() + centre
    )
    
    nest_corners = tuple(nest_area_polygon)
    ds.attrs["nest_corners_Px"] = nest_corners
    # %%
    # Our central region is a solid shape without any interior holes.
    # To create the appropriate RoI, we just pass the coordinates in either
    # clockwise or counter-clockwise order.
    plt.close('all')
    central_region = PolygonOfInterest(core_boundary, name="Central region")
    central_region_CM = PolygonOfInterest(core_boundary * [pixel_to_cm_x,pixel_to_cm_y], name="Central region")
    
    # %%
    # Likewise, the nest is also just a solid shape without any holes.
    # Note that we are only registering the "floor" of the nest here.
    nest_region = PolygonOfInterest(nest_corners, name="Nest region")
    
    nest_corners_cm = [
    (x * pixel_to_cm_x, y * pixel_to_cm_y) for x, y in nest_corners
]
    nest_region_CM = PolygonOfInterest(nest_corners_cm, name="Nest region")
    ds= get_head_direction(ds,frame,positions_cm,nest_region_CM) # update ds with head direction
    
    # %%
    # To create an RoI representing the ring region, we need to provide an interior
    # boundary so that ``movement`` knows our ring region has a "hole".
    # ``movement``'s ``PolygonsOfInterest`` can actually support multiple
    # (non-overlapping) holes, which is why the ``holes`` argument takes a
    # ``list``.
    ring_region = PolygonOfInterest(
        ring_outer_boundary, holes=[core_boundary], name="Ring region"
    )
    
    ring_region_CM = PolygonOfInterest(
        ring_outer_boundary* [pixel_to_cm_x,pixel_to_cm_y], holes=[core_boundary* [pixel_to_cm_x,pixel_to_cm_y]], name="Ring region"
    )
    
    arena_fig, arena_ax = plt.subplots(1, 1)
    # Overlay an image of the experimental arena
    arena_ax.imshow(frame)
    
    central_region.plot(arena_ax, facecolor="lightblue", alpha=0.25)
    nest_region.plot(arena_ax, facecolor="green", alpha=0.25)
    ring_region.plot(arena_ax, facecolor="blue", alpha=0.25)
    arena_ax.legend()
    # sphinx_gallery_thumbnail_number = 2
    arena_fig.show()
    figfilename='arena_regions.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
    
    ds.attrs["central_region_Px"] = central_region
    ds.attrs["nest_region_Px"] = nest_region
    ds.attrs["ring_region_Px"] = ring_region
    
    ds.attrs["central_region_CM"] = central_region_CM
    ds.attrs["nest_region_CM"] = nest_region_CM
    ds.attrs["ring_region_CM"] = ring_region_CM
    
    # %%
    # View individual paths inside the arena
    # --------------------------------------
    # We can now overlay the paths that the individuals followed on top of our
    # image of the arena and the RoIs that we have defined.
    
    arena_fig, arena_ax = plt.subplots(1, 1)
    # Overlay an image of the experimental arena
    arena_ax.imshow(frame)
    
    central_region.plot(arena_ax, facecolor="lightblue", alpha=0.25)
    nest_region.plot(arena_ax, facecolor="green", alpha=0.25)
    #ring_region.plot(arena_ax, facecolor="blue", alpha=0.25)
    
    # Plot trajectories of the individuals
    mouse_names_and_colours = list(
        zip(positions.individuals.values, ["k","r", "g", "b"], strict=False)
    )
    for mouse_name, col in mouse_names_and_colours:
        plot_centroid_trajectory(
            positions,
            individual=mouse_name,
            ax=arena_ax,
            linestyle="-",
            marker=".",
            s=1,
            c=col,
            label=mouse_name,
        )
    arena_ax.set_title("Individual trajectories within the arena")
    #arena_ax.legend()
    
    arena_fig.show()
    figfilename='arena_centriod.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
    
    
    
    # %%
    # At a glance, it looks like all the individuals remained inside the
    # ring-region for the duration of the experiment. We can verify this
    # programmatically, by asking whether the ``ring_region``
    # contained the individuals' locations, at all recorded time-points.
    
    # This is a DataArray with dimensions: time, keypoint, and individual.
    # The values of the DataArray are True/False values, indicating if at the given
    # time, the keypoint of individual was inside ring_region.
   
    # individual_was_inside = ring_region.contains_point(positions)
    # all_individuals_in_ring_at_all_times = individual_was_inside.all()
    
    # if all_individuals_in_ring_at_all_times:
    #     print(
    #         "All recorded positions, at all times,\n"
    #         "and for all individuals, were inside ring_region."
    #     )
    # else:
    #     print("At least one position was recorded outside the ring_region.")
    
    # %%
    # Compute the distance to the nest
    # --------------------------------
    # Defining RoIs means that we can efficiently extract information from our data
    # that depends on the location or relative position of an individual to an RoI.
    # For example, we might be interested in how the distance between an
    # individual and the nest region changes over the course of the experiment. We
    # can query the ``nest_region`` that we created for this information.
    
    # Compute all distances to the nest; for all times, keypoints, and
    # individuals.
    #distances_to_nest = nest_region.compute_distance_to(positions)
    distances_to_nest_CM = nest_region_CM.compute_distance_to(positions_cm)
   
    distances_fig, distances_ax = plt.subplots(1, 1)
    for mouse_name, col in mouse_names_and_colours:
        distances_ax.plot(
            distances_to_nest_CM.sel(individuals=mouse_name),
            c=col,
            label=mouse_name,
        )
    #distances_ax.legend()
    distances_ax.set_xlabel("Time (frames)")
    distances_ax.set_ylabel("Distance to nest_region (CM)")
    distances_fig.show()
    ds.attrs["distances_to_nest_CM"] = distances_to_nest_CM
    
    distances_fig.show()
    figfilename='distances_to_nest_CM.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
  
    
    
    # %%
    # We can see that the ``AEON38_TP2`` individual appears to be moving towards
    # the nest during the experiment, whilst the other two individuals are
    # moving away from the nest. The "plateau" in the figure between frames 200-400
    # is when the individuals meet in the ``ring_region``, and remain largely
    # stationary in a group until they can pass each other.
    #
    # One other thing to note is that ``compute_distance_to`` is returning the
    # distance "as the crow flies" to the ``nest_region``. This means that
    # structures potentially in the way (such as the ``ring_region`` walls) are not
    # accounted for in this distance calculation. Further to this, the "distance to
    # a RoI" should always be understood as "the distance from a point to the
    # closest point within an RoI".
    #
    # If we wanted to check the direction of closest approach to a region, referred
    # to as the **approach vector**, we can use the ``compute_approach_vector``
    # method.
    # The distances that we computed via ``compute_distance_to`` are just the
    # magnitudes of the approach vectors.
    
    #approach_vectors = nest_region.compute_approach_vector(positions)
    #approach_vectors_CM = nest_region.compute_approach_vector(positions_cm)
    
    # %%
    # The ``boundary_only`` keyword
    # -----------------------------
    # From our plot of the distances to the nest, we saw a time-window
    # in which the individuals are grouped up, possibly trying to pass each other
    # as they approach from different directions.
    # We might be interested in whether they move to opposite walls of the ring
    # while doing so. To examine this, we can plot the distance between each
    # individual and the ``ring_region``, using the same commands as above.
    #
    # However, we get something slightly unexpected:
    
    #distances_to_ring_wall = ring_region.compute_distance_to(positions)
    print(f"computing distances_to_ringCM")
    distances_to_ring_wall_CM=ring_region_CM.compute_distance_to(positions_cm)
    distances_fig, distances_ax = plt.subplots(1, 1)
    for mouse_name, col in mouse_names_and_colours:
        distances_ax.plot(
            distances_to_ring_wall_CM.sel(individuals=mouse_name),
            c=col,
            label=mouse_name,
        )
    distances_ax.legend()
    distances_ax.set_xlabel("Time (frames)")
    distances_ax.set_ylabel("Distance to closest ring_region wall (CM)")
    
    print(
        "distances_to_ring_wall are all zero:",
        np.allclose(distances_to_ring_wall_CM, 0.0),
    )
    
    distances_fig.show()
    figfilename='distances_to_ringCM.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
    # %%
    # The distances are all zero because when a point is inside a region, the
    # closest point to it is itself.
    #
    # To find distances to the ring's walls instead, we can use
    # ``boundary_only=True``, which tells ``movement`` to only look at points on
    # the boundary of the region, not inside it.
    # Note that for 1D regions (``LineOfInterest``), the "boundary" is just the
    # endpoints of the line.
    #
    # Let's try again with ``boundary_only=True``:
    
    # distances_to_ring_wall = ring_region.compute_distance_to(
    #     positions, boundary_only=True
    # )
    print(f"computing distances_to_wall_CM")
    distances_to_ring_wall_CM = ring_region_CM.compute_distance_to(
        positions_cm, boundary_only=True
    )
    distances_fig, distances_ax = plt.subplots(1, 1)
    for mouse_name, col in mouse_names_and_colours:
        distances_ax.plot(
            distances_to_ring_wall_CM.sel(individuals=mouse_name),
            c=col,
            label=mouse_name,
        )
    distances_ax.legend()
    distances_ax.set_xlabel("Time (frames)")
    distances_ax.set_ylabel("Distance to closest ring_region wall (CM)")
    
    print(
        "distances_to_ring_wall are all zero:",
        np.allclose(distances_to_ring_wall_CM, 0.0),
    )
    
    distances_fig.show()
    figfilename='distances_to_wall_CM.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
    ds.attrs["distances_to_ring_wall_CM"] = distances_to_ring_wall_CM
    # %%
    # The resulting plot looks much more like what we expect, but is again
    # not very helpful; we get the distance to the closest point on *either*
    # the interior or exterior wall of the ``ring_region``. This means that we
    # can't tell if the individuals do move to opposite walls when passing each
    # other. Instead, let's ask for the distance  to just the exterior wall.
    
    # Note that the exterior_boundary of the ring_region is a 1D RoI (a series of
    # connected line segments). As such, boundary_only needs to be False.
    print(f"computing distances_to_exterior")
    distances_to_exterior_CM = ring_region_CM.exterior_boundary.compute_distance_to(
        positions_cm
    )
    distances_exterior_fig, distances_exterior_ax = plt.subplots(1, 1)
    for mouse_name, col in mouse_names_and_colours:
        distances_exterior_ax.plot(
            distances_to_exterior_CM.sel(individuals=mouse_name),
            c=col,
            label=mouse_name,
        )
    distances_exterior_ax.legend()
    distances_exterior_ax.set_xlabel("Time (frames)")
    distances_exterior_ax.set_ylabel("Distance to exterior wall (CM)")
    distances_exterior_fig.show()
    figfilename='distances_to_exterior.png'
    full_path = Path(paths['preprocessed']).joinpath(figfilename)
    plt.savefig(full_path)
    plt.close()
    
    ds.attrs["distances_to_exterior_CM"] = distances_to_exterior_CM
    ds["position_in_px"] =  positions
    ds["positions_cm"] = positions_cm
    


    # ds_filename=Path(rf"{paths['preprocessed']}").joinpath("ds.h5").as_posix()    
    
  
    # dt=ds.to_dict()
    # df.to_hdf(ds_filename)
   

  
  

    
    index=np.where(ds.keypoints.values=='b_back')[0][0]
    print("end of get_arena_dim_from_movement")
 
    return ds,distances_to_nest_CM.squeeze(), [pixel_to_cm_x,pixel_to_cm_y], nest_corners
    
    # %%
    # This output is much more helpful. We see that the individuals are largely the
    # same distance from the exterior wall during frames 250-350, and then notice
    # that;
    #
    # - Individual ``AEON_TP1`` moves far away from the exterior wall,
    # - ``AEON3B_NTP`` moves almost up to the exterior wall,
    # - and ``AEON3B_TP2`` seems to remain between the other two individuals.
    #
    # After frame 400, the individuals appear to go back to chaotic distances from
    # the exterior wall again, which is consistent with them having passed each
    # other in the ``ring_region`` and once again having the entire width of the
    # ring to explore.
    
    


def extract_sleap(paths, vframerate, node=None,view=None):


    """
    path: path to sleap file
    vpath: path to video
    node: which node to take for calculating velocity
    calculates velocity in cm per second (assuming the camera angle/ pixel number didn't change!!')
    # this function assumes x,y pixel to cm ratios are the same !!!!!!!!!!! - TODO


    """
    session=paths['session']
    animal=paths['Mouse_ID']
    if view=='top':
        mouse_tracking_path = paths['mouse_tracking']
        camera_video_path = paths['video']    
        Cm2Pixel_from_paths=paths['Cm2Pixel_xy']
        Shelter_xy_from_paths=paths['Shelter_xy']
        center_of_arena=paths['arena_center_xy']
    elif view=='bottom':
        mouse_tracking_path = paths['bottom_mouse_tracking']
        camera_video_path = paths['bottom_video']    
        Cm2Pixel_from_paths=paths['bottom_Cm2Pixel_xy']
        Shelter_xy_from_paths=paths['bottom_Shelter_xy']
        center_of_arena=paths['bottom_arena_center_xy']
    else:
        raise ValueError(f"view argument must be 'top' or 'bottom' ")
        
        
    arena_width_arena_height= paths['arena_width_arena_height']
    
    
    
    from movement.io import load_poses
    from matplotlib import pyplot as plt
    from scipy.signal import welch    
    import movement.kinematics as kin
    from movement.filtering import (
        interpolate_over_time,
        savgol_filter)
    
    
    vpath= camera_video_path
    vframerate=vframerate
    
    #load sleap data using movement
  

    # if  np.isnan(paths['mouse_tracking']) or Path(paths['mouse_tracking']).is_file() == False:
    #     raise ValueError('missing tracking file in csv file')
    print(f"{paths['mouse_tracking']=}")
    ds = load_poses.from_sleap_file(paths['mouse_tracking'], vframerate)
  
    # with h5py.File(mouse_tracking_path, "r") as f:
    #     #dset_names = list(f.keys())
    #     locations = np.squeeze(f["tracks"][:].T )#frames * nodes * x,y * animals
    #     node_names = np.array([n.decode() for n in f["node_names"][:]])
    
    #print mean confidence scores for each index
    print(ds.confidence.mean(dim='time').to_dataframe())
    # filter out keypoints with low confidence and replace outliers with nans
    ds.update({"position": filter_by_confidence(ds.position, ds.confidence, threshold=0.5)})
    #interpolate over missing values
    ds.update({"position": interpolate_over_time(ds.position, max_gap=1000)})
    
    # mean_loc=np.nanmean(locations, axis=1)
    # dist=hf.euclidean_distance_old(locations, mean_loc[:,None,:], axis=2) #dist is in frames (?)  
    # outlierframes=np.where(dist>75)[0] #this is specific for the round arena
    # locations[outlierframes]=np.nan

    # # fill in nans via interpolation 
    
    
    # Choose node locations
    #node_index=np.where(node_names==node)[0]

    try:
        locations = np.transpose(ds.position.sel(individuals='individual_0').to_numpy(), (0,2,1))
    except:
        locations = np.transpose(ds.position.sel(individuals='id_0').to_numpy(), (0,2,1))
    locations = fill_missing(locations)
    
    

    ##Distance to shelter
   
    if Cm2Pixel_from_paths is None or np.isnan(Cm2Pixel_from_paths).all() or Shelter_xy_from_paths is None or np.isnan(Shelter_xy_from_paths).all() or np.isnan(center_of_arena).all() or center_of_arena is None:
        locate_shelter = True
        print('click on shelter location in pop up plot. ')
        ds,distance2shelter, Cm2Pixel, shelterpoint=get_arena_dim_from_movement(ds,session,animal,vframerate)
        plt.close('all')
   #     distance2shelter, Cm2Pixel, shelterpoint=hf.get_shelterdist(paths, locations, vframerate, vpath, locate_shelter) #the loc is the locations from before but only for one node, vector is in cm
    else:
        locate_shelter=False
        kwargs={
            'center_of_arena': center_of_arena,# a tuple of two points: x,y
            'nest_area_points': Shelter_xy_from_paths,# a list of x,y tuples
            'pixel_to_cm_x': Cm2Pixel_from_paths[0],
            'pixel_to_cm_y': Cm2Pixel_from_paths[1] ,
            'arena_width': arena_width_arena_height[0],
            'arena_height': arena_width_arena_height[1]
            }
        ds,distance2shelter, Cm2Pixel, shelterpoint=get_arena_dim_from_movement(ds,session,animal,vframerate,**kwargs)# change later so can extract needed info from csv
        # Cm2Pixel=hf.check_and_convert_variable(Cm2Pixel_from_paths)
        # distance2shelter=(Shelter_xy_from_paths[0],Shelter_xy_from_paths[1])
        # loc=np.squeeze(locations[:,node_index,:])
        # loc=loc*Cm2Pixel
        # 
        # ds_filename = Path(paths['preprocessed']).joinpath("ds.h5")
        # if Path(ds_filename).is_file(): 
        #     ds2 = fetch_dataset(ds_filename)
        # else:
        #     kwargs={
        #         locate_shelter:False,
        #         Cm2Pixel:Cm2Pixel_from_paths,
        #         shelterpoint:shelterpoint                
        #         }
        
        # ds,distance2shelter, Cm2Pixel, shelterpoint=get_arena_dim_from_movement(ds,session,animal,vframerate,**kwargs)
        
        # distance2shelter, Cm2Pixel, shelterpoint=hf.get_shelterdist(paths, locations, vframerate, vpath, locate_shelter)
   
    #turn locations in cm
    if type(Cm2Pixel)==str:
        Cm2Pixel=check_and_convert_variable(Cm2Pixel,float)
    plt.close('all')
    ds['cm_to_pixel'] = Cm2Pixel
    locations[:,:,0] *= Cm2Pixel[0]
    locations[:,:,1] *= Cm2Pixel[1]
    
    
    
    ds['position'] = (('time', 'keypoints', 'space'), locations)
    
    #smoothing
    ds.update({"position": savgol_filter(ds.position, 25, polyorder=3)})
    locations=ds['position'].data
    #compute velocity
    ds['velocity'] = (('time', 'space', 'keypoints'), kin.compute_velocity(ds.positions_cm).squeeze().data)
    #compute speed
    ds['speed'] = (('time', 'keypoints'), kin.compute_speed(ds.positions_cm).squeeze().data)
    
    
   
    #save in dataset xarray
    #ds['position_in_cm'] = (('time', 'keypoints', 'space'), locations)
    ds.update({"position_cm": savgol_filter(ds.position_cm, 25, polyorder=3)})
    shelterpoint_list = list(shelterpoint)    
    ds.attrs['shelter']=(('space'), shelterpoint_list)      
    ds.attrs['distance_to_shelter'] = distance2shelter
    ds.update({"distance_to_shelter": savgol_filter(ds.distance_to_shelter, 50, polyorder=3)})
    

    
    

    
   
   
    
    # # #smoothing
    # smooth_node_loc = np.zeros_like((locations[:,0,:]))
    # for c in range(locations[:,0,:].shape[-1]):
    #     smooth_node_loc[:, c] = savgol_filter(locations[:,0,c], window_length=25, polyorder=3, deriv=1)
        

    # node_vel = np.linalg.norm(smooth_node_loc,axis=1)  
    # # convert to s 
    # node_vel*= vframerate #this is now cm/s
    # sqz_loc=np.squeeze(locations)
    

    return ds.speed.sel(keypoints=node).data, ds.position_cm.sel(keypoints=node).data, ds.keypoints.data, ds, Cm2Pixel, distance2shelter, shelterpoint_list



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
    if path.endswith('.tsv'):
        delimiter = '\t'
    else:
        delimiter = ','

    # Read the file using the appropriate delimiter
    data = pd.read_csv(path, delimiter=delimiter)
   
    
    #Get relevant columns/ data
    behaviours=data['Behavior'].to_numpy()
    behavioural_category=data['Behavioral category'].to_numpy()
    frames=data['Image index'].to_numpy()
    start_stop=data['Behavior type'].to_numpy()
    try:
        modifier=data['Modifier #1'].to_numpy()
    except:
        modifier=[]
        
    if np.sum(np.isnan(frames)): # sum nan values>0
        raise ValueError('There is nan values in boris frame numbers, go over the annotations again')

    return behaviours, behavioural_category, frames, start_stop, modifier





#%% general
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

def get_paths_mac(animal=None, session=None):
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
      
    paths=pd.read_csv(r"/Volumes/stem/data/project_hierarchy/data/paths-Copy.csv")
    
   
    if animal is not None:
        apaths=paths[paths['Mouse_ID']==animal]
        if session is not None:
            sesline=apaths[apaths['session']==session]
            return sesline.squeeze()
        return apaths
    
    elif animal is None:
        sesline=paths[paths['session']==session]
        return sesline.squeeze().astype(str)
    else:
        return paths
    
    
   
def convert_unc_path(win_path_str: str) -> Path:
    """
    Converts a Windows UNC path to a POSIX path. Also accounting for server names and shares.
    Author: chatGPT o3-mini
    """
    # Parse the Windows UNC path.
    p = PureWindowsPath(win_path_str)
    
    # p.parts for a UNC path is something like:
    # ('\\\\gpfs.corp.brain.mpg.de\\stem', 'data', 'project_hierarchy', 'data', 'paths-Copy.csv')
    if not p.parts:
        raise ValueError("Invalid UNC path format.")
    
    # The first part contains both server and share.
    unc_root = p.parts[0].lstrip("\\")  # e.g. "gpfs.corp.brain.mpg.de\\stem"
    try:
        server, share = unc_root.split("\\", 1)
        share = share.rstrip("\\")  # Remove trailing backslash if present
    except ValueError:
        raise ValueError("Invalid UNC path format.")
    
    # Map the UNC share to your local mount.
    if server.lower() == 'gpfs.corp.brain.mpg.de':
        local_root = Path("/gpfs") / share
    else:
        raise ValueError(f"Server '{server}' is not recognized. Please update the mapping.")

    # Use the remaining parts from the UNC path.
    tail_parts = p.parts[1:]
    return local_root.joinpath(*tail_parts)    

def maybe_convert(cell):
    """
    This must be applied to Pandas dataframe as
    my_dataframe.map(maybe_convert).
    Looks for all path strings and converts them to POSIX paths.
    author: chatGPT o3-mini
    
    """
    if isinstance(cell, str) and cell.startswith(r"\\gpfs.corp.brain.mpg.de"):
        try:
            # Convert the UNC path and return its string representation.
            return convert_unc_path(cell)
        except Exception as e:
            # Optionally log the exception e.
            return cell  # If conversion fails, return the original cell.
    else:
        return cell

def get_paths(animal=None, session=None,
        csvfile=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv",
        csvfile_posix="/gpfs/stem/data/project_hierarchy/data/paths-Copy.csv"):
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
    
    # Reads the full paths file 
    if os.name == 'nt':#windows
        csvfile=Path(csvfile).as_posix()
        try:
            paths=pd.read_csv(csvfile)
        except:
            raise ValueError(f"could not open {csvfile}. check you have access to the folder (GPFS) if running as administrator")
    elif os.name == 'posix': # Linux
        csvfile = Path(csvfile_posix)
        if not csvfile.exists():
            raise ValueError(f"Path {csvfile_posix} does not exist. Please check the path.")
        paths = pd.read_csv(csvfile)
         
    if animal is not None:
        apaths=paths[paths['Mouse_ID']==animal]
        _ret = apaths
        if session is not None:
            sesline=apaths[apaths['session']==session]
            _ret = sesline.squeeze()
    
    elif animal is None:
        sesline=paths[paths['session']==session]
        _ret = sesline.squeeze().astype(str)
    else:
        _ret = paths
    
    if os.name == 'nt':
        return _ret
    elif os.name == 'posix':
        _ret2 = _ret.map(maybe_convert)
        return _ret2


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

def check_for_bad_times(animal,session, t1_s, tend_s):    
    from pathlib import Path 
    import matplotlib
    import spikeinterface
    import spikeinterface.full as si
    import spikeinterface.extractors as se
    import spikeinterface.sorters as ss
    import spikeinterface.comparison as sc
    import spikeinterface.widgets as sw
    from spikeinterface.exporters import export_report
    from probeinterface import Probe, get_probe
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()
    from pathlib import Path
    import IPython
    job_kwargs=si.get_best_job_kwargs();
    from typing import Union
    import time
    import spikeinterface_helper_functions as sf
    import helperFunctions as hf                
    import preprocessFunctions as pp               
    #%matplotlib inline
    
    
    def find_problematic_periods_in_LFP(spikeglx_folder):
        
        # import matplotlib
        # import spikeinterface
        # import spikeinterface.full as si
        # import spikeinterface.extractors as se
        # import spikeinterface.sorters as ss
        # import spikeinterface.comparison as sc
        # import spikeinterface.widgets as sw
        # from spikeinterface.exporters import export_report
        # from probeinterface import Probe, get_probe
        # import numpy as np
        # import matplotlib.pyplot as plt
        # plt.ion()
        # from pathlib import Path
        # import IPython
        # job_kwargs=si.get_best_job_kwargs();
        # from typing import Union
        # import time
        # import spikeinterface_helper_functions as sf
        # import helperFunctions as hf
        
        spikeglx_folder=Path(spikeglx_folder).as_posix()
        try:
            recording = si.read_spikeglx(spikeglx_folder,stream_id='imec0.lf',load_sync_channel=False)
            
        except:
            print(f"\ndidnt find LF stream, switiching to ap\n")
            recording = si.read_spikeglx(spikeglx_folder,stream_id='imec0.ap',load_sync_channel=False)
            
        print(recording)
        
        t_start=recording.get_time_info()['t_start']
        recording.shift_times(shift=-t_start)
       # recording=recording.time_slice(start_time=t1_s, end_time=tend_s)
        t_start=recording.get_time_info()['t_start']
        
        #recording=recording.time_slice(t_start+3150, t_start+3200)
        
        t_stop=recording.get_duration()
        channel_ids = recording.get_channel_ids()
        selected_channels = channel_ids[::50]
        #selected_channels = [channel_ids[0]] #take only 1 channel for debugging
        print(F"using N={len(selected_channels)} channels")
        
    
        
        #recording = sf.preprocess_NP_rec(spikeglx_folder,kwargs,stream_id='')
        recording = si.phase_shift(recording)
        recording = si.highpass_filter(recording,300)
        # recording = si.bandpass_filter(
        #     recording,
        #     freq_min=300,
        #     freq_max=1250,
        
        #     margin_ms=1500.,
        #     filter_order=3,
        #     dtype="float32",
        #     add_reflect_padding=True,
        # )
        
        
        resample_rate=1250
        print(f"downsampling to {resample_rate}Hz...\n")
        recording = si.resample(recording, resample_rate=resample_rate, margin_ms=1000)
        from spikeinterface.preprocessing import scale_to_uV
        recording = scale_to_uV(recording)
        #physical_value = raw_value * gain + offset
        
        # Convert to physical units (whatever they may be)
        #recording_physical = scale_to_physical_units(recording)
        
        #sw.plot_traces(recording,time_range=(t_start+3175,t_start+3195),mode='line',channel_ids=selected_channels)
        fs= recording.get_sampling_frequency()
        print(f"getting samples...")
        all_traces = recording.get_traces(channel_ids=selected_channels, return_scaled=True)
        print(f"scanning for zeros and saturations...")
        
        #
        
        
        def find_zeros_for(all_traces,saturation_threshold = 2000,N_samples=10):
            from itertools import groupby
            from operator import itemgetter
            import numpy as np
            
            # Compute the mean across channels for each sample (axis=1)
            mean_traces = all_traces.mean(axis=1)
            
            # Find indices where the mean is exactly zero
            zero_or_above2000_inds = np.where((mean_traces == 0) | (mean_traces > saturation_threshold))[0]
           # zero_mean_inds = np.where(mean_traces == 0)[0]
            
            # Find runs of at least 10 consecutive zeros
            groups = []
            for k, g in groupby(enumerate(zero_or_above2000_inds), lambda ix: ix[0] - ix[1]):
                group = list(map(itemgetter(1), g))
                if len(group) >= N_samples:
                    groups.append(group)
            
            # Find indices where after a run of zeros, the next sample is non-zero
            after_zero_runs = []
            for group in groups:
                last_idx = group[-1]
                if last_idx + 1 < len(mean_traces) and mean_traces[last_idx + 1] != 0:
                    after_zero_runs.append(last_idx + 1)
            
            # Find margins: 3000 samples before the first zero and after the last zero in each group
            margin_in_samples=3000
            margins = []
            n_samples = len(mean_traces)
            for group in groups:
                first = group[0]
                last = group[-1]
                start_margin = max(0, first - margin_in_samples)
                end_margin = min(n_samples - 1, last + margin_in_samples)
                margins.append((start_margin, end_margin))
                margins_seconds = [(start / fs, end / fs) for start, end in margins]
               
            return margins_seconds
        
        def find_zeros_vectorized_GPT41(all_traces, fs=recording.get_sampling_frequency(),saturation_threshold = 2000,N_samples=30,margin_in_samples=0):
            
            mean_traces = all_traces.mean(axis=1)
            zero_or_above2000 = (mean_traces == 0) 
            idx = np.flatnonzero(zero_or_above2000)
        
            # Find runs of at least 10 consecutive indices
            if idx.size == 0:
                return []
        
            # Find the breaks between runs
            breaks = np.where(np.diff(idx) != 1)[0]
            run_starts = np.insert(idx[breaks + 1], 0, idx[0])
            run_ends = np.append(idx[breaks], idx[-1])
        
            # Filter runs by length >= 10
            valid = (run_ends - run_starts + 1) >= N_samples
            run_starts = run_starts[valid]
            run_ends = run_ends[valid]
        
           
            n_samples = len(mean_traces)
            margins = []
            for start, end in zip(run_starts, run_ends):
                start_margin = max(0, start - margin_in_samples)
                end_margin = min(n_samples - 1, end + margin_in_samples)
                margins.append((start_margin, end_margin))
        
            margins_seconds = [(start / fs, end / fs) for start, end in margins]
            return margins_seconds
        
        def find_zeros_and_saturations_vectorized_GPT41(
            all_traces, 
            fs, 
            saturation_threshold=2000, 
            return_threshold=50, 
            N_samples=30, 
            margin_in_samples=10
        ):
            """
            Find margins (in seconds) around:
            1. Runs of zeros in mean_traces (length >= N_samples)
            2. Runs where abs(mean_traces) exceeds saturation_threshold, until it returns below return_threshold
        
            Returns:
                margins_seconds_zeros: list of (start, end) in seconds for zero runs
                margins_seconds_saturation: list of (start, end) in seconds for saturation runs
            """
            
            mean_traces = all_traces.mean(axis=1)
            zero_or_above2000 = (mean_traces == 0) 
            idx = np.flatnonzero(zero_or_above2000)
        
            # Find runs of at least 10 consecutive indices
            if idx.size == 0:
                return []
        
            # Find the breaks between runs
            breaks = np.where(np.diff(idx) != 1)[0]
            run_starts = np.insert(idx[breaks + 1], 0, idx[0])
            run_ends = np.append(idx[breaks], idx[-1])
        
            # Filter runs by length >= 10
            valid = (run_ends - run_starts + 1) >= N_samples
            run_starts = run_starts[valid]
            run_ends = run_ends[valid]
        
           
            n_samples = len(mean_traces)
            margins = []
            for start, end in zip(run_starts, run_ends):
                start_margin = max(0, start - margin_in_samples)
                end_margin = min(n_samples - 1, end + margin_in_samples)
                margins.append((start_margin, end_margin))
        
            margins_seconds = [(start / fs, end / fs) for start, end in margins]
        
            # --- Saturation ---
            abs_trace = np.abs(mean_traces)
            above = abs_trace > saturation_threshold
            margins_saturation = []
            n_samples = len(mean_traces)
            i = 0
            while i < n_samples:
                if above[i]:
                    start = i
                    # Move forward until abs(mean_traces) < return_threshold
                    while i < n_samples and abs_trace[i] > return_threshold:
                        i += 1
                    end = i - 1
                    # Add margin
                    start_margin = max(0, start - margin_in_samples)
                    end_margin = min(n_samples - 1, end + margin_in_samples)
                    margins_saturation.append((start_margin, end_margin))
                else:
                    i += 1
            margins_seconds_saturation = [(start / fs, end / fs) for start, end in margins_saturation]
        
            return margins_seconds, margins_seconds_saturation
        
        def check_overlap(margins_second):
            # First, sort the intervals by their start times.
            sorted_intervals = sorted(margins_second, key=lambda x: x[0])
            
            merged_intervals = []
            for interval in sorted_intervals:
                # If there are no intervals in merged_intervals, simply add the interval.
                if not merged_intervals:
                    merged_intervals.append(interval)
                    continue
            
                prev_start, prev_end = merged_intervals[-1]
                curr_start, curr_end = interval
                
                # Check for overlap: if current start is less than or equal to the previous end,
                # they overlap and we need to merge.
                if curr_start <= prev_end:
                    # Merge by updating the end to the maximum of both intervals' ends.
                    merged_intervals[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    # No overlap, so append the current interval as is.
                    merged_intervals.append(interval)
            
            #print("Merged intervals:", merged_intervals)
            return merged_intervals
        
        #start_time = time.time()
        #margins_second= find_zeros_vectorized_o3(all_traces,fs)
        #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n{margins_second[0]}")
        #start_time = time.time()
        margins_second= find_zeros_vectorized_GPT41(all_traces,fs)
        margins_seconds_zeros, margins_seconds_saturation = find_zeros_and_saturations_vectorized_GPT41(
            all_traces, 
            fs, 
            saturation_threshold=800, 
            return_threshold=50, 
            N_samples=10, 
            margin_in_samples=10
        )
        
        all_margins=np.sort(margins_seconds_saturation+margins_second)#merge to a single list, sort by time
        all_margins=check_overlap(all_margins)#remove overlaps
        
        #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n")
        #start_time = time.time()
        #margins_second= find_zeros_for(all_traces)
        #print(f"Elapsed time: {time.time() - start_time:.6f} seconds\n{margins_second[0]}")
        
        
          
        
        # Create arrays for each field
        
        times = np.array([float(pair[0]) for pair in margins_seconds_zeros])#only start times
        zero_all_times = np.array([[float(pair[0]), float(pair[1])] for pair in margins_seconds_zeros])#also stop times
        zeros_duration = np.array([float(pair[1] - pair[0]) for pair in margins_second])
        label = np.array(["zeros"] * len(margins_seconds_zeros))
        zeros_duration_sum=np.sum(zeros_duration)
        # Assemble the dictionary
        zero_dict = {"time": time, "duration": zeros_duration, "label": label}
        
        
        
        sat_times = np.array([float(pair[0]) for pair in margins_seconds_saturation])#only start times
        sat__all_times = np.array([[float(pair[0]), float(pair[1])] for pair in margins_seconds_saturation])#also stop times
        sat_duration = np.array([float(pair[1] - pair[0]) for pair in margins_seconds_saturation])
        label = np.array(["saturations"] * len(margins_seconds_saturation))
        
        sat_dict= {"time": sat_times, "duration": sat_duration, "label": label}
        #all_values = [value+offset for pair in margins_second for value in pair]
        
        
        
        
        
        duration = recording.get_duration()
        
        # all_values = np.array([value for pair in margins_second for value in pair])
        # sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]],events=all_values,events_color='red',events_alpha=.5,return_scaled=True)
        
        # all_values = np.array([value for pair in margins_seconds_zeros for value in pair])
        # sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]],events=all_values,events_color='green',events_alpha=.5,return_scaled=True)
        
        all_values = np.array([value for pair in all_margins for value in pair])
        
        zeros_duration = np.array([float(pair[1] - pair[0]) for pair in all_margins])
        zeros_duration_sum=np.sum(zeros_duration)
        zeros_duration_sum/duration
        
        
        #plot behaviors on traces timeline
        sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]]
                        ,events=all_values,events_color='blue',events_alpha=.8,return_scaled=True)
        plt.savefig(Path.joinpath(out_path, 'bad_times.png')) 
        plt.show()
        
        #plot behaviors on traces timeline
        try:
         plt.close('all')
         frames_dropped, behaviour, ndata,n_spike_times, n_time_index, n_cluster_index,n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s=hf.load_preprocessed(animal, session)
         behaviour.iloc[0].video_start_s 
         behavior_times=np.array(behaviour.iloc[:].frames_s)   
         sw.plot_traces(recording,time_range=(t_start, int(t_start+duration)),mode='line',channel_ids=[channel_ids[0]]
                       ,events=np.array(behavior_times),events_color='red',events_alpha=.8,return_scaled=True)
         plt.savefig(Path.joinpath(out_path, 'behavior_times.png')) 
         plt.show()
        except:
            pass
        
        plt.close('all')
        step = 21
        T = np.arange(0, duration + step, step)
        
        # Suppose all_margins might have 1D arrays
        
        shared_bin_indices = []
        
        for margins in all_margins:
            # Ensure margins is at least 2D.
            margins2d = np.atleast_2d(margins)
            
            # Check that the second dimension is even so we can reshape it into pairs.
            if margins2d.shape[1] % 2 != 0:
                raise ValueError("The number of elements isn't even; cannot form start-stop pairs.")
            
            # Reshape each array so that each row represents an event [start, stop]
            events = margins2d.reshape(-1, 2)
            
            for event in events:
                start, stop = event
                
                # Determine bin indices for start and stop times.
                bin_index_start = np.digitize(start, T) - 1
                bin_index_stop  = np.digitize(stop, T) - 1
                
                # Only if both times fall in the same bin do we record the bin index.
                if bin_index_start == bin_index_stop:
                    shared_bin_indices.append(bin_index_start)
        shared_bin_indices=np.unique(shared_bin_indices)
        #print("Bin indices for events contained within a single bin:", shared_bin_indices)
        print({spikeglx_folder})
        print(f"\n duration:{duration:.2f}\n ")
        print(f"\n lost seconds:{zeros_duration_sum:.2f}\n ")
        print(f"\n lost seconds/duration :{zeros_duration_sum/duration:.5f}\n ")
        print(f"\n {len(shared_bin_indices)} chunks, {len(shared_bin_indices)*step} seconds of data will be lost \n")
        plt.show(block=False)
        
        return all_margins
    
    
    
    ###############################################################################
    
    #for session in sessions:
    paths=pp.get_paths(session=session,animal=animal)
    spikeglx_folder=Path(paths['lf']).parent
    out_path =Path(paths['preprocessed'])

    print(spikeglx_folder)
    
    all_margins = find_problematic_periods_in_LFP(spikeglx_folder)
    
    np.save(Path.joinpath(out_path, 'bad_times.npy'),all_margins)
    
    return all_margins

#%% labelling bad times in boris tags

from collections import defaultdict
import numpy as np
import math

def _normalize_and_merge_bad_times(bad_times, merge_bad=True):
    """Normalize bad_times: drop invalid/zero-length, fix flipped, sort, and merge if requested."""
    if not bad_times:
        return []
    clean = []
    for pair in bad_times:
        if pair is None or len(pair) != 2:
            continue
        bs, be = pair
        if bs is None or be is None:
            continue
        if isinstance(bs, float) and math.isnan(bs): continue
        if isinstance(be, float) and math.isnan(be): continue
        if bs == be:
            continue  # zero-length
        if bs > be:
            bs, be = be, bs
        clean.append((float(bs), float(be)))
    if not clean:
        return []
    clean.sort()
    if not merge_bad:
        return clean
    # merge overlapping/contiguous (positive-length by default; treat touching as contiguous merge here)
    merged = [clean[0]]
    for s, e in clean[1:]:
        ps, pe = merged[-1]
        if s <= pe:  # overlap or touch -> merge
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _events_to_intervals(behaviours, start_stop, boris_frames_s, boris_frames, behavioural_category):
    """
    Convert mixed event stream (with POINTs, interleaved behaviours) into well-formed intervals:
    returns [(beh, s_time, e_time, s_frame, e_frame, cat)], plus problems dict.
    """
    n = len(behaviours)
    if not (len(start_stop)==len(boris_frames_s)==len(boris_frames)==len(behavioural_category)==n):
        raise ValueError("All input lists must have the same length.")

    stacks = defaultdict(list)  # behaviour -> list of open STARTs
    intervals = []
    problems = {'unmatched_starts': [], 'unmatched_stops': []}

    for i in range(n):
        beh = behaviours[i]
        typ = start_stop[i]
        t   = float(boris_frames_s[i])
        fr  = int(boris_frames[i])
        cat = behavioural_category[i]

        if typ == 'POINT':
            continue
        if typ == 'START':
            stacks[beh].append({'s_time': t, 's_frame': fr, 'cat': cat})
        elif typ == 'STOP':
            if stacks[beh]:
                start = stacks[beh].pop()  # LIFO
                s_time = start['s_time']
                e_time = t
                if e_time <= s_time:
                    problems['unmatched_stops'].append(
                        f"{beh}: STOP @ {e_time} <= START @ {s_time} (idx {i})"
                    )
                else:
                    intervals.append((beh, s_time, e_time, start['s_frame'], fr, start['cat']))
            else:
                problems['unmatched_stops'].append(f"{beh}: stray STOP at t={t} (idx {i})")
        else:
            problems['unmatched_stops'].append(f"{beh}: unknown event type '{typ}' at idx {i}")

    for beh, stack in stacks.items():
        for start in stack:
            problems['unmatched_starts'].append(f"{beh}: unmatched START @ {start['s_time']}")

    intervals.sort(key=lambda x: x[1])  # by start time
    return intervals, problems

def _build_time_to_frame(boris_frames_s, boris_frames):
    """
    Create a vectorized time->frame interpolator using the original event stream.
    Works even if events are irregular; we sort by time and linearly interpolate.
    """
    t = np.asarray(boris_frames_s, float)
    f = np.asarray(boris_frames, int)
    # sort by time and deduplicate times
    order = np.argsort(t)
    t_sorted = t[order]
    f_sorted = f[order]
    # deduplicate identical times by keeping the first occurrence
    uniq_idx = np.concatenate(([0], np.where(np.diff(t_sorted) != 0)[0] + 1))
    t_u = t_sorted[uniq_idx]
    f_u = f_sorted[uniq_idx]
    def to_frame(times):
        times = np.asarray(times, float)
        # extrapolate using end values; round to nearest int
        frames = np.interp(times, t_u, f_u, left=f_u[0], right=f_u[-1])
        return np.rint(frames).astype(int)
    return to_frame

def label_bad_behaviours(
    behaviours,
    start_stop,
    boris_frames_s,
    boris_frames,
    behavioural_category,
    bad_times,
    touching_counts=False,   # False -> positive-length only; True -> touching counts as overlap
    strict_intervals=False,  # True -> raise on unmatched/invalid; False -> proceed & report problems
    merge_bad=True           # Merge overlapping/touching bad periods before insertion
):
    """
    1) Build behavior intervals from mixed events (ignoring POINT).
    2) Normalize (and optionally merge) bad_times.
    3) Relabel behavior intervals that overlap any bad period -> category 'ignore'.
    4) Insert new 'ignore' behaviour intervals for every bad period (with frames via interpolation).
    5) Return unified, time-sorted START/STOP streams.

    Returns
    -------
    behaviours_f, start_stop_f, times_f, frames_f, cats_f, problems
    """
    # Step 1: intervals from events
    intervals, problems = _events_to_intervals(
        behaviours, start_stop, boris_frames_s, boris_frames, behavioural_category
    )
    if strict_intervals and (problems['unmatched_starts'] or problems['unmatched_stops']):
        msg = "Invalid event stream:\n"
        if problems['unmatched_starts']:
            msg += "  Unmatched STARTS:\n    " + "\n    ".join(problems['unmatched_starts']) + "\n"
        if problems['unmatched_stops']:
            msg += "  STOP issues:\n    " + "\n    ".join(problems['unmatched_stops']) + "\n"
        raise ValueError(msg.rstrip())

    bad_norm=bad_times

    def overlaps(s, e, bs, be, closed):
        # closed=True → touching counts; closed=False → positive-length only
        return not ((e < bs or s > be) if closed else (e <= bs or s >= be))

    # Step 3: relabel behavior intervals’ categories if overlapping
    relabeled = []
    for beh, s_time, e_time, s_frame, e_frame, cat in intervals:
        cat_out = cat
        for (bs, be) in bad_norm:
            if overlaps(s_time, e_time, bs, be, touching_counts):
                cat_out = 'ignore'
                break
        relabeled.append((beh, s_time, e_time, s_frame, e_frame, cat_out))

    # Step 4: create 'ignore' intervals for each bad period with frames from interpolation
    to_frame = _build_time_to_frame(boris_frames_s, boris_frames)
    ignore_intervals = []
    if bad_norm:
        bs_arr = np.array([b[0] for b in bad_norm], float)
        be_arr = np.array([b[1] for b in bad_norm], float)
        fs = to_frame(bs_arr)
        fe = to_frame(be_arr)
        for (bs, be), s_frame, e_frame in zip(bad_norm, fs, fe):
            if touching_counts:
                # allow zero-length if caller wants touching to count; still emit as tiny interval
                if be < bs:  # should not happen after normalization
                    bs, be = be, bs
            else:
                if be <= bs:
                    continue  # skip non-positive length
            ignore_intervals.append(('ignore', bs, be, int(s_frame), int(e_frame), 'ignore'))

    # Step 5: fuse intervals and convert to event arrays
    all_intervals = relabeled + ignore_intervals

    # Build event list: (time, order, beh, START/STOP, frame, cat)
    # 'order' ensures START sorts before STOP at identical times
    events = []
    for beh, s_time, e_time, s_frame, e_frame, cat in all_intervals:
        events.append((s_time, 0, beh, 'START', int(s_frame), cat))
        events.append((e_time, 1, beh, 'STOP',  int(e_frame), cat))

    events.sort(key=lambda x: (x[0], x[1]))  # by time, START before STOP

    behaviours_f = np.array([e[2] for e in events], dtype=object)
    start_stop_f = np.array([e[3] for e in events], dtype=object)
    times_f      = np.array([e[0] for e in events], dtype=float)
    frames_f     = np.array([e[4] for e in events], dtype=int)
    cats_f       = np.array([e[5] for e in events], dtype=object)

    return behaviours_f, start_stop_f, times_f, frames_f, cats_f, problems
