# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:29:54 2025

@author: su-weisss
"""
from spikeinterface import create_sorting_analyzer, load_sorting_analyzer

from spikeinterface import load_extractor 
import numpy as np
import spikeinterface as si
import  spikeinterface.extractors as se

##

    
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

##
sampling_frequency = 30000.0
duration = 20.0
num_timepoints = int(sampling_frequency * duration)
num_units = 4
num_spikes = 1000

times0 = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
labels0 = np.random.randint(1, num_units + 1, size=num_spikes)

times1 = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
labels1 = np.random.randint(1, num_units + 1, size=num_spikes)

sorting = se.NumpySorting.from_times_labels([times0, times1], [labels0, labels1], sampling_frequency)
print(sorting)

print("Unit ids = {}".format(sorting.get_unit_ids()))
st = sorting.get_unit_spike_train(unit_id=1, segment_index=0)
print("Num. events for unit 1seg0 = {}".format(len(st)))
st1 = sorting.get_unit_spike_train(unit_id=1, start_frame=0, end_frame=30000, segment_index=1)
print("Num. events for first second of unit 1 seg1 = {}".format(len(st1)))


#https://spikeinterface.readthedocs.io/en/stable/tutorials/core/plot_2_sorting_extractor.html#sphx-glr-tutorials-core-plot-2-sorting-extractor-py


import pandas as pd
#file paths
csv1=r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\KS2_whiten_afm16924_supercat.imec0.ap_imec3b2.csv'
csv2=r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\KS2_whiten_afm16924_supercat.imec0.ap_imec3b2_quality.csv'

#import pandas as pd
#csv1='KS2_whiten_afm16924_supercat.imec0.ap_imec3b2.csv'
#csv2='KS2_whiten_afm16924_supercat.imec0.ap_imec3b2_quality.csv'
#read spikes times file
df=pd.read_csv(csv1, header=None)
df.columns = ['spike_times', 'unit_id', 'max_amp_site']

# Group by 'cluster_number' and 'max_amp_site', and aggregate spike times into lists
df_grouped = df.groupby(['unit_id', 'max_amp_site'])['spike_times'].apply(list).reset_index()


# Remove rows with negative cluster numbers
df_filtered = df[df['unit_id'] > 0]

#spiketimes group and sort by unit id 
df_grouped = df_filtered.groupby('unit_id')['spike_times'].apply(list).reset_index()

# Select the first max_amp_site for each cluster_number
df_grouped['max_amp_site'] = df_filtered.groupby('unit_id')['max_amp_site'].first().values

#read quality file
df_quality=pd.read_csv(csv2)

#merge to single dataframe
df_irc = pd.merge(df_grouped, df_quality, on='unit_id')
df_irc = df_irc[df_irc['note'] == 'single'] # keep only single units

print(df_irc.columns)



amplitudes = np.load(os.path.join(datapath, 'amplitudes.npy'))
channel_positions = np.load(os.path.join(datapath, 'channel_positions.npy'))
spike_clusters = np.load(os.path.join(datapath, 'spike_clusters.npy'))
spike_times = df_irc.spike_times[::]
df_irc['xpos'][ 'ypos']
spike_positions = np.load(os.path.join(datapath, 'spike_positions.npy'))




# Group by 'cluster_number' and 'max_amp_site', and aggregate spike times into lists
df_grouped = df_filtered.groupby(['cluster_number', 'max_amp_site'])['spike_times'].apply(list).reset_index()

sampling_frequency=30_000
sorting = se.NumpySorting.from_times_labels([times0,  [labels0 ], sampling_frequency)
# df_grouped is your desired dataframe


#times =  df.iloc[:, 0]
#clusters =df.iloc[:, 1] 
#sites =df.iloc[:, 2] 

#df=pd.read_csv(csv2)
#for column in df.columns:
#    globals()[column] = df[column]


import pynapple as nap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

spikes = nap.TsGroup(my_ts)
spikes['clustID'] = np.array(clusterID)
spikes['posX'] = np.array(chanPosX)
spikes['posY'] = np.array(chanPosY)
spikes.save(os.path.join(datapath, 'clusters.npz'))
plt.figure()
plt.subplot(211)
for n in range(len(spikes)):
 plt.eventplot(spikes[n].t,lineoffsets = n,linelengths = 0.3)
plt.xlabel('time (s)')
plt.ylabel('unit #')
