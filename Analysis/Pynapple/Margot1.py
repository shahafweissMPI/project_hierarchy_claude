import pynapple as nap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
# specify dir and ks output
dirPath = r"A:\data\np2\afm17307\concat\supercat\supercat_afm17307_241024_hunting_pups_escape_g0\afm17307_241024_hunting_pups_escape_g0_imec0"
datapath = os.path.join(dirPath, r'kilosort4_long\sorter_output') 
from pathlib import Path
for meta_filename in Path(dirPath).glob("*ap.meta"):
 print(meta_filename)

word = 'imSampRate'
with open(meta_filename, 'r') as meta_info:
 # read all lines in a list
 lines = meta_info.readlines()
 for line in lines:
 # check if string present on a current line
  if line.find(word) != -1:
   index = lines.index(line)

line = lines[index]
fs = line[11:]
fs = float(fs)
print(fs)
# Load the data

# phy .npy outputs:
amplitudes = np.load(os.path.join(datapath, 'amplitudes.npy'))
channel_positions = np.load(os.path.join(datapath, 'channel_positions.npy'))
spike_clusters = np.load(os.path.join(datapath, 'spike_clusters.npy'))
spike_times = np.load(os.path.join(datapath, 'spike_times.npy'))
spike_positions = np.load(os.path.join(datapath, 'spike_positions.npy'))

# non-standard .npy file containing mean waveforms for each cluster:
#mean_waveforms = np.load(os.path.join(datapath, 'mean_waveforms.npy'))


# tsv file containing cluster labels:
cluster_groups = pd.read_csv(os.path.join(datapath, 'cluster_group.tsv'),sep='\t')
type(cluster_groups)
#features_name = cluster_groups.columns.tolist()
#print(features_name)
#print(cluster_groups['group'].iloc[0] == 'good')
clusterID = cluster_groups['cluster_id'].values
print(clusterID)
my_ts = {}
chanPosX = []
chanPosY = []
count = 0
for index, n in enumerate(clusterID):
 spikeTimesN = spike_times[spike_clusters == n] / fs
 chanN = spike_positions[spike_clusters == n,:]
 my_ts[count] = np.array(spikeTimesN)
 chanX = np.mean(chanN[:,0],axis = 0)
 chanY = np.mean(chanN[:,1],axis = 0)
 chanPosX = np.append(chanPosX,chanX)
 chanPosY = np.append(chanPosY,chanY)
 count = count + 1 
spikes = nap.TsGroup(my_ts)
spikes['clustID'] = np.array(clusterID)
spikes['posX'] = np.array(chanPosX)
spikes['posY'] = np.array(chanPosY)
spikes.save(os.path.join(datapath, 'clusters.npz'))
#plt.figure()
#plt.subplot(211)
#for n in range(len(spikes)):
# plt.eventplot(spikes[n].t,lineoffsets = n,linelengths = 0.3)
#plt.xlabel('time (s)')
#plt.ylabel('unit #')