# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:49:25 2025

@author: su-weisss
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:45:26 2025

@author: su-weisss
"""

# download channel maps for probes
from kilosort.utils import download_probes
from kilosort.io import load_prob, save_to_phy
from kilosort import run_kilosort, DEFAULT_SETTINGS
import os
from pathlib import Path
import numpy as np
#plotting output
import IPython
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import pandas as pd

spikeglx_folder = Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0")
data_file_path = Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap.bin")
#spikeglx_folder = Path(r"B:\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0")
#data_file_path = Path(r"B:\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap.bin")

datadir=spikeglx_folder
filename=data_file_path
                 
probe_dir=Path(r'F:\scratch\probes')
#download_probes(probe_dir)
probe_name = Path(probe_dir / 'NeuroPix1_default.mat')
probe=load_probe(probe_name)
tmin = 7271.95
tmax = 12574.965015#float('inf')7271.942530,12574.965015
if type(tmax) == 'int':
    output_folder = spikeglx_folder / 'kilosort4_spyder2' 
else:
    output_folder = spikeglx_folder / 'kilosort4_spyder'
results_dir=Path(output_folder)
acg_threshold = 0.1
ccg_threshold = 0.05


# get sampling rate

for meta_filename in Path(spikeglx_folder).glob("*ap.meta"):
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

# customize settings
settings={}
settings['n_chan_bin']=385
         
# settings = DEFAULT_SETTINGS
settings['data_dir'] = spikeglx_folder
#settings['probe_name'] = probe_name
# settings['data_file_path'] = data_file_path
# settings['results_dir'] = output_folder
# settings['filename'] = data_file_path

# settings['do_CAR'] = False
# settings['invert_sign'] = False
# settings['NTbuff'] = 60122
# settings['n_chan_bin'] = 385
# settings['Nchan'] = 383
# settings['torch_device'] = 'cuda'

#settings['data_dtype'] = 'int16'
settings['fs'] = fs
settings['batch_size'] = 60000
#settings['nblocks'] = 1
#settings['Th_universal'] =  9.0
#settings['Th_learned'] =  8.0
settings['tmin'] =  tmin
settings['tmax'] = tmax
# settings['nt'] = 61
# settings['shift'] = None
# settings['scale'] = None
# #settings['artifact_threshold'] = inf
# settings['nskip'] = 25
# settings['whitening_range'] = 32
# settings['highpass_cutoff'] = 300.0
# settings['binning_depth'] = 5.0
# settings['sig_interp'] = 20.0
# settings['drift_smoothing'] = [0.5, 0.5, 0.5]
# settings['nt0min'] = 20
# settings['dmin'] = None
# settings['dminx'] = 32.0
settings['min_template_size'] = 16.0 # default = 10
settings['template_sizes'] = 8 # default = 5
settings['nearest_chans'] = 32 # default = 10
# settings['nearest_templates'] = 100
# settings['max_channel_distance'] = None
# settings['templates_from_data'] = True
# settings['n_templates'] = 6
settings['n_pcs'] = 9
# settings['Th_single_ch'] = 6.0
settings['acg_threshold'] = acg_threshold
settings['ccg_threshold'] = ccg_threshold
settings['cluster_downsampling'] = 10 # default 20
# settings['x_centers'] = None
# settings['duplicate_spike_ms'] = 0.25
#settings['save_preprocessed_copy'] = True


# del settings['data_file_path']
# del settings['invert_sign']
# del settings['NTbuff']
# del settings[ 'Nchan']
# del settings['torch_device']
# del settings['data_dtype']
# del settings['save_preprocessed_copy']

print(settings)
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings,probe=probe,probe_name=probe_name,
                 filename=filename,data_dir=datadir,results_dir=results_dir,
                 do_CAR=False,verbose_log=True,verbose_console=False,save_extra_vars=True,save_preprocessed_copy=True)
    
#kilosort.io.save_to_phy(st=st, clu=clu, tF=tF, Wall=Wall, probe=probe, ops=ops, imin=np.int64(tmax*fs)-in64(tmin*fs), results_dir=Path(r'E:\scratch\KS4_phy'),data_dtype='int16', save_extra_vars=True,  save_preprocessed_copy=False)
    
IPython.embed()    
# rcParams['axes.spines.top'] = False
# rcParams['axes.spines.right'] = False
# gray = .5 * np.ones(3)

# fig = plt.figure(figsize=(10,10), dpi=100)
# grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

# ax = fig.add_subplot(grid[0,0])
# ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
# ax.set_xlabel('time (sec.)')
# ax.set_ylabel('drift (um)')

# ax = fig.add_subplot(grid[0,1:])
# t0 = 0
# t1 = np.nonzero(st > ops['fs']*5)[0][0]
# ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
# ax.set_xlim([0, 5])
# ax.set_ylim([chan_map.max(), 0])
# ax.set_xlabel('time (sec.)')
# ax.set_ylabel('channel')
# ax.set_title('spikes from units')
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
chan_map =  np.load(results_dir / 'channel_map.npy')
templates =  np.load(results_dir / 'templates.npy')
chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
chan_best = chan_map[chan_best]
amplitudes = np.load(results_dir / 'amplitudes.npy')
st = np.load(results_dir / 'spike_times.npy')
clu = np.load(results_dir / 'spike_clusters.npy')
firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
dshift = ops['dshift']    

## plotting
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
gray = .5 * np.ones(3)

fig = plt.figure(figsize=(10,10), dpi=100)
grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

ax = fig.add_subplot(grid[0,0])
ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
ax.set_xlabel('time (sec.)')
ax.set_ylabel('drift (um)')

ax = fig.add_subplot(grid[0,1:])
t0 = 0
t1 = np.nonzero(st > ops['fs']*5)[0][0]
ax.scatter(st/ops['fs'], chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
ax.set_xlim([0, 5])
ax.set_ylim([chan_map.max(), 0])
ax.set_xlabel('time (sec.)')
ax.set_ylabel('channel')
ax.set_title('spikes from units')

ax = fig.add_subplot(grid[1,0])
nb=ax.hist(firing_rates, 20, color=gray)
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,1])
nb=ax.hist(camps, 20, color=gray)
ax.set_xlabel('amplitude')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,2])
nb=ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color=gray)
ax.plot([10, 10], [0, nb[0].max()], 'k--')
ax.set_xlabel('% contamination')
ax.set_ylabel('# of units')
ax.set_title('< 10% = good units')

for k in range(2):
    ax = fig.add_subplot(grid[2,k])
    is_ref = contam_pct<10.
    ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
    ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xlabel('firing rate (Hz)')
    ax.legend()
    if k==1:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('loglog')
        
##
probe = ops['probe']
# x and y position of probe sites
xc, yc = probe['xc'], probe['yc']
nc = 16 # number of channels to show
good_units = np.nonzero(contam_pct <= 0.1)[0]
mua_units = np.nonzero(contam_pct > 0.1)[0]


gstr = ['good', 'mua']
for j in range(2):
    print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('title = number of spikes from each unit')
    units = good_units if j==0 else mua_units
    fig = plt.figure(figsize=(12,3), dpi=150)
    grid = gridspec.GridSpec(2,20, figure=fig, hspace=0.25, wspace=0.5)

    for k in range(40):
        wi = units[np.random.randint(len(units))]
        wv = templates[wi].copy()
        cb = chan_best[wi]
        nsp = (clu==wi).sum()

        ax = fig.add_subplot(grid[k//20, k%20])
        n_chan = wv.shape[-1]
        ic0 = max(0, cb-nc//2)
        ic1 = min(n_chan, cb+nc//2)
        wv = wv[:, ic0:ic1]
        x0, y0 = xc[ic0:ic1], yc[ic0:ic1]

        amp = 4
        for ii, (xi,yi) in enumerate(zip(x0,y0)):
            t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
            t /= wv.shape[0] / 20
            ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color='k')

        ax.set_title(f'{nsp}', fontsize='small')
        ax.axis('off')
    plt.show()
    

###############
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kilosort.io import load_ops
from kilosort.data_tools import (
    mean_waveform, cluster_templates, get_good_cluster, get_cluster_spikes,
    get_spike_waveforms, get_best_channels
    )
# Pick a random good cluster
cluster_id = get_good_cluster(results_dir, n=1)

# Get the mean spike waveform and mean templates for the cluster
mean_wv, spike_subset = mean_waveform(cluster_id, results_dir, n_spikes=100,
                                      bfile=None, best=True)
mean_temp = cluster_templates(cluster_id, results_dir, mean=True,
                              best=True, spike_subset=spike_subset)

# Get time in ms for visualization
ops = load_ops(results_dir / 'ops.npy')
t = (np.arange(ops['nt']) / ops['fs']) * 1000

fig, ax = plt.subplots(1,1)
ax.plot(t, mean_wv, c='black', linestyle='dashed', linewidth=2, label='waveform')
ax.plot(t, mean_temp, linewidth=1, label='template')
ax.set_title(f'Mean single-channel template and spike waveform for cluster {cluster_id}')
ax.set_xlabel('Time (ms)')
ax.legend()

%matplotlib ipympl
# Get n spike times for this cluster
spike_times, _ = get_cluster_spikes(cluster_id, results_dir, n_spikes=100)
# Time in s for spike time axis
t2 = spike_times / ops['fs']
# Get single-channel waveform for each spike
chan = get_best_channels(results_dir)[cluster_id]
waves = get_spike_waveforms(spike_times, results_dir, chan=chan)

# Plot each waveform, using spike time as 3rd dimension
fig, ax = plt.subplots(1, 1, figsize=(6,6), subplot_kw={'projection': '3d'})
for i in range(waves.shape[1]):
    # TODO: color by spike time
    ax.plot(t, t2[i], zs=waves[:,i], zdir='z');
ax.set_xlabel('Time (ms)');
ax.set_ylabel('Spike time (s)');
ax.view_init(azim=-100, elev=20);
ax.set_title(f'Spike waveforms for cluster {cluster_id}')
ax.set_box_aspect(None, zoom=0.85)

plt.tight_layout()

#######
# Can also visualize this as a heatmap
fig2, ax2 = plt.subplots(1,1,figsize=(6,6))
pos = ax2.imshow(waves.T, aspect='auto', extent=[t[0], t[-1], t2[0], t2[-1]]);
fig2.colorbar(pos, ax=ax2);
ax2.set_xlabel('Time (ms)');
ax2.set_ylabel('Spike time (s)');
