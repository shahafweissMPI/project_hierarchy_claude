# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:12:53 2025

@author: su-weisss
"""

# download channel maps for probes
from kilosort.utils import download_probes
from kilosort.io import load_probe, save_to_phy
from kilosort import run_kilosort, DEFAULT_SETTINGS
import os
from pathlib import Path
import numpy as np
#plotting output
import IPython
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import pandas as pd

import platform
from pathlib import Path

import time
import os

# Set the OPENBLAS_NUM_THREADS environment variable
os.environ["OPENBLAS_NUM_THREADS"] = "20"
# Pause the script for 1 hour (3600 seconds)

# === INPUT: Only specify your data file path and desired output directory in Windows-style ===
raw_data_file_path = Path(r"F:\stempel\afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap.bin")
raw_out_dir = Path(r"F:\stempel\afm16924_240522_g0\afm16924_240522_g0_imec0")
probe_dir=Path('F:\stempel\afm16924_240522_g0\afm16924_240522_g0_imec0')
#print(probe_dir)
#download_probes(probe_dir)
probe_name = Path(r"F:\stempel\afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap_kilosortChanMap.mat")
#"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap_kilosortChanMap.mat"
#probe_name = Path(probe_dir / 'NeuroPix1_default.mat')
probe_name
probe=load_probe(probe_name)

# === Function to convert Windows-style path to Linux /mnt/* mount ===
def windows_to_linux_path(p: Path) -> Path:
    parts = p.parts
    if parts and parts[0].endswith(':'):
        drive_letter = parts[0][0].lower()
        return Path("/mnt") / drive_letter / Path(*parts[1:])
    return p

# === OS-aware Path Handling ===
if platform.system() == "Linux":
    data_file_path = windows_to_linux_path(raw_data_file_path)
    spikeglx_folder = data_file_path.parent
    out_dir = windows_to_linux_path(raw_out_dir)
else:
    data_file_path = raw_data_file_path
    spikeglx_folder = data_file_path.parent
    out_dir = raw_out_dir

# === Optional Debug Prints ===
print(f"spikeglx_folder: {spikeglx_folder}")
print(f"data_file_path:   {data_file_path}")
print(f"out_dir:          {out_dir}")



#sorting_path=Path(r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\240523\trial0\ephys\raw\afm16924_240523_g0\afm16924_240523_g0_imec0\kilosort4_jupyter_0_5303')

# spikeglx_folder=Path(r"F:\scratch\supercat_afm17365_241202_0_g0\afm17365_241202_0_g0_imec0")
# data_file_path = Path(r"F:\scratch\supercat_afm17365_241202_0_g0\afm17365_241202_0_g0_imec0\afm17365_241202_0_g0_tcat.imec0.ap.bin")
# out_dir=spikeglx_folder
# out_dir=Path("E:\scratch\KS4")

datadir=out_dir
filename=data_file_path
                 



tmin = float(0)
tmax = float('inf')#7271.942530,12574.965015

if type(tmax) == float('inf'):
    output_folder = out_dir / 'afm17365_kilosort4'
    
else:
    output_folder = out_dir / 'afm17365_kilosort4' f'_{tmin}_{tmax}'
results_dir=Path(output_folder)
acg_threshold = 0.1
ccg_threshold = 0.01


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
settings['Th_universal'] =  9.0
settings['Th_learned'] =  8.0
settings['tmin'] =  tmin
settings['tmax'] = tmax
#settings['nt'] = 61
# settings['shift'] = None
# settings['scale'] = None
# #settings['artifact_threshold'] = inf
# settings['nskip'] = 25
settings['whitening_range'] = 32
settings['highpass_cutoff'] = 300.0
settings['binning_depth'] = 5.0
settings['sig_interp'] = 20.0
settings['drift_smoothing'] = [0.5, 0.5, 0.5]
# settings['nt0min'] = 20
# settings['dmin'] = None
# settings['dminx'] = 32.0
settings['min_template_size'] = 10#margot 16.0 # default = 10
settings['template_sizes'] = 5#margot 8 # default = 5
settings['nearest_chans'] = 10#Margot 32 # default = 10
#settings['nearest_templates'] = 100
# settings['max_channel_distance'] = None
settings['templates_from_data'] = True
# settings['n_templates'] = 6
settings['n_pcs'] = 6#32
# settings['Th_single_ch'] = 6.0
settings['acg_threshold'] = acg_threshold
settings['ccg_threshold'] = ccg_threshold
settings['cluster_downsampling'] = 500 # default 20
settings['max_cluster_subset']=5000
settings['cluster_neighbors'] = 10 # default 10


# settings['x_centers'] = None
settings['duplicate_spike_ms'] = 0.2

print(settings)
print(f"{results_dir}")
#results_dir = Path('F:/scratch/KS4_v4')
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Running kilosort with settings: {settings}")
print(f"Probe: {probe_name}")
print(f"Data file: {filename}")
from kilosort.io import save_to_phy
import kilosort
combined_list = list(range(0, 40)) + list(range(210, 383))
bad_channels= combined_list
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings,
                 probe=probe,probe_name=probe_name,
                filename=filename,data_dir=datadir,results_dir=results_dir,
                 do_CAR=False,verbose_log=False,verbose_console=False,
                 save_extra_vars=True,save_preprocessed_copy=False,bad_channels=bad_channels,clear_cache=True)

