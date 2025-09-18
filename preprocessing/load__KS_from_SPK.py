# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 08:54:01 2025

@author: su-weisss
"""

import matplotlib
#%matplotlib inline
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

import spikeinterface_helper_functions as sf

spikeglx_folder = Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0")
data_file_path = Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\afm16924_240522_g0_tcat.imec0.ap.bin")

kwargs={
'bandpass':True,
'common_reference':True,
'detect_bad_channels':False,
'remove_channels':False,
'frame_slice':True,
'start_frame':7271.94,
'end_frame':12574}




recording = sf.preprocess_NP_rec(spikeglx_folder,kwargs)
fs=recording.get_sampling_frequency()


output_folder = spikeglx_folder / 'kilosort4_spyder' 
sorter_name = 'kilosort'
sorting_folder=output_folder

sorting = sf.load_sorting_from_folder(sorter_name, sorting_folder)
sorting.to_shared_memory_sorting()

sorting.save(format='npz_folder',folder=sorting_folder / 'spk',overwrite=True)
sorting=sf.load_sorting_from_folder(sorter_name='npz',sorting_folder=folder)
sf.load_sorting_from_folder(sorter_name: str, sorting_folder: str | Path)
sorting.frame_slice(int(kwargs['start_frame']*fs), int(kwargs['end_frame']*fs))
#sorting=sorting.register_recording(recording)
sorting=spikeinterface.curation.remove_excess_spikes(sorting,recording)
#sorting=sorting.remove_empty_units()

from spikeinterface.curation import remove_redundant_units

# remove redundant units from BaseSorting object
clean_sorting = remove_redundant_units(
    sorting,
    duplicate_threshold=0.9,
    remove_strategy="max_spikes"
)

# remove redundant units from SortingAnalyzer object
# note this returns a cleaned sorting
clean_sorting = remove_redundant_units(
    sorting_analyzer,
    duplicate_threshold=0.9,
    remove_strategy="min_shift"
)


job_kwargs=si.get_best_job_kwargs()
sorting_analyzer = sf.analyze_results(sorting_out_folder=output_folder,sorting=sorting,recording=recording,export_to_phy=False,export_report=True,job_kwargs=job_kwargs)