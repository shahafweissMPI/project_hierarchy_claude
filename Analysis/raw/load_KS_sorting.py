# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 09:08:02 2025

@author: su-weisss
"""
import importlib.util
import os
if os.name != 'nt':#windows
            if importlib.util.find_spec("cuml") is not None:
                print("cuml Module is installed.trying to speed up sklearn with RAPIDS")
                from cuml.accel import install
                install()
                #%load_ext cuml.accel​
                import sklearn
                
            else:
                print("cuml Module is not installed.")
                

#recordingpath='//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/data/afm16924/240526/trial0/ephys/preprocessed/catgt_afm16924_240526_pup_retrieval_g0/afm16924_240526_pup_retrieval_g0_imec0'
#sortingpath='//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/data/afm16924/240526/trial0/ephys/preprocessed/catgt_afm16924_240526_pup_retrieval_g0/afm16924_240526_pup_retrieval_g0_imec0/kilosort4'
from spikeinterface_gui import run_mainwindow
from pathlib import Path
import spikeinterface.full as si
import matplotlib.pyplot as plt
from spikeinterface import download_dataset
from spikeinterface import create_sorting_analyzer, load_sorting_analyzer
import spikeinterface.extractors as se
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
#job_kwargs = dict(n_jobs=0.9, progress_bar=True, chunk_duration="1s")
#job_kwargs=si.get_best_job_kwargs();

job_kwargs = dict(pool_engine='thread',n_jobs=8, progress_bar=True,mp_context='spawn',max_threads_per_worker=1)
job_kwargs=score.fix_job_kwargs(job_kwargs)
si.set_global_job_kwargs(**job_kwargs)


#recordingpath=Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\np2\afm17365\supercat\concat\supercat_afm17365_241202_0_g0\afm17365_241202_0_g0_imec0").as_posix()
#recordingpath=Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\np2\afm17365\supercat\concat\supercat_afm17365_241202_0_g0\afm17365_241202_0_g0_imec0").as_posix()
#recordingpath=Path(r"F:\supercat_afm17307_241024_hunting_pups_escape_g0\afm17307_241024_hunting_pups_escape_g0_imec0")
recordingpath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0"
sortingpath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\kilosort42025"
sortingpath=r"F:\scratch\kilosort4_spyder"
#sortingpath=Path(r"F:\supercat_afm17307_241024_hunting_pups_escape_g0\afm17307_241024_hunting_pups_escape_g0_imec0\kilosort4")
#sortingpath='//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/data/afm16924/concat/supercat_afm16924_240522_g0/afm16924_240522_g0_imec0/kilosort4_spyder'
import os
if os.name == 'nt':#windows
    print(f"windows detected")
    recording_new=Path(recordingpath)
    sorting_new=Path(sortingpath)
    
elif os.name == 'posix': #linux
    recordingpath=Path(recordingpath).as_posix()
    sortingpath=Path(sortingpath).as_posix()
    
    str_to_rep='\\\\gpfs.corp.brain.mpg.de\\stem\\data'
    str_to_put='\mnt\stem'
    recording_new_str = str(recordingpath).replace(str_to_rep,str_to_put)
    sorting_new_str   = str(sortingpath).replace(str_to_rep,str_to_put)
    recording_new_str = str(recording_new_str).replace('\\',"/")
    sorting_new_str   = str(sorting_new_str).replace('\\',"/")
    
    
    
    # If needed, convert the new string paths back to Path objects
    recording_new = Path(recording_new_str)
    sorting_new   = Path(sorting_new_str)
    
    print("New recording path:", f"{recording_new}")
    print("New sorting path:", f"{sorting_new}")
    
recording = si.read_spikeglx(recording_new,stream_id='imec0.ap')
recording_filtered = si.bandpass_filter(recording,freq_min=300,freq_max=3000)
fs=recording_filtered.get_sampling_frequency()
tmin = 7271.95
tmax = 12574.965015#float('inf')7271.942530,12574.965015
#recording_filtered=recording_filtered.frame_slice(start_frame=int(tmin*fs), end_frame=int(tmax*fs))
recording_filtered=recording_filtered.time(tmin,tmax)


#sorting = si.read_kilosort(folder_path=sorting_new)
sorting = si.read_phy(folder_path=sorting_new)
#sorting = sorting.frame_slice(start_frame=int(tmin*fs), end_frame=int(tmax*fs))
sorting.register_recording(recording_filtered)
sorting.get_total_duration()
#sorting=sorting.to_multiprocessing(job_kwargs['n_jobs'])

#sorting = scur.remove_excess_spikes(sorting, recording_filtered)
#sorting=scur.remove_duplicated_spikes(sorting,method= "keep_first_iterative")
#sorting=sorting.remove_empty_units()    


# from spikeinterface.curation import remove_redundant_units

# # remove redundant units from BaseSorting object
# clean_sorting = remove_redundant_units(
#     sorting,
#     duplicate_threshold=0.9,
#     remove_strategy="max_spikes"
# )

# remove redundant units from SortingAnalyzer object
# note this returns a cleaned sorting

    
#temp_analyzer_path=Path(r'F:\scratch\sorting_analyzer_folder_23_0')
temp_analyzer_path= sortingpath / 'analyzer_folder'
temp_analyzer_path.mkdir(parents=True, exist_ok=True)

sorting_analyzer = si.create_sorting_analyzer(sorting, recording_filtered,
                                              format="binary_folder", folder= temp_analyzer_path,
                                              verbose=True,overwrite=True,
                                              return_scaled=True,
                                              sparse=True,
                                              method='radius',
                                              radius_um=40,ms_before= 0.5,ms_after= 1.0,                                             
                                              **job_kwargs)

sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,seed=2205)
sorting_analyzer.compute("noise_levels",**job_kwargs)       
sorting_analyzer.compute("waveforms", ms_before=0.5, ms_after=1.0, **job_kwargs)
sorting_analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
sorting_analyzer.compute(input="template_similarity", method='cosine_similarity',**job_kwargs); 
sorting_analyzer.compute(input="template_metrics", include_multi_channel_metrics=False,**job_kwargs) ; 

sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
sorting_analyzer.compute(
                input="correlograms",
                window_ms=200.0,
                bin_ms=2.0,
                method="numba",
                verbose=True,
                **job_kwargs)
sorting_analyzer.compute(
                input="isi_histograms",
                window_ms=500.0,
                bin_ms=1.0,
                method="numba",
                verbose=True,
                **job_kwargs) 
sorting_analyzer.compute("unit_locations", method="monopolar_triangulation",**job_kwargs)
sorting_analyzer.compute( input="spike_locations",ms_before=0.5,ms_after=1,**job_kwargs)
sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"],**job_kwargs)
#sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_local', **job_kwargs)

# clean_sorting = remove_redundant_units(
#     sorting_analyzer,
#     duplicate_threshold=0.9,
#     remove_strategy="min_shift")

# qm_ext = sorting_analyzer.compute(input={"principal_components": dict(n_components=3, mode="by_channel_local"),
#                                 "quality_metrics": dict(skip_pc_metrics=False)})
# metrics = qm_ext.get_data()
# assert 'isolation_distance' in metrics.columns
 
# FR = sqm.compute_firing_rates(sorting_analyzer)
# rp_contamination, rp_violations = sqm.compute_refrac_period_violations(sorting_analyzer)      
    
# unit_peak_shifts= score.get_template_extremum_channel_peak_shift(sorting_analyzer, peak_sign= 'neg')
# sorting_aligned=spost.align_sorting(sorting, unit_peak_shifts)


#sorting_analyzer.save_as( format="binary_folder", folder= f"{str(temp_analyzer_path)}saved")
ext_wf = sorting_analyzer.get_extension("waveforms")
for unit_id in sorting_analyzer.unit_ids:
    wfs = ext_wf.get_waveforms_one_unit(unit_id)
    print(unit_id, ":", wfs.shape)
ext_templates = sorting_analyzer.get_extension("templates")

av_templates = ext_templates.get_data(operator="average")
print(av_templates.shape)

median_templates = ext_templates.get_data(operator="median")
print(median_templates.shape)


# for unit_index, unit_id in enumerate(analyzer.unit_ids[:3]):
#     fig, ax = plt.subplots()
#     template = av_templates[unit_index]
#     ax.plot(template)
#     ax.set_title(f"{unit_id}")
# sorting_analyzer.save_as(folder=f"{sortingpath.as_posix()}/sorting_analyzer/analyzer.zarr", format="zarr")
    

# reload the SortingAnalyzer
folder= f"{str(temp_analyzer_path)}"
sorting_analyzer = si.load_sorting_analyzer(folder)
# open and run the Qt app
run_mainwindow(sorting_analyzer, mode="web",verbose=True)
# open and run the Web app
#run_mainwindow(sorting_analyzer, mode="web")



# sw.plot_sorting_summary(sorting_analyzer, backend="spikeinterface_gui")
  
# from spikeinterface_gui import run_mainwindow
# run_mainwindow(sorting_analyzer, curation=True,mode="web")

# from spikeinterface.widgets import plot_sorting_summary
# sw.plot_sorting_summary(sorting_analyzer, curation=True, backend="spikeinterface_gui")