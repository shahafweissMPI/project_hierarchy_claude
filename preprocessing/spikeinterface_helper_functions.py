# -*- coding: utf-8 -*-
"""
Spike Sorting Analysis Toolkit

This script provides a collection of functions to run, load, preprocess, analyze,
and compare spike sorting results using the SpikeInterface library.

--- Table of Contents ---

- preprocess_NP_rec:
    Loads and preprocesses Neuropixels data from a SpikeGLX folder.
    Applies filtering, bad channel removal, and common reference correction.

- fetch_probe_from_library:
    Retrieves a probe's layout and metadata from the probeinterface library.

- run_sorting:
    Runs a single specified spike sorter on a recording object.

- analyze_results:
    Creates a SortingAnalyzer to compute waveforms, templates, quality metrics,
    and other post-processing analyses for a given sorting result.

- load_sorting_from_folder:
    Dynamically loads a sorting result from a folder by matching the sorter
    name to the correct SpikeInterface extractor.

- load_analyzer:
    Loads a previously saved SortingAnalyzer object from a folder.

- dynamic_run_and_compare_sorters:
    Runs multiple spike sorters on a recording and then compares their outputs
    to find consensus units.

- dynamic_compare_sortings:
    Compares the results of multiple, pre-loaded sorting objects.
    
    Created on Thu Jul  3 14:08:10 2025
    @author: su-weisss
"""

import matplotlib
#%matplotlib inline
import spikeinterface
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as scomp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw
from spikeinterface.exporters import export_report
import spikeinterface.core as score 
import spikeinterface.postprocessing as spost
from probeinterface import Probe, get_probe
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from pathlib import Path
import IPython
from typing import Union
import pandas as pd
job_kwargs=si.get_best_job_kwargs();



def dynamic_compare_sortings(sorting_objects, name_list, show_graph=True):
    """
    Dynamically compares multiple spike sorting outputs.

    Parameters:
      sorting_objects: A list of pre-computed spike sorting objects.
      name_list: A list of names corresponding to each sorting object.
      show_graph: If True, the multi-comparison graph will be plotted.
      
    Returns:
      mcmp: The MultiSortingComparison object.
      
    Example usage:
        Assume sorting_MS4, sorting_HS, sorting_TDC are precomputed sorting objects:
        sorting_list = [sorting_MS4, sorting_HS, sorting_TDC]
        names = ['MS4', 'HS', 'TDC']
        mcmp = dynamic_compare_sortings(sorting_objects=sorting_list, name_list=names)
    """
    
    # Import necessary spikeinterface modules
    import spikeinterface.comparison as sc
    import spikeinterface.widgets as sw
    from itertools import combinations

    # Compare the multiple sorters
    mcmp = sc.compare_multiple_sorters(
        sorting_list=sorting_objects,
        name_list=name_list,
        verbose=True,
    )

    # Iterate pairwise through the provided sorters and print the comparisons
    for s1, s2 in combinations(name_list, 2):
        cmp_obj = mcmp.comparisons.get((s1, s2)) or mcmp.comparisons.get((s2, s1))
        if cmp_obj:
            print(f"Comparison between {s1} and {s2}:")
            print("Sorting 1:", cmp_obj.sorting1)
            print("Sorting 2:", cmp_obj.sorting2)
            print("Matching:", cmp_obj.get_matching())
            print()

    # Plot the overall multi-comparison graph if desired
    if show_graph:
        sw.plot_multicomp_graph(multi_comparison=mcmp)

    # Optionally, get and print consensus units from all comparisons
    agr_all = mcmp.get_agreement_sorting(minimum_agreement_count=len(name_list))
    print(f'Units in agreement for all {len(name_list)} sorters:', agr_all.get_unit_ids())

    return mcmp


def dynamic_run_and_compare_sorters(recording, sorter_names, show_graph=True):
    """
    Dynamically compares multiple spike sorting outputs.

    Parameters:
      recording: The recording object to be sorted.
      sorter_names: A list of sorter names as strings (e.g., ['mountainsort4', 'herdingspikes', 'tridesclous']).
      show_graph: If True, the multi-comparison graph will be plotted.

    Returns:
      mcmp: The MultiSortingComparison object.
      # Example usage:
      # recording = ...  # Initialize your recording object
      # sorter_names = ['mountainsort4', 'herdingspikes', 'tridesclous']
      # mcmp = dynamic_compare_sorters(recording, sorter_names)
    """

    # Import spikeinterface modules if needed
    import spikeinterface.sorters as ss
    import spikeinterface.comparison as sc
    import spikeinterface.widgets as sw

    # Run each spike sorter and store the results
    sorting_list = []
    for sorter in sorter_names:
        print(f"Running sorter: {sorter}")
        sorting = ss.run_sorter(sorter_name=sorter, recording=recording)
        sorting_list.append(sorting)

    # Compare the multiple sorters
    mcmp = sc.compare_multiple_sorters(
        sorting_list=sorting_list,
        name_list=sorter_names,
        verbose=True,
    )

    # Optionally display pairwise comparison information for each pair
    from itertools import combinations
    for s1, s2 in combinations(sorter_names, 2):
        cmp_dict = mcmp.comparisons.get((s1, s2)) or mcmp.comparisons.get((s2, s1))
        if cmp_dict:
            print(f"Comparison between {s1} and {s2}:")
            print("Sorting 1:", cmp_dict.sorting1)
            print("Sorting 2:", cmp_dict.sorting2)
            print("Matching:", cmp_dict.get_matching())
            print()

    # Plot the overall multi-comparison graph if desired
    if show_graph:
        sw.plot_multicomp_graph(multi_comparison=mcmp)

    # Example: Get consensus units that are found by all sorters
    agr_all = mcmp.get_agreement_sorting(minimum_agreement_count=len(sorter_names))
    print(f'Units in agreement for all {len(sorter_names)} sorters:', agr_all.get_unit_ids())

    return mcmp

def get_kilosort4_params(spikeglx_folder,tmin,tmax,acg_threshold,ccg_threshold):
    settings={}
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
    return settings


def run_sorting(recording,sorter_name,sorter_params=None,base_folder=None,docker_image=False):
    """
    sorting funciton
    input: 
        recording,sorter_name,sorter_params,base_folder
    returns:
        sorting, sorting_out_folder
    """
    ## **spykingcircus2** 
    #sorter_name='spykingcircus2'
    
    #sorter_params=spykingcircus2_params
    
    sorting_out_folder=base_folder / f'{sorter_name}_output'
    print(f'sorting with: {sorter_name}, to {sorting_out_folder}')
    if sorter_params is not None:
        sorting = si.run_sorter(sorter_name, recording, folder=sorting_out_folder,
                            docker_image=docker_image, verbose=True, **sorter_params,
                            remove_existing_folder=True,delete_output_folder=True)
    else:
        sorting = si.run_sorter(sorter_name, recording, folder=sorting_out_folder,
                            docker_image=docker_image, verbose=True, 
                            remove_existing_folder=True,delete_output_folder=True)
    sorting.save_to_folder(name=sorter_name,folder=sorting_out_folder,overwrite=True)
    
    
    return sorting, sorting_out_folder

def load_analyzer(folder):
    sorting_analyzer= si.load_sorting_analyzer(folder)
    return sorting_analyzer

def load_sorter(folder,sorter_name):
    sorting_analyzer= si.sorting_extractor_full_dict
    return sorting_analyzer

def auto_label(sorting_analyzer):
    # Apply the noise/not-noise model
    noise_neuron_labels = scur.auto_label_units(
        sorting_analyzer=sorting_analyzer,
        repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
        trust_model=True,
    )
    
    noise_units = noise_neuron_labels[noise_neuron_labels['prediction']=='noise']
    analyzer_neural = sorting_analyzer.remove_units(noise_units.index)
    
    # Apply the sua/mua model
    sua_mua_labels = scur.auto_label_units(
        sorting_analyzer=analyzer_neural,
        repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
        trust_model=True,
    )
    
    all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
    print(all_labels)
    return all_labels

def analyze_results(sorting_out_folder,sorting,recording,export_to_phy=False,export_report=False,save_rasters=False,job_kwargs=si.get_best_job_kwargs()):
    """
    sorting analyzer
    input:
        sorting_out_folder,sorting,rec
    returns:
            sorting_analyzer,temp_analyzer_path
            
    """
    job_kwargs['n_jobs']=int(job_kwargs['n_jobs']*0.8)
    temp_analyzer_path= sorting_out_folder / 'sorting_analyzer_results'
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording,
                                                  format="binary_folder", folder= temp_analyzer_path,
                                                  verbose=True,overwrite=True,
                                                  return_scaled=True,
                                                  sparse=True,
                                                  method='radius',radius_um=40,
                                                  ms_before= 1.5,ms_after= 1.5,**job_kwargs)
    print('new analyzer created at temp_analyzer_path')
    
    
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,seed=2205)
    sorting_analyzer.compute("noise_levels",**job_kwargs)       
    sorting_analyzer.compute("waveforms", ms_before=1.5, ms_after=1.5, **job_kwargs)
    sorting_analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
    #print('calculating time shifts')
    #unit_peak_shifts= score.get_template_extremum_channel_peak_shift(sorting_analyzer, peak_sign= 'neg')
    #sorting=spost.align_sorting(sorting, unit_peak_shifts)           
    #print('recalculating waveforms and templates after shift corrections')
    #sorting_analyzer.compute("waveforms",  ms_before=1.5, ms_after=1.5, **job_kwargs)
    #sorting_analyzer.compute("templates", operators=["average", "median", "std"],**job_kwargs)
    print('computing template simliarity and metrics')
    sorting_analyzer.compute(input="template_similarity", method='cosine_similarity',**job_kwargs); 
    sorting_analyzer.compute(input="template_metrics", include_multi_channel_metrics=False,**job_kwargs) 
    print('calculating amplitudes') 
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
    print('calculating ISI and CCG histograms') 
    
    sorting_analyzer.compute(
                    input="correlograms",
                    window_ms=100.0,
                    bin_ms=2.0,
                    method="numba",
                    verbose=True,
                    **job_kwargs)    
    sorting_analyzer.compute(
                    input="isi_histograms",
                    window_ms=100.0,
                    bin_ms=1.0,
                    method="numba",
                    verbose=True,
                    **job_kwargs) 
    
    print('calculating locations') 
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation",**job_kwargs)
    sorting_analyzer.compute( input="spike_locations",ms_before=1.0,ms_after=1.5,**job_kwargs)
    
    print('calculating PCA') 
    sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_global', whiten=True, **job_kwargs)
    print('calculating quality metrics') 
    sorting_analyzer.compute("quality_metrics", metric_names=["num_spikes","snr", "firing_rate"],**job_kwargs)
#    'isi_violation',    'rp_violation',    'sliding_rp_violation','isolation_distance',
#     'l_ratio',     'd_prime'],**job_kwargs)
    print(f'extentiond computed. stored at:\n {temp_analyzer_path}')
    print(sorting_analyzer)
    
    
    #export to phy
    if export_to_phy==True:      
        si.export_to_phy(sorting_analyzer=sorting_analyzer, output_folder=temp_analyzer_path / 'phy_folder',
                         remove_if_exists=True,compute_pc_features=False,compute_amplitudes=True,dtype='int16',**job_kwargs)
        #export a report
    if export_report==True:
        si.export_report(sorting_analyzer=sorting_analyzer, output_folder=temp_analyzer_path / 'report_folder',**job_kwargs)
    if save_rasters==True:
        import spikeinterface.widgets as sw
        sw.plot_rasters(sorting_analyzer,figsize=(80, 40))
        plt.savefig(temp_analyzer_path / 'spiker rasters.png')
        plt.close('all')
        
    return sorting_analyzer,temp_analyzer_path

def fetch_probe_from_library(manufacturer='neuropixels',probe_name='NP1010'):
    probe = get_probe(manufacturer, probe_name)
    print(probe)
    return probe

def load_sorting_from_folder(sorter_name: str, sorting_folder: str | Path):
    """
    Loads spike sorting output from a folder using the appropriate SpikeInterface extractor.

    This function dynamically selects the correct SortingExtractor based on the
    provided sorter name.

    Args:
        sorter_name (str): The name of the spike sorter used (e.g., 'kilosort', 'tridesclous').
                           This must be a key in the SpikeInterface sorter dictionary.
        sorting_folder (str | Path): The path to the folder containing the spike sorter's output files.

    Returns:
        spikeinterface.BaseSorting | None: A SpikeInterface Sorting object if the sorter name is valid
                                            and the data is loaded successfully, otherwise None.
    """
    # Get the dictionary that maps sorter names to their corresponding extractor classes
    # si.get_sorting_extractor_dict() is the canonical way to get this mapping.
    sorting_extractor_dict = si.sorting_extractor_full_dict

    # Ensure the provided sorter_name is valid
    if sorter_name not in sorting_extractor_dict:
        print(f"Error: Sorter '{sorter_name}' not found.")
        print(f"Available sorters are: {list(sorting_extractor_dict.keys())}")
        return None

    try:
        # Get the specific SortingExtractor class from the dictionary
        ExtractorClass = sorting_extractor_dict[sorter_name]
        
        # Instantiate the extractor with the provided folder path
        # Most, if not all, folder-based extractors accept a 'folder_path' or positional argument.
        print(f"Loading from '{sorting_folder}' using '{ExtractorClass.__name__}'...")
        sorting_result = ExtractorClass(sorting_folder)
        print(sorting_result)
        print("Loading successful!")
        return sorting_result

    except Exception as e:
        print(f"An error occurred while loading the sorting data for '{sorter_name}':")
        print(e)
        return None




def preprocess_NP_rec(recordingpath,kwargs=None,stream_id=None):
    """
    recordingpath a Path object to the recording folder
    kwargs: a dictionary with the following example fields:
   'bandpass':True,
   'common_reference':True,
   'detect_bad_channels':False,
   'remove_channels':False,
   'time_slice':True,
   'start_time':7271.942530,
   'end_time':12574.965015
    """
    if kwargs:
        pass
    else:
        kwargs={
    'bandpass':True,
     'common_reference':False,
     'detect_bad_channels':False,
     'remove_channels':False,
     'time_slice':False}
        
        
    if stream_id is None:
        stream_id='imec0.ap'
    
    recording = si.read_spikeglx(str(recordingpath.as_posix()),stream_id=stream_id)
    t_start=recording.get_time_info()['t_start']
    recording.shift_times(shift=-t_start)
    
    fs=recording.get_sampling_frequency()
    
    #recording.set_probe((probe))
    print(recording)
    recording.get_probe().to_dataframe()
    print(f'preprocessing from \n {recordingpath}')
    if kwargs['time_slice']==True:
        start_frame=kwargs['start_time']
        end_frame=kwargs['end_time']
        #recording_filtered=recording.frame_slice(start_frame=int(start_frame*fs), end_frame=int(end_frame*fs))
        recording_filtered=recording.time_slice(start_time=start_time, end_time=end_time)
    else:
        #start_frame=recording.get_start_time()
#        end_frame=start_frame+recording.get_duration()
        recording_filtered=recording
    
    if kwargs['bandpass']:
        recording_filtered = si.bandpass_filter(recording_filtered,freq_min=300,freq_max=3000)
    if kwargs['detect_bad_channels']:
        bad_channel_ids, channel_labels = si.detect_bad_channels(recording_filtered)            
        print('bad_channel_ids', bad_channel_ids)
    if kwargs['remove_channels']:
        recording_filtered = recording_filtered.remove_channels(bad_channel_ids)
    if kwargs['common_reference']:
        recording_filtered = si.common_reference(recording_filtered, operator="median", reference="local",local_radius=(40,160))
    
    print(recording_filtered)
    return recording_filtered