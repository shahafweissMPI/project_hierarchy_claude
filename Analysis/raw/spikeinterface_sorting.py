# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:09:08 2025

@author: su-weisss
"""
# Clear memory
%reset -f
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

import  spikeinterface_helper_functions as sf

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




def run_sorting(recording,sorter_name,sorter_params,base_folder):
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
    sorting = si.run_sorter(sorter_name, recording, folder=sorting_out_folder,
                            docker_image=False, verbose=True, **sorter_params,
                            remove_existing_folder=True,delete_output_folder=True)
    sorting.save_to_folder(name=sorter_name,folder=sorting_out_folder,overwrite=True)
    
    
    return sorting, sorting_out_folder

def load_analyzer(folder):
    sorting_analyzer= si.load_sorting_analyzer(folder)
    return sorting_analyzer

def s(folder,sorter_name):
    sorting_analyzer= si.sorting_extractor_full_dict
    return sorting_analyzer


def analyze_results(sorting_out_folder,sorting,rec,export_to_phy=False,export_report=True,job_kwargs=si.get_best_job_kwargs()):
    """
    sorting analyzer
    input:
        sorting_out_folder,sorting,rec
    returns:
            sorting_analyzer,temp_analyzer_path
            
    """
    temp_analyzer_path= sorting_out_folder / 'analyzer'
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting_KS3, recording=recording_filtered,
                                                  format="binary_folder", folder= temp_analyzer_path,
                                                  verbose=True,overwrite=True,
                                                  return_scaled=True,
                                                  sparse=True,
                                                  method='radius',radius_um=40,
                                                  ms_before= 0.5,ms_after= 1.0,**job_kwargs)
    
    
    
    
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
                    window_ms=50.0,
                    bin_ms=1.0,
                    method="numba",
                    verbose=True,
                    **job_kwargs) 
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation",**job_kwargs)
    sorting_analyzer.compute( input="spike_locations",ms_before=0.5,ms_after=1,**job_kwargs)
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"],**job_kwargs)
    sorting_analyzer.compute("principal_components", n_components=3, **job_kwargs)
    #export to phy
    if export_to_phy==True:
        si.export_to_phy(sorting_analyzer=sorting_analyzer, output_folder=temp_analyzer_path / 'phy_folder',**job_kwargs,remove_if_exists=True,compute_pc_features=False,compute_amplitudes=False,)
        #export a report
    if export_report==True:
        si.export_report(sorting_analyzer=sorting_analyzer, output_folder=temp_analyzer_path / 'report_folder',**job_kwargs,)
        
    return sorting_analyzer,temp_analyzer_path

def fetch_probe_from_library(manufacturer='neuropixels',probe_name='NP1010')
    probe = get_probe(manufacturer, probe_name)
    print(probe)
    return probe

def load_sorting_from_folder(sorter_name, sorting_folder):
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
        sorting_result = ExtractorClass(folder_path=sorting_folder)
        
        print("Loading successful!")
        return sorting_result

    except Exception as e:
        print(f"An error occurred while loading the sorting data for '{sorter_name}':")
        print(e)
        return None




def preprocess_NP_rec(recordingpath,kwargs=None):
    """
    recordingpath a Path object to the recording folder
    """
    if kwargs:
        pass
    else:
        kwargs={
    {'bandpass':True,
     'common_reference':False,
     'detect_bad_channels':False,
     'remove_channels':False,
     'frame_slice':False}
    
    recording = si.read_spikeglx(recordingpath,stream_id='imec0.ap')
    
    fs=recording.get_sampling_frequency()
    
    #recording.set_probe((probe))
    print(recording)
    recording.get_probe().to_dataframe()
    print(f'preprocessing from \n {recordingpath}')
    if kwargs['frame_slice']==True
        start_frame=kwargs['start_frame']
        end_frame=kwargs['end_frame']
    else:
        start_frame=0
        end_frame=recording.get_end_time()
    recording_filtered=recording.frame_slice(start_frame=start_frame, end_frame=end_frame)
    if kwargs{'bandpass']==True:
              recording_filtered = si.bandpass_filter(recording_filtered,freq_min=400,freq_max=3000)
    if kwargs['detect_bad_channels']:
        bad_channel_ids, channel_labels = si.detect_bad_channels(recording_filtered)            
        print('bad_channel_ids', bad_channel_ids)
    if kwargs['remove_channels']:
        recording_filtered = recording_filtered.remove_channels(bad_channel_ids)
    if kwargs['common_reference']:
        recording_filtered = si.common_reference(recording_filtered, operator="median", reference="local",local_radius=(40,160))
    
    
    return recording_filtered



recordingpath=Path(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0")
output_folder=Path(r"F:\scratch")
base_folder= output_folder

from spikeinterface.sorters import installed_sorters
installed_sorters()

def plot_spikes(rec,base_folder):
    %matplotlib widget
    si.plot_traces(rec, backend='matplotlib')
    
    # plot some channels
    fig, ax = plt.subplots(figsize=(20, 10))
    some_chans = rec.channel_ids[[100, 150, 200, ]]
    si.plot_traces(rec, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans)
    
    # we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
    noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
    noise_levels_int16 = si.get_noise_levels(rec, return_scaled=False)
    
    fig, ax = plt.subplots()
    _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
    ax.set_xlabel('noise  [microV]')
    fig.savefig(base_folder / 'noise_lvls.png')
    
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    
    job_kwargs = dict(n_jobs=0.9, chunk_duration='1s', progress_bar=True)
    
    
    peaks = detect_peaks(rec,  method='locally_exclusive', noise_levels=noise_levels_int16,
                          detect_threshold=5, radius_um=50., **job_kwargs)
    peaks
    
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    
    peak_locations = localize_peaks(rec, peaks, method='center_of_mass', radius_um=50., **job_kwargs)
    
    # check for drifts
    fs = rec.sampling_frequency
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(peaks['sample_ind'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)
    
    # we can also use the peak location estimates to have an insight of cluster separation before sorting
    fig, ax = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(rec, ax=ax, with_channel_ids=True)
    ax.set_ylim(-100, 150)
    
    ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
    fig.savefig(base_folder / 'peaks.png')
##############################################################################
## sorting
#fixed paths
si.Kilosort3Sorter.set_kilosort3_path(r'D:\sorters\Kilosort-3.0.2\Kilosort-3.0.2')
si.IronClustSorter.set_ironclust_path(r'D:\sorters\ironclustgit')


# check default params for kilosort2.5
kilosort2_5_params=si.get_default_sorter_params('kilosort2_5')
kilosort3_params=si.get_default_sorter_params('kilosort3')

kilosort3_params['minFR']=0.1
kilosort3_params['car']=False
kilosort3_params['whiteningRange']=48
kilosort3_params['NT']=65_472
kilosort3_params['save_rez_to_mat']=True


IRC_params = si.get_default_sorter_params('ironclust')
IRC_params['n_jobs']=-1 
IRC_params['fParfor']=True
IRC_params['clip_post']=0.5
IRC_params['detect_threshold']=4.5
IRC_params['prm_template_name']=r'\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\afm16924\concat\supercat_afm16924_240522_g0\afm16924_240522_g0_imec0\KS2_whiten_afm16924_supercat.imec0.ap_imec3b2.prm'
IRC_params['pc_per_chan']=9
IRC_params['common_ref_type']='none'
IRC_params['nSites_whiten']=48



spykingcircus2_params=si.get_default_sorter_params('spykingcircus2')
#kilosort4_params=si.get_default_sorter_params('kilosort4')


# run kilosort2.5 without drift correction
params_kilosort2_5 = {'do_correction': False}


IPython.embed()

#KS3
sorter_name='kilosort3'
sorter_params=kilosort3_params
sorting_out_folder=base_folder / f'{sorter_name}_output'
print(f'sorting with: {sorter_name}, to {sorting_out_folder}')
sorting_KS3 = si.run_sorter(sorter_name, recording_filtered, folder=sorting_out_folder,
                        docker_image=False, verbose=True, **sorter_params,
                        remove_existing_folder=True,delete_output_folder=True)
sorting_KS3.save(folder, save_kwargs)
sorting_KS3.save_to_folder(name='kilosort3',folder=sorting_out_folder /"results",overwrite=True)



def analyze_results(sorting_out_folder,sorting,rec):
    temp_analyzer_path= sorting_out_folder / 'analyzer'
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting_KS3, recording=recording_filtered,
                                                  format="binary_folder", folder= temp_analyzer_path,
                                                  verbose=True,overwrite=True,
                                                  return_scaled=True,
                                                  sparse=True,
                                                  method='radius',radius_um=40,
                                                  ms_before= 0.5,ms_after= 1.0,**job_kwargs)
    
    
    
    
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
                    window_ms=50.0,
                    bin_ms=1.0,
                    method="numba",
                    verbose=True,
                    **job_kwargs) 
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation",**job_kwargs)
    sorting_analyzer.compute( input="spike_locations",ms_before=0.5,ms_after=1,**job_kwargs)
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"],**job_kwargs)
    sorting_analyzer.compute("principal_components", n_components=3, **job_kwargs)
    return sorting_analyzer,temp_analyzer_path

# # the results can be read back for futur session
# sorting = si.read_sorter_folder(sorting_out_folder)
# sorting=sorting_spykingcircus2
# analyzer = si.create_sorting_analyzer(sorting, rec4, sparse=True, format="memory")
# analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
# analyzer.compute("waveforms",  ms_before=1.5,ms_after=2., **job_kwargs)
# analyzer.compute("templates", operators=["average", "median", "std"])
# analyzer.compute("noise_levels")
# analyzer.compute("correlograms")
# analyzer.compute("unit_locations")
# analyzer.compute("spike_amplitudes", **job_kwargs)
# analyzer.compute("template_similarity")
# analyzer


analyzer_saved = analyzer.save_as(folder=base_folder / "analyzer", format="binary_folder")
analyzer_saved


metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']


# metrics = analyzer.compute("quality_metrics").get_data()
# equivalent to
metrics = si.compute_quality_metrics(analyzer, metric_names=metric_names)

metrics

amplitude_cutoff_thresh = 0.1
isi_violations_ratio_thresh = 1
presence_ratio_thresh = 0.9

our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
print(our_query)

keep_units = metrics.query(our_query)
keep_unit_ids = keep_units.index.values
keep_unit_ids









analyzer_clean = analyzer.select_units(keep_unit_ids, folder=base_folder / 'analyzer_clean', format='binary_folder')
analyzer_clean

# export spike sorting report to a folder
si.export_report(analyzer_clean, base_folder / 'report', format='png')
analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')
analyzer_clean
si.plot_sorting_summary(analyzer_clean, backend='sortingview')

