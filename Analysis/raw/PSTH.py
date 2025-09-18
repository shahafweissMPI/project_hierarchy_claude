"""
Created by Tom Kern
Last modified 05/08/24

Makes a PSTH plot per neuron, with one subplot for each behaviour in target_bs 
-If the same behaviour happens twice in close sequence (e.g. two attacks),
    the second behaviour will not be plotted in a new line, but will just be
    shaded in a less intense color
    
""" 

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import os
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
from time import time

import math
import matplotlib; matplotlib.use('Agg')  # Use a non-interactive backend
from pathlib import Path
#Parameters
animal = 'afm16924'
sessions = '240522'
plt.ion()
plt.style.use('default')
window=5 #s How much time before and after event start should be plotted
density_bins=.5 #s; Over what time should avg activity in axs[1] be averaged

target_regions=['LPAG','DMPAG','DLPAG','Su3','Su3C'] # From which regions should the neurons be?, leave empty for all
# target_regions=[]
#target_bs=['nesting','pup_snif','pup_grab','pup_retrieve','pup_run','pup_sit','loom']
target_bs=[] #which behacviors to plot? leave empty for all

#savepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\PSTH"
#savepath=Path(rf"C:\Users\chalasp\Documents\GitHub\test")
savepath_0=Path(rf"E:\2025\Figures\PSTH\{animal}")


for session in [sessions]:

    
    if not os.path.exists(fr'{savepath_0}\{session}'):
        os.makedirs(fr'{savepath_0}\{session}',exist_ok=True)
    savepath  =  Path(rf"{savepath_0}\{session}")
    
# load data
    [dropped, 
     behaviour, 
     ndata, 
     n_time_index, 
     n_cluster_index, 
     n_region_index, 
     n_channel_index, 
     velocity, 
     _, 
     _, 
     frame_index_s] = hf.load_preprocessed(animal, session)
    
    if session=='240522':
        velocity=velocity[0:len(frame_index_s)]
    
    frame_index_s=frame_index_s[:len(velocity)]

    #Get base firing
    base_mean, _= hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s)
    base_mean=np.round(base_mean, 2)
    
    unique_behaviours = behaviour.behaviours.unique()
    if not target_bs:
        target_bs = np.sort(unique_behaviours)
    
    
   
    if target_regions:
        target_n=np.where(np.isin(n_region_index, target_regions))[0]
    else:
        target_n = np.arange(len(n_region_index))


   
    # Get neurons from target regions
    
   
    
    
    
    
    
#%%plot PSTH

      
    def plotting(i_neuron, region, channel, neurondata):
        # Set figsize depending on number of subplots
        figlength=int(8*len(target_bs)/2)
        figheight=int(6*len(target_bs)/2)
        
            
        #Make big plot for all behaviours
        rows, cols, gs, fig=pf.subplots(len(target_bs), gridspec=True, figsize=(figlength, figheight))
        fig.suptitle(f'{region}, neuron: {i_neuron}\nBaseline: {base_mean[i_neuron]}Hz site: {channel}',fontsize=72, fontname='Arial')
        gs.update(hspace=0.5) #vertical distance between plots
    
        ys=[]
        Hz_axs=[]
        
        #Go through each behaviour
        for i, b_name in enumerate(target_bs):
            
            #divide one subplot into three (velocity, avg Hz, firing)
            gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
            axs=[]
            for j in range(3):
                axs.append(plt.subplot(gs_sub[j]))
                                    
            #get behaviour frames
            start_stop=hf.start_stop_array(behaviour, b_name)  
            if len(start_stop) == 0:
                print(f'skiiped {i_neuron}')
                continue                         
    
            #Make PSTH           
            pf.psth(neurondata, 
                    n_time_index, 
                    start_stop, 
                    velocity,
                    frame_index_s,
                    axs, 
                    window=window, 
                    density_bins=density_bins)
            
            
            # make x/y labels
            axs[0].set_title(b_name, fontsize=50, fontname='Arial')
            if i in [0,3]:
                axs[0].set_ylabel('Velocity [cm/s]')            
                axs[1].set_ylabel('avg firing [Hz]')
                axs[2].set_ylabel('trials')
            axs[2].set_xlabel('time [s]')
            
        #Collect y_max in all plots, to have the same ylim in all subplots
            ys.append(axs[1].get_ylim()[1])
            Hz_axs.append(axs[1])
        max_y=np.max(ys)
        [ax.set_ylim((0,max_y)) for ax in Hz_axs]
        
    

        plt.savefig(Path(rf"{savepath}\{session}_cell{i_neuron}_ch{channel}_{region}.png").as_posix())
        # plt.savefig(rf"{savepath}\{i_neuron}_{channel}_{region}.svg")
        #plt.close()
        plt.close()


# Use joblib to parallelize the loop
    start=time()

    for i_neuron, (region, channel, neurondata) in enumerate(zip(n_region_index[target_n], n_channel_index, ndata)):
       # print(i_neuron,channel)
        plotting(i_neuron, region, channel, neurondata)

    # num_cores = os.cpu_count()
    # half_cores = math.ceil(num_cores / 2)
    # Parallel(half_cores)(delayed(plotting)(i_neuron, 
    #                                   region, 
    #                                   channel, 
    #                                   neurondata) for (
    #                                       i_neuron,
    #                                       (region,  
    #                                        channel,  
    #                                        neurondata)) in enumerate(zip(
    #                                                         n_region_index,
    #                                                         n_channel_index,
    #                                                         ndata)))
    stop=time()
    print(f'{session} took {hf.convert_s(stop-start)[0]}')
plt.ion()
  
