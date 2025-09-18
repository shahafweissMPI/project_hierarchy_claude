"""
Created by Tom Kern
Last modified 04.08.2024

Do neurons respond differently to escapes, dependeing on the peak velocity?
-Creates PSTHs for escapes. The y-axis is not just trials, but peak escape 
velocity
-marks looms and escape duration

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import os

session='231213_0'

savepath=r"F:\scratch\escape_firing_threshold_test"
prewindow=5 #s
postwindow=10 #s, relative to escape START






if not os.path.exists(fr'{savepath}\{session}'):
    os.makedirs(fr'{savepath}\{session}')
plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']


[frames_dropped, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index, 
 velocity, 
 locations, 
 node_names, 
 frame_index_s] = hf.load_preprocessed(session, load_lfp=False)


#%%
plt.ioff()

n_srate=1/n_time_index[1]

escapes_f, escape_peaks=hf.peak_escape_vel(behaviour, velocity, exclude_switches=False)
escapes_s=np.hstack((frame_index_s[escapes_f[:,0],None], frame_index_s[escapes_f[:,1],None]))
turns_s=behaviour[behaviour['behaviours']=='turn']['frames_s']
looms_s=behaviour[behaviour['behaviours']=='loom']['frames_s']

#saniy check
if len(turns_s)!=len(escapes_s):
    raise ValueError('unequal number of turns vs escapes!')


for neuron, cluster, region in zip(ndata, n_cluster_index, n_region_index):
    all_e_change=[] #relative change of firing during escape
    n_hz=np.sum(neuron)/n_time_index[-1]
    
    # format figure
    plt.figure(figsize=(10,10))
    plt.ylabel('velocity (cm/s')
    plt.xlabel('time (s), centred at turn')
    plt.title(f'{region}, {np.round(n_hz,1)} Hz')
    pf.remove_axes()
    plt.xlim((-prewindow, postwindow))
    
    for e, turn, vel in zip(escapes_s, turns_s, escape_peaks):
       centre=turn
       loom=looms_s[(looms_s>centre-2*prewindow) & (looms_s<centre+postwindow)] 
       
        # get spikes that happen during escape
       n_ind= (n_time_index>(centre-prewindow)) & (n_time_index<(centre+postwindow))
       
       # get the time of spikes, in s, relative to escape start
       n_turn=np.where(neuron[n_ind])[0]/n_srate-prewindow
       
       #mark escape duration
       plt.axhline(vel, c='gray', lw=.5)
       plt.barh(vel, e[1]-centre, left= 0, height=1, color='lightgray')
       
       # mark loom 
       plt.scatter(loom-centre,np.ones_like(loom)*vel,c='plum', s=80)
       
       # plot
       plt.scatter(n_turn,np.ones_like(n_turn)*vel, c='navy', s=.9)


    # skip figure if there is no significant increase
       e_ind=(n_time_index>centre) & (n_time_index<e[1])
       e_spikes=neuron[e_ind]
       e_hz=sum(e_spikes)/(e[1]-centre)
       all_e_change.append(e_hz/n_hz)
    
    all_e_change=np.array(all_e_change)
    num_sig_change=sum(all_e_change>2)
    if  num_sig_change>2:  
        plt.savefig(rf"{savepath}\{session}\{num_sig_change}_{region}_{cluster}.tiff")
        plt.close()
   
    else:
       
       plt.close()

plt.ion()
hf.endsound()



