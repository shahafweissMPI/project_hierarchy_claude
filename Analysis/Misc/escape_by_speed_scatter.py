"""
Created by Tom Kern
Last modified 04.08.2024

Is there a correlation between escape velocity and firing intensity?
     --> From the plots I would say no, but I compute this not optimally
         also, we're not in dPAG
- Determine escape-reactive neurons
     -for each neuron, divide firing during escape periods by avg firing 
         (i.e. firing change)
     -This is done separately per escape period
     -neurons that have a firing change of above 2 for at least 4 escapes
         are considered 'escape-active'
- take the mean firing rate during each escape from the escape active neurons
- plot this against peak - escape velocity

Limitations
-This way of computing 'escape-active' ignores variance in firing
- Scaling of firing with peak-velocity should better be calculated on the single-neuron 
    level first, and then say the proportion of neurons that do scaling. 

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import os
from scipy.stats import sem

session='231213_0'
exclude_switches=False


savepath=r"F:\scratch\escape_firing_threshold_test"
if not os.path.exists(fr'{savepath}\{session}'):
    os.makedirs(fr'{savepath}\{session}')

plt.style.use('dark_background')

[_, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index, 
 velocity, 
 _, 
 _, 
 frame_index_s] = hf.load_preprocessed(session, load_lfp=False)


#%%

n_srate=1/n_time_index[1]

escapes_f, escape_peaks=hf.peak_escape_vel(behaviour, velocity, exclude_switches=False)
escapes_s=np.hstack((frame_index_s[escapes_f[:,0],None], frame_index_s[escapes_f[:,1],None]))
turns_s=behaviour[behaviour['behaviours']=='turn']['frames_s']

#exclude siwtches
if exclude_switches:
    escapes_s, mask=hf.exclude_switch_trials(escapes_s, behaviour, return_mask=True)
    turns_s=turns_s[mask]
    escape_peaks=escape_peaks[mask]

#saniy check
if len(turns_s)!=len(escapes_s):
    raise ValueError('unequal number of turns vs escapes!')



    
    
# get firing of each neuron in Hz 
n_hz=np.sum(ndata, axis=1)/n_time_index[-1]

all_e_change=[] #relative change of firing during escape
for e, turn in zip(escapes_s, turns_s):


   e_ind=(n_time_index>turn) & (n_time_index<e[1])
   e_spikes=ndata[:,e_ind]
   e_hz=np.sum(e_spikes, axis=1)/(e[1]-turn)
   e_change=e_hz/n_hz
   all_e_change.append(e_change)
   
all_e_change=np.array(all_e_change).T


# get only neurons that are modulated by escape 
modulated_ind=np.sum(all_e_change>2, axis=1)>3
modulated_e_change=all_e_change[modulated_ind]
tot_change=np.mean(modulated_e_change, axis=0)
sem_change=sem(modulated_e_change, axis=0)

plt.figure()
plt.scatter(escape_peaks, tot_change)
plt.errorbar(escape_peaks, tot_change, yerr=sem_change, linestyle='None')

plt.ylabel('avg firing change')
plt.xlabel('peak velocity')
plt.title(f'{session}\neach entry is one escape')
pf.remove_axes()
plt.xlim((28,135))
plt.ylim((.9,6))
