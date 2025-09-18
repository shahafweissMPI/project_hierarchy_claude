"""
Created by Tom Kern
Last modified 04.08.2024

avg latency in min between introduction of cricket and first occurrence of 
hunting behaviours

Eating behaviour is a bit of problem here because the mice tend to go back to
and eat a dead cricket, even if the new one is already in the arena. 
This going back is usually a lot shorter than the eating right after having 
killed the cricket. I get rid of all eating periods that are shorter than
30s in this script (line 51)
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
import seaborn as sns
import scipy



animals=['afm16924']
plt.style.use('default')



#%% Collect all behaviour in one big v

# bigB, all_frame_index, _=hf.get_bigB(animal)
bigB, all_frame_index, all_vel, _= hf.bigB_multiple_animals(animals)

b_names=bigB['behaviours'].to_numpy()
start_stop=bigB['start_stop'].to_numpy()
frames_s=bigB['frames_s'].to_numpy()

#%% Latency

hunting_bs=['approach','pursuit','attack', 'eat']
cs=['tan','coral','firebrick','slategray']
hunting_latencies=[] #trials*hunting_bs
for intro in frames_s[b_names=='introduction']:
    
    latency=[]
    for b_name in hunting_bs:
        
        # Filter out eating oeriods that are too short
        if b_name=='eat':
            b_start_stop=hf.start_stop_array(bigB,b_name)
            eat_dur=b_start_stop[:,1]-b_start_stop[:,0]
            b_starts=b_start_stop[eat_dur>30, 0]
        else:
            b_starts=frames_s[(b_names==b_name) & (start_stop=='START')]
        first_b=b_starts[b_starts>intro][0]
        latency.append(first_b-intro)
    hunting_latencies.append(latency)
hunting_latencies=np.array(hunting_latencies) #trials*hunting_bs
hunting_latencies /= 60 # convert to minutes

mean_latencies=np.mean(hunting_latencies, axis=0)

plt.figure()
plt.bar(range(len(hunting_bs)),mean_latencies, color=cs)
for i,h in enumerate(hunting_latencies.T):
    sem=scipy.stats.sem(h)
    mean=np.mean(h)
    plt.plot([i,i], [mean-sem, mean+sem],c='k', lw=.5)
plt.xticks(range(len(hunting_bs)), hunting_bs)
plt.ylabel('latency (min)')
pf.remove_axes()
plt.title(animals)

