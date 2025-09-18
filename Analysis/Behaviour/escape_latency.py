"""
Created by Tom Kern
Last modified 04.08.2024

-avg time delay between start of loom and turn (i.e. escape onset) (fig1)
-comparison between regular escape and switch escape (if hunting happened before; fig2)
-distribution of escape latencies (fig3)
-statistic tests between switch-escape and regular-escape latencies
-max_reaction: maximmum time delay between loom and escape onset
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
import seaborn as sns




animals=['afm16924']
plt.style.use('default')

max_reaction=7

#%% Collect all behaviour in one big v

bigB, all_frame_index, _, animal_border=hf.bigB_multiple_animals(animals)


b_names=bigB['behaviours'].to_numpy()
start_stop=bigB['start_stop'].to_numpy()
frames_s=bigB['frames_s'].to_numpy()

#%% Loom reactions




def get_rt(looms):
    rt=[]
    rt_border=0
    for loom in looms:
        event_ind=(frames_s> loom) & (frames_s<loom+max_reaction)
        reactions_b=b_names[event_ind]
        reactions_s=frames_s[event_ind]
        
        if 'loom' in reactions_b:
            loom_index = np.where(reactions_b == 'loom')[0][0]
            reactions_b = reactions_b[:loom_index]
            reactions_s = reactions_s[:loom_index]
        
        if 'escape' in reactions_b:
            if 'turn' in reactions_b:
                rt.append(reactions_s[reactions_b=='turn'][0] - loom)
            else:
                rt.append(reactions_s[reactions_b=='escape'][0] - loom)
            if loom<(animal_border[0]/50):
                rt_border+=1
    return rt, rt_border


########################## Indiscriminative ###################################
looms=frames_s[b_names=='loom']
rt, _=get_rt(looms)

plt.figure()
sns.violinplot(rt, color='slategray', inner='point')
plt.scatter(np.zeros_like(rt),rt, color='w')
pf.remove_axes()
plt.ylabel('escape latency')
plt.title(animals)


###################### during hunt vs regular #################################

otherlooms, huntlooms=hf.divide_looms(all_frame_index, bigB, radiance=2)

other_rt, other_border=get_rt(otherlooms)
hunt_rt, hunt_border=get_rt(huntlooms)

plt.figure()
plt.bar(range(2),[np.mean(other_rt), np.mean(hunt_rt)], color=['lightblue', 'violet'])

plt.scatter(np.zeros_like(other_rt[:other_border]),other_rt[:other_border],  color='slategray')
plt.scatter(np.zeros_like(other_rt[other_border:]),other_rt[other_border:],  color='navy')

plt.scatter(np.ones_like(hunt_rt[:hunt_border]),hunt_rt[:hunt_border],  color='thistle')
plt.scatter(np.ones_like(hunt_rt[hunt_border:]),hunt_rt[hunt_border:],  color='purple')


plt.title(animals)

plt.xticks(range(2), ['escape','switch'])
pf.remove_axes()
plt.ylabel('escape latency (s)')


#%% statistic tests
from scipy import stats
hf.test_normality(other_rt)
hf.test_normality(hunt_rt)

# --> not normal

#Mann-whitney U test
u_statistic, p_value = stats.mannwhitneyu(other_rt, hunt_rt)

print(f"U statistic: {u_statistic}")
print(f"P-value: {p_value *2}") # because two tailed testing

plt.figure()
plt.hist(other_rt, label='escape', facecolor='none', edgecolor='lightsteelblue', linewidth=2)
plt.hist(hunt_rt, label='switch', facecolor='none', edgecolor='salmon', linewidth=2) 
plt.legend()