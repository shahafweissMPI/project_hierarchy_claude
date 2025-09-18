"""
Created by Tom Kern
Last modified 04.08.2024

compares peak velocity during regular escape vs during switch escape. 
same-sheded dots are from the same animal (figure1)

also makes satistic tests, fig2 shows the populations that are being compared 
(i.e. distribution of peak escape velocities)
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
import seaborn as sns




# animal='afm16618'
plt.style.use('default')
animals=['afm16505','afm16618']


#%% Collect all behaviour in one big v

    
bigB, all_frame_index, all_vel, animal_border=hf.bigB_multiple_animals(animals)


b_names=bigB['behaviours'].to_numpy()
start_stop=bigB['start_stop'].to_numpy()
frames_s=bigB['frames_s'].to_numpy()


#%% get escapes

escapes=hf.start_stop_array(bigB, 'escape')
switches=hf.start_stop_array(bigB, 'switch')


#get peak velocities
#escape
peak_vels_e=[]
e_border=0
for e in escapes:
    vel=all_vel[(all_frame_index>e[0]) & (all_frame_index<e[1])]
    peak_vels_e.append(np.nanmax(vel))
    if e[1]<(animal_border[0]/50):
        e_border+=1
peak_vels_e=np.array(peak_vels_e)

#switch
peak_vels_s=[]
s_border=0
for s in switches:
    vel=all_vel[(all_frame_index>s[0]) & (all_frame_index<s[1])]
    peak_vels_s.append(np.nanmax(vel))
    if s[1]<(animal_border[0]/50):
        s_border+=1
peak_vels_s=np.array(peak_vels_s)




plt.figure()
plt.bar(range(2),  [np.mean(peak_vels_e),
                    np.mean(peak_vels_s)],
        color=['lightblue','violet'])

plt.scatter(np.zeros_like(peak_vels_e[:e_border]),peak_vels_e[:e_border], c='lightslategray')
plt.scatter(np.zeros_like(peak_vels_e[e_border:]),peak_vels_e[e_border:], c='navy')

plt.scatter(np.ones_like(peak_vels_s[:s_border]),peak_vels_s[:s_border], c='thistle')
plt.scatter(np.ones_like(peak_vels_s[s_border:]),peak_vels_s[s_border:], c='purple')

plt.xticks(range(2), ['escape','switch'])
plt.ylabel('velocity (cm/s)')
pf.remove_axes()
plt.title(f'{animals}\npeak escape velocity')


#%% statistic tests
# test for normality
from scipy import stats

hf.test_normality(peak_vels_e)
hf.test_normality(peak_vels_s)
 
# --> not normal

#Mann-whitney U test
u_statistic, p_value = stats.mannwhitneyu(peak_vels_e, peak_vels_s)

print(f"U statistic: {u_statistic}")
print(f"P-value: {p_value *2}") # because two tailed testing

plt.figure()
plt.hist(peak_vels_e, label='escape', facecolor='none', edgecolor='lightsteelblue', linewidth=2)
plt.hist(peak_vels_s, label='switch', facecolor='none', edgecolor='salmon', linewidth=2) 
plt.legend()





