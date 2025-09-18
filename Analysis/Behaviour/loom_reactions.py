"""
Created by Tom Kern
Last modified 04.08.2024

Shows escape probability across sessions and animals total (figure1) and
divided into regular escapes and switch-escapes. It also has the category
'some reaction' which means startle or freeze, but no escape 

max_reaction: time after loom for whcih behaviour is collected. 
"""



import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd





animals=['afm16924']
plt.style.use('default')

max_reaction=7#s


#%% Collect all behaviour in one big v

bigB, all_frame_index, _, _=hf.bigB_multiple_animals(animals)

b_names=bigB['behaviours'].to_numpy()
start_stop=bigB['start_stop'].to_numpy()
frames_s=bigB['frames_s'].to_numpy()

#%% Loom reactions

def loom_proportions(looms,frames_s, b_names, max_reaction=5):
    r=[]
    
    for loom in looms:
        
        reactions=b_names[(frames_s> loom) & (frames_s<loom+max_reaction)]
        
        if 'loom' in reactions:
            loom_index = np.where(reactions == 'loom')[0][0]
            reactions = reactions[:loom_index]
        
        if 'escape' in reactions:
            r.append('escape')
        elif sum(np.isin(['startle','freeze'], reactions))>0:
            r.append('some reaction')
        else:
            r.append('no reaction')
    r=np.array(r)
    names=['escape', 'some reaction', 'no reaction']
    sums=np.array([sum(r==names[0]),
                       sum(r==names[1]),
                       sum(r==names[2])]
                  )/ len(r)
    return r,names,sums

########################## Indiscriminative ###################################
(r,
 names,
 sums)=loom_proportions(frames_s[b_names=='loom'],
                              frames_s,
                              b_names, 
                              max_reaction)

#fig: total loom reactions
plt.figure()

plt.bar(range(3), sums, color='slategray')
plt.xticks(range(len(names)), names)
pf.remove_axes()
plt.title(f'{animals}\nreactions to looms -- total')
plt.ylabel('percent')
plt.ylim((0,1))
###################### during hunt vs regular #################################

otherlooms, huntlooms=hf.divide_looms(all_frame_index, bigB, radiance=2)


(hunt_r,
 _,
 hunt_sums)=loom_proportions(huntlooms,
                        frames_s,
                        b_names, 
                        max_reaction)

(other_r,
 _,
 other_sums)=loom_proportions(otherlooms,
                        frames_s,
                        b_names, 
                        max_reaction)
                              
plt.figure()
plt.bar([0,3,6], other_sums, label='escape looms', color='lightblue')
plt.bar([1,4,7], hunt_sums, label='switch looms', color='violet')
plt.xticks([.5,3.5,6.5], names)
plt.legend()
pf.remove_axes()
plt.title(f'{animals}\nreactions to looms')
plt.ylabel('percent')
plt.ylim((0,1))