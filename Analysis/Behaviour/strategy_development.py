
"""
Created by Tom Kern
Last modified 04.08.2024

______________________________________________________________________________________
HUNTING STRATEGIES DEFINITION
-more hunt: Is true if within x seconds after finishing either approach or pursuit,
    a new approach or pursuit is started

- stop hunt: is true if within x seconds after finishing approach or pursuit, NO new
    approach or pursuit is started
- pullback: is true if within x seconds after hunt a pullback happens

--> x is determined by h_max_reaction
______________________________________________________________________________________
BAYESIAN MODEL
-adapted from Maggi et al (2024); https://elifesciences.org/articles/86491
-Models probability of binary strategies, e.g. escape vs no escape after loom
-Prediction for probability of startegy A being executed is calculated by 
calculating the proportion of previous trials in which this strategy was executed.
When calculating this, long ago trials receive less weight than recent trials

ESCAPE STRATEGIES
-escape/ no escape

HUNTING STRATEGIES
- more hunt/ not more hunt
- pullback/ no pullback
- stop hunt/ not stop hunt (i.e. either pullbak or more hunt)


PLOTS
-in figure 1 and 3, the dots represent times when the strategy could be executed,
i.e. in figure 1 they stand for looms, in figure 3 they mark times when hunting stops
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd
import seaborn as sns




animal='afm16505'
animals=['afm16505','afm16618']
plt.style.use('default')
e_max_reaction=5#s; how long after a loom an escape is allowed to happen
h_max_reaction=3#s; how long after a hunt will the next hunt be counted as continuation


#% model strategy developemnt
def strat_dev(strtg, decay_factor=.9, prior=.5):
    """
    models probability of a binary strategy being used

    Parameters
    ----------
    strtg : bool
        whether or not a strategy is used at each timepoint.
    decay_factor : float, optional
        how fast should long-ago startegy be forgotten. closer to 0 means 
        faster forgetting The default is .9.
    prior: what is the likelhood of the strategy before start
    Returns
    -------
    probs: float array
        for each trial, what is the probability of the given strategy being used.

    """
    probs=[]
    for i, s in enumerate(strtg):
        w_prior=prior*np.power(decay_factor, i)
        decay_function=np.power(decay_factor, range(i+1))[::-1]
        prob = (np.sum(decay_function * strtg[:i+1])+ w_prior) / (np.sum(decay_function) +1)
        
        probs.append(prob)
    return np.array(probs)





#% Collect all behaviour in one big v
bigB, all_frame_index, all_vel, sessionstarts=hf.get_bigB(animal, get_sessionstarts=True)
b_names=bigB['behaviours'].to_numpy()
start_stop=bigB['start_stop'].to_numpy()
frames_s=bigB['frames_s'].to_numpy()


          
#%% Escape strategy development

#get escape strategy
e_strtg=[]
e_strtg_time=[]
looms=frames_s[b_names=='loom']
for loom in looms:
    reactions=b_names[(frames_s> loom) & (frames_s<loom+e_max_reaction)]
    
    if 'escape' in reactions:
        e_strtg.append('escape')
    else:
        e_strtg.append('no escape')

    e_strtg_time.append(loom)
e_strtg=np.array(e_strtg)
e_strtg_time=np.array(e_strtg_time)

e_probs=strat_dev(e_strtg=='escape')


#plot this
fig, axs=plt.subplots(2,1)
pf.plot_events(bigB[np.isin(b_names, ['approach','pursuit','attack'])], ax=axs[1])
for i in sessionstarts:
    plt.axvline(i, ls='--',c='k', label='session start')
axs[1].plot(e_strtg_time,e_probs, c='k')
axs[1].scatter(looms, e_probs, c='slateblue', label='looms', s=50)
axs[1].set_ylabel('cumulative likelihood')
axs[1].set_title(f"{animal}\nlikeligood of escaping after loom")
pf.remove_axes(axs)
axs[1].set_xlabel('time (s)')
axs[1].set_ylim((.4,1.05))

hf.unique_legend()

axs[0].plot(looms,(e_strtg=='escape').astype(int),'.')
axs[0].set_yticks((0,1),('no escape','escape'))
axs[0].set_xlabel('number of looms')








#%% hunt strategy development

#Get times of hunting stops
hunt_bs=['approach','pursuit']
hunt_stops=frames_s[np.isin(b_names, hunt_bs) & (start_stop=='STOP')]


#Get index for when hunting is happening
hunt_ind=np.zeros_like(all_frame_index)
for i, b_name in enumerate(hunt_bs):

    b_start_stop=hf.start_stop_array(bigB, b_name, frame=False)
      
    for b in b_start_stop:
        hunt_ind+=(all_frame_index>(b[0])) & (all_frame_index<b[1])
hunt_ind=hunt_ind.astype(bool)
        
        
#get pullback strategy
h_strtg=[]
h_strtg_time=[]
for hunt in hunt_stops:
    reactions=b_names[ (frames_s> hunt) & (frames_s<hunt+h_max_reaction)]
    after_hunt=hunt_ind[(all_frame_index>hunt )& (all_frame_index< hunt+ h_max_reaction)]
    
    if 'pullback' in reactions:
        h_strtg.append('pullback')
    elif np.sum(after_hunt)>0:
        h_strtg.append('more hunt')
    else :
        h_strtg.append('stop hunting')
    
    h_strtg_time.append(hunt)
h_strtg=np.array(h_strtg)
h_strtg_time=np.array(h_strtg_time)





#%% plot hunt

#barplot
plt.figure()
plt.bar(range(3), [np.sum(h_strtg=='more hunt')/len(h_strtg),
                   np.sum(h_strtg=='stop hunting')/len(h_strtg),
                   np.sum(h_strtg=='pullback')/len(h_strtg)],
        color=['coral', 'gray','cadetblue'])

plt.ylabel('Percent')
plt.xticks(range(3), ['more hunt','stop hunting','pullback'])
pf.remove_axes()
plt.title(f'{animal}\nwhat happens after a hunting period?')


#%% All strategies
# plt.close('all')
fig, axs=plt.subplots(4)


for ax, name, c in zip(axs,
                       ['more hunt','stop hunting', 'pullback','escape'], 
                       ['coral','gray','cadetblue','lighblue']):
    
    pf.plot_events(bigB[np.isin(b_names, ['approach','pursuit','attack'])], ax=ax)
    for i in sessionstarts:
        ax.axvline(i, ls='--',c='k', label='session start')

    
    if name=='escape':
        ax.plot(e_strtg_time,e_probs, c='lightblue')
    else:
        ax.plot(h_strtg_time, strat_dev(h_strtg==name), label=name, c=c)
        ax.scatter(h_strtg_time, strat_dev(h_strtg==name), label=name, c='k')
    ax.set_xlim((h_strtg_time[0]-500, h_strtg_time[-1]+500))
    ax.set_ylim((0,1))
    ax.set_title(name, y=.9)
    ax.set_ylabel('predicted probability')
    
    # pf.plot_events(bigB[np.isin(b_names, ['loom'])], ax=ax)
    # hf.unique_legend(ax=ax)

axs[-2].set_xlabel('time(s)')
pf.remove_axes(axs)


