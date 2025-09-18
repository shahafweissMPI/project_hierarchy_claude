"""
Created by Tom Kern
Last modified 05/08/24

Takes output from s_num_perm.py from multiple sessions gives the percentage 
of neurons that are responsive to each behaviour in each region.
-can summarise different behaviours into a common categroy, e.g. approach and 
    pursuit into 'hunt'
    




Important
-at the moment this file only works, if the parameter 'taregt_bs' was the same 
    in all sessions that s_num_perm.py ran through. It is possibe to specify 
    behaviours that are not present in a certain session


Limitations
-if a neuron is positively tuned to one hunting behaviur, 
      but negativel to another, it will basically be counted like 2 tuned neurons.
- NOt all sessions have all behaviours, so there will be a different number of neurons
    for each behaviour, even within the same region
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf

#Parameters
plt.style.use('default')

#Choose behaviours to plot
quant_bs=[['approach','pursuit','attack'],'pullback','switch','escape']
    #Which behaviours to quantify. By putting multiple behaviours in 
    # brackets (like with [approach,pursuit,attack]), they will be 
    # counted together

quant_b_names=['hunt','pullback','switch','escape']
    # Necessary to give a name to mergd behvaiours



#Load data
savepath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\plotdata"

[all_region_change, #Sessions --> regions --> neurons*behaviours
 all_regions, #index into all_region_change
 target_bs #index into all_region_change
 ]=hf.load_tuning(savepath)



#Preassign
plus=np.full((len(all_regions), len(quant_bs)), np.nan)
minus=plus.copy()
num_neurons=plus.copy().astype(int)




#%%collect numbers of tuned neurons for each area across sessions
for i_r,region in enumerate(all_regions):
    
    # Merge data from the same regions across sessions
    r=[]
    for ses in all_region_change:
        r.append(ses[i_r])        
    r=np.vstack(r) #neurons* behaviour
    

    # Summarise data for each behaviour
    for i_b, b in enumerate(quant_bs):
        b_neurons=r[:,np.isin(target_bs, b)]
        
        # check if nans are he same for behaviours that are taken together
        isnan=np.isnan(b_neurons)
        condition = np.logical_not(np.logical_xor(isnan.all(axis=1), isnan.any(axis=1)))
        if sum(condition==False)!=0:
            print(f'there are {sum(condition==False)} inconsistent nan values, best would be 0')
            # raise ValueError('nan values inconsistent')

        b_neurons=b_neurons[~isnan.any(axis=1)]
        
        #filter for positive/ negativey tuned neurons
        plus[i_r, i_b]=a=np.sum(b_neurons>0, axis=1).astype(bool).sum()
        minus[i_r, i_b]=a=np.sum(b_neurons<0, axis=1).astype(bool).sum()
        num_neurons[i_r, i_b]=len(b_neurons)


# Get overall percentages
tuned=plus+minus
perc_tuned=tuned/num_neurons
perc_plus=plus/num_neurons
perc_minus=minus/num_neurons






#%%Plot
fig, axs=pf.subplots(len(all_regions))

ploti=0
for i_r, region in enumerate(all_regions):
    if np.sum(tuned[i_r])==0:
        print(f'{region} skipped, noting tuned to anything')
        continue

    axs[ploti].set_title(region)
    
    
    axs[ploti].bar(np.arange(len(quant_bs)), perc_tuned[i_r],color='k', label='total')
    axs[ploti].bar(np.arange(len(quant_bs))-.2, perc_plus[i_r], width=.3,linewidth=4, edgecolor='mediumseagreen', color='k', label='positive')
    axs[ploti].bar(np.arange(len(quant_bs))+.2, perc_minus[i_r], width=.3,linewidth=4, edgecolor='indianred',color='k', label='negative')
    
    axs[ploti].set_ylim((0,1.05))
    
    
    
    for i in range(len(perc_tuned[i_r])):
        axs[ploti].text(i, perc_tuned[i_r,i] + 0.05, f'n={num_neurons[i_r,i]}', ha = 'center')
    
    
    ploti+=1
    
axs[3].legend()
axs[0].set_ylabel('% responsive neurons')
axs[4].set_ylabel('% responsive neurons')

for i, ax in enumerate(axs):
    if i in [3,4,5,6]:
        axs[i].set_xticks(range(len(quant_bs)), quant_b_names)
    else:
        axs[i].set_xticks([])

