"""
Created by Tom Kern
Last modified 05/08/24

Takes output from s_num_perm.py from multiple sessions. It l tells you how many behaviours each neuron is tuned to
in each area. 
!!!Unless specifed otherwise, the percentages refer not to the total 
number of neurons, but to the number of neurons tuned to something!!!

Figures
-Fig1: Barplot, showing all existing tuning combinations in an area.
    Barcolors correspond to number of behaviours
-Fig2: same information as Fig1, but in Venn plot. This only works properly
    for up to 3 behaviours
-Fig3: Avg number of behaviours a neuron in a certain rgeion is tuned to.
    Blue line indicates average across regions
-Fig4: From all neurons in a region, how many are responsive to at least
    one behaviour?

Limitations
- can only do 3-way Venns, even if a neuron is tuned to 4 different behaviours
    (for doing 4-way Venns you would need to install venny4py, which isn't a 
     great library)
        -->For the Venns to work properly you can only specify 3 quant_bs 
        -->However, for figure 1 you can give as many behaviours as you want
        
- Neurons that weren't exposed to all behaviuors are excluded. E.g. if you have 
    3 sessions where there is only escape and nothing else, you might think that
    there are more only-escape-neurons than there actually are. However, this might
    be a bit annoying when trying to integrate hunting-escape and pup-retrieval-escape 
    sessions
    
Notes
-You might have to install 'matplotlib-venn'
    --> conda install -c conda-forge matplotlib-venn

    
"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import pandas as pd
import helperFunctions as hf
from matplotlib_venn import venn3_circles, venn3
from venny4py.venny4py import venny4py

#Paramters
svg_optimised=False #Gives you only circles, bit nicer to handle in AI
loadpath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\plotdata"

quant_bs=[['approach','pursuit','attack'],'pullback','switch','escape']
    #Which behaviours to quantify. By putting multiple behaviours in 
    # brackets (like with [approach,pursuit,attack]), they will be 
    # counted together

quant_b_names=['hunt','pullback','switch','escape']
    # Necessary to give a name to mergd behvaiours
plt.style.use('default')



#Load data
all_region_change, all_regions,target_bs=hf.load_tuning(loadpath)

target_regions=all_regions.copy()
target_regions=['DpWh','LPAG','VLPAG']


#Premake figure
barfig, baraxs= pf.subplots(len(target_regions))
baraxs=baraxs.flatten()
pf.remove_axes(baraxs)

vfig, v_axs=pf.subplots(len(target_regions))

v4fig, v4_axs=pf.subplots(len(target_regions))


#%%collect numbers of tuned neurons for each area across sessions
avg_bs=[]
tot_perc=[]
iax=0

#Go through all regions
for i_r,region in enumerate(all_regions):
    if not region in target_regions:
        continue
    
   
    # Stack all data from the same region across sessions
    r=[]
    for ses in all_region_change:
        r.append(ses[i_r])
    r=np.vstack(r) #neurons* behaviour   
    
    num_neurons=len(r)
    
    
    #make a matrix with just the quant behaviours
    quant_tuning=[] #neurons * behaviour
    for b in quant_bs:
        b_neurons=r[:,np.isin(target_bs, b)]
        b_tuning=np.sum(b_neurons!=0, axis=1).astype(bool).astype(float)
        b_tuning[np.isnan(b_neurons).any(axis=1)]=np.nan
        
        # check if nans are the same for behaviours that are taken together
        isnan=np.isnan(b_neurons)
        condition = np.logical_not(np.logical_xor(isnan.all(axis=1), isnan.any(axis=1)))
        if sum(condition==False)!=0:
            print(f'\nthere are {sum(condition==False)} inconsistent nan values, best would be 0\n')
            # raise ValueError('nan values inconsistent')        
        quant_tuning.append(b_tuning)
    
        
    
    # Exclude neurons that weren't exposed to all behaviours
    quant_tuning=pd.DataFrame(quant_tuning).T.dropna(axis=0).to_numpy().astype(int)
    
    #Skip regions with no tuned neurons
    if np.sum(quant_tuning)==0:
        print(f'skipping {region}, nothing is tuned')
        avg_bs.append(np.nan)
        tot_perc.append(np.nan)
        continue
    
    
    
    

#%% Get unique tuning combinations

    # Get all possible combinations of tuning
    all_combs=hf.combos(len(quant_bs))
    
    #How many neurons are tuned to each combination (not uniquely)
    comb_values=[]
    for comb in all_combs:
        ind = sum(quant_tuning[:, i] for i in comb) == len(comb)
        comb_values.append(ind.sum())
    
    
   #Calculae out the overlap, to get only unique combinations
    unique_values=comb_values.copy()
    for i_high in range(len(all_combs)):
        
        # get combinations from largest to smallest
        reverse_i=len(all_combs)-i_high -1
        high_comb=all_combs[reverse_i]
        
        #go through all combinations of lower order, that contain relevant numbers
        for i_low, low_comb in enumerate(all_combs):            
            
            if (sum(np.isin(low_comb, high_comb))== len(low_comb)) & (len(high_comb)>len(low_comb)):
                unique_values[i_low]-=unique_values[reverse_i]
    
    
    #sanity check
    tuned_neurons=np.sum(np.sum(quant_tuning, axis=1).astype(bool))
    if tuned_neurons != sum(unique_values):
        raise ValueError('unique value computation wrong')
    if sum(np.array(unique_values)<0)>0:
        raise ValueError('uniqueee value computation wrong')
    
    #Get percentages
    unique_perc=np.array(unique_values)/tuned_neurons
    tot_perc.append(np.sum(unique_values)/num_neurons)


    # get only combinations that really exist
    names=[]
    values=[]
    existing_combs=[]
    for value, comb in zip(unique_perc, all_combs):
        if value==0:
            continue
        
        names.append([quant_b_names[i] for i in comb])
        values.append(value)
        existing_combs.append(comb)
    
    """
    all_combs: All possible combinations of tuning. Numbers refer to behaviours in quant_bs
        e.g. 1,2 means a neuron that is tuned to both pullback and switch
    unique_perc: For each combination in all_combs, how many neurons are there with exactly 
        this combination? ( neurons tuned to 1,2,3 are also tuned to 1,2 or just 1.
        I call this 'unique' combinations, since I calculate out all of this
        overlap, so that 1,2 only contains neurons tuned to exctly this 
        combination, and nothing more)
        
    existing_combs: Contains only those combinations where that at least one 
        neuron in the region has
    values: For each combination in existing_combs, what percentage of neurons
        shows this combination of tuning?
    
    """
    



#%% Plot all existing combinations as barplot
    ticnames=[', '.join(n) for n in names]
    
    # plot
    
    baraxs[iax].set_title(f'{region}\n {len(quant_tuning)} neurons\n {tuned_neurons} tuned neurons')
    for i, (v, c) in enumerate(zip(values, existing_combs)):
        if len(c)==1:
            baraxs[iax].bar(i, v, color='silver')
        if len(c)==2:
            baraxs[iax].bar(i, v, color='dimgrey')
        if len(c)==3:
            baraxs[iax].bar(i, v, color='darkblue')
        if len(c)==4:
            baraxs[iax].bar(i, v, color='k')
            
    baraxs[iax].set_xticks(range(len(names)), ticnames, rotation=20 )
    
    baraxs[iax].set_ylabel('fraction of neurons, from all tuned neurons')
    baraxs[iax].set_ylim((0,1))
    # plt.savefig(rf"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\area_tuning\combinations\{region}.svg")
    # plt.savefig(rf"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\Figures\area_tuning\combinations\{region}.png")








#%% Make 3-way Venn of tuning combinations
 
    vind=np.array(['(0,)','(1,)','(0, 1)','(2,)','(0, 2)','(1, 2)','(0, 1, 2)'])
    
    str_combs=np.array([str(t) for t in all_combs])
    
    p=[unique_perc[str_combs==i][0] for i in vind]
    
    # (Abc, aBc, ABc, abC, AbC, aBC, ABC)

    if svg_optimised:
        venn=venn3_circles(subsets=p, ax=v_axs[iax])
    else:
        venn=venn3(subsets=np.round(p, 2), ax=v_axs[iax],set_labels=quant_b_names)

    v_axs[iax].set_title(region)

    
    
    
    
    num_n=([len(i) for i in all_combs])
    avg_bs.append(sum(num_n*unique_perc))
   



#%% Make 4-way Venn of tuning combinations

    sets={}
    df=pd.DataFrame(quant_tuning, columns=quant_b_names)
    for column in df.columns:
        sets[column] = set(df[df[column] == True].index)
    
    venny4py(sets=sets, asax=v4_axs[iax])

    iax+=1




#%% Summary plots
# how many behaviours per neuron???
plt.figure()
plt.bar(range(len(target_regions)), avg_bs, color='dimgrey')
plt.ylabel('avg number of behaviours 1 neuron is tuned to')
plt.xticks(range(len(target_regions)), target_regions)
plt.axhline(np.nanmean(avg_bs))



#How many neyurons tuned per area???
plt.figure()
plt.bar(range(len(tot_perc)),tot_perc, color='dimgrey')
plt.xticks(range(len(tot_perc)), target_regions)
plt.ylabel('% of tuned neurons')
pf.remove_axes()
plt.axhline(np.nanmean(tot_perc))

