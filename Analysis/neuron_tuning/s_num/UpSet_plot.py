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
from upsetplot import from_memberships, UpSet


#Paramters
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



#%%collect numbers of tuned neurons for each area across sessions



def get_unique_combs(region, only_pos=False, only_neg=False):
    if only_pos and only_neg:
        raise ValueError ('They cant both be True at the same time, idot')
   
    # Stack all data from the same region across sessions
    r=[]
    i_r= np.where(all_regions==region)[0][0]
    for ses in all_region_change:
        r.append(ses[i_r])
    r=np.vstack(r) #neurons* behaviour  
    
    #Get total number of tuned neurons (irrespective of what are the target behaviours !!!) 
    nans_dropped[np.isnan(nans_zeroed)]=0
    tuned_neurons=np.sum(nans_zeroed!=0, axis=1).astype(bool).sum()
    #Filter out posirtively or negative tuned neuorns
    if only_pos:
        r[r<1]=0
    if only_neg:
        r[r>1]=0
    
    num_neurons=len(r)
    
    
    #make a matrix with just the quant behaviours
    quant_tuning=[] #neurons * behaviour
    nans=np.zeros(num_neurons)
    for b in quant_bs:
        b_neurons=r[:,np.isin(target_bs, b)]
        b_tuning_pos=np.sum(b_neurons>0,axis=1)
        b_tuning_neg=np.sum(b_neurons<0,axis=1)
        nans+=np.sum(np.isnan(b_neurons), axis=1)
        
        """FIX THIS!!!!
        
        """
        
        
        # In case  one quant_b is made up of multiple behaviours (like hunting), set all nan if one is nan
        b_tuning[np.isnan(b_neurons).any(axis=1)]=np.nan
        
       
        quant_tuning.append(b_tuning)
       
        
    
    # Exclude neurons that weren't exposed to all behaviours
    quant_tuning=pd.DataFrame(quant_tuning).T.dropna(axis=0).to_numpy().astype(int)
    
    #Skip regions with no tuned neurons
    if np.sum(quant_tuning)==0:
        print(f'skipping {region}, nothing is tuned')
        return [], [], []
    
    
    
    

#% Get unique tuning combinations

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
    # # tuned_neurons=np.sum(np.sum(quant_tuning, axis=1).astype(bool))
    # if tuned_neurons != sum(unique_values):
    #     raise ValueError('unique value computation wrong')
    if sum(np.array(unique_values)<0)>0:
        raise ValueError('uniqueee value computation wrong')
    
    #Get percentages
    unique_perc=np.array(unique_values)/tuned_neurons
    tot_perc=np.sum(unique_values)/num_neurons


    return unique_perc, all_combs, tot_perc
    
# """
# all_combs: All possible combinations of tuning. Numbers refer to behaviours in quant_bs
#     e.g. 1,2 means a neuron that is tuned to both pullback and switch
# unique_perc: For each combination in all_combs, how many neurons are there with exactly 
#     this combination? ( neurons tuned to 1,2,3 are also tuned to 1,2 or just 1.
#     I call this 'unique' combinations, since I calculate out all of this
#     overlap, so that 1,2 only contains neurons tuned to exctly this 
#     combination, and nothing more)

# """




#%% Make UpSetPlot

for region in target_regions:

    pos_perc, _, _ =get_unique_combs(region, only_pos=True)
    neg_perc, _, _ =get_unique_combs(region, only_neg=True)
    overall_perc, all_combs, tot_perc =get_unique_combs(region)



    string_combs=[]
    for entry in all_combs:
        new_entry=[quant_b_names[i] for i in entry]
        string_combs.append(new_entry)
    
    data=from_memberships(string_combs, unique_perc)
    
    
    # Create the UpSet plot
    fig=plt.figure()
    UpSet(data, 
          show_percentages=False,
          with_lines=False,
          sort_by='cardinality').plot(fig=fig)
    
    fig.suptitle(region)
