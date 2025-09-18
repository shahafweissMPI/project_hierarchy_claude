"""
Created by Tom Kern
Last modified 04.08.2024

Neural dimensionality: How many neural factors do you need to cover 90% 
    of the variance  in an area(Fig1)
    
    
PCA of different behaviours:
    - take neurons from each area
    - Compute U factors of those neurons, using SVD
    - for specified factors, take only the timepoinst when target behaviour is happening
    - plot the actvity at these timepoints for factor 1 at axis 1, for factor 2 at axis 2, etc    

Plots
-For each area, makes plot 
    \\comparing SVD of  all target_bs against baseline separately
    \\Comparing SVD of different target_bs directly against each other
    

Limitations
-The different behaviours (switch, attack, baseline, escape) are not really well 
    comparable, since they have such vastly different number of timepoints
-downsampled to 2s bins, but mostly because that makes computation faster

"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore
import seaborn as sns
from time import time
from joblib import Parallel, delayed


# Parameters
session='231213_0'
plt.ioff()
plt.style.use('dark_background')
res=2#s; is so large only to speed up computation
target_areas=[['CIC','DCIC'],['DpWh'],['LPAG'],['VLPAG'], ['isRt'],['DRL']] # which areas to plot togetehr
target_bs=np.array(['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat'])
factor_plotting=[[0,1],[0,2],[1,2]] # which factors should be plotted against each other inwhich order?

direct_comp=['eat','attack','pullback'] #which behaviours should be compard?
cmaps=['Greens','Reds','Purples'] #COLORMAPS, used for density plot
colors=['green','red','purple'] # COLORS used for dots on density plot
    
savepath=r"E:\awesome_files\F_SSD_content\dimplots"


#load data
[dropped, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index, 
 velocity, 
 _, 
 _, 
 frame_index_s] = hf.load_preprocessed(session)




#%%SVD

plt.close('all')


#Resample n

ndata, n_time_index=hf.resample_ndata(ndata, n_time_index, res)



def run_script(anames, ndata, n_time_index, target_areas, frame_index_s, spath):
    plt.style.use('dark_background')
    
    region_ind=np.isin(n_region_index, anames)
    n=zscore(ndata[region_ind,:], axis=1)
    
    
    
    # run SVD
    U, S, Vt = np.linalg.svd(n.T, full_matrices=False)
    #S: Singular values (how much variance does each dimension cover)
    #U: time* factors
    #Vt: neurons*factors, shows the loading of neurons on factors
    
    # take only relevant factors
    U=U[:,:np.max(factor_plotting)+1]
    
    #Neural dimensionality
    cumsum=np.cumsum(S**2)
    threshold=cumsum[-1]*.9
    dim=np.min(np.where(cumsum>threshold))
    rel_dim=dim/len(n)
    
    plt.figure()
    plt.plot(S)
    plt.axvline(dim, c='w',ls='--')
    plt.xlabel('Factors')
    plt.ylabel('squared singular values')
    plt.title(f'{anames}\ndimensionality {dim}/{len(n)}= {np.round(rel_dim,2)}')
    pf.remove_axes()
    
    # plt.show()
    plt.savefig(fr'{savepath}\{anames}_dimensionality.tiff')
    plt.close()    



#SVD factors during different behaviours
    factor_b=[]
    
    for b_name in target_bs:
        b=behaviour[behaviour['behaviours']==b_name]       
        
        start_stops=hf.start_stop_array(behaviour, b_name, frame=False)
        
        
        #Make boolean index for when behaviour happens
        b_ind=np.zeros(ndata.shape[1])
        for stst in start_stops:        
            b_ind+=(n_time_index>stst[0]) & (n_time_index<stst[1])
        if sum(b_ind>1)>0:
            raise ValueError('sth is wrong with the computation of your b_ind')
        b_ind=b_ind.astype(bool)
        
        # get factor activity during b
        factor_b.append(U[b_ind,:])
        
    
    _, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:-3])
    factor_base=U[base_ind,:]
    
    #% probaility density

    
    #All against baseline
    for f in (factor_plotting):
        fig, axs = plt.subplots(3,3, figsize=(15,15))
        axs=axs.flatten()
        fig.suptitle(f'Probability density\n factors {f[0]+1}, {f[1]+1}')
        for i,(b_name, ax) in enumerate(zip(target_bs, axs)):
            
            
            ax.set_title(f'{b_name}')           
            
            ax.scatter(x=factor_base[:, f[0]], y=factor_base[:, f[1]], color='grey', s=5)
            ax.scatter(x=factor_b[i][:, f[0]], y=factor_b[i][:, f[1]], color='mediumorchid', s=10)
            
            sns.kdeplot(x=factor_base[:, f[0]], y=factor_base[:, f[1]], cmap='binary', fill=True,thresh=.2, levels=10, alpha=.5, ax=ax)
            sns.kdeplot(x=factor_b[i][:, f[0]], y=factor_b[i][:, f[1]], cmap='inferno',  fill=True,thresh=0.2, levels=10, alpha=.6,ax=ax)
        
        pf.remove_axes(axs)
        axs[8].set_xlabel(f'Factor {f[0]+1}')
        axs[4].set_ylabel(f'Factor {f[1]+1}')
        
        plt.savefig(fr'{spath}\{anames}_factors{f}_b_baseline.tiff')
        plt.close()  
        

# attack vs switch vs escape vs base
    for f in (factor_plotting):
        fig, ax = plt.subplots(figsize=(15,15))
        plt.xlabel(f'Factor {f[0]+1}')
        plt.ylabel(f'Factor {f[1]+1}')
        plt.title(f'Probability density')
        pf.remove_axes()
        
        # base
        plt.scatter(x=factor_base[:, f[0]], y=factor_base[:, f[1]], color='grey', s=5, label='baseline')
        sns.kdeplot(x=factor_base[:, f[0]], y=factor_base[:, f[1]], cmap='binary', fill=True,thresh=.2, levels=10, alpha=.5, ax=ax)

        # direct comparison
        for i, sp_b in enumerate(direct_comp):
            index=np.where(target_bs==sp_b)[0][0]
            
            plt.scatter(x=factor_b[index][:, f[0]], y=factor_b[index][:, f[1]], color=colors[i], s=10, label=sp_b)
            sns.kdeplot(x=factor_b[index][:, f[0]], y=factor_b[index][:, f[1]], cmap=cmaps[i],  fill=True,thresh=0.2, levels=10, alpha=.6,ax=ax)
        
        plt.legend()

        plt.savefig(fr'{savepath}\{anames}_factors{f}_b_comp.tiff')
        plt.close()  


############################# RUN SCRIPT ########################################

    
start=time()
Parallel(n_jobs=15)(delayed(run_script)(anames, 
                                                        ndata,
                                                        n_time_index, 
                                                        target_areas,
                                                        frame_index_s, 
                                                        savepath
                                                        ) for anames in target_areas)
stop=time()
print(f'whole script took {hf.convert_s(stop-start)}')



plt.ion()






