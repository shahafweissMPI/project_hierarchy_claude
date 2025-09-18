"""
Created by Tom Kern
Last modified 04.08.2024

Neural dimensionality: How many neural factors do you need to cover 90% 
    of the variance  in the dataset (Fig1)

PCA of different behaviours:
    - Compute U factors using SVD
    - for specified factors, take only the timepoinst when target behaviour is happening
    - plot the actvity at these timepoints for factor 1 at axis 1, for factor 2 at axis 2, etc    

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

# Parameters
session='231213_0'
res=2#s Only so high to make plotting faster, you can decrase it as well
target_bs=['baseline','eat','attack','pullback'] #which behaviours should be compard?
cmaps=['Greys','Greens','Reds','Purples'] #COLORMAPS, used for density plot
colors=['grey','green','red','purple'] # COLORS used for dots on density plot
factor_plotting=[[0,1], [0,3], [1,3]] # How should facrtors be plotted? Each entry is for one 2D plot, numbers indicate factors that should be plotted togetehr

plt.style.use('dark_background')

           
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
#Resample n

ndata, n_time_index=hf.resample_ndata(ndata, n_time_index, res)

# run SVD
U, S, Vt = np.linalg.svd(zscore(ndata, axis=1).T, full_matrices=False)
#S: Singular values (how much variance does each dimension cover)
#U: time* factors
#Vt: neurons*factors, shows the loading of neurons on factors

# Restrict U to relevant number of factors
U=U[:,:np.max(factor_plotting)+1] 

#Neural dimensionality
cumsum=np.cumsum(S**2)
threshold=cumsum[-1]*.9
dim=np.min(np.where(cumsum>threshold))


plt.figure()
plt.plot(S)
plt.axvline(dim, c='w',ls='--')
plt.xlabel('Factors')
plt.ylabel('squared singular values')
plt.title(f'all neurons\ndimensionality {dim}/{len(ndata)} = {np.round(dim/len(ndata),2)}')
pf.remove_axes()



#%% SVD factors during different behaviours
factor_b=[]

for b_name in target_bs:
    
    if b_name == 'baseline':
        _, base_ind=hf.baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s[:-3])
        factor_b.append(U[base_ind,:])
    else:
        b=behaviour[behaviour['behaviours']==b_name]
        frames_s=b['frames_s'].to_numpy()
        
    
        
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
    





# This is commented out because it takes ages to compute
# Compares SVD of each behaviour against baselien separately, for factors 1-3

# for f in ([0,1], [0,3],[1,3]):
#     fig, axs = plt.subplots(3,3)
#     axs=axs.flatten()
#     fig.suptitle(f'Probability density\n factors {f[0]+1}, {f[1]+1}')
#     for i,(b_name, ax) in enumerate(zip(target_bs, axs)):
        
        
#         ax.set_title(f'{b_name}')
        
        
#         ax.scatter(x=factor_base[:, f[0]], y=factor_base[:, f[1]], color='grey', s=5)
#         ax.scatter(x=factor_b[i][:, f[0]], y=factor_b[i][:, f[1]], color='mediumorchid', s=10)
        
#         sns.kdeplot(x=factor_base[:, f[0]], y=factor_base[:, f[1]], cmap='binary', fill=True,thresh=.2, levels=10, alpha=.5, ax=ax)
#         sns.kdeplot(x=factor_b[i][:, f[0]], y=factor_b[i][:, f[1]], cmap='inferno',  fill=True,thresh=0.2, levels=10, alpha=.6,ax=ax)
    
#     pf.remove_axes(axs)
#     axs[8].set_xlabel(f'Factor {f[0]+1}')
#     axs[4].set_ylabel(f'Factor {f[1]+1}')




#%% attack vs switch vs escape vs base

for f in (factor_plotting):
    fig, ax = plt.subplots()
    plt.xlabel(f'Factor {f[0]+1}')
    plt.ylabel(f'Factor {f[1]+1}')
    plt.title(f'Probability density')
    pf.remove_axes()
    
    for ib, b_name in enumerate(target_bs):

        plt.scatter(x=factor_b[ib][:, f[0]], y=factor_b[ib][:, f[1]], color=colors[ib], s=10, label=b_name)  
        sns.kdeplot(x=factor_b[ib][:, f[0]], y=factor_b[ib][:, f[1]], cmap=cmaps[ib],  fill=True,thresh=0.2, levels=10, alpha=.6,ax=ax)

    plt.legend()










