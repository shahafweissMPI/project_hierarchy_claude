"""
Created by Tom Kern
Last modified 04.08.2024

Using SVD + regression to understand how much neurons respond to the 
instinctive behvaiours that we marked

- Compute SVD on neural data
- Predict each SVD factor with all the behaviour information from boris (binary
     indices of when behaviour happens)
    --> Fig 2 shows how well each factor is predicted by instinctive behaviours
- to know which behaviorus contribute how much to the prediction of each factor,
    we can have a look at the beta weights from the regression. By multiplying
    beta weights with prediction performance (r2), we have a measure of how much
    each behaviour is reflected in each behaviour
    --> fig1 shows this for the 4 (num_factors) best predicted factors 

NOtes
- considerable downsampling necessary
- 'eating' has a major influence on this, especially the first factor

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore


session='231213_0'
plt.style.use('dark_background')

res=.5#s
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback']
# which behaviours should be included in X

num_factors=4 # how many factors should be plotted

      
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

resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, res)



#%%SVD

U, s, VT = np.linalg.svd(zscore(resampled_n, axis=1).T, full_matrices=False)

#Restrict factors to a reasonable number
U=U[:,:100]


#%% Regression

# Make X
X=np.zeros((len(resampled_t), len(target_bs)))

for i, b_name in enumerate(target_bs):
        
    start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
    for b in start_stop:
        b_ind=(resampled_t>(b[0])) & (resampled_t<b[1])
        X[:,i]+=b_ind
if np.sum(X[:,:-1]>1)!=0:
    raise ValueError('check your calculation of X')
    
    
# regression
Y=U.copy()
perf_r2, bs=hf.OLS_regression(X,Y,nfolds=5, normalise=False)




#%% Get weights

avg_weight=np.mean(np.array(bs), axis=0)
scaled_weight=avg_weight/np.sum(np.abs(avg_weight), axis=0)
weighted_weight=scaled_weight*perf_r2



#%%plot
#Factor weighting
best_f=np.where(perf_r2==np.max(perf_r2))[0][0]
target_factors=range(num_factors)

fig, axs=plt.subplots(1,len(target_factors), figsize=(20,10))
fig.suptitle('beta weights scaled by r2')
for i, ax in zip(target_factors, axs.flatten()):
    ax.set_title(f'Factor: {i+1}\nr2: {np.round(perf_r2[i],2)}')
    if i == best_f:
        ax.bar(range(len(weighted_weight)),weighted_weight[:,i], color='lightsteelblue')
    else:
        ax.bar(range(len(weighted_weight)),weighted_weight[:,i], color='lightslategray')
    ax.set_xticks(range(len(target_bs)), target_bs, rotation=45)
    ax.set_ylim((-.1,.1))
axs[0].set_ylabel('r2 * beta weights')
pf.remove_axes(axs)


# perf
plt.figure()
plt.title('explained variance per factor')
pf.logplot(perf_r2)
plt.xlabel('factors')
plt.ylabel('r2')
pf.remove_axes()

#raw factor (the one with highest r2)
plt.figure()
pf.plot_events(behaviour)
plt.xlabel('time (s)')
plt.ylabel('Factor activity')
plt.title(f'global\nfactor {best_f+1}\nr2: {np.round(perf_r2[best_f],3)}')

plt.plot(resampled_t, U[:,best_f], c='w')


