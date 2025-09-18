"""
Created by Tom Kern
Last modified 04.08.2024

Predicts each neurons activity based on behaviour labels. If the prediction works 
well (high r2), you know that this neuron changes it's firing when the behaviours 
happen. To know which neuron responds to which behaviour, you can look at how 
much each behaviour contributed to the prediction (beta weights)

- X is a 10 dimensional (1 dimension per behaviour) binary predictor
    that is 1 when the behaviour happens and 0 when it doesn't
- Y is what is predicted. It has as many dimensions as there are neurons
- The way regression works, all 10 predictors predict each dimension of Y separately
- The beta weights tell you how much each predictor contributed to each neurons prediction
- r2, between 0-1, tells you how good a prediction is. 0 means horrible 1 means perfect

Tuning
-For each neuron, multiply r2 by beta weights, to have a mixed measure of prediction
    performance and contribution of each behaviour

Figures
 Fig1: Tuning of each neuron to each behavipiur
 Fig2: r2 of each neuron
 Fig3: beta weighst of each behaviour on each neuron
 Fig4: beta weights of 3 example neurons
 Fig5 same beta weights after scaling
 Fig6: Neural traces together with predictions, for same example neurons

Limitations
-Needs some downsampling, as it works better with non-binary data
-If a neuron reacts strongly to escape, and dooes random shit the other times, it 
    will have a low r2, since escape time contributes only minimally to total time
-you can probably increase performance a lot by also doing permutations


"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore


session='231213_0'
plt.style.use('dark_background')
target_bs=['approach', 'pursuit', 'attack', 'switch', 'startle','freeze', 'escape','pullback',  'eat']#, 'velocity']
SVD_analysis=False
           
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


#%%RUN REGRESSION
resolution=1#s
resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, resolution)

#Make predictor (X) out of behaviour labels
X=np.zeros((len(resampled_t), len(target_bs)))
for i, b_name in enumerate(target_bs):
    if b_name =='velocity':
        vel=hf.interp(velocity, frame_index_s[:-dropped], resampled_t)
        X[:,-1]=zscore(vel)
        continue
        
    start_stop=hf.start_stop_array(behaviour, b_name, frame=False)
    for b in start_stop:
        b_ind=(resampled_t>(b[0])) & (resampled_t<b[1])
        X[:,i]+=b_ind
if np.sum(X[:,:-1]>1)!=0:
    raise ValueError('check your calculation of X')


# Get predicted (Y) from downsampled neural data
Y=zscore(resampled_n.T, axis=0)


# Run regression
perf_r2, b_weights=hf.OLS_regression(X,Y,nfolds=5, normalise=False)


# Scale beta weights by prediction performance (r2)
b_weights=np.mean(b_weights, axis=0).T
scaled_weights=b_weights/np.sum(np.abs(b_weights), axis=1)[:,None]
tuning=scaled_weights*perf_r2[:,None]



#Plot
values=[-.1, -.02,.02,  .1]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)
cbar=pf.plot_tuning(tuning[:,:], 
               target_bs[:], 
               n_region_index,
               cmap=cmap, 
               vmin=values[0], 
               vmax=values[-1])
cbar.set_label('scaled_weight')
plt.title(f'regression tuning\n resolution: {resolution}s')





#%% plot examples to illustrate what you're doing here
      
plt.figure()     
cbar=plt.imshow(perf_r2[:,None],aspect='auto')  
plt.title('r2')
cbar=plt.colorbar()
cbar.set_label('r2')
# pf.region_ticks(n_region_index)
pf.remove_axes(bottom=True, ticks=False)

cbar=pf.plot_tuning(scaled_weights, 
               target_bs, 
               n_region_index,
               cmap='viridis', 
               vmin=-.5,#np.percentile(b_weights, 3), 
               vmax=.5)#np.percentile(b_weights, 97))
cbar.set_label('weight')
plt.title(f'beta weights\n resolution: {resolution}s')
      

#weights
fig, axs=plt.subplots(3,1)
fig.suptitle('beta weights')
for n, ax in zip([153,158,161], axs):

    ax.set_ylabel('scaled beta weights')
    ax.bar(range(len(target_bs)),scaled_weights[n], color='w')
    ax.set_xticks(range(len(target_bs)), target_bs, rotation=45)
    ax.set_title(f'neuron: {n}')
pf.remove_axes(axs)

#tuning
fig, axs=plt.subplots(3,1)
fig.suptitle('tuning')
for n, ax in zip([153,158,161], axs):

    ax.set_ylabel('tuning')
    ax.bar(range(len(target_bs)),tuning[n], color='w')
    ax.set_xticks(range(len(target_bs)), target_bs, rotation=45)
    ax.set_title(f'neuron: {n}')
    ax.set_ylim((-.1,.1))
pf.remove_axes(axs)



  
      
#PLOT PREDICTION NO CROSS VALIDATION
Y_OLS, B_OLS, r2= hf.regression_no_kfolds(X,Y)


b_names=behaviour['behaviours'].to_numpy()
    
fig, axs=plt.subplots(3,1, sharex=True)
for i, n in enumerate([153,158,161]):
    axs[i].set_title(f'neuron {n}')
    axs[i].set_ylabel('zscore', y=.9)
    axs[i].plot(Y[:,n], c='w', label='data')
    axs[i].plot(Y_OLS[:,n], label='prediction', c='orange')
    
    pf.plot_events(behaviour[np.isin(b_names, ['attack','escape','switch'])], ax=axs[i])
    
axs[2].set_xlabel('time (s)')
pf.remove_axes(axs)
plt.xlim((3050,3650))
hf.unique_legend(ax=axs[0])








