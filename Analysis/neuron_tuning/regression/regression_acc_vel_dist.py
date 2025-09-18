"""
Created by Tom Kern
Last modified 04.08.2024

Predicts each neurons activity based on velocity, accelaration, and shelter distance
. If the prediction works 
well (high r2), you know that this neuron changes it's firing with respective variables 
To know which neuron responds to which variable uniquely, you can look at how 
much each behaviour contributed to the prediction (beta weights)

- X is a 3 dimensional (1 dimension per variable) predictor, all zscored
- Y is what is predicted. It has as many dimensions as there are neurons
- The way regression works, all 3 predictors predict each dimension of Y separately
- The beta weights tell you how much each predictor contributed to each neurons prediction
- r2, between 0-1, tells you how good a prediction is. 0 means horrible 1 means perfect

Tuning
-For each neuron, multiply r2 by beta weights, to have a mixed measure of prediction
    performance and contribution of each variable

Figures
 Fig1: Tuning of each neuron to each behavipiur
 Fig2: r2 of each neuron
 Fig3: beta weighst of each behaviour on each neuron
 Fig4-6: Predictions of example neurons
 
Limitations
- needs downsampling
- Vanessa used a different way to compute this, you can find it in this paper:
    https://www.sciencedirect.com/science/article/pii/S0143416021000440
-you can probably increase performance a lot by also doing permutations


"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from scipy.stats import zscore


session='231213_0'
plt.style.use('dark_background')
target_bs=['Velocity', 'Acceleration','ShelterDist']
           
[dropped, 
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index, 
 velocity, 
 locations, 
 node_names, 
 frame_index_s] = hf.load_preprocessed(session)


#%%RUN REGRESSION
resolution=1#s
plt.close('all')
resampled_n, resampled_t=hf.resample_ndata(ndata, n_time_index, resolution)



#Calculate values
acc=hf.diff(velocity)
shelterdist=hf.get_shelterdist(locations, node_names=='f_back')

#resample
vel=hf.interp(velocity, frame_index_s[:-dropped], resampled_t)
acc=hf.interp(acc, frame_index_s[:-dropped], resampled_t)
shelterdist=hf.interp(shelterdist, frame_index_s[:-dropped], resampled_t)

#Make X and Y
X=np.array([vel,acc,shelterdist]).T
X=zscore(X, axis=0)

Y=zscore(resampled_n.T, axis=0)

#Regression
perf_r2, b_weights=hf.OLS_regression(X,Y,nfolds=5, normalise=False)

#Weight tuning
b_weights=np.mean(b_weights, axis=0).T
scaled_weights=b_weights/np.sum(np.abs(b_weights), axis=1)[:,None]

tuning=scaled_weights*perf_r2[:,None]

#Plot
values=[-.1, -.02,.02,  .1]
cmap=pf.make_cmap(['darkslategray', 'w', 'w', 'saddlebrown'], values)
cbar=pf.plot_tuning(tuning, 
               target_bs, 
               n_region_index,
               cmap=cmap, 
               vmin=values[0], 
               vmax=values[-1],
               lines=False)
cbar.set_label('scaled_weight')
plt.title(f'regression tuning\n resolution: {resolution}s')
print("""
      problems here
      -------------
      -If a neuron reacts strongly to escape, and dooes random shit the other times, it 
      will have a low r2, since escape time contributes only minimally to total time
      
      """)

#%% plot other random shit
      
plt.figure()     
cbar=plt.imshow(perf_r2[:,None],aspect='auto')  
plt.title('r2')
cbar=plt.colorbar()
cbar.set_label('r2')
# pf.region_ticks(n_region_index)
pf.remove_axes(bottom=True, ticks=False)

cbar=pf.plot_tuning(b_weights, 
               target_bs, 
               n_region_index,
               cmap='viridis', 
               vmin=np.percentile(b_weights, 3), 
               vmax=np.percentile(b_weights, 97),
               lines=False)
cbar.set_label('weight')
plt.title(f'beta weights\n resolution: {resolution}s')
      

      
      
#PLOT best PREDICTION NO CROSS VALIDATION
Y_OLS, B_OLS, r2= hf.regression_no_kfolds(X,Y)

best_n=np.argpartition(r2, -3)[-3:]

for n_ind in best_n:
    plt.figure()
    plt.title(f'neuron: {n_ind}\n resolution: {resolution}s')

    plt.plot(Y[:,n_ind], label='data')
    plt.plot(Y_OLS[:,n_ind], label='prediction')
    pf.plot_events(behaviour)
    hf.unique_legend
    pf.remove_axes()
    plt.ylabel('firing Hz')
    plt.xlabel('time (s)')
    plt.suptitle('not cross validated!!!')
    

