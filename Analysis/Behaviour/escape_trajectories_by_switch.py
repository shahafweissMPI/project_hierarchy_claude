"""
Created by Tom Kern
Last modified 04.08.2024

Shows velocity and position during escapes for all preprocessed sessions from
specified animal. Divides them into regular escapes and switch escapes

rad: how many frames (s*50) before an escape must a hunting behaviour terminate,
for that escape to be considered a switch

"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd



animal='afm16924'
node_ind=2 #this si f_back
plt.style.use('default')
rad=3 *50#how many seconds before an escape must hunting occur for it to be a switch
hunting_bs=['approach','pursuit','attack','eat']

paths=hf.get_paths()

cmap=pf.make_cmap(['steelblue','w','darkorange'], [0,20,80])

#%%Initialise figure
fig, axs=plt.subplots(1,2)
plt.suptitle(f'{animal}\n escape trajectories')
pf.remove_axes(axs, rem_all=True )


# plot arena as background 
a_paths=paths[paths['Mouse_ID']=='afm16618']
imframes=hf.read_frames(a_paths['video'].iloc[0], range(500))
im=np.percentile(imframes, 70, axis=0)

for ax in axs:
    ax.imshow(im,cmap='binary_r')
#%%

bigB, all_frame_index, all_vel, all_locs=hf.get_bigB(animal, locs=True)
b_frames=bigB['frames']
bs=bigB['behaviours']

for ax, bname in zip(axs, ['escape', 'switch']):

   e_start_stop=hf.start_stop_array(bigB, 'escape', frame=True)
   
    
   frames=[]
   for i, e in enumerate(e_start_stop):
       
       around_e_ind=(b_frames>(e[0]-rad)) & (b_frames< e[1])
       around_e=bs[around_e_ind]
       
       if (sum(np.isin(around_e, hunting_bs))>0) & (bname=='switch'):
           frames.append(np.arange(e[0], e[1]))
       if (sum(np.isin(around_e, hunting_bs))==0) & (bname=='escape'):
           frames.append(np.arange(e[0], e[1]))
   frames=np.hstack(frames)
    
   
    
   
    
   
    
    # plot velocity
    
   val=ax.scatter(all_locs[frames, node_ind, 0], 
                all_locs[frames, node_ind, 1],
                c=all_vel[frames],
                cmap=cmap,
                vmin=0,
                vmax=100,
                s=5)
   ax.set_title(bname)
        
cbar_ax = fig.add_axes([0.93, 0.2, 0.01, 0.6]) 
#distance from:        left, bott,width, height
cbar_ax.set_title('Velocity (cm/s)')
fig.colorbar(val,cax=cbar_ax)
    
    


