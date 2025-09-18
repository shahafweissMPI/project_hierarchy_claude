"""
Created by Tom Kern
Last modified 04.08.2024

plots all locations of the animal across the entire session for each session with 
velocity information
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd


separate_sessions=True # whether to make a new subplot for each session
animal='afm16924'
node_ind=2 #this is f_back
plt.style.use('dark_background')


paths=hf.get_paths()


# Premake figure
# plt.close('all')
fig=plt.figure()
plt.suptitle(f'{animal}\nall paths')
a_paths=paths[paths['Mouse_ID']==animal]
cols = int(np.ceil(np.sqrt(len(a_paths))))
rows = int(np.ceil(len(a_paths) / cols))

# plot arena
imframes=hf.read_frames(a_paths['video'].iloc[0], range(0,200))
im=np.percentile(imframes, 80, axis=0)
if not separate_sessions:
    plt.imshow(im,cmap='binary_r')
    pf.remove_axes(bottom=True, left=True, ticks=False)


for i, (index,ses) in enumerate(a_paths.iterrows()): 
    path=str(ses['preprocessed'])
    if path =='nan': # test if there is preprocessed data
        continue
    
    # load_data  
    tracking=np.load(fr'{path}\tracking.npy', allow_pickle=True).item()
    velocity=tracking['velocity']
    locations=tracking['locations']
    node_names=tracking['node_names']
    frame_index_s=tracking['frame_index_s']
    
    #Make plot
    if separate_sessions:
        plt.subplot(cols,rows, i+1)
        plt.title(f"{ses['session']}\nlength: {hf.convert_s(frame_index_s[-1])[0]}")
        plt.imshow(im,cmap='binary_r')
        pf.remove_axes(bottom=True, left=True, ticks=False)
    
    
    
    
    
    
   
    
    #Plot vels and locations of mouse in these frames
    
    val=plt.scatter(locations[:, node_ind, 0], 
                locations[:, node_ind, 1],
                c=velocity,
                cmap='inferno',
                vmin=0,
                vmax=90,
                s=.2)

cbar_ax = fig.add_axes([0.93, 0.2, 0.01, 0.6]) 
#distance from:        left, bott,width, height
cbar_ax.set_title('Velocity (cm/s)')
fig.colorbar(val,cax=cbar_ax)




