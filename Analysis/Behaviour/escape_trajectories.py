"""
Created by Tom Kern
Last modified 04.08.2024

for sessions that were preprocessed, plots velocity and position of escape frames.
separate_sessions: Wether to plot all escapes from one animal on a single plot,
or to make a separate plot per session
"""


import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
import pandas as pd


separate_sessions=True # whether to make a new subplot for each session
animal='afm16505'
node_ind=2 #this si f_back
plt.style.use('dark_background')


paths=hf.get_paths()


# Premake figure
# plt.close('all')
fig=plt.figure()
plt.suptitle(f'{animal}\nonly escapes')
a_paths=paths[paths['Mouse_ID']==animal]

# plot arena
imframes=hf.read_frames(a_paths['video'].iloc[0], range(0,200))
im=np.percentile(imframes, 80, axis=0)
if not separate_sessions:
    plt.imshow(im,cmap='binary_r')
    pf.remove_axes(bottom=True, left=True, ticks=False)


for i, (index,ses) in enumerate(a_paths.iterrows()): 
    if separate_sessions:
        plt.subplot(1,len(a_paths), i+1)
        plt.title(ses['session'])
        plt.imshow(im,cmap='binary_r')
        pf.remove_axes(bottom=True, left=True, ticks=False)
    
    path=str(ses['preprocessed'])
    if path =='nan':
        continue
    
    
    
    
    # load_data
    behaviour=pd.read_csv(fr'{path}\behaviour.csv')    
    tracking=np.load(fr'{path}\tracking.npy', allow_pickle=True).item()
    velocity=tracking['velocity']
    locations=tracking['locations']
    node_names=tracking['node_names']
    frame_index_s=tracking['frame_index_s']



    
    # In which frames is an escape actively happening?
    escapes=behaviour[behaviour['behaviours']=='escape']
    start_stop=escapes['start_stop'].to_numpy().astype(str)
    eframes=escapes['frames'].to_numpy().astype(int)
    if eframes.size ==0:
        print(f"{ses['session']} has no escapes, skipping")
        continue
    #Deleteme
    e=eframes[start_stop=='START']
    if np.nanmin(np.diff(e))< (50*5):
        raise ValueError('there are two escapes which should be merged')
    

    
    
    all_eframes=[]
    for i, s in enumerate(start_stop):
        if s!='START':
            continue
        all_eframes.append(np.arange(eframes[i], eframes[i+1]))
        
    all_eframes=np.hstack(all_eframes)
    
    #Plot vels and locations of mouse in these frames
    
    
    
    val=plt.scatter(locations[all_eframes, node_ind, 0], 
                locations[all_eframes, node_ind, 1],
                c=velocity[all_eframes],
                cmap='inferno',
                vmin=0,
                vmax=130,
                s=5)
    
cbar_ax = fig.add_axes([0.93, 0.2, 0.01, 0.6]) 
#distance from:        left, bott,width, height
cbar_ax.set_title('Velocity (cm/s)')
fig.colorbar(val,cax=cbar_ax)




