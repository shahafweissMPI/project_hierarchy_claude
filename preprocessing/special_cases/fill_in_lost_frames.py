import helperFunctions as hf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import cv2
from tqdm import tqdm



#%%Inputs:
session='231218_0'    
new_vid_path=r"F:\scratch\deleteme.avi"


paths=hf.get_paths(session)

#Path to csv timestamps
csv_path=paths['csv_timestamp']
    
#Path to video
vpath= paths['video']

#Under which path should the new video be saved?






#%% Insert frames

#read video and create out writer
cap = cv2.VideoCapture(vpath)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out=cv2.VideoWriter(new_vid_path, fourcc, fps, (width, height))




#Read camera timestamps
csv=pd.read_csv(csv_path)
camstamps=csv['Value']
diffs=hf.diff(camstamps)
std_diffs=diffs/np.median(diffs) # the normal distance between two subsequent frames

#Sanity check
if np.nanmax(np.abs( np.unique(std_diffs) -np.round((np.unique(std_diffs)))))> 0.001:
    raise ValueError ('some distances between frames are not multiples of normal distance')
    #This means you don't exactly know how many frames were lost there


#Fill in stamps
filled_stamps = []
exc_count=1
for i, diff, stamp in zip( tqdm(range(len(camstamps)), desc="Processing Video..."), std_diffs, camstamps):
    #read and write first frame
    if i==0:
        ret, frame = cap.read()        
        out.write(frame)
        continue
    
    
    # handle (very rare) exceptioon of negative diff
    try:
        if  (diffs[i-1]<-0.001) or (diffs[i+1]<-0.001) or (diff<-0.001): #this is to deal with one special case where no frame was lost, but they had a wrong order
            print(f'special case: wrong order of frames {exc_count}/3 \n(this should happen EXACTLY thrice)')
            ret, frame = cap.read()        
            out.write(frame)
            exc_count+=1
            continue
    except IndexError  as e:
        #  this error is because it is the last iteration
        
        if e.args[0] != f'index {i+1} is out of bounds for axis 0 with size {i+1}': # in case a different error occurs
            raise IndexError(e)
    
    
    # insert the previous frame, if there was a frame skipped    
    while np.abs(diff-1) > 0.001:
        out.write(frame) 
        diff-=1
        filled_stamps.append(i)
    
    
    
    # read frame and write it, if there was no exception
    ret, frame = cap.read()
    
    out.write(frame)
    
cap.release()
out.release()
print(f'\n\nfilled in {len(filled_stamps)} stamps')











