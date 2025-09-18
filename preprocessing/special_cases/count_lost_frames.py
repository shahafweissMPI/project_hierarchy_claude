import numpy as np
import preprocessFunctions as pp
import pandas as pd
import helperFunctions as hf
import matplotlib.pyplot as plt

# session numbers from paths.csv file
sessions=['240212', '240214', '240216', '240223', '240226_0', '240226_1', '240228']


for s in sessions:
    paths=pp.get_paths(s)
    
    #Get nidq signal
    (frame_index_s, 
     t1_s, 
     tend_s, 
     vframerate, 
     cut_rawData, 
     nidq_frame_index, 
     nidq_meta)= pp.cut_rawData2vframes(paths['nidq'])
    
    #count frames in video
    frames_in_video=pp.count_frames(paths['video'])
    

    print(f'session: {s}')
    print(f'video framerate: {vframerate}')
    print('unique distances between video frames:')
    print(pp.unique_float(np.diff(frame_index_s)))
    print(f'{frames_in_video-len(frame_index_s)} frames difference!!\n\n')
    

#%%

# nidq_path=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\test\240429_frame_drop_test2_g0\240429_frame_drop_test2_g0_t0.nidq.bin"
# vpath=r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\test\240429_frame_drop_test2_g0\cam0_2024-04-29-14-41-52USB.avi"
# (frame_index_s, 
#  t1_s, 
#  tend_s, 
#  vframerate, 
#  cut_rawData, 
#  nidq_frame_index, 
#  nidq_meta)= pp.cut_rawData2vframes(nidq_path)

# #count frames in video
# frames_in_video=pp.count_frames(vpath)
# print(f'{frames_in_video-len(frame_index_s)} frames difference!!\n\n')
