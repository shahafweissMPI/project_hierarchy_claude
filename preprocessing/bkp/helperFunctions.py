import numpy as np
import preprocessFunctions as pp
import scipy.io
try:
    import cv2
except ModuleNotFoundError:
    print("cv2 couldn't be imported, video processing won't work")
#import winsound
import sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import KFold
from scipy.stats import zscore
from numpy.random import choice
from scipy import stats
import os

def unique_float(vector, precision=10):
    """ calculates unique values in vector. It ignores differences that are lower than 'precision' decimals """
    
    rounded_vector = np.round(vector, precision)
    unique_values = np.unique(rounded_vector)
    return unique_values

def mat2npy (filepath, savepath=None, openfile=True):
    """
    Loads mat data as numpy file, or saves it in numpy format
    
    
    Parameters
    ----------
    filepath : Where the .mat file is

    savepath : optional; under what path the new variable should be saved.
                If empty, same path as filepath will be used

    openfile : if True, opens the npy file WITHOUT saving it

    Returns
    -------
    npy data.

    """
    mat_data=scipy.io.loadmat(filepath)
    keys=list(mat_data.keys())
    extracted_data=mat_data[keys[3]]
    
    if savepath==None and openfile==False:        
        np.save(filepath[:-3]+'npy',extracted_data)
    elif savepath!=None and openfile==False:
        np.save(savepath, extracted_data)
    elif openfile==True:
        return np.squeeze(extracted_data)
    
def extract_numbers(string):
    numbers = ''.join(filter(str.isdigit, string))
    return numbers# if numbers else 0

def unique_legend(ax=None, loc='upper right'):
    # Dictionary to keep track of labels and handles
    if ax is None:
        ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    
    # Add a legend with unique labels
    ax.legend(label_dict.values(), label_dict.keys(), loc=loc)



# def endsound():
#     winsound.Beep(300, 150)
#     winsound.Beep(400, 150)
#     winsound.Beep(500, 150)
#     winsound.Beep(700, 500)

# def errorsound():
#     winsound.Beep(500, 150)
#     winsound.Beep(400, 150)
#     winsound.Beep(200, 700)
#     sys.exit()
    
def count_frames(videopath):
    """
    The filelist needs to be separately for cam0 and cam1

    """

    cap = cv2.VideoCapture(videopath)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return frame_count

    
def diff(array,fill=0):
    """
    Inserts a 0 at the start of the array, so that it has the same shape as the input array


    fill : default: 0, changes the value that is used at the first location

    """
    diff=np.diff(array)
    old_shape=np.hstack((fill,diff))
    return old_shape


def load_preprocessed (session, load_pd=False, load_lfp=False):
    """
    loads the data output from preprocess_all.py

    Parameters
    ----------
    path : path to folder with preprocessed data
        
    b_n_t : which output to load (b=behaviour; n=neural; t=tracking)
    
    Returns
    --------
    all the stuff from preprocessing
    frames_dropped: frame difference between frame index and velocity vector (positive numbers mean there is more in nidq frame index)

    """
    if load_pd and load_lfp:
        raise ValueError('the function is not adapted for that')
    paths=pp.get_paths(session)
    path=paths['preprocessed']

    behaviour=pd.read_csv(fr'{path}\behaviour.csv')
    
    
    tracking=np.load(fr'{path}\tracking.npy', allow_pickle=True).item()
    velocity=tracking['velocity']
    locations=tracking['locations']
    node_names=tracking['node_names']
    frame_index_s=tracking['frame_index_s']
    
    frames_dropped=int(float(paths['frame_loss']))
    
    if load_pd:
        ndata_pd=pd.read_csv(fr'{path}\pd_neural_data.csv')
        return behaviour,ndata_pd , velocity, locations, node_names, frame_index_s
    
    
    ndata_dict=np.load(fr'{path}\np_neural_data.npy', allow_pickle=True).item()
    ndata=ndata_dict['n_by_t']
    n_time_index=ndata_dict['time_index']
    n_cluster_index=ndata_dict['cluster_index']
    n_region_index=ndata_dict['region_index']
    n_channel_index=ndata_dict['cluster_channels']
    
    if load_lfp==True:
        lfp_dict=np.load(fr'{path}\lfp.npy', allow_pickle=True).item()
        lfp=lfp_dict['lfp']
        lfp_time=lfp_dict['lfp_time']
        lfp_framerate=lfp_dict['lfp_framerate']
        
        return frames_dropped,behaviour, ndata, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s, lfp, lfp_time, lfp_framerate
    else:

        return frames_dropped, behaviour, ndata, n_time_index, n_cluster_index, n_region_index, n_channel_index, velocity, locations, node_names, frame_index_s



    
    
def mat_struct(file_path, struct_name):
    """
    Read a MATLAB struct from a .mat file.

    Parameters:
    - file_path: Path to the MATLAB file (.mat).
    - struct_name: Name of the struct to read.

    Returns:
    - struct_data: Dictionary containing the data from the MATLAB struct.
    """

    # Open the MATLAB file using h5py
    with h5py.File(file_path, 'r') as mat_file:

        # Check if the specified struct exists in the file
        if struct_name not in mat_file:
            raise ValueError(f"The struct '{struct_name}' does not exist in the MATLAB file.")

        # Extract the struct data
        mat_struct_group = mat_file[struct_name]

        # Convert structured array to dictionary for easier access
        if isinstance(mat_struct_group, h5py.Group):
            struct_data = {name: mat_struct(file_path, struct_name + '/' + name) for name in mat_struct_group.keys()}
        elif isinstance(mat_struct_group, h5py.Dataset):
            struct_data = mat_struct_group[()].tolist()
        else:
            raise ValueError(f"The specified struct '{struct_name}' is not a valid group or dataset in the MATLAB file.")

    return struct_data
    

    
    
def get_paths(session=None, animal=None):
    """
    Gets csv file where paths to datafiles are stored
    works bot if you just specify the animal or just the session
    if you specify nothing, the whole file is being returned
    
    animal/ session names need to be EXACTLY like they are on the csv
   
    Returns
    -------
    paths, either all sessions from all animals, all sessions from one animal,
    or just one session.

    """
    paths=pd.read_csv(r"\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths-Copy.csv")
    if session is None:
        if animal is None:
            return paths
        else:
            return paths[paths['Mouse_ID']==animal]
    else:
        if animal is None:
            sesline=paths[paths['session']==session]
        else:
            sesline=paths[(paths['session']==session) & (paths['Mouse_ID'==animal])]
        return sesline.squeeze().astype(str)

    
    
    
    
def padded_reshape(array, binfactor, axis=0):
    """
    reshapes array by taking binfactor number of entries and putting them in a new
    axis. Works with 1D or 2D arrays
    If reshape is not evenly possible, the end is padded

    Parameters
    ----------
    array : either vector or array

    binfactor : how many values should make up one new entry?
      
    axis : if array 2D, walong which axis should be reshaped?


    """
    binfactor=np.array(binfactor)
    if binfactor!= binfactor.astype(int):
        raise TypeError ('binfactor must be int value')
    binfactor=binfactor.astype(int).item()
    pad_size = np.ceil(array.shape[axis] / binfactor) * binfactor - array.shape[axis]
    pad_size=pad_size.astype(int)
    old_shape=list(array.shape)

    if (len(old_shape)==2) and (axis==0):

        padded = np.pad(array, ((0,pad_size),(0,0)), 'constant', constant_values=0)
        reshaped = padded.reshape(-1,old_shape[1], binfactor)
        print('check if padding is on correct spot')
    elif (len(old_shape)==2) and (axis==1):

        padded = np.pad(array, ((0,0),(0,pad_size)), 'constant', constant_values=0)
        reshaped = padded.reshape(old_shape[0],-1, binfactor)
    elif (len(old_shape)==1):

        padded = np.pad(array, (0,pad_size), 'constant', constant_values=0)
        reshaped = padded.reshape(-1, binfactor)
    else:
        raise ValueError('this function only works in 1d/2d')
        
    
    return reshaped
    
def resample_ndata(ndata, n_time_index, resolution, method='sum'):
    """
    takes ndata that is neurons*time, and resamples it to new resolution

    Parameters
    ----------
    ndata :  neurons*time, with number of spikes per time bins

    n_time_index : same shape[1] as ndata, with the time in s per timebin
        .
    resolution : the new size of timebins, in s
    
    method: whether to take the mean or the sum in resampling step

    Returns
    -------
    resampled_ndata : neurons*time, but in new resolution
        
    new_time_index : index for each timebin, what is the time

    """

    old_resolution=unique_float(np.diff(n_time_index),precision=10)
    
    if len(old_resolution)!= 1:
        raise ValueError('there is something wrong with the time index')
        
    binfactor=resolution/old_resolution
    
    if not binfactor == binfactor.astype(int):
        raise ValueError('new resolution needs to be a multiple of old resoluition')
    
    
    reshaped_ndata=padded_reshape(ndata, binfactor, axis=1)
    if method=='sum':
        resampled_ndata=np.sum(reshaped_ndata, axis=2)
    elif method=='mean':
        resampled_ndata=np.mean(reshaped_ndata, axis=2)
    
    new_time_index=padded_reshape(n_time_index, binfactor)[:,0]
    
    if np.sum(ndata)!= np.sum(resampled_ndata):
        print('the resampling went wrong')
    
    return resampled_ndata, new_time_index 
    
    
#%% Video things
def count_frames(videopath):
    """
    The filelist needs to be separately for cam0 and cam1

    """

    cap = cv2.VideoCapture(videopath)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return frame_count
    
    
def read_frames(video_path, desired_frames):
    """
    reads only the indicated frames from video, not all
    averages out color dimension. This reduces size, but now cv2 plotting doesn't work anymore
    
    Parameters
    ----------
    video_path : path to video
    desired_frames : list with frame numbers
    """
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError("Could not open video.")
    # Set the frame position

    for i,frame in enumerate(desired_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        # Read frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            cap.release()
            raise ValueError("Video was opened, but could not read frame.")
            
        # average out color dimension
        frame = np.mean(frame, axis=-1)
        
        # save frames to array 
        if i==0:
            all_frames=np.zeros((len(desired_frames),*frame.shape))
        all_frames[i]=frame

    cap.release()

    return np.squeeze(all_frames)
    
# def read_frames(video_path, start_frame, end_frame, bw=True):
#     """
#     reads pixel data for each frame

#     Parameters
#     ----------
#     video_path : path to video
#         DESCRIPTION.
#     start_frame : which should be the first frame?
#         .
#     end_frame : whch should be the last frame to be read?
#         .
#     bw : bool, optional
#         should the 3rd dimension of each frame (color) be averaged out?.
#         The default is True.

#     Returns
#     -------
#     frames_np : np array
#         frames*height*width.

#     """
#     # Load video
#     clip = VideoFileClip(video_path)

#     # Get the frame rate (fps) of the video
#     fps = clip.fps

#     # Convert frame number to time
#     start_time = start_frame / fps
#     end_time = end_frame / fps

#     # Get subclip (i.e., frames between start_time and end_time)
#     subclip = clip.subclip(start_time, end_time)

#     # Convert subclip to frames and then to numpy array
#     frames = [frame for frame in subclip.iter_frames()]
#     frames_np = np.array(frames)
    
#     if bw:
#         frames_np=np.mean(frames_np,axis=3)

#     return frames_np

def convert_s(time_in_seconds):
    """
    Converts time in seconds to hh:mm:ss:msmsms format

    Parameters
    ----------
    time_in_seconds : float number in s, can be array of floats

    Returns
    -------
    time_string : string, time in hh:mm:ss:msms format

    """
    if (not isinstance(time_in_seconds, list)) and (not  isinstance(time_in_seconds,np.ndarray)):
        time_in_seconds=[time_in_seconds]
        
    time_strings=[]
    for i, t in enumerate(time_in_seconds):
        # Calculate hours, minutes, seconds and milliseconds
        hours = t // 3600
        minutes = (t % 3600) // 60
        seconds = (t % 60) // 1
        milliseconds = np.floor((t % 1) * 1000)
    
        # Format the time
        time_strings.append( "{:02d}h:{:02d}m:{:02d}s:{:03d}ms".format(int(hours), int(minutes), int(seconds), int(milliseconds)))

    return np.array(time_strings)



def euclidean_distance_old(vector, point, axis):
    """
    calculates distance between vector and time at each point in time
    
    vector: dimension 0 has to be time, can have multiple other dimensions
    point: same dimensions as vector, dimensions that are misssing shou;d be filled with None
    axis: The x,y axis

    """
    
    sq_diff=np.square(vector-point)
    # axis=tuple(range(1,np.ndim(vector))) # this feels very error prone
    distances=np.sqrt(np.nansum(sq_diff, axis )) 
    return distances


def euclidean_distance(vector, point, axis):
    """
    calculates distance between vector and time at each point in time
    
    vector: dimension 0 has to be time, can have multiple other dimensions
    point: same dimensions as vector, dimensions that are misssing shou;d be filled with None
    axis: The x,y axis

    """

    Px=point[0]
    Py=point[1]
    x= np.array(vector[:,0])
    y= np.array(vector[:,1])
    
    dx = (x - Px)
    dy = (y - Py)
#    sq_diff=np.square(vector-point)
    distance = np.sqrt(dx**2 + dy**2)
    return (dx, dy), distance




def convert_csv_time_to_s(timestamps, start_time):
    # Convert start_time to datetime
    start_time = pd.to_datetime(start_time)

    # Convert timestamps to datetime
    timestamps = pd.to_datetime(timestamps)

    # Calculate time difference in seconds
    time_diff_seconds = (timestamps - start_time).total_seconds()

    return time_diff_seconds


def start_stop_array(behaviour, b_name, frame=False, merge_attacks=None, pre_time=7.5):
    """
    take behaviour pd for one behaviour and turns it into matrix
    with rows for one behaviour, and start, stop in the columns
    IMPORTANT: retruns 'turns' as escape START

    Parameters
    ----------
    behaviour : pd with all behaviours, directly from preprocessing
    b_name: the target behaviuour that should be converted e.g. 'escape'
        (NEEDS TO BE STATE BEHAVIOUR)
    frame: if true, framenumber will be returned, otherwise the s of the frame
    merge_attacks: whether attack periods that follow each other shortly should 
        be taken together. if yes, give the minimum distance in s that attacks 
        should be allowed to have
    pre_time: how many s before escape should loom have occured (note this is before running onset, not turn)

    Returns
    -------
    vec: np array with first column being starts, second column being stops. 
            each row is a new instance of the behaviour

    """
    hunting_bs=['approach','pursuit','attack','eat']
    
    if frame==True:
        f='frames'
    elif frame==False:
        f='frames_s'
    #Get escape frames
    

    b_pd=behaviour[behaviour['behaviours']==b_name]
        
        
    
    #If behaviour is point behaviour, just return the frames like this
    if b_pd['start_stop'].iloc[0] =='POINT':
        if not b_pd['start_stop'].nunique() ==1:
            raise ValueError ('something is weird here')
        
        # print('behaviour is point behaviour, returning vector')
        return b_pd[f].to_numpy()

    #Sanity check
    starts=b_pd['start_stop']=='START'
    stops=b_pd['start_stop']=='STOP'    
    

    if len(b_pd) ==0:
        raise ValueError('specified behaviour doesnt exist')
    if (starts.sum()==0) or (stops.sum()==0):
        raise ValueError('specified doesnt seem to be state behaviour')
    if not all(starts.iloc[::2]) and all(stops.iloc[1::2]):
        raise ValueError('starts and stops are not alternating')
    if len(b_pd['behaviours'].unique()) > 1:
        raise ValueError('there is more than one behaviour')
    
    # Make vector
    vec=[]
    acount=0
    for i in range(len(b_pd)):
        if b_pd['start_stop'].iloc[i]!='START':
            continue
        
        #Apppend [start, stop]
        if b_name== 'escape':
            e=b_pd['frames_s'].iloc[i]
            
        
            around_e=behaviour[(behaviour['frames_s']>e-pre_time) & (behaviour['frames_s']<e+.1)]   
            #t=around_e
            t=around_e[around_e['behaviours']=='turn']
            
            #Exclude trials where the loom is too far away
            
            # if not around_e.isin(['loom']).any().any():
            #     print('escape excluded, no loom before')
            #     continue
            
           
            
            # if all(e-t['frames_s']>10):
            #     print('detected turns are more than 10 seconds before escape')
            #     continue
            
            if len(t) >1:
                print('more than one turn before escape. taking the last one')
                t=t.iloc[-1]

                #raise ValueError('more than 1 turn before escape')

            if len(t) == 0: #If there is no turn before escape, use escape start
                t=b_pd.iloc[i]
            
            vec.append([t[f].squeeze(),
                       b_pd[f].iloc[i+1]])
        
           
        
        # elif b_name=='switch':
            
        #     s=b_pd['frames_s'].iloc[i]
            
        #     around_s=behaviour[(behaviour['frames_s']>s-7.5) & (behaviour['frames_s']<s+.1)]   
            
        #     #exclude escapes
        #     if np.sum(np.isin(around_s, hunting_bs))==0:
        #         continue
            
        #     #Exclude trials where the loom is too far away
        #     if not around_s.isin(['loom']).any().any():
        #         continue
            
            
       
            
        #     s=b_pd['frames_s'].iloc[i]
            
        #     around_s=behaviour[(behaviour['frames_s']>s) & (behaviour['frames_s']<s+7)]
            
        #     e_stop=around_s[(around_s['behaviours']=='escape') & 
        #                     (around_s['start_stop']=='STOP')]
            
        #     if len(e_stop)==0:
        #         print('a')
        #         continue #switch not counted, because there was no escape afterwards
            
        #     vec.append([b_pd[f].iloc[i],
        #                 e_stop[f].squeeze()])
        
        # elif (b_name== 'attack') and merge_attacks:
            
        #     # if too  little distance between attack start and previous stop
        #     if ((b_pd[f].iloc[i] - b_pd[f].iloc[i-1])> merge_attacks) or (i==0):            
        #         vec.append([b_pd[f].iloc[i], 
        #                     b_pd[f].iloc[i+1]])

        #     else:
        #         vec[-1][1]=b_pd[f].iloc[i+1]
        #         acount+=1
        else: 
           
                vec.append([b_pd[f].iloc[i], 
                            b_pd[f].iloc[i+1]])

    

    return np.array(vec)


def exclude_switch_trials(event_frames, behaviour , window=10, startstop=0, unit='s', return_mask=False):
    """"excludes those events where there is a switch happening in the window s
    before or after the event
    ___________________
    Parameters:
        event_frames: vector with event times, column 0 is starts, column 1 is stops (as output from hf.start_stop_array)
        behaviour: the behaviour df that is output from preprocessing
        window:  scalar of how long before and after the event a switch should be considered for exclusion of trial
            ALWAYS in s
            
        startstop: if 0, starts are used as eventtime, otherwise stops
        unit: if s: event_frames and window has to be in seconds
              if f: eventrframes and window has to be in frames
        return_mask: should the mask used to exclude switch trials be returned?
    """

    if unit=='f':
        window*=50
        if window> 30*50:
            raise ValueError('window seems a bit large, check that you have it in s')
    
    #Deal with point behaviours
    if len(event_frames.shape)==1:
        event_frames=event_frames[:,None]
    
    
    mask=np.ones(len(event_frames), dtype=bool)
    
    if unit=='s':
        frame_column='frames_s'
    elif unit=='f':
        frame_column='frames'
    else:
        raise ValueError("wrong value for 'unit'")
    
    for i,ev_frame in enumerate( event_frames[:,startstop]): # the 0 is to only take the starts
        
        #get behaviours around event
        windowstart=ev_frame-window
        windowstop=ev_frame+window
        around_event=behaviour[(behaviour[frame_column]>=windowstart) & (behaviour[frame_column]<=windowstop)]
    
        
        #excluse escapes with a switch 5 sec before or after
        if 'switch' in around_event['behaviours'].values:
            
            mask[i]=0
            
    if return_mask:
        return np.squeeze(event_frames[mask,:]), mask
    return np.squeeze(event_frames[mask,:])

def peak_escape_vel(behaviour, velocity, exclude_switches=True):
    """
    get peak velocity for each escape

    Parameters
    ----------
    behaviour : df from preprocesing
    velocity: np vector from preprocessing
    exclude_switches: should switch escapes be excluded?

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    start_stop=start_stop_array(behaviour, 'escape', frame=True)
    if exclude_switches:
        start_stop=exclude_switch_trials(start_stop, behaviour, 5*50, 0, unit='f')
    peak_vels=[]
    for escape in start_stop:
        vel=velocity[escape[0]:escape[1]]
        peak_vels.append(np.nanmax(vel))
    
    return start_stop, np.array(peak_vels)


def baseline_firing(behaviour, n_time_index, ndata, velocity, frame_index_s, window=5, vel_cutoff=[7, 3]):
    """
    Calculates baseline firing during exploration periods. For this, 
    periods are excluded...
    - that are during/ before/ after ANY state/ point behaviour
    - where the animal is not locomoting for some time 
    - 
    Parameters
    ----------
    behaviour : pd table from preprocessing        
    n_time_index :       from preprocessing       
    ndata :              from preprocessing       
    velocity :           from preprocessing
    frame_index_s :      from preprocessing
    
    window : int, optional
        How many s before and after a point/ state behaviour should the neural activity 
        be excluded from baseline. The default is 5s.
    vel_cutoff : list, optional
        first number is below what value the velocity has to drop,
        second number is for what period of time, so that the neural activity is excluded.
        The default is [7 cm/s, 3s].

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    mean_frg_hz : For each neuron, what is the average firing rate in Hz
        during the baseline period.
    base_ind : boolean index, True during baseline period. Shape matches n_time_index


    """
    
    if n_time_index[0] != 0:
        raise ValueError('fix your n_time_index!!!!')
        
    n_sampling=n_time_index[1]
    
    #get baseline
    base_ind=np.ones(ndata.shape[1]).astype(int)
    
    #Cut out periods of behviour
    for b_name in behaviour['behaviours'].unique():
        b=behaviour[behaviour['behaviours']==b_name]
        
        if b['start_stop'].iloc[0]=='START':
            start_stop=start_stop_array(behaviour, b_name, frame=False)
            for start_stop_s in start_stop:
                ind=(n_time_index>(start_stop_s[0]-window)) & (n_time_index<(start_stop_s[1]+window))
                base_ind-=ind
        
        elif b['start_stop'].iloc[0] =='POINT':
            for event_s in b['frames_s']:
                ind=(n_time_index>(event_s-window)) & (n_time_index<(event_s+window))
                base_ind-=ind
        
        else:
            raise ValueError('unexpected content of start_stop, check this')
    
    #Cut out periods of low velocity

    interp_vel=interp(velocity, frame_index_s[:], n_time_index)
    
    num_consecutive_samples = int(vel_cutoff[1] / n_sampling)
    
    # Create a rolling window of the required size and check if all velocities are below 7
    rolling_windows = sliding_window_view(interp_vel, num_consecutive_samples)
    mask = np.all(rolling_windows < vel_cutoff[0], axis=-1)
    mask = np.pad(mask, (num_consecutive_samples - 1, 0), mode='constant', constant_values=False)
    
    base_ind-= mask
    
    
    #Get final index/ mask
    base_ind[base_ind!=1]=0
    base_ind=base_ind.astype(bool)
    
    #calculate baseline firing
    base_sum=np.sum(ndata[:,base_ind], axis=1)
    base_time=np.sum(base_ind)*n_sampling
    mean_frg_hz=base_sum/ base_time
    
    print(f'baseline period covers {convert_s(base_time)[0]}')
    
    return mean_frg_hz, base_ind

def nanmean(array,fill=0, axis=None):
    if np.sum(np.isinf(array))!=0:
        raise ValueError('there are inf values in this array!')
    return np.mean(np.nan_to_num(array, nan=fill), axis)

def interp(vector,t, new_t):
    #Cut out periods of low velocity
    interp_func = interp1d(t, vector, bounds_error=False, fill_value="extrapolate")
    interpolated_vector= interp_func(new_t)
    return interpolated_vector


#%% Regression

def regression(X,Y):

    #Centering X and Y to the mean


    # Calculate covariance matrix of X
    CXX = np.dot(X.T,X) # + sigma * np.identity(np.size(X,1))
    # Calculate covariance matrix of X with Y
    CXY = np.dot(X.T,Y)
    # Regress Y onto X/ calculate the B that gets you best from X to Y
    B_OLS = np.dot(np.linalg.pinv(CXX), CXY)
    # Use the cacluated B to make a prediction on Y
    Y_OLS = np.dot(X,B_OLS)
    # Perform SVD on the predicted Y values
    # _U, _S, V = np.linalg.svd(Y_OLS, full_matrices=False)

    return Y_OLS, B_OLS


 
def OLS_regression(X,Y,nfolds=5, random_state=None, normalise=True):
    """
    

    Parameters
    ----------
    X : Predictor; time*neurons
        .
    Y : Predicted; time*neurons
        .
    nfolds : int; how often cross validation
        Data is divided into nfolds random splits (not consecutive; along time dimension)
        each of the folds is predicted separately by all the other folds together.
        The default is 5.
    random_state : int, optional
        For always having the same random split. The default is None.
    normalise : Bool, optional
        Whether the data should be zscored. The default is True.

    Returns
    -------
    perf_r2 : Vector
        r2 (explained variance) per factor.
    bs : 3d matrix
        regression weights fold*X*Y .

    """
    

    is_gpu = False
    sigma=0
    kfolds = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    taxis = range(X.shape[0])
    perf_r2=[]
    bs=[]

    for ifold, (train, test) in enumerate(kfolds.split(taxis)):
        Xtrain=X[train,:]
        Ytrain=Y[train,:]
        Xtest=X[test,:]
        Ytest=Y[test,:]
        
        if normalise:
            Xtrain=np.nan_to_num(zscore(Xtrain, axis=0))
            Ytrain=np.nan_to_num(zscore(Ytrain, axis=0))
            Xtest=np.nan_to_num(zscore(Xtest, axis=0))
            Ytest=np.nan_to_num(zscore(Ytest, axis=0))

        _, B_OLS = regression(Xtrain,Ytrain)
        # Calculate TEST error by multiplying unseen X values with B 
        Ytestpred = np.dot(Xtest,B_OLS)
        test_err = Ytest - Ytestpred
        mse=np.sum(np.square(test_err),axis=0)
        sst=np.sum(np.square(Ytest-np.mean(Ytest, axis=0)), axis=0)
        perf_r2.append(1-(mse/sst))
        bs.append(B_OLS)

    perf_r2=np.mean(np.array(perf_r2), axis=0)

    return perf_r2, np.array(bs)

def regression_no_kfolds(X,Y):

    #Centering
    mX = np.mean(X, axis = 0,keepdims=True)
    mY = np.mean(Y, axis = 0, keepdims=True)
    stdX=np.std(X,axis=0)
    stdY=np.std(Y, axis=0)
    X = (X - mX)/stdX
    Y = (Y - mY)/stdY


    Y_OLS, B_OLS = regression(X,Y)
    
    #Calculate r2
    test_err = Y_OLS - Y
    mse=np.sum(np.square(test_err),axis=0)
    sst=np.sum(np.square(Y-np.mean(Y, axis=0)), axis=0)
    r2=1-(mse/sst)
    

    return Y_OLS, B_OLS, r2

#%%Permutation
def shift_data(data, num_shift, column_direction=False):
    """
    assumes input to be 2d
    Positive shift values: Data is shifted forward, bottom values are moved on top
    negative shift: Data is shifted backwards, top values are moved to bottom

    Parameters
    ----------
    data : 2d
        .
    num_shift : TYPE
        DESCRIPTION.
    column_direction : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    shifted_data : TYPE
        DESCRIPTION.

    """
    
    if column_direction==True:
        data=data.T
    
    top=data[:-num_shift,:]
    bottom=data[-num_shift:,:]
    shifted_data=np.concatenate((bottom,top),axis=0)
    
    if column_direction==True:
        shifted_data=shifted_data.T
    
    return shifted_data


def rand_base_periods (base_ind, n_time_index, num_base_samples, base_period_length, overlapping=False):
    """
    gets random samples of non-overlapping baseline periods, as a control for behaviours

    Parameters
    ----------
    base_ind : Boolean 1D vector
        indicates periods of baseline; output from hf.baseline_firing() .
    n_time_index : float 1D vector
        indicates time in s for each entry in base_in; output from preprocessing.
    num_base_samples : int
        how many samples/ periods should be drawn?.
    base_period_length : float; in s
        how long should each period be?

    Returns
    -------
    base_start_stop_s : 1D float vector; in s
        in s; each row is a baseline period, column[0] is start, column [1] is stop .

    """
   
    
   
    #NOTE:
   
    """
   
    -You could solve your current problem (shit takes too long) by 
    calculating average necessary distance between two random periods 
    and then taking random sample within that period
    
    -otherwise do as you did before, it doesn't feel so bad
    
    
    """
    
    
    # initial computations
    n_srate=len(n_time_index)/n_time_index[-1]
    num_base_samples=int(num_base_samples)
    base_period_length = int(base_period_length*n_srate)
    
    # test if command is possible
    if (num_base_samples * 2* base_period_length > np.sum(base_ind)) and not overlapping:
        max_samples=np.sum(base_ind)/ (2*base_period_length)
        raise ValueError (f'With this sample length, you can have a maximum of {int(max_samples)} samples')
    
    # Exclude periods that are too close to behaviour
    n=base_period_length*2 # distance to the sides
    padded_base = np.convolve(base_ind, np.ones(n, dtype=int), mode='same') >= n
    
    # get random periods, that are at least  'base_period_length' apart
    diff=0 
    if not overlapping:
        while np.min(diff)<=((base_period_length)/n_srate):
            base_starts=np.sort(choice(n_time_index[padded_base], size=num_base_samples, replace=False ))
            diff=np.diff(base_starts)
    else:
        base_starts=np.sort(choice(n_time_index[padded_base], size=num_base_samples, replace=False ))
        diff=np.diff(base_starts)
        print(f'random samples average diff: {np.median(diff)}')
        
        #wrong unit, needs to be /srate
        
    
    base_stops=base_starts.copy() + (base_period_length/ n_srate)
    base_start_stop_s=np.array([base_starts,base_stops]).T
    return base_start_stop_s
import matplotlib.pyplot as plt
import numpy as np

def convert_loc_to_cm_big_arena(px_coords,py_coords):
    radius_cm = 45 #float(input("Enter the radius of the arena in centimeters: "))
    
    # Calculate the factor to convert pixels to centimeters
    max_pixel_radius = np.max(np.sqrt(px_coords**2 + py_coords**2))
    conversion_factor = radius_cm*2 / max_pixel_radius
    
    return px_coords*conversion_factor,py_coords*conversion_factor

def plot_and_convert(locations,vframerate,radius_cm=45):
    # we assume that N pixels x == N pixels Y
    # Plot every 50th x, y coordinate
    x_coords = locations[::vframerate, 1, 0]
    y_coords = locations[::vframerate, 1, 1]
    
    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, 'o', markersize=2)
    ax.set_title('Click to select shelter location')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    shelter_point = np.array([0, 0])  # Initialize shelter point
    
    # Function to handle click event
    def onclick(event):
        nonlocal shelter_point
        shelter_x, shelter_y = event.xdata, event.ydata
        shelter_point = np.array([shelter_x, shelter_y])
        print(f'Shelter location: ({shelter_x}, {shelter_y}) pixels')
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)  # Close the plot after clicking
    
    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show(block=False)
    
    # Wait for the user to click on the plot
    plt.waitforbuttonpress()
    
    # Ask user for the radius of the arena in centimeters or just hard coded for now
    #radius_cm = 45 #float(input("Enter the radius of the arena in centimeters: "))
    
    # Calculate the factor to convert pixels to centimeters
    max_pixel_radius = np.max(np.sqrt(x_coords**2 + y_coords**2))
    conversion_factor = radius_cm*2 / max_pixel_radius
    
    print(f'Conversion factor: {conversion_factor} cm/pixel')
    
    return conversion_factor, shelter_point
# def plot_and_convert(locations):
#     # Plot every 50th x, y coordinate
#     x_coords = locations[::50, 1, 0]
#     y_coords = locations[::50, 1, 1]
    
#     fig, ax = plt.subplots()
#     ax.plot(x_coords, y_coords, 'o', markersize=2)
#     ax.set_title('Click to select shelter location')
#     ax.set_xlabel('X (pixels)')
#     ax.set_ylabel('Y (pixels)')
    
#     shelter_point = np.array([0, 0])  # Initialize shelter point
    
#     # Function to handle click event
#     def onclick(event):
#         nonlocal shelter_point
#         shelter_x, shelter_y = event.xdata, event.ydata
#         shelter_point = np.array([shelter_x, shelter_y])
#         print(f'Shelter location: ({shelter_x}, {shelter_y}) pixels')
#         fig.canvas.mpl_disconnect(cid)
    
#     # Connect the click event to the handler
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
#     plt.show()
    
#     # Ask user for the radius of the arena in centimeters or just hard coded for now
#     radius_cm = 45 #float(input("Enter the radius of the arena in centimeters: "))
    
#     # Calculate the factor to convert pixels to centimeters
#     max_pixel_radius = np.max(np.sqrt(x_coords**2 + y_coords**2))
#     conversion_factor = radius_cm*2 / max_pixel_radius
    
#     print(f'Conversion factor: {conversion_factor} cm/pixel')
    
#     return conversion_factor, shelter_point

def check_and_convert_variable(x):
    
    if  isinstance(x, np.ndarray):
        return x
    if isinstance(x, str): #x is a string

        if '.' in x:
            x = list(map(float, x.split(',')))
        else:            
            x = list(map(int, x.split(',')))
            
    elif isinstance(x, int):
        if np.isnan(x): #x is NaN
            x=[]            # Do nothing if x is NaN
        else:#x is a float
            x = list(map(int, str(x).split(',')))
    elif isinstance(x, list): # x is a list
        # No conversion needed for lists
        return x
    elif x is None: #x is None
        x=[]
    else: #x is of an unknown type
        x=[]

    return x

def get_shelterdist(locations, node_ind,vframerate ):
    
    #TODO if exits in csv use that
    #if paths['cm/pixel'] AND paths['Shelter location (pixels)']
    pixel2cm,shelterpoint = plot_and_convert(locations,vframerate)
    #Sanity check
    # xmax=np.nanmax(locations[:,:,0])
    # ymax=np.nanmax(locations[:,:,1])
    # if (abs(xmax-1280)>70) or (abs(ymax-1024)>50):
    #     raise ValueError ('check that video layout is really 1280*1024')
    
    # print('double check shelterpoint')
    
    loc=np.squeeze(locations[:,node_ind,:])
    loc=loc*pixel2cm
    shelterpoint=shelterpoint*pixel2cm
    # shelterpoint=np.array([650,910])
    # pixel2cm=88/1005
    vector,distance2shelter=euclidean_distance(loc,shelterpoint,axis=1)    
    return distance2shelter,loc,vector


def sample_neurons(ndata,n_region_index, res, nmb_n,  target_frg_rate, tolerance, return_factors=None):
    """
    Draws semi-random sample from neurons. Final sample has mean: 'target_frg_rate'
    +/- tolerance. Optional: return SVD factors, calculated on sample of neurons

    Parameters
    ----------
    ndata : from preprocessing
    n_region_index : from preprocessing

    res : what is the resolution of the data (first entry of n_time_index)

    nmb_n : int
        how many neurons should be drawn as sample
   
    target_frg_rate : float
        what firing rate shoul dthe samples from each area have
        
    tolerance : float
        What deviation from this avg frg rate is allowed?

    return_factors : int, optional
        If specified, how many factors SVD factors should be returned
        Otherwise, nmb_n neurons are returned. The default is None.


    Returns
    -------
    n_sample:  time* factors / neurons*time matrix with chosen neurons
    
    areass_sample: areanames for neurons
    
    OPTIONAL:
        s2_factors: s^2 values for the returned factors

    """

    avg_hz=np.mean(ndata, axis=1)/res
    areas=np.unique(n_region_index)
    
    n_sample=[]
    areas_sample=[]
    s2_factors=[]
    for aname in areas:
        region_ind=np.isin(n_region_index, aname)
        
        
        #check if area has enough neurons
        if np.sum(region_ind)<nmb_n:
            print(f'{aname} skipped, too few neurons')
            continue
        
        #Choose samples with same avg firing
        pop_avg=target_frg_rate-2*tolerance# just to  have a value outside of alowed range
        i=0
        while np.abs(pop_avg-target_frg_rate)>tolerance:
            n_ind=np.random.choice(np.where(region_ind)[0], nmb_n, replace=False)
            pop_avg=np.mean(avg_hz[n_ind])
            
            #break the loop, if conditions cant be fulfilled by area
            i+=1
            if i>10000:
                raise ValueError(f'sample cant be taken from {aname} with current restrictions')
  
        
        # collect neurons in vector
        if return_factors is None:
            n_sample.append(ndata[n_ind,:])
            areas_sample.append(n_region_index[n_ind])
            
        # OR collect factors
        else:
            n=zscore(ndata[n_ind,:], axis=1).T
            U, s, VT = np.linalg.svd(n, full_matrices=False)
            
            n_sample.append(U[:,:return_factors])
            areas_sample.append([aname]*return_factors)
            s2_factors.append(s[:return_factors]**2)
            
            
            
    if return_factors is None:
        return np.vstack(n_sample), np.hstack(areas_sample)
    else:
        return np.hstack(n_sample), np.hstack(areas_sample), np.hstack(s2_factors)



# # choose neurons with  high firing rate
# region_ind=np.isin(n_region_index, aname)
# sort_ind=np.argsort(avg_hz)
# sort_region_ind=np.isin(ind, np.where(region_ind)[0])
# n_ind=sort_ind[sort_region_ind][:nmb_n]




        
def get_bigB(animal, locs=False, get_sessionstarts=False):
    """
    Collects all the boris annotations from one animal in one long pandas
    and adapts the frame_index_s to match this

    Parameters
    ----------
    animal : str
        EXACT animal name like it is in csv.

    Returns
    -------
    bigB : pd matrix 
        like behaviour.csv, but concatenated from all sessions.
        time values are counted continuously from the start of the first session.
    all_frame_index : concatenated frme_index_s, with continuously increasing time.
.

    """
    from pathlib import Path
    paths=get_paths(None, animal)
    sessionlist=paths['session']
    
    previous_dur_s=0
    previous_dur_f=0
    bigB=[]
    all_locs=[]
    all_frame_index=[]
    all_vel=[]
    sessionstarts=[]
    for ses in sessionlist:  
        
        #Load data
        datapath=paths.loc[paths['session']==ses, 'preprocessed'].values[0]
        if datapath is np.nan:
            print(f'{ses} skipped, no preprocessing path')
            continue
        if os.path.exists(datapath):
            print("The path exists.")
        else:
            print("The path does not exist.")
            continue
        if os.path.exists(fr'{datapath}/behaviour.csv')==False:
            print("The path does not exist.")
            continue
          
        dropped=int(paths.loc[paths['session']==ses, 'frame_loss'].values[0])

         
        behaviour=pd.read_csv(fr'{datapath}/behaviour.csv')
        
        tracking=np.load(fr'{datapath}\tracking.npy', allow_pickle=True).item()
        if dropped != 0:
            frame_index_s=tracking['frame_index_s'][:-dropped]  
            locations=tracking['locations'][:-dropped]
        else:
            frame_index_s=tracking['frame_index_s']
            locations=tracking['locations']
        velocity=tracking['velocity']

        #check if data is aligned
        if len(velocity)!=len(frame_index_s):
            raise ValueError(f'{ses} datastreams are misaligned by {len(velocity)-len(frame_index_s)} frames')
        
        behaviour['frames_s']+=previous_dur_s
        behaviour['frames']+=previous_dur_f

        
        bigB.append(behaviour)
        all_frame_index.append(frame_index_s+previous_dur_s)
        all_vel.append(velocity)
        all_locs.append(locations)
        
        previous_dur_f+=len(frame_index_s)
        previous_dur_s+=frame_index_s[-1]
        sessionstarts.append(previous_dur_s)
        
        
        # plt.figure();plt.title(ses);plt.plot(behaviour['frames_s'].diff())
        # print(f'{ses}\n{frame_index_s[-1]}\n{previous_dur}\n\n')
    print(bigB) 
    bigB=pd.concat(bigB)
    all_frame_index=np.hstack(all_frame_index)
    all_vel=np.hstack(all_vel)
    
    if locs:
        all_locs=np.concatenate(all_locs, axis=0)
        return bigB, all_frame_index, all_vel, all_locs
    if get_sessionstarts:
        return bigB, all_frame_index, all_vel, sessionstarts
    return bigB, all_frame_index, all_vel

def bigB_multiple_animals(animals, get_locs=False):
    """
    Collects all the boris annotations from one animal in one long pandas
    and adapts the frame_index_s to match this
    if multiple animals are given, it concatentes all sessions from all animals

    Parameters
    ----------
    animal : str
        EXACT animal name like it is in csv.

    Returns
    -------
    bigB : pd matrix 
        like behaviour.csv, but concatenated from all sessions.
        time values are counted continuously from the start of the first session.
    all_frame_index : concatenated frme_index_s, with continuously increasing time.
    animal_borders: when do annotations from new animal start

    """
    
    bigB=[]
    all_frame_index=[]
    all_vel=[]
    prev_num_frames=0
    animal_borders=[]
    all_locs=[]
    for i, animal in enumerate(animals):
        
        smallB, frame_index, vel, locs=get_bigB(animal, True)
        
        if i==0:
            all_frame_index.append(frame_index)
            last_t_end=0
            res=0
        else:
            all_frame_index.append(frame_index+last_t_end+res)
        
        smallB['frames_s']+= last_t_end+res
        smallB['frames']+= prev_num_frames
        bigB.append(smallB)
        all_vel.append(vel)
        all_locs.append(locs)
        
        last_t_end=all_frame_index[-1][-1]
        prev_num_frames+= len(frame_index)
        res=frame_index[1]
        animal_borders.append(prev_num_frames)
    if get_locs:
        return pd.concat(bigB), np.hstack(all_frame_index), np.hstack(all_vel), np.concatenate(all_locs, axis=0)
    return pd.concat(bigB), np.hstack(all_frame_index), np.hstack(all_vel), animal_borders

        
        
        

def divide_looms(all_frame_index, bigB, radiance=2):
    """
    divide looms into looms that happen during hunt vs 
    looms that happen outside of hunting

    Parameters
    ----------
    all_frame_index : from hf.get_bigB
        .
    bigB : from hf.get_bigB
        .
    radiance : float, in s
        how long before/ after a hunting event must a loom happen to be considered 'other'. The default is 2.

    Returns
    -------
    otherlooms : list, in s
        when do looms happen that are outside of hunting.

    huntlooms : list, in s
        when do looms happen that are during hunting.

    """
    
    b_names=bigB['behaviours'].to_numpy()
    frames_s=bigB['frames_s'].to_numpy()
    
    hunting_bs=['approach','pursuit','attack', 'eat']
    
    #Get index for when hunting is happening
    hunt_ind=np.zeros_like(all_frame_index)
    for i, b_name in enumerate(hunting_bs):
    
        start_stop=start_stop_array(bigB, b_name, frame=False)
    
            
                
        for b in start_stop:
            hunt_ind+=(all_frame_index>(b[0])) & (all_frame_index<b[1])
    hunt_ind=hunt_ind.astype(bool)
           
    
    #Divide looms in during hunt and other
    huntlooms=[]
    otherlooms=[]
    for loom in frames_s[b_names=='loom']:
        
        around_loom=hunt_ind[(all_frame_index>loom-radiance )& (all_frame_index< loom+radiance)]
    
        if np.sum(around_loom) == 0:
            otherlooms.append(loom)
        elif np.sum(around_loom) > 0:
            huntlooms.append(loom)
        else:
            raise ValueError ('sth is weird here')
    return otherlooms, huntlooms

import itertools

def combos(n, skip_singulars=False):
    # Create a list of variables
    variables = range(n)
    
    # Generate all combinations
    combinations = []
    for r in range(1, n+1):
        combinations.extend(itertools.combinations(variables, r))

    
    if skip_singulars:
        combinations=combinations[n:]
    
    
        
    return combinations

def test_normality(data):
    ad_test = stats.anderson(data)
    print(f"Anderson-Darling test statistic: {ad_test.statistic}")
    for i in range(len(ad_test.critical_values)):
        sl, cv = ad_test.significance_level[i], ad_test.critical_values[i]
        if ad_test.statistic < ad_test.critical_values[i]:
            print(f"Significance level: {sl}, Critical value: {cv}, data looks normal (fail to reject H0)")
        else:
            print(f"Significance level: {sl}, Critical value: {cv}, data does not look normal (reject H0)")
        
    shapiro_test = stats.shapiro(data)
    print(f"\nShapiro test statistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}")
    
    # Kolmogorov-Smirnov test
    ks_test = stats.kstest(data, 'norm')
    print(f"\nKolmogorov-Smirnov test statistic: {ks_test.statistic}, p-value: {ks_test.pvalue}")
    print('\n\n')
    
    

def load_tuning(savepath):
    files=os.listdir(savepath)
    
    all_change=[]
    all_regions=[]
    all_bs=[]
    for file in files:
        data=np.load(f'{savepath}/{file}', allow_pickle=True).item()
        all_change.append(data['region_change'])
        all_regions.append(data['regions'])
        all_bs.append(data['target_bs'])
    
    if not np.all(all_bs==all_bs[0]):
        raise ValueError ("""
                          For this to work all target_bs in all sessions must
                          be the same (at least in the current version of the code)
                          This doesn't mean that you can't have different 
                          behaviours in different sessions, just that the 
                          target_bs you specify in the s_num_permutations.py 
                          script must stay the same
                          """)
    
    unique_bs=np.unique(np.hstack(all_bs))
    unique_regions=np.unique(np.hstack(all_regions))
    
    
    all_region_change=[]
    for i, session in enumerate(all_change):
        region_change=[]
        for region in unique_regions:

                region_change.append(session[all_regions[i]==region])

        all_region_change.append(region_change)
    return all_region_change, unique_regions, all_bs[-1]