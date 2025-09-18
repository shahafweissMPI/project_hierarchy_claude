from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor,as_completed
from multiprocessing import Pool
from joblib import Parallel, delayed
import preprocessFunctions as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helperFunctions as hf
import plottingFunctions as pf
import os
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import IPython
import time
from decord import VideoReader, cpu
import matplotlib

##plotting params
plt.style.use('dark_background')
import sys
from pathlib import Path
sys.path.append(Path(r'F:\stempel\code\Project_hierarchy\Analysis\raw').as_posix())

import PyNvVideoCodec as nvc
from PyNvVideoCodec_samples import Utils
import numpy as np
import pycuda.driver as cuda

##User defined Parameters
#cachepath=r"F:\stempel\data\vids" #Where should the video be stored? (best an SSD)
cachepath=r"E:\test\vids" #Where should the video be stored? (best an SSD)
animal = 'afm16924'
session = '240522'
###   target_behavior='pup_run'
target_regions = [ 'DpWh','DMPAG', 'DLPAG','LPAG', 'Su3']


plt.style.use('dark_background')

all_neurons=True #whether to plot activity in the whole probe or zoom in on a certain region
zoom_on_mouse=False



windowsize_s=7 #how long before+ after the loom should the video play
view_window_s=5 #with what window should be plotted around the current time?
    
    
#%% load data
[dropped_frames,
 behaviour, 
 ndata, 
 n_time_index, 
 n_cluster_index, 
 n_region_index, 
 n_channel_index,
 velocity, 
 locations, 
 node_names, 
 frame_index_s, 
] = hf.load_preprocessed(session, load_lfp=False)

vframerate=len(frame_index_s)/max(frame_index_s)
paths=pp.get_paths(session)
windowsize_f=np.round(windowsize_s*vframerate).astype(int)
view_window_f=np.round(view_window_s*vframerate).astype(int)
#target node
node_ind=np.where(node_names=='f_back')[0][0]#node to use for tracking


##Distance to shelter
if paths['Cm2Pixel']=='nan' or paths['Shelter_xy']=='nan':
    print('click on shelter location in pop up plot')
    distance2shelter,loc,vector=hf.get_shelterdist(locations, node_ind )
else:
    Cm2Pixel=hf.check_and_convert_variable(paths['Cm2Pixel'])[0]
    distance2shelter=hf.check_and_convert_variable(paths['Shelter_xy'])
    distance2shelter=(distance2shelter[0],distance2shelter[1])
    loc=np.squeeze(locations[:,node_ind,:])
    loc=loc*Cm2Pixel
   
    
    
max_dist=max(distance2shelter)
max_vel=max(velocity)

##Target neurons
target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
n_ybottom_ind = np.max(target_neurons_ind) + 2
n_ytop_ind = np.min(target_neurons_ind) - 2

if all_neurons:
    fileend = 'all'
    ndot = .1  #size of raster dots
else:
    fileend = 'zoom'
    ndot = .6

#%% Plot!!!
#matplotlib.use('Agg')  #Use the Agg backend for non-interactive plotting
os.chdir(cachepath)
from PyNvVideoCodec_samples.Utils import cast_address_to_1d_bytearray, AppFrame, FetchGPUFrame
import PyNvVideoCodec as nvc
import numpy as np
#import pycuda.driver as cuda

from pathlib import Path


def encode_images_to_video(images, enc_file_path, width, height, fmt, config_params):
    """
    Encode an array of images to a .mp4 video file using H264 Nvenc encoder.

    Parameters:
        - images (list of np.array): List of image frames to encode (each image should be a numpy array)
        - enc_file_path (str): Path to output video file (encoded as H.264)
        - width (int): Width of the encoded frame
        - height (int): Height of the encoded frame
        - fmt (str): Surface format string in uppercase (e.g., NV12, RGB)
        - config_params (dict): Key-value pairs providing fine-grained control on encoding

    Returns:
        - None
    """
    with open(enc_file_path, "wb") as encFile:
        nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)  # Create encoder object
        
        # Loop over images and encode them
       
        
        for img in images:
            # Convert image to appropriate format if necessary (e.g., NV12)
            # Assuming images are in numpy format (height, width, channels)
            
            input_frame = AppFrame(width, height, fmt)  # Prepare AppFrame for the current image
            
            # Transfer image to GPU and get the corresponding GPU frame
            input_gpu_frame = FetchGPUFrame([input_frame], img,1)
#            num_frames=?
#            input_gpu_frame = FetchGPUFrame([input_frame],num_frames, img)
            
            # Encode the current GPU frame
            bitstream = nvenc.Encode(input_gpu_frame)
            bitstream = bytearray(bitstream)
            encFile.write(bitstream)  # Write encoded bitstream to file
        
        # Flush the encoder queue
        bitstream = nvenc.EndEncode()
        bitstream = bytearray(bitstream)
        encFile.write(bitstream)

    
   
    
    

def decode_to_numpy_range(gpu_id, enc_file_path, use_device_memory, start_frame, num_frames):
    """
    Function to decode a media file and return a range of raw frames as a NumPy array.

    This function reads a media file, decodes frames between `start_frame` and `end_frame` using the GPU (if available), 
    and returns the decoded frames as a NumPy array. It no longer writes the frames to a file.

    Parameters:
    - gpu_id (int): Ordinal of GPU to use [Currently not in use]
    - enc_file_path (str): Path to the input media file to be decoded
    - use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface;
                               else it is Host memory.
    - start_frame (int): The index of the first frame to decode.
    - end_frame (int): The index of the last frame to decode (exclusive).

    Returns:
    - decoded_frames (List[np.ndarray]): A list of decoded raw frames (YUV format).
    """
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=gpu_id,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=use_device_memory)

    decoded_frames = []
    decoded_frame_size = 0
    raw_frame = None

    seq_triggered = False
    current_frame = 0
    extracted_frames =0
    # Print FPS and pixel format of the stream for convenience
    print("FPS = ", nv_dmx.FrameRate())

    for packet in nv_dmx:
        # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
        # size of (decode picture buffer) depends on GPU, for Turing series it's 8
        for decoded_frame in nv_dec.Decode(packet):
            
            if current_frame < start_frame:
                current_frame += 1
                continue
            else:
                # 'decoded_frame' contains a list of views implementing CUDA array interface
                # For nv12, it would contain 2 views for each plane and two planes would be contiguous
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                    seq_triggered = True
    
                luma_base_addr = decoded_frame.GetPtrToPlane(0)
    
                if use_device_memory:
                    # Copy data from GPU device memory to host memory
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    # Convert the frame into a NumPy array
                    decoded_frames.append(raw_frame.copy())
                else:
                    # If not using device memory, handle the CPU-side memory transfer
                    new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                    decoded_frames.append(np.array(new_array))
                extracted_frames += 1
            if extracted_frames>num_frames:
                return decoded_frames
                
def decode_to_numpy_selected_frames(gpu_id, enc_file_path, use_device_memory, frame_indices):
    """
    Function to decode a media file and return specific frames as a NumPy array.

    This function reads a media file, decodes frames at the indices provided in `frame_indices` using the GPU (if available), 
    and returns the decoded frames as a NumPy array. It no longer writes the frames to a file.

    Parameters:
    - gpu_id (int): Ordinal of GPU to use [Currently not in use]
    - enc_file_path (str): Path to the input media file to be decoded.
    - use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface;
                               else it is Host memory.
    - frame_indices (List[int]): List of specific frame indices to extract from the video.

    Returns:
    - decoded_frames (List[np.ndarray]): A list of decoded raw frames (YUV format).
    """
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=gpu_id,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=use_device_memory)

    decoded_frames = []
    decoded_frame_size = 0
    raw_frame = None

    seq_triggered = False
    current_frame = 0
    extracted_frames = 0
    target_frame_indices = set(frame_indices)  # Convert list to a set for O(1) lookups

    # Print FPS and pixel format of the stream for convenience
    print("FPS = ", nv_dmx.FrameRate())

    for packet in nv_dmx:
        # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
        # size of (decode picture buffer) depends on GPU, for Turing series it's 8
        for decoded_frame in nv_dec.Decode(packet):
            if current_frame not in target_frame_indices:
                current_frame += 1
                continue
            else:
                # 'decoded_frame' contains a list of views implementing CUDA array interface
                # For nv12, it would contain 2 views for each plane and two planes would be contiguous
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                    seq_triggered = True

                luma_base_addr = decoded_frame.GetPtrToPlane(0)

                if use_device_memory:
                    # Copy data from GPU device memory to host memory
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    # Convert the frame into a NumPy array
                    decoded_frames.append(raw_frame.copy())
                else:
                    # If not using device memory, handle the CPU-side memory transfer
                    new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                    decoded_frames.append(np.array(new_array))

                extracted_frames += 1

                # Remove the extracted frame from the set
                target_frame_indices.remove(current_frame)

            if not target_frame_indices:  # Break if all frames have been extracted
                return decoded_frames
            
            current_frame += 1

    return decoded_frames


def decode_PyNvVideoCodec_to_numpy(gpu_id, enc_file_path, use_device_memory, start_frame, end_frame):
    """
    Function to decode a media file and return a range of raw frames as a NumPy array.

    This function reads a media file, decodes frames between `start_frame` and `end_frame` using the GPU (if available), 
    and returns the decoded frames as a NumPy array. It no longer writes the frames to a file.

    Parameters:
    - gpu_id (int): Ordinal of GPU to use [Currently not in use]
    - enc_file_path (str): Path to the input media file to be decoded
    - use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface;
                               else it is Host memory.
    - start_frame (int): The index of the first frame to decode.
    - end_frame (int): The index of the last frame to decode (exclusive).

    Returns:
    - decoded_frames (List[np.ndarray]): A list of decoded raw frames (YUV format).
    """
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=gpu_id,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=use_device_memory)

    decoded_frames = []
    decoded_frame_size = 0
    raw_frame = None

    seq_triggered = False
    current_frame = 0

    # Print FPS and pixel format of the stream for convenience
    print("FPS = ", nv_dmx.FrameRate())

    # Demuxer can be iterated, fetch the packet from the demuxer
    for packet in nv_dmx:
        # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
        for decoded_frame in nv_dec.Decode(packet):
            # Only start storing frames after reaching start_frame
            if current_frame >= start_frame:
                # Stop decoding if we've reached the end frame
                if current_frame >= end_frame:
                    return decoded_frames

                # 'decoded_frame' contains a list of views implementing CUDA array interface
                # For nv12, it would contain 2 views for each plane and two planes would be contiguous
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                    seq_triggered = True

                luma_base_addr = decoded_frame.GetPtrToPlane(0)

                if use_device_memory:
                    # Copy data from GPU device memory to host memory
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    # Convert the frame into a NumPy array
                    decoded_frames.append(raw_frame.copy())
                else:
                    # If not using device memory, handle the CPU-side memory transfer
                    new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                    decoded_frames.append(np.array(new_array))

            current_frame += 1
    print(f"seq_triggered is {seq_triggered}")
    return decoded_frames

# def cast_address_to_1d_bytearray(base_address, size):
#     """
#     Helper function to cast a memory address to a NumPy array.
#     This is only used when use_device_memory is set to 0 (Host memory).
    
#     Parameters:
#     - base_address: The base memory address to cast.
#     - size: The size of the frame in bytes.

#     Returns:
#     - NumPy array containing the raw frame data.
#     """
#     return np.ctypeslib.as_array((ctypes.c_ubyte * size).from_address(base_address)) # ignore any warning about this line. it runs fine

def read_frames_decord(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))  
    frames = vr.get_batch(frame_indices)
    return frames.asnumpy()

def write_video_moviepy(images, output_path, fps=25):
    clip = ImageSequenceClip(list(images), fps=fps)
    clip.write_videofile(output_path, codec='libx264', ffmpeg_params=['-vcodec', 'h264_nvenc'])

    
def save_vid_from_images_array(images,ltime,encoding_type:str='moviepy'):
    print('saving movie file')
    movie_file_name= fr'{cachepath}\{target_behavior}_at_{np.round(ltime/60,2)}_{fileend}.mp4'
   
  
    if encoding_type == 'moviepy':
        start_time = time.time()
        write_video_moviepy(images,movie_file_name)         
        end_time = time.time()
        print(f"moviepy: saving video took {end_time - start_time:.4f} seconds")
    elif encoding_type == 'PyNvVideoCodec':        
        
        config = {
        "codec" : "h264",
        "preset" : "P1",
        "tuning_info" : "high_quality",
        "rc": "vbr",
        "fps"  : 25,
        "gop" : 250,
        "bf" : 3,
        "bitrate" : "10M",
        "maxbitrate" : 0,
        "vbvinit" : 0,
        "vbvbufsize" : 0,
        "initqp" : 32,
        "qmin"         : "0,0,0",
        "qmax"         : "0,0,0",
        "initqp"       : "0,0,0",
        "format" : "ABGR",#"RGB",
        }
        start_time = time.time()
       
        encode_images_to_video(images=images, enc_file_path=Path(movie_file_name).as_posix(), width=1920, height=1200, fmt='ABGR',config_params=config)
        end_time = time.time()
        print(f"PyNvVideoCodec: saving video took {end_time - start_time:.4f} seconds")
   
   
    
      

def process_frame(i, window_frame, lframe, frame):

    #process and plot a single frame
    window_frame = float(window_frame)
    window_time = window_frame / vframerate

    # Make figure
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(20, 12))

    # Add subplots
    ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column
    ax2 = fig.add_subplot(gs[:, 1])  # All rows, second column

    windowtime = frame_index_s[window_frame]
    plt.suptitle(hf.convert_s(windowtime))

    # plot frame + sleap labels
    ax0.imshow(frame, cmap='binary_r')
    pf.remove_axes(ax0, rem_all=True)

    # zoom in on mouse
    if zoom_on_mouse:
        x_min, x_max, y_min, y_max, new_centre = pf.make_window(frame, locations[window_frame, node_ind, :], 200)
        ax0.set_xlim((x_min, x_max))
        ax0.set_ylim((y_min, y_max))
    else:
        x_min = 650
        y_max = 300

    
    if target_behavior=='loom': # show loom
        if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)

    # plot distance and velocity
    plot_start = window_frame - view_window_f  # this is in units of video frames
    plot_end = window_frame + view_window_f
    x_v = np.linspace(-5, 5, plot_end - plot_start)

    # velocity
    line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity')
    ax1.set_ylabel('velocity (cm/s)')
    ax1.set_ylim((0, max_vel))

    # distance to shelter
    ax1_1 = ax1.twinx()
    line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter')
    ax1_1.set_ylabel('distance to shelter (cm)')
    ax1_1.set_ylim((0, max_dist))

    # Add loom line, legend, xlabel, remove top axis
    ax1.set_xlim(x_v[0], x_v[-1])
    ax1.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
    ax1.axvline(0, linestyle='--', color='Gray')  # current time
    ax1.spines['top'].set_visible(False)
    ax1_1.spines['top'].set_visible(False)
    ax1.legend(handles=[line1, line2])
    ax1.set_xlabel('time (s)')

    # plot raster
    plot_start = window_time - view_window_s
    plot_end = window_time + view_window_s

    ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
    for j, n in enumerate(ndata):
        spikeind = n.astype(bool)
        all_spiketimes = n_time_index[spikeind]
        window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
        spiketime = all_spiketimes[window_ind] - window_time
        ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[j], color='w', s=ndot)

    pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
    pf.remove_axes(ax2)
    ax2.set_xlabel('time (s)')
    ax2.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)  # loom
    ax2.axvline(0, linestyle='--', color='Gray')  # current time
    ax2.set_xlim((-5, 5))

    if not all_neurons:  # zoom in to target areas
        bottom = ycoords[n_ybottom_ind]
        top = ycoords[n_ytop_ind]
        ax2.set_ylim((bottom, top))

    # save figure to array
    fig.canvas.draw()  # Force update to the figure
#    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    IPython.embed()
    plt.close()

    return i, image

def process_loom(args):# takes about 400 seconds for a 3 second window
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)  # plot every other frame (25 instead of 60 FPS)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:  # if frame numbers are not close to integers
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    start_frame= around_lframe[0]
    num_of_frames = around_lframe[-1]-around_lframe[0]
    # IPython.embed()
    # start_time = time.time()
    # #frames = decode_to_numpy_range(gpu_id=0, enc_file_path= Path(paths['video']).as_posix(), use_device_memory=1, start_frame=start_frame, num_frames=num_of_frames) # takes 145 seconds with use_device_memory=1
    # frames =  decode_to_numpy_selected_frames(gpu_id=0, enc_file_path=Path(paths['video']).as_posix(), use_device_memory=1, frame_indices=around_lframe) # takes 145 seconds with use_device_memory=1
    # end_time = time.time()   
    # print(f"decode_to_numpy_range took {end_time - start_time:.4f} seconds")
    
    
    start_time = time.time()
    frames = read_frames_decord(paths['video'], around_lframe) #~15-30 seconds runtime
    end_time = time.time()   
    print(f"read_frames_decord took {end_time - start_time:.4f} seconds")
    
  

    images = [None] * len(around_lframe)
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)]
        for future in as_completed(futures):
            i, image = future.result()
            images[i] = image
    end_time = time.time()
    print(f"ThreadPoolExecutor took {end_time - start_time:.4f} seconds")
    return images
    
    



def process_loom_ProcessPoolExecutor(args):
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  

    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")  
        
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        i, image = future.result()
        images[i] = image
         
    end_time = time.time()

    print(f"ProcessPoolExecutor took {end_time - start_time:.4f} seconds")
    
    return images  


def process_frame_wrapper(args):
    return process_frame(*args)

def process_loom_Pool(args):
    
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")  
        
    start_time = time.time()
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_frame_wrapper, [(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)]), total=len(around_lframe)))
    
    for i, image in results:
        images[i] = image
         
    end_time = time.time()

    print(f"Pool took {end_time - start_time:.4f} seconds")
    
    return images



def process_loom_joblib(args):
   
    lframe, ltime, target_behavior = args
    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
#    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
 #       raise ValueError(f" Something is wrong in the calculation of frames (either here or in the preprocessing script), got difference of {np.abs(np.mean(around_lframe - np.round(around_lframe)))}")
    around_lframe = np.round(around_lframe).astype(int)
    print("loading frames")  
       
    frames = read_frames_decord(paths['video'], around_lframe)

    images = [None] * len(around_lframe)
   
    print("plotting frames to figure array")  
        
    start_time = time.time()
    results = Parallel()(delayed(process_frame)(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe))
    
    for i, image in tqdm(results, total=len(results)):
        images[i] = image
         
    end_time = time.time()

    print(f"joblib took {end_time - start_time:.4f} seconds")
    
    return images


import time
import concurrent.futures
from multiprocessing import Pool
from joblib import Parallel, delayed
import time
import concurrent.futures
from multiprocessing import Pool
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Assuming that all other necessary variables such as vframerate, 
# locations, etc., are already defined in your script.

# The `process_frame` function that returns the index and generated image
def process_frame(i, window_frame, lframe, frame):
    # Convert window_frame to float
    window_frame = float(window_frame)
    window_time = window_frame / vframerate
    
    # Create figure with subplots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(20, 12))
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    # (Assumed all required variables and plots are processed here as shown previously)

    # Draw the figure to get the image
    fig.canvas.draw()
    
    # Save the figure as an image in a numpy array (RGB format)
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close the figure after saving the image to free memory
    plt.close()

    # Return the index and the generated image
    return i, image

# Sample data (replace these with actual data)
frame_all_looms = [f'frame_{i}' for i in range(100)]
time_all_looms = [f'time_{i}' for i in range(100)]
around_lframe = [str(i) for i in range(10)]
frames = [f'frame_{i}' for i in range(10)]

# Function to time and execute different parallel strategies
def time_execution(fn, description):
    start_time = time.time()
    images = fn()  # Execute and store returned images
    end_time = time.time()
    print(f'{description}: {end_time - start_time:.4f} seconds')
    return images

# 1. Sequential execution (baseline)
def sequential_execution(args):
    lframe, ltime, target_behavior = args
    images = []
    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        for i, window_frame in enumerate(around_lframe):
            idx, image = process_frame(i, window_frame, lframe, frames[i])
            images.append(image)  # Store the image
    return images

# 2. Parallel using ThreadPoolExecutor
def thread_pool_execution():
    images = [None] * len(around_lframe)

    def thread_worker(lframe, iframe):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_frame, i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)]
            for future in concurrent.futures.as_completed(futures):
                idx, image = future.result()
               # images.append(image)  # Store the image
                images[idx]=image

    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        thread_worker(lframe, iframe)
    
    return images

# 3. Parallel using ProcessPoolExecutor
def process_pool_execution(args):
    
    images_joblib = time_execution(joblib_parallel_execution, "Joblib parallel execution") # 19 / 35 seconds
    images = [None] * len(around_lframe)
    def process_worker(lframe, iframe):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_frame, i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)]
            for future in concurrent.futures.as_completed(futures):
                idx, image = future.result()
               # images.append(image)  # Store the image
                images[idx]=image

    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        process_worker(lframe, iframe)
    
    return images

# 4. Multiprocessing with multiprocessing.Pool
def multiprocessing_pool_execution(args):
    lframe, ltime, target_behavior = args
    images = [None] * len(around_lframe)

    def pool_worker(lframe, iframe):
        with Pool() as pool:
            results = pool.starmap(process_frame, [(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe)])
            for idx, image in results:
                #images.append(image)  # Store the image
                images[idx]=image

    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        pool_worker(lframe, iframe)

    return images

# 5. Parallel execution using Joblib
def joblib_parallel_execution():

   
   
    print("plotting frames to figure array")  
        
    #start_time = time.time()
    # results = Parallel()(delayed(process_frame)(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe))
    
    # for i, image in tqdm(results, total=len(results)):
    #     images[i] = image
         
    # end_time = time.time()

    #print(f"joblib took {end_time - start_time:.4f} seconds")
    images = [None] * len(around_lframe)
   
      
    def joblib_worker(lframe, iframe):
        results = Parallel(n_jobs=-1)(delayed(process_frame)(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe))
        for idx, image in results:
           # images.append(image)  # Store the image
            images[idx]=image
   
   
    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        joblib_worker(lframe, iframe)
    
    return images

# Measure execution times for each method and store images




def vid_by_behav(target_behavior):
    #%%Precompute

    #Windows

    loc_all_looms = np.where([behaviour['behaviours'] == target_behavior])[1]
    time_all_looms = behaviour['frames_s'][loc_all_looms]
    frame_all_looms = time_all_looms * vframerate
    # args = [(lframe, ltime, target_behavior) for lframe, ltime in zip(frame_all_looms, time_all_looms)] # loop through events in target behavior
    # with Pool() as pool:
    #    pool.map(process_loom, args)
    
    for iframe, (lframe, ltime) in enumerate(zip(frame_all_looms, time_all_looms)):
        around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)#take every other frame
       # if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
         #   IPython.embed()
#            raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
        around_lframe = np.round(around_lframe).astype(int)
       
        print("loading frames using decord...")             
        start_time = time.time()        
        frames = read_frames_decord(paths['video'], around_lframe) # 23 seconds for 3 second window
        end_time = time.time()
        print(f'frames loaded in: {end_time - start_time:.4f} seconds')
        args=lframe, ltime, target_behavior        
        IPython.embed()
        image = process_frame(100, around_lframe[100], lframe, frames[100])
        start_time = time.time()
        
        results = Parallel()(delayed(process_frame)(i, window_frame, lframe, frames[i]) for i, window_frame in enumerate(around_lframe))
        images = [None] * len(around_lframe)
        for i, image in tqdm(results, total=len(results)):
            images[i] = image
        end_time = time.time()
        print(f"joblib took {end_time - start_time:.4f} seconds")
        
        #images_joblib = time_execution(joblib_parallel_execution, "Joblib parallel execution") # 19 / 35 seconds
      
        save_vid_from_images_array(images,ltime,encoding_type='moviepy') # save as video
     #   save_vid_from_images_array(images_joblib,ltime,encoding_type='PyNvVideoCodec') # save as video
       
       
        print(f" lframe {lframe} ltime {ltime}")
       # IPython.embed()
       
       



        # print("Comparing runtimes for parallel processing:")
        
        # images_sequential = time_execution(sequential_execution, "Sequential execution")#~136 seconds for 3 second window
        # images_thread = time_execution(thread_pool_execution, "ThreadPoolExecutor execution") #139.2940 seconds
        # images_process = time_execution(process_pool_execution, "ProcessPoolExecutor execution")
        # images_multiprocessing = time_execution(multiprocessing_pool_execution, "Multiprocessing Pool execution")
        # images_joblib = time_execution(joblib_parallel_execution, "Joblib parallel execution") # 19 / 35 seconds

        # # Example of accessing the stored images
        # print(f"Number of images processed in sequential execution: {len(images_sequential)}")
        # print(f"Number of images processed in thread pool execution: {len(images_thread)}")
        # print(f"Number of images processed in process pool execution: {len(images_process)}")
        # print(f"Number of images processed in multiprocessing pool execution: {len(images_multiprocessing)}")
        # print(f"Number of images processed in joblib parallel execution: {len(images_joblib)}")

        # IPython.embed()
        # images = process_loom([lframe, ltime, target_behavior]) #get array of frames #takes 402.2655 seconds on 44 threads
        # images = process_loom_ProcessPoolExecutor([lframe, ltime, target_behavior]) #get array of frames
        # images = process_loom_Pool([lframe, ltime, target_behavior]) #get array of frames
        # images = process_loom_joblib([lframe, ltime, target_behavior]) #get array of frames
        
        
        # save_vid_from_images_array(images,ltime) # save as video
    
    
#IPython.embed()
unique_behaviours = behaviour.behaviours.unique()
#unique_behaviours = [ 'escape'] #debug
for target_behavior in unique_behaviours:
    print(f"target_behavior: {target_behavior}")
    vid_by_behav(target_behavior)