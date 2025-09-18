# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:57:35 2024

@author: su-weisss
"""
from PyNvVideoCodec_samples.Utils import cast_address_to_1d_bytearray
import PyNvVideoCodec as nvc
import numpy as np
import pycuda.driver as cuda

def decode_to_numpy(gpu_id, enc_file_path, use_device_memory):
    """
    Function to decode a media file and return raw frames as a NumPy array.

    This function reads a media file, decodes it using the GPU (if available), and returns the decoded frames as
    a NumPy array. It no longer writes the frames to a file.

    Parameters:
    - gpu_id (int): Ordinal of GPU to use [Currently not in use]
    - enc_file_path (str): Path to the input media file to be decoded
    - use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface;
                               else it is Host memory.

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

    # Print FPS and pixel format of the stream for convenience
    print("FPS = ", nv_dmx.FrameRate())

    # Demuxer can be iterated, fetch the packet from the demuxer
    for packet in nv_dmx:
        # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
        # size of (decode picture buffer) depends on GPU, for Turing series it's 8
        for decoded_frame in nv_dec.Decode(packet):
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

    return decoded_frames

from PyNvVideoCodec_samples.Utils import cast_address_to_1d_bytearray
import PyNvVideoCodec as nvc
import numpy as np
import pycuda.driver as cuda

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
                

       
       
        
from pathlib import Path
enc_file_path=Path(r"F:\stempel\data\vids\afm16924_240524_pup_retrieval_0_sharp.mp4.mouse_generalized.mp4").as_posix()
decoded_frames_range=[]
start_time = time.time()
decoded_frames_range = decode_to_numpy_range(0, enc_file_path, use_device_memory=1, start_frame=100, num_frames=500)
end_time = time.time()
print(f"moviepy: saving video took {end_time - start_time:.4f} seconds")
