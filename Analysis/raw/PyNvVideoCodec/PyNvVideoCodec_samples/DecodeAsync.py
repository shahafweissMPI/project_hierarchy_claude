# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from struct import pack
import sys
import os
import argparse
from pathlib import Path
from enum import Enum
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import numpy as np
from pycuda.compiler import SourceModule


def decode(gpu_id, enc_file_path, dec_file_path, enable_async_allocations):
    """
    Function to demonstrate how to decode media file into output surfaces allocated on non default cuda stream.

                This function will read a media file and split it into chunks of data (packets).
                A Packet contains elementary bitstream belonging to one frame and conforms to annex.b standard.
                Packet is sent to decoder for parsing and hardware accelerated decoding. Decoder returns list of raw YUV
                frames which can be iterated upon.

                Parameters: - gpu_id (int): Ordinal of GPU to use [Parameter not in use] - enc_file_path (str): Path
                to file to be decoded - enc_file_path (str): Path to output file into which raw frames are stored -
                enable_async_allocations (int): if set to 1 output surface is allocated on stream else its allocated
                on device else its Host memory Returns: - None.

                Example:
                >>> decode(0, "path/to/input/media/file","path/to/output/yuv", 1)
                Function to decode media file and write raw frames into an output file.
        """
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_ctx = cuda_device.retain_primary_context()
    print(f'Context created on device: {cuda_device.name()}')
    cuda_ctx.push()
    cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
    cuda_stream_app = cuda.Stream()

    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    decoded_frame_size = 0
    raw_frame = None
    seq_triggered = False
    nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=cuda_ctx.handle,
                               cudastream=cuda_stream_nv_dec.handle,
                               usedevicememory=True,
                               enableasyncallocations=enable_async_allocations)

    with open(dec_file_path, "wb") as decFile:
        for packet in nv_dmx:
            for decoded_frame in nv_dec.Decode(packet):
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    raw_frame = cuda.pagelocked_empty(shape=decoded_frame_size, dtype=np.uint8, order='C',mem_flags=0)  # for stream aware allocations, we need to create page locked host
                    # memory
                    seq_triggered = True

                luma_base_addr = decoded_frame.GetPtrToPlane(0)  # when async allocations are enabled, we create
                # allocations which are stream aware and do not depend on any context, since allocations happen on a
                # 'cuda_stream_nv_dec' whereas application wants to access the allocation on 'cuda_stream_app',
                # we record an event on 'cuda_stream_nv_dec' and application has to wait 'cuda_stream_nv_dec',
                # this wait operation mostly does not introduce CPU wait thus helping in pipelining work with other
                # frameworks
                nv_dec.WaitOnCUStream(cuda_stream_app.handle)
                cuda.memcpy_dtoh_async(raw_frame, luma_base_addr, cuda_stream_app)
                cuda_stream_app.synchronize()
                bits = bytearray(raw_frame)
                decFile.write(bits)

    cuda_ctx.pop()
    print('Context removed.\nEnd of decode session')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This sample application demonstrates how to decode media file into output surfaces allocated on non default "
        "cuda stream."
    )
    parser.add_argument(
        "-g", "--gpu_id", type=int, help="GPU id, check nvidia-smi, only for demo, do not use", )
    parser.add_argument(
        "-i", "--encoded_file_path", required=True, type=Path, help="Encoded video file (read from)", )
    parser.add_argument(
        "-o", "--raw_file_path", required=True, type=Path, help="Raw NV12 video file (write to)", )
    parser.add_argument(
        "-e", "--enable_async_alloc", action="store_true",
        help="Make all allocations within Decoder as asynchronous, Application needs to provide stream else decoder "
             "would create its own")

    args = parser.parse_args()

    decode(
        args.gpu_id, args.encoded_file_path.as_posix(), args.raw_file_path.as_posix(), args.enable_async_alloc)
