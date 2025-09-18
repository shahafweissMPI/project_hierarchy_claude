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
import time
import pycuda.autoinit


def decode_parallel(gpu_id, enc_file_path):
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    decoded_frame_size = nv_dmx.FrameSize()

    nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=True)
    nv_dec.setDecoderSessionID(0)
    start = time.perf_counter()

    num_decoded_frames = 0
    for packet in nv_dmx:
        for decoded_frames in nv_dec.Decode(packet):
            num_decoded_frames += 1

    elapsed_time = time.perf_counter() - start
    session_overhead = nvc.PyNvDecoder.getDecoderSessionOverHead(0)
    session_overhead /= 1000.00
    elapsed_time -= session_overhead
    print(f"FPS = ", num_decoded_frames / elapsed_time)


import multiprocessing
from multiprocessing import Process, Queue
import random
import time


def run_parallel(N, gpu_id, enc_file_path):
    """
            This function measures decoding performance in FPS.

            The application creates multiple python processes and runs a different decoding session on each process.
            The number of sessions can be controlled by the CLI option "-n".
            The application supports measuring the decode performance only (keeping decoded frames in device memory as well as measuring the decode performance including transfer of frames to the host memory.

            Parameters: - gpu_id (int): Ordinal of GPU to use [Parameter not in use] - enc_file_path (str): Path
            to file to be decoded
            Returns: - None.

            Example:
            >>> run_parallel(1, 0,"path/to/input/media/file")
            Create 1 decoder in 1 process to measure FPS
        """
    multiprocessing.set_start_method('spawn')
    processes = []

    for m in range(0, N):
        p = Process(target=decode_parallel, args=(gpu_id, enc_file_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This sample application measures decoding performance in FPS."
    )
    parser.add_argument(
        "-g","--gpu_id",type=int,help="GPU id, check nvidia-smi, only for demo, do not use",)
    parser.add_argument(
        "-i","--encoded_file_path",type=Path,help="Encoded video file (read from)",)
    parser.add_argument(
        "--number","-n",required=True,type=int,help="Number of parallel runs")

    args = parser.parse_args()

    run_parallel(args.number,
                 args.gpu_id,
                 args.encoded_file_path.as_posix()
                 )
