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

import argparse
from pathlib import Path
from enum import Enum
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from pycuda.autoinit import context
import time
import nvtx
import mmap
import json
import multiprocessing
from multiprocessing import Process, Queue
import random
from Utils import AppFramePerf

total_num_frames = 1000


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, h, framesize, config_params):
    """
                This function measures encoding performance in FPS.
    """
    nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)  # create encoder object
    devicedata = cuda.IPCMemoryHandle(h)
    input_gpu_frame = AppFramePerf(width, height, fmt, devicedata, 0)

    begin = time.perf_counter_ns()
    for i in range(total_num_frames):
        rng = nvtx.start_range(color="blue")
        input_gpu_frame.gpuAlloc = int(devicedata) + (i * framesize)
        bitstream = nvenc.Encode(input_gpu_frame)  # encode frame one by one
        nvtx.end_range(rng)

    bitstream = nvenc.EndEncode()  # flush encoder queue
    end = time.perf_counter_ns()
    duration = (end - begin) / (1000 * 1000 * 1000)
    print("Duration : ", duration)
    print("FPS : ", total_num_frames / duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'This function measures encoding performance in FPS.'
    )

    parser.add_argument(
        "--gpu_id", "-g", type=int, default=0, help="GPU id, check nvidia-smi, only for demo, do not use", )
    parser.add_argument(
        "--raw_file_path", "-i", required=True, type=Path, help="Raw NV12 video file (write to)", )
    parser.add_argument(
        "--encoded_file_path", "-o", required=True, type=Path, help="Encoded video file (read from)", )
    parser.add_argument(
        "--size", "-s", required=True, type=str, help="WidthxHeight of raw frame", )
    parser.add_argument(
        "--format", "-if", required=True, type=str, help="Format of input file", )
    parser.add_argument(
        "--codec", "-c", required=True, type=str, help="HEVC, H264, AV1")
    parser.add_argument(
        "--config_file", "-json", type=str, default='', help="path of json config file")
    parser.add_argument(
        "--number","-n",required=True,type=int,help="Number of parallel runs")
    

    args = parser.parse_args()
    size = args.size.split("x")
    config = {}

    if len(args.config_file):
        with open(args.config_file) as jsonFile:
            jsonContent = jsonFile.read()
        config = json.loads(jsonContent)
        config["preset"] = config["preset"].upper()

    args.codec = args.codec.lower()
    args.format = args.format.upper()
    config["codec"] = args.codec

    multiprocessing.set_start_method('spawn')

    processes = []

    with open(args.raw_file_path.as_posix(), "rb") as decFile, open(args.encoded_file_path.as_posix(), "wb") as encFile:
        width = int(size[0])
        height = int(size[1])
        framesize = width * height * 1.5
        if format == "ARGB" or format == "ABGR":
            framesize = width * height * 4
        if format == "YUV444" or format == "P010":
            framesize = width * height * 3
        if format == "YUV444_16BIT":
            framesize = width * height * 3 * 2

        m = mmap.mmap(decFile.fileno(), 0, prot=mmap.PROT_READ)
        hostdata = m.read(m.size())
        devicedata = cuda.mem_alloc(m.size())
        cuda.memcpy_htod(devicedata, hostdata)
        devptrhandle = cuda.mem_get_ipc_handle(devicedata)

        for m in range(0, args.number):
            p = Process(target=encode, args=(args.gpu_id,
                                             args.raw_file_path.as_posix(),
                                             args.encoded_file_path.as_posix(),
                                             int(size[0]),
                                             int(size[1]),
                                             args.format,
                                             devptrhandle,
                                             framesize,
                                             config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
