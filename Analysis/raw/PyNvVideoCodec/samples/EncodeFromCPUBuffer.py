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

import PyNvVideoCodec as nvc
import numpy as np
import json
import argparse
from pathlib import Path

total_num_frames = 100


def GetFrameSize(width, height, surface_format):
    frame_size = int(width * height * 3 / 2)
    if surface_format == "ARGB" or surface_format == "ABGR":
        frame_size = width * height * 4
    if surface_format == "YUV444":
        frame_size = width * height * 3
    if surface_format == "YUV420":
        frame_size = int(width * height * 3 / 2)
    if surface_format == "P010":
        frame_size = int(width * height * 3 / 2 * 2)
    if surface_format == "YUV444_16BIT":
        frame_size = int(width * height * 3 * 2)
    return frame_size


def FetchCPUFrame(dec_file, frame_size):
    yield np.fromfile(dec_file, np.uint8, count=frame_size)


def encode(gpuID, dec_file_path, enc_file_path, width, height, fmt, use_cpu_memory, config_params):
    """
                    This function illustrates encoding of frames using host memory buffers as input.

                    The application reads the image data from file subsequently copies the CUDA buffers and submits
                    them to NVENC hardware for encoding as part of Encode() function. Video memory buffer is allocated
                    by the application to get the NVENC hardware output. The application copies the NVENC output
                    from video memory buffer to host memory buffer in order to dump to a file, but this is not needed
                    if application choose to use it in some other way.

                    Parameters:
                        - gpu_id (int): Ordinal of GPU to use [Parameter not in use]
                        - dec_file_path (str): Path to
                    file to be decoded
                        - enc_file_path (str): Path to output file into which raw frames are stored
                        - width (int): width of encoded frame
                        - height (int): height of encoded frame
                        - fmt (str) : surface format string in uppercase, for e.g. NV12
                        - config_params(key value pairs) : key value pairs providing fine-grained control on encoding

                    Returns: - None.

                    Example:
                    >>> encode(0, "path/to/input/yuv/file","path/to/output/elementary/bitstream",1920,1080,"NV12", 1)
                    Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec and P4 preset
            """
    frame_size = GetFrameSize(width, height, fmt)
    with open(dec_file_path, "rb") as dec_file, open(enc_file_path, "wb") as enc_file:
        nvenc = nvc.CreateEncoder(width, height, fmt, use_cpu_memory, **config_params)  # create encoder object
        for i in range(total_num_frames):
            chunk = np.fromfile(dec_file, np.uint8, count=frame_size)
            if chunk.size != 0:
                bitstream = nvenc.Encode(chunk)  # encode frame one by one
                bitstream = bytearray(bitstream)
                enc_file.write(bitstream)
        bitstream = nvenc.EndEncode()  # flush encoder queue
        bitstream = bytearray(bitstream)
        enc_file.write(bitstream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'This Sample Application illustrates encoding of frames using host memory buffers as input.'
    )

    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="Unused variable, do not use", )
    parser.add_argument("-i", "--raw_file_path", type=Path, required=True, help="Raw video file (read from)", )
    parser.add_argument("-o", "--encoded_file_path", type=Path, required=True, help="Encoded video file (write to)", )
    parser.add_argument("-s", "--size", type=str, required=True, help="widthxheight of raw frame. Eg: 1920x1080", )
    parser.add_argument("-if", "--format", type=str, required=True, help="Format of input file", )
    parser.add_argument("-c", "--codec", type=str, required=True, help="HEVC, H264, AV1", )
    parser.add_argument("-json", "--config_file", type=str, default='', help="path of json config file", )
    parser.add_argument("-cb", "--use_cpu_memory", required=True, type=int,
                        help="encode accepts CPU buffer directly else accepts CAI or DLPack", )
    

    args = parser.parse_args()
    config = {}

    if len(args.config_file):
        with open(args.config_file) as jsonFile:
            jsonContent = jsonFile.read()
        config = json.loads(jsonContent)
        config["preset"] = config["preset"].upper()

    args.codec = args.codec.lower()
    args.format = args.format.upper()
    config["codec"] = args.codec
    size = args.size.split("x")

    encode(args.gpu_id,
           args.raw_file_path.as_posix(),
           args.encoded_file_path.as_posix(),
           int(size[0]),
           int(size[1]),
           args.format,
           args.use_cpu_memory,
           config)
