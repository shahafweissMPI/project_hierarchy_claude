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
from Utils import AppFrame
from Utils import FetchCPUFrame
from Utils import FetchGPUFrame


total_num_frames = 1000


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params):
    """
    The application demonstrates bitrate change at runtime without the need to reset the encoder session. The
    application reduces the bitrate by half and then restores it to the original value after 100 frames.

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
                    >>> encode(0, "path/to/input/yuv/file","path/to/output/elementary/bitstream",1920,1080,"NV12")
                    Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec and P4 preset
            """
    with open(dec_file_path, "rb") as decFile, open(enc_file_path, "wb") as encFile:
        nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)  # create encoder object
        input_frame_list = list([AppFrame(width, height, fmt) for x in range(1, 5)])
        i = 0
        t = nvenc.GetEncodeReconfigureParams()
        even = False
        for input_gpu_frame in FetchGPUFrame(input_frame_list, FetchCPUFrame(decFile, input_frame_list[0].frameSize),
                                             total_num_frames):
            if i & i % 100==0:  # i == 100, 200, 300, 400

                if even == False:
                    t.averageBitrate = int(t.averageBitrate / 2)
                    even = True
                else:
                    t.averageBitrate = int(t.averageBitrate * 2)
                    even = False
                
                t.vbvBufferSize = int(t.averageBitrate * t.frameRateDen / t.frameRateNum)
                t.vbvInitialDelay = t.vbvBufferSize
                nvenc.Reconfigure(t)
            i = i + 1
            bitstream = nvenc.Encode(input_gpu_frame)  # encode frame one by one
            bitstream = bytearray(bitstream)
            encFile.write(bitstream)

        bitstream = nvenc.EndEncode()  # flush encoder queue
        bitstream = bytearray(bitstream)
        encFile.write(bitstream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'The application demonstrates bitrate change at runtime without the need to reset the encoder session.'
        'The application reduces the bitrate by half and then restores it to the original value after 100 frames.'
    )

    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="Unused variable, do not use", )
    parser.add_argument("-i", "--raw_file_path", type=Path, required=True, help="Raw video file (read from)", )
    parser.add_argument("-o", "--encoded_file_path", type=Path, required=True, help="Encoded video file (write to)", )
    parser.add_argument("-s", "--size", type=str, required=True, help="widthxheight of raw frame. Eg: 1920x1080", )
    parser.add_argument("-if", "--format", type=str, required=True, help="Format of input file", )
    parser.add_argument("-c", "--codec", type=str, required=True, help="HEVC, H264, AV1", )
    parser.add_argument("-json", "--config_file", type=str, default='', help="path of json config file", )

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
           config)
