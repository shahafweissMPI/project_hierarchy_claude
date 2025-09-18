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

from Utils import *
import torch


def transcode(gpu_id, in_file_path, out_file_path):
    """
    This function demonstrates transcoding of an input video stream. Pixel value of decoded content
    Pixel value of decoded content will be clamped between 0 and 127 before encoding.

                        Parameters:
                            - gpu_id (int): Ordinal of GPU to use [Parameter not in use]
                            - in_file_path (str): Path to
                        file to be decoded
                            - out_file_path (str): Path to output file into which raw frames are stored


                        Returns: - None.

                        Example:
                        >>> transcode(0, "path/to/input/video/file","path/to/output/elementary/bitstream")
        """
    nv_dmx = nvc.CreateDemuxer(filename=in_file_path)
    width = 0
    height = 0
    nv12_frame_size = 0
    input_frame_list = 0
    encoder_created = False
    nvdec = nvc.CreateDecoder(gpuid=0, codec=nv_dmx.GetNvCodecId(), cudacontext=0, cudastream=0, usedevicememory=True)

    with open(out_file_path, "wb") as dec_file:
        frame_cnt = 0

        for packet in nv_dmx:
            for decoded_frame in nvdec.Decode(packet):
                if not encoder_created:
                    width = nvdec.GetWidth()
                    height = nvdec.GetHeight()
                    print("width : , height :", width, height)
                    nv12_frame_size = nvdec.GetFrameSize()
                    input_frame_list = list([AppFrame(width, height, "NV12") for x in range(0, 5)])
                    nvenc = nvc.CreateEncoder(width, height, "NV12", False, codec="h264", preset="P4")
                    encoder_created = True
 
                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                src_tensor = torch.from_dlpack(decoded_frame)
                dst_tensor = torch.clamp(src_tensor, min=0, max=127)
                bitstream = nvenc.Encode(dst_tensor)
                bitstream = bytearray(bitstream)
                dec_file.write(bitstream)

        bitstream = nvenc.EndEncode()
        bitstream = bytearray(bitstream)
        dec_file.write(bitstream)
        print("frame count : ", frame_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This sample application demonstrates transcoding of an input video stream. Pixel value of decoded content "
        "will be "
        "clamped between 0 and 127 before encoding."
    )

    parser.add_argument(
        "-g", "--gpu_id", type=int, help="GPU id, check nvidia-smi. Do not use", )
    parser.add_argument(
        "-i", "--in_file_path", required=True, type=Path, help="Encoded video file (read from)", )
    parser.add_argument(
        "-o", "--out_file_path", required=True, type=Path, help="Encoded video file (write to)", )

    args = parser.parse_args()

    transcode(args.gpu_id,
              args.in_file_path.as_posix(),
              args.out_file_path.as_posix())
