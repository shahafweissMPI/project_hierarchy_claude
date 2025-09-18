import ffmpeg
import sys
from pprint import pprint # for printing Python dictionaries in a human-readable way

def get_video_metadata(media_file):

# read the audio/video file from the command line arguments
#media_file = r"E:\afm16924\afm16924_240523_1_sharp.mp4"
# uses ffprobe command to extract all possible metadata from the media file
    pprint(ffmpeg.probe(media_file)["streams"])