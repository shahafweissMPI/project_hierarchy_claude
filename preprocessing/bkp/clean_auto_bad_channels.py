import os
import re
import spikeinterface.full as si
from pathlib import Path
#from threading import Thread
#import tkinter as tk
from tkinter import filedialog
import ttkbootstrap
import concurrent.futures
import IPython
import subprocess

catGT_path = Path(r'G:\scratch\CatGTWinApp43\CatGT-win').as_posix()

os.chdir(catGT_path)
# Create an empty list to store the commands
commands = []

            
global_job_kwargs = dict(n_jobs=0.5, chunk_duration="1s", progress_bar=True)
si.set_global_job_kwargs(**global_job_kwargs)




def get_stream(input_folder):       
        
        # Use regex to find the imecX part
        match = re.search(r'imec\d+', input_folder)
        if match:
            imec_part = match.group()  # This will be 'imec0', 'imec1', etc., depending on the folder
            stream_id = f'{imec_part}.ap'
        else:
            stream_id = 'imec0.ap'  # Fallback value in case imecX is not found

      #  print(f"stream_id: {stream_id}")
        return stream_id
    

def make_IRC_prb(rec,base_folder):
    
    probe=rec.get_probe()
    dic=probe.to_dict()
    
    geometry = dic["contact_positions"]
    channels = [x + 1 for x in dic["contact_annotations"]["channel_ids"]]
    
    
    # Step 3: Generate MATLAB commands
    matlab_commands ="% Neuropixels probe info from spikeinterface;\n"
    matlab_commands += "channels = {};\n".format(channels)
    matlab_commands += "geometry = {};\n".format((geometry))
    #matlab_commands += "sRateHz = {};\n".format(rec.get_sampling_frequency())
#    matlab_commands += "uV_per_bit = {};\n".format(rec.get_channel_gains()[0])
#    matlab_commands += "nChans = {};\n".format(rec.get_num_channels())
 #   matlab_commands += "vcDataType = {};\n".format(rec.get_dtype())
    

    file_path=Path(base_folder).joinpath(base_folder,'IRC_spk.prb').as_posix()
    # Optionally, print the MATLAB commands to check

    with open(file_path, 'w') as file:
        file.write(matlab_commands)
        
def preprocess_rec(input_folder):
       # rec=preprocess_rec(input_folder,stream_id)
        stream_id= get_stream(input_folder)
        
        if "ap" not in stream_id:
            pass
        print('processing recording: ',input_folder)
        recording= si.read_spikeglx(folder_path=input_folder,stream_id=stream_id)
        make_IRC_prb(recording,input_folder)

        tenth_duration=recording.get_total_duration()*0.1 #take 10% of recording 
        slic_rec=recording.frame_slice(0,recording.sampling_frequency*tenth_duration)#default 30 seconds
        bad_channel_ids, channel_labels = si.detect_bad_channels(slic_rec,outside_channel_threshold=-0.3)
        print('bad_channel_ids:', bad_channel_ids)
        message=f"bad channel detection estimation should be reliable"
        if bad_channel_ids.size > recording.get_num_channels() / 3:
                
            message = (f"""NOISY RECORDING! ".
           Over 1/3 of channels are detected as bad. In the presence of a high.
           number of dead / noisy channels, bad channel detection may fail.
           good channels may be erroneously labeled as dead.""")

        print('number of bad_channel: ',len(bad_channel_ids))
        channels = slic_rec.get_channel_ids()
        if len(channels) != len(channel_labels):
            raise Exception("Number of channels does not match number of channel labels.")
        channels = channels[channel_labels != 'good']
        channel_labels = channel_labels[channel_labels != 'good']

        return {'message': message,'bad_channel_ids': channels, 'channel_labels': channel_labels}
       


def find_imec_folders(parent_folder):
    imec_folders = []

    # Walk through the directory structure
    for dirpath in Path(parent_folder).rglob('*'):
        # Check if the directory name ends with "imec" followed by a number
        if re.match(r'.*imec\d+$', dirpath.name):
            # If it does, add its path to the list
            imec_folders.append(str(dirpath))

    return imec_folders



def parse_bad_channel_id(bad_channel_ids):
    bad_channels_int = []
    for channel_id in bad_channel_ids:
        match = re.search(r'#AP(\d+)', channel_id)
        if match:
            channel_int = int(match.group(1))
            bad_channels_int.append(channel_int)
    return bad_channels_int

def parse_input_folder(input_folder):
    # Split the path into parts
    parts = input_folder.split('/')
    
    # Initialize variables
    dir_path = []
    run = ""
    g = ""
    probe = ""
    
    # Iterate over parts to find the relevant sections
    for part in parts:
        if "_g" in part:
            # Extract run and g
            run, g_part = part.split("_g", 1)
            g = g_part[0]  # Get the character immediately after "_g"
            break  # Stop the loop as we found the run
        else:
            dir_path.append(part)  # Keep building the dir path
    
    # Find the probe value
    for part in parts:
        if "_imec" in part:
            probe_index = part.find("_imec") + len("_imec")
            probe = part[probe_index]  # Get the character immediately after "_imec"
            break  # Stop the loop as we found the probe
    
    # Join dir_path back into a string
    dir_path = '/'.join(dir_path)


    return dir_path,run,g, probe
    

# def write_catGT(input_folder, channels_to_exclude):
#     dir_path,run_folder,g_number, prb_number=parse_input_folder(input_folder)
#     # Parse the input folder path

#     # Convert the list of channels to a string
#     channels_str = ','.join(map(str, channels_to_exclude))

#     # Create the command string
#     command = (f":: {input_folder}\n\n"
#                f"@echo on\n"
#                f"@setlocal enableextensions\n"
#                f"@cd /d \"%~dp0\"\n\n"
#                f"set LOCALARGS=-dir={dir_path} -run={run_folder} -g={g_number} -t=0,10 ^\n"
#                f"-prb_fld -t_miss_ok ^\n"
#                f"-ap -lf -ni -prb={prb_number} ^\n"
#                f"-apfilter=butter,12,300,6000 -loccar_um=40,160 ^\n"
#                f"-chnexcl={{0;{channels_str}}} ^\n"
#                f"-gfix=0,0.1,0.02 -gfix=0.40,0.10,0.02 ^\n" 
#                f"-dest={input_folder}\n\n"
#                f"if [%1]==[] (set ARGS=%LOCALARGS%) else (set ARGS=%*)\n\n"
#                f"%~dp0CatGT %ARGS%\n")
    
#     print(command)
   
#     # Write the command string to a file
#     with open('command.bat', 'w') as file:
#         file.write(command)
#         # Write the command string to a file in the input_folder
#         command_file_path = Path(catGT_path) / 'command.bat'
#         command_file_path=Path(command_file_path).as_posix()
#         with open(command_file_path, 'w') as file:
#             file.write(command)
            

#     try:
#         result = subprocess.run([command_file_path], capture_output=True, text=True, check=False)
#         print("Command output:", result.stdout)
#     except subprocess.CalledProcessError as e:
#         print("Error running command:", e.stderr)
        


def list_for_run_catGT_directly(input_folder, channels_to_exclude, run_from_directory):
    # Parse the input folder path
    dir_path,run_folder,g_number, prb_number=parse_input_folder(input_folder)
    out_path = dir_path
    if dir_path.endswith("raw"):
        out_path = dir_path[:-3] + "preprocessed"

   
    # Convert the list of channels to a string
    channels_str = ','.join(map(str, channels_to_exclude))
    # Prepare the command arguments
    if len(channels_str)>0: #exclude bad channels

        command_args = [
            f"CatGT",
            f"-dir={dir_path}",
            f"-run={run_folder}",
            f"-g={g_number}",
            f"-t=0,100",
            f"-prb_fld",
            f"-out_prb_fld",
            f"-t_miss_ok",
            f"-prb={prb_number}",
            f"-ni",
            f"-lf",
            f"-ap",
            f"-apfilter=butter,12,300,6000",
            f"-loccar_um=40,160",
            f"-chnexcl={{{prb_number};{channels_str}}}",
            f"-gfix=0,0.1,0.02",
            f"-gfix=0.40,0.10,0.02",
            f"-dest={out_path}"
        ]
    else: 
        command_args = [
            f"CatGT",
            f"-dir={dir_path}",
            f"-run={run_folder}",
            f"-g={g_number}",
            f"-t=0,100",
            f"-prb_fld",
            f"-out_prb_fld",
            f"-t_miss_ok",
            f"-prb={prb_number}",
            f"-ni",
            f"-lf",
            f"-ap",
            f"-apfilter=butter,12,300,6000",
            f"-loccar_um=40,160",
            f"-gfix=0,0.1,0.02",
            f"-gfix=0.40,0.10,0.02",
            f"-dest={out_path}"
            ]

    return command_args

def run_catGT_directly(command_args):
    print(command_args)
    # Run the command from a specific directory and capture errors
    result = subprocess.run(command_args, cwd=catGT_path, capture_output=True, text=True)
   
    # Check if there was an error and print it
    if result.returncode != 0:
        print("Error running command:", result.stderr)
    else:
        print("Command output:", result.stdout)

    return result


def process_folder(folder):
        input_folder = Path(str(folder)).as_posix()
        if "catgt" in input_folder:
            print('skipping catgt output')
            pass
        bad_dict = preprocess_rec(input_folder)
       
        # Open the bad_dict.txt file for writing
        filepath=f"{input_folder}/bad_channels.txt"
        with open(filepath, "w") as file:    
            file.write(f"message: {bad_dict['message']}\n")  # Add this line to write the "message" variable to the file                 

            # Iterate over the bad_channel_ids and channel_labels
            for bad_channel_id, channel_label in zip(bad_dict['bad_channel_ids'], bad_dict['channel_labels']):
                # Write the bad_channel_id and channel_label as a line in the file
                file.write(f"{bad_channel_id}\t{channel_label}\n")

        print('bad_channels file written to: ',filepath)
        # Call the parse_bad_channel_id function to get the list of bad channels as integers
        bad_channels_int = parse_bad_channel_id(bad_dict['bad_channel_ids'])
       


        print(input_folder)
        command=list_for_run_catGT_directly(input_folder, bad_channels_int, catGT_path)
        commands.append(command)
        
########################################################################################################

            
style = ttkbootstrap.Style('lumen')
root = style.master

parent_folder = filedialog.askdirectory()  # Open the folder selection dialog

imec_folders = find_imec_folders(parent_folder)


# Iterate over the imec_folders and process each folder
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(process_folder, imec_folders)

print('done detecting all bad channels for all sessions')
# Run the run_catGT function for each command

os.chdir(catGT_path)
print('pre processing recordings with catGT')
try:    #try running catGT in parallel , 2 workers should use ~ 1GB/sec on server with 1Gig connection.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(run_catGT_directly, commands)
except: # fall back to a for loop for cat_GT
    for command in commands:
        run_catGT_directly(command)


print('done processing recordings with catGT')

 

    

    
    
