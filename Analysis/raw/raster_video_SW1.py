import os
import numpy as np
import cv2
cv2.setUseOptimized(True)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

import helperFunctions as hf
import plottingFunctions as pf
import preprocessFunctions as pp
from tqdm import tqdm

nest_asyncio.apply()

cachepath = r"F:\stempel\code\data\vids"
animal = 'afm16924'
session = '240524'
plt.style.use('dark_background')

all_neurons = True
zoom_on_mouse = False
target_regions = ['LPAG', 'DMPAG', 'DLPAG']

windowsize_s = 7
view_window_s = 5

[dropped_frames, behaviour, ndata, n_time_index, n_cluster_index, n_region_index, n_channel_index,
 velocity, locations, node_names, frame_index_s] = hf.load_preprocessed(session, load_lfp=False)

vframerate = len(frame_index_s) / max(frame_index_s)
paths = hf.get_paths(session)

windowsize_f = np.round(windowsize_s * vframerate).astype(int)
view_window_f = np.round(view_window_s * vframerate).astype(int)
node_ind = np.where(node_names == 'f_back')[0][0]

distance2shelter = hf.get_shelterdist(locations, node_ind)
max_dist = max(distance2shelter)
max_vel = max(velocity)

target_neurons_ind = np.where(np.isin(n_region_index, target_regions))[0]
n_ybottom_ind = np.max(target_neurons_ind) + 2
n_ytop_ind = np.min(target_neurons_ind) - 2

if all_neurons:
    fileend = 'all'
    ndot = .1
else:
    fileend = 'zoom'
    ndot = .6

os.chdir(cachepath)

loc_all_looms = np.where([behaviour['behaviours'] == 'loom'])[1]
time_all_looms = behaviour['frames_s'][loc_all_looms]
frame_all_looms = time_all_looms * vframerate

def process_loom(lframe, ltime):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fr'{cachepath}\loom_at_{np.round(ltime / 60, 2)}_{fileend}.mp4',
                          fourcc,
                          25.0,
                          (720, 480))

    around_lframe = np.arange(lframe - windowsize_f, lframe + windowsize_f, 2)
    if np.abs(np.mean(around_lframe - np.round(around_lframe))) > .001:
        raise ValueError(' Something is wrong in the calculation of frames (either here or in the preprocessing script)')
    around_lframe = np.round(around_lframe).astype(int)
    frames = hf.read_frames(paths['video'], desired_frames=around_lframe)

    for i, window_frame in enumerate(around_lframe):
        window_time = window_frame / vframerate

        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(10, 6))

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[:, 1])

        windowtime = frame_index_s[window_frame]
        plt.suptitle(hf.convert_s(windowtime))

        ax0.imshow(frames[i], cmap='binary_r')
        pf.remove_axes(ax0, rem_all=True)

        if zoom_on_mouse:
            x_min, x_max, y_min, y_max, new_centre = pf.make_window(frames[i], locations[window_frame, node_ind, :], 200)
            ax0.set_xlim((x_min, x_max))
            ax0.set_ylim((y_min, y_max))
        else:
            x_min = 650
            y_max = 300

        if (window_frame >= lframe) and (window_frame < lframe + 5 * vframerate):
            pf.show_loom_in_video_clip(window_frame, lframe, vframerate, (x_min, y_max), ax0)

        plot_start = window_frame - view_window_f
        plot_end = window_frame + view_window_f
        x_v = np.linspace(-5, 5, plot_end - plot_start)

        line1, = ax1.plot(x_v, velocity[plot_start:plot_end], color='firebrick', label='velocity')
        ax1.set_ylabel('velocity (cm/s)')
        ax1.set_ylim((0, max_vel))

        ax1_1 = ax1.twinx()
        line2, = ax1_1.plot(x_v, distance2shelter[plot_start:plot_end], color='peru', label='distance to shelter')
        ax1_1.set_ylabel('distance to shelter (cm)')
        ax1_1.set_ylim((0, max_dist))

        ax1.set_xlim(x_v[0], x_v[-1])
        ax1.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)
        ax1.axvline(0, linestyle='--', color='Gray')
        ax1.spines['top'].set_visible(False)
        ax1_1.spines['top'].set_visible(False)
        ax1.legend(handles=[line1, line2])
        ax1.set_xlabel('time (s)')

        plot_start = window_time - view_window_s
        plot_end = window_time + view_window_s

        ycoords = np.linspace(0, len(ndata) * 4, len(ndata)) * -1
        for i, n in enumerate(ndata):
            spikeind = n.astype(bool)
            all_spiketimes = n_time_index[spikeind]
            window_ind = (all_spiketimes > plot_start) & (all_spiketimes < plot_end)
            spiketime = all_spiketimes[window_ind] - window_time
            ax2.scatter(spiketime, np.zeros_like(spiketime) + ycoords[i], color='w', s=ndot)

        pf.region_ticks(n_region_index, ycoords=ycoords, ax=ax2)
        pf.remove_axes(ax2)
        ax2.set_xlabel('time (s)')
        ax2.axvline((lframe - window_frame) / vframerate, color='w', lw=1.5)
        ax2.axvline(0, linestyle='--', color='Gray')
        ax2.set_xlim((-5, 5))

        if not all_neurons:
            bottom = ycoords[n_ybottom_ind]
            top = ycoords[n_ytop_ind]
            ax2.set_ylim((bottom, top))

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

        plt.close()
        buf.close()

    print(f'done with loom at {hf.convert_s(ltime)}')
    cv2.destroyAllWindows()
    out.release()

async def main():
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(4)
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, process_loom, lframe, ltime)
            for lframe, ltime in zip(frame_all_looms, time_all_looms)
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    main_thread = ThreadPoolExecutor().submit(asyncio.run, main())
    main_thread.result()

print("""
      Making this script faster
      -------------------------
      
      -Have separate processes for each loom 
      (that should be really easy actually)
      
      -Divide each video into chunks of 100 frames. Make separate videos for
      each chunk and then concatenate the videos in the end
      
      -decrease figsize/ resolution
      
      - use a different codec?? .avi consumes a lot of space""")