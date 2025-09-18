#-*- coding: utf-8 -*-
"""
Created on 2025-08-05
Author: Dylan Festa

The goal here is to select specific neurons and compare them to movie and behavior.

Visualize movie of a specific mouse, with the possiblity of navigating it.
At the same time, show raster plot for selected neurons at time of video playback.
Also, visualize behavior intervals.

I will use neurons that appear important for a specific behavior, such as pup run.
Identified through logistic regression analysis.
"""

#%%

import sys, time, os, re
import numpy as np, pandas as pd, xarray as xr
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox)
import pyqtgraph as pg

from pathlib import Path

import h5py

import read_data_light as rdl
from preprocess import SpikeTrains

movies_path = Path.home() / 'Videos800Temp'

assert movies_path.exists(), f"Movies path {movies_path} does not exist."


def get_videos_df(movies_path):
    """
    Get a DataFrame with the video files in the movies_path directory.
    The DataFrame will have columns 'video_path', 'mouse', and 'session'.
    """
    pattern = re.compile(r'^(?P<mouse>afm\d{5})_(?P<session>.+?)(?=\.mp4$)')
    records = []
    for p in movies_path.glob("*.mp4"):          # or rglob if you need recursion
        m = pattern.match(p.name)
        if m:                             # skip anything that doesn’t match
            rec = m.groupdict()           # {'mouse': 'afm16924', 'session': '240522'}
            rec["video_path"] = p.resolve()
            records.append(rec)
    ret = pd.DataFrame(records, columns=["video_path", "mouse", "session"])
    ret.sort_values(by=["mouse", "session"], inplace=True)
    return ret

#%%

the_mouse = 'afm16924'
the_session = '240524'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(the_mouse, the_session)
# unpack the data
behaviour = all_data_dict['behaviour']
spike_times = all_data_dict['spike_times']
time_index = all_data_dict['time_index']
cluster_index = all_data_dict['cluster_index']
region_index = all_data_dict['region_index']
channel_index = all_data_dict['channel_index']

t_data_load_seconds = time.time() - t_data_load_start
t_data_load_minutes = t_data_load_seconds / 60
print(f"Data loaded successfully in {t_data_load_seconds:.2f} seconds ({t_data_load_minutes:.2f} minutes).")

t_start_all = 0.0
t_stop_all = time_index[-1]


spiketrains=SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')

n_units = spiketrains.n_units
print(f"Number of PAG units: {n_units}.")

#%% Select only pup_run important neurons
# manually, for now
important_neurons = [344,393,394,579,673,688,693]

spiketrains = spiketrains.filter_by_units(important_neurons)

print(f"Number of important PAG units: {spiketrains.n_units}.") 

#%%
df_videos = get_videos_df(movies_path)

the_video_file = df_videos[(df_videos['mouse'] == the_mouse) & (df_videos['session'] == the_session)]['video_path'].values[0]
print(f"Video file: {the_video_file}")
if not Path(the_video_file).exists():
    raise FileNotFoundError(f"Video file {the_video_file} does not exist.")

#%%
# behavior time intervals selection
behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(the_mouse,the_session,behaviour)
# keep only behaviours that occur at least 15 times
behaviour_timestamps_df = behaviour_timestamps_df[behaviour_timestamps_df['n_trials'] >= 15]
# now generate a dictionary
dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# remove 'loom'
dict_behavior_label_to_index.pop('loom', None)
# add 'none'
dict_behavior_label_to_index['none'] = -1
# pup grab and pup retrieve should have same label
dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']

#%%
# select only start_stop_times from pup_run
start_stop_times = behaviour_timestamps_df[behaviour_timestamps_df['behaviour'] == 'pup_run'].start_stop_times.values[0]

#%%

# some parameters here
plot_dist_time_window = 10.0  # seconds around the current time to plot
plot_dist_half_window = plot_dist_time_window / 2.0  # half window for plotting


class VideoPlayerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualize movie and raster plot")

        # Video player setup
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(100)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        # Set video file
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(the_video_file))))

        # Play/Pause button
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.player.positionChanged.connect(self.update_slider_and_plot)
        self.player.durationChanged.connect(self.on_duration_changed)

        # Behavior plot (PyQtGraph)
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setMinimumHeight(100)
        self.behavior_plot.setBackground("w")
        self.behavior_plot.setMouseEnabled(x=False, y=False)

        # Raster plot (PyQtGraph)
        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setMinimumHeight(180)
        self.raster_plot.setBackground("w")
        self.raster_plot.setMouseEnabled(x=False, y=False)

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)

        main = QVBoxLayout(self)
        main.addWidget(self.video_widget, stretch=5)
        main.addWidget(self.behavior_plot, stretch=1)
        main.addWidget(self.raster_plot, stretch=2)
        main.addLayout(controls)

        # Data holders
        self.times = None
        self.dist_shelter_vect = None

        # Initial plot
        self.update_plot(plot_dist_time_window*2.0)


    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.playBtn.setText("▶")
        else:
            self.player.play()
            self.playBtn.setText("⏸")

    def update_slider_and_plot(self, ms):
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)
        # Update plots below video
        t = ms / 1000.0
        self.update_plot(t)

    def on_slider_moved(self, ms):
        # Set video position in ms
        self.player.setPosition(ms)

    def on_duration_changed(self, duration):
        # Set slider maximum to video duration in ms
        self.slider.setMaximum(duration)

    def update_plot(self, t):
        # Center plots at time t, window = plot_dist_time_window
        t0 = t - plot_dist_time_window / 2
        t1 = t + plot_dist_time_window / 2

        # --- Behavior plot ---
        self.behavior_plot.clear()
        self.behavior_plot.setXRange(-plot_dist_half_window, plot_dist_half_window)
        self.behavior_plot.setYRange(0, 1)
        # Draw behavior intervals in window
        for start, stop in start_stop_times:
            _start_rel = start - t
            _stop_rel = stop - t
            # if stop < t0 or start > t1:
            #     continue
            xs = [_start_rel, _stop_rel, _stop_rel, _start_rel]
            ys = [0, 0, 1, 1]
            fill_brush = pg.mkBrush(100, 100, 255, 50)
            self.behavior_plot.plot(xs + [xs[0]], ys + [ys[0]], pen=None, brush=fill_brush, fillLevel=0)
        # Add vertical line at t
        vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))
        self.behavior_plot.addItem(vline)
        # Add time label in MM:SS format at x=3, y=0.5
        minutes = int(t // 60)
        seconds = int(t % 60)
        time_label = f"t = {minutes:02d}:{seconds:02d}"
        text_item = pg.TextItem(text=time_label, anchor=(0, 0.5), color='k')
        text_item.setPos(3, 0.5)
        self.behavior_plot.addItem(text_item)

        # --- Raster plot ---
        self.raster_plot.clear()
        self.raster_plot.setXRange(-plot_dist_half_window, plot_dist_half_window)
        self.raster_plot.setYRange(0, spiketrains.n_units + 1)
        # Get raster data in window
        _t_start_spikes = np.maximum(0.1, t0)
        _t_stop_spikes = np.minimum(np.maximum(0.2, t1), t_stop_all)
        _t_offset_spikes = _t_start_spikes + (_t_stop_spikes - _t_start_spikes) / 2
        # Get line segments for raster plot
        xplot, yplot = spiketrains.get_line_segments_xynans(
            t_start=_t_start_spikes,
            t_stop=_t_stop_spikes, time_offset=_t_offset_spikes)
        pen = pg.mkPen(color='k', width=1)
        self.raster_plot.plot(xplot, yplot, pen=pen)
        vline2 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))
        self.raster_plot.addItem(vline2)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoPlayerWidget()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())

#%%