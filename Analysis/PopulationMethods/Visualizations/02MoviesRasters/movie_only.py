#%%
import sys, numpy as np, pathlib
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox)


from pathlib import Path

import pyqtgraph as pg
import pandas as pd
import re
import os
import h5py

import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

movies_path = Path.home() / 'Videos800Temp'

assert movies_path.exists(), f"Movies path {movies_path} does not exist."

#%%

# read_dict = rdl.load_preprocessed_dict(mouse_test, session_test)
# #%%

# dist_shelter = read_dict['distance_from_shelter']
# dist_shelter_vect = dist_shelter[:,3]
# times = read_dict['frame_index_s'] 


def get_time_and_distance_from_shelter(mouse, session):
    read_dict = rdl.load_preprocessed_dict(mouse, session)
    dist_shelter = read_dict['distance_from_shelter']
    dist_shelter_vect = dist_shelter[:,3]
    times = read_dict['frame_index_s']
    # reduce to minimum size of the two vectors
    min_length = min(len(times), len(dist_shelter_vect))
    if len(times) > min_length:
        times = times[:min_length]
    if len(dist_shelter_vect) > min_length:
        dist_shelter_vect = dist_shelter_vect[:min_length]
    return times, dist_shelter_vect


def cut_curve_around_t0(times,xs,t0,*, time_window=3.0):
    t_start = t0 - time_window
    t_end = t0 + time_window
    # find index of t_start and t_end, consider times as sorted
    idx_start = np.searchsorted(times, t_start)
    idx_end = np.searchsorted(times, t_end)
    # cut the arrays
    times_cut = times[idx_start:idx_end]
    xs_cut = xs[idx_start:idx_end]
    return times_cut, xs_cut




# mouse_test,session_test = "afm16924","240525"
# paths_test = rdl.get_paths(mouse_test, session_test)
# read_dict_test = rdl.load_preprocessed_dict(session_test, mouse_test)
# time_test, dist_shelter_test = get_time_and_distance_from_shelter(mouse_test, session_test)

#%%

#for k in read_dict.keys():
#    print(f"{k}: {read_dict[k].shape if isinstance(read_dict[k], np.ndarray) else type(read_dict[k])}")

#%%
# get mice and sessions
pattern = re.compile(r'^(?P<mouse>afm\d{5})_(?P<session>.+?)(?=\.mp4$)')

records = []
for p in movies_path.glob("*.mp4"):          # or rglob if you need recursion
    m = pattern.match(p.name)
    if m:                             # skip anything that doesn’t match
        rec = m.groupdict()           # {'mouse': 'afm16924', 'session': '240522'}
        rec["video_path"] = p.resolve()
        records.append(rec)

df_videos = pd.DataFrame(records, columns=["video_path", "mouse", "session"])
# sort by mouse
df_videos.sort_values(by=["mouse", "session"], inplace=True)


#%%

# some parameters here
plot_dist_time_window = 10.0  # seconds around the current time to plot


mice_sessions = []
for idx, row in df_videos.iterrows():
    mice_sessions.append(f"{row.mouse} / {row.session}")

class VideoSelectorWidget(QWidget):
    def __init__(self, df_videos, mice_sessions):
        super().__init__()
        self.df_videos = df_videos
        self.mice_sessions = mice_sessions
        self.setWindowTitle("Select and Play Movie")

        # ComboBox for selection
        self.combo = QComboBox()
        self.combo.addItems(self.mice_sessions)
        self.combo.currentIndexChanged.connect(self.on_selection_changed)

        # Video player setup
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(100)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)

        # Play/Pause button
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self.update_slider_and_plot)
        self.player.durationChanged.connect(self.slider.setMaximum)

        # PyQtGraph plot
        self.plot = pg.PlotWidget(labels={'left': 'distance from shelter', 'bottom': 'time (s)'})
        self.curve = self.plot.plot(pen='y')
        self.vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot.addItem(self.vline)

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)

        main = QVBoxLayout(self)
        main.addWidget(self.combo)
        main.addWidget(self.video_widget, stretch=5)
        main.addWidget(self.plot, stretch=2)
        main.addLayout(controls)

        # Data holders
        self.times = None
        self.dist_shelter_vect = None

        # Load first video and data if available
        if len(self.df_videos) > 0:
            self.load_video_and_data(0)

    def on_selection_changed(self, idx):
        self.load_video_and_data(idx)

    def load_video_and_data(self, idx):
        video_path = self.df_videos.iloc[idx]["video_path"]
        mouse = self.df_videos.iloc[idx]["mouse"]
        session = self.df_videos.iloc[idx]["session"]
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(video_path))))
        self.player.pause()
        self.playBtn.setText("▶")
        # Load time and distance data only once per selection
        self.times, self.dist_shelter_vect = get_time_and_distance_from_shelter(mouse, session)
        # Initial plot
        self.update_plot(0.0)

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
        # Update plot below video
        t = ms / 1000.0
        self.update_plot(t)

    def update_plot(self, t):
        if self.times is None or self.dist_shelter_vect is None:
            self.curve.setData([], [])
            self.vline.setPos(0)
            return
        times_cut, xs_cut = cut_curve_around_t0(self.times, self.dist_shelter_vect, t, time_window=plot_dist_time_window)
        # x axis: time relative to t0
        rel_times = times_cut - t
        self.curve.setData(rel_times, xs_cut)
        self.vline.setPos(0)
        self.plot.setXRange(-plot_dist_time_window, plot_dist_time_window, padding=0)
        self.plot.setYRange(-5, 80, padding=0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoSelectorWidget(df_videos, mice_sessions)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())
