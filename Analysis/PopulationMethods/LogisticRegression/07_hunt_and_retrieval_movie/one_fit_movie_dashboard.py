#-*- coding: utf-8 -*-
"""
Created on 2025-08-26
Author: Dylan Festa

Plot the outcome of 01_one_fold_fit.py, with movie.

So we have: movie, behaviors, raster of most important neurons and "prediction".

Note that is not really a prediction because the video covers both test set and train set,
and at this stage I did not separate the two very rigorously.

The real goal here is to take a good look at the raster, and try to figure out 
what those neurons are doing.
"""

#%%

import sys, time, os, re
import numpy as np, pandas as pd, xarray as xr
from PyQt5.QtCore import Qt, QUrl, QEvent, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox, QLabel, QFrame)  # added QLabel, QFrame
import pyqtgraph as pg

from pathlib import Path

import pickle

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
the_session = '240527'

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


#%% videos
df_videos = get_videos_df(movies_path)

the_video_file = df_videos[(df_videos['mouse'] == the_mouse) & (df_videos['session'] == the_session)]['video_path'].values[0]
print(f"Video file: {the_video_file}")
if not Path(the_video_file).exists():
    raise FileNotFoundError(f"Video file {the_video_file} does not exist.")

#%%  Now load fit data, unpicking it
fit_load_path = Path(__file__).parent / f"{the_mouse}_{the_session}_one_fit.pkl"
if not fit_load_path.exists():
    raise FileNotFoundError(f"Fit data file {fit_load_path} does not exist.")

with open(fit_load_path, 'rb') as f:
    fit_data = pickle.load(f)

fit_predictions_xr = fit_data['predictions']
df_ranking = fit_data['df_ranking'] 
behaviour_timestamps_df = fit_data['behaviour_timestamps']
dict_behavior_label_to_index = fit_data['index_to_behaviour']
behaviour_representation_df = fit_data['behaviour_representation_df']
latest_train_time = fit_data['latest_train_time']

dict_classindex = fit_predictions_xr.attrs['class_labels_dict']


#%% select 30 neurons with highest SUMMED absolute rank val in df_ranking 
df_ranking['abs_rank'] = df_ranking['rank'].abs()

# now sum abs_rank grouping by neuron
df_ranking_sum = df_ranking.groupby('unit')['abs_rank'].sum().reset_index()

important_neurons = df_ranking_sum.nlargest(30, 'abs_rank')['unit'].values

spiketrains = spiketrains.filter_by_units(important_neurons).generate_sorted_by_rate()

print(f"Number of important PAG units: {spiketrains.n_units}.") 


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
        self.player.positionChanged.connect(self.update_slider_and_all_plots)
        self.player.durationChanged.connect(self.on_duration_changed)

        # Store train cutoff time
        self.latest_train_time = latest_train_time

        # Overlay (color after training time)
        self.post_train_overlay = QWidget(self)
        self.post_train_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.post_train_overlay.setStyleSheet("background-color: rgba(255,0,0,90); border: none;")
        self.post_train_overlay.hide()

        # Track slider geometry changes
        self.slider.installEventFilter(self)
        QTimer.singleShot(0, self.update_slider_overlays)

        # Behavior plot (PyQtGraph)
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setMinimumHeight(100)
        self.behavior_plot.setBackground("w")
        self.behavior_plot.setMouseEnabled(x=False, y=False)

        # Prepare sorted behaviours once
        self.behaviour_rows = behaviour_representation_df.sort_values('plot_index').reset_index(drop=True)

        # Build legend widget (left side)
        self.legend_widget = QWidget()
        _leg_layout = QVBoxLayout(self.legend_widget)
        _leg_layout.setContentsMargins(2, 2, 2, 2)
        _leg_layout.setSpacing(2)
        for _row in self.behaviour_rows.itertuples():
            # color consistent with plotting
            color_id = _row.plot_index
            n_behaviours = self.behaviour_rows.shape[0]
            base_color = pg.intColor(color_id, hues=n_behaviours, values=1, maxValue=255)
            base_color_a = pg.mkColor(base_color)
            base_color_a.setAlpha(160)
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            patch = QFrame()
            patch.setFixedSize(14, 14)
            patch.setStyleSheet(f"background-color: rgba({base_color_a.red()},{base_color_a.green()},{base_color_a.blue()},{base_color_a.alpha()});"
                                "border: 1px solid #444;")
            lab = QLabel(str(_row.label))
            lab.setStyleSheet("font-size:10px;")
            row_l.addWidget(patch, 0)
            row_l.addWidget(lab, 1)
            _leg_layout.addWidget(row_w)
        _leg_layout.addStretch(1)

        # Wrap legend + behavior plot together
        behaviour_container = QWidget()
        _beh_layout = QHBoxLayout(behaviour_container)
        _beh_layout.setContentsMargins(0, 0, 0, 0)
        _beh_layout.setSpacing(4)
        _beh_layout.addWidget(self.legend_widget, 0)
        _beh_layout.addWidget(self.behavior_plot, 1)

        # Raster plot (PyQtGraph)
        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setMinimumHeight(180)
        self.raster_plot.setBackground("w")
        self.raster_plot.setMouseEnabled(x=False, y=False)

        # New prediction plot (PyQtGraph)
        self.prediction_plot = pg.PlotWidget()
        self.prediction_plot.setMinimumHeight(140)
        self.prediction_plot.setBackground("w")
        self.prediction_plot.setMouseEnabled(x=False, y=False)
        self.prediction_plot.setLabel('left', 'Predicted p')
        self.prediction_plot.setLabel('bottom', 'Time (s, centered)')

        # Color maps (class_index -> pen) using plot_index hues
        # Expect columns: class_index, plot_index
        if 'class_index' in self.behaviour_rows.columns:
            self._beh_by_class = self.behaviour_rows.set_index('class_index')
            self._class_color = {}
            n_behaviours = self.behaviour_rows.shape[0]
            for r in self.behaviour_rows.itertuples():
                c = pg.intColor(r.plot_index, hues=n_behaviours, values=1, maxValue=255)
                c_a = pg.mkColor(c); c_a.setAlpha(220)
                self._class_color[r.class_index] = pg.mkPen(c_a, width=2)
        else:
            self._beh_by_class = None
            self._class_color = {}

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)

        main = QVBoxLayout(self)
        main.addWidget(self.video_widget, stretch=5)
        main.addWidget(behaviour_container, stretch=1)  # replaced self.behavior_plot with container
        main.addWidget(self.raster_plot, stretch=2)
        main.addWidget(self.prediction_plot, stretch=1)  # added prediction plot
        main.addLayout(controls)

        # Data holders
        self.times = None
        self.dist_shelter_vect = None

        # Initial plot
        self.update_all_plots(plot_dist_time_window*2.0)


    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.playBtn.setText("▶")
        else:
            self.player.play()
            self.playBtn.setText("⏸")

    def update_slider_and_all_plots(self, ms):
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)
        t = ms / 1000.0
        self.update_all_plots(t)

    def on_slider_moved(self, ms):
        self.player.setPosition(ms)

    def on_duration_changed(self, duration):
        self.slider.setMaximum(duration)
        self.update_slider_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_slider_overlays()

    def eventFilter(self, obj, event):
        if obj is self.slider and event.type() in (QEvent.Resize, QEvent.Move):
            self.update_slider_overlays()
        return super().eventFilter(obj, event)

    def update_slider_overlays(self):
        """Position/size the red overlay showing post‑training segment."""
        duration_ms = self.slider.maximum()
        if duration_ms <= 0 or self.latest_train_time * 1000 >= duration_ms:
            # No overlay needed (either no duration yet or training covered full video)
            self.post_train_overlay.hide()
            return
        slider_geom = self.slider.geometry()
        # Groove approximate height (small bar)
        groove_h = 6
        y_pos = slider_geom.y() + (slider_geom.height() - groove_h) // 2
        frac = max(0.0, min(1.0, (self.latest_train_time * 1000) / duration_ms))
        start_x = slider_geom.x() + int(frac * slider_geom.width())
        width = slider_geom.x() + slider_geom.width() - start_x
        if width <= 0:
            self.post_train_overlay.hide()
            return
        self.post_train_overlay.setGeometry(start_x, y_pos, width, groove_h)
        self.post_train_overlay.show()

    def update_all_plots(self, t):
        # Center plots at time t, window = plot_dist_time_window
        t0 = t - plot_dist_time_window / 2
        t1 = t + plot_dist_time_window / 2

        # --- Behavior plot ---
        self.behavior_plot.clear()
        self.behavior_plot.setXRange(-plot_dist_half_window, plot_dist_half_window)

        n_behaviours = self.behaviour_rows.shape[0]
        # One row per behaviour (centered at integer y)
        self.behavior_plot.setYRange(-0.5, n_behaviours - 0.5)
        self.behavior_plot.hideAxis('left')
        self.behavior_plot.hideAxis('bottom')  # optional; comment out if you want ticks

        for i, _row in enumerate(self.behaviour_rows.itertuples()):
            color_id = _row.plot_index
            base_color = pg.intColor(color_id, hues=n_behaviours, values=1, maxValue=255)
            base_color_a = pg.mkColor(base_color)
            base_color_a.setAlpha(160)
            brush = pg.mkBrush(base_color_a)
            pen = pg.mkPen(pg.mkColor(base_color).darker(200), width=1)

            xs, ys, ws, hs = [], [], [], []
            for start, stop in _row.intervals:
                if stop < (t - plot_dist_half_window) or start > (t + plot_dist_half_window):
                    continue
                s = max(start, t - plot_dist_half_window)
                e = min(stop, t + plot_dist_half_window)
                width = e - s
                if width <= 0:
                    continue
                x_center = ((s + e) / 2.0) - t
                xs.append(x_center)
                ys.append(i)        # row index
                ws.append(width)
                hs.append(0.8)

            if xs:
                bar_item = pg.BarGraphItem(x=xs, y=ys, width=ws, height=hs, brush=brush, pen=pen)
                self.behavior_plot.addItem(bar_item)

        # Center line
        vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))
        self.behavior_plot.addItem(vline)

        # Time label at top row
        minutes = int(max(0, t) // 60)
        seconds = int(max(0, t) % 60)
        time_label = f"t = {minutes:02d}:{seconds:02d}"
        text_item = pg.TextItem(text=time_label, anchor=(0, 0), color='k')
        text_item.setPos(-plot_dist_half_window + 0.2, n_behaviours - 0.4)
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

        # --- Prediction plot ---
        self.prediction_plot.clear()
        # Clamp t within data range
        t_center = float(np.clip(t, 0.0, t_stop_all))
        win_left = max(0.0, t_center - plot_dist_half_window)
        win_right = min(t_stop_all, t_center + plot_dist_half_window)

        # Determine coord names robustly
        time_dim = 'time' if 'time' in fit_predictions_xr.dims else fit_predictions_xr.dims[0]
        label_dim = 'label' if 'label' in fit_predictions_xr.dims else [d for d in fit_predictions_xr.dims if d != time_dim][0]

        pred_slice = fit_predictions_xr.sel({time_dim: slice(win_left, win_right)})
        if pred_slice[time_dim].size > 1:
            rel_time = pred_slice[time_dim].values - t_center
            self.prediction_plot.setXRange(-plot_dist_half_window, plot_dist_half_window, padding=0)
            self.prediction_plot.setYRange(0.0, 1.0, padding=0.02)

            # Iterate labels
            for lbl in pred_slice[label_dim].values:
                arr = pred_slice.sel({label_dim: lbl}).values
                # Map lbl to class_index (assuming they are identical or lbl is class_index)
                class_index = int(lbl)
                pen = self._class_color.get(class_index, pg.mkPen('k', width=1))
                self.prediction_plot.plot(rel_time, arr, pen=pen)

            vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#008000', width=2))
            self.prediction_plot.addItem(vline)

            # Optional zero line (comment out if not needed)
            # hline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#888', style=Qt.DashLine))
            # self.prediction_plot.addItem(hline)

            # Time label (reuse formatting)
            minutes = int(max(0, t_center) // 60)
            seconds = int(max(0, t_center) % 60)
            time_label = f"t = {minutes:02d}:{seconds:02d}"
            txt = pg.TextItem(time_label, anchor=(1, 0))
            txt.setPos(plot_dist_half_window, 0.0)
            self.prediction_plot.addItem(txt)
        else:
            self.prediction_plot.setXRange(-plot_dist_half_window, plot_dist_half_window, padding=0)
            self.prediction_plot.setYRange(0.0, 1.0)

# --- Add this class below imports ---
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QSize

# Remove obsolete SliderMarker class (and related imports if any)
# (Delete the SliderMarker class definition block entirely)
# ...existing code...

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoPlayerWidget()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())

#%%