# -*- coding: utf-8 -*-
"""
§ created on 2025-09-08

@ author: Dylan Festa

Select mouse, session, show behaviours that last in total more than `Tmin_beh` seconds (default 15s).

Ask for padding time before and after the behaviour (default 5s).

Once the behaviour is selected, show the movie ONLY when that behaviour is happening.

"""
#%%
from __future__ import annotations
from typing import List,Dict,Tuple
#%%
import os,sys

from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox, QLabel, QFrame)

import pyqtgraph as pg


import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import tempfile
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

movies_path = Path.home() / 'Videos800Compressed'

local_load_path = Path(tempfile.gettempdir()) / "TempDatadictSaves"
# check that exists
if not local_load_path.exists():
    raise FileNotFoundError(f"Local load path {local_load_path} does not exist. Please run save_data_locally.py first.")

assert movies_path.exists(), f"Movies path {movies_path} does not exist."


Tmin_beh = 15.0  # seconds
padding_video_left = 5.0  # seconds before the behaviour
padding_video_right = 5.0  # seconds after the behaviour

#%%

# workaround: a dictionary with every behaviour, behaviour index and color
# assigned to it

behaviours_color_dict = {
    'pup_run': ("#b7e1ff", 0),
    'pup_grab': ("#61b0ff", 1),
    'pup_retrieve': ("#00027a", 2),
    'nesting': ('#2ca02c', 3),
    'bed_retrieve': ("#004b25", 4),
    'approach': ('#ff7f0e', 5),
    'pursuit': ("#e5ff00", 6),
    'chase': ('#d62728', 7),
    'attack': ("#490000", 8),
    'eat': ("#553624", 9),
    'sniff': ('#9467bd', 10),
    'run_away': ("#ff00b3", 11),
}



#%%
def get_mouse_session_from_video_path(video_path: Path) -> tuple[str, str]:
    pattern = re.compile(r'^(?P<mouse>afm\d{5})_(?P<session>.+?)(?=\.mp4$)')
    m = pattern.match(video_path.name)
    if m:
        return m.group('mouse'), m.group('session')
    else:
        raise ValueError(f"Video path {video_path} does not match expected pattern.")
def generate_mouse_session_df(movies_path: Path,
                              *,
                              session_filter: List[str]) -> pd.DataFrame:
    df_rows = []
    for p in movies_path.glob("*.mp4"):          # or rglob if you need recursion
        try:
            mouse, session = get_mouse_session_from_video_path(p)
            # make sure that no element of session_filter is in session
            if not any(filter_str in session for filter_str in session_filter):
                mouse_session_str = f"{mouse} / {session}"
                rec = {
                    "mouse": mouse,
                    "session": session,
                    "video_path": p.resolve(),
                    "mouse_session_name": mouse_session_str}
                df_rows.append(rec)
        except ValueError:
            continue
    df_videos = pd.DataFrame(df_rows, columns=["video_path", "mouse", "session", "mouse_session_name"])
    df_videos.sort_values(by=["mouse", "session"], inplace=True)
    return df_videos


def generate_beh_dataframe(the_mouse:str,the_session:str,the_data_dict:Dict):
    beh_df = rdl.convert_to_behaviour_timestamps(
        the_mouse, the_session, the_data_dict['behaviour'])
    # select only behaviours that last more than Tmin_beh seconds
    beh_df_ret = beh_df.query("total_duration >= @Tmin_beh").copy()
    # error if behaviour not in behaviours_color_dict
    for b in beh_df_ret['behaviour'].values:
        if b not in behaviours_color_dict:
            raise ValueError(f"Behaviour {b} not in behaviours_color_dict.")
    return beh_df_ret


#%%
session_filter = ['test','Kilosort', 'coded','overlap','raw']
df_videos = generate_mouse_session_df(movies_path, session_filter=session_filter)

def generate_video_behaviour_intervals(beh_df:pd.DataFrame, behaviour_name:str, *, pad_left:float=5.0, pad_right:float=5.0) -> List[Tuple[float,float]]:
    """
    Given a behaviour dataframe and a behaviour name, return a list of tuples (start_time, end_time)
    for each interval where the behaviour occurs, padded by pad_left and pad_right seconds.
    """
    if behaviour_name == "all":
        return [] # no filtering, return empty list
    # filter the dataframe for the given behaviour
    intervals = beh_df.query("behaviour == @behaviour_name")['start_stop_times'].values[0]
    return rdl.add_padding_to_intervals(intervals, pad_left=pad_left, pad_right=pad_right)

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


#%%
the_mouse = 'afm16924'
the_session = '240526'

paths_test = rdl.get_paths(the_mouse, the_session)
read_dict = rdl.load_local_preprocessed_dict(the_mouse, the_session, local_load_path)

beh_test = generate_beh_dataframe(the_mouse, the_session, read_dict)
list_beh_times_test = generate_video_behaviour_intervals(beh_test, 'pup_run', pad_left=5.0, pad_right=5.0)

#%%

#%%

# some parameters here
plot_dist_time_window = 10.0  # seconds around the current time to plot
plot_dist_half_window = plot_dist_time_window / 2.0


mice_sessions = df_videos["mouse_session_name"].tolist()

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

        # Behaviour selector UI
        self.behaviour_selector_layout = QVBoxLayout()
        self.behaviour_buttons = []
        self.behaviour_label = QLabel("No behaviour selected.")
        self.behaviour_label.setStyleSheet("color: #d32f2f;")  # red by default
        self.behaviour_selector_layout.addWidget(self.behaviour_label)

        # intervals of interest
        self.time_intervals_list = []
        # state for interval-restricted playback
        self.intervals_ms: List[Tuple[int, int]] = []
        self.current_interval_idx: int | None = None
        self._seeking: bool = False  # guard to avoid recursive positionChanged loops

        # Video player setup
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(100)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)

        # Behavior plot (with left-side legend)
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setMinimumHeight(100)
        self.behavior_plot.setBackground("w")
        self.behavior_plot.setMouseEnabled(x=False, y=False)
        self.behavior_plot.setLabel('bottom', 'time (s, centered)')

        self.legend_widget = QWidget()
        _leg_layout = QVBoxLayout(self.legend_widget)
        _leg_layout.setContentsMargins(2, 2, 2, 2)
        _leg_layout.setSpacing(2)

        behaviour_container = QWidget()
        _beh_layout = QHBoxLayout(behaviour_container)
        _beh_layout.setContentsMargins(0, 0, 0, 0)
        _beh_layout.setSpacing(4)
        _beh_layout.addWidget(self.legend_widget, 0)
        _beh_layout.addWidget(self.behavior_plot, 1)

        # Play/Pause button
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)
        # Jump buttons
        self.back3Btn = QPushButton("−3s")
        self.back3Btn.setToolTip("Jump back 3 seconds")
        self.back3Btn.clicked.connect(lambda: self._jump_by_ms(-3000))
        self.fwd3Btn = QPushButton("+3s")
        self.fwd3Btn.setToolTip("Jump forward 3 seconds")
        self.fwd3Btn.clicked.connect(lambda: self._jump_by_ms(3000))
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        # use custom handler so we can respect allowed intervals when active
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.player.positionChanged.connect(self.update_slider_and_plot)
        self.player.durationChanged.connect(self.slider.setMaximum)

        # PyQtGraph plot for behaviour intervals / time-series (kept)
        self.plot = pg.PlotWidget(labels={'left': 'behaviour', 'bottom': 'time (s)'})
        self.curve = self.plot.plot(pen='y')
        self.vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot.addItem(self.vline)

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(self.back3Btn)
        controls.addWidget(self.playBtn)
        controls.addWidget(self.fwd3Btn)
        controls.addWidget(self.slider)

        main = QVBoxLayout(self)
        main.addWidget(self.combo)
        main.addLayout(self.behaviour_selector_layout)
        main.addWidget(self.video_widget, stretch=5)
        main.addWidget(behaviour_container, stretch=1)
        main.addLayout(controls)

        # Data holders
        self.times = None
        self.dist_shelter_vect = None
        self.data_dict = None
        self.behaviour_df = None

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
        # reset intervals when switching video
        self.time_intervals_list = []
        self.set_intervals_ms(self.time_intervals_list)
        # Load data only once per selection
        self.data_dict = rdl.load_local_preprocessed_dict(mouse, session, local_load_path)
        self.behaviour_df = generate_beh_dataframe(mouse, session, self.data_dict)
        self.update_behaviour_selector()

    def update_behaviour_selector(self):
        # Remove old buttons
        for btn in self.behaviour_buttons:
            self.behaviour_selector_layout.removeWidget(btn)
            btn.deleteLater()
        self.behaviour_buttons = []
        # Reset label to default state (red)
        self.behaviour_label.setText("No behaviour selected.")
        self.behaviour_label.setStyleSheet("color: #d32f2f;")
        # first button is "all"
        btn = QPushButton("ALL (full video)")
        btn.clicked.connect(lambda checked, name="all": self.on_behaviour_selected(name))
        self.behaviour_selector_layout.addWidget(btn)
        self.behaviour_buttons.append(btn)
        if self.behaviour_df is not None and len(self.behaviour_df) > 0:
            for idx, row in self.behaviour_df.iterrows():
                beh_text = f"{row['behaviour']} | Trials: {row['n_trials']} | Duration: {row['total_duration']:.1f}s"
                btn = QPushButton(beh_text)
                btn.clicked.connect(lambda checked, name=row['behaviour']: self.on_behaviour_selected(name))
                self.behaviour_selector_layout.addWidget(btn)
                self.behaviour_buttons.append(btn)
        else:
            self.behaviour_label.setText("No behaviour available.")
            self.behaviour_label.setStyleSheet("color: #d32f2f;")
        # Rebuild legend for current session
        self.build_behaviour_legend()

    # Build/update the legend using behaviours_color_dict
    def build_behaviour_legend(self):
        layout = self.legend_widget.layout()
        # clear previous
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self.behaviour_df is None or len(self.behaviour_df) == 0:
            return
        for row in self.behaviour_df.itertuples():
            beh_ = row.behaviour
            color_hex = behaviours_color_dict[beh_][0]
            base_color_a = pg.mkColor(color_hex)
            base_color_a.setAlpha(160)
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            patch = QFrame()
            patch.setFixedSize(14, 14)
            patch.setStyleSheet(
                f"background-color: rgba({base_color_a.red()},{base_color_a.green()},{base_color_a.blue()},{base_color_a.alpha()});" \
                "border: 1px solid #444;"
            )
            lab = QLabel(str(beh_))
            lab.setStyleSheet("font-size:10px;")
            row_l.addWidget(patch, 0)
            row_l.addWidget(lab, 1)
            layout.addWidget(row_w)
        layout.addStretch(1)

    def on_behaviour_selected(self, behaviour_name):
        if behaviour_name == "all":
            self.behaviour_label.setText("Showing: ALL (full video)")
            self.behaviour_label.setStyleSheet("color: #666666;")  # gray for ALL
        else:
            self.behaviour_label.setText(f"Video locked on: {behaviour_name}")
            self.behaviour_label.setStyleSheet("color: #d32f2f;")  # red for specific behaviours
        self.time_intervals_list = generate_video_behaviour_intervals(
            self.behaviour_df, behaviour_name, pad_left=padding_video_left, pad_right=padding_video_right)
        # build ms intervals and enforce playback rules
        self.set_intervals_ms(self.time_intervals_list)
        # If currently playing and outside allowed range, jump to next allowed
        if self.player.state() == QMediaPlayer.PlayingState and self.intervals_ms:
            tgt = self._next_allowed_position(self.player.position())
            if tgt is None:
                self.player.pause()
                self.playBtn.setText("▶")
            else:
                self._safe_seek(tgt)

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.playBtn.setText("▶")
        else:
            # ensure we start within an allowed interval if any
            if self.intervals_ms:
                tgt = self._next_allowed_position(self.player.position())
                if tgt is None:
                    self.player.pause()
                    self.playBtn.setText("▶")
                    return
                self._safe_seek(tgt)
            self.player.play()
            self.playBtn.setText("⏸")

    def on_slider_moved(self, ms: int):
        # When intervals are active, jump to the beginning of the next available interval
        if not self.intervals_ms:
            self.player.setPosition(ms)
            return
        # find next interval whose start >= ms; if ms is inside an interval, jump to that interval's start
        idx_inside = self._interval_index_for_time(ms)
        if idx_inside is not None:
            start_ms, _ = self.intervals_ms[idx_inside]
            if self.player.position() != start_ms:
                self._safe_seek(start_ms)
            return
        idx_next = self._next_interval_index_from_time(ms)
        if idx_next is not None:
            start_ms, _ = self.intervals_ms[idx_next]
            if self.player.position() != start_ms:
                self._safe_seek(start_ms)
        else:
            # no next interval; loop to start of first interval
            first_start, _ = self.intervals_ms[0]
            if self.player.position() != first_start:
                self._safe_seek(first_start)

    def update_slider_and_plot(self, ms):
        # Enforce interval-restricted playback when active
        if self.intervals_ms and not self._seeking:
            # If outside any interval, or past the end of current interval, jump accordingly
            idx_inside = self._interval_index_for_time(ms)
            if idx_inside is None:
                idx_next = self._next_interval_index_from_time(ms)
                if idx_next is not None:
                    start_ms = self.intervals_ms[idx_next][0]
                    if ms != start_ms:
                        self._safe_seek(start_ms)
                        return
                else:
                    # after last interval: loop to first interval
                    first_start = self.intervals_ms[0][0]
                    if ms != first_start:
                        self._safe_seek(first_start)
                        return
            else:
                # if we just passed the end of the interval, jump to next or loop
                start_ms, end_ms = self.intervals_ms[idx_inside]
                if ms >= end_ms:
                    next_idx = idx_inside + 1
                    if next_idx < len(self.intervals_ms):
                        next_start = self.intervals_ms[next_idx][0]
                        if ms != next_start:
                            self._safe_seek(next_start)
                    else:
                        # loop to the start of the first interval
                        first_start = self.intervals_ms[0][0]
                        if ms != first_start:
                            self._safe_seek(first_start)
                    return
                # track current interval index
                self.current_interval_idx = idx_inside
        # update slider without triggering setPosition
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)
        # Update plot below video
        t = ms / 1000.0
        self.update_beh_plot(t)

    # --- Interval helpers ---
    def set_intervals_ms(self, intervals_sec: List[Tuple[float, float]]):
        # Convert to ms, sort, merge overlaps, and clip to media duration
        if not intervals_sec:
            self.intervals_ms = []
            self.current_interval_idx = None
            return
        dur = int(self.player.duration()) if self.player.duration() > 0 else None
        ints = []
        for s, e in intervals_sec:
            s_ms = max(0, int(round(s * 1000)))
            e_ms = int(round(e * 1000))
            if dur is not None:
                s_ms = max(0, min(s_ms, dur))
                e_ms = max(0, min(e_ms, dur))
            if e_ms > s_ms:
                ints.append((s_ms, e_ms))
        # sort and merge
        ints.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        for s, e in ints:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        self.intervals_ms = merged
        self.current_interval_idx = None

    def _interval_index_for_time(self, ms: int) -> int | None:
        # binary search over intervals
        lo, hi = 0, len(self.intervals_ms) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s, e = self.intervals_ms[mid]
            if ms < s:
                hi = mid - 1
            elif ms > e:
                lo = mid + 1
            else:
                return mid
        return None

    def _next_interval_index_from_time(self, ms: int) -> int | None:
        # first interval with start >= ms
        for i, (s, _) in enumerate(self.intervals_ms):
            if s >= ms:
                return i
        return None

    def _next_allowed_position(self, ms: int) -> int | None:
        # If inside interval, allow current; else return next interval start or loop to first
        idx_inside = self._interval_index_for_time(ms)
        if idx_inside is not None:
            return ms
        idx_next = self._next_interval_index_from_time(ms)
        if idx_next is not None:
            return self.intervals_ms[idx_next][0]
        # loop to first if intervals exist
        return self.intervals_ms[0][0] if self.intervals_ms else None

    def _safe_seek(self, ms: int):
        # Guarded seek that avoids recursive positionChanged handling
        if self.player.position() == ms:
            return
        self._seeking = True
        self.player.setPosition(ms)
        # Defer releasing the seeking flag to the next event loop turn
        QTimer.singleShot(0, lambda: setattr(self, '_seeking', False))

    # Helper to jump by a delta in milliseconds, respecting intervals and bounds
    def _jump_by_ms(self, delta_ms: int):
        cur = int(self.player.position())
        dur = int(self.player.duration()) if self.player.duration() > 0 else None
        target = cur + int(delta_ms)
        if target < 0:
            target = 0
        if dur is not None and target > dur:
            target = dur
        if self.intervals_ms:
            if self._interval_index_for_time(target) is None:
                nxt = self._next_allowed_position(target)
                if nxt is None:
                    return
                target = nxt
        self._safe_seek(target)
        
    def update_beh_plot(self, t):
        # Center plots at time t, window = plot_dist_time_window
        t0 = t - plot_dist_time_window / 2
        t1 = t + plot_dist_time_window / 2

        # --- Behavior plot ---
        self.behavior_plot.clear()
        if self.behaviour_df is None or len(self.behaviour_df) == 0:
            return
        self.behavior_plot.setXRange(-plot_dist_half_window, plot_dist_half_window)

        n_behaviours = len(self.behaviour_df)
        # One row per behaviour (centered at integer y)
        self.behavior_plot.setYRange(-0.5, n_behaviours - 0.5)
        self.behavior_plot.hideAxis('left')
        self.behavior_plot.showAxis('bottom')
        self.behavior_plot.setLabel('bottom', 'time (s, centered)')

        for i, _row in enumerate(self.behaviour_df.itertuples()):
            beh_ = _row.behaviour
            base_color = behaviours_color_dict[beh_][0]
            base_color_a = pg.mkColor(base_color)
            base_color_a.setAlpha(160)
            brush = pg.mkBrush(base_color_a)
            pen = pg.mkPen(pg.mkColor(base_color).darker(200), width=1)

            xs, ys, ws, hs = [], [], [], []
            for start, stop in _row.start_stop_times:
                if stop < t0 or start > t1:
                    continue
                s = max(start, t0)
                e = min(stop, t1)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoSelectorWidget(df_videos, mice_sessions)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())
