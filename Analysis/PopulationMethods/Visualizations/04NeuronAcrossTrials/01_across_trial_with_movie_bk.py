# -*- coding: utf-8 -*-
"""
📃 ./04NeuronsAcrossTrials/01_across_trial_with_movie.py

🕰️  created on 2025-09-12

🤡 author: Dylan Festa

Select mouse, session, show ALL behaviours in that session and ALL available units.

Units and behaviours are selectable from dropdowns.

Once selected, show a raster plot of the selected unit across ALL trials of that behaviour.
It also shows the video, synchronized to the raster plot (because why not).

"""
#%%
from __future__ import annotations
from time import time
from typing import List,Dict,Tuple,Union
#%%
import os,sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox, QLabel, QFrame)

# Use VLC backend for video playback
import vlc

import pyqtgraph as pg


# This is to save the plots
import plotly.express as px
import plotly.graph_objects as go

import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import tempfile
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains, SpikeTrainByTrials



movies_path = Path.home() / 'Videos800Compressed'

local_load_path = Path(tempfile.gettempdir()) / "TempDatadictSaves"
# check that exists
if not local_load_path.exists():
    raise FileNotFoundError(f"Local load path {local_load_path} does not exist. Please run save_data_locally.py first.")

assert movies_path.exists(), f"Movies path {movies_path} does not exist."


#%%

path_this_file = Path(__file__).resolve()

path_save_raster_plots = path_this_file.parent / "local_outputs_plots"
path_save_raster_plots.mkdir(parents=True, exist_ok=True)

#%%

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
    'hunt_switch': ("#000000", 12),
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
    # error if behaviour not in behaviours_color_dict
    for b in beh_df['behaviour'].values:
        if b not in behaviours_color_dict:
            raise ValueError(f"Behaviour {b} not in behaviours_color_dict.")
    return beh_df


def get_all_units(the_mouse:str,the_session:str,*,
            the_data_dict:Union[None,Dict]=None,
            spiketrains:Union[None,SpikeTrains]=None) -> SpikeTrains:
    # error if both the_data_dict and spiketrains are provided, because it's ambiguous
    if the_data_dict is not None and spiketrains is not None:
        raise ValueError("Provide either the_data_dict or spiketrains, or neither, not both! ")
    
    if the_data_dict is None and spiketrains is None:
        the_data_dict = rdl.load_local_preprocessed_dict(the_mouse, the_session, local_load_path)
    if spiketrains is None:
        time_index = the_data_dict['time_index']
        spike_times = the_data_dict['spike_times']
        cluster_index = the_data_dict['cluster_index']
        region_index = the_data_dict['region_index']
        spiketrains = SpikeTrains.from_spike_list(spike_times,
                                    units=cluster_index,
                                    unit_location=region_index,
                                    isi_minimum=1/200.0,
                                    t_start=0.0,t_stop=time_index[-1])
        spiketrains = spiketrains.filter_by_unit_location('PAG')
    units_str = [f"unit {u}" for u in spiketrains.units]
    return spiketrains.units.copy(), units_str



def generate_spiketrain_by_trials(spiketrains: SpikeTrains,unit_id:Union[str,int],
                            *,
                            start_stop_times:Union[List[Tuple[float,float]],None]=None,
                            point_times:Union[List[float],None]=None,
                            padding_beh_left:float=5.0,
                            padding_beh_right:float=15.0) -> SpikeTrainByTrials:
    if start_stop_times is not None and point_times is not None:
        raise ValueError("Provide either start_stop_times or point_times, not both.")
    if start_stop_times is None and point_times is None:
        raise ValueError("Provide either start_stop_times or point_times, not both.")
    if start_stop_times is not None:
        return SpikeTrainByTrials.from_spike_trains(
            unit_id=unit_id,
            spike_trains=spiketrains,
            time_event_starts=[x[0] for x in start_stop_times],
            time_event_stops=[x[1] for x in start_stop_times],
            t_pad_left=padding_beh_left,
            t_pad_right=padding_beh_right
        )
    else:
        return SpikeTrainByTrials.from_spike_trains(
            unit_id=unit_id,
            spike_trains=spiketrains,
            time_event_starts=point_times,
            time_event_stops=None,
            t_pad_left=padding_beh_left,
            t_pad_right=padding_beh_right
        )


def get_time_intervals_in_raster(spiketrainsbytrials:SpikeTrainByTrials) -> List[Tuple[float,float]]:
    intervals = []
    for t_event in spiketrainsbytrials.event_times:
        t_start = t_event - spiketrainsbytrials.t_pad_left
        t_end = t_event + spiketrainsbytrials.t_pad_right
        intervals.append((t_start, t_end))
    return intervals


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


def do_raster_at_time(spiketrains:SpikeTrains,t0:float, t_plot:float,
                          output_full_path:Union[str,Path],):
    raise NotImplementedError("This function is not implemented yet!")
    t_start = t0 - t_plot/2
    t_end = t0 + t_plot/2
    n_units = spiketrains.n_units
    # Labels (string) for each unit index 0..n_units-1
    unit_labels = [str(u) for u in spiketrains.units]
    spiketrains_windowed = spiketrains.filter_by_time(t_start,t_end)
    x_plot,y_plot = spiketrains_windowed.get_line_segments_xynans(
                            t_start=t_start,t_stop=t_end,time_offset=t0)
    fig = go.Figure()
    # Raster: line segments already encoded via NaN breaks in x_plot/y_plot
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode='lines',
            line=dict(color='black',width=1),
            hoverinfo='skip',
            name='spikes'
        )
    )
    # Vertical reference line at t=0
    fig.add_shape(
        type='line', x0=0, x1=0, y0=-0.5, y1=n_units-0.5,
        line=dict(color='green', width=1)
    )
    # Axes & grid (horizontal only every unit)
    fig.update_xaxes(title=f'time relative to t0={t0} s', showgrid=False, zeroline=False,
                     range=[-t_plot/2, t_plot/2])
    fig.update_yaxes(title='unit', showgrid=True, gridcolor='lightgrey', dtick=1,
                     range=[-0.5, n_units-0.5], autorange=False, zeroline=False,
                     tickmode='array', tickvals=list(range(n_units)), ticktext=unit_labels)
    # Dynamic size (explicit width/height so static exports respect it)
    fig_height = max(300, int(n_units * 20))  
    fig_width = 1100
    fig.update_layout(height=fig_height, width=fig_width, margin=dict(l=100,r=25,t=40,b=50),
                      paper_bgcolor='white', plot_bgcolor='white')
    # Ensure parent folder exists & save
    output_full_path = Path(output_full_path)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_full_path.with_suffix('.html')))
    # Provide explicit width/height to force size; scale>1 for higher DPI if desired
    fig.write_image(str(output_full_path.with_suffix('.png')), width=fig_width, height=fig_height, scale=2)
    fig.write_image(str(output_full_path.with_suffix('.pdf')), width=fig_width, height=fig_height, scale=1)
    return fig

#%%
# the_mouse = 'afm16924'
# the_session = '240526'

# paths_test = rdl.get_paths(the_mouse, the_session)
# read_dict = rdl.load_local_preprocessed_dict(the_mouse, the_session, local_load_path)

# beh_test = generate_beh_dataframe(the_mouse, the_session, read_dict)
# list_beh_times_test = generate_video_behaviour_intervals(beh_test, 'pup_run', pad_left=5.0, pad_right=5.0)

# unit_selector_df_test = get_unit_selector_df(the_mouse, the_session, read_dict)

#%%


# some parameters here
padding_beh_left = 5.0  # seconds before the behaviour
padding_beh_right = 15.0  # seconds after the behaviour

plot_dist_time_window = 10.0  # seconds around the current time to plot
plot_dist_half_window = plot_dist_time_window / 2.0


mice_sessions = df_videos["mouse_session_name"].tolist()

class VideoSelectorWidget(QWidget):
    def __init__(self, df_videos, mice_sessions):
        super().__init__()
        self.df_videos = df_videos
        self.mouse_sessions = mice_sessions
        self.setWindowTitle("Select and Play Movie")

        # ComboBox for selection
        self.mouse_session_combo = QComboBox()
        self.mouse_session_combo.addItems(self.mouse_sessions)
        self.mouse_session_combo.currentIndexChanged.connect(self.on_mouse_session_selection_changed)

        # ComboBox for unit selection (populated after loading data)
        self.unit_selector_combo = QComboBox()
        self.unit_selector_combo.currentIndexChanged.connect(self.on_unit_selection_changed)

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

        # Video player setup (VLC backend)
        self.vlc_instance = vlc.Instance()
        self.media_player = self.vlc_instance.media_player_new()
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")

        # Behavior plot (with left-side legend)
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setMinimumHeight(100)
        self.behavior_plot.setBackground("w")
        self.behavior_plot.setMouseEnabled(x=False, y=False)
        # no need: raster already has a label
        #self.behavior_plot.setLabel('bottom', 'time (s, centered)')

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

        # Raster plot
        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setBackground("w")
        self.raster_plot.setMouseEnabled(x=False, y=False)
        self.raster_plot.setLabel('bottom', 'time (s, centered)')
        self.raster_plot.setLabel('left', 'unit')

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
        
        # Slider (we will manage range via VLC get_length in poll)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider_moved)

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
        main.addWidget(self.mouse_session_combo)
        main.addWidget(self.unit_selector_combo)
        main.addLayout(self.behaviour_selector_layout)
        main.addWidget(self.video_frame, stretch=5)
        main.addWidget(behaviour_container, stretch=1)
        main.addWidget(self.raster_plot, stretch=2)
        main.addLayout(controls)
        # New save raster button at bottom
        self.saveRasterBtn = QPushButton("Save Raster")
        self.saveRasterBtn.setToolTip("Save current raster window around t")
        self.saveRasterBtn.clicked.connect(self.save_current_raster)
        main.addWidget(self.saveRasterBtn)

        # Data holders
        self.times = None
        self.dist_shelter_vect = None
        self.data_dict = None
        self.behaviour_df = None
        self.unit_selector = None
        self.spiketrains_full = None
        self.spiketrains_to_plot = None
        self.t_stop_all = None

        # Poll timer to sync UI with VLC
        self._last_length_ms = 0
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll_player)
        self.poll_timer.start()

        # Load first video and data if available
        if len(self.df_videos) > 0:
            self.load_video_and_data(0)

    def _set_video_output_handle(self):
        """Bind VLC video output to our QWidget handle."""
        wid = int(self.video_frame.winId())
        if sys.platform.startswith("linux"):
            self.media_player.set_xwindow(wid)
        elif sys.platform == "win32":
            self.media_player.set_hwnd(wid)
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(wid)

    def _poll_player(self):
        """Periodic UI update: slider, plots, and interval enforcement."""
        if not self.media_player:
            return
        length = int(self.media_player.get_length())
        if length > 0 and length != self.slider.maximum():
            self.slider.setMaximum(length)
        ms = int(self.media_player.get_time())
        if ms < 0:
            ms = 0
        self.update_slider_and_plot(ms)

    def on_mouse_session_selection_changed(self, idx):
        self.load_video_and_data(idx) # this also updates unit_selector_df
        self.update_unit_selector()

    def load_video_and_data(self, idx):
        video_path = self.df_videos.iloc[idx]["video_path"]
        mouse = self.df_videos.iloc[idx]["mouse"]
        session = self.df_videos.iloc[idx]["session"]
        # Store for file naming
        self.mouse = mouse
        self.session = session
        # Load media via VLC
        self.open_file(video_path)
        self.media_player.pause()
        self.playBtn.setText("▶")
        # reset intervals when switching video
        self.time_intervals_list = []
        self.set_intervals_ms(self.time_intervals_list)
        # Load data only once per selection
        self.data_dict = rdl.load_local_preprocessed_dict(mouse, session, local_load_path)
        # and the full spiketrains
        cluster_index = self.data_dict['cluster_index']
        region_index = self.data_dict['region_index']
        time_index = self.data_dict['time_index']
        t_stop_spiketrain = time_index[-1]
        spike_times = self.data_dict['spike_times']
        self.spiketrains_full = SpikeTrains.from_spike_list(spike_times,
                                    units=cluster_index,
                                    unit_location=region_index,
                                    isi_minimum=1/200.0, 
                                    t_start=0.0,
                                    t_stop=t_stop_spiketrain)
        self.t_stop_all = t_stop_spiketrain
        self.behaviour_df = generate_beh_dataframe(mouse, session, self.data_dict)
        self.unit_selector = get_all_units(mouse, session, spiketrains=self.spiketrains_full)
        self.update_unit_selector()
        self.update_behaviour_selector()
        # initialize selection to first entry
        if self.unit_selector_combo.count() > 0:
            self.unit_selector_combo.setCurrentIndex(0)
            self.on_unit_selection_changed(0)

    def open_file(self, path):
        """Open media file in VLC and bind to widget."""
        try:
            self.media_player.stop()
        except Exception:
            pass
        if not path:
            self.playBtn.setText("▶")
            return
        p = Path(path)
        if not p.exists():
            print(f"File does not exist: {p}")
            return
        media = self.vlc_instance.media_new(str(p))
        self.media_player.set_media(media)
        # Ensure video output is set
        self._set_video_output_handle()
        # Autoplay briefly to ensure metadata (length) is available, then pause
        self.media_player.play()
        QTimer.singleShot(150, lambda: (self.media_player.pause(), self.playBtn.setText("▶")))

    def update_unit_selector(self):
        # Populate unit selection combo from self.unit_selector
        names = self.unit_selector[1] if self.unit_selector is not None else []
        block = self.unit_selector_combo.blockSignals(True)
        self.unit_selector_combo.clear()
        self.unit_selector_combo.addItems(names)
        self.unit_selector_combo.blockSignals(block)

    def on_unit_selection_changed(self, idx:int):
        # generate spiketrains_to_plot based on current selection
        if self.spiketrains_full is None or self.unit_selector_df is None:
            return
        if idx < 0 or idx >= len(self.unit_selector_df):
            return
        sel_name = self.unit_selector_combo.currentText()
        try:
            self.spiketrains_to_plot = get_subselection_of_units(sel_name, self.spiketrains_full, self.unit_selector_df)
        except Exception:
            self.spiketrains_to_plot = None
        # refresh raster for current time
        pos_ms = int(self.media_player.get_time())
        if pos_ms < 0:
            pos_ms = 0
        t = pos_ms / 1000.0
        self.update_raster_plot(t)

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
        if self.media_player.is_playing() and self.intervals_ms:
            tgt = self._next_allowed_position(int(self.media_player.get_time()))
            if tgt is None:
                self.media_player.pause()
                self.playBtn.setText("▶")
            else:
                self._safe_seek(tgt)

    def toggle_play(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.playBtn.setText("▶")
        else:
            # ensure we start within an allowed interval if any
            if self.intervals_ms:
                tgt = self._next_allowed_position(int(self.media_player.get_time()))
                if tgt is None:
                    self.media_player.pause()
                    self.playBtn.setText("▶")
                    return
                self._safe_seek(tgt)
            self.media_player.play()
            self.playBtn.setText("⏸")

    def on_slider_moved(self, ms: int):
        # When intervals are active, jump to the beginning of the next available interval
        if not self.intervals_ms:
            self.media_player.set_time(int(ms))
            return
        # find next interval whose start >= ms; if ms is inside an interval, jump to that interval's start
        idx_inside = self._interval_index_for_time(ms)
        if idx_inside is not None:
            start_ms, _ = self.intervals_ms[idx_inside]
            if int(self.media_player.get_time()) != start_ms:
                self._safe_seek(start_ms)
            return
        idx_next = self._next_interval_index_from_time(ms)
        if idx_next is not None:
            start_ms, _ = self.intervals_ms[idx_next]
            if int(self.media_player.get_time()) != start_ms:
                self._safe_seek(start_ms)
        else:
            # no next interval; loop to start of first interval
            first_start, _ = self.intervals_ms[0]
            if int(self.media_player.get_time()) != first_start:
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
        self.update_raster_plot(t)

    # --- Interval helpers ---
    def set_intervals_ms(self, intervals_sec: List[Tuple[float, float]]):
        # Convert to ms, sort, merge overlaps, and clip to media duration
        if not intervals_sec:
            self.intervals_ms = []
            self.current_interval_idx = None
            return
        dur_raw = int(self.media_player.get_length())
        dur = dur_raw if dur_raw > 0 else None
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
        if int(self.media_player.get_time()) == int(ms):
            return
        self._seeking = True
        self.media_player.set_time(int(ms))
        QTimer.singleShot(0, lambda: setattr(self, '_seeking', False))

    # Helper to jump by a delta in milliseconds, respecting intervals and bounds
    def _jump_by_ms(self, delta_ms: int):
        cur = int(self.media_player.get_time())
        if cur < 0:
            cur = 0
        dur_raw = int(self.media_player.get_length())
        dur = dur_raw if dur_raw > 0 else None
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
        # no need: raster already has a label
        #self.behavior_plot.setLabel('bottom', 'time (s, centered)')

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
    
    def update_raster_plot(self,t):
        # Center plots at time t, window = plot_dist_time_window
        t0 = t - plot_dist_time_window / 2
        t1 = t + plot_dist_time_window / 2
        # --- Raster plot ---
        self.raster_plot.clear()
        self.raster_plot.setXRange(-plot_dist_half_window, plot_dist_half_window)
        if self.spiketrains_to_plot is None:
            return
        # Y range based on selected units
        self.raster_plot.setYRange(0, self.spiketrains_to_plot.n_units + 1)
        # Get raster data in window
        _t_start_spikes = np.maximum(0.1, t0)
        t_stop_all = self.t_stop_all if self.t_stop_all is not None else t1
        _t_stop_spikes = np.minimum(np.maximum(0.2, t1), t_stop_all)
        _t_offset_spikes = _t_start_spikes + (_t_stop_spikes - _t_start_spikes) / 2
        # Get line segments for raster plot
        xplot, yplot = self.spiketrains_to_plot.get_line_segments_xynans(
            t_start=_t_start_spikes,
            t_stop=_t_stop_spikes, time_offset=_t_offset_spikes)
        pen = pg.mkPen(color='k', width=1)
        self.raster_plot.plot(xplot, yplot, pen=pen)
        vline2 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))
        self.raster_plot.addItem(vline2)

    def save_current_raster(self):
        """Save raster using current selection & time window."""
        if self.spiketrains_to_plot is None:
            return
        try:
            pos_ms = int(self.media_player.get_time())
            if pos_ms < 0:
                pos_ms = 0
            t = pos_ms / 1000.0
            fname = f"{getattr(self,'mouse','unk')}_{getattr(self,'session','unk')}_t{t:.3f}"
            out_path = path_save_raster_plots / fname
            do_raster_at_time(self.spiketrains_to_plot, t0=t, t_plot=plot_dist_time_window, output_full_path=out_path)
        except Exception:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoSelectorWidget(df_videos, mice_sessions)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())
