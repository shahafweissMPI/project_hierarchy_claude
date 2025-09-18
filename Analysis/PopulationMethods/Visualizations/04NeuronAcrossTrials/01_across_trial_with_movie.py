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
    QPushButton, QSlider, QFileDialog, QComboBox, QLabel, QFrame
)
# Use VLC backend for video playback
import vlc

import pyqtgraph as pg
# REMOVE: overlay/stack imports
# from PyQt5.QtWidgets import QStackedLayout
from PyQt5.QtGui import QFont
# REMOVE: event import
# from PyQt5.QtCore import QEvent


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
    'loom': ("#ff0000", 13),
    'turn': ("#00ff00", 14),
    'startle': ("#ff8000", 15),
    'escape': ("#8c564b", 16),
    'look_up': ("#e377c2", 17),
    'grab_play': ("#7f7f7f", 18),
    'pullback': ("#bcbd22", 19),
    'pup_drop': ("#348d77", 20),
    'gas_escape': ("#ff1493", 21),
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
    if (start_stop_times is not None) and len(start_stop_times)==0:
        start_stop_times=None
    if (point_times is not None) and len(point_times)==0:
        point_times=None

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




#%%
session_filter = ['test','Kilosort', 'coded','overlap','raw']
df_videos = generate_mouse_session_df(movies_path, session_filter=session_filter)

def generate_video_intervals_from_spiketrains(spiketrainsbytrials:Union[SpikeTrainByTrials|None]) -> List[Tuple[float,float]]:
    if spiketrainsbytrials is None:
        return [(0.0,100.0)]
    intervals = []
    for t_event in spiketrainsbytrials.event_times:
        t_start = t_event - spiketrainsbytrials.t_pad_left
        t_end = t_event + spiketrainsbytrials.t_pad_right
        intervals.append((t_start, t_end))
    return intervals


def generate_spiketrains_from_data_dict(data_dict:Dict) -> SpikeTrains:
    cluster_index = data_dict['cluster_index']
    region_index = data_dict['region_index']
    time_index = data_dict['time_index']
    spike_times = data_dict['spike_times']
    spiketrains = SpikeTrains.from_spike_list(\
                spike_times,
                units=cluster_index,
                unit_location=region_index,
                isi_minimum=1/200.0, 
                t_start=0.0,
                t_stop=time_index[-1])
    spiketrains = spiketrains.filter_by_unit_location('PAG')
    return spiketrains

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

        # NEW: behaviour selection combo (replaces button list panel)
        self.behaviour_selector_combo = QComboBox()
        self.behaviour_selector_combo.currentIndexChanged.connect(self.on_behaviour_combo_changed)

        # remove old behaviour button machinery
        # self.behaviour_selector_layout = QVBoxLayout()
        # self.behaviour_buttons = []
        # self.behaviour_label = QLabel("No behaviour selected.")
        # self.behaviour_selector_layout.addWidget(self.behaviour_label)

        # intervals of interest
        self.time_intervals_list = []
        # state for interval-restricted playback
        self.intervals_ms: List[Tuple[int, int]] = []
        self.current_interval_idx: int | None = None
        self._seeking: bool = False  # guard to avoid recursive positionChanged loops
        self.selected_behaviour_name = "all"
        # Add: opacity for duration bars (0..1)
        self.duration_bar_opacity = 0.5

        # Video player setup (VLC backend)
        self.vlc_instance = vlc.Instance()
        self.media_player = self.vlc_instance.media_player_new()

        # RESTORE: plain video frame (no overlay/stack)
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")

        # Time row under the video
        self.time_hms_label = QLabel("00:00:00")
        self.time_secs_label = QLabel("t = 0 s")
        self.time_row_font_pointsize = 10
        self._apply_time_font()

        time_row = QHBoxLayout()
        time_row.setContentsMargins(6, 4, 6, 4)
        time_row.addWidget(self.time_hms_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        time_row.addStretch()
        time_row.addWidget(self.time_secs_label, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.time_row_widget = QWidget()
        self.time_row_widget.setLayout(time_row)

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

        unit_beh_layout = QHBoxLayout()
        unit_beh_layout.addWidget(self.unit_selector_combo)
        unit_beh_layout.addWidget(self.behaviour_selector_combo)
        main.addLayout(unit_beh_layout)

        # PLACE: video, then time row, then raster
        main.addWidget(self.video_frame, stretch=5)
        main.addWidget(self.time_row_widget)
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
        self.spiketrains_full = None
        self.spiketrains_by_trials = None
        self.t_stop_all = None

        # Poll timer to sync UI with VLC
        self._last_length_ms = 0
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll_player)
        self.poll_timer.start()

        # Load first video
        if len(self.df_videos) > 0:
            self.load_video_and_data(0)
            # removed overlay geometry calls

    # NEW: font applier for time row
    def _apply_time_font(self):
        try:
            f = QFont()
            f.setPointSize(int(self.time_row_font_pointsize))
            self.time_hms_label.setFont(f)
            self.time_secs_label.setFont(f)
        except Exception:
            pass

    # REPLACE: overlay time updater -> time row updater
    def _format_hms(self, total_seconds: int) -> str:
        if total_seconds < 0:
            total_seconds = 0
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_time_row(self, ms: int):
        if ms < 0:
            ms = 0
        secs = ms // 1000
        self.time_hms_label.setText(self._format_hms(secs))
        self.time_secs_label.setText(f"t = {secs} s")

    def _set_video_output_handle(self):
        """Bind VLC video output to our QWidget handle."""
        wid = int(self.video_frame.winId())
        if sys.platform.startswith("linux"):
            self.media_player.set_xwindow(wid)
        elif sys.platform == "win32":
            self.media_player.set_hwnd(wid)
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(wid)

    # REMOVE: overlay event filter and geometry helpers
    # def eventFilter(...): ...
    # def _ensure_overlay_geometry(...): ...

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
        # removed overlay geometry calls

    def update_unit_selector(self):
        # Updated to use self.unit_names
        names = self.unit_names if self.unit_names is not None else []
        block = self.unit_selector_combo.blockSignals(True)
        self.unit_selector_combo.clear()
        self.unit_selector_combo.addItems(names)
        self.unit_selector_combo.blockSignals(block)

    def on_unit_selection_changed(self, idx:int):
        if len(self.unit_ids) == 0:
            return
        if idx < 0 or idx >= len(self.unit_ids):
            return
        self.selected_unit_id = self.unit_ids[idx]
        # Rebuild trials (depends on current behaviour selection)
        self.rebuild_trials_data()
        # Refresh raster
        pos_ms = int(self.media_player.get_time())
        if pos_ms < 0:
            pos_ms = 0
        self.update_raster_plot(pos_ms / 1000.0)

    def update_behaviour_selector(self):
        # Rewritten: populate combo instead of buttons
        block = self.behaviour_selector_combo.blockSignals(True)
        self.behaviour_selector_combo.clear()
        self.behaviour_selector_combo.addItem("ALL (full video)", userData="all")
        if self.behaviour_df is not None and len(self.behaviour_df) > 0:
            for _, row in self.behaviour_df.iterrows():
                text = f"{row['behaviour']} | Trials: {row['n_trials']} | Dur: {row['total_duration']:.1f}s"
                self.behaviour_selector_combo.addItem(text, userData=row['behaviour'])
        block = self.behaviour_selector_combo.blockSignals(block)
        self.behaviour_selector_combo.setCurrentIndex(0)

    def on_behaviour_combo_changed(self, idx:int):
        if idx < 0:
            return
        behaviour_name = self.behaviour_selector_combo.itemData(idx)
        if behaviour_name is None:
            behaviour_name = "all"
        self.on_behaviour_selected(behaviour_name)

    def on_behaviour_selected(self, behaviour_name):
        self.selected_behaviour_name = behaviour_name
        # Rebuild trials first, then derive playback intervals from trials
        self.rebuild_trials_data()
        self.time_intervals_list = generate_video_intervals_from_spiketrains(self.spiketrains_by_trials)
        self.set_intervals_ms(self.time_intervals_list)
        if self.media_player.is_playing() and self.intervals_ms:
            tgt = self._next_allowed_position(int(self.media_player.get_time()))
            if tgt is None:
                self.media_player.pause()
                self.playBtn.setText("▶")
            else:
                self._safe_seek(tgt)
        pos_ms = int(self.media_player.get_time())
        if pos_ms < 0:
            pos_ms = 0
        self.update_raster_plot(pos_ms / 1000.0)

    def rebuild_trials_data(self):
        """Recompute trial-aligned spikes for current behaviour & unit."""
        self.spiketrains_by_trials = None
        self.trial_relative_spikes = []
        self.trial_abs_intervals = []
        self.max_rel_right = None
        if getattr(self, 'selected_behaviour_name', "all") == "all":
            return
        if (self.behaviour_df is None or
            getattr(self, 'selected_unit_id', None) is None or
            self.spiketrains_full is None):
            return
        row = self.behaviour_df[self.behaviour_df.behaviour == self.selected_behaviour_name]
        if row.empty:
            return
        row = row.iloc[0]
        start_stop_times = row.get('start_stop_times', None)
        point_times = row.get('point_times', None)
               
        self.spiketrains_by_trials = generate_spiketrain_by_trials(
                self.spiketrains_full,
                unit_id=self.selected_unit_id,
                start_stop_times=start_stop_times,
                point_times=point_times,
                padding_beh_left=padding_beh_left,
                padding_beh_right=padding_beh_right
            )
        #print(f"DEBUG: t_start for spiketrains_by_trials: {self.spiketrains_by_trials.t_start}")
        #min_t_spiketrains = np.min([tr[0] for tr in self.spiketrains_by_trials.trains])
        #print(f"DEBUG: min t in spiketrains_by_trials: {min_t_spiketrains}")
        return None


    def update_raster_plot(self, t: float):
        if self.spiketrains_full is None:
            return

        # if all behaviours, or no unit selected, show full raster
        if getattr(self, 'selected_behaviour_name', "all") == "all"\
                or self.selected_behaviour_name is None:
            t0 = max(self.spiketrains_full.t_start, t - padding_beh_left)
            t1 = min(self.spiketrains_full.t_stop, t + padding_beh_right)
            self.raster_plot.clear()
            self.raster_plot.setXRange(t0, t1)
            spiketrain_to_plot = self.spiketrains_full
            # xyplot = spiketrain_to_plot.get_line_segments_xynans(
            #         t_start=t0,
            #         t_stop=t1),
            # print(xyplot)
            (xplot, yplot) = spiketrain_to_plot.get_line_segments_xynans(
                    t_start=t0,
                    t_stop=t1
                    #time_offset=_t_offset_spikes
            )
            self.raster_plot.setYRange(0, spiketrain_to_plot.n_units + 1)
            self.raster_plot.plot(xplot, yplot, pen=pg.mkPen('k', width=1))
            self.raster_plot.addItem(pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('g', width=2)))
            return

        # If behaviour selected, show raster for that behaviour trials!
        self.raster_plot.clear()
        spiketrain_to_plot=self.spiketrains_by_trials
        if spiketrain_to_plot is None:
            return
        n_trials = spiketrain_to_plot.n_trials
        t_left = -padding_beh_left
        t_right = padding_beh_right
        self.raster_plot.setXRange(t_left, t_right)
        self.raster_plot.setYRange(-0.5, n_trials - 0.5)
        self.raster_plot.setLabel('left', 'trial')
        tnow_idxtrial,tnow_reltimes = spiketrain_to_plot.get_trials_and_relative_times(t)
        # Add a red dot at each (tnow_reltimes, tnow_idxtrial)
        for rt, it in zip(tnow_reltimes, tnow_idxtrial):
            if rt is None or it is None:
                continue
            dot = pg.ScatterPlotItem([rt], [it], symbol='o', size=8, brush=pg.mkBrush('r'))
            self.raster_plot.addItem(dot)

        # Add horizontal duration bars behind spikes (like test_obj.py)
        if getattr(spiketrain_to_plot, 'is_start_stop_event', False):
            durations = getattr(spiketrain_to_plot, 'event_durations', None)
            if durations is not None:
                alpha = int(max(0.0, min(1.0, getattr(self, 'duration_bar_opacity', 0.5))) * 255)
                pen = pg.mkPen(color=(255, 0, 255, alpha), width=1)
                for idx, dur in enumerate(durations):
                    if dur is None:
                        continue
                    try:
                        d = float(dur)
                    except Exception:
                        continue
                    if not np.isfinite(d) or d <= 0:
                        continue
                    bar_item = self.raster_plot.plot([0.0, d], [idx, idx], pen=pen)
                    try:
                        bar_item.setZValue(-1)  # behind spikes
                    except Exception:
                        pass

        (xplot, yplot) = spiketrain_to_plot.get_line_segments_xynans(time_offset=0.0)
        spikes_item = self.raster_plot.plot(xplot, yplot, pen=pg.mkPen('k', width=1))
        try:
            spikes_item.setZValue(0)
        except Exception:
            pass
        self.raster_plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2)))

    def toggle_play(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.playBtn.setText("▶")
        else:
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
        if not self.intervals_ms:
            self.media_player.set_time(int(ms))
            return
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
            first_start, _ = self.intervals_ms[0]
            if int(self.media_player.get_time()) != first_start:
                self._safe_seek(first_start)

    def update_slider_and_plot(self, ms: int):
        if self.intervals_ms and not self._seeking:
            idx_inside = self._interval_index_for_time(ms)
            if idx_inside is None:
                idx_next = self._next_interval_index_from_time(ms)
                if idx_next is not None:
                    start_ms = self.intervals_ms[idx_next][0]
                    if ms != start_ms:
                        self._safe_seek(start_ms)
                        return
                else:
                    first_start = self.intervals_ms[0][0]
                    if ms != first_start:
                        self._safe_seek(first_start)
                        return
            else:
                s_ms, e_ms = self.intervals_ms[idx_inside]
                if ms >= e_ms:
                    nxt = idx_inside + 1
                    if nxt < len(self.intervals_ms):
                        ns = self.intervals_ms[nxt][0]
                        if ms != ns:
                            self._safe_seek(ns)
                    else:
                        fs = self.intervals_ms[0][0]
                        if ms != fs:
                            self._safe_seek(fs)
                    return
                self.current_interval_idx = idx_inside
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)

        # UPDATE: refresh time row
        self._update_time_row(ms)

        self.update_raster_plot(ms / 1000.0)

    def set_intervals_ms(self, intervals_sec: List[Tuple[float, float]]):
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
                s_ms = min(s_ms, dur)
                e_ms = min(e_ms, dur)
            if e_ms > s_ms:
                ints.append((s_ms, e_ms))
        ints.sort()
        merged = []
        for s, e in ints:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        self.intervals_ms = merged
        self.current_interval_idx = None

    def _interval_index_for_time(self, ms: int) -> int | None:
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
        for i, (s, _) in enumerate(self.intervals_ms):
            if s >= ms:
                return i
        return None

    def _next_allowed_position(self, ms: int) -> int | None:
        idx_inside = self._interval_index_for_time(ms)
        if idx_inside is not None:
            return ms
        idx_next = self._next_interval_index_from_time(ms)
        if idx_next is not None:
            return self.intervals_ms[idx_next][0]
        return self.intervals_ms[0][0] if self.intervals_ms else None

    def _safe_seek(self, ms: int):
        if int(self.media_player.get_time()) == int(ms):
            return
        self._seeking = True
        self.media_player.set_time(int(ms))
        QTimer.singleShot(0, lambda: setattr(self, '_seeking', False))

    def _jump_by_ms(self, delta_ms: int):
        cur = int(self.media_player.get_time())
        if cur < 0:
            cur = 0
        dur_raw = int(self.media_player.get_length())
        dur = dur_raw if dur_raw > 0 else None
        target = cur + int(delta_ms)
        target = max(0, target)
        if dur is not None and target > dur:
            target = dur
        if self.intervals_ms and self._interval_index_for_time(target) is None:
            nxt = self._next_allowed_position(target)
            if nxt is None:
                return
            target = nxt
        self._safe_seek(target)

    def save_current_raster(self):
        if getattr(self, 'selected_behaviour_name', 'all') != "all":
            return
        if self.spiketrains_full is None or getattr(self, 'selected_unit_id', None) is None:
            return
        try:
            pos_ms = int(self.media_player.get_time())
            if pos_ms < 0:
                pos_ms = 0
            t = pos_ms / 1000.0
            fname = f"{getattr(self,'mouse','unk')}_{getattr(self,'session','unk')}_unit{self.selected_unit_id}_t{t:.3f}"
            out_path = path_save_raster_plots / fname
            if hasattr(self.spiketrains_full, 'filter_by_units'):
                subset = self.spiketrains_full.filter_by_units([self.selected_unit_id])
            else:
                subset = self.spiketrains_full
            do_raster_at_time(subset, t0=t, t_plot=plot_dist_time_window, output_full_path=out_path)
        except Exception:
            pass
    # ------------ END ADDED METHODS ------------

    def on_mouse_session_selection_changed(self, idx):
        """Handle mouse/session combo changes."""
        self.load_video_and_data(idx)
        self.update_unit_selector()

    def load_video_and_data(self, idx):
        """Load selected video, preprocessed data, and initialize selectors."""
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

        # Reset intervals when switching video (no restriction until behaviour is selected)
        self.set_intervals_ms([])

        # Load preprocessed dict and build spiketrains
        self.data_dict = rdl.load_local_preprocessed_dict(mouse, session, local_load_path)
        time_index = self.data_dict['time_index']
        self.t_stop_all = time_index[-1] if len(time_index) else None
        self.spiketrains_full = generate_spiketrains_from_data_dict(self.data_dict)

        # Behaviours dataframe
        self.behaviour_df = generate_beh_dataframe(mouse, session, self.data_dict)

        # Units list for selector
        self.unit_ids, self.unit_names = get_all_units(mouse, session, spiketrains=self.spiketrains_full)

        # Refresh UI selectors
        self.update_unit_selector()
        self.update_behaviour_selector()

        # Initialize unit selection
        if self.unit_selector_combo.count() > 0:
            self.unit_selector_combo.setCurrentIndex(0)
            self.on_unit_selection_changed(0)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoSelectorWidget(df_videos, mice_sessions)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())