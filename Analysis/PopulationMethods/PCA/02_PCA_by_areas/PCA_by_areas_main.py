# -*- coding: utf-8 -*-
"""
📃 ./PCA/02_PCA_by_areas_main

🕰️  created on 2025-09-16

🤡 author: Dylan Festa

PyQt app that allows selection of mouse/session, area considered, bin size
then computes first 3 PC on all units of that area and plots the result as a
pyqtgraph 3D plot (GLScatterPlotItem) 

Shows the behaviours and lets the user highlight them with color.
"""
#%%
from __future__ import annotations
from time import time
from typing import List,Dict,Tuple,Union
#%%
import os,sys
from pathlib import Path


import numpy as np, pandas as pd, xarray as xr

import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains

# PCA stuff
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from PyQt5.QtCore import Qt, QTimer
# NEW: Qt widgets and WebEngine for Plotly rendering
from PyQt5.QtWidgets import QWidget,\
    QVBoxLayout, QHBoxLayout, QLabel,\
    QComboBox, QMessageBox,QApplication
from PyQt5.QtGui import QVector3D  # <-- added
# ADD: QCheckBox
from PyQt5.QtWidgets import QCheckBox

import pyqtgraph
import pyqtgraph.opengl as gl


behaviours_color_dict = {
    'pup_run': ("#78c7ff", 0),
    'pup_grab': ("#003061", 1),
    'pup_retrieve': ("#00027a", 2),
    'nesting': ('#2ca02c', 3),
    'bed_retrieve': ("#35684f", 4),
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
    'ignore': ("#9b0000", 22),
    'escape_switch': ("#ffa500", 23),
}


behaviours_labels_dict = {k:v[1] for k, v in behaviours_color_dict.items()}

#%%
animals_all = rdl.get_good_animals()
print(f"Found {len(animals_all)} animals.")
print("Animals:", animals_all)

#%%
# dictionary of list of sessions for each animal
dict_sessions = {}
for _animal in animals_all:
    sessions_for_animal = rdl.get_good_sessions(_animal)
    dict_sessions[_animal] = sessions_for_animal

# exclude sessions with things in their name
exclude_things=['test','Kilosort', 'coded','overlap' ]

for _animal in animals_all:
    dict_sessions[_animal] = [session for session in dict_sessions[_animal] if not any(thing in session for thing in exclude_things)]

#%%


def generate_mouse_session_df(session_exclude_filter: List[str]) -> pd.DataFrame:

    animals_all = rdl.get_good_animals()
    df_ret_rows = []
    for _animal in animals_all:
        sessions_for_animal = rdl.get_good_sessions(_animal)
        for _session in sessions_for_animal:
            if not any(thing in _session for thing in session_exclude_filter):
                mouse_session_str = f"{_animal} / {_session}"
                df_ret_rows.append({'mouse': _animal,
                                    'session': _session,
                                   'mouse_session': mouse_session_str})
    df_ret = pd.DataFrame(df_ret_rows)
    return df_ret

df_mouse_session = generate_mouse_session_df(session_exclude_filter=exclude_things)

#%%

Tmin_beh = 10.0 # s

#data_dict = rdl.load_preprocessed_dict(mouse, session)

def get_behaviours_df(data_dict) -> pd.DataFrame:
    beh = data_dict.get('behaviours', None)
    if beh is None:
        raise ValueError("No 'behaviours' key found in data_dict.")

    return beh

def generate_beh_df(the_mouse:str,the_session:str,the_data_dict:Dict):
    beh_df = rdl.convert_to_behaviour_timestamps(
        the_mouse, the_session, the_data_dict['behaviour'])
    # error if behaviour not in behaviours_color_dict
    for b in beh_df['behaviour'].values:
        if b not in behaviours_color_dict:
            raise ValueError(f"Behaviour {b} not in behaviours_color_dict.")
    return beh_df

def get_beh_list(beh_df:pd.DataFrame) -> List[str]:
    beh_df_duration= beh_df.query("total_duration >= @Tmin_beh").copy()
    return beh_df_duration['behaviour'].unique().tolist()

def get_spiketrains(data_dict) -> SpikeTrains:
    """
    Load preprocessed data and wrap it into a SpikeTrains container.
    """

    time_index = data_dict['time_index']
    spike_times = data_dict['spike_times']
    cluster_index = data_dict['cluster_index']
    region_index = data_dict['region_index']
    spiketrains = SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0,
                                t_start=0.0,t_stop=time_index[-1])
    # ONLY KEEP PAG!
    return spiketrains.filter_by_unit_location('PAG')


def get_area_list(spiketrains: SpikeTrains) -> List[str]:
    """
    Returns a sorted list of unique brain areas from the SpikeTrains object.
    """
    unique_areas = np.unique(spiketrains.unit_location)
    return sorted(unique_areas.tolist())

def get_area_counts(spiketrains: SpikeTrains) -> Dict[str, int]:
    """
    Returns a dict mapping area -> number of units for that area.
    """
    areas, counts = np.unique(spiketrains.unit_location, return_counts=True)
    return {str(a): int(c) for a, c in zip(areas, counts)}

def compute_PCAs(spiketrains: SpikeTrains, dt: float, area_filter: Union[str, None]=None) -> Dict[str, xr.DataArray]:
    """
    Filters spiketrains by area (if provided), then applies binning operation
    finally computes the PCA of the spike counts and returns PCA results as xarray DataArrays,
    with dimensions (time, component).
    """
    
    if (area_filter is not None):
        spiketrains = spiketrains.filter_by_unit_location(area_filter)

    # assert we have some units left
    if spiketrains.n_units == 0:
        raise ValueError(f"No units found in area '{area_filter}'. Wrong area name?")
    # Bin the data
    binned_data = pre.do_binning_operation(spiketrains, 'count', dt=dt,
                                          t_start=spiketrains.t_start,
                                          t_stop=spiketrains.t_stop)

    # Define the processing pipeline
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=3)
    )

    # Fit-transform the pipeline
    binned_data_lagged_pca = pipe.fit_transform(binned_data.values)

    # Convert to xarray DataArray
    pca_xarray = xr.DataArray(
        binned_data_lagged_pca,
        dims=['time', 'component'],
        coords={'time': binned_data.coords['time_bin_center'].values}
    )

    return pca_xarray

#%%

# mouse_test = 'afm16924'
# session_test = '240525'
# data_dict_test = rdl.load_preprocessed_dict(mouse_test, session_test)
# spiketrains_test = get_spiketrains(data_dict_test)
# print(f"Loaded spiketrains with {spiketrains_test.n_units} units")

# #%%
# dt_test = 100E-3
# beh_df_test = generate_beh_df(mouse_test, session_test, data_dict_test)

# beh_list_test = get_beh_list(beh_df_test)


# beh_df_test_less = beh_df_test.query("behaviour in @beh_list_test").copy()
# behaviours_labels_dict_less = {k:v for k, v in behaviours_labels_dict.items() if k in beh_list_test}

# beh_labels_test = rdl.generate_behaviour_labels_inclusive(beh_df_test_less,\
#                             0.0,data_dict_test['time_index'][-1],\
#                             dt_test,behaviour_labels_dict=behaviours_labels_dict_less)

# #%%
# spikecounts_test = pre.do_binning_operation(spiketrains_test, 'count', dt=dt_test,
#                                           t_start=spiketrains_test.t_start,
#                                           t_stop=spiketrains_test.t_stop)
    
#%% 
    
class PCAVisualizerWidget(QWidget):
    """
    PyQt5 widget to:
    - Select mouse/session (labels from df_mouse_session['mouse_session'])
    - Select dt among [20e-3, 50e-3, 100e-3, 200e-3]
    - Select area from get_area_list(self.spiketrains_full)
    - Compute and display 3D PCA scatter with pyqtgraph GLScatterPlotItem
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_dict = None
        self.spiketrains_full: Union[SpikeTrains, None] = None
        # NEW: behaviour-related state
        self.beh_df: Union[pd.DataFrame, None] = None
        self.beh_list: Union[List[str], None] = None
        self.beh_labels: Union[xr.DataArray, None] = None
        # ADD: storage for PCA/labels and behaviour checkbox state
        self.pca_arr: Union[np.ndarray, None] = None
        self.beh_checkboxes: Dict[int, QCheckBox] = {}
        self.index_to_beh: Dict[int, str] = {}

        # UI
        main_layout = QVBoxLayout(self)
        controls = QHBoxLayout()

        # Mouse/session selector
        controls.addWidget(QLabel("Mouse/Session:", self))
        self.mouse_session_cb = QComboBox(self)
        # Populate with labels from df_mouse_session mouse_session column
        # Store (mouse, session) as itemData
        for _, row in df_mouse_session.iterrows():
            self.mouse_session_cb.addItem(row["mouse_session"], (row["mouse"], row["session"]))
        self.mouse_session_cb.currentIndexChanged.connect(self.on_mouse_session_changed)
        # Default to afm16924 / 240525
        self.mouse_session_cb.blockSignals(True)
        for i in range(self.mouse_session_cb.count()):
            if self.mouse_session_cb.itemData(i) == ("afm16924", "240525"):
                self.mouse_session_cb.setCurrentIndex(i)
                break
        self.mouse_session_cb.blockSignals(False)
        controls.addWidget(self.mouse_session_cb)

        # dt selector
        controls.addWidget(QLabel("dt (s):", self))
        self.dt_cb = QComboBox(self)
        for dt in [20e-3, 50e-3, 100e-3, 200e-3]:
            self.dt_cb.addItem(f"{dt}", dt)
        self.dt_cb.currentIndexChanged.connect(self.on_params_changed)
        # Default dt to 0.1 s
        idx_dt = self.dt_cb.findData(0.1)
        if idx_dt != -1:
            self.dt_cb.setCurrentIndex(idx_dt)
        controls.addWidget(self.dt_cb)

        # area selector (filled after loading spiketrains)
        controls.addWidget(QLabel("Area:", self))
        self.area_cb = QComboBox(self)
        self.area_cb.currentIndexChanged.connect(self.on_params_changed)
        controls.addWidget(self.area_cb)

        main_layout.addLayout(controls)

        # ADD: behaviour filter row (checkboxes under the selector)
        beh_row = QHBoxLayout()
        beh_row.addWidget(QLabel("Show behaviours:", self))
        self.beh_checks_layout = QHBoxLayout()
        beh_row.addLayout(self.beh_checks_layout)
        main_layout.addLayout(beh_row)

        # pyqtgraph GLViewWidget for 3D scatter
        self.gl_view = gl.GLViewWidget(self)
        self.gl_view.setBackgroundColor('w')  # white background
        self.gl_view.opts['distance'] = 20
        # Dot color RGBA (darker grey, opaque for contrast on white)
        self.dot_color = np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32)
        main_layout.addWidget(self.gl_view)

        # Store scatter items for updating/removal
        self.scatter_items: List[gl.GLScatterPlotItem] = []  # replaces single-item storage

        # Trigger initial load if available
        if self.mouse_session_cb.count() > 0:
            self.on_mouse_session_changed(self.mouse_session_cb.currentIndex())

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        """Convert '#RRGGBB' to normalized RGBA tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, float(alpha))

    def on_mouse_session_changed(self, idx: int):
        data = self.mouse_session_cb.itemData(idx)
        if not data:
            return
        mouse, session = data
        try:
            # Load data and build spiketrains
            print(f"Loading data for {mouse}/{session}...")
            self.data_dict = rdl.load_preprocessed_dict(mouse, session)
            print(f"Data loaded, building spiketrains...")
            self.spiketrains_full = get_spiketrains(self.data_dict)
            print(f"Loaded spiketrains with {self.spiketrains_full.n_units} units!")

            # Populate area list
            # areas = get_area_list(self.spiketrains_full)
            area_counts = get_area_counts(self.spiketrains_full)
            total_units = self.spiketrains_full.n_units
            self.area_cb.blockSignals(True)
            self.area_cb.clear()
            self.area_cb.addItem(f"All areas ({total_units})", None)  # Option for no filtering
            for area in sorted(area_counts.keys()):
                self.area_cb.addItem(f"{area} ({area_counts[area]})", area)
            self.area_cb.blockSignals(False)

            # NEW: compute and store behaviours df/list for this selection
            self.beh_df = generate_beh_df(mouse, session, self.data_dict)
            self.beh_list = get_beh_list(self.beh_df)

            # ADD: rebuild behaviour checkboxes for this session
            self.rebuild_behaviour_checkboxes()

            # Compute PCA/labels and plot with current params
            self.compute_pca_and_labels()
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Failed to load {mouse}/{session}:\n{e}")

    def on_params_changed(self, _idx: int):
        # Only recompute PCA when dt/area changes
        self.compute_pca_and_labels()

    # ADD: build/update behaviour checkboxes based on current beh_list
    def rebuild_behaviour_checkboxes(self):
        # clear existing
        while self.beh_checks_layout.count():
            item = self.beh_checks_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.beh_checkboxes.clear()
        # always add 'none' for unlabeled (-1)
        cb_none = QCheckBox("none", self)
        cb_none.setChecked(True)
        cb_none.stateChanged.connect(self.on_behaviour_filter_changed)
        self.beh_checkboxes[-1] = cb_none
        self.beh_checks_layout.addWidget(cb_none)
        # add one checkbox per behaviour in list (sorted by label index)
        if self.beh_list:
            for beh_name in sorted(self.beh_list, key=lambda n: behaviours_labels_dict[n]):
                beh_idx = behaviours_labels_dict[beh_name]
                cb = QCheckBox(beh_name, self)
                cb.setChecked(True)
                cb.stateChanged.connect(self.on_behaviour_filter_changed)
                self.beh_checkboxes[beh_idx] = cb
                self.beh_checks_layout.addWidget(cb)
        # OPTIONAL: try to update counts if already available
        self.update_behaviour_checkbox_counts()

    # ADD: update checkbox labels with counts, e.g. "pup_run (1200)"
    def update_behaviour_checkbox_counts(self):
        if self.beh_labels is None:
            return
        labels_np = self.beh_labels.values.astype(int)

        # unlabeled (-1)
        if -1 in self.beh_checkboxes:
            n_unlabeled = int(np.sum(labels_np == -1))
            self.beh_checkboxes[-1].setText(f"none ({n_unlabeled})")

        # behaviours
        for beh_idx, cb in self.beh_checkboxes.items():
            if beh_idx < 0:
                continue
            beh_name = self.index_to_beh.get(beh_idx, None)
            if beh_name is None:
                continue
            n = int(np.sum(labels_np == beh_idx))
            cb.setText(f"{beh_name} ({n})")

    # ADD: handle checkbox toggles -> replot only
    def on_behaviour_filter_changed(self, _state: int):
        self.plot_only()

    # ADD: helper to gather selected label indices (-1 for unlabeled)
    def get_selected_label_indices(self) -> List[int]:
        return [idx for idx, cb in self.beh_checkboxes.items() if cb.isChecked()]

    # ADD: split compute and plot
    def compute_pca_and_labels(self):
        if self.spiketrains_full is None or self.area_cb.count() == 0:
            return
        try:
            dt = float(self.dt_cb.currentData())
            area = self.area_cb.currentData()
            print(f"Computing PCA with dt={dt}s, area={area}...")
            pca = compute_PCAs(self.spiketrains_full, dt=dt, area_filter=area)
            self.pca_arr = pca.values  # shape (time, 3)
            print("PCA computed, preparing behaviour labels...")

            # compute behaviour labels aligned to PCA time bins
            if self.beh_df is None:
                self.beh_labels = xr.DataArray(np.full(self.pca_arr.shape[0], -1, dtype=int), dims=['time'])
            else:
                beh_df_less = self.beh_df.query("behaviour in @self.beh_list").copy() if self.beh_list else self.beh_df.iloc[0:0].copy()
                if beh_df_less.empty:
                    self.beh_labels = xr.DataArray(np.full(self.pca_arr.shape[0], -1, dtype=int), dims=['time'])
                else:
                    behaviours_labels_dict_less = {k: v for k, v in behaviours_labels_dict.items() if k in self.beh_list}
                    t0 = 0.0
                    t1 = float(self.data_dict['time_index'][-1])
                    labels_xr = rdl.generate_behaviour_labels_inclusive(
                        beh_df_less, t0, t1, dt,
                        behaviour_labels_dict=behaviours_labels_dict_less)
                    if self.pca_arr.shape[0] != labels_xr.shape[0]:
                        raise ValueError(f"Length mismatch between PCA ({self.pca_arr.shape[0]}) and behaviour labels ({labels_xr.shape[0]}).")
                    self.beh_labels = labels_xr

            # build reverse map for present behaviours
            if self.beh_list:
                behaviours_labels_dict_less = {k: v for k, v in behaviours_labels_dict.items() if k in self.beh_list}
                self.index_to_beh = {v: k for k, v in behaviours_labels_dict_less.items()}
            else:
                self.index_to_beh = {}

            # UPDATE: refresh checkbox labels with counts
            self.update_behaviour_checkbox_counts()

            # Update title and camera based on data extent
            self.gl_view.setWindowTitle(f"3D PCA - {self.area_cb.currentText()}, dt={dt}s")
            center = self.pca_arr.mean(axis=0)
            self.gl_view.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
            extent = self.pca_arr.max(axis=0) - self.pca_arr.min(axis=0)
            radius = max(1.0, float(np.linalg.norm(extent)) * 0.5)
            self.gl_view.opts['distance'] = radius * 2.0

            # Ensure grid exists and is opaque/black
            if not hasattr(self, 'grid_items'):
                self.grid_items = []
                for i, axis in enumerate(['x', 'y', 'z']):
                    grid = gl.GLGridItem()
                    grid.setSize(20, 20)
                    grid.setSpacing(3, 3)
                    grid.setGLOptions('translucent')
                    grid.setColor('k') # black
                    if axis == 'x':
                        grid.rotate(90, 0, 1, 0)
                    elif axis == 'y':
                        grid.rotate(90, 1, 0, 0)
                    self.gl_view.addItem(grid)
                    self.grid_items.append(grid)
            else:
                for grid in self.grid_items:
                    grid.setColor('k')  # black
                    grid.setGLOptions('translucent')

            # draw according to current selection without recomputing PCA
            self.plot_only()
        except Exception as e:
            QMessageBox.warning(self, "PCA error", f"Failed to compute PCA/labels:\n{e}")

    # ADD: plotting-only routine driven by checkbox selection
    def plot_only(self):
        if self.pca_arr is None or self.beh_labels is None:
            return
        try:
            # remove old scatter items
            for it in getattr(self, 'scatter_items', []):
                self.gl_view.removeItem(it)
            self.scatter_items = []

            labels_np = self.beh_labels.values.astype(int)
            sel_indices = set(self.get_selected_label_indices())

            # plot unlabeled if selected (-1)
            if -1 in sel_indices:
                mask = labels_np == -1
                if np.any(mask):
                    item = gl.GLScatterPlotItem(
                        pos=self.pca_arr[mask].astype(np.float32),
                        color=(0.5, 0.5, 0.5, 1.0),
                        size=1,
                        pxMode=True
                    )
                    item.setGLOptions('opaque')
                    self.gl_view.addItem(item)
                    self.scatter_items.append(item)

            # plot selected behaviours
            for beh_idx in sorted(i for i in sel_indices if i >= 0):
                beh_name = self.index_to_beh.get(beh_idx, None)
                if beh_name is None:
                    continue
                mask = labels_np == beh_idx
                if not np.any(mask):
                    continue
                #print(f"Plotting behaviour '{beh_name}', index {beh_idx}, count {np.sum(mask)}")
                color_rgba = self._hex_to_rgba(behaviours_color_dict[beh_name][0], alpha=1.0)
                #print(f"Color RGBA: {color_rgba}")
                item = gl.GLScatterPlotItem(
                    pos=self.pca_arr[mask].astype(np.float32),
                    color=color_rgba,
                    size=4,
                    pxMode=True
                )
                item.setGLOptions('opaque')
                self.gl_view.addItem(item)
                self.scatter_items.append(item)
        except Exception as e:
            QMessageBox.warning(self, "Plot error", f"Failed to plot PCA points:\n{e}")

    # OLD: compute_and_plot now delegates to split methods (kept for compatibility)
    def compute_and_plot(self):
        self.compute_pca_and_labels()
# %%

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PCAVisualizerWidget()
    window.setWindowTitle("PCA by Areas Visualizer")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())