# -*- coding: utf-8 -*-
"""
📃 ./PCA/02_PCA_by_areas_main

🕰️  created on 2025-09-16

🤡 author: Dylan Festa

PyQt app that allows selection of mouse/session, area considered, bin size
then computes first 3 PC on all units of that area and plots the result as a plotly
3D plot. 

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

# plot with plotly
import plotly.express as px
import plotly.graph_objects as go

from PyQt5.QtCore import Qt, QTimer
# NEW: Qt widgets and WebEngine for Plotly rendering
from PyQt5.QtWidgets import QWidget,\
    QVBoxLayout, QHBoxLayout, QLabel,\
    QComboBox, QMessageBox,QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView



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


#data_dict = rdl.load_preprocessed_dict(mouse, session)

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
    
class PCAVisualizerWidget(QWidget):
    """
    PyQt5 widget to:
    - Select mouse/session (labels from df_mouse_session['mouse_session'])
    - Select dt among [20e-3, 50e-3, 100e-3, 200e-3]
    - Select area from get_area_list(self.spiketrains_full)
    - Compute and display 3D PCA scatter with Plotly
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_dict = None
        self.spiketrains_full: Union[SpikeTrains, None] = None

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
        controls.addWidget(self.mouse_session_cb)

        # dt selector
        controls.addWidget(QLabel("dt (s):", self))
        self.dt_cb = QComboBox(self)
        for dt in [20e-3, 50e-3, 100e-3, 200e-3]:
            self.dt_cb.addItem(f"{dt}", dt)
        self.dt_cb.currentIndexChanged.connect(self.on_params_changed)
        controls.addWidget(self.dt_cb)

        # area selector (filled after loading spiketrains)
        controls.addWidget(QLabel("Area:", self))
        self.area_cb = QComboBox(self)
        self.area_cb.currentIndexChanged.connect(self.on_params_changed)
        controls.addWidget(self.area_cb)

        main_layout.addLayout(controls)

        # Plotly view
        self.web_view = QWebEngineView(self)
        main_layout.addWidget(self.web_view)

        # Trigger initial load if available
        if self.mouse_session_cb.count() > 0:
            self.on_mouse_session_changed(self.mouse_session_cb.currentIndex())

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
            areas = get_area_list(self.spiketrains_full)
            self.area_cb.blockSignals(True)
            self.area_cb.clear()
            self.area_cb.addItem("All areas", None)  # Option for no filtering
            for area in areas:
                self.area_cb.addItem(area)
            self.area_cb.blockSignals(False)

            # Compute and plot with current params
            self.compute_and_plot()
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Failed to load {mouse}/{session}:\n{e}")

    def on_params_changed(self, _idx: int):
        self.compute_and_plot()

    def compute_and_plot(self):
        if self.spiketrains_full is None:
            return
        if self.area_cb.count() == 0:
            return
        try:
            dt = self.dt_cb.currentData()
            area = self.area_cb.currentText() or None
            if area == "All areas":
                area = None
            print(f"Computing PCA with dt={dt}s, area={area}...")
            pca = compute_PCAs(self.spiketrains_full, dt=float(dt), area_filter=area)
            arr = pca.values  # shape (time, 3)
            print(f"PCA computed, plotting...")
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        z=arr[:, 2],
                        mode="markers",
                        marker=dict(size=2, opacity=0.7)
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                title=f"3D PCA - Area: {area}, dt={dt}s"
            )
            print("...still plotting...")
            html = fig.to_html(include_plotlyjs="cdn", full_html=False)
            print("Plot completed!")
            self.web_view.setHtml(html)
        except Exception as e:
            QMessageBox.warning(self, "PCA error", f"Failed to compute/plot PCA:\n{e}")
# %%

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PCAVisualizerWidget()
    window.setWindowTitle("PCA by Areas Visualizer")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())