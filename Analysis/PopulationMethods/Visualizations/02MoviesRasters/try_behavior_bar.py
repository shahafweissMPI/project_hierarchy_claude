#-*- coding: utf-8 -*-
"""
Created on 2025-08-05
Author: Dylan Festa

Example of plot of behavior start/stop

"""
#%%
from PyQt5 import QtWidgets

import numpy as np, pandas as pd, xarray as xr

import numpy as np
import pyqtgraph as pg
import time


import read_data_light as rdl
from preprocess import SpikeTrains

animal = 'afm16924'
session = '240524'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(animal, session)
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


behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setMinimumSize(800, 200)
        #self.plot_graph.setMaximumSize(800, 200)
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")

        # Set up the time axis
        self.plot_graph.setXRange(t_start_all, t_stop_all)

        # Plot rectangles for each interval in start_stop_times as filled rectangles between y=0 and y=1
        for start, stop in start_stop_times:
            x = [start, stop, stop, start]
            y = [0, 0, 1, 1]
            fill_brush = pg.mkBrush(100, 100, 255, 50)
            rect_item = pg.PlotDataItem(x + [x[0]], y + [y[0]], pen=None, brush=fill_brush, fillLevel=0)
            self.plot_graph.addItem(rect_item)

# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()

#         # Temperature vs time plot
#         self.plot_graph = pg.PlotWidget()
#         self.setCentralWidget(self.plot_graph)
#         self.plot_graph.setBackground("w")
#         # HELP ME

app = QtWidgets.QApplication([])
main = MainWindow()
main.show()
app.exec()

#%%