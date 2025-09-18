#-*- coding: utf-8 -*-
"""
Created on 2025-08-05
Author: Dylan Festa

Example of plot of single raster.

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


spiketrains=SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG').generate_sorted_by_rate()


#%%

xplot,yplot=spiketrains.get_line_segments_xynans(t_start=500.0,t_stop=1000.0,
                                            time_offset=750.0)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Temperature vs time plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        # black pen for the plot
        pen = pg.mkPen(color='k', width=2)
        self.plot_graph.plot(xplot, yplot, pen=pen)
        # Add vertical line at x=0
        vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot_graph.addItem(vline)

app = QtWidgets.QApplication([])
main = MainWindow()
main.show()
app.exec()

#%%