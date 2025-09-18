# -*- coding: utf-8 -*-
"""
📃 ./test_generate_raster.py

🕰️ created on 2025-09-10

🤡 author: Dylan Festa

Test the generation of a raster plot of a portion of time for selected neurons
"""
#%%
from __future__ import annotations
from time import time
from typing import List,Dict,Tuple,Union
#%%
import os,sys


import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import tempfile
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

local_load_path = Path(tempfile.gettempdir()) / "TempDatadictSaves"
# check that exists
if not local_load_path.exists():
    raise FileNotFoundError(f"Local load path {local_load_path} does not exist. Please run save_data_locally.py first.")

import plotly.express as px
import plotly.graph_objects as go

#%%

path_this_file = Path(__file__).resolve()
path_data = path_this_file.parent / "local_outputs"
path_plots_output = path_this_file.parent / "local_outputs_plots"
# create if not exists
path_plots_output.mkdir(parents=True, exist_ok=True)

if not path_data.exists():
    raise FileNotFoundError(f"Data path {path_data} does not exist")

path_unit_selection = path_data / "02_suggested_units.pkl"
df_suggested_units = pd.read_pickle(path_unit_selection)



#%%

def do_raster_at_time(spiketrains:SpikeTrains,t0:float, t_plot:float,
                          output_full_path:Union[str,Path],):
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
    # Dynamic height proportional to number of units (≈12 px per unit)
    fig.update_layout(height=max(200, int(n_units*12)), margin=dict(l=100,r=25,t=40,b=50),
                      paper_bgcolor='white', plot_bgcolor='white')
    # Ensure parent folder exists & save
    output_full_path = Path(output_full_path)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    # export as HTML, PNG 1000 px wide and as pdf
    fig.write_html(str(output_full_path.with_suffix('.html')))
    fig.write_image(str(output_full_path.with_suffix('.png')), width=1000)
    fig.write_image(str(output_full_path.with_suffix('.pdf')))
    return fig

#%%
mouse_test = 'afm16924'
session_test = '240525'


data_dict_test = rdl.load_preprocessed_dict(mouse_test,session_test)
# unpack the data
spike_times = data_dict_test['spike_times']
time_index = data_dict_test['time_index']
cluster_index = data_dict_test['cluster_index']
region_index = data_dict_test['region_index']

spiketrains_test=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=0.0,
                                t_stop=time_index[-1])
#%%
# select only PAG units
spiketrains_test_pag = spiketrains_test.filter_by_unit_location('PAG')
spiketrains_test_few = spiketrains_test_pag.filter_by_units([352,361,366,391])

#%%
raster_test = do_raster_at_time(
    spiketrains=spiketrains_test_few,
    t0=500.0,
    t_plot=10.0,
    output_full_path=path_plots_output / "eraseme")

#%%
raster_test.show()

# %%
