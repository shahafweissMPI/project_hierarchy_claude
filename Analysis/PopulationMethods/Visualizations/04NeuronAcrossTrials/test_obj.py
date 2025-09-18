# -*- coding: utf-8 -*-
"""
📃 ./04NeuronAcrossTrials/test_obj.py

🕰️ created on 2025-09-12

🤡 author: Dylan Festa

Tests function that creates a SpikeTrainByTrial object and then plots it 
as a raster plot. 
"""


#%%
from __future__ import annotations
from typing import List,Dict,Tuple

import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import tempfile
import read_data_light as rdl
import preprocess as pre
import pickle

# This is to save the plots
import plotly.express as px
import plotly.graph_objects as go

from preprocess import SpikeTrains, SpikeTrainByTrials

# date string in the format YYYYMMDD
import time,datetime

#%%
# get mouse/session combinations to save


movies_path = Path.home() / 'Videos800Compressed'

assert movies_path.exists(), f"Movies path {movies_path} does not exist."


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


session_filter = ['test','Kilosort', 'coded','overlap','raw']
df_videos = generate_mouse_session_df(movies_path, session_filter=session_filter)


#%%


mouse_test, session_test = df_videos.iloc[0][['mouse','session']].to_list()

print(f"Testing with mouse {mouse_test}, session {session_test}")

#%%
# get behaviours, all of them
data_dict = rdl.load_preprocessed_dict(mouse_test, session_test)
beh_raw = data_dict['behaviour']
beh_df = rdl.convert_to_behaviour_timestamps(mouse_test, session_test, data_dict['behaviour'])
# %%
# test that behaviours are the same in both
beh1 = beh_raw['behaviours'].unique()
beh2 = beh_df['behaviour'].unique()
if set(beh1) != set(beh2):
    raise ValueError("Behaviours in raw and df do not match")
#%%

beh_test = 'loom'

is_beh_start_stop = beh_df.query("behaviour == @beh_test")['is_start_stop'].iloc[0]

if is_beh_start_stop:
    start_times_test = [ x[0] for x in  beh_df.query("behaviour == @beh_test")['start_stop_times'].iloc[0]]
    end_times_test = [ x[1] for x in  beh_df.query("behaviour == @beh_test")['start_stop_times'].iloc[0]]
    start_times_test = np.array(start_times_test)
    end_times_test = np.array(end_times_test)

    # print min and max duration
    durations = end_times_test - start_times_test
    print(f"Min duration: {durations.min():.2f} s, Max duration: {durations.max():.2f} s")
else:
    start_times_test = beh_df.query("behaviour == @beh_test")['point_times'].iloc[0]
    start_times_test = np.array(start_times_test)
    end_times_test = None

# %%
# create spiketrain and check unit list and rates too


# unpack the data
spike_times =data_dict['spike_times']
time_index =data_dict['time_index']
cluster_index =data_dict['cluster_index']
region_index =data_dict['region_index']

spiketrains=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=0.0,
                                t_stop=time_index[-1])
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')
n_units = spiketrains.n_units
the_units = spiketrains.units
the_unit_locations = spiketrains.unit_location
the_rates = [len(tr)/ (spiketrains.t_stop - spiketrains.t_start) for tr in spiketrains.trains]
print(f"Number of PAG units: {n_units}.")

# print 10 units with highest firing rates
top_n = 10
top_indices = np.argsort(the_rates)[-top_n:][::-1]
print(f"Top {top_n} units by firing rate:")
for idx in top_indices:
    print(f"  Unit {the_units[idx]} (Location: {the_unit_locations[idx]}): {the_rates[idx]:.2f} Hz")
# %%
unit_test = 688
# create SpikeTrainByTrials object
trainbytrials = SpikeTrainByTrials.from_spike_trains(
    unit_id=unit_test,
    spike_trains=spiketrains,
    time_event_starts=start_times_test,
    time_event_stops=end_times_test,
    t_pad_left=3.0,
    t_pad_right=10.0
)

#%%


x_plot,y_plot = trainbytrials.get_line_segments_xynans(
                            t_start=trainbytrials.t_start,t_stop=trainbytrials.t_stop,time_offset=0.0)
n_trials = trainbytrials.n_trials
the_trials = trainbytrials.units
durations = trainbytrials.event_durations

fig = go.Figure()
# Duration as lines behind the spikes
duration_line_width = 1
duration_opacity = 0.5

if trainbytrials.is_start_stop_event:
    for idx, dur in enumerate(durations):
        fig.add_shape(
            type='line',
            x0=0, x1=dur,
            y0=idx, y1=idx,
            line=dict(color='magenta', width=duration_line_width),
            layer='below',  # Draw behind spikes
            name=f'duration_{idx}',
            opacity=duration_opacity
        )

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
    type='line', x0=0, x1=0, y0=-0.5, y1=n_trials-0.5,
    line=dict(color='green', width=1)
)
# Axes & grid (horizontal only every unit)
fig.update_xaxes(title=f'time relative to start (s)', showgrid=False, zeroline=False,
                 range=[trainbytrials.t_start, trainbytrials.t_stop])
fig.update_yaxes(title='trial idx', showgrid=True, gridcolor='lightgrey', dtick=1,
                 range=[-0.5, n_trials-0.5], autorange=False, zeroline=False,
                 tickmode='array', tickvals=list(range(n_trials)), ticktext=[ str(u) for u in the_trials])
# Dynamic height proportional to number of units (≈12 px per unit)
fig.update_layout(height=max(200, int(n_trials*20)), margin=dict(l=100,r=25,t=40,b=50),
                  paper_bgcolor='white', plot_bgcolor='white')
fig.show()

#%%