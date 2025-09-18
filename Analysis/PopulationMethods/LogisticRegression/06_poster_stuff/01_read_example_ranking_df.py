# -*- coding: utf-8 -*-
"""
Created on 2025-08-21

@author: Dylan Festa

Reads data from `fit_and_save` and builds a Pandas dataframe with the ranking of each unit.
Then it saves it in some temporary file.
"""
#%%
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go

the_mouse='afm16924'

read_file = os.path.join(os.path.dirname(__file__), "local_outputs", f"df_ranking_{the_mouse}.pkl")

if not os.path.exists(read_file):
    raise FileNotFoundError(f"File {read_file} does not exist. Please run `fit_and_save.py` first.")

with open(read_file, "rb") as f:
    df_ranking = pickle.load(f)
#%%

the_sessions = df_ranking['session'].unique()
print(f"Found {len(the_sessions)} sessions: {the_sessions}")

#%% for each session, print number of units, and ratio for each location

for session in the_sessions:
    print(f"Session: {session}")
    df_session = df_ranking[df_ranking['session'] == session]
    df_unitsonly = df_session[['unit', 'unit_location']].drop_duplicates()
    n_units = len(df_unitsonly)
    print(f"Number of PAG units: {n_units}")

    # Calculate ratio for each location
    location_counts = df_unitsonly['unit_location'].value_counts()
    location_ratios = location_counts / n_units if n_units > 0 else 0
    print(f"Location ratios (all neurons):\n{location_ratios}\n")

#%%
# now, select only ranks between 0 and 9 (10 most important neurons per behavior) and count again
max_k = 10
for session in the_sessions:
    print(f"Session: {session}")
    # use query instead
    df_session = df_ranking.query("session == @session and rank < @max_k")
    df_unitsonly = df_session[['unit', 'unit_location']].drop_duplicates()
    n_units = len(df_unitsonly)
    print(f"Number of PAG units with ranking below {max_k}: {n_units}")

    # Calculate ratio for each location
    location_counts = df_unitsonly['unit_location'].value_counts()
    location_ratios = location_counts / n_units if n_units > 0 else 0
    print(f"Location ratios (all neurons):\n{location_ratios}\n")

# %%
# given a session, and a given rank level, print
# the number of units that are at that rank level or below for only one behaviour,
# those that appear in two behaviours, three behaviours, etc etc.

def print_rank_counts(session: str, rank_level: int):
    filtered = df_ranking.query("session == @session and rank < @rank_level")
    # count how many times each unit appears
    unit_counts = filtered["unit"].value_counts()
    # now invert, for each value count, from the highest, print how many units there are
    inverted_counts = unit_counts.value_counts().sort_index(ascending=False)
    # print on screen, starting from highest
    print(f"Rank counts for session {session}")
    n_units = df_ranking["unit"].nunique()
    print(f"Total unique units: {n_units}")
    n_unique_units_highrank = filtered["unit"].nunique()
    print(f"Total units below rank {rank_level} : {n_unique_units_highrank}")
    for count, num_units in inverted_counts.items():
        # singular or plural
        beh_string = "behaviour" if count == 1 else "behaviours"
        print(f"In {count} {beh_string}: {num_units} units")
    print("")
    return None

print_rank_counts(the_sessions[0], max_k)
print_rank_counts(the_sessions[1], max_k)

# %%
# finally, for each session, each behavior, simply print the high-rank units and their locations
max_k=10

def print_high_rank_units(df: pd.DataFrame, max_rank: int):
    cols_needed = {'session', 'behaviour', 'unit', 'unit_location', 'rank'}
    missing = cols_needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_filtered = (
        df.query("rank < @max_rank")
          .sort_values(['session', 'behaviour', 'unit'])
    )

    # Group and print
    for (session, behavior), g in df_filtered.groupby(['session', 'behaviour'], sort=True):
        print(f"Session: {session} | Behavior: {behavior}")
        # Preserve first occurrence order of units
        seen = set()
        for _, row in g.iterrows():
            u = row['unit']
            if u in seen:
                continue
            seen.add(u)
            print(f"{row['unit']}\t{row['unit_location']}")
        print("")  # blank line between groups

print_high_rank_units(df_ranking, max_k)



# %%
# for each session, behavior, location, plot relative contribution to behaviours versus relative presence in dataset
# plots are with plotly
def plot_relative_contribution(df_data: pd.DataFrame,
                               session: str,
                               behaviour: str,
                               max_rank: int=10):

    df_filtered_allranks = df_data.query("session == @session and behaviour == @behaviour")
    df_filtered_lowrank = df_filtered_allranks.query("rank < @max_rank")

    # Aggregate unique units per location
    def agg(df):
        if df.empty:
            return pd.DataFrame(columns=['unit_location', 'unit'])
        out = (df.groupby('unit_location')
                 .agg(unit=('unit', 'nunique'))
                 .reset_index())
        return out

    df_all = agg(df_filtered_allranks)
    df_low = agg(df_filtered_lowrank)

    # Prevent division by zero
    total_all = df_all['unit'].sum()
    total_low = df_low['unit'].sum()
    df_all['relative_contribution'] = df_all['unit'] / total_all if total_all else 0
    df_low['relative_contribution'] = df_low['unit'] / total_low if total_low else 0

    # Ensure both have same set of locations
    all_locations = sorted(set(df_all['unit_location']) | set(df_low['unit_location']))
    df_all = df_all.set_index('unit_location').reindex(all_locations, fill_value=0).reset_index()
    df_low = df_low.set_index('unit_location').reindex(all_locations, fill_value=0).reset_index()

    df_all['RankGroup'] = 'All Ranks'
    df_low['RankGroup'] = f'Top < {max_rank}'
    df_plot = pd.concat([df_all, df_low], ignore_index=True)

    fig = px.bar(
        df_plot,
        x='unit_location',
        y='relative_contribution',
        color='RankGroup',
        barmode='group',
        title=f"Relative Contribution of Locations for {session}, behaviour: {behaviour}",
        labels={'unit_location': 'Location', 'relative_contribution': 'Relative Contribution'},
        color_discrete_map={
            'All Ranks': "#6d6d6d",
            f'Top < {max_rank}': '#000000'
        },
        text='unit'  # absolute counts
    )
    # Make black bars fully opaque, light gray slightly softer
    for tr in fig.data:
        if tr.name == 'All Ranks':
            tr.marker.update(opacity=0.85)
        else:
            tr.marker.update(opacity=1.0)

    # Clean text: hide zeros, ensure integers
    for tr in fig.data:
        cleaned = []
        for y_val, t in zip(tr.y, tr.text):
            try:
                num = int(float(t))
            except (TypeError, ValueError):
                num = 0
            cleaned.append("" if (y_val == 0 or num == 0) else str(num))
        tr.text = cleaned
        tr.textposition = "outside"

    # Add some headroom for labels
    max_y = (df_plot['relative_contribution'].max() or 0) * 1.15 + 0.02
    fig.update_yaxes(range=[0, max_y])

    fig.update_layout(
        legend_title_text='Group',
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )
    fig.show()
    return None


# test it
k_max = 10
plot_relative_contribution(df_ranking, the_sessions[0], 'pup_retrieve', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[0], 'pup_run', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[0], 'bed_retrieve', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[0], 'notmuch', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[0], 'attack', max_rank=k_max)

#%%
k_max=5
plot_relative_contribution(df_ranking, the_sessions[0], 'pup_retrieve', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[1], 'pup_retrieve', max_rank=k_max)

#%%

k_max=5
plot_relative_contribution(df_ranking, the_sessions[0], 'pup_run', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[1], 'pup_run', max_rank=k_max)
# %%
# consistency across sessions
plot_relative_contribution(df_ranking, the_sessions[0], 'attack', max_rank=k_max)
plot_relative_contribution(df_ranking, the_sessions[1], 'attack', max_rank=k_max)


# %%
