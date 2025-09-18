# -*- coding: utf-8 -*-
"""
+ created on 2025-09-11

@ author: Dylan Festa

+ `05_find_decoding_neurons_by_exclusion.py` does the fit and saves the results
+ `05_read_and_plot` reads those results, and saves the subset of the top decoding neurons in a separate dataframe
   `top_subset_df.pkl`  

Here I read that dataframe and convert it in a simpler format.
mouse,session, selection_name, units 
Behaviours are integrated into selection_name.

I order the units by their rate (which requires reading the data again).
"""
#%%

from __future__ import annotations
from typing import List,Dict,Tuple
import os,sys

import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains


#%%
def generate_rates_df(mouse:str,session:str):
    """
    Given mouse and session, reads spiking data
    and generates a dataframe with columns: 
    'mouse','session','unit','rate'
    """

    data_ = rdl.load_preprocessed_dict(mouse,session)
    spike_times = data_['spike_times']
    time_index = data_['time_index']
    cluster_index = data_['cluster_index']
    t_end = time_index[-1]
    rates = [len(st)/t_end for st in spike_times]
    df_ret = pd.DataFrame({
        'mouse':mouse,
        'session':session,
        'unit':cluster_index,
        'rate':rates
    })
    return df_ret


#%%
path_this_file = Path(__file__).resolve()
path_data = path_this_file.parent / "local_outputs_05byexclusion"

if not path_data.exists():
    raise FileNotFoundError(f"Data path {path_data} does not exist")

path_file = path_data / "top_subset_df.pkl"
if not path_file.exists():
    raise FileNotFoundError(f"Data file {path_file} does not exist")
df = pd.read_pickle(path_file)
data = df['data']
data_df = pd.DataFrame(data)
#%%
# test rates 
#mouse_test,session_test = data_df.iloc[0][['mouse','session']]
#rates_df_test = generate_rates_df(mouse_test,session_test)

# Compute and merge firing rates per (mouse, session) so each row has its unit's rate
unique_mouse_session = data_df[['mouse','session']].drop_duplicates()
_rates_list = []
for _m, _s in unique_mouse_session.itertuples(index=False):
    _rates_list.append(generate_rates_df(_m, _s))

_rates_all_df = (pd.concat(_rates_list, ignore_index=True)
                    .rename(columns={'rate':'rates'}))
# Merge (units can repeat in data_df); left join preserves original ordering
data_df = data_df.merge(_rates_all_df, on=['mouse','session','unit'], how='left')
# Optional sanity check: warn if any rates missing
if data_df['rates'].isna().any():
    missing_ct = data_df['rates'].isna().sum()
    print(f"Warning: {missing_ct} rows have missing rates after merge.")

#%%
# Group and aggregate units into a list (unique order preserved by first appearance)
# (Updated: now order units within each group by descending firing rate)

group_cols = ["mouse","session","behaviour"]
# Sort rows by descending rate first, then aggregate ordered units per group
_units_sorted = data_df.sort_values('rates', ascending=False)
units_grouped_df = (_units_sorted
                    .groupby(group_cols, dropna=False, sort=False)
                    .apply(lambda g: g.sort_values('rates', ascending=False)['unit'].tolist())
                    .reset_index(name='units'))

selection_name_col = [ f"top coding for {beh_}" for beh_ in units_grouped_df['behaviour'] ]
units_grouped_df['selection_name'] = selection_name_col

#%%
# save as pickled dataframe called 'top_subset_as_list.pkl'

path_output_file = path_data / "top_subset_as_list.pkl"
units_grouped_df.to_pickle(path_output_file)
print(f"Saved top subset unit selection dataframe to {path_output_file}")
