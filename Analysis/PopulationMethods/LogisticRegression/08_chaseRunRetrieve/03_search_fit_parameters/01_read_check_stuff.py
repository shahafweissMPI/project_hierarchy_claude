# -*- coding: utf-8 -*-
"""
Created on 2025-08-28

@author: Dylan Festa

Check the performance as a function of the different hyperparameters from
the script `fit_score_only.py`

"""
#%%

import os
from pathlib import Path


path_this_file = Path(__file__).resolve()
path_storage = path_this_file.parent / "local_outputs"

import pickle, json, jsonpickle
import numpy as np, pandas as pd, xarray as xr
import time


import plotly.express as px
import plotly.graph_objects as go

#%% Functions for reading the data stored by sacred
def check_ids(datafolder):
  ids=[]
  labels=[]
  dates=[]
  has_binary=[]
  def read_json(id,filename):
    filepath = os.path.join(datafolder,'%d'%id,filename)
    return jsonpickle.decode(open(filepath,'r').read())

  for mydir in os.listdir(datafolder):
    # if does not contain config.json, move on
    filepath = os.path.join(datafolder,mydir,'config.json')
    if not os.path.exists(filepath):
      continue
    theid = int(mydir)
    ids.append(theid)
    rundict = read_json(theid,'run.json')
    infodic = read_json(theid,'info.json')
    configdic = read_json(theid,'config.json')
    labels.append(configdic['series_id'])
    dates.append(rundict['start_time'][:9])
    has_binary.append(rundict['artifacts'] != [])
  # now sort by ID
  sorted_ids = np.argsort(ids)
  for i in sorted_ids:
    print(f"ID {ids[i]} -- {dates[i]} -- {labels[i]} -- Saved data: {has_binary[i]}")
  # and return sorted lists
  return [ [ids[i] for i in sorted_ids],
          [labels[i] for i in sorted_ids], 
          [dates[i] for i in sorted_ids], 
          [has_binary[i] for i in sorted_ids]]

# reads the simulation details, and stores them in three separate
# sub-dictionaries
def read_stuff(id,datafolder):
  def read_json(filename):
    filepath = os.path.join(datafolder,'%d'%id,filename)
    return jsonpickle.decode(open(filepath,'r').read())
  rundict = read_json('run.json')
  infodic = read_json('info.json')
  configdic = read_json('config.json')
  # also , read what was printed on terminal
  cout_path = os.path.join(datafolder,'%d'%id,'cout.txt')
  cout_txt = open(cout_path,'r').read()
  return rundict,infodic,configdic,cout_txt

# this reads the ONE binary save that was produced by the run
def read_binary_save(id,datafolder):
  rundict = read_stuff(id,datafolder)[0]
  # WARNING: we assume there is only one artifact per run!
  artifact_name = rundict['artifacts'][0]
  filepath = os.path.join(datafolder,'%d'%id,artifact_name)
  return pickle.load(open(filepath,'rb'))

def read_all_data(id,datafolder):
  rundict,infodic,configdic,cout_txt = read_stuff(id,datafolder)
  outputdict = read_binary_save(id,datafolder)
  return {'run':rundict,'info':infodic,'config':configdic,
          'output':outputdict,'cout':cout_txt}

def extract_run_retrieve_performance(id,datafolder):
    data_all = read_all_data(id,datafolder)
    config_dict = data_all['config']
    output_dict = data_all['output']
    beh_df = output_dict['per_behaviour_avg_metrics_df']
    # get mouse and session
    mouse = beh_df['mouse'].values[0]
    session = beh_df['session'].values[0]
    # get f1-score for behaviour_label pup_retrieve
    f1_score_retrieve = beh_df.query("behavior_label == 'pup_retrieve'")['f1-score'].values[0]
    # sane for pup_run
    f1_score_run = beh_df.query("behavior_label == 'pup_run'")['f1-score'].values[0]
    # geometric_mean
    f1_score_geo = (f1_score_retrieve * f1_score_run) ** 0.5
    dict_ret = {
        'mouse': mouse,
        'session': session,
        'f1_score_retrieve': f1_score_retrieve,
        'f1_score_run': f1_score_run,
        'f1_score_mean': f1_score_geo
    }
    # merge with config_dict
    dict_ret.update(config_dict)
    return dict_ret
  

all_good_ids = check_ids(path_storage)[0]

all_data_test = read_all_data(1,path_storage)

#%%
# df with all run and retrieve score

df_all_scores_rows = []

for _id in all_good_ids:
    print(f"Processing ID {_id}")
    df_all_scores_rows.append(extract_run_retrieve_performance(_id,path_storage))

df_all_scores = pd.DataFrame(df_all_scores_rows)

#%%
mouse_test = 'afm16924'
session_test = '240529'

df_one_mousesession = df_all_scores.query("mouse == @mouse_test and session == @session_test").copy()

# %% Bar plot: f1_score_mean vs C_regression grouped by (dt, penalty)
def do_the_plot(df_plot):
    df_plot = df_plot.copy()
    _min_score = df_plot['f1_score_mean'].min()
    if _min_score > 0.6:
        y_plot_range = [0.6, 1]
    else:
        y_plot_range = [0.0, 1]
    df_plot['dt_penalty'] = df_plot['dt'].astype(str) + "_" + df_plot['penalty'].astype(str)
    # create readable categorical labels for C
    c_sorted = sorted(df_plot['C_regression'].unique())
    c_label_map = {c: f"C={c:g}" for c in c_sorted}
    df_plot['C_regression_label'] = df_plot['C_regression'].map(c_label_map)
    combo_order = (
        df_plot[['dt', 'penalty', 'dt_penalty']]
        .drop_duplicates()
        .sort_values(['dt', 'penalty'])['dt_penalty']
        .tolist()
    )
    fig = px.bar(
        df_plot,
        x='C_regression_label',
        y='f1_score_mean',
        color='dt_penalty',
        barmode='group',
        category_orders={'C_regression_label': [c_label_map[c] for c in c_sorted],
                         'dt_penalty': combo_order},
        title=f"Geometric mean F1 (retrieve/run) vs C_regression ({mouse_test} / {session_test})"
    )
    fig.update_yaxes(range=y_plot_range, title='F1-score geometric mean')
    fig.update_xaxes(title='C_regression')
    fig.update_layout(legend_title='dt_penalty', bargap=0.15)
    fig.show()
    return fig

# %%
# unique combinations of mouse session columns
df_mouse_sessions = df_all_scores[['mouse', 'session']].drop_duplicates()

# now, for each row, do the plot
for idx, row in df_mouse_sessions.iterrows():
    mouse_test = row['mouse']
    session_test = row['session']
    df_one_mousesession = df_all_scores.query("mouse == @mouse_test and session == @session_test").copy()
    fig=do_the_plot(df_one_mousesession)
    # export as png on /tmp/ folder
    fig.write_image(f"/tmp/plot_mouse_{mouse_test}_session_{session_test}.png")

#%%