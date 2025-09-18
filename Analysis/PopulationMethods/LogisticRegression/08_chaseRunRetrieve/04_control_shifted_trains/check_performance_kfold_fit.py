# -*- coding: utf-8 -*-
"""
Created on 2025-08-28

@author: Dylan Festa

After performing k-fold cross-validated fit of all sessions using `k_fold_fit_and_save.py`,
this script checks the performance of the fitted models by behaviour label.
"""
#%%
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go
# NEW: persistence and cleanup helpers
import gc, json
from joblib import dump, load
import sys 
from pathlib import Path

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl

#%%

path_this_file = Path(__file__).resolve()
path_data = path_this_file.parent / "local_outputs"
if not path_data.exists():
    raise FileNotFoundError(f"Data directory not found: {path_data}")

#%%
# now read all subfolders in path_data and extract mouse,session combinations
records = []
for p in [p for p in path_data.iterdir() if p.is_dir()]:
    name = p.name
    if "_" not in name:
        print(f"Skipping (no underscore): {name}")
        continue
    mouse_candidate, session = name.split("_", 1)
    # validate mouse pattern: 3 letters + 5 digits
    if len(mouse_candidate) == 8 and mouse_candidate[:3].isalpha() and mouse_candidate[3:].isdigit():
        records.append({
            "mouse": mouse_candidate,
            "session": session,
            "path": p,              # Path object (kept for convenience)
            "full_path": str(p)     # Added string version
        })
    else:
        print(f"Skipping (mouse pattern mismatch): {name}")

mouse_session_df = pd.DataFrame(records).sort_values(["mouse", "session"]).reset_index(drop=True)

print(f"Discovered {len(mouse_session_df)} mouse-session folders.")
print(mouse_session_df[["mouse","session","full_path"]])

#%%

def read_json_file_as_dict(_path_file:str):
    data_dict = {}
    if os.path.exists(_path_file):
        with open(_path_file, "r") as f:
            data_dict = json.load(f)
    else:
        raise FileNotFoundError(f"File not found: {_path_file}")
    return data_dict

def read_run_params(_path:str):
    runparams_file = "run_params.json"
    return read_json_file_as_dict(Path(_path) / runparams_file)

def read_classidx_to_beh(_path:str):
    classidx_file="dict_classindex_to_behaviour.json"
    return read_json_file_as_dict(Path(_path) / classidx_file)

def read_report_fold(_path:str, fold_idx:int):
    report_file = f"report_fold_{fold_idx}.json"
    return read_json_file_as_dict(Path(_path) / report_file)

def get_avg_performance(_path:str,n_folds:int):
    all_reports = [read_report_fold(_path, fold_idx=i) for i in range(n_folds)]
    # Compute average performance metrics across all folds
    _class_idx_dict = read_classidx_to_beh(_path)

    _df_rows = []
    for (class_idx, beh_label) in _class_idx_dict.items():
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        support_avg = 0.0
        # if class_idx not in _report_dict, continue
        if str(class_idx) not in all_reports[0].keys(): 
            continue
        for _report_dict in all_reports:
            _report_dict_beh = _report_dict[str(class_idx)]
            precision += _report_dict_beh['precision']
            recall += _report_dict_beh['recall']
            f1_score += _report_dict_beh['f1-score']
            support_avg += _report_dict_beh['support']
        precision /= n_folds
        recall /= n_folds
        f1_score /= n_folds
        support_avg /= n_folds
        _df_rows.append({
            "class_idx": class_idx,
            "behaviour": beh_label,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support_avg
        })

    return pd.DataFrame(_df_rows)

# %%

idx_row_test = 2 

row_test = mouse_session_df.loc[idx_row_test]
mouse_test = row_test['mouse']
session_test = row_test['session']
path_test = row_test['full_path']

run_params_dict = read_run_params(path_test)
classidx_to_beh_dict = read_classidx_to_beh(path_test)
n_total_folds = run_params_dict['n_total_folds']
# %%

#report_test = read_report_fold(path_test, fold_idx=3)

# %%
avg_performance_test = get_avg_performance(path_test, n_folds=n_total_folds)

# %%
# mouse_session_df...

performance_df_list = []
# iterate over rows in mouse_session_df
for idx, row in mouse_session_df.iterrows():
    mouse = row['mouse']
    session = row['session']
    path = row['full_path']

    avg_performance_df = get_avg_performance(path, n_folds=n_total_folds)
    # add mouse and session columns
    avg_performance_df['mouse'] = mouse
    avg_performance_df['session'] = session
    performance_df_list.append(avg_performance_df)

performance_df_all = pd.concat(performance_df_list, ignore_index=True)

# %%
def plot_session_performance(df_all: pd.DataFrame, save: bool = True, show: bool = False):
    """
    Generate one grouped bar plot per (mouse, session) with behaviours on x and
    precision/recall/f1_score as grouped bars. Y axis fixed to [0,1].
    Now also annotates per-behaviour support (rounded to 1 decimal) above each group.
    """
    out_dir = Path('/tmp')

    metrics = ["precision", "recall", "f1_score"]

    for (mouse, session), df_sub in df_all.groupby(["mouse", "session"]):
        df_sub = df_sub.sort_values("behaviour")
        fig = go.Figure()
        for metric in metrics:
            fig.add_bar(
                x=df_sub["behaviour"],
                y=df_sub[metric],
                name=metric
            )
        # Add support annotations
        for _, r in df_sub.iterrows():
            beh = r["behaviour"]
            support_val = r["support"]
            # max metric height for this behaviour
            max_h = max(r[m] for m in metrics)
            # place annotation slightly above bar but inside axis range
            y_annot = min(max_h + 0.02, 0.98)
            fig.add_annotation(
                x=beh,
                y=y_annot,
                text=f"{support_val:.1f}",
                showarrow=False,
                font=dict(size=10, color="black"),
                xanchor="center",
                yanchor="bottom"
            )
        fig.update_layout(
            title=f"SHUFFLE CONTROL! mouse: {mouse} | s: {session}",
            barmode="group",
            xaxis_title="Behaviour",
            yaxis=dict(title="Score", range=[0, 1]),
            legend_title="Metric",
            template="plotly_white"
        )
        if show:
            fig.show()
        if save:
            fname = f"plot_{mouse}_{session}_performance_CONTROL.png"
            # save as png in /tmp/ directory
            fig.write_image(out_dir / fname)

# Execute plotting
plot_session_performance(performance_df_all, save=False, show=True)
# === END NEW ===


# %%
