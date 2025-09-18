# -*- coding: utf-8 -*-
"""
Created on 2025-08-29

@author: Dylan Festa

For each session, makes a bar plot of behaviours and f1-score, comparing full model 
with shuffled control.

Uses the already saved data for the plot.
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
path_data_control = path_this_file.parent / "local_outputs"
if not path_data_control.exists():
    raise FileNotFoundError(f"Data directory for control model not found: {path_data_control}")

path_data_fullfit = path_this_file.parent.parent / "local_outputs"
if not path_data_fullfit.exists():
    raise FileNotFoundError(f"Data directory for full model not found: {path_data_fullfit}")

#%%
# now read all subfolders in path_data and extract mouse,session combinations
records = []
for p_fullfit in [p for p in path_data_fullfit.iterdir() if p.is_dir()]:
    name = p_fullfit.name
    if "_" not in name:
        print(f"Skipping (no underscore): {name}")
        continue
    mouse_candidate, session = name.split("_", 1)
    # validate mouse pattern: 3 letters + 5 digits
    if len(mouse_candidate) == 8 and mouse_candidate[:3].isalpha() and mouse_candidate[3:].isdigit():
        # control full path
        p_control = path_data_control / f"{mouse_candidate}_{session}"
        if not p_control.exists() or not p_control.is_dir():
            raise FileNotFoundError(f"Control data not found: {p_control}")
        records.append({
            "mouse": mouse_candidate,
            "session": session,
            "full_path_fullfit": str(p_fullfit),    
            "full_path_control": str(p_control)    
        })
    else:
        print(f"Skipping (mouse pattern mismatch): {name}")

mouse_session_df = pd.DataFrame(records).sort_values(["mouse", "session"]).reset_index(drop=True)

print(f"Discovered {len(mouse_session_df)} mouse-session folders.")
print(mouse_session_df[["mouse","session","full_path_control","full_path_fullfit"]])

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
path_test = row_test['full_path_fullfit']

run_params_dict = read_run_params(path_test)
classidx_to_beh_dict = read_classidx_to_beh(path_test)
n_total_folds = run_params_dict['n_total_folds']
# %%

#report_test = read_report_fold(path_test, fold_idx=3)

# %%
avg_performance_test = get_avg_performance(path_test, n_folds=n_total_folds)

# %%
performance_df_full_list = []
performance_df_control_list = []
# iterate over rows in mouse_session_df
for idx, row in mouse_session_df.iterrows():
    mouse = row['mouse']
    session = row['session']
    path_full = row['full_path_fullfit']
    path_control = row['full_path_control']  # FIX: use control path

    avg_performance_full_df = get_avg_performance(path_full, n_folds=n_total_folds)
    avg_performance_control_df = get_avg_performance(path_control, n_folds=n_total_folds)  # FIX: correct source
    # add mouse and session columns
    avg_performance_full_df['mouse'] = mouse
    avg_performance_full_df['session'] = session
    performance_df_full_list.append(avg_performance_full_df)

    avg_performance_control_df['mouse'] = mouse
    avg_performance_control_df['session'] = session
    performance_df_control_list.append(avg_performance_control_df)

performance_df_full_all = pd.concat(performance_df_full_list, ignore_index=True)
performance_df_control_all = pd.concat(performance_df_control_list, ignore_index=True)

# %%
# def plot_session_performance_withcontrol(df_full: pd.DataFrame, df_control: pd.DataFrame, save: bool = True, show: bool = False):
#     """
#     Generate one grouped bar plot per (mouse, session) with behaviours on x and
#     precision/recall/f1_score as grouped bars. Y axis fixed to [0,1].
#     Now also annotates per-behaviour support (rounded to 1 decimal) above each group.
#     """
#     out_dir = Path('/tmp')
#     out_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists

#     metrics = ["precision", "recall", "f1_score"]

#     for (mouse, session), df_sub in df_full.groupby(["mouse", "session"]):
#         df_sub = df_sub.sort_values("behaviour")
#         fig = go.Figure()
#         for metric in metrics:
#             fig.add_bar(
#                 x=df_sub["behaviour"],
#                 y=df_sub[metric],
#                 name=metric
#             )
#         # Add support annotations
#         for _, r in df_sub.iterrows():
#             beh = r["behaviour"]
#             support_val = r["support"]
#             # max metric height for this behaviour
#             max_h = max(r[m] for m in metrics)
#             # place annotation slightly above bar but inside axis range
#             y_annot = min(max_h + 0.02, 0.98)
#             fig.add_annotation(
#                 x=beh,
#                 y=y_annot,
#                 text=f"{support_val:.1f}",
#                 showarrow=False,
#                 font=dict(size=10, color="black"),
#                 xanchor="center",
#                 yanchor="bottom"
#             )
#         fig.update_layout(
#             title=f"SHUFFLE CONTROL! mouse: {mouse} | s: {session}",
#             barmode="group",
#             xaxis_title="Behaviour",
#             yaxis=dict(title="Score", range=[0, 1]),
#             legend_title="Metric",
#             template="plotly_white"
#         )
#         if show:
#             fig.show()
#         if save:
#             base = f"plot_{mouse}_{session}_performance_CONTROL"
#             for ext in ("png", "svg", "pdf"):
#                 fig.write_image(out_dir / f"{base}.{ext}")

# # Execute plotting
# plot_session_performance_withcontrol(performance_df_full_all, performance_df_control_all, save=False, show=True)
# # === END NEW ===


# %%
def plot_session_f1_full_vs_control(df_full: pd.DataFrame,
                                    df_control: pd.DataFrame,
                                    save: bool = True,
                                    show: bool = False):
    """
    For each (mouse, session):
      - Plot f1-score only.
      - Two bars per behaviour: Full (blue) and Control (grey).
      - Annotate support (from full) above the higher of the two bars.
    """
    # Colors (web-safe)
    COLOR_FULL = "#0A2F5A"      # dark blue
    COLOR_CONTROL = "#7F7F7F"   # grey
    out_dir = Path('/tmp/plots_f1_full_vs_control')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate sessions
    for (mouse, session), df_full_sub in df_full.groupby(["mouse", "session"]):
        # Match control subset
        try:
            df_ctrl_sub = df_control.query("mouse == @mouse and session == @session")
        except Exception:
            continue
        if df_ctrl_sub.empty:
            print(f"[WARN] No control data for {mouse} {session}")
            continue

        # Merge on behaviour (inner join keeps intersection)
        m = (df_full_sub[['behaviour', 'f1_score', 'support']]
             .rename(columns={'f1_score': 'f1_full', 'support': 'support_full'})
             .merge(df_ctrl_sub[['behaviour', 'f1_score']]
                    .rename(columns={'f1_score': 'f1_control'}),
                    on='behaviour', how='inner'))

        if m.empty:
            print(f"[WARN] No overlapping behaviours for {mouse} {session}")
            continue

        m = m.sort_values('behaviour')
        # Reorder behaviours: alphabetical, but move 'chase' to end if present
        behaviours_sorted = sorted(m['behaviour'].unique())
        if 'chase' in behaviours_sorted:
            behaviours_sorted = [b for b in behaviours_sorted if b != 'chase'] + ['chase']
        # Reindex dataframe to this order
        m = m.set_index('behaviour').loc[behaviours_sorted].reset_index()
        behaviours = behaviours_sorted
        f1_full = m['f1_full'].tolist()
        f1_control = m['f1_control'].tolist()
        fig = go.Figure()
        fig.add_bar(name="full model", x=behaviours, y=f1_full, marker_color=COLOR_FULL)
        fig.add_bar(name="shuffled control", x=behaviours, y=f1_control, marker_color=COLOR_CONTROL)

        # Annotations: support from full above the taller bar of each pair
        for _, r in m.iterrows():
            y_max = max(r['f1_full'], r['f1_control'])
            y_annot = min(y_max + 0.02, 0.98)
            fig.add_annotation(
                x=r['behaviour'],
                y=y_annot,
                text=f"{round(r['support_full']):d}",
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
                yanchor="bottom"
            )

        fig.update_layout(
            title=f"F1-score Full vs shuffled | mouse: {mouse} | session: {session}",
            barmode="group",
            xaxis_title="behaviour",
            yaxis=dict(title="f1-score", range=[0, 1]),
            legend_title="",
            template="plotly_white"
        )

        if show:
            fig.show()
        if save:
            base = f"f1_full_vs_control_{mouse}_{session}"
            for ext in ("png", "svg", "pdf"):
                fig.write_image(out_dir / f"{base}.{ext}")

# Execute new plotting (replaces old incorrect call)
plot_session_f1_full_vs_control(performance_df_full_all, performance_df_control_all, save=True, show=True)

# === END SCRIPT ===

#%%