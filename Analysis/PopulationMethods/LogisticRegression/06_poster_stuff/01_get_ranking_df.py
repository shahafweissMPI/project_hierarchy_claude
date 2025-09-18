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
import json
from joblib import load


# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre



all_mice = rdl.get_good_animals()
print(f"Found {len(all_mice)} animals.")
print("Animals:", all_mice)

animal = 'afm16924'
sessions_for_animal = rdl.get_good_sessions(animal)
print(f"Found {len(sessions_for_animal)} sessions for animal {animal}.")
print("Sessions:", sessions_for_animal)

#%%

sessions = ['240527', '240529']

print("Loading data...")



#%% Read run parameters saved by training
outdir = os.path.join(os.path.dirname(__file__), "local_outputs", f"{animal}_{sessions[0]}")
params_path = os.path.join(outdir, "run_params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as f:
        _run_params = json.load(f)
    dt = float(_run_params.get("dt", 10e-3))
    n_lags = int(_run_params.get("n_lags", 29))
    n_k_folds = int(_run_params.get("n_k_folds", 1000))
    print(f"Loaded run params from {params_path}: dt={dt}, n_lags={n_lags}, n_k_folds={n_k_folds}")
else:
    raise FileNotFoundError(f"Run params file not found: {params_path}")


#%% Per-session loaders and caches (artifacts saved by fit_and_save.py)
_run_params_by_session = {}
_units_by_session = {}
_unit_locations_by_session = {}
_labeldict_by_session = {}  # {'behaviour_to_index': ..., 'index_to_behaviour': ...}
_folds_by_session = {}
_labels_by_session = {}  # simple list of available behaviour labels per session

def _session_outdir(session: str) -> str:
    return os.path.join(os.path.dirname(__file__), "local_outputs", f"{animal}_{session}")

def load_session_artifacts(session: str):
    """Load and cache all artifacts for a given session (id string, e.g. '240529')."""
    if session in _folds_by_session:
        return

    base = _session_outdir(session)

    # Load run params
    rp_path = os.path.join(base, "run_params.json")
    if not os.path.exists(rp_path):
        raise FileNotFoundError(f"run_params.json not found for session {session}: {rp_path}")
    with open(rp_path, "r") as f:
        rp = json.load(f)
    _run_params_by_session[session] = {
        'dt': float(rp.get('dt', 10e-3)),
        'n_lags': int(rp.get('n_lags', 29)),
        'n_total_folds': int(rp.get('n_total_folds', 5)),
    }

    # Load units and label dicts
    labels_meta_path = os.path.join(base, "labels_meta.joblib")
    if not os.path.exists(labels_meta_path):
        raise FileNotFoundError(f"labels_meta.joblib not found for session {session}: {labels_meta_path}")
    labels_meta = load(labels_meta_path)
    _units_by_session[session] = labels_meta.get('units')
    _unit_locations_by_session[session] = labels_meta.get('unit_locations', np.array([]))
    behaviour_to_index = labels_meta.get('dict_behaviour_label_to_index', {})
    _labeldict_by_session[session] = {
        'behaviour_to_index': behaviour_to_index,
        'index_to_behaviour': labels_meta.get('dict_classindex_to_behaviour', {}),
    }
    # Populate simple list of available behaviour labels (exclude 'none')
    _labels_by_session[session] = sorted([lbl for lbl in behaviour_to_index.keys() if lbl != 'none'])

    # Load folds list from summary
    folds_summary_csv = os.path.join(base, "folds_summary.csv")
    if not os.path.exists(folds_summary_csv):
        raise FileNotFoundError(f"folds_summary.csv not found for session {session}: {folds_summary_csv}")
    folds_summary_df = pd.read_csv(folds_summary_csv)
    ks = folds_summary_df['k'].tolist()

    all_folds_data = []
    for k in ks:
        pipe_path = os.path.join(base, f"pipe_fold_{k}.joblib")
        report_path = os.path.join(base, f"report_fold_{k}.json")
        cm_path = os.path.join(base, f"confusion_matrix_fold_{k}.npy")

        pipe_k = load(pipe_path)
        with open(report_path, "r") as f:
            report_k = json.load(f)
        cm_k = np.load(cm_path)

        all_folds_data.append({
            'k': k,
            'pipeline': pipe_k,
            'report': report_k,
            'confusion_matrix': cm_k,
        })
    _folds_by_session[session] = all_folds_data


# Small helper to access available labels for a session
def get_available_behaviours(session: str) -> list:
    load_session_artifacts(session)
    return _labels_by_session.get(session, [])

#%%
# get the best fold for a session (optional helper)
def get_best_fold(session: str) -> int:
    base = _session_outdir(session)
    folds_summary_df = pd.read_csv(os.path.join(base, "folds_summary.csv"))
    return int(folds_summary_df.iloc[folds_summary_df['macro_f1'].values.argmax()]['k'])


#%%
# some utility functions to get the coefficients
def get_loadings(fold: int, behaviour: str, session: str) -> xr.DataArray:
    """
    Return LogisticRegression loadings for a behaviour and fold as an xarray.DataArray
    with dims ('lag','unit') and coords:
      - lag: 0..n_lags
      - unit: units_fit (saved by fit_and_save.py)
    """
    load_session_artifacts(session)
    n_lags_sess = _run_params_by_session[session]['n_lags']
    units = _units_by_session[session]
    unit_locations = _unit_locations_by_session[session]
    n_units = len(units)

    pipe = _folds_by_session[session][fold]['pipeline']
    lr = pipe.named_steps['lda']  # step name kept as 'lda' in fit_and_save.py
    classes = lr.classes_

    beh_to_idx = _labeldict_by_session[session]['behaviour_to_index']
    if behaviour not in beh_to_idx:
        raise KeyError(f"Behaviour '{behaviour}' not found in label dict for session {session}.")
    class_index = beh_to_idx[behaviour]

    # find the row corresponding to this class index in the fitted model
    match = np.where(classes == class_index)[0]
    if len(match) == 0:
        raise ValueError(f"Class index {class_index} (beh='{behaviour}') not present in fold {fold} model for session {session}.")
    row = int(match[0])

    coef = lr.coef_[row]  # shape: (n_features,)
    coef_2d = coef.reshape(n_lags_sess + 1, n_units)

    # build DataArray with coords
    da = xr.DataArray(
        coef_2d,
        dims=("lag", "unit"),
        coords={
            "lag": np.arange(n_lags_sess + 1),
            "unit": units,
        },
        name=f"loadings_{behaviour}_fold{fold}_{session}",
        attrs={
            "unit_locations": unit_locations,
            "n_lags": n_lags_sess + 1,
        }
    )
    return da


def get_loading_average(behaviour: str, session: str) -> xr.DataArray:
    load_session_artifacts(session)
    n_folds = len(_folds_by_session[session])
    acc = None
    for k in range(n_folds):
        da = get_loadings(k, behaviour, session)
        acc = da if acc is None else (acc + da)
    acc = acc / n_folds
    # copy attributes from last da
    for key, value in da.attrs.items():
        acc.attrs[key] = value
    return acc


# just sum over lags, +1 if positive, -1 if negative
def get_loading_sign(behaviour: str, session: str) -> np.ndarray:
    da_avg = get_loading_average(behaviour, session)
    sums = da_avg.sum(dim="lag").values  # per-unit sum
    _signs = np.sign(sums)
    _signs[_signs == 0] = 1.0
    return _signs

#%%

beh_0 = get_available_behaviours(sessions[0])  # just to load the session artifacts
beh_1 = get_available_behaviours(sessions[1])  # just to load the session artifacts

#%%
# now, build the dataframe

_main_df_rows = []

# Merged loop over sessions; ensure consistent key 'rank_val' for all rows
for _session in sessions:
    behaviours = get_available_behaviours(_session)
    for beh in behaviours:
        loading_avg = get_loading_average(beh, _session)
        loading_sign = get_loading_sign(beh, _session)
        loading_avg_sq = loading_avg ** 2
        rank_val = loading_avg_sq.sum(dim="lag").values
        # sort descending by rank_val (largest magnitude first)
        rank_sorting = np.argsort(-rank_val)
        rank = np.empty_like(rank_sorting)
        rank[rank_sorting] = np.arange(len(rank_sorting))
        units = loading_avg.coords["unit"].values
        unit_location = loading_avg.attrs['unit_locations']
        for _idx_unit, _unit in enumerate(units):
            _main_df_rows.append({
                "behaviour": beh,
                "session": _session,
                "unit": _unit,
                "unit_location": unit_location[_idx_unit],
                "rank_sign": loading_sign[_idx_unit],
                "rank": int(rank[_idx_unit]),
                "rank_val": float(rank_val[_idx_unit] * loading_sign[_idx_unit]),
            })

# now created the main DataFrame
df_ranking = pd.DataFrame(_main_df_rows)
# sort by session, behaviour, unit
df_ranking.sort_values(by=["session", "behaviour", "unit"], inplace=True)

#%%
# sanity check, plot with plotly scatter of rank vs rank_val


fig = px.scatter(df_ranking, x="rank", y="rank_val", color="session", 
                 title="Rank vs Rank Value",
                 labels={"rank": "Rank", "rank_val": "Rank Value"})
fig.show()

#%%
# second check, given a session, and a given rank level,
# the number of units that are at that rank level or below for only one behaviour,
# those that appear in two behaviours, three behaviours, etc etc.

def print_rank_counts(session: str, rank_level: int):
    filtered = df_ranking[(df_ranking["session"] == session) & (df_ranking["rank"] <= rank_level)]
    # count how many times each unit appears
    unit_counts = filtered["unit"].value_counts()
    # now invert, for each value count, from the highest, print how many units there are
    inverted_counts = unit_counts.value_counts().sort_index(ascending=False)
    # print on screen, starting from highest
    print(f"Rank counts for session {session} at rank level {rank_level}")
    n_units = df_ranking["unit"].nunique()
    print(f"Total unique units: {n_units}")
    n_unique_units_highrank = filtered["unit"].nunique()
    print(f"Total units at rank level {rank_level} or below: {n_unique_units_highrank}")
    for count, num_units in inverted_counts.items():
        # singular or plural
        beh_string = "behaviour" if count == 1 else "behaviours"
        print(f"In {count} {beh_string}: {num_units} units")
    print("")
    return None

print_rank_counts(sessions[0], 9)
print_rank_counts(sessions[1], 9)

# %%
# dump df as a temporary binary in local_outputs
savedir = os.path.join(os.path.dirname(__file__), "local_outputs")
if not os.path.exists(savedir):
    raise FileNotFoundError(f"Output directory {savedir} does not exist.")

df_ranking_path = os.path.join(savedir, f"df_ranking_{animal}.pkl")
df_ranking.to_pickle(df_ranking_path)

print(f"DataFrame saved to {df_ranking_path}")

exit()
