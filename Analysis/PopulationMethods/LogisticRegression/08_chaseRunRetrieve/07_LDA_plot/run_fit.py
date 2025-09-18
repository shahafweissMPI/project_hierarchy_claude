# -*- coding: utf-8 -*-
"""
Created on 2025-08-31

Reworked: Use LinearDiscriminantAnalysis and output 2D latent coordinates
for each test sample (per fold) with its true behaviour label.
All performance / coefficient logic removed.
"""
#%%
from __future__ import annotations
from typing import List, Tuple, Dict, Any

#%load_ext cuml.accel
import cuml
#cuml.set_global_output_type('numpy')
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
# NEW: persistence and cleanup helpers
import gc, json
from joblib import dump, load
import sys  # NEW: for simple CLI parsing
try:
    import cupy as cp
except Exception:
    cp = None

def _free_gpu():
    """Best-effort free of CuPy memory pools."""
    try:
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

path_this_file = Path(__file__).resolve()
path_data = path_this_file.parent / "local_outputs"
if not path_data.exists():
    raise FileNotFoundError(f"Data directory not found: {path_data}")

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

# import stuff from sklear: pipeline, lagged data, Z-score, PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# simple argument parsing (before using the_animal/the_session) ---
DEFAULT_MOUSE = 'afm16924'
DEFAULT_SESSION = '240529'

# Fixed parameters (removed C, penalty – not needed for LDA)
dt = 20*1E-3
n_lags = 29 # 49
n_folds_total = 3

def _parse_cmdline(default_mouse: str, default_session: str):
    mouse = default_mouse
    session = default_session
    for arg in sys.argv[1:]:
        if arg in ('-h', '--help'):
            print(f"Usage: python {os.path.basename(sys.argv[0])} mouse=<mouse_id> session=<session_id>")
            print(f"Defaults: mouse={default_mouse} session={default_session}")
            sys.exit(0)
        if arg.startswith('mouse='):
            mouse = arg.split('=', 1)[1]
        elif arg.startswith('session='):
            session = arg.split('=', 1)[1]
    return mouse, session

the_mouse, the_session = _parse_cmdline(DEFAULT_MOUSE, DEFAULT_SESSION)
print(f"[INFO] Using mouse={the_mouse} session={the_session}")


interesting_behaviours = ['chase', 'pup_run', 'pup_retrieve','escape','escape_switch']
required_behaviours = [behaviour\
        for behaviour in interesting_behaviours if (behaviour != 'escape') and (behaviour != 'escape_switch')]


print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(the_mouse, the_session)
# unpack the data
data_behaviour = all_data_dict['behaviour']
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


#%% Now, process behaviour data to get labels

behaviour_timestamps_df_all = rdl.convert_to_behaviour_timestamps(the_mouse, the_session, data_behaviour)

# first, for representation, keep all interesting behaviours 
behaviour_timestamps_df_toplot = behaviour_timestamps_df_all[\
    behaviour_timestamps_df_all['behaviour'].isin(interesting_behaviours)].copy()
# second one, for fitting, keep only required behaviours
behaviour_timestamps_df = behaviour_timestamps_df_all[\
    behaviour_timestamps_df_all['behaviour'].isin(required_behaviours)].copy()

#%%
# back to data_behaviour, find the smallest frames_s element, and report what is in that row
smallest_frame_row = data_behaviour.loc[data_behaviour['frames_s'].idxmin()]
time_first_beh = smallest_frame_row['frames_s']
time_min_str = time.strftime("%M:%S", time.gmtime(time_first_beh))


#%%

t_nolabel_start = 1 * 60.0  # 1 minute after the start of the session
# end is min between 8 minutes and 2 min before first behavior
t_nolabel_end = np.min([8 * 60.0, time_first_beh - 2 * 60.0])

# #time_first_beh - 2 * 60.0  # 2 minutes before the first behavior

k_intervals = 30  # number of intervals to generate
t_interval_duration = 2.5  # duration of each interval in seconds

intervals_nolabel_fromzero = rdl.generate_random_intervals_within_time(
        t_nolabel_end,k_intervals,t_interval_duration)

intervals_nobabel = [(_start+ t_nolabel_start, _end + t_nolabel_start) for _start, _end in intervals_nolabel_fromzero]


# add a row to behaviour dataframe
new_row = pd.DataFrame([{
    'mouse': the_mouse,
    'session': the_session,
    'behavioural_category': 'no_label',
    'behaviour': 'no_label',
    'n_trials': k_intervals,
    'is_start_stop': True,
    'total_duration': t_interval_duration * k_intervals,
    'start_stop_times': intervals_nobabel,
    'point_times': []
}])
behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)
behaviour_timestamps_df_toplot = pd.concat([behaviour_timestamps_df_toplot, new_row], ignore_index=True)

#%%
# now generate a dictionary
dict_behaviour_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df_toplot['behaviour'].values)}
# add 'none'
dict_behaviour_label_to_index['none'] = -1

#%% for convenience, have a dictionary that maps index to label, merging labels with same index
# so the above becomes pup_grab_and_pup_retrieve
# dict_classindex_to_behavior={}
# Build inverse with merging of duplicate indices
_inv = {}
for label, idx in dict_behaviour_label_to_index.items():
    _inv.setdefault(idx, []).append(label)

# Deduplicate and merge with "_and_" for multi-label indices
dict_classindex_to_behaviour = {
    idx: (labels[0] if len(labels) == 1 else "_and_".join(labels))
    for idx, labels in ((i, lst) for i, lst in _inv.items())
}

#%%
# build behaviour_representation_df
# keys are label and class index as in dict_classindex_to_behaviour
_rows_temp = []
for class_index, label in dict_classindex_to_behaviour.items():
    if label == 'none':
        continue
    _intervals = []
    for (_key,_val) in dict_behaviour_label_to_index.items():
        if _val == class_index:
            #print(f"Adding intervals for label: {label} (class index: {class_index}), key: {_key}")
            _to_add = behaviour_timestamps_df_toplot[behaviour_timestamps_df_toplot['behaviour'] == _key]['start_stop_times'].values[0]
            _intervals.append(_to_add)
    # merge if needed
    if len(_intervals) == 2:
        _intervals = rdl.merge_time_interval_list(_intervals[0], _intervals[1])
    elif len(_intervals) > 2:
        raise ValueError("More than 2 intervals found")
    else:
        _intervals=_intervals[0]
    _rows_temp.append({'label':label,'class_index':class_index,'intervals':_intervals})

behaviour_representation_df = pd.DataFrame(_rows_temp,columns=['label','class_index','intervals'])
behaviour_representation_df['plot_index'] = behaviour_representation_df.index


spiketrains=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')
n_units = spiketrains.n_units
the_units = spiketrains.units
the_unit_locations = spiketrains.unit_location
print(f"Number of PAG units: {n_units}.")




def get_train_test_data(X:np.ndarray,y:np.ndarray,k:int):
    # fix: use logical or and guard range correctly
    if (k < 0) or (k > n_folds_total - 1):
        raise ValueError(f"k must be between 0 and {n_folds_total-1}, got {k}.")
    skf = StratifiedKFold(n_splits=n_folds_total, shuffle=False)

    # extract the fold from the set without the escape
    train_index,test_index = list(\
        skf.split(X,y))[k]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    return X_train, y_train, X_test, y_test


def build_pipeline():
    """Return a simple scaling + LDA pipeline (2D latent space)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=2))
    ])

outdir = path_data


def run_one_fit(units_to_exclude: List):
    """
    Fit LDA across folds and return 2D latent coordinates for each test sample.
    Output DataFrame columns:
        fold, unit_set_id, time, behaviour, class_index, ld1, ld2
    """
    # --- Select units ---
    if not units_to_exclude:
        units_to_keep = the_units
    else:
        units_to_keep = list(set(the_units) - set(units_to_exclude))
    if len(units_to_keep) == 0:
        raise ValueError("No units left to fit.")
    spiketrains_ = spiketrains.filter_by_units(units_to_keep)
    units_less = spiketrains_.units

    # --- Bin & lag ---
    X_alltime = pre.do_binning_operation(
        spiketrains_, 'count', dt=dt, t_start=t_start_all, t_stop=t_stop_all
    )
    beh_labels_xr = rdl.generate_behaviour_labels_inclusive(
        behaviour_timestamps_df,
        t_start=0.0,
        t_stop=t_stop_all,
        dt=dt,
        behaviour_labels_dict=dict_behaviour_label_to_index
    )
    y_alltime = beh_labels_xr.values
    if len(y_alltime) != X_alltime.shape[0]:
        raise AssertionError("Mismatch between X rows and y length.")

    X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion_xr(X_alltime, n_lags)
    X_alltime_maxlagged = X_alltime_maxlagged[n_lags:, :]
    y_alltime = y_alltime[n_lags:]
    if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
        raise AssertionError("Mismatch after lag trimming.")
    t_alltime = X_alltime_maxlagged.coords['time_bin_center'].values

    # --- Keep labeled samples only (exclude 'none' = -1) ---
    y_idx_behavior = y_alltime != -1
    y_behaviour = y_alltime[y_idx_behavior]
    X_behaviour_maxlagged = X_alltime_maxlagged.values[y_idx_behavior, :]
    t_behaviour = t_alltime[y_idx_behavior]

    # Check we have enough distinct classes for 2D LDA
    unique_classes = np.unique(y_behaviour)
    if unique_classes.size < 3:
        raise ValueError(
            f"Need at least 3 classes for 2D LDA (current={unique_classes.size})."
        )

    skf = StratifiedKFold(n_splits=n_folds_total, shuffle=False)
    folds = list(skf.split(X_behaviour_maxlagged, y_behaviour))

    latent_rows = []
    unit_set_id = hash(tuple(sorted(units_less)))  # simple identifier

    for fold_idx, (train_index, test_index) in enumerate(folds):
        X_train_, X_test_ = X_behaviour_maxlagged[train_index], X_behaviour_maxlagged[test_index]
        y_train_, y_test_ = y_behaviour[train_index], y_behaviour[test_index]
        t_test_ = t_behaviour[test_index]

        pipe_ = build_pipeline()
        t_start_fit = time.time()
        pipe_.fit(X_train_, y_train_)
        t_end_fit = time.time()
        print(f"[INFO] Fold {fold_idx}: LDA fit time: {t_end_fit - t_start_fit:.2f} seconds.")
        lda_step = pipe_.named_steps['lda']
        # Ensure we actually got 2 components
        if lda_step.scalings_.shape[1] < 2 and lda_step.scalings_.ndim == 2:
            raise RuntimeError("LDA produced fewer than 2 components.")

        latent_test = pipe_.transform(X_test_)  # shape (n_test, 2) because n_components=2
        if latent_test.shape[1] != 2:
            raise RuntimeError(f"Expected 2D latent space, got shape {latent_test.shape}.")

        for i in range(latent_test.shape[0]):
            class_index = int(y_test_[i])
            behaviour_label = dict_classindex_to_behaviour.get(class_index, 'UNKNOWN')
            latent_rows.append({
                'fold': fold_idx,
                'unit_set_id': unit_set_id,
                'time': float(t_test_[i]),
                'behaviour': behaviour_label,
                'class_index': class_index,
                'ld1': float(latent_test[i, 0]),
                'ld2': float(latent_test[i, 1]),
            })

        # cleanup
        del X_train_, X_test_, y_train_, y_test_, t_test_, latent_test, pipe_, lda_step
        gc.collect()
        _free_gpu()

    latent_df = pd.DataFrame(
        latent_rows,
        columns=['fold', 'unit_set_id', 'time', 'behaviour', 'class_index', 'ld1', 'ld2']
    )
    return latent_df

#%%
# --- Run with all units by default ---
latent_df = run_one_fit(units_to_exclude=[])

#%%
# --- New: plotting utilities for latent LDA space ---------------------------------
DEFAULT_BEHAVIOUR_COLORS = {
    'chase': '#1f77b4',
    'pup_run': '#ff7f0e',
    'pup_retrieve': '#2ca02c',
    'escape': '#d62728',
    'escape_switch': '#9467bd',
    'no_label': '#7f7f7f'
}

def plot_fold_latent(
        latent_df: pd.DataFrame,
        fold: int,
        color_map: Dict[str, str] | None = None,
        title: str | None = None,
        *,
        save: bool = False,
        show: bool = False,
        out_dir: str | Path | None = None,
        basename: str | None = None,
        size: int = 700):
    """
    Plot ld1 vs ld2 scatter for a given fold with explicit class colors (square plot).

    Parameters
    ----------
    latent_df : DataFrame with columns: fold, behaviour, ld1, ld2
    fold : int
    color_map : dict behaviour -> color hex (optional)
    title : plot title (optional)
    save : whether to save (html + png/svg/pdf if possible)
    show : whether to display interactively
    out_dir : directory to save into (defaults to global outdir)
    basename : base filename (defaults to 'lda_latent_fold{fold}')
    size : width/height in pixels (square)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if color_map is None:
        color_map = DEFAULT_BEHAVIOUR_COLORS.copy()

    df_fold = latent_df[latent_df['fold'] == fold].copy()
    if df_fold.empty:
        raise ValueError(f"No rows found for fold {fold}.")

    missing = sorted(set(df_fold['behaviour']) - set(color_map))
    if missing:
        fallback_palette = [
            '#17becf', '#bcbd22', '#8c564b', '#e377c2',
            '#7f7f7f', '#aec7e8', '#ffbb78', '#98df8a'
        ]
        for i, beh in enumerate(missing):
            color_map[beh] = fallback_palette[i % len(fallback_palette)]

    df_fold['behaviour'] = df_fold['behaviour'].astype(str)
    beh_order = sorted(df_fold['behaviour'].unique())
    # NEW: ensure 'chase' is plotted last (on top)
    if 'chase' in beh_order:
        beh_order = [b for b in beh_order if b != 'chase'] + ['chase']
    fig = px.scatter(
        df_fold,
        x='ld1',
        y='ld2',
        color='behaviour',
        color_discrete_map={b: color_map[b] for b in beh_order},
        category_orders={'behaviour': beh_order},
        hover_data=['time', 'class_index'],
        title=title or f"LDA latent space (fold={fold})"
    )
    fig.update_layout(
        xaxis_title='LD1',
        yaxis_title='LD2',
        legend_title='Behaviour',
        template='plotly_white',
        width=size,
        height=size,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    # lock aspect ratio
    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    if save:
        if out_dir is None:
            out_dir = Path(outdir)
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if basename is None:
            basename = f"lda_latent_fold_{the_mouse}_{the_session}_fold_{fold}"
        html_path = out_dir / f"{basename}.html"
        fig.write_html(html_path, include_plotlyjs='cdn')
        for ext in ("png", "svg", "pdf"):
            try:
                fig.write_image(out_dir / f"{basename}.{ext}")
            except Exception as e:
                print(f"[WARN] export {ext} failed: {e}")
        print(f"[SAVE] Wrote latent plot to: {out_dir} (basename={basename})")

    if show:
        fig.show()

    return fig


##
# for cycle on all folds
for fold_ in range(n_folds_total):
    try:
        plot_fold_latent(latent_df, fold=fold_, save=True, show=False, out_dir=outdir)
    except Exception as e:
        print(f"[WARN] Could not create example plot for fold {fold_}: {e}")

# %%
