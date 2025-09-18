# -*- coding: utf-8 -*-
"""
Created on 2025-08-31

@author: Dylan Festa

Once the top units are found, compare fits. Fit of all, fit of top only
fit of all with top.
f1-score is averaged across 5 folds.

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
path_data = path_this_file.parent / "local_outputs_05byexclusion"
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
from sklearn.linear_model import LogisticRegression

# to check explicitely the weights used for different classes
from sklearn.metrics import classification_report, confusion_matrix


# simple argument parsing (before using the_animal/the_session) ---
DEFAULT_MOUSE = 'afm16924'
DEFAULT_SESSION = '240529'

# Fixed parameters
dt = 20*1E-3 # in ms
n_lags = 49
C = 10.0
penalty='l2'
n_folds_total=5

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
    return Pipeline([
        ('regularizer',StandardScaler()),
        ('lda', LogisticRegression(
                        penalty=penalty, 
                        solver='liblinear',
                        class_weight='balanced',
                        verbose=False,
                        n_jobs=1,
                        C=C,
                        max_iter=5000,
                        tol=1e-4,
                        random_state=0,
        )),
    ])

# FIX: use the_animal / the_session (previously undefined animal/session)
outdir = os.path.join(os.path.dirname(__file__), "local_outputs_05byexclusion", f"{the_mouse}_{the_session}")
os.makedirs(outdir, exist_ok=True)

# NEW: save behaviour/label metadata to disk (binary + human-readable)
labels_meta_path = os.path.join(outdir, "labels_meta.joblib")
dump(
    {
        'C': C,
        'penalty': penalty,
        'dt': dt,
        'behaviour_timestamps_df': behaviour_timestamps_df,
        'dict_behaviour_label_to_index': dict_behaviour_label_to_index,
        'dict_classindex_to_behaviour': dict_classindex_to_behaviour,
        'n_units': n_units,
        'units': the_units,
        'unit_locations': the_unit_locations,
    },
    labels_meta_path,
    compress=3
)
behaviour_timestamps_df.to_csv(os.path.join(outdir, "behaviour_timestamps_df.csv"), index=False)
with open(os.path.join(outdir, "dict_behaviour_label_to_index.json"), "w") as f:
    json.dump(dict_behaviour_label_to_index, f, indent=2)
with open(os.path.join(outdir, "dict_classindex_to_behaviour.json"), "w") as f:
    json.dump(dict_classindex_to_behaviour, f, indent=2)


def run_one_fit(units_to_exclude: List):
    if units_to_exclude is None or len(units_to_exclude) == 0:
        units_to_keep = the_units
    else:
        units_to_keep = list(set(the_units) - set(units_to_exclude))

    if len(units_to_keep) == 0:
        raise ValueError("No units left to fit!!!")

    spiketrains_ = spiketrains.filter_by_units(units_to_keep)
    units_less = spiketrains_.units
    n_units_less = len(units_less)
    
    X_alltime=pre.do_binning_operation(spiketrains_,
                        'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)
    # get labels
    beh_labels_xr = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                            t_start =0.0,t_stop= t_stop_all,
                                            dt=dt,
                                            behaviour_labels_dict=dict_behaviour_label_to_index)
    y_alltime = beh_labels_xr.values
    if len(y_alltime) != X_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after binning.")
    # apply lag
    X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion_xr(X_alltime, n_lags)
    # drop NaN rows from X and y
    X_alltime_maxlagged = X_alltime_maxlagged[n_lags:,:]  # keep only the last n_lags_max rows
    y_alltime = y_alltime[n_lags:]  # keep only the last n_lags_max rFalsews
    if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")
    t_alltime = X_alltime_maxlagged.coords['time_bin_center'].values
    # now, select only behavioural labels
    y_idx_behavior = y_alltime != -1  # -1 is the 'none' label
    y_behaviour = y_alltime[y_idx_behavior]
    X_behaviour_maxlagged = X_alltime_maxlagged.values[y_idx_behavior, :]
    # For each fold
    df_output_full_all = []
    df_output_reduced_all = []
    t_start_folds=time.time()
    for fold_ in range(n_folds_total):
        X_train_,y_train_,X_test_,y_test_ = get_train_test_data(X_behaviour_maxlagged,y_behaviour,fold_)
        pipe_ = build_pipeline()
        t0 = time.time()
        pipe_.fit(X_train_, y_train_)
        fit_secs = time.time() - t0
        print(f"Fitting (fold {fold_}) took {fit_secs:.2f}s")
        y_pred_ = pipe_.predict(X_test_)
        report_ = classification_report(y_test_, y_pred_, output_dict=True, zero_division=0)
        lr_ = pipe_.named_steps['lda']
        # dictionary behaviour_idx to weight_score for all neurons
        coefs = lr_.coef_  # shape: (n_classes, n_features)
        weight_score_dict = {}
        weight_sign_dict = {}
        # row by row
        for k in range(coefs.shape[0]):
            class_index = lr_.classes_[k]
            behaviour = dict_classindex_to_behaviour.get(class_index, 'ERROR')
            if behaviour == 'ERROR':
                raise ValueError(f"Class index {class_index} not found in dict_classindex_to_behaviour.")
            coef_2d = coefs[k].reshape(n_lags + 1, n_units_less)
            coef_2d_abs = np.abs(coef_2d)
            coef_abs_sums = coef_2d_abs.sum(axis=0)
            coef_sums = coef_2d.sum(axis=0)
            scores_all_units = coef_abs_sums
            signs_all_units = np.sign(coef_sums)
            # turn 0 into +1
            signs_all_units[signs_all_units == 0] = 1
            # FIX: remove redundant nested key {behaviour: array}
            weight_score_dict[behaviour] = scores_all_units
            weight_sign_dict[behaviour] = signs_all_units

        # dictionary for classification, same as classification report, but
        # keys are behaviours and not indices
        classification_dict = {}
        for class_,behaviour in dict_classindex_to_behaviour.items():
            class_str_ = str(class_)
            if not class_str_ in report_.keys():
                continue
            classification_dict[behaviour] = report_[class_str_]

        output_df_rows_ = []
        for beh_, scores_all_units in weight_score_dict.items():
            signs_all_units = weight_sign_dict[beh_]
            for (k, unit) in enumerate(units_less):
                weight_score_ = float(scores_all_units[k])
                f1_score_ = classification_dict.get(beh_, {}).get('f1-score', np.nan)
                support_ = classification_dict.get(beh_, {}).get('support', np.nan)
                sign_ = int(signs_all_units[k])  # retained (not stored) for possible future use
                if np.isnan(f1_score_) or np.isnan(support_) or support_ == 0:
                    raise ValueError(f"F1 score for behaviour '{beh_}' not found!?")
                output_df_rows_.append({
                    'fold': fold_,
                    'unit': unit,
                    'behaviour': beh_,
                    'weight_score': weight_score_,
                    'f1_score': f1_score_,
                    'support': support_,
                    'weight_sign': sign_,
                })
        output_df = pd.DataFrame(output_df_rows_, columns=['fold','unit','behaviour','weight_score','weight_sign','f1_score','support'])
        # also output smaller df with behaviour, score, support 
        output_df_reduced = output_df[['fold','behaviour','f1_score','support']].drop_duplicates().reset_index(drop=True)
        # Cleanup RAM/GPU before next iteration
        del X_train_, y_train_, X_test_, y_test_, y_pred_, pipe_, lr_
        gc.collect()
        _free_gpu()
        df_output_full_all.append(output_df)
        df_output_reduced_all.append(output_df_reduced)
    output_df = pd.concat(df_output_full_all, ignore_index=True)
    output_df_reduced = pd.concat(df_output_reduced_all, ignore_index=True)
    t_end_folds = time.time()
    print(f"Total time for all folds: {t_end_folds - t_start_folds:.2f}s")
    return units_less,output_df,output_df_reduced

#%%

test1_,output_df_all,performance_beh_allfolds_df = run_one_fit(units_to_exclude=[])




#%%
# read dictionary top_subset_df.plk
# HELP!

# Load the top subset dictionary saved by 05_read_and_plot.py
top_subset_pkl_path = path_data / "top_subset_df.pkl"
if not top_subset_pkl_path.exists():
    raise FileNotFoundError(f"Top subset dictionary not found: {top_subset_pkl_path}")
with open(top_subset_pkl_path, "rb") as f:
    top_subset_dict = pickle.load(f)
print(f"[INFO] Loaded top_subset_df.pkl (keys={list(top_subset_dict.keys())})")
# Optional: DataFrame view (commented to keep it minimal)
top_subset_df = pd.DataFrame(top_subset_dict['data'])
# select mouse and session
top_subset_df = top_subset_df.query("mouse == @the_mouse and session == @the_session").copy()


# %%

the_top_units = top_subset_df['unit'].unique()
n_units = len(the_units)
n_top_units = len(the_top_units)
n_nottop_units = n_units - n_top_units
print(f"[INFO] Number of top units: {n_top_units}")

# %%
_,output_df_notop,performance_beh_allfolds_notop_df = run_one_fit(units_to_exclude=the_top_units)

# the_units but without the_top_units
the_units_no_top = [u for u in the_units if u not in set(the_top_units)]

_,output_df_toponly,performance_beh_allfolds_toponly_df = run_one_fit(units_to_exclude=the_units_no_top)


#%%

#%%
# for each behaviour, average across folds
performance_all_df = performance_beh_allfolds_df.groupby('behaviour').mean().reset_index()
performance_all_df = performance_all_df.drop(columns=['fold'])

performance_toponly_df = performance_beh_allfolds_toponly_df.groupby('behaviour').mean().reset_index()
performance_toponly_df = performance_toponly_df.drop(columns=['fold'])

performance_notop_df = performance_beh_allfolds_notop_df.groupby('behaviour').mean().reset_index()
performance_notop_df = performance_notop_df.drop(columns=['fold'])




#%% Bar plot function comparing F1 across unit subsets

# Hexadecimal colors
COLOR_ALL_UNITS = "#77797a"
COLOR_TOP_ONLY = "#1f00ce"
COLOR_NO_TOP = "#cf56ff"

def plot_f1_bar(perf_all_df: pd.DataFrame,
                perf_top_df: pd.DataFrame,
                perf_notop_df: pd.DataFrame,
                *,
                mouse: str,
                session: str,
                show: bool = True,
                save: bool = False,
                out_dir: str | Path | None = None,
                title: str | None = None):
    """
    Uses already aggregated dataframes (one row per behaviour) with columns:
      - behaviour
      - f1_score
      - support (needed only in perf_all_df for annotation; ignored if absent in others)
    Unit counts are taken from outer-scope variables:
      n_units, n_top_units, n_nottop_units (see lines where they are defined).
    """
    # Minimal column checks
    for name, df in [('all', perf_all_df), ('top', perf_top_df), ('no_top', perf_notop_df)]:
        if 'behaviour' not in df.columns or 'f1_score' not in df.columns:
            raise ValueError(f"Input DF '{name}' must contain 'behaviour' and 'f1_score'.")

    # Pull unit counts from outer scope
    try:
        n_units_all = n_units
        n_units_top = n_top_units
        n_units_notop = n_nottop_units
    except NameError as e:
        raise RuntimeError("Unit count variables (n_units, n_top_units, n_nottop_units) not defined in outer scope.") from e

    # If the dataframes were NOT aggregated (multiple rows per behaviour), aggregate now (idempotent)
    def _ensure_agg(df: pd.DataFrame):
        if df.groupby('behaviour').size().max() > 1:
            return df.groupby('behaviour', as_index=False)['f1_score'].mean()
        return df[['behaviour', 'f1_score']].copy()
    agg_all = _ensure_agg(perf_all_df)
    agg_top = _ensure_agg(perf_top_df)
    agg_notop = _ensure_agg(perf_notop_df)

    # Support (only from all-units df if present)
    if 'support' in perf_all_df.columns:
        support_series = perf_all_df.groupby('behaviour')['support'].mean().round().astype(int)
    else:
        support_series = pd.Series(dtype=int)

    beh_order = agg_all['behaviour'].tolist()

    # Use legend labels with unit counts
    label_all = f"all ({n_units_all})"
    label_top = f"top-coding ({n_units_top})"
    label_notop = f"top-coding excluded ({n_units_notop})"

    tidy_df = pd.concat([
        agg_all.assign(model=label_all),
        agg_top.assign(model=label_top),
        agg_notop.assign(model=label_notop),
    ], ignore_index=True)
    tidy_df['behaviour'] = pd.Categorical(tidy_df['behaviour'], categories=beh_order, ordered=True)

    color_map = {
        label_all: COLOR_ALL_UNITS,
        label_top: COLOR_TOP_ONLY,
        label_notop: COLOR_NO_TOP
    }

    if title is None:
        title = f"f1-score, neurons comparison (mouse={mouse}, session={session})"

    fig = px.bar(
        tidy_df,
        x='behaviour',
        y='f1_score',
        color='model',
        barmode='group',
        category_orders={'behaviour': beh_order,
                         'model': [label_all, label_top, label_notop]},
        color_discrete_map=color_map,
        title=title
    )
    fig.update_layout(
        xaxis_title='Behaviour',
        yaxis_title='F1 score',
        legend_title='Subset (units)',
        bargap=0.15,
        bargroupgap=0.05,
        template='plotly_white'
    )

    max_f1_overall = tidy_df['f1_score'].max()
    fig.update_yaxes(range=[0, min(1.05, max(1.0, max_f1_overall + 0.08))])

    # Add integer support annotations (from all units only)
    for beh in beh_order:
        beh_max = tidy_df.loc[tidy_df['behaviour'] == beh, 'f1_score'].max()
        supp = support_series.get(beh, None)
        if pd.notna(supp):
            fig.add_annotation(
                x=beh,
                y=beh_max + 0.04,
                text=f"{int(supp)}",
                showarrow=False,
                font=dict(size=11, color="#222"),
                yanchor='bottom'
            )

    if save:
        if out_dir is None:
            out_dir = Path(outdir)
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"f1_bar_compare_{mouse}_{session}"
        html_path = out_dir / f"{base_name}.html"
        fig.write_html(html_path)
        # Multi-format static exports
        for ext in ("png", "svg", "pdf"):
            try:
                fig.write_image(out_dir / f"{base_name}.{ext}")
            except Exception as e:
                print(f"[WARN] {ext} export failed: {e}")
        print(f"[SAVE] Wrote plot (html + static formats) to: {out_dir}")
    if show:
        fig.show()
    return fig

#%% Generate and save/show bar plot
_ = plot_f1_bar(perf_all_df=performance_all_df,
                perf_top_df=performance_toponly_df,
                perf_notop_df=performance_notop_df,
                mouse=the_mouse,
                session=the_session,
                show=False,
                save=True,
                out_dir=outdir)

#%%

#exit()
#%%
# additional code to save df ranking
# must save a df with unit,behaviour,rank_val_rel
# using output_df_toponly


def _majority_sign(series: pd.Series) -> int:
    """Return sign of majority; 1 if more positives, -1 if more negatives, +1 on tie."""
    s = series.sum()
    if s > 0:
        return 1
    if s < 0:
        return -1
    return +1  # tie (e.g., equal counts)


# output_df_toponly but only for unit,behaviour combinations that appear in 
# top_subset_df
if 'behaviour' not in top_subset_df.columns:
    raise KeyError("Expected column 'behaviour' in top_subset_df.")
_top_pairs = top_subset_df[['unit','behaviour']].drop_duplicates()
output_df_selected_top_only = (
    output_df_toponly.merge(_top_pairs, on=['unit','behaviour'], how='inner')
)


df_rankings = (
    output_df_selected_top_only
    .groupby(['unit', 'behaviour'], as_index=False)
    .agg(
        weight_score=('weight_score', 'mean'),
        weight_sign=('weight_sign', _majority_sign),
    )
)

df_rankings['rank_val'] = df_rankings['weight_score'] * df_rankings['weight_sign']

#%%
# add column rank_val_rel, that goes from -1 to 1
# so that zero stays zero, min is -1 , max is +1
def rescale_rank_val_signed(df, col="rank_val", 
                            condition_col="behaviour",
                            new_col="rank_val_rel"):
    """
    Per-behaviour (group) scaling of `col` into [-1,1].
    Rules per group:
      - If group has both negative (<0) and positive (>0) values:
            min -> -1, max -> +1, 0 stays 0 (piecewise linear on each side).
      - If only positives (min >= 0): values / max -> [0,1].
      - If only negatives (max <= 0): values / abs(min) -> [-1,0].
      - If all zeros: stays 0.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe.")

    def _scale_group(g: pd.DataFrame) -> pd.DataFrame:
        s = g[col]
        s_min = s.min()
        s_max = s.max()

        if (s_min < 0) and (s_max > 0):
            # Mixed signs
            neg_scale = -s_min  # positive
            pos_scale = s_max   # positive
            scaled = s.copy()
            neg_mask = s < 0
            pos_mask = s > 0
            scaled[neg_mask] = s[neg_mask] / neg_scale
            scaled[pos_mask] = s[pos_mask] / pos_scale
            scaled[s == 0] = 0.0
        elif s_max > 0:  # only non‑negative (some positives or all zero)
            if s_max == 0:
                scaled = s * 0.0
            else:
                scaled = s / s_max
        elif s_min < 0:  # only non‑positive (some negatives)
            scaled = s / (-s_min)  # s_min is negative → -s_min positive
        else:
            # all zeros
            scaled = s * 0.0

        g[new_col] = scaled.astype(float)
        return g

    return (
        df.groupby(condition_col, group_keys=False)
          .apply(_scale_group)
          .reset_index(drop=True)
    )

df_session_ranking = rescale_rank_val_signed(df_rankings, col="rank_val", new_col="rank_val_rel")


#%%
# Now the rest of the script...
df_export = df_session_ranking.copy()


#%%
# ---------- helpers for colors ----------
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def _lerp(a, b, t):
    return tuple(int(round((1-t)*a[i] + t*b[i])) for i in range(3))

# Diverging palette: blue (-1) → gray (0) → red (+1)
POS = _hex_to_rgb('#2c7bb6')  # blue
NEU = _hex_to_rgb('#bdbdbd')  # neutral gray
NEG = _hex_to_rgb('#d73027')  # red

def strength_to_color(s):
    if pd.isna(s):
        return _rgb_to_hex(NEU)
    s = max(-1.0, min(1.0, float(s)))
    if s >= 0:
        c = _lerp(NEU, POS, s)
    else:
        c = _lerp(NEU, NEG, -s)
    return _rgb_to_hex(c)

# ---------- build nodes ----------
units = sorted(df_export["unit"].unique())
behaviours = sorted(df_export["behaviour"].unique())

nodes = []
# Units on level 0 (for a clean two‑column hierarchical layout)
for u in units:
    nodes.append({
        "id": f"u{u}",
        "label": str(u),
        "group": "unit",
        "level": 0
    })

# behaviours on level 1
for b in behaviours:
    nodes.append({
        "id": f"b:{b}",
        "label": str(b),
        "group": "behaviour",
        "level": 1
    })

# ---------- build edges ----------
has_strength = "rank_val_rel" in df_export.columns
edges = []
for row in df_export.itertuples(index=False):
    u = getattr(row, "unit")
    b = getattr(row, "behaviour")
    edge = {
        "from": f"u{u}",
        "to":   f"b:{b}",
        "smooth": False
    }
    if has_strength:
        s = getattr(row, "rank_val_rel")
        edge["strength"] = None if pd.isna(s) else float(s)   # keep raw strength if you'd rather color in JS
        # If you prefer to bake color/width in Python, uncomment the next two lines:
        edge["color"] = strength_to_color(s)
        edge["width"] = 1 + 3*abs(0.0 if pd.isna(s) else float(s))  # 1–4 px
        edge["title"] = f"strength: {float(s):+.2f}" if not pd.isna(s) else "strength: n/a"
    edges.append(edge)

# ---------- write JSON ----------

savedir = outdir

with open(f"{savedir}/network_data.json", "w", encoding="utf-8") as f:
    json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

print("Wrote network_data.json with", len(nodes), "nodes and", len(edges), "edges.")

# %%
