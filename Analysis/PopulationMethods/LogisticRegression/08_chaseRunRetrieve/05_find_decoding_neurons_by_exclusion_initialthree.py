# -*- coding: utf-8 -*-
"""
Created on 2025-08-28

@author: Dylan Festa

This script identifies neurons important for decoding with an erosion approach.

First it picks a random k-fold, then it fits and test it, then it 
progressively removes neurons that have the strongest weights.
Stores details at each step, until only ~5 neurons remain.

Based on this, I can figure out where to make a cut and which neurons I pick as imporant.
Every behavior is handled independently.

Flaw of this approach: cannot identify multi-modal neurons that are important for multiple behaviors.

Given the coding neurons, I find here, I will need another fit to figure out which ones 
are multi-modal.

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
DEFAULT_N_UNITS_STOP = 260  # NEW

# Fixed parameters
dt = 20*1E-3 # in ms
n_lags = 49
C = 10.0
penalty='l2'
n_folds_total=5

def _parse_cmdline(default_mouse: str, default_session: str, default_n_units_stop: int):
    mouse = default_mouse
    session = default_session
    n_units_stop = default_n_units_stop  # NEW
    for arg in sys.argv[1:]:
        if arg in ('-h', '--help'):
            print(f"Usage: python {os.path.basename(sys.argv[0])} mouse=<mouse_id> session=<session_id> n_units_stop=<int>")
            print(f"Defaults: mouse={default_mouse} session={default_session} n_units_stop={default_n_units_stop}")
            sys.exit(0)
        if arg.startswith('mouse='):
            mouse = arg.split('=', 1)[1]
        elif arg.startswith('session='):
            session = arg.split('=', 1)[1]
        elif arg.startswith('n_units_stop='):
            try:
                n_units_stop = int(arg.split('=', 1)[1])
            except ValueError:
                print("[WARN] Invalid n_units_stop value; using default.")
    return mouse, session, n_units_stop  # CHANGED

the_animal, the_session, n_units_stop = _parse_cmdline(DEFAULT_MOUSE, DEFAULT_SESSION, DEFAULT_N_UNITS_STOP)  # CHANGED
print(f"[INFO] Using mouse={the_animal} session={the_session} n_units_stop={n_units_stop}")  # CHANGED


interesting_behaviours = ['chase', 'pup_run', 'pup_retrieve','escape','escape_switch']
required_behaviours = [behaviour\
        for behaviour in interesting_behaviours if (behaviour != 'escape') and (behaviour != 'escape_switch')]


print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(the_animal, the_session)
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

behaviour_timestamps_df_all = rdl.convert_to_behaviour_timestamps(the_animal, the_session, data_behaviour)

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
    'mouse': the_animal,
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



#%%



#%% Define train-test split


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


def get_train_test_data_random(X:np.ndarray,y:np.ndarray):
    k_random = np.random.randint(0, n_folds_total)
    return get_train_test_data(X,y,k_random)

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
outdir = os.path.join(os.path.dirname(__file__), "local_outputs_05byexclusion", f"{the_animal}_{the_session}")
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
    t_behaviour = t_alltime[y_idx_behavior]
    X_train_,y_train_,X_test_,y_test_ = get_train_test_data_random(X_behaviour_maxlagged,y_behaviour)
    pipe_ = build_pipeline()
    t0 = time.time()
    pipe_.fit(X_train_, y_train_)
    fit_secs = time.time() - t0
    print(f"Fitting took {fit_secs:.2f}s")
    y_pred_ = pipe_.predict(X_test_)
    report_ = classification_report(y_test_, y_pred_, output_dict=True, zero_division=0)
    lr_ = pipe_.named_steps['lda']
    # dictionary behaviour_idx to weight_score for all neurons
    coefs = lr_.coef_  # shape: (n_classes, n_features)
    weight_score_dict = {}
    # row by row
    for k in range(coefs.shape[0]):
        class_index = lr_.classes_[k]
        behaviour = dict_classindex_to_behaviour.get(class_index, 'ERROR')
        if behaviour == 'ERROR':
            raise ValueError(f"Class index {class_index} not found in dict_classindex_to_behaviour.")
        coef_2d_abs =np.abs(coefs[k].reshape(n_lags + 1, n_units_less))
        scores_all_units = coef_2d_abs.sum(axis=0)  # sum across lags
        weight_score_dict[behaviour] = {behaviour:scores_all_units}

    # dictionary for classification, same as classification report, but
    # keys are behaviours and not indices
    classification_dict = {}
    for class_,behaviour in dict_classindex_to_behaviour.items():
        class_str_ = str(class_)
        if not class_str_ in report_.keys():
            continue
        classification_dict[behaviour] = report_[class_str_]

    output_df_rows_ = []
    for beh_,weight_scores in weight_score_dict.items():
        for (k,unit) in enumerate(units_less):
            weight_score_ = float(weight_scores[beh_][k])
            f1_score_ = classification_dict.get(beh_, {}).get('f1-score', np.nan)
            support_ = classification_dict.get(beh_, {}).get('support', np.nan)
            if np.isnan(f1_score_) or np.isnan(support_) or support_ == 0:
                raise ValueError(f"F1 score for behaviour '{beh_}' not found!?")
            output_df_rows_.append({
                'unit': unit,
                'behaviour': beh_,
                'weight_score': weight_score_,
                'f1_score': f1_score_,
                'support': support_
            })
    output_df = pd.DataFrame(output_df_rows_, columns=['unit','behaviour','weight_score','f1_score','support'])
    # also output smaller df with behaviour, score, support 
    output_df_reduced = output_df[['behaviour','f1_score','support']].drop_duplicates().reset_index(drop=True)
    # Cleanup RAM/GPU before next iteration
    del X_train_, y_train_, X_test_, y_test_, y_pred_, pipe_, lr_
    gc.collect()
    _free_gpu()
    return units_less,output_df,output_df_reduced



def _print_behaviour_scores(df: pd.DataFrame, iteration: int):
    if df is None or df.empty:
        print(f"\n=== Behaviour scores (iteration {iteration}) ===")
        print("No behaviour rows to display.")
        return
    cols = [c for c in ['behaviour','f1_score','support'] if c in df.columns]
    df_show = df[cols].drop_duplicates().copy()
    if 'f1_score' in df_show.columns:
        df_show['f1_score'] = df_show['f1_score'].map(lambda v: f"{v:.3f}")
    if 'support' in df_show.columns and df_show['support'].notna().all():
        # cast to int if they are whole numbers
        df_show['support'] = df_show['support'].astype(int)
    df_show = df_show.sort_values(by='behaviour', ascending=False)
    width_beh = max(9, df_show['behaviour'].str.len().max())
    print(f"\n=== Behaviour scores (iteration {iteration}) ===")
    header = f"{'behaviour'.ljust(width_beh)}  f1_score  support"
    print(header)
    print("-" * len(header))
    for _, r in df_show.iterrows():
        print(f"{r['behaviour'].ljust(width_beh)}  {r['f1_score']:>8}  {r.get('support',''):>7}")


def scores_by_reduction(n_units_stop:int=100):
    # run first fit
    units_all,start_df,beh_start_df = run_one_fit([])

    out_dict_first = {
        'units': units_all,
        'output_df': start_df,
        'output_beh_df': beh_start_df
    }

    # now for each behaviour, keep the 3 highest weight_score rows
    # (if a behaviour has <3 units, all its rows are kept)
    start_df_top3 = (
        start_df
        .sort_values(['behaviour', 'weight_score'], ascending=[True, False])
        .groupby('behaviour', group_keys=False)
        .head(3)
        .copy()
    )
    start_df_top3.reset_index(drop=True, inplace=True)

    out_dict_first['output_df_top3'] = start_df_top3

    units_keep_start = start_df_top3['unit'].unique().tolist() 

    n_units_left = len(units_all) - len(units_keep_start)
    units_to_exclude = units_keep_start
    out_dict_first['units_first_excluded'] = units_to_exclude
    print(f"Starting reduction with {n_units_left} units.")

    reduction_log_df_rows = []
    iteration = 0

    while n_units_left > n_units_stop:
        print(f"\nFitting with {n_units_left} units (iteration {iteration})...")
        # Fit using all units except those in units_to_exclude
        units_left_here, df_fit, beh_df_fit = run_one_fit(units_to_exclude)

        # show beh_df_fit, behavior and f1_score columns, plotted neatly
        _print_behaviour_scores(beh_df_fit, iteration)

        # Sum weight_score across behaviours per unit
        weight_scores = df_fit.groupby('unit')['weight_score'].sum()

        if weight_scores.empty:
            raise ValueError("No weight scores available. This should not happen")

        # Identify unit with highest cumulative weight (to exclude next)
        unit_to_remove = weight_scores.idxmax()
        max_score = weight_scores.loc[unit_to_remove]
        print(f"Marking unit {unit_to_remove} (sum weight_score={max_score:.4f}) for exclusion in next iteration.")

        # Store outputs for this cycle BEFORE exclusion of selected unit
        output_dict_cycle = {
            'units': units_left_here,
            'output_df': df_fit,
            'output_beh_df': beh_df_fit
        }
        reduction_log_df_rows.append({
            'iteration': iteration,
            'neuron_excluded': unit_to_remove,
            'weight_score_summed': max_score,
            'output_dict': output_dict_cycle
        })

        # Apply exclusion for next loop
        units_to_exclude.append(unit_to_remove)

        # Update counters
        n_units_left = len(units_all) - len(units_to_exclude)
        iteration += 1

    # Build log dataframe
    reduction_log_df = pd.DataFrame(reduction_log_df_rows, columns=['iteration','neuron_excluded','weight_score_summed','output_dict'])

    # Return both initial snapshot and erosion log
    return out_dict_first, reduction_log_df


#%%

#test1_,test2_,test3_ = run_one_fit([])  # fit with all units


# VEEERY LONG!
first_out_dict,reduction_log_df = scores_by_reduction(n_units_stop)  # CHANGED (was 260)


#%%
# save as binary files in output directory with pickle

print(f"[INFO] Pickle saving to: {outdir}")
_first_pickle = os.path.join(outdir, "first_out_dict.pkl")
_reduction_pickle = os.path.join(outdir, "reduction_log_df.pkl")
with open(_first_pickle, "wb") as f:
    pickle.dump(first_out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(_reduction_pickle, "wb") as f:
    pickle.dump(reduction_log_df, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[INFO] Saved first_out_dict (pickle) -> {_first_pickle}")
print(f"[INFO] Saved reduction_log_df (pickle) -> {_reduction_pickle}")

print(f"[INFO] Pickle saving completed! Exiting!")

exit()


#%%

#%%
scores_df_rows =[]
# for each row if test2_
for i, row in reduction_log_df.iterrows():
    iteration_ = row['iteration']
    out_beh_df = row['output_dict']['output_beh_df']
    f1_chase = out_beh_df.loc[out_beh_df['behaviour'] == 'chase', 'f1_score'].values[0]
    f1_pup_run = out_beh_df.loc[out_beh_df['behaviour'] == 'pup_run', 'f1_score'].values[0]
    f1_pup_retrieve = out_beh_df.loc[out_beh_df['behaviour'] == 'pup_retrieve', 'f1_score'].values[0]
    f1_no_label = out_beh_df.loc[out_beh_df['behaviour'] == 'no_label', 'f1_score'].values[0]   
    scores_df_rows.append({
        'iteration': iteration_,
        'f1_chase': f1_chase,
        'f1_pup_run': f1_pup_run,
        'f1_pup_retrieve': f1_pup_retrieve,
        'f1_no_label': f1_no_label
    })
scores_df = pd.DataFrame(scores_df_rows)

#%%
if not scores_df.empty:
    fig = go.Figure()
    behaviours_map = [
        ('f1_chase', 'chase', '#1f77b4'),
        ('f1_pup_run', 'pup_run', '#ff7f0e'),
        ('f1_pup_retrieve', 'pup_retrieve', '#2ca02c'),
        ('f1_no_label', 'no_label', '#7f7f7f'),
    ]
    for col, name, color in behaviours_map:
        if col in scores_df.columns:
            fig.add_trace(go.Scatter(
                x=scores_df['iteration'],
                y=scores_df[col],
                mode='lines+markers',
                name=name,
                line=dict(width=2, color=color),
                marker=dict(size=6)
            ))
    fig.update_layout(
        title='F1 scores vs neuron exclusion iteration',
        xaxis_title='number of excluded neurons',
        yaxis_title='F1 score',
        template='plotly_white',
        legend_title='Behaviour',
        hovermode='x unified',
        height=600  # NEW: make plot taller
    )
    # NEW: horizontal grid lines every 0.1
    fig.update_yaxes(showgrid=True, dtick=0.1, gridcolor='rgba(0,0,0,0.2)', gridwidth=1)
    # plot_path = os.path.join(outdir, 'f1_scores_by_iteration.html')
    # fig.write_html(plot_path)
    fig.show()
else:
    print("[WARN] scores_df is empty; skipping F1 plot.")


# %%

cutoff_val = 40


#%%


iter_scores_df_rows = []
for i, row in reduction_log_df.iterrows():
    iteration_ = row['iteration']
    output_df = row['output_dict']['output_df']
    # add iteration column to df
    output_df = output_df.copy()
    output_df['iteration'] = iteration_
    iter_scores_df_rows.append(output_df)

iter_scores_df = pd.concat(iter_scores_df_rows, ignore_index=True)

# filter up to cutoff value
iter_scorescut_df = iter_scores_df.query(f"iteration <= {cutoff_val}").copy()


#%%
# now take MAX weight_score grouping by behaviour and unit (across iterations)
iter_scores_beh_df = (
    iter_scorescut_df
    .groupby(['behaviour', 'unit'], as_index=False)['weight_score']
    .max()
)

#%% remove no_label and chase
iter_scores_beh_df_less = iter_scores_beh_df.query("behaviour != 'no_label' and behaviour != 'chase'").copy()

#%% for each neuron, select max weight_score and behaviour
iter_scores_max_df = iter_scores_beh_df.loc[iter_scores_beh_df_less.groupby('unit')['weight_score'].idxmax()]


# weight score cutoff is the minimum of the weigth_score column
weight_score_cutoff = iter_scores_max_df['weight_score'].min()

#%%

# --- NEW: density plots of weight_score per behaviour (one figure per behaviour) ---
behaviours_for_plot = iter_scores_beh_df['behaviour'].unique()
for beh in behaviours_for_plot:
    df_b = iter_scores_beh_df[iter_scores_beh_df['behaviour'] == beh]
    if df_b.empty:
        print(f"[WARN] No data for behaviour '{beh}', skipping density plot.")
        continue

    # Base histogram (density normalized)
    fig = px.histogram(
        df_b,
        x='weight_score',
        histnorm='density',
        nbins=min(60, max(10, df_b['weight_score'].nunique())),
        opacity=0.55
    )

    # Add KDE curve if possible
    xs = None
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        vals = df_b['weight_score'].values
        if np.allclose(vals.min(), vals.max()):
            # All values identical; create a narrow spike
            xs = np.linspace(vals.min() - 1e-6, vals.max() + 1e-6, 50)
            kde_y = np.zeros_like(xs)
            kde_y[len(kde_y)//2] = 1.0
        else:
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            kde_y = kde(xs)
        fig.add_trace(go.Scatter(
            x=xs, y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='black', width=2)
        ))
    except Exception:
        # Fallback: simple density line from numpy histogram
        counts, edges = np.histogram(df_b['weight_score'].values, bins=40, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        fig.add_trace(go.Scatter(
            x=centers, y=counts,
            mode='lines',
            name='hist density',
            line=dict(color='black', width=2)
        ))

    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(
        title=f"Density of weight_score for behaviour '{beh}'",
        xaxis_title='weight_score',
        yaxis_title='density',
        template='plotly_white',
        bargap=0.05
    )
    # --- NEW: vertical cutoff line ---
    fig.add_vline(
        x=weight_score_cutoff,
        line_color='red',
        line_dash='dash',
        annotation_text=f"cutoff={weight_score_cutoff:.3f}",
        annotation_position='top'
    )
    # --- END NEW ---
    fig.show()

#%%
# --- NEW: stacked bar plot of fractional weight_score contribution per unit ---
if not iter_scores_beh_df.empty:
    _df_frac = iter_scores_beh_df.copy()
    totals = _df_frac.groupby('unit')['weight_score'].sum().rename('total_weight')
    _df_frac = _df_frac.merge(totals, on='unit', how='left')
    _df_frac = _df_frac[_df_frac['total_weight'] > 0].copy()
    _df_frac['fraction'] = _df_frac['weight_score'] / _df_frac['total_weight']
    frac_wide = _df_frac.pivot_table(index='unit', columns='behaviour', values='fraction', fill_value=0.0)
    # --- CHANGED: order units by pup_run fraction descending (fallback: total weight) ---
    order_units = frac_wide.sort_values('pup_retrieve', ascending=False).index.tolist()
    frac_wide = frac_wide.reindex(order_units)
    # --- END CHANGED ---
    fig = go.Figure()
    color_map = {
        'chase': '#1f77b4',
        'pup_run': '#ff7f0e',
        'pup_retrieve': '#2ca02c',
        'no_label': '#7f7f7f'
    }
    for beh in frac_wide.columns:
        fig.add_trace(go.Bar(
            x=frac_wide.index.astype(str),
            y=frac_wide[beh].values,
            name=beh,
            marker=dict(color=color_map.get(beh, None)),
            hovertemplate="unit=%{x}<br>behaviour="+beh+"<br>fraction=%{y:.3f}<extra></extra>"
        ))
    fig.update_layout(
        barmode='stack',
        title='Fractional weight_score contribution per unit (sorted by pup_retrieve fraction)',
        xaxis_title='Unit',
        yaxis_title='Fraction of total weight_score',
        yaxis=dict(range=[0,1]),
        template='plotly_white',
        legend_title='Behaviour',
        hovermode='x unified',
        bargap=0.15
    )
    fig.show()
else:
    print("[WARN] iter_scores_beh_df empty; skipping stacked bar plot.")
# --- END NEW ---



# %%
