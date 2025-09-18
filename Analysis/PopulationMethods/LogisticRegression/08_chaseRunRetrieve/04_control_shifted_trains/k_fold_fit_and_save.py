# -*- coding: utf-8 -*-
"""
Created on 2025-08-26

@author: Dylan Festa

clone of `02_k_fold_fit_and_save.py`, but it shifts spiketrains randomly in time
as a control, shuffling the temporal structure of the data and destroying correlations so that labels should lose
meaning.  Of course `no_label` might still be correctly labelled at the single neuron level.

This control sill preserves unit identity and mean rate of each unit.
"""
#%%
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


# --- NEW: simple argument parsing (before using the_animal/the_session) ---
DEFAULT_MOUSE = 'afm16924'
DEFAULT_SESSION = '240529'

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

the_animal, the_session = _parse_cmdline(DEFAULT_MOUSE, DEFAULT_SESSION)
print(f"[INFO] Using mouse={the_animal} session={the_session}")
# --------------------------------------------------------------------------

#the_animal = 'afm17365'
# (Removed hard-coded assignments; now handled above)
# the_animal = 'afm16924'
# the_session = '240529'


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


spiketrains_unshuffled=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains_unshuffled = spiketrains_unshuffled.filter_by_unit_location('PAG')

# WARNING: this is the part where I reshuffle the spike trains, so that they are expected to lose
# meaning! Don't use this outside of negative controls!
print("WARNING: shuffling ALL spike trains! This is a control!")
t_shuffle_min = 3*60.0 # shifts trains of at least 3 minutes!
spiketrains = spiketrains_unshuffled.generate_shuffled_control(
    shuffle_neurons=False,
    minimum_shift=t_shuffle_min
)
n_units = spiketrains.n_units
the_units = spiketrains.units
the_unit_locations = spiketrains.unit_location
print(f"Number of PAG units: {n_units}.")
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
print(f"Smallest frames_s element is at: {time_min_str} (MM:SS)\n")
print("Contents of the row:")
print(smallest_frame_row)


#%%

t_nolabel_start = 1 * 60.0  # 1 minute after the start of the session
# end is min between 8 minutes and 2 min before first behavior
t_nolabel_end = np.min([8 * 60.0, time_first_beh - 2 * 60.0])

# #time_first_beh - 2 * 60.0  # 2 minutes before the first behavior

k_intervals = 20  # number of intervals to generate
t_interval_duration = 5.0  # duration of each interval in seconds

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

#%% Plot the timestamps of the selected labels


beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df_toplot,dict_behaviour_label_to_index)
#beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

n_beh_keys = len(beh_plot_dict)

do_the_plot = False

if do_the_plot:

    thefig = go.Figure()
    thefig.update_layout(
        title=f"behaviour labels, mouse:{the_animal}, session:{the_session}",
        xaxis_title="time (s)",
        yaxis_title="beh. idx"
    )
    bar_height = 0.8

    # define color dictionary
    color_map = px.colors.qualitative.Plotly
    beh_color_dict = {}
    for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
        beh_color_dict[beh_idx] = color_map[idx % len(color_map)]

    # Add custom legend
    for idx, (beh, beh_idx) in enumerate(beh_plot_dict.items()):
        thefig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=beh_color_dict[beh_idx]),
            legendgroup=beh,
            showlegend=True,
            name=str(beh)
        ))

    for k, (x, y) in enumerate(beh_plot_xy):
        # Prepare color mapping for legend and rectangles
        offset = bar_height / 2.0  # half the height of the bar
        for i in range(len(x) - 1):
            if not (np.isnan(x[i]) or np.isnan(x[i+1]) or np.isnan(y[i]) or np.isnan(y[i+1])):
                beh_idx = y[i]
                fillcolor = beh_color_dict.get(beh_idx, "#000000")
                thefig.add_shape(
                    type="rect",
                    x0=x[i], y0=y[i] - offset,
                    x1=x[i + 1],
                    y1=y[i] + offset,
                    fillcolor=fillcolor,
                    line=dict(color=fillcolor),
                    opacity=1.0,
                    layer='above',
                )
            i += 1
    thefig.update_xaxes(range=[0, t_stop_all])
    thefig.update_yaxes(range=[0, n_beh_keys + 1])
        
    thefig.show()


#%%
# these are fixed
dt = 20*1E-3 # in ms
n_lags = 49
C = 10.0
penalty = 'l2'
n_folds_total = 5

#%%
# generate features and labels for the complete dataset

X_alltime=pre.do_binning_operation(spiketrains,
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

units_fit = X_alltime.coords['neuron'].values


#%% Define train-test split


def get_train_test_data(k:int):
    # fix: use logical or and guard range correctly
    if (k < 0) or (k > n_folds_total - 1):
        raise ValueError(f"k must be between 0 and {n_folds_total-1}, got {k}.")
    skf = StratifiedKFold(n_splits=n_folds_total, shuffle=False)

    # extract the fold from the set without the escape
    train_index,test_index = list(\
        skf.split(X_behaviour_maxlagged, y_behaviour))[k]
    X_train = X_behaviour_maxlagged[train_index]
    y_train = y_behaviour[train_index]
    X_test = X_behaviour_maxlagged[test_index]
    y_test = y_behaviour[test_index]
    return X_train, y_train, X_test, y_test

    
#%%
# let's test the function above

X_train1,y_train1, X_test1, y_test1 = get_train_test_data(0)


#%% Data is ready! Now fit one model per escape-fold and keep per-fold results




def build_pipeline():
    return Pipeline([
        ('regularizer',StandardScaler()),
        ('lda', LogisticRegression(
                        penalty=penalty, 
                        solver='liblinear',
                        class_weight='balanced',
                        verbose=True,
                        n_jobs=-1,
                        C=C,
                        max_iter=5000,
                        tol=1e-4,
                        random_state=0,
        )),
    ])

# FIX: use the_animal / the_session (previously undefined animal/session)
outdir = os.path.join(os.path.dirname(__file__), "local_outputs", f"{the_animal}_{the_session}")
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

# keep only what is needed in memory
fold_macro_f1 = []
# NEW: accumulators for per-behaviour coefficient averaging
coeff_sums = {}       # behaviour label -> summed coefficient vector
coeff_counts = {}     # behaviour label -> number of folds contributing

print("Starting per-fold logistic regressions (saving each fold to disk and freeing memory)...")
t_loop_start = time.time()
for k in range(n_folds_total):
    X_train_k, y_train_k, X_test_k, y_test_k = get_train_test_data(k)
    # save train/test arrays for this fold (temporary artifact)
    fold_arrays_path = os.path.join(outdir, f"fold_{k}_arrays.joblib")
    dump(
        {
            'X_train': X_train_k,
            'y_train': y_train_k,
            'X_test': X_test_k,
            'y_test': y_test_k,
        },
        fold_arrays_path,
        compress=3
    )
    pipe_k = build_pipeline()
    t0 = time.time()
    pipe_k.fit(X_train_k, y_train_k)
    fit_secs = time.time() - t0

    # --- NEW: accumulate coefficients per behaviour (exclude 'none') ---
    lr_k = pipe_k.named_steps['lda']
    for class_idx, coef_row in zip(lr_k.classes_, lr_k.coef_):
        beh_label = dict_classindex_to_behaviour.get(class_idx)
        if (beh_label is None) or (beh_label == 'none'):
            continue
        coeff_sums[beh_label] = coeff_sums.get(beh_label, 0.0) + coef_row
        coeff_counts[beh_label] = coeff_counts.get(beh_label, 0) + 1
    # -------------------------------------------------------------------

    y_pred_k = pipe_k.predict(X_test_k)
    report_k = classification_report(y_test_k, y_pred_k, output_dict=True, zero_division=0)
    cm_k = confusion_matrix(y_test_k, y_pred_k, labels=pipe_k.named_steps['lda'].classes_)
    macro_f1_k = report_k.get('macro avg', {}).get('f1-score', np.nan)

    # Save pipeline and metrics to disk
    pipe_path = os.path.join(outdir, f"pipe_fold_{k}.joblib")
    dump(pipe_k, pipe_path, compress=3)
    with open(os.path.join(outdir, f"report_fold_{k}.json"), "w") as f:
        json.dump(report_k, f)
    np.save(os.path.join(outdir, f"confusion_matrix_fold_{k}.npy"), cm_k)

    fold_macro_f1.append(macro_f1_k)
    print(f"Fold {k}: fit in {fit_secs:.2f}s, macro-F1={macro_f1_k:.4f} -> saved to {outdir}")

    # Cleanup RAM/GPU before next fold
    del X_train_k, y_train_k, X_test_k, y_test_k, y_pred_k, pipe_k, lr_k
    gc.collect()
    _free_gpu()

print(f"All folds completed in {time.time()-t_loop_start:.2f}s.")

# Save overall summary and exit early to avoid heavy downstream steps
folds_summary_df = pd.DataFrame({'k': list(range(n_folds_total)), 'macro_f1': fold_macro_f1})
folds_summary_df.to_csv(os.path.join(outdir, "folds_summary.csv"), index=False)
with open(os.path.join(outdir, "folds_summary.json"), "w") as f:
    json.dump({'k': list(range(n_folds_total)), 'macro_f1': fold_macro_f1}, f)
best_k = int(np.nanargmax(fold_macro_f1))
print(f"Best fold is k={best_k} with macro-F1={fold_macro_f1[best_k]:.4f}")
print(f"Per-fold artifacts saved under: {outdir}. Reloading artifacts into RAM...")

# Save core run parameters for reference
run_params = {
    'animal': the_animal,
    'session': the_session,
    'dt': float(dt),
    'n_lags': int(n_lags),
    'C': float(C),
    'penalty': str(penalty),
    'n_total_folds': int(n_folds_total),
}
with open(os.path.join(outdir, "run_params.json"), "w") as f:
    json.dump(run_params, f, indent=2)

# NEW: build ranking dataframe (df_ranking) similarly to 01_get_ranking_df.py
print("Building df_ranking from averaged fold coefficients...")
n_units = len(units_fit)
lag_block = n_lags + 1  # number of lag slices
ranking_rows = []
for beh_label, sum_vec in coeff_sums.items():
    count = coeff_counts[beh_label]
    avg_vec = sum_vec / count
    coef_2d = avg_vec.reshape(lag_block, n_units)  # shape (lag, unit)
    # magnitude-based ranking (sum of squared loadings across lags)
    rank_val_mag = (coef_2d ** 2).sum(axis=0)  # per unit
    lag_sum = coef_2d.sum(axis=0)
    sign = np.sign(lag_sum)
    sign[sign == 0] = 1.0
    # ranking (descending by magnitude)
    order = np.argsort(-rank_val_mag)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    for ui, unit in enumerate(units_fit):
        ranking_rows.append({
            'behaviour': beh_label,
            'session': the_session,
            'unit': unit,
            'unit_location': the_unit_locations[ui],
            'rank_sign': float(sign[ui]),
            'rank': int(ranks[ui]),
            'rank_val': float(rank_val_mag[ui] * sign[ui]),
        })

df_ranking = pd.DataFrame(ranking_rows).sort_values(by=['session','behaviour','unit'])
df_ranking_path_pkl = os.path.join(outdir, "df_ranking.pkl")
df_ranking_path_csv = os.path.join(outdir, "df_ranking.csv")
df_ranking.to_pickle(df_ranking_path_pkl)
df_ranking.to_csv(df_ranking_path_csv, index=False)
print(f"df_ranking saved to:\n  {df_ranking_path_pkl}\n  {df_ranking_path_csv}")

exit()

