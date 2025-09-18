# -*- coding: utf-8 -*-
"""
Created on 2025-08-21

@author: Dylan Festa

Applies logistic regression with l1 regularizer to promote sparsity.

Fits on parenting and hunting behaviours, focusing on sessions
20240527 and 20250529
K-fold fit, with K=5, saves all results in a folder called `local_output`

"""
#%%
%load_ext cuml.accel
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
from sklearn.metrics import classification_report, confusion_matrix

#%%

all_mice = rdl.get_good_animals()
print(f"Found {len(all_mice)} animals.")
print("Animals:", all_mice)

animal = 'afm16924'
sessions_for_animal = rdl.get_good_sessions(animal)
print(f"Found {len(sessions_for_animal)} sessions for animal {animal}.")
print("Sessions:", sessions_for_animal)

#%%

#session = '240527'
session = '240529'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(animal, session)
# unpack the data
behaviour = all_data_dict['behaviour']
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


spiketrains=pre.SpikeTrains.from_spike_list(spike_times,
                                units=cluster_index,
                                unit_location=region_index,
                                isi_minimum=1/200.0, 
                                t_start=t_start_all,
                                t_stop=t_stop_all)
# filter spiketrains to only include PAG units
spiketrains = spiketrains.filter_by_unit_location('PAG')

the_units = spiketrains.units
the_unit_locations = spiketrains.unit_location
n_units = spiketrains.n_units
print(f"Number of PAG units: {n_units}.")
#%% Now, process behaviour data to get labels

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter labels for training, myst be only start_stop, at least 5 trials
# and also remove the hunt-related behaviours (for now). Also switches can be ignored as labels
# beh_to_remove = ['attack', 'pursuit', 
#                  'chase', 'approach','hunt_switch','run_away',
#                  'escape_switch','pup_grab']

# let's remove pup grab, just in case there are grabs without retrieves
# also, either I remove pursuit, or I remove attack and run_away. I cannot keep all three.
# I am also removing escape for now, because it happens a few times and I dont' want to mix 
# parenting with hunting with escape just yet.
beh_to_remove = ['pup_grab','pursuit','hunt_switch','escape']

behaviour_timestamps_df = behaviour_timestamps_df[ (behaviour_timestamps_df['n_trials'] >= 5)
                                                  & (behaviour_timestamps_df['is_start_stop'])
                                                  & (~behaviour_timestamps_df['behaviour'].isin(beh_to_remove))]

#%% add a "notmuch" behavior, in the form of 
# k randomly arranged, non superimposing 10 second intervals
# after the first 5 min but before the 5 min preceding the first labeled behavior

t_first_behav = behaviour['frames_s'].min()
t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

#%%

t_notmuch_start = 5 * 60.0  # 5 minutes after the start of the session
t_notmuch_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

k_intervals = 30  # number of intervals to generate
t_interval_duration = 5.0  # duration of each interval in seconds

intervals_notmuch_fromzero = rdl.generate_random_intervals_within_time(
        t_notmuch_end,k_intervals,t_interval_duration)

intervals_notmuch = [(_start+ t_notmuch_start, _end + t_notmuch_start) for _start, _end in intervals_notmuch_fromzero]


# add a row to behaviour dataframe
new_row = pd.DataFrame([{
    'mouse': animal,
    'session': session,
    'behavioural_category': 'notmuch',
    'behaviour': 'notmuch',
    'n_trials': k_intervals,
    'is_start_stop': True,
    'total_duration': t_interval_duration * k_intervals,
    'start_stop_times': intervals_notmuch,
    'point_times': []
}])
behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)
                                    

#%%
# now generate a dictionary
dict_behaviour_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# add 'none'
dict_behaviour_label_to_index['none'] = -1

# impose explicit label for notmuch
label_notmuch = 101
dict_behaviour_label_to_index['notmuch'] = label_notmuch

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
            _to_add = behaviour_timestamps_df[behaviour_timestamps_df['behaviour'] == _key]['start_stop_times'].values[0]
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


beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df,dict_behaviour_label_to_index)
#beh_plot_inverse_dict = {v: k for k, v in beh_plot_dict.items()}

n_beh_keys = len(beh_plot_dict)

thefig = go.Figure()
thefig.update_layout(
    title="time labels for each behavior",
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
    
#thefig.show(renderer="browser")
#thefig.show(renderer="vscode")
thefig.show()


#%%
# these are fixed
dt = 20*1E-3 # in ms
n_lags = 29 # crashes when above 29 :-(
C = 0.02
total_folds = 5

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

def get_train_test_data(k:int,tot_k_folds:int=5):
    # fix: use logical or and guard range correctly
    skf = StratifiedKFold(n_splits=tot_k_folds, shuffle=False)

    # extract the fold from the set without the escape
    train_index,test_index = list(\
        skf.split(X_behaviour_maxlagged, y_behaviour))[k]
    X_train_folded = X_behaviour_maxlagged[train_index]
    y_train_folded = y_behaviour[train_index]
    X_test_folded = X_behaviour_maxlagged[test_index]
    y_test_folded = y_behaviour[test_index]
    return X_train_folded, y_train_folded, X_test_folded, y_test_folded

    
#%%
# let's test the function above

X_train1,y_train1, X_test1, y_test1 = get_train_test_data(1,tot_k_folds=total_folds)

#%% 
# second test, count number of train and test elements for some labels
# (''notmuch','pup_run','eat') and for  for each k, 1 to total_folds, make a dataframe with that
# df_test_kfold_rows = []

# for _k in range(total_folds):
#     label_notmuch = dict_behaviour_label_to_index['notmuch']
#     label_pup_run = dict_behaviour_label_to_index['pup_run']
#     label_eat = dict_behaviour_label_to_index['eat']
#     _,_y_train_k,_,_y_test_k = get_train_test_data(_k)
#     # count all train labels
#     counted_labels = pd.Series(_y_train_k).value_counts().to_dict()
#     # count all test labels
#     counted_labels_test = pd.Series(_y_test_k).value_counts().to_dict()
#     # add to the dataframe
#     df_test_kfold_rows.append({
#         'k': _k,
#         'n_train_notmuch': counted_labels.get(label_notmuch, 0),
#         'n_test_notmuch': counted_labels_test.get(label_notmuch, 0),
#         'n_train_pup_run': counted_labels.get(label_pup_run, 0),
#         'n_test_pup_run': counted_labels_test.get(label_pup_run, 0),
#         'n_train_eat': counted_labels.get(label_eat, 0),
#         'n_test_eat': counted_labels_test.get(label_eat, 0),
#     })

# df_test_kfold = pd.DataFrame(df_test_kfold_rows)


# # check that n_train_escape and n_test_escape ALWAYS have the same sum for each row

# _test_columns_sum = df_test_kfold[['n_train_eat', 'n_test_eat']].sum(axis=1)
# # Assert that the sums are equal
# assert np.all(_test_columns_sum == _test_columns_sum[0]), "Total bin number not consistent across k-folds!"



#%% Data is ready! Now fit one model per escape-fold and keep per-fold results


def build_pipeline():
    return Pipeline([
        ('regularizer',StandardScaler()),
        ('lda', LogisticRegression(
                        penalty='l1', 
                        #solver='saga',
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

outdir = os.path.join(os.path.dirname(__file__), "local_outputs", f"{animal}_{session}")
os.makedirs(outdir, exist_ok=True)

# NEW: save behaviour/label metadata to disk (binary + human-readable)
labels_meta_path = os.path.join(outdir, "labels_meta.joblib")
dump(
    {
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

print("Starting per-fold logistic regressions (saving each fold to disk and freeing memory)...")
t_loop_start = time.time()
for k in range(total_folds):
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
    del X_train_k, y_train_k, X_test_k, y_test_k, y_pred_k, pipe_k
    gc.collect()
    _free_gpu()

print(f"All folds completed in {time.time()-t_loop_start:.2f}s.")

# Save overall summary and exit early to avoid heavy downstream steps
folds_summary_df = pd.DataFrame({'k': list(range(total_folds)), 'macro_f1': fold_macro_f1})
folds_summary_df.to_csv(os.path.join(outdir, "folds_summary.csv"), index=False)
with open(os.path.join(outdir, "folds_summary.json"), "w") as f:
    json.dump({'k': list(range(total_folds)), 'macro_f1': fold_macro_f1}, f)
best_k = int(np.nanargmax(fold_macro_f1))
print(f"Best fold is k={best_k} with macro-F1={fold_macro_f1[best_k]:.4f}")
print(f"Per-fold artifacts saved under: {outdir}. Reloading artifacts into RAM...")

# Save core run parameters for reference
run_params = {
    'animal': animal,
    'session': session,
    'dt': float(dt),
    'n_lags': int(n_lags),
    'C': float(C),
    'n_total_folds': int(total_folds),
}
with open(os.path.join(outdir, "run_params.json"), "w") as f:
    json.dump(run_params, f, indent=2)

exit()