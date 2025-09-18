# -*- coding: utf-8 -*-
"""
Created on 2025-08-19

@author: Dylan Festa

Applies logistic regression with l1 regularizer to promote sparsity.
Focus on session 240529, animal afm16924, with both escape and parenting behavior.
K-fold method, leave-one-out, where one full escape trial is removed in each fold.
Single fit, includes a label for "loiter", corresponding to time intervals before
the pups are introduced, while the mouse is simply moving around and not doing much.

Quick, saves everything on temporary files (necessary otherwise the GPU can't handle memory load)

Read files with `kfold_fit_examine.py`
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

# to check explicitely the weights used for different classes
from sklearn.utils.class_weight import compute_class_weight



#%%

all_mice = rdl.get_good_animals()
print(f"Found {len(all_mice)} animals.")
print("Animals:", all_mice)

animal = 'afm16924'
sessions_for_animal = rdl.get_good_sessions(animal)
print(f"Found {len(sessions_for_animal)} sessions for animal {animal}.")
print("Sessions:", sessions_for_animal)

#%%

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

n_units = spiketrains.n_units
print(f"Number of PAG units: {n_units}.")
#%% Now, process behaviour data to get labels

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter labels for training, myst be only start_stop, at least 5 trials
# and also remove the hunt-related behaviours (for now). Also switches can be ignored as labels
beh_to_remove = ['attack', 'pursuit', 
                 'chase', 'approach','hunt_switch','run_away',
                 'escape_switch','pup_grab']

behaviour_timestamps_df = behaviour_timestamps_df[ (behaviour_timestamps_df['n_trials'] >= 5)
                                                  & (behaviour_timestamps_df['is_start_stop'])
                                                  & (~behaviour_timestamps_df['behaviour'].isin(beh_to_remove))]

#%% add a "loiter" behavior, in the form of 
# k randomly arranged, non superimposing 10 second intervals
# after the first 5 min but before the 5 min preceding the first labeled behavior

t_first_behav = behaviour['frames_s'].min()
t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

#%%

t_loiter_start = 5 * 60.0  # 5 minutes after the start of the session
t_loiter_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

k_intervals = 20  # number of intervals to generate
t_interval_duration = 20.0  # duration of each interval in seconds

intervals_loiter_fromzero = rdl.generate_random_intervals_within_time(
        t_loiter_end,k_intervals,t_interval_duration)

intervals_loiter = [(_start+ t_loiter_start, _end + t_loiter_start) for _start, _end in intervals_loiter_fromzero]


# add a row to behaviour dataframe
new_row = pd.DataFrame([{
    'mouse': animal,
    'session': session,
    'behavioural_category': 'loiter',
    'behaviour': 'loiter',
    'n_trials': k_intervals,
    'is_start_stop': True,
    'total_duration': t_interval_duration * k_intervals,
    'start_stop_times': intervals_loiter,
    'point_times': []
}])
behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)
                                    

#%%
# now generate a dictionary
dict_behaviour_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# add 'none'
dict_behaviour_label_to_index['none'] = -1
# pup grab... removed!
# pup grab and pup retrieve should have same label
# dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']
# impose explicit label for escape, so we can use it explictely in the script
label_escape = 101
dict_behaviour_label_to_index['escape'] = label_escape

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
# Non-standard split.  Our focus is the squeeze the moust out of the 6 
# escapes. Therefore we will use a k-fold approach, with k=6, where
# we inject a fold for the escape behavior manually.

n_escapes = behaviour_timestamps_df[behaviour_timestamps_df['behaviour'] == 'escape']['n_trials'].values[0]

y_idx_noescape = y_behaviour != label_escape 
y_behaviour_noescape = y_behaviour[y_idx_noescape]
X_behaviour_maxlagged_noescape = X_behaviour_maxlagged[y_idx_noescape, :]

def get_train_test_data_by_escape_trials(k:int):
    # fix: use logical or and guard range correctly
    if (k < 0) or (k > n_escapes - 1):
        raise ValueError(f"k must be between 0 and {n_escapes-1}, got {k}.")
    skf = StratifiedKFold(n_splits=n_escapes, shuffle=False)

    # extract the fold from the set without the escape
    train_index_noescape,test_index_noescape = list(\
        skf.split(X_behaviour_maxlagged_noescape, y_behaviour_noescape))[k]
    X_train_folded = X_behaviour_maxlagged_noescape[train_index_noescape]
    y_train_folded = y_behaviour_noescape[train_index_noescape]
    X_test_folded = X_behaviour_maxlagged_noescape[test_index_noescape]
    y_test_folded = y_behaviour_noescape[test_index_noescape]
    # Now, find manually the k-fold for the escape only
    start_stop_all = pre.get_start_stop_idx(y_behaviour,label_escape)
    if len(start_stop_all)!=n_escapes :
        raise ValueError(f"Expected {n_escapes} escape intervals, got {len(start_stop_all)}.")
    # use the k-th interval for testing, keep a copy of the element
    start_stop_test_keep = start_stop_all[k]
    # add these to the test set
    X_test_folded = np.concatenate([X_test_folded, X_behaviour_maxlagged[start_stop_test_keep[0]:start_stop_test_keep[1]]])
    y_test_folded = np.concatenate([y_test_folded, y_behaviour[start_stop_test_keep[0]:start_stop_test_keep[1]]])
    # I must *exclude* my k-th interval from the train list
    # that is, drop the kth element
    start_stop_keep = np.delete(start_stop_all, k, axis=0)
    # now, using start stop inices, extend X_train and y_train
    for _start, _stop in start_stop_keep:
        X_train_folded = np.concatenate([X_train_folded, X_behaviour_maxlagged[_start:_stop]])
        y_train_folded = np.concatenate([y_train_folded, y_behaviour[_start:_stop]])
    return X_train_folded, y_train_folded, X_test_folded, y_test_folded

    
#%%
# let's test the function above

X_train1,y_train1, X_test1, y_test1 = get_train_test_data_by_escape_trials(1)

#%%
test_theclasses = np.unique(y_train1)
test_classweights = compute_class_weight('balanced', 
                                    classes=test_theclasses,y=np.concatenate([y_train1, y_test1]))
class_weight_dict = dict(zip(test_theclasses, test_classweights))

# double the weight of 'escape'
# nah, this seems totally useless
#class_weight_dict[label_escape] *= 2

#%% 
# second test, count number of train and test elements for some labels
# ('escape','loiter','pup_run_and_pup_retrieve') and for  for each k, 1 to 6, make a dataframe with that
df_test_kfold_rows = []

for _k in range(n_escapes):
    label_loiter = dict_behaviour_label_to_index['loiter']
    label_pup_run = dict_behaviour_label_to_index['pup_run']
    _,_y_train_k,_,_y_test_k = get_train_test_data_by_escape_trials(_k)
    # count all train labels
    counted_labels = pd.Series(_y_train_k).value_counts().to_dict()
    # count all test labels
    counted_labels_test = pd.Series(_y_test_k).value_counts().to_dict()
    # add to the dataframe
    df_test_kfold_rows.append({
        'k': _k,
        'n_train_escape': counted_labels.get(label_escape, 0),
        'n_test_escape': counted_labels_test.get(label_escape, 0),
        'n_train_loiter': counted_labels.get(label_loiter, 0),
        'n_test_loiter': counted_labels_test.get(label_loiter, 0),
        'n_train_pup_run': counted_labels.get(label_pup_run, 0),
        'n_test_pup_run': counted_labels_test.get(label_pup_run, 0),
    })

df_test_kfold = pd.DataFrame(df_test_kfold_rows)


# check that n_train_escape and n_test_escape ALWAYS have the same sum for each row

_test_columns_sum = df_test_kfold[['n_train_escape', 'n_test_escape']].sum(axis=1)
# Assert that the sums are equal
assert np.all(_test_columns_sum == _test_columns_sum[0]), "Total bin number not consistent across k-folds!"



#%% Data is ready! Now fit one model per escape-fold and keep per-fold results

from sklearn.metrics import classification_report, confusion_matrix

def build_pipeline():
    return Pipeline([
        ('regularizer',StandardScaler()),
        ('lda', LogisticRegression(
                        penalty='l1', 
                        #solver='saga',
                        solver='liblinear',
                        #class_weight='balanced',
                        class_weight=class_weight_dict,
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
for k in range(n_escapes):
    X_train_k, y_train_k, X_test_k, y_test_k = get_train_test_data_by_escape_trials(k)
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

    # Save compact coefficients summary (top-10 units per behaviour) for this fold
    lda = pipe_k.named_steps['lda']
    rows_best_units = []
    lag_seconds = np.arange(n_lags+1) * dt
    for beh_idx, beh_name in dict_classindex_to_behaviour.items():
        if beh_idx not in lda.classes_:
            continue
        coefficients_beh = lda.coef_[lda.classes_ == beh_idx]
        # reshape to (lags, units)
        coefficients_beh_reshaped = coefficients_beh.reshape(n_lags+1, n_units)
        # rank units by sum of squares across lags
        score_for_kbest = (coefficients_beh_reshaped ** 2).sum(axis=0)[:]
        top10 = np.argsort(score_for_kbest)[-10:][::-1]
        top_units = np.array(units_fit)[top10]
        for idx_u, unit in zip(top10, top_units):
            unit_coefficients = coefficients_beh_reshaped[:, idx_u]
            rows_best_units.append({
                'fold': k,
                'unit': unit,
                'behaviour_name': beh_name,
                'behaviour_idx': int(beh_idx),
                'coefficients': unit_coefficients.tolist(),
                'coefficients_lag': lag_seconds.tolist(),
                'summed_coefficients': float(unit_coefficients.sum()),
            })
    pd.DataFrame(rows_best_units).to_csv(
        os.path.join(outdir, f"best10_units_per_behaviour_fold_{k}.csv"), index=False
    )

    fold_macro_f1.append(macro_f1_k)
    print(f"Fold {k}: fit in {fit_secs:.2f}s, macro-F1={macro_f1_k:.4f} -> saved to {outdir}")

    # Cleanup RAM/GPU before next fold
    del X_train_k, y_train_k, X_test_k, y_test_k, y_pred_k, pipe_k, lda
    gc.collect()
    _free_gpu()

print(f"All folds completed in {time.time()-t_loop_start:.2f}s.")

# Save overall summary and exit early to avoid heavy downstream steps
folds_summary_df = pd.DataFrame({'k': list(range(n_escapes)), 'macro_f1': fold_macro_f1})
folds_summary_df.to_csv(os.path.join(outdir, "folds_summary.csv"), index=False)
with open(os.path.join(outdir, "folds_summary.json"), "w") as f:
    json.dump({'k': list(range(n_escapes)), 'macro_f1': fold_macro_f1}, f)
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
    'label_escape': int(label_escape),
}
with open(os.path.join(outdir, "run_params.json"), "w") as f:
    json.dump(run_params, f, indent=2)

exit()