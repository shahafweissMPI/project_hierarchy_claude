# -*- coding: utf-8 -*-
"""
Created on 2025-08-25

@author: Dylan Festa

Applies logistic regression with regularizers that promote sparsity
(elasticnet) Single fit, measure speed when using cuML and RAPIDS on the GPU

Here I also include a label for "no behaviour", corresponding to time intervals before
the pups are introduced, while the mouse is simply moving around (exploring?)

"""
#%%
%load_ext cuml.accel
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

# import stuff from sklear: pipeline, lagged data, Z-score, PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix


#%%


the_animal = 'afm16924'
the_session = '240527'

print("Loading data...")
t_data_load_start = time.time()
# load data using read_data_light library
all_data_dict = rdl.load_preprocessed_dict(the_animal, the_session)
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

units=spiketrains.units
unit_locations=spiketrains.unit_location

n_units = len(units)
print(f"Number of PAG units: {n_units}.")

#%%

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(the_animal,the_session,behaviour)
# filter labels for training, myst be only start_stop, at least 5 trials
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
    'mouse': the_animal,
    'session': the_session,
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
    title=f"time labels for each behavior of session {the_session}",
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
C = 0.04 # 0.04
n_split_kfold = 4

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



#%% Train-test split


skf = StratifiedKFold(n_splits=n_split_kfold, shuffle=False)
# Get the last split
train_index, test_index = list(skf.split(X_behaviour_maxlagged, y_behaviour))[-1]
# sort indices to keep arrays in time order
train_index = np.sort(train_index)
test_index = np.sort(test_index)

X_train, X_test = X_behaviour_maxlagged[train_index], X_behaviour_maxlagged[test_index]
y_train, y_test = y_behaviour[train_index], y_behaviour[test_index]

#%% Find the latest time point in the train set
latest_train_time_point = t_behaviour[train_index].max()


#%% Data is ready! Now build the pipeline and apply the grid search
pipe= Pipeline([
    ('regularizer',StandardScaler()),
    ('lda', LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    class_weight='balanced',
                    verbose=False,
                    n_jobs=1,
                    C=C,
                    max_iter=5000,
                    tol=1e-4,
                    random_state=0,
                    )),])

time_start_onefit = time.time()
print("Starting logistic regression...")
pipe.fit(X_train, y_train)
time_end_onefit = time.time()
time_onefit_string = time.strftime("%H:%M:%S", time.gmtime(time_end_onefit - time_start_onefit))
print(f"Logistic regression completed in: {time_onefit_string}")

#%% Check performance on test set

y_pred = pipe.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


#%% Full decoding ranking dataframe



trained_behaviours = behaviour_timestamps_df['behaviour'].values
lda = pipe.named_steps['lda']
lag_seconds = np.arange(n_lags+1) * dt

def get_loading(beh):
    # Implement the logic to retrieve the loading average for the given behaviour
    _coefs = lda.coef_[lda.classes_ == dict_behaviour_label_to_index[beh]]
    _coefs_reshaped = _coefs.reshape(n_lags+1,n_units)
    return xr.DataArray(
        _coefs_reshaped,
        coords=[lag_seconds, units_fit],
        dims=["lag", "unit"],
        attrs={'unit_locations': unit_locations}
    )

def get_loading_sign(beh):
    _loading = get_loading(beh)
    #sum over lags
    _loading_sums = _loading.sum(dim="lag").values
    _signs = np.sign(_loading_sums)
    # turn zero into +1
    _signs[_signs == 0] = 1
    return _signs

_main_df_rows = []
for beh in trained_behaviours:
    loading_avg = get_loading(beh)
    loading_sign = get_loading_sign(beh)
    loading_avg_sq = loading_avg ** 2
    rank_val = loading_avg_sq.sum(dim="lag").values
    # sort descending by rank_val (largest magnitude first)
    rank_sorting = np.argsort(-rank_val)
    rank = np.empty_like(rank_sorting)
    rank[rank_sorting] = np.arange(len(rank_sorting))
    unit_location = loading_avg.attrs['unit_locations']
    for (_idx_unit, _unit) in enumerate(units):
        _main_df_rows.append({
            "behaviour": beh,
            "session": the_session,
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
# get probability estimate for each label for the full dataset (memory-aware batching)

def batched_predict_proba(estimator, X_da, *,
                          base_batch_size=100_000,
                          dtype=np.float32,
                          target_chunk_mb=200,
                          memmap_threshold_mb=2048,
                          memmap_path=None):
    """
    Predict probabilities in batches to avoid RAM spikes.

    Parameters
    ----------
    estimator : fitted sklearn estimator with predict_proba
    X_da : xarray DataArray or numpy-like (n_samples, n_features)
    base_batch_size : int, upper cap for batch size
    dtype : np.dtype, storage dtype for output
    target_chunk_mb : int, aim for each batch to be about this size in MB
    memmap_threshold_mb : int, if total output exceeds this, use memmap
    memmap_path : str or None, optional custom memmap filename

    Returns
    -------
    probs : ndarray (possibly memmap) shape (n_samples, n_classes)
    """
    n_samples = X_da.shape[0]
    # Peek one batch to know number of classes
    # (We assume at least one sample exists)
    n_features = X_da.shape[1]
    n_classes = len(estimator.classes_)

    itemsize = np.dtype(dtype).itemsize
    total_bytes = n_samples * n_classes * itemsize
    total_mb = total_bytes / (1024**2)

    # Derive adaptive batch size
    bytes_per_row = n_classes * itemsize
    adaptive_batch = max(1, int((target_chunk_mb * 1024**2) // bytes_per_row))
    batch_size = min(base_batch_size, adaptive_batch)

    use_memmap = total_mb > memmap_threshold_mb
    if use_memmap:
        if memmap_path is None:
            memmap_path = f"predictions_{estimator.__class__.__name__.lower()}_{n_samples}.mmap"
        print(f"[batched_predict_proba] Using memmap file: {memmap_path} (~{total_mb:.1f} MB total)")
        probs = np.memmap(memmap_path, mode='w+', dtype=dtype, shape=(n_samples, n_classes))
    else:
        print(f"[batched_predict_proba] Allocating in RAM: ~{total_mb:.1f} MB")
        probs = np.empty((n_samples, n_classes), dtype=dtype)

    print(f"[batched_predict_proba] batch_size={batch_size} (target_chunk_mb={target_chunk_mb}, base_cap={base_batch_size})")

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        X_batch = X_da[start:stop, :].values  # xarray -> np
        batch_probs = estimator.predict_proba(X_batch)
        if batch_probs.dtype != dtype:
            batch_probs = batch_probs.astype(dtype, copy=False)
        probs[start:stop] = batch_probs

    if use_memmap:
        probs.flush()
    return probs

# Allow user override via environment variables
_env_target_mb = os.environ.get("PREDICT_PROBA_TARGET_CHUNK_MB")
_env_memmap_thresh_mb = os.environ.get("PREDICT_PROBA_MEMMAP_THRESHOLD_MB")

target_chunk_mb = int(_env_target_mb) if _env_target_mb else 200
memmap_threshold_mb = int(_env_memmap_thresh_mb) if _env_memmap_thresh_mb else 2048  # 2 GB

time_pred_start = time.time()

y_predprob_full = batched_predict_proba(
    pipe,
    X_alltime_maxlagged,
    base_batch_size=1_000,
    dtype=np.float32,
    target_chunk_mb=target_chunk_mb,
    memmap_threshold_mb=memmap_threshold_mb,
)

time_pred_end = time.time()
time_pred_str = time.strftime("%H:%M:%S", time.gmtime(time_pred_end - time_pred_start)) 

print(f"Prediction of probabilities completed in: {time_pred_str}\nNow saving!")

#%%


# Build xarray (works with memmap too)
predictions_xr = xr.DataArray(
    y_predprob_full,
    coords=[t_alltime, pipe.classes_],
    dims=["time", "label"],
    attrs={
        "description": "Predicted probabilities for each behavior label",
        "dtype": str(y_predprob_full.dtype),
        "units": "probability",
        "class_labels_dict": dict_classindex_to_behaviour
    }
)


#%%
dict_save = {
    "latest_train_time": latest_train_time_point,
    "df_ranking": df_ranking,
    "predictions": predictions_xr,
    "behaviour_timestamps": behaviour_timestamps_df,
    "index_to_behaviour": dict_classindex_to_behaviour,
    "behaviour_representation_df": behaviour_representation_df,
}

# savename with mouse and session
saveaname = f"{the_animal}_{the_session}_one_fit.pkl"

print(f"Saving results to: {saveaname}...")

#%%

with open(saveaname, "wb") as f:
    pickle.dump(dict_save, f)

#%%
# exit the script
exit()