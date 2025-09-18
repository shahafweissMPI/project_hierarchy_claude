# -*- coding: utf-8 -*-
"""
Created on 2025-08-25

@author: Dylan Festa

After reading the unit ranking dataframe, makes a new series of logistic regression fits 
with the goal of comparing performance when we only consider 'good' units instead of all
of them. 

"""
#%%
from __future__ import annotations
from typing import List

%load_ext cuml.accel
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px

# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains

# import stuff from sklear: pipeline, lagged data, Z-score, PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

# to apply arbitrary functions to the data in the pipeline
from sklearn.preprocessing import FunctionTransformer
# NEW: optional CuPy + gc for explicit GPU/RAM cleanup
import gc
try:
    import cupy as cp
except Exception:
    cp = None

def _free_gpu():
    """Best-effort release of CuPy memory pools (if CuPy is available)."""
    try:
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass



#%%


def run_the_fit(animal_fit:str, session_fit:str,
                *,
                units_to_keep:List=[]) :
    # Fixed parameters used in the fit
    n_lags = 49 
    dt = 20*1E-3 
    C = 0.04  #0.02
    logistic_regression_kwargs = dict(
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        verbose=False,
        n_jobs=1,
        C=C,
        max_iter=5000,
        tol=1e-4,
    )
    # Cross-validation splitter (internal automatic splitting)
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    all_data_dict = rdl.load_preprocessed_dict(animal_fit, session_fit)
    # unpack the data
    behaviour_data_df = all_data_dict['behaviour']
    spike_times = all_data_dict['spike_times']
    time_index = all_data_dict['time_index']
    cluster_index = all_data_dict['cluster_index']
    region_index = all_data_dict['region_index']


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
    # if unit list is not empty, apply filter_by_unit
    if len(units_to_keep) > 0:
        spiketrains = spiketrains.filter_by_units(units_to_keep)

    n_units = spiketrains.n_units
    print(f"Number of units: {n_units}.")
    behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal_fit,session_fit,behaviour_data_df)
    # remove pursuit and run_away behaviour and select enough trials

    beh_to_remove = ['pup_grab','pursuit','hunt_switch','escape','loom']

    behaviour_timestamps_df = behaviour_timestamps_df[ (behaviour_timestamps_df['n_trials'] >= 5)
                                                  & (behaviour_timestamps_df['is_start_stop'])
                                                  & (~behaviour_timestamps_df['behaviour'].isin(beh_to_remove))]

    t_first_behav = behaviour_data_df['frames_s'].min()
    t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
    print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

    # add a "notmuch" behavior, in the form of 
    # k randomly arranged, non superimposing 10 second intervals
    # after the first 5 min but before the 5 min preceding the first labeled behavior

    t_notmuch_start = 5 * 60.0  # 5 minutes after the start of the session
    t_notmuch_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

    k_intervals = 30  # number of intervals to generate
    t_interval_duration = 5.0  # duration of each interval in seconds

    intervals_notmuch_fromzero = rdl.generate_random_intervals_within_time(
            t_notmuch_end,k_intervals,t_interval_duration)

    intervals_notmuch = [(_start+ t_notmuch_start, _end + t_notmuch_start) for _start, _end in intervals_notmuch_fromzero]


    # add a row to behaviour dataframe
    new_row = pd.DataFrame([{
        'mouse': animal_fit,
        'session': session_fit,
        'behavioural_category': 'notmuch',
        'behaviour': 'notmuch',
        'n_trials': k_intervals,
        'is_start_stop': True,
        'total_duration': t_interval_duration * k_intervals,
        'start_stop_times': intervals_notmuch,
        'point_times': []
    }])
    behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)
                                        

    # now generate a dictionary
    dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
    # add 'none'
    dict_behavior_label_to_index['none'] = -1

    # for convenience, have a dictionary that maps index to label, merging labels with same index
    # so the above becomes pup_grab_and_pup_retrieve
    # Build inverse with merging of duplicate indices
    _inv = {}
    for label, idx in dict_behavior_label_to_index.items():
        _inv.setdefault(idx, []).append(label)

    # Deduplicate and merge with "_and_" for multi-label indices
    dict_classindex_to_behaviour = {
        idx: (labels[0] if len(labels) == 1 else "_and_".join(labels))
        for idx, labels in ((i, lst) for i, lst in _inv.items())
    }

    # build behaviour_representation_df
    # keys are label and class index as in dict_classindex_to_behaviour
    _rows_temp = []
    for class_index, label in dict_classindex_to_behaviour.items():
        if label == 'none':
            continue
        _intervals = []
        for (_key,_val) in dict_behavior_label_to_index.items():
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

    X_alltime=pre.do_binning_operation(spiketrains,
                        'count',dt=dt,t_start=t_start_all,t_stop=t_stop_all)
    # get labels
    beh_labels_xr = rdl.generate_behaviour_labels_inclusive(behaviour_timestamps_df,
                                            t_start =0.0,t_stop= t_stop_all,
                                            dt=dt,
                                            behaviour_labels_dict=dict_behavior_label_to_index)
    y_alltime = beh_labels_xr.values
    if len(y_alltime) != X_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after binning.")
    # apply lag
    X_alltime_maxlagged = rdl.generate_lag_dimensions_expansion_xr(X_alltime, n_lags)
    # drop NaN rows from X and y
    X_alltime_maxlagged = X_alltime_maxlagged[n_lags:,:]  # keep only the last n_lags_max rows
    y_alltime = y_alltime[n_lags:]  # keep only the last n_lags_max rows
    if X_alltime_maxlagged.shape[0] != y_alltime.shape[0]:
        raise AssertionError("Mismatch between number of samples in X and y after 'drop_nan' step.")
    t_alltime = X_alltime_maxlagged.coords['time_bin_center'].values
    # now, select only behavioural labels
    y_idx_behavior = y_alltime != -1  # -1 is the 'none' label
    y_behaviour = y_alltime[y_idx_behavior]
    X_behaviour_maxlagged = X_alltime_maxlagged.values[y_idx_behavior, :]
    t_behaviour= t_alltime[y_idx_behavior]

    units_fit = X_alltime.coords['neuron'].values
    if len(units_fit)!=n_units:
        raise AssertionError("Mismatch between number of units in X and n_units after binning.")

    # ---------------------------
    # Cross-validated evaluation
    # ---------------------------
    unique_classes = np.unique(y_behaviour)
    n_classes = unique_classes.size
    confusion_matrix_total = np.zeros((n_classes, n_classes), dtype=int)
    # per-class metric accumulators: {class_idx: {'precision':[], 'recall':[], 'f1-score':[], 'support': total_support}}
    per_class_metrics = {
        int(c): {'precision': [], 'recall': [], 'f1-score': [], 'support': 0}
        for c in unique_classes
    }

    fold_id = 0
    for train_index, test_index in skf.split(X_behaviour_maxlagged, y_behaviour):
        fold_id += 1
        X_train, X_test = X_behaviour_maxlagged[train_index], X_behaviour_maxlagged[test_index]
        y_train, y_test = y_behaviour[train_index], y_behaviour[test_index]

        pipe_fold = Pipeline([
            ('regularizer', StandardScaler()),
            ('lda', LogisticRegression(**logistic_regression_kwargs)),
        ])

        t0 = time.time()
        pipe_fold.fit(X_train, y_train)
        t1 = time.time()
        print(f"[CV fold {fold_id}] fit time: {t1 - t0:.2f}s")

        y_pred = pipe_fold.predict(X_test)

        # Accumulate confusion matrix
        cm_fold = confusion_matrix(y_test, y_pred, labels=unique_classes)
        confusion_matrix_total += cm_fold

        # Classification report dict for this fold
        report_dict_fold = classification_report(y_test, y_pred, labels=unique_classes,
                                                 output_dict=True, zero_division=0)
        for _cls in unique_classes:
            cls_key = str(int(_cls))
            per_class_metrics[int(_cls)]['precision'].append(report_dict_fold[cls_key]['precision'])
            per_class_metrics[int(_cls)]['recall'].append(report_dict_fold[cls_key]['recall'])
            per_class_metrics[int(_cls)]['f1-score'].append(report_dict_fold[cls_key]['f1-score'])
            per_class_metrics[int(_cls)]['support'] += report_dict_fold[cls_key]['support']
        # enhanced cleanup (RAM + VRAM)
        del pipe_fold, X_train, X_test, y_train, y_test, y_pred, cm_fold, report_dict_fold
        gc.collect()
        _free_gpu()
    # Aggregate per-class metrics (mean over folds)
    rows = []
    for _cls in unique_classes:
        metrics = per_class_metrics[int(_cls)]
        rows.append({
            'class_index': int(_cls),
            'behavior': dict_classindex_to_behaviour.get(int(_cls), str(int(_cls))),
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'f1-score': np.mean(metrics['f1-score']),
            'support': metrics['support'],
        })
    classification_report_df = pd.DataFrame(rows).sort_values('class_index')


    the_confusion_matrix = confusion_matrix_total

    return  classification_report_df,\
        the_confusion_matrix,\
        behaviour_representation_df,\
        dict_classindex_to_behaviour

#%%

time_start = time.time()

the_animal = 'afm16924'
the_session = '240527'
#the_session = '240529'

classification_report_df_allneus,\
    the_confusion_matrix_allneus,\
    behaviour_representation_df_allneus,\
    dict_classindex_to_behaviour_allneus = run_the_fit(the_animal, the_session)

# Now classification_report_df is available as a DataFrame with scores and behavior names

time_end = time.time()
time_end_str = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
print(f"Fit with all neurons completed in: {time_end_str}")

#%%
# read dataframe with all rankings

the_mouse='afm16924'

read_file = os.path.join(os.path.dirname(__file__), "local_outputs", f"df_ranking_{the_mouse}.pkl")

if not os.path.exists(read_file):
    raise FileNotFoundError(f"File {read_file} does not exist. Please run `fit_and_save.py` first.")

with open(read_file, "rb") as f:
    df_ranking = pickle.load(f)

the_sessions = df_ranking['session'].unique()
print(f"Found {len(the_sessions)} sessions: {the_sessions}")

#%%
# For a given session, select only ranking below k_ranking_max
# use query
sesson_test = the_sessions[0]
k_ranking_max = 5

df_session_test = df_ranking.query("session == @sesson_test")
all_units = df_session_test['unit'].unique()

df_good_units = df_session_test.query("session == @sesson_test and rank < @k_ranking_max")
good_units = df_good_units['unit'].unique()
n_good_units = len(good_units)

# now, control units are randomly selected from units that are not good
notgood_units = np.setdiff1d(all_units, good_units)
control_units = np.random.choice(notgood_units, size=n_good_units, replace=False)

#%%
# now let's run classifications!
classification_report_df_best,\
    confusion_matrix_best,\
    behaviour_representation_df_best,\
    dict_classindex_to_behaviour_best = run_the_fit(the_animal, the_session, units_to_keep=good_units)

classification_report_df_nobest,\
    confusion_matrix_nobest,\
    behaviour_representation_df_nobest,\
    dict_classindex_to_behaviour_nobest = run_the_fit(the_animal, the_session, units_to_keep=control_units)


#%%

# Grouped bar plot: F1-score per behavior for
# 1) all units
# 2) high-ranking (good) units
# 3) random control units (same count as good)
beh_base = classification_report_df_allneus[['class_index','behavior']].copy()
# Ensure unique behaviors (one row per class_index)
beh_base = beh_base.sort_values('class_index')

f1_all = classification_report_df_allneus[['behavior','f1-score']].rename(columns={'f1-score':'f1_all_units'})
f1_good = classification_report_df_best[['behavior','f1-score']].rename(columns={'f1-score':'f1_high_ranking'})
f1_ctrl = classification_report_df_nobest[['behavior','f1-score']].rename(columns={'f1-score':'f1_random_control'})

merged = beh_base.merge(f1_all, on='behavior', how='left') \
                 .merge(f1_good, on='behavior', how='left') \
                 .merge(f1_ctrl, on='behavior', how='left')

tidy = merged.melt(
    id_vars=['class_index','behavior'],
    value_vars=['f1_all_units','f1_high_ranking','f1_random_control'],
    var_name='model',
    value_name='f1-score'
)

model_name_map = {
    'f1_all_units': 'All units',
    'f1_high_ranking': 'High-ranking subset',
    'f1_random_control': 'Random control subset'
}
tidy['model'] = tidy['model'].map(model_name_map)

# Sort behaviors by class_index to keep consistent ordering
beh_order = merged.sort_values('class_index')['behavior'].tolist()
tidy['behavior'] = pd.Categorical(tidy['behavior'], categories=beh_order, ordered=True)

color_map = {
    'All units': "#505050",            # blue
    'High-ranking subset': '#2ca02c',  # green
    'Random control subset': "#b17300" # purple
}

fig = px.bar(
    tidy,
    x='behavior',
    y='f1-score',
    color='model',
    barmode='group',
    category_orders={'behavior': beh_order, 'model': list(model_name_map.values())},
    color_discrete_map=color_map,
    title='Per-behavior F1-score across neuron selection strategies'
)
fig.update_layout(
    xaxis_title='Behavior',
    yaxis_title='F1-score',
    legend_title='Model',
    bargap=0.15,
    bargroupgap=0.05
)
fig.update_yaxes(range=[0,1])

# Ensure output directory exists and save
_out_dir = os.path.join(os.path.dirname(__file__), "local_outputs")
os.makedirs(_out_dir, exist_ok=True)
_out_file = os.path.join(_out_dir, "f1_compare_units.html")
fig.write_html(_out_file)
print(f"Saved F1 comparison figure to: {_out_file}")

fig.show()

# # %%

# %%

# second approach, select high-value units, the top k in terms 
# of summed abs rank value
sesson_test = the_sessions[0]
k_top_max = 20


df_session_test = df_ranking.query("session == @sesson_test").copy()

# now sum abs of rank_val on all rows for each unit
df_session_test.loc[:,'rank_val_abs'] = df_session_test['rank_val'].abs()
df_unit_rank_sums = (
    df_session_test
        .groupby('unit')['rank_val_abs']
        .sum()
        .reset_index(name='rank_val_total')
)

# sort by rank, highest to lowest
df_unit_rank_sums = df_unit_rank_sums.sort_values('rank_val_total', ascending=False)

top_units_by_total = df_unit_rank_sums.nlargest(k_top_max, 'rank_val_total')['unit'].values

# now pick randomly nontopk units, similar to before
nontopk_units_all = np.setdiff1d(df_session_test['unit'].unique(), top_units_by_total)
nontopk_units = np.random.choice(nontopk_units_all, size=k_top_max, replace=False)

# now let's run classifications with this new selection of units
classification_report_df_topk,\
    confusion_matrix_topk,\
    behaviour_representation_df_topk,\
    dict_classindex_to_behaviour_topk = run_the_fit(the_animal, the_session, units_to_keep=top_units_by_total)

classification_report_df_nontopk,\
    confusion_matrix_nontopk,\
    behaviour_representation_df_nontopk,\
    dict_classindex_to_behaviour_nontopk = run_the_fit(the_animal, the_session, units_to_keep=nontopk_units)

#%%
# Bar plot: F1-score per behavior for (a) all units, (b) top-k by rank_val_total, (c) random non-topk
beh_base2 = classification_report_df_allneus[['class_index','behavior']].drop_duplicates().sort_values('class_index')
f1_all2   = classification_report_df_allneus[['behavior','f1-score']].rename(columns={'f1-score':'f1_all_units'})
f1_topk   = classification_report_df_topk[['behavior','f1-score']].rename(columns={'f1-score':'f1_topk_total'})
f1_nontop = classification_report_df_nontopk[['behavior','f1-score']].rename(columns={'f1-score':'f1_random_nontopk'})

merged2 = (beh_base2
           .merge(f1_all2, on='behavior', how='left')
           .merge(f1_topk, on='behavior', how='left')
           .merge(f1_nontop, on='behavior', how='left'))

tidy2 = merged2.melt(
    id_vars=['class_index','behavior'],
    value_vars=['f1_all_units','f1_topk_total','f1_random_nontopk'],
    var_name='model',
    value_name='f1-score'
)

model_name_map2 = {
    'f1_all_units': 'all units',
    'f1_topk_total': f'top {k_top_max}',
    'f1_random_nontopk': f'random non-top {k_top_max}'
}
tidy2['model'] = tidy2['model'].map(model_name_map2)
beh_order2 = merged2.sort_values('class_index')['behavior'].tolist()
tidy2['behavior'] = pd.Categorical(tidy2['behavior'], categories=beh_order2, ordered=True)

color_map2 = {
    'all units': '#505050',
    f'top {k_top_max}': '#1f77b4',
    f'random non-top {k_top_max}': "#b8c958"
}

fig2 = px.bar(
    tidy2,
    x='behavior',
    y='f1-score',
    color='model',
    barmode='group',
    category_orders={'behavior': beh_order2,
                     'model': list(model_name_map2.values())},
    color_discrete_map=color_map2,
    title='per-behavior F1-score: top-k vs random non-top-k'
)
fig2.update_layout(
    xaxis_title='behavior',
    yaxis_title='F1-score',
    legend_title='model',
    bargap=0.15,
    bargroupgap=0.05
)
fig2.update_yaxes(range=[0,1])

_out_dir = os.path.join(os.path.dirname(__file__), "local_outputs")
os.makedirs(_out_dir, exist_ok=True)
_out_file2 = os.path.join(_out_dir, "f1_compare_units_topk.html")
fig2.write_html(_out_file2)
print(f"Saved Top-k comparison figure to: {_out_file2}")
fig2.show()



# %%
