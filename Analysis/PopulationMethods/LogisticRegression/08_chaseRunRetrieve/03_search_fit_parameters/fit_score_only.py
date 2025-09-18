# -*- coding: utf-8 -*-
"""
Created on 2025-08-27

@author: Dylan Festa

The purpose of this script is exploring hyperparameters, to see
if there are specific choices that lead to much better prediction 
performance in cross-validation.
"""
#%%
#%load_ext cuml.accel
import cuml

import os
from pathlib import Path

# sacred!
from sacred import Experiment
from sacred.observers import FileStorageObserver

path_this_file = Path(__file__).resolve()
path_storage = path_this_file.parent / "local_outputs"
ex=Experiment("Testing logistic regression performance across hyperparameters")
ex.observers.append(FileStorageObserver(str(path_storage)))

import pickle
import numpy as np, pandas as pd, xarray as xr
import time
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


@ex.config
def default_configuration():
    seed = 0
    the_mouse = 'afm16924'
    the_session = '240529'
    dt = 20*1E-3 # in ms
    n_lags = 49
    C_regression = 0.02
    penalty = 'l1'  # 'l1' or 'l2'
    n_splits = 5
    notes = "Testing logistic regression performance across hyperparameters."
    save_id = f"{the_mouse}_{the_session}_{C_regression}_{dt}_{penalty}_{n_splits}"
    series_id = f"{the_mouse}_{the_session}_{C_regression}_{dt}_{penalty}_{n_splits}"



@ex.automain
def main_function(the_mouse, the_session,
            dt,n_lags,
            C_regression, penalty,n_splits,
            notes,save_id,series_id,_seed,_run):

    # if session is an integer, convert it to string
    if isinstance(the_session, int):
        the_session = str(the_session)

    _run.info["notes"] = notes
    print(f"Starting fit, series name: {series_id}")
    print(f"mouse={the_mouse} session={the_session}")
    print(f"C={C_regression}, dt={dt}, penalty={penalty}")


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
    y_alltime = y_alltime[n_lags:]  # keep only the last n_lags_max rows
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
        if (k < 0) or (k > n_splits - 1):
            raise ValueError(f"k must be between 0 and {n_splits-1}, got {k}.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

        # extract the fold from the set without the escape
        train_index,test_index = list(\
            skf.split(X_behaviour_maxlagged, y_behaviour))[k]
        X_train = X_behaviour_maxlagged[train_index]
        y_train = y_behaviour[train_index]
        X_test = X_behaviour_maxlagged[test_index]
        y_test = y_behaviour[test_index]
        return X_train, y_train, X_test, y_test

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
                            C=C_regression,
                            max_iter=5000,
                            tol=1e-4,
                            random_state=0,
            )),
        ])

    # keep only what is needed in memory
    fold_macro_f1 = []
    # NEW: accumulators for per-behaviour coefficient averaging
    coeff_sums = {}
    coeff_counts = {}
    # NEW: collect per-fold classification metrics
    per_fold_metrics_rows = []

    print("Starting per-fold logistic regressions...")
    t_loop_start = time.time()
    for k in range(n_splits):
        X_train_k, y_train_k, X_test_k, y_test_k = get_train_test_data(k)
        pipe_k = build_pipeline()

        pipe_k.fit(X_train_k, y_train_k)

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
        # NEW: store macro F1 for fold summary
        if 'macro avg' in report_k:
            fold_macro_f1.append(report_k['macro avg'].get('f1-score', np.nan))
        # NEW: expand per-class metrics rows
        for class_key, metrics in report_k.items():
            if class_key in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            try:
                class_idx = int(class_key)
            except ValueError:
                continue
            beh_label = dict_classindex_to_behaviour.get(class_idx, str(class_idx))
            per_fold_metrics_rows.append({
                'mouse': the_mouse,
                'session': the_session,
                'fold_number': k,
                'behavior_label': beh_label,
                'f1-score': metrics.get('f1-score', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
            })

        # Cleanup RAM/GPU before next fold
        del X_train_k, y_train_k, X_test_k, y_test_k, y_pred_k, pipe_k, lr_k
        gc.collect()
        _free_gpu()

    print(f"All folds completed in {time.time()-t_loop_start:.2f}s.")

    # Save overall summary and exit early to avoid heavy downstream steps
    folds_summary_df = pd.DataFrame({'k': list(range(n_splits)), 'macro_f1': fold_macro_f1})
    # build per-fold metrics dataframe
    df_per_fold_metrics = pd.DataFrame(per_fold_metrics_rows,
                                       columns=['mouse','session','fold_number','behavior_label',
                                                'f1-score','precision','recall'])

    # NEW: averages across folds per behaviour (no fold count column)
    df_per_behaviour_avg_metrics = (
        df_per_fold_metrics
        .groupby(['mouse','session','behavior_label'], as_index=False)[['f1-score','precision','recall']]
        .mean()
    )

    # Now, dataframe with averages across all folds
    
    # build ranking dataframe (df_ranking) similarly to 01_get_ranking_df.py
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

    save_dictionary = {
        'ranking_df': df_ranking,
        'folds_summary_df': folds_summary_df,
        'per_fold_metrics_df': df_per_fold_metrics,
        'per_behaviour_avg_metrics_df': df_per_behaviour_avg_metrics,
    }

    path_savetemp = path_storage / f"fit_score_only_temp_{save_id}.pkl"
    while path_savetemp.exists():
        save_id += 1
        path_savetemp = path_storage / f"fit_score_only_temp_{save_id}.pkl"

    with open(path_savetemp, "wb") as f:
        pickle.dump(save_dictionary, f)
    _run.add_artifact(path_savetemp)
    print(f"Created temporary file {path_savetemp} to store data. It will be deleted now.")
    os.remove(path_savetemp)
    print("ALL DONE! Exiting...")
    return None

