# -*- coding: utf-8 -*-
"""
Created on 2025-08-11

@author: Dylan Festa

Applies logistic regression with regularizers that promote sparsity
(elasticnet) Single fit. I include a label for "no behaviour", corresponding to time intervals before
the pups are introduced, while the mouse is simply moving around (exploring?)


The goal here is to compare performance between all neurons, high-scoring neurons only,
and random neurons that are not high-scoring.

High scoring neurons are defined as the k neurons with the highest sum of squared loadings.

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



#%%


animal = 'afm16924'
session = '240523_0'


def run_the_fit(animal_fit:str, session_fit:str, kbest:int = 50,
                units_to_keep:List=None) :

    # Fixed parameters used in the fit
    n_lags = 29
    dt = 10*1E-3  # 10 ms
    l1_ratio = 1.0
    C = 0.02
    # Decide penalty type: if l1_ratio == 1 use pure L1 to avoid elastic-net overhead
    use_elasticnet = (l1_ratio is not None) and (l1_ratio != 1.0)
    logistic_regression_kwargs = dict(
        penalty='elasticnet' if use_elasticnet else 'l1',
        solver='saga',
        class_weight='balanced',
        verbose=True,
        n_jobs=-1,
        C=C,
        max_iter=5000,
        tol=1e-4,
        random_state=0,
    )
    if use_elasticnet:
        logistic_regression_kwargs['l1_ratio'] = l1_ratio

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
    # if unit selection is provided, apply it
    if units_to_keep is not None and len(units_to_keep) > 0:
        spiketrains = spiketrains.filter_by_units(units_to_keep)

    n_units = spiketrains.n_units

    print(f"Number of PAG units: {n_units}.")
    behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal_fit,session_fit,behaviour_data_df)

    # behaviour selection and refinement here:
    # remove pursuit and run_away behaviour and select enough trials
    behaviour_timestamps_df = behaviour_timestamps_df[ (behaviour_timestamps_df['n_trials'] >= 15)
                                                  & (behaviour_timestamps_df['is_start_stop'])
                                                  & (behaviour_timestamps_df['behaviour'] != 'pursuit')
                                                  & (behaviour_timestamps_df['behaviour'] != 'run_away')
                                                  ]

    t_first_behav = behaviour_data_df['frames_s'].min()
    t_first_behav_str = time.strftime("%M:%S", time.gmtime(t_first_behav))
    print(f"First behavior starts at: {t_first_behav_str} (MM:SS)")

    # add a "loiter" behavior, in the form of 
    # k randomly arranged, non superimposing 10 second intervals
    # after the first 5 min but before the 5 min preceding the first labeled behavior

    t_loiter_start = 5 * 60.0  # 5 minutes after the start of the session
    t_loiter_end = t_first_behav - 5 * 60.0  # 5 minutes before the first behavior

    k_intervals = 18  # number of intervals to generate
    t_interval_duration = 10.0  # duration of each interval in seconds

    intervals_loiter_fromzero = rdl.generate_random_intervals_within_time(
            t_loiter_end,k_intervals,t_interval_duration)

    intervals_loiter = [(_start+ t_loiter_start, _end + t_loiter_start) for _start, _end in intervals_loiter_fromzero]


    # add a row to behaviour dataframe
    new_row = pd.DataFrame([{
        'mouse': animal_fit,
        'session': session_fit,
        'behavioural_category': 'loiter',
        'behaviour': 'loiter',
        'n_trials': k_intervals,
        'is_start_stop': True,
        'total_duration': t_interval_duration * k_intervals,
        'start_stop_times': intervals_loiter,
        'point_times': []
    }])
    behaviour_timestamps_df = pd.concat([behaviour_timestamps_df, new_row], ignore_index=True)

    # now generate a dictionary
    dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
    # remove 'loom'
    dict_behavior_label_to_index.pop('loom', None)
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

    # Train-test split using StratifiedKFold (no shuffle). Use the LAST fold as test.
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    # Get the last split
    train_index, test_index = list(skf.split(X_behaviour_maxlagged, y_behaviour))[-1]
    # sort indices to keep arrays in time order
    train_index = np.sort(train_index)
    test_index = np.sort(test_index)

    X_train, X_test = X_behaviour_maxlagged[train_index], X_behaviour_maxlagged[test_index]
    y_train, y_test = y_behaviour[train_index], y_behaviour[test_index]
    X_train_idx, X_test_idx = train_index, test_index

    # Build a dataframe with, for each label, the end time of training samples for that label
    times_train = t_behaviour[X_train_idx]
    rows_last_per_label = []
    for _lab in np.unique(y_train):
        _lab = int(_lab)
        _name = dict_classindex_to_behaviour.get(_lab, str(_lab))
        mask = (y_train == _lab)
        _lab_last_time = float(np.max(times_train[mask]))
        rows_last_per_label.append({
            'behaviour_idx': _lab,
            'behaviour_name': _name,
            'last_train_time_s': _lab_last_time,
        })
    last_train_time_per_label_df = (
        pd.DataFrame(rows_last_per_label)
        .sort_values('behaviour_idx')
        .reset_index(drop=True)
    )
    last_train_time_per_label_df['last_train_time_str'] = last_train_time_per_label_df['last_train_time_s'].apply(
        lambda s: time.strftime("%H:%M:%S", time.gmtime(s))
    )

    # Define the pipeline for logistic regression
    pipe= Pipeline([
        ('regularizer',StandardScaler()),
        ('lda', LogisticRegression(**logistic_regression_kwargs)),
    ])

    time_start_onefit = time.time()
    print("Starting logistic regression...")
    pipe.fit(X_train, y_train)
    time_end_onefit = time.time()
    time_onefit_string = time.strftime("%H:%M:%S", time.gmtime(time_end_onefit - time_start_onefit))
    print(f"Logistic regression completed in: {time_onefit_string}")

    # Check performance on test set
    y_pred = pipe.predict(X_test)
    the_confusion_matrix = confusion_matrix(y_test, y_pred)
    the_classification_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(the_classification_report)
    print("Confusion Matrix:")
    print(the_confusion_matrix)

    # Convert classification report to pandas DataFrame with behavior names
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    # Only keep rows corresponding to class indices (not 'accuracy', 'macro avg', etc.)
    report_rows = []
    for class_idx, row in report_dict.items():
        try:
            idx = int(class_idx)
            beh_name = dict_classindex_to_behaviour.get(idx, str(idx))
            row_with_label = dict(row)
            row_with_label['class_index'] = idx
            row_with_label['behavior'] = beh_name
            report_rows.append(row_with_label)
        except ValueError:
            continue  # skip non-integer keys
    classification_report_df = pd.DataFrame(report_rows)
    # Reorder columns for clarity
    cols = ['class_index', 'behavior'] + [c for c in classification_report_df.columns if c not in ['class_index', 'behavior']]
    classification_report_df = classification_report_df[cols]

    # Dataframe of all scores for each label and neuron
    lda = pipe.named_steps['lda']
    all_scores_df_row = []
    for beh_idx,beh_name in dict_classindex_to_behaviour.items():
        # avoid behaviours not used for training
        if beh_idx not in lda.classes_:
            continue
        # get all coefficients for the behavior, and reshape them
        coefficients_beh = lda.coef_[lda.classes_ == beh_idx]
        coefficients_beh_reshaped = coefficients_beh.reshape(n_lags+1,n_units)
        coefficients_beh_sum = coefficients_beh_reshaped.sum(axis=0)[:]
        # score to classify a neuron as best is sum of squares of its coefficients
        score_for_kbest =  (coefficients_beh_reshaped ** 2).sum(axis=0)[:]
        # now row for dataframe, one per neuron, for all neurons
        for (k,unit) in enumerate(units_fit):
            all_scores_df_row.append({
                'class': beh_name,
                'class_idx': int(beh_idx),
                'unit': unit,
                'coefficients_lag': coefficients_beh_reshaped[:,k],
                'summed_coefficients': coefficients_beh_sum[k],
                'score_kbest': score_for_kbest[k]
            })

    all_scores_df = pd.DataFrame(all_scores_df_row, 
                columns=['class', 'class_idx', 'unit', 'coefficients_lag',\
                    'summed_coefficients', 'score_kbest'])
    # add is_k_best column, true for the k_best neurons with the highest score_kbest value
    # summed across all classes
    all_scores_df['is_k_best'] = False
    # Compute per-unit total score across all classes
    unit_total_scores = all_scores_df.groupby('unit')['score_kbest'].sum().sort_values(ascending=False)
    # Keep a handy column with the per-unit total on each row
    all_scores_df['score_kbest_allclasses'] = all_scores_df['unit'].map(unit_total_scores)
    # Select top-k units (guard for k > number of units)
    if kbest > unit_total_scores.shape[0]:
        raise ValueError("kbest is greater than the number of available units.")
    top_units = set(unit_total_scores.head(kbest).index)
    all_scores_df.loc[all_scores_df['unit'].isin(top_units), 'is_k_best'] = True

    # ALL DONE!

    return pipe,\
        all_scores_df,\
        last_train_time_per_label_df,\
        the_classification_report,\
        the_confusion_matrix,\
        behaviour_representation_df,\
        dict_classindex_to_behaviour,\
        classification_report_df

#%%

# parameter: how many neurons in total are considered as good?
the_kbest_number = 10

time_start = time.time()

pipe, all_scores_df,\
    last_train_time_per_label_df,\
    the_classification_report,\
    the_confusion_matrix,\
    behaviour_representation_df,\
    dict_classindex_to_behaviour,\
    classification_report_df = run_the_fit(animal, session,kbest=the_kbest_number)

# Now classification_report_df is available as a DataFrame with scores and behavior names

time_end = time.time()
time_end_str = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
print(f"One fit completed in: {time_end_str}")

best_scores_df = all_scores_df[all_scores_df['is_k_best']]
# sort by class and unit
all_scores_df = all_scores_df.sort_values(by=['class', 'unit'])
best_scores_df = best_scores_df.sort_values(by=['class', 'unit'])

#%% Now, heatmap of ALL scores for each class
# sort by highest score
heatmap_data = all_scores_df.pivot(index="unit", columns="class", values="summed_coefficients")
heatmap_data = heatmap_data.fillna(0)
# sort order based on sum of absolute value across all classes
_value_sum = heatmap_data.abs().sum(axis=1)
_sort_order = _value_sum.argsort()[::-1]
heatmap_data = heatmap_data.iloc[_sort_order.values]

# x values as strings
heatmap_data.index = [f"unit : {i}" for i in heatmap_data.index]
#%%

# Plot with plotly

fig = px.imshow(
    heatmap_data,
    aspect="auto",
    color_continuous_scale="Picnic_r",
    color_continuous_midpoint=0.0  # <-- ensure zero is mapped to white
)
fig.update_layout(
    title="scores for each class",
    xaxis_title="unit",
    yaxis_title="class",
    coloraxis_colorbar=dict(title="Summed Coefficients"),
    height=800,
)
fig.show()

#%% Now, run the fit again in two cases: keeping only best units
# and keeping same number of non-best units


all_units = all_scores_df['unit'].unique()
best_units = best_scores_df['unit'].unique()

if len(best_units) != the_kbest_number:
    raise ValueError("Number of best units does not match kbest parameter.")

non_best_units  = [u for u in all_units if u not in best_units]

n_best_units = the_kbest_number

all_minus_best_units = [u for u in all_units if u not in best_units]
# pick them randomly
non_best_to_keep = np.random.choice(non_best_units, size=n_best_units, replace=False)

print('***************************\n')
print("\nKEEPING BEST UNITS\n")

pipe_nobest, all_scores_df_nobest,\
    last_train_time_per_label_df_nobest,\
    classification_report_nobest,\
    confusion_matrix_nobest,\
    behaviour_representation_df_nobest,\
    dict_classindex_to_behaviour_nobest,\
    classification_report_df_nobest = run_the_fit(animal, session,
                                                  kbest=the_kbest_number,
                                                    units_to_keep=best_units)

print("\nKEEPING NON-BEST UNITS\n")

pipe_noothers, all_scores_df_noothers,\
    last_train_time_per_label_df_noothers,\
    the_classification_report_noothers,\
    the_confusion_matrix_noothers,\
    behaviour_representation_df_noothers,\
    dict_classindex_to_behaviour_noothers,\
    classification_report_df_noothers = run_the_fit(animal, session,
                                                    kbest=the_kbest_number,
                                                    units_to_keep=non_best_to_keep)

#%%

# Grouped bar plot of F1-scores per behavior for "remove best" vs "remove non-best"
# Uses behaviors from the baseline classification_report_df as x categories
beh_order = classification_report_df['behavior'].drop_duplicates().tolist()

base_behaviors_df = pd.DataFrame({'behavior': beh_order})
f1_allunits = classification_report_df[['behavior', 'f1-score']].rename(columns={'f1-score': 'f1_allunits'})
f1_nobest = classification_report_df_nobest[['behavior', 'f1-score']].rename(columns={'f1-score': 'f1_nobest'})
f1_noothers = classification_report_df_noothers[['behavior', 'f1-score']].rename(columns={'f1-score': 'f1_noothers'})

merged = base_behaviors_df.merge(f1_nobest, on='behavior', how='left')\
    .merge(f1_noothers, on='behavior', how='left')\
    .merge(f1_allunits, on='behavior', how='left')

tidy = merged.melt(
    id_vars='behavior',
    value_vars=['f1_allunits', 'f1_nobest', 'f1_noothers'],
    var_name='condition',
    value_name='f1-score'
)
# Rename legend entries
tidy['condition'] = tidy['condition'].map({
    'f1_allunits': 'all units',
    'f1_nobest': 'high-scoring',
    'f1_noothers': 'random non high-scoring'
})

fig = px.bar(
    tidy,
    x='behavior',
    y='f1-score',
    color='condition',
    barmode='group',
    title=f'f1-score per behavior: all units vs best {n_best_units} units vs non-best {n_best_units} units',
    category_orders={'condition': ['all units', 'high-scoring', 'random non high-scoring']},
    color_discrete_map={
        'all units': "#525252",                 # dark gray
        'high-scoring': '#2ca02c',              # green
        'random non high-scoring': '#9467bd'    # purple
    }
)
fig.update_layout(
    xaxis_title='behavior',
    yaxis_title='F1-score',
    legend_title='Units kept'
)
fig.show()

# %%
