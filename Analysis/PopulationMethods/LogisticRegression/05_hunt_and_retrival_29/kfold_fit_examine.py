# -*- coding: utf-8 -*-
"""
Created on 2025-08-20

@author: Dylan Festa

This is to examine the output of `kfold_fit_and_save.py`
See comments there for details on the fit.
"""
#%%
import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json
from joblib import load


# impot local modules in PopulationMethods/lib 
import read_data_light as rdl
import preprocess as pre
from preprocess import SpikeTrains,IFRTrains


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
units_fit = spiketrains.units
print(f"Number of PAG units: {n_units}.")
#%% Now, process behaviour data to get labels

behaviour_timestamps_df = rdl.convert_to_behaviour_timestamps(animal,session,behaviour)
# filter labels for training, myst be only start_stop, at least 5 trials
# and also remove the hunt-related behaviours (for now). Also switches can be ignored as labels
beh_to_remove = ['attack', 'pursuit', 'chase', 'approach','hunt_switch','run_away','escape_switch']

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
dict_behavior_label_to_index = {label: idx for idx, label in enumerate(behaviour_timestamps_df['behaviour'].values)}
# add 'none'
dict_behavior_label_to_index['none'] = -1
# pup grab and pup retrieve should have same label
dict_behavior_label_to_index['pup_grab'] = dict_behavior_label_to_index['pup_retrieve']
# impose explicit label for escape, so we can use it explictely in the script
label_escape = 101
dict_behavior_label_to_index['escape'] = label_escape

#%% for convenience, have a dictionary that maps index to label, merging labels with same index
# so the above becomes pup_grab_and_pup_retrieve
# dict_classindex_to_behavior={}
# Build inverse with merging of duplicate indices
_inv = {}
for label, idx in dict_behavior_label_to_index.items():
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

#%% Plot the timestamps of the selected labels


beh_plot_xy, beh_plot_dict = rdl.generate_behaviour_startstop_segments(behaviour_timestamps_df,dict_behavior_label_to_index)
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


#%% Read run parameters saved by training
outdir = os.path.join(os.path.dirname(__file__), "local_outputs", f"{animal}_{session}")
params_path = os.path.join(outdir, "run_params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as f:
        _run_params = json.load(f)
    dt = float(_run_params.get("dt", 10e-3))
    n_lags = int(_run_params.get("n_lags", 29))
    print(f"Loaded run params from {params_path}: dt={dt}, n_lags={n_lags}")
else:
    print(f"Warning: {params_path} not found. Falling back to defaults.")
    dt = 10e-3
    n_lags = 29


#%%

outdir = os.path.join(os.path.dirname(__file__), "local_outputs", f"{animal}_{session}")
folds_summary_df = pd.read_csv(os.path.join(outdir, "folds_summary.csv"))
ks = folds_summary_df['k'].tolist()

all_folds_data = []
print(f"Reloading artifacts for folds: {ks}")
for k in ks:
    print(f" - Loading fold {k}")
    pipe_path = os.path.join(outdir, f"pipe_fold_{k}.joblib")
    report_path = os.path.join(outdir, f"report_fold_{k}.json")
    cm_path = os.path.join(outdir, f"confusion_matrix_fold_{k}.npy")
    best10_path = os.path.join(outdir, f"best10_units_per_behaviour_fold_{k}.csv")

    pipe_k = load(pipe_path)
    with open(report_path, "r") as f:
        report_k = json.load(f)
    cm_k = np.load(cm_path)
    best10_df_k = pd.read_csv(best10_path)

    all_folds_data.append({
        'k': k,
        'pipeline': pipe_k,
        'report': report_k,
        'confusion_matrix': cm_k,
        'best10_units_per_behaviour': best10_df_k,
    })
    print(f" - Loaded fold {k}")

# Determine best fold from reloaded summary
best_k_reloaded = int(folds_summary_df.iloc[folds_summary_df['macro_f1'].values.argmax()]['k'])
print(f"Reload complete. Folds loaded: {len(all_folds_data)}. Best fold (reloaded) = {best_k_reloaded}")

n_escapes = len(all_folds_data)
# At this point, all_folds_data holds all per-fold artifacts in RAM for further analysis.

#%%
# some utility functions to get the coefficients
def get_loadings(fold, behaviour):
    lda = all_folds_data[fold]['pipeline'].named_steps['lda']
    coefficients = lda.coef_[lda.classes_ == dict_behavior_label_to_index[behaviour]]
    return coefficients.reshape(n_lags+1, n_units)


def get_loading_average(behaviour):
    loading_sum = np.zeros((n_lags+1, n_units))
    for _k in range(n_escapes):
        loading_sum += get_loadings(_k, behaviour)
    return loading_sum / n_escapes

# just sum over lags, +1 if positive, -1 if negative
def get_loading_sign(behaviour):
    loading_mat = get_loading_average(behaviour)
    loading_sums = loading_mat.sum(axis=0)[:]
    _signs = np.sign(loading_sums)
    # replace 0 with 1.0
    _signs[_signs == 0] = 1
    return _signs

#%%
# now average reshaped loading matrix over k-folds, and plot it
lag_seconds = np.arange(n_lags+1) * dt

behaviour_check = 'escape'
coefficients_check  = get_loading_average(behaviour_check)
column_sorting_on_mean = np.argsort(-np.abs(coefficients_check).sum(axis=0))
coefficients_check_resorted = coefficients_check[:, column_sorting_on_mean]
units_fit_resorted = units_fit[column_sorting_on_mean]

n_neus_show = 25
fig_heatmap_check = px.imshow(
    coefficients_check_resorted[:,:n_neus_show],
    labels=dict(x="units", y="bin lag (s)", color="regression coefficient"),
    y=lag_seconds,  # Use numerical labels for y-axis
    x=[f"unit {u}" for u in units_fit_resorted][:n_neus_show],
    title=f"averaged regression coefficients for behaviour: {behaviour_check}",
    color_continuous_scale='PuOr_r',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

fig_heatmap_check.show(renderer="vscode")

# %%
# Now, take a look at the coefficients of the logistic regression model on each fold
kfold_check=2
behaviour_check = 'escape'
lda = all_folds_data[kfold_check]['pipeline'].named_steps['lda']
coefficients_all = lda.coef_

coefficients_behaviour_check = coefficients_all[lda.classes_ == dict_behavior_label_to_index['escape']]

coefficients_check_reshaped = coefficients_behaviour_check.reshape(n_lags+1,n_units)
lag_seconds = np.arange(n_lags+1) * dt

# apply sorting
coefficients_check_resorted = coefficients_check_reshaped[:, column_sorting_on_mean]
units_fit_resorted = units_fit[column_sorting_on_mean]

# Print as heatmap
n_neus_show = 30
fig_heatmap_check = px.imshow(
    coefficients_check_resorted[:,:n_neus_show],
    labels=dict(x="units", y="bin lag (s)", color="LDA coefficient"),
    y=lag_seconds,  # Use numerical labels for y-axis
    x=[f"unit {u}" for u in units_fit_resorted][:n_neus_show],
    title=f"LDA Coefficients for fold {kfold_check} and behaviour: {behaviour_check}",
    color_continuous_scale='PuOr_r',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

fig_heatmap_check.show(renderer="vscode")


#%% display all of them in a for cycle!
behaviour_check = 'escape'
sorting_check = column_sorting_on_mean
n_neus_show = 30
for k_fold_check in range(n_escapes):
    coefficients_check_reshaped = get_loadings(k_fold_check, behaviour_check)
    lag_seconds = np.arange(n_lags+1) * dt

    # apply sorting
    coefficients_check_resorted = coefficients_check_reshaped[:, sorting_check]
    units_fit_resorted = units_fit[sorting_check]

    # Print as heatmap
    fig_heatmap_check = px.imshow(
        coefficients_check_resorted[:,:n_neus_show],
        labels=dict(x="units", y="bin lag (s)", color="regression coefficient"),
        y=lag_seconds,  # Use numerical labels for y-axis
        x=[f"unit {u}" for u in units_fit_resorted][:n_neus_show],
        title=f"regression coefficients for fold {k_fold_check} and behaviour: {behaviour_check}",
        color_continuous_scale='PuOr_r',
        color_continuous_midpoint=0.0,
        height=900  # Add this line to make the figure taller
    )

    fig_heatmap_check.show(renderer="vscode")


#%% same for pup_run
# start with average over k-folds
behaviour_check = 'pup_run'
n_neus_show = 30
coefficients_check = get_loading_average(behaviour_check)

column_sorting_on_mean = np.argsort(-np.abs(coefficients_check).sum(axis=0))
coefficients_check_resorted = coefficients_check[:, column_sorting_on_mean]
units_fit_resorted = units_fit[column_sorting_on_mean]

fig_heatmap_check = px.imshow(
    coefficients_check_resorted[:,:n_neus_show],
    labels=dict(x="units", y="bin lag (s)", color="regression coefficient"),
    y=lag_seconds,  # Use numerical labels for y-axis
    x=[f"unit {u}" for u in units_fit_resorted][:n_neus_show],
    title=f"averaged regression coefficients for behaviour: {behaviour_check}",
    color_continuous_scale='PuOr_r',
    color_continuous_midpoint=0.0,
    height=900  # Add this line to make the figure taller
)

fig_heatmap_check.show(renderer="vscode")

#%% Now all 5 folds
sorting_check = column_sorting_on_mean
for k_fold_check in range(n_escapes):
    coefficients_check_reshaped = get_loadings(k_fold_check, behaviour_check)
    lag_seconds = np.arange(n_lags+1) * dt

    # apply sorting
    coefficients_check_resorted = coefficients_check_reshaped[:, sorting_check]
    units_fit_resorted = units_fit[sorting_check]

    # Print as heatmap
    fig_heatmap_check = px.imshow(
        coefficients_check_resorted[:,:n_neus_show],
        labels=dict(x="units", y="bin lag (s)", color="regression coefficient"),
        y=lag_seconds,  # Use numerical labels for y-axis
        x=[f"unit {u}" for u in units_fit_resorted][:n_neus_show],
        title=f"regression coefficients for fold {k_fold_check} and behaviour: {behaviour_check}",
        color_continuous_scale='PuOr_r',
        color_continuous_midpoint=0.0,
        height=900  # Add this line to make the figure taller
    )

    fig_heatmap_check.show(renderer="vscode")


#%% Next test:
# using again the coefficients averaged over all folds, I assign a rank value to each neuron for each label
# df with unit, rank_escape, rank_pup_run... rank_<behaviour_name>

all_labels = behaviour_representation_df['label'].values
_df_row_temp=[]
#dict_behavior_label_to_index['pup_grab_and_pup_retrieve'] = dict_behavior_label_to_index['pup_grab']

for _lab in all_labels:
    _coefficients = get_loading_average(_lab)
    _signs  = get_loading_sign(_lab)
    # score per unit: sum of squares across lags
    units_score = (_coefficients ** 2).sum(axis=0)  # shape: (n_units,)
    # order (asc) and invert to get rank per unit index
    order_asc = np.argsort(units_score)
    rank_per_unit_index = np.empty_like(order_asc)
    rank_per_unit_index[order_asc] = np.arange(order_asc.size)

    for i, _u in enumerate(units_fit):
        _df_row_temp.append({
            "unit": _u,
            "label": _lab,
            "score": float(units_score[i]),
            "rank": int(rank_per_unit_index[i]),  # 0 = highest score
            "sign": int(_signs[i]), # 1 = positive, -1 = negative
            "signed_rank": int((rank_per_unit_index[i]+1) * _signs[i]), 
        })

df_unit_ranks = pd.DataFrame(_df_row_temp)

# Pivot to wide (one row per unit, one column per behaviour rank)
label_order = list(all_labels)
df_unit_ranks_wide = df_unit_ranks.pivot(index="unit", columns="label", values="signed_rank")
df_unit_ranks_wide = df_unit_ranks_wide.reindex(columns=label_order)  # keep original label order
df_unit_ranks_wide.columns = [f"rank_{c}" for c in df_unit_ranks_wide.columns]
df_unit_ranks_wide = df_unit_ranks_wide.reset_index()

# ensure integer dtype
rank_cols = [c for c in df_unit_ranks_wide.columns if c.startswith("rank_")]
df_unit_ranks_wide[rank_cols] = df_unit_ranks_wide[rank_cols].astype(int)

# replace original variable
df_unit_ranks = df_unit_ranks_wide

# assert unit column is made of unique elements
if not df_unit_ranks['unit'].is_unique:
    raise ValueError("Unit column is not unique")


#%% 
# Now scatter plot one rank against the other
# Grab and retrieve have similar rank!

def plot_rank_scatter(rank_test_1, rank_test_2):

    # Positions: absolute ranks (ignore sign in position)
    x_signed = df_unit_ranks[rank_test_1].to_numpy()
    y_signed = df_unit_ranks[rank_test_2].to_numpy()
    xplot = np.abs(x_signed)
    yplot = np.abs(y_signed)

    # Signs for color/legend; force zeros to +1 just in case
    sx = np.sign(x_signed)
    sy = np.sign(y_signed)
    sx[sx == 0] = 1
    sy[sy == 0] = 1
    sx_str = np.where(sx > 0, "+", "-")
    sy_str = np.where(sy > 0, "+", "-")

    # Legend label per point: "+,+" "+,-" "-,+" "-,-" (no spaces)
    cat_labels = [f"{sx_str[i]},{sy_str[i]}" for i in range(len(sx_str))]

    # Build plotting dataframe
    df_plot = pd.DataFrame({
        "x": xplot,
        "y": yplot,
        "signs": cat_labels
    })

    # Fixed categories and colors (greens when first sign '+', reds/orange when '-')
    c_pp = "+,+"
    c_pn = "+,-"
    c_np = "-,+"
    c_nn = "-,-"
    color_map = {
        c_pp: "#1a9850",  # green (darker)
        c_pn: "#a6d96a",  # green (lighter)
        c_np: "#fdae61",  # orange (lighter)
        c_nn: "#d73027",  # red (darker)
    }

    # Short names without the "rank_" prefix for labeling
    name1 = rank_test_1[5:] if rank_test_1.startswith("rank_") else rank_test_1
    name2 = rank_test_2[5:] if rank_test_2.startswith("rank_") else rank_test_2

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="signs",
        category_orders={"signs": [c_pp, c_pn, c_np, c_nn]},
        color_discrete_map=color_map,
        labels={
            "x": name1,
            "y": name2,
            "signs": f"{name1} and {name2} signs"
        },
        title=f"Scatter plot of {name1} vs {name2}",
        height=600
    )
    fig.show(renderer="vscode")

    return None

#%%
 
rank_test_1 = "rank_pup_grab_and_pup_retrieve"
rank_test_2 = "rank_pup_run"

plot_rank_scatter(rank_test_1, rank_test_2)


#%%
# Let's try run and escape
rank_test_1 = "rank_pup_run"
rank_test_2 = "rank_escape"
plot_rank_scatter(rank_test_1, rank_test_2)

# WOW,they avoid each other

#%%
# let's try loiter and escape
rank_test_1 = "rank_loiter"
rank_test_2 = "rank_escape"
plot_rank_scatter(rank_test_1, rank_test_2)

#%%
# finally, loiter and pup run
rank_test_1 = "rank_loiter"
rank_test_2 = "rank_pup_run"
plot_rank_scatter(rank_test_1, rank_test_2)


#%% Compare retrieve and escape
rank_test_1 = "rank_pup_grab_and_pup_retrieve"
rank_test_2 = "rank_escape"
plot_rank_scatter(rank_test_1, rank_test_2)


#%% get score of escape behaviour for each fold
escape_recall = []
escape_precision = []
escape_support = []
escape_f1 = []

for _k in range(n_escapes):
    _lab = str(label_escape)
    report_k = all_folds_data[_k]['report']
    escape_recall.append(report_k[_lab]['recall'])
    escape_precision.append(report_k[_lab]['precision'])
    escape_support.append(int(report_k[_lab]['support']))
    escape_f1.append(report_k[_lab]['f1-score'])

# print a couple
print(f"Escape recall all folds: {escape_recall}")
print(f"Escape precision all folds: {escape_precision}")
print(f"Escape f1-score all folds: {escape_f1}")
print(f"Escape support all folds: {escape_support}")


#%% Get average f1 score of all labels, not just escape label!
# create dataframe with label, and f1-average
_df_temp_rows=[]

for _k in range(n_escapes):
    # iterate over rows of behaviour_representation_df
    report_k = all_folds_data[_k]['report']
    for _beh_row in behaviour_representation_df.itertuples(index=False):
        _lab_name = _beh_row.label
        _lab_id = str(_beh_row.class_index)
        _df_temp_rows.append({
            "fold": _k,
            "label": _lab_name,
            "class_index": _lab_id,
            "f1-score": report_k[_lab_id]['f1-score']
        })

df_f1_by_fold = pd.DataFrame(_df_temp_rows)

# now group by label and average
df_f1_avg = df_f1_by_fold.groupby("label")["f1-score"].mean().reset_index()

# and make a bar plot using plotly express
fig_bar = px.bar(df_f1_avg, x="label", y="f1-score", title="Average F1 Score by Label")
# limits between 0 and 1
fig_bar.update_yaxes(range=[0, 1])
fig_bar.show(renderer="vscode")

#%%
# Print and plot confusion matrix for each fold using saved .npy files (plotly express)
for _k in range(n_escapes):
    cm = all_folds_data[_k]['confusion_matrix']
    lda = all_folds_data[_k]['pipeline'].named_steps['lda']
    classes = lda.classes_
    class_names = [dict_classindex_to_behaviour.get(int(c), str(c)) for c in classes]

    # Pretty print as a labeled table
    df_cm = pd.DataFrame(cm, index=[f"{n}" for n in class_names],
                            columns=[f"{n}" for n in class_names])
    # Plot annotated heatmap with plotly express
    fig = px.imshow(
        cm,
        x=[f"{n}" for n in class_names],
        y=[f"{n}" for n in class_names],
        color_continuous_scale='Blues',
        text_auto=True,
        aspect='equal'
    )
    fig.update_layout(
        title=f"Confusion matrix - fold {_k}",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        height=max(450, 35 * len(class_names) + 200),
        width=max(500, 35 * len(class_names) + 250),
        coloraxis_showscale=True
    )
    fig.update_xaxes(side='top', tickangle=45)
    fig.show(renderer="vscode")

# %%
