# -*- coding: utf-8 -*-
"""
Created on 2025-08-31

@author: Dylan Festa

Reads and plots data saved by `05_find_decoding_neurons_by_exclusion.py`

"""
#%%
from __future__ import annotations

import os
import numpy as np, pandas as pd, xarray as xr
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

#%%


path_this_file = Path(__file__).resolve()
path_data = path_this_file.parent / "local_outputs_05byexclusion"
if not path_data.exists():
    raise FileNotFoundError(f"Data directory not found: {path_data}")

#%%
# now read all subfolders in path_data and extract mouse,session combinations
records = []
for p in [p for p in path_data.iterdir() if p.is_dir()]:
    name = p.name
    if "_" not in name:
        print(f"Skipping (no underscore): {name}")
        continue
    mouse_candidate, session = name.split("_", 1)
    # validate mouse pattern: 3 letters + 5 digits
    if len(mouse_candidate) == 8 and mouse_candidate[:3].isalpha() and mouse_candidate[3:].isdigit():
        records.append({
            "mouse": mouse_candidate,
            "session": session,
            "path": p,              # Path object (kept for convenience)
            "full_path": str(p)     # Added string version
        })
    else:
        print(f"Skipping (mouse pattern mismatch): {name}")

mouse_session_df = pd.DataFrame(records).sort_values(["mouse", "session"]).reset_index(drop=True)

print(f"Discovered {len(mouse_session_df)} mouse-session folders.")
print(mouse_session_df[["mouse","session","full_path"]])

#%%

def read_pkl_file(filename,path_mouse_session):
    # check if it exists
    if not os.path.exists(path_mouse_session):
        raise FileNotFoundError(f"Path not found: {path_mouse_session}")
    # make it Path if not already a Path
    if not isinstance(path_mouse_session, Path):
        path_mouse_session = Path(path_mouse_session)
    file_full_path = path_mouse_session / filename
    if not file_full_path.exists():
        raise FileNotFoundError(f"File not found: {file_full_path}")
    # read the pickle file
    with open(file_full_path, "rb") as f:
        data = pickle.load(f)
    return data


def read_reduction_df(path_mouse_session):
    return read_pkl_file("reduction_log_df.pkl", path_mouse_session)
def read_first_out_dict(path_mouse_session):
    return read_pkl_file("first_out_dict.pkl", path_mouse_session)


def f1_score_weight(f1_score: float | pd.Series | np.ndarray, *, min_f1: float = 0.65):
    """
    Vectorized weight:
      - If f1_score >= min_f1 -> weight 0
      - Else linear penalty up to 1 as f1_score -> 0
    Accepts scalar or array-like.
    """
    f1_arr = np.asarray(f1_score, dtype=float)
    f1_less = np.maximum(0.0, f1_arr-min_f1)
    weights = f1_less / (1 - min_f1)
    return weights

def extract_iteration_output_df(reduction_df, *, min_f1: float = 0.65):
    iteration_output_df_rows_ = []
    for _, row in reduction_df.iterrows():
        iteration_ = row['iteration']
        output_df = row['output_dict']['output_df']
        # add iteration column to df
        output_df = output_df.copy()
        output_df['iteration'] = iteration_
        iteration_output_df_rows_.append(output_df)
    iteration_output_df = pd.concat(iteration_output_df_rows_, ignore_index=True)
    # vectorized weighting over the f1_score column
    f1_weights = f1_score_weight(iteration_output_df['f1_score'], min_f1=min_f1)
    f1_weights = pd.Series(f1_weights, index=iteration_output_df.index, name='f1_weight')
    iteration_output_df['f1_weight'] = f1_weights
    iteration_output_df['weightf1_score'] = iteration_output_df['weight_score'] * f1_weights
    return iteration_output_df

def extract_iteration_df_from_path(path_mouse_session,*,min_f1: float = 0.65):
    reduction_df = read_reduction_df(path_mouse_session)
    return extract_iteration_output_df(reduction_df, min_f1=min_f1)


def extract_interation_bestiter(extract_iteration_df):
    group_keys = ['mouse','session','unit', 'behaviour']
    idx_max = extract_iteration_df.groupby(group_keys)['weightf1_score'].idxmax()

    # columns we want to retain (support may not always exist)
    base_cols = ['mouse', 'session', 'unit', 'behaviour', 'iteration', 'weight_score', 'f1_score', 'weightf1_score']

    ret_df = (
    extract_iteration_df
    .loc[idx_max, base_cols]
    .reset_index(drop=True))
    ret_df.sort_values(by=['mouse','session','unit', 'behaviour'], inplace=True)
    return ret_df

#%%

extract_iteration_all_df_row_all = []
# iterate on mouse_session_df
for _, row in mouse_session_df.iterrows():
    the_mouse_ = row['mouse']
    the_session_ = row['session']
    df_to_add = extract_iteration_df_from_path(row['full_path'], min_f1=0.65)
    # add mouse, session columns
    df_to_add['mouse'] = the_mouse_
    df_to_add['session'] = the_session_
    extract_iteration_all_df_row_all.append(df_to_add)


extract_iteration_all_df = pd.concat(extract_iteration_all_df_row_all, ignore_index=True)
extract_iteration_bestiter_unit_df = extract_interation_bestiter(extract_iteration_all_df)

# For each mouse, session, behavior, print on screen unique units in extract_iteration_bestiter_unit_df
bestiter_unique_units_counts_df = (
    extract_iteration_bestiter_unit_df
    .groupby(['mouse','session','behaviour'])['unit']
    .nunique()
    .reset_index(name='n_unique_units')
    .sort_values(['mouse','session','behaviour'])
)
print("[INFO] Unique units per (mouse, session, behaviour) using best iteration per unit:")
print(bestiter_unique_units_counts_df.to_string(index=False))
print(f"[INFO] Total rows: {bestiter_unique_units_counts_df.shape[0]}")
#%%

# Now create a new dataframe where for each mouse, session, behaviour, I keep only the top 10% (flexible)
# with the highest values of weightf1_score

def select_top_fraction(df: pd.DataFrame,
                        *,
                        group_cols: list[str],
                        score_col: str = 'weightf1_score',
                        frac: float = 0.10,
                        min_rows: int = 1,
                        tie_handling: str = 'include_ties') -> pd.DataFrame:
    """
    Return top fraction (by score_col) within each group.

    Parameters
    ----------
    df : input dataframe (must contain group_cols and score_col)
    group_cols : columns defining groups (e.g. ['mouse','session','behaviour'])
    score_col : column to rank by (descending)
    frac : fraction (0<frac<=1); number kept per group = ceil(frac * group_size), >= min_rows
    min_rows : minimum rows to keep per group
    tie_handling :
        'include_ties' -> include all rows whose score equals the cutoff score
        'strict'       -> take exactly computed number (after sorting)
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0,1].")
    missing_cols = [c for c in group_cols + [score_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    # Drop NaN scores (cannot rank them)
    work = df.dropna(subset=[score_col]).copy()
    if work.empty:
        return work

    parts = []
    for g_vals, g_df in work.groupby(group_cols, sort=False):
        g_df_sorted = g_df.sort_values(score_col, ascending=False)
        g_n = g_df_sorted.shape[0]
        keep_n = max(min_rows, int(np.ceil(frac * g_n)))
        keep_n = min(keep_n, g_n)
        #print(f'[INFO] Group: {g_vals}, Size: {g_n}, Keep: {keep_n}')
        if keep_n == 0:
            continue
        if tie_handling == 'include_ties':
            cutoff_score = g_df_sorted.iloc[keep_n - 1][score_col]
            g_keep = g_df_sorted[g_df_sorted[score_col] >= cutoff_score]
        else:  # 'strict'
            g_keep = g_df_sorted.head(keep_n)
        parts.append(g_keep)
    if not parts:
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(parts, ignore_index=True)
    # Optional: re-sort for readability
    out = out.sort_values(group_cols + [score_col], ascending=[True]*len(group_cols) + [False])
    return out

# Create the best subset (top 10%)
extract_iteration_top_subset_df = select_top_fraction(
    extract_iteration_bestiter_unit_df,
    group_cols=['mouse', 'session', 'behaviour'],
    score_col='weightf1_score',
    frac=0.10,
    min_rows=3,
    tie_handling='strict'
)

print(f"[INFO] Top subset shape: {extract_iteration_top_subset_df.shape} "
      f"(from {extract_iteration_all_df.shape})")
print(extract_iteration_top_subset_df.groupby(['mouse','session','behaviour']).size()
      .rename('rows_kept').head())

#%%
# print of screen number of unique neurons left in extract_iteration_best_df for each moise,session combination
unique_units_per_session = (
    extract_iteration_top_subset_df
    .groupby(['mouse', 'session'])['unit']
    .nunique()
    .reset_index(name='n_unique_units')
    .sort_values(['mouse', 'session'])
)
print("[INFO] Unique units per (mouse, session) in top subset:")
print(unique_units_per_session.to_string(index=False))


#%% 
# for each mouse, session, count how many neurons are repetead more than once
# in extract_iteration_top_subset_df and print on screen.
print("[INFO] Unit repetition summary within top subset (counts are per unit across behaviours).")
if extract_iteration_top_subset_df.empty:
    print("[WARN] Top subset is empty; skipping repetition summary.")
else:
    for (mouse_, session_), grp in extract_iteration_top_subset_df.groupby(['mouse','session']):
        unit_occurrences = grp.groupby('unit').size()              # occurrences per unit
        freq_distribution = unit_occurrences.value_counts()        # key: times a unit appears -> number of units
        freq_distribution = freq_distribution.sort_index()
        total_unique_units = int(unit_occurrences.shape[0])
        # Build narrative parts
        parts = []
        for times, n_units in freq_distribution.items():
            label_times = "time" if times == 1 else "times"
            parts.append(f"{n_units} units appear {times} {label_times}")
        # Consistency check
        if freq_distribution.sum() != total_unique_units:
            print(f"[ERROR] Inconsistency for {mouse_} {session_}: freq sum {freq_distribution.sum()} "
                  f"!= total unique {total_unique_units}")
        summary_line = (f"[UNITS] mouse={mouse_} session={session_} | total_unique_units={total_unique_units} | " +
                        ", ".join(parts))
        print(summary_line)

#%%
# save in path_data the dictionary extract_iteration_top_subset_df
top_dict_save_name = "top_subset_df"
top_dict_save_full_path = path_data / top_dict_save_name
# --- SAVE TOP SUBSET AS PICKLED DICT ---
top_subset_dict = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_script": path_this_file.name,
    "path_data": str(path_data),
    "n_rows": int(extract_iteration_top_subset_df.shape[0]),
    "columns": list(extract_iteration_top_subset_df.columns),
    "data": extract_iteration_top_subset_df.to_dict(orient="records"),
}
top_subset_pkl_path = path_data / f"{top_dict_save_name}.pkl"
with open(top_subset_pkl_path, "wb") as f:
    pickle.dump(top_subset_dict, f)
print(f"[SAVE] Top subset dict written: {top_subset_pkl_path} "
      f"(rows={top_subset_dict['n_rows']})")
# Optional: also save a CSV view (comment out if not needed)
# extract_iteration_top_subset_df.to_csv(path_data / f"{top_dict_save_name}.csv", index=False)
# --- END SAVE BLOCK ---

# OLD STUFF

#%%
# print of screen number of unique neurons left in extract_iteration_best_df for each moise,session combination


#%%

row_read = mouse_session_df.loc[0] 

the_mouse_ = row_read['mouse']
the_session_= row_read['session']
the_path_ = row_read['path']

reduction_df_ = read_reduction_df(the_path_)
first_out_dict_ = read_first_out_dict(the_path_)

#%%
units_start = first_out_dict_.get("units", [])
n_units_start = len(units_start)

#%%
# check that all units are still present on first round
units_start_check = reduction_df_.loc[0]['output_dict']['units']

if not np.array_equal(units_start, units_start_check):
    raise ValueError("Units in first_out_dict and reduction_df do not match.")
# %%



def plot_performance_by_reduction(reduction_df, mouse, session,
                                  save: bool = False,
                                  show: bool = True,
                                  out_dir: Path | str | None = None):
    """
    Plot F1 scores across exclusion iterations.

    Parameters
    ----------
    reduction_df : DataFrame
    mouse, session : identifiers
    save : if True, save PNG to out_dir (default /tmp)
    show : if True, display interactive plot (previous default behavior)
    out_dir : optional directory for saving (created if missing)
    """
    scores_df_rows =[]
    for i, row in reduction_df.iterrows():
        iteration_ = row['iteration']
        out_beh_df = row['output_dict']['output_beh_df']
        f1_chase = out_beh_df.loc[out_beh_df['behaviour'] == 'chase', 'f1_score'].values[0]
        f1_pup_run = out_beh_df.loc[out_beh_df['behaviour'] == 'pup_run', 'f1_score'].values[0]
        f1_pup_retrieve = out_beh_df.loc[out_beh_df['behaviour'] == 'pup_retrieve', 'f1_score'].values[0]
        f1_no_label = out_beh_df.loc[out_behDf['behaviour'] == 'no_label', 'f1_score'].values[0]   # FIX: was out_behDf
        scores_df_rows.append({
            'iteration': iteration_,
            'f1_chase': f1_chase,
            'f1_pup_run': f1_pup_run,
            'f1_pup_retrieve': f1_pup_retrieve,
            'f1_no_label': f1_no_label
        })
    scores_df_ = pd.DataFrame(scores_df_rows)
    if scores_df_.empty:
        raise ValueError("Scores dataframe is empty!!! Something is wrong!")

    fig = go.Figure()
    behaviours_map = [
        ('f1_chase', 'chase', '#1f77b4'),
        ('f1_pup_run', 'pup_run', '#ff7f0e'),
        ('f1_pup_retrieve', 'pup_retrieve', '#2ca02c'),
        ('f1_no_label', 'no_label', '#7f7f7f'),
    ]
    for col, name, color in behaviours_map:
        if col in scores_df_.columns:
            fig.add_trace(go.Scatter(
                x=scores_df_['iteration'],
                y=scores_df_[col],
                mode='lines+markers',
                name=name,
                line=dict(width=2, color=color),
                marker=dict(size=6)
            ))
    fig.update_layout(
        title=f'f1-scores exclusion test -- mouse: {mouse}, session: {session}',
        xaxis_title='number of excluded neurons',
        yaxis_title='F1 score',
        template='plotly_white',
        legend_title='Behaviour',
        hovermode='x unified',
        height=600
    )
    fig.update_yaxes(showgrid=True, dtick=0.1, gridcolor='rgba(0,0,0,0.2)', gridwidth=1)

    if save:
        if out_dir is None:
            out_dir = Path("/tmp/plots_05byexclusion")
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"f1_exclusion_{mouse}_{session}.png"
        fig.write_image(out_dir / fname)
    if show:
        fig.show()
    return None


# %%

# iterate on mouse_session_df
for _, row in mouse_session_df.iterrows():
    the_mouse_ = row['mouse']
    the_session_ = row['session']
    the_path_ = row['path']
    reduction_df_ = read_reduction_df(the_path_)
    plot_performance_by_reduction(reduction_df_, the_mouse_, the_session_,save=True)

# %%

# reduction df from first row
reduction_df_ = read_reduction_df(mouse_session_df.iloc[0]['path'])
iter_scoresweights_all_df = extract_iteration_output_df(reduction_df_,min_f1=0.8)
# remove all rows with f1_weight is zero
iter_scoresweights_df = iter_scoresweights_all_df.query('weightf1_score != 0').copy()

#sort by unit
iter_scoresweights_df.sort_values(by=['unit'], inplace=True)
# print first 20 rows
print(iter_scoresweights_df.head(20))
# %%

# now take MAX weight_score grouping by behaviour and unit (across iterations)

group_keys = ['unit', 'behaviour']
idx_max = iter_scoresweights_df.groupby(group_keys)['weightf1_score'].idxmax()

# columns we want to retain (support may not always exist)
base_cols = ['unit', 'behaviour', 'iteration', 'weight_score', 'f1_score', 'weightf1_score']
if 'support' in iter_scoresweights_df.columns:
    base_cols.insert(3, 'support')  # after iteration

iter_scores_beh_df = (
    iter_scoresweights_df
    .loc[idx_max, base_cols]
    .reset_index(drop=True)
)

# sort by unit, behaviour
iter_scores_beh_df.sort_values(by=['unit', 'behaviour'], inplace=True)
print(iter_scores_beh_df.head(20))

# (unit, behaviour) uniqueness already validated above.
# Optional diagnostic: how many behaviours per unit?
beh_per_unit = iter_scores_beh_df.groupby('unit')['behaviour'].nunique()
multi_beh_units = beh_per_unit[beh_per_unit > 1]
print(f"[INFO] Units appearing in >1 behaviour: {multi_beh_units.shape[0]} (expected if multi-class).")

#%%
# check the unit,behavior combinations are unique!
combo_cols = ['unit', 'behaviour']
dupe_mask = iter_scores_beh_df.duplicated(subset=combo_cols, keep=False)
if dupe_mask.any():
    dupes_df = (
        iter_scores_beh_df
        .loc[dupe_mask, combo_cols + [c for c in ['iteration','support','weight_score','f1_score','weightf1_score'] if c in iter_scores_beh_df.columns]]
        .sort_values(combo_cols)
    )
    print("[ERROR] Duplicate (unit, behaviour) combinations detected:")
    print(dupes_df.to_string(index=False))
    # Raise with a concise summary
    dup_combo_list = dupes_df[combo_cols].drop_duplicates().to_dict(orient='records')
    raise ValueError(f"Non-unique (unit, behaviour) combos found: {dup_combo_list}")
else:
    print(f"[OK] All {iter_scores_beh_df.shape[0]} (unit, behaviour) rows are unique.")

# %%

# --- NEW: density plots of weight_score per behaviour (one figure per behaviour) ---
behaviours_for_plot = iter_scores_beh_df['behaviour'].unique()
for beh in behaviours_for_plot:
    df_b = iter_scores_beh_df.query('behaviour == @beh')
    if df_b.empty:
        print(f"[WARN] No data for behaviour '{beh}', skipping density plot.")
        continue
    # Duplicate units WITHIN this behaviour (should be none after aggregation)
    intra_dupe_mask = df_b.duplicated(subset=['unit'], keep=False)
    if intra_dupe_mask.any():
        print(f"[ERROR] Unexpected duplicate units within behaviour '{beh}':")
        print(df_b.loc[intra_dupe_mask].to_string(index=False))
        raise ValueError(f"Duplicate units within behaviour '{beh}' after reduction step.")
    # show number of rows
    histvalues = df_b['weightf1_score'].values
    print(f"Behaviour '{beh}': {len(histvalues)} data points for density plot.")

    # Base histogram (density normalized)
    fig = px.histogram(
        df_b,
        x='weightf1_score',
        histnorm='density',
        nbins=60,
        opacity=0.7
    )

    # Add KDE curve if possible
    xs = None
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        vals = histvalues
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
        counts, edges = np.histogram(histvalues, bins=60, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        fig.add_trace(go.Scatter(
            x=centers, y=counts,
            mode='lines',
            name='hist density',
            line=dict(color='black', width=2)
        ))

    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(
        title=f"Density of weightf1_score for behaviour '{beh}'",
        xaxis_title='weightf1_score',
        yaxis_title='density',
        template='plotly_white',
        bargap=0.05
    )
    # fig.add_vline(
    #     x=weight_score_cutoff,
    #     line_color='red',
    #     line_dash='dash',
    #     annotation_text=f"cutoff={weight_score_cutoff:.3f}",
    #     annotation_position='top'
    # )
    # --- END NEW ---
    fig.show()

# %%
