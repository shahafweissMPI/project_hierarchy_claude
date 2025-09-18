# -*- coding: utf-8 -*-
"""
Created on 2025-08-22

@author: Dylan Festa

This adapts and re-exports the ranking dataframe, to create a graph 
representation of it, with only the k_top units for each behaviour.
"""
#%%
import os
import numpy as np, pandas as pd, xarray as xr
import json
import math
import pickle

the_mouse='afm16924'

read_file = os.path.join(os.path.dirname(__file__), "local_outputs", f"df_ranking_{the_mouse}.pkl")

if not os.path.exists(read_file):
    raise FileNotFoundError(f"File {read_file} does not exist. Please run `fit_and_save.py` first.")

with open(read_file, "rb") as f:
    df_ranking = pickle.load(f)
#%%

the_sessions = df_ranking['session'].unique()
print(f"Found {len(the_sessions)} sessions: {the_sessions}")

the_session = the_sessions[0]

#%%

k_top = 10

df_session_ranking = df_ranking.query("session == @the_session and rank < @k_top").copy()

# add column rank_val_rel, that goes from -1 to 1
# so that zero stays zero, min is -1 , max is +1
def rescale_rank_val_signed(df, col="rank_val", new_col="rank_val_rel"):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe.")
    s = df[col].copy()

    neg_mask = s < 0
    pos_mask = s > 0
    zero_mask = s == 0

    if not neg_mask.any():
        raise ValueError("No negative values found in 'rank_val'; cannot create symmetric scaling.")

    # Scale negatives: most negative -> -1
    neg_vals = s[neg_mask]
    neg_min = neg_vals.min()  # most negative (e.g., -5)
    neg_scaled = neg_vals / (-neg_min)  # (-5)/5 = -1

    # Scale positives: max positive -> +1
    if pos_mask.any():
        pos_vals = s[pos_mask]
        pos_max = pos_vals.max()
        if pos_max == 0:
            # Should not happen due to mask, but guard anyway
            pos_scaled = pos_vals * 0
        else:
            pos_scaled = pos_vals / pos_max
    else:
        pos_scaled = pd.Series(dtype=float)

    # Assemble new series
    out = pd.Series(index=s.index, dtype=float)
    out[neg_mask] = neg_scaled
    out[pos_mask] = pos_scaled
    out[zero_mask] = 0.0

    df[new_col] = out
    return df

df_session_ranking = rescale_rank_val_signed(df_session_ranking)


#%%
# Now the rest of the script...
df_export = df_session_ranking.copy()


#%%
# ---------- helpers for colors ----------
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def _lerp(a, b, t):
    return tuple(int(round((1-t)*a[i] + t*b[i])) for i in range(3))

# Diverging palette: blue (-1) → gray (0) → red (+1)
POS = _hex_to_rgb('#2c7bb6')  # blue
NEU = _hex_to_rgb('#bdbdbd')  # neutral gray
NEG = _hex_to_rgb('#d73027')  # red

def strength_to_color(s):
    if pd.isna(s):
        return _rgb_to_hex(NEU)
    s = max(-1.0, min(1.0, float(s)))
    if s >= 0:
        c = _lerp(NEU, POS, s)
    else:
        c = _lerp(NEU, NEG, -s)
    return _rgb_to_hex(c)

# ---------- build nodes ----------
units = sorted(df_export["unit"].unique())
behaviours = sorted(df_export["behaviour"].unique())

nodes = []
# Units on level 0 (for a clean two‑column hierarchical layout)
for u in units:
    nodes.append({
        "id": f"u{u}",
        "label": str(u),
        "group": "unit",
        "level": 0
    })

# behaviours on level 1
for b in behaviours:
    nodes.append({
        "id": f"b:{b}",
        "label": str(b),
        "group": "behaviour",
        "level": 1
    })

# ---------- build edges ----------
has_strength = "rank_val_rel" in df_export.columns
edges = []
for row in df_export.itertuples(index=False):
    u = getattr(row, "unit")
    b = getattr(row, "behaviour")
    edge = {
        "from": f"u{u}",
        "to":   f"b:{b}",
        "smooth": False
    }
    if has_strength:
        s = getattr(row, "rank_val_rel")
        edge["strength"] = None if pd.isna(s) else float(s)   # keep raw strength if you'd rather color in JS
        # If you prefer to bake color/width in Python, uncomment the next two lines:
        edge["color"] = strength_to_color(s)
        edge["width"] = 1 + 3*abs(0.0 if pd.isna(s) else float(s))  # 1–4 px
        edge["title"] = f"strength: {float(s):+.2f}" if not pd.isna(s) else "strength: n/a"
    edges.append(edge)

# ---------- write JSON ----------

savedir = "/tmp"

with open(f"{savedir}/network_data.json", "w", encoding="utf-8") as f:
    json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

print("Wrote network_data.json with", len(nodes), "nodes and", len(edges), "edges.")

# %%
