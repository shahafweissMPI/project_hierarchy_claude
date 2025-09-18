# -*- coding: utf-8 -*-
"""
📃 ./save_data_locally.py

🕰️ created on 2025-09-09

Read data and save it locally, to make it easier to access it later.

Data to save is chosen based on the mouse/session found in the videos folder!

"""



#%%
from __future__ import annotations
from typing import List,Dict,Tuple

import numpy as np, pandas as pd, xarray as xr
import re
from pathlib import Path
import tempfile
import read_data_light as rdl
import preprocess as pre
import pickle


# date string in the format YYYYMMDD
import time,datetime
date_str = datetime.datetime.now().strftime("%Y%m%d")

#%%
# get mouse/session combinations to save


movies_path = Path.home() / 'Videos800Compressed'

assert movies_path.exists(), f"Movies path {movies_path} does not exist."


#%%

def get_mouse_session_from_video_path(video_path: Path) -> tuple[str, str]:
    pattern = re.compile(r'^(?P<mouse>afm\d{5})_(?P<session>.+?)(?=\.mp4$)')
    m = pattern.match(video_path.name)
    if m:
        return m.group('mouse'), m.group('session')
    else:
        raise ValueError(f"Video path {video_path} does not match expected pattern.")
def generate_mouse_session_df(movies_path: Path,
                              *,
                              session_filter: List[str]) -> pd.DataFrame:
    df_rows = []
    for p in movies_path.glob("*.mp4"):          # or rglob if you need recursion
        try:
            mouse, session = get_mouse_session_from_video_path(p)
            # make sure that no element of session_filter is in session
            if not any(filter_str in session for filter_str in session_filter):
                mouse_session_str = f"{mouse} / {session}"
                rec = {
                    "mouse": mouse,
                    "session": session,
                    "video_path": p.resolve(),
                    "mouse_session_name": mouse_session_str}
                df_rows.append(rec)
        except ValueError:
            continue
    df_videos = pd.DataFrame(df_rows, columns=["video_path", "mouse", "session", "mouse_session_name"])
    df_videos.sort_values(by=["mouse", "session"], inplace=True)
    return df_videos


session_filter = ['test','Kilosort', 'coded','overlap','raw']
df_videos = generate_mouse_session_df(movies_path, session_filter=session_filter)


#%%

path_temp_saves = Path(tempfile.gettempdir()) / "TempDatadictSaves"
path_temp_saves.mkdir(parents=True, exist_ok=True)

#%%
t_start = datetime.datetime.now()

for _row in df_videos.itertuples():
    mouse = _row.mouse
    session = _row.session
    print(f"Processing {mouse} / {session}")
    rdl.save_preprocessed_dict(mouse, session, path_temp_saves)
    print(f"Saved {mouse} / {session} to {path_temp_saves}")

t_delta = datetime.datetime.now() - t_start
duration_mm_ss_string = time.strftime("%M:%S", time.gmtime(t_delta.total_seconds()))
print(f"Done! Total time: {duration_mm_ss_string}")
