from __future__ import annotations
from typing import List,Union,Tuple,Optional
import os

import numpy as np
import xarray as xr
import pandas as pd
import warnings

from dataclasses import dataclass
from collections.abc import Sequence
from numpy.typing import NDArray
from abc import ABC
from preprocess.binning_and_statistics import SpikeTrains



# Some utility functions

def cut_train_at_event(t_event: float,
        spike_train: NDArray[np.float_],t_pad_left: float, t_pad_right: float,
        *, warn_on_empty: bool = False) -> NDArray[np.float_]:
    """
    Generates a new spike train cut around the specificed event time
    
    """
    t_start = t_event - t_pad_left
    t_end = t_event + t_pad_right
    new_train = spike_train[(spike_train >= t_start) & (spike_train <= t_end)] - t_event
    if warn_on_empty and new_train.size == 0:
        print(f"Warning: cut_train_at_event produced an empty spike train for event at {t_event} with padding ({t_pad_left}, {t_pad_right}). Original train had {spike_train.size} spikes.")
    return new_train


class SpikeTrainByTrials(SpikeTrains):
    """
    Extension of SpikeTrains for trial-based spike trains.
    Inherits all methods and attributes from SpikeTrains.

    The `units` attribute now indicates the trial number, always starting from 0

    Additional attributes:
    - event_times: np.ndarray
            time in seconds at which the events occur
    - t_pad_left: float
            padding to the left of the event
    - t_pad_right: float
            total padding to the right of the event.
            This is a hard limit, if events are longer than this, the spike train 
            will be cut *before* the end of the event.
    - event_durations: np.ndarray
            duration of each event in seconds. If events are POINT-like, this will be an array of zeros.
    - unit_id : Union[str, int] the neuron displayed across trials

    Attributes with different meaning:
    - n_units: int
            now indicates the number of trials
    - units: np.ndarray
            now indicates the trial number, always starting from 0
            (i.e. units = np.arange(n_trials))
    - trains: tuple[NDArray[np.float64], ...]
            These are the spikes for the *same* unit corresponding to the `unit_id` attribute,
            cut around each of the event times with the specified padding.
            The time is *relative* to the event time, i.e. 0 corresponds to the event time.
            Negative spike time indicates spikes before the event, etc.
    
    """
    def __init__(self, *args, t_pad_left: float = 0.0, t_pad_right: float = 0.0,
                 event_times: np.ndarray = None, event_durations: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_pad_left = t_pad_left
        self.t_pad_right = t_pad_right
        self.event_times = event_times if event_times is not None else np.array([])
        self.event_durations = event_durations if event_durations is not None else np.array([])

    @classmethod
    def from_spike_trains(cls,unit_id,
                    spike_trains: SpikeTrains,
                    time_event_starts: Sequence[float],
                    *,
                    time_event_stops: Optional[Sequence[float]] = None,
                    t_pad_left: float = 3.0,
                    t_pad_right: float = 10.0):
        """
        Constructs a SpikeTrainByTrials object from a SpikeTrains object by first picking the 
        spike train of the unit specified by `unit_id`, and then cutting it at each of the event times
        provided in `time_event_starts`, with optional padding before and after the event.
        For START-STOP events please provide `time_event_stops` as well.
 
        If `time_event_stops` is provided, but some events are longer than the padding,
        throws a warning.
 
        
        Arguments
        ----------
          - unit_id: Union[str, int] the unit to be extracted from spike_trains
                     If missing, or multiple matches, it raises a ValueError
          - spike_trains: SpikeTrains the SpikeTrains object containing the spike trains
          - time_event_starts: Sequence[float] the start times of the events
          - time_event_stops: Optional[Sequence[float]] the stop times of the events
                     If missing, it assumes POINT events
          - t_pad_left: float, default 3.0
                     the padding to apply to the left of the event
          - t_pad_right: float, default 10.0
                     the padding to apply to the right of the event
 
        """ 
        if t_pad_left < 0 or t_pad_right < 0:
            raise ValueError("t_pad_left and t_pad_right must be non‑negative")
        # find the unit index
        unit_idx_match = np.where(spike_trains.units == unit_id)[0]
        if unit_idx_match.size == 0:
            raise ValueError(f"Unit {unit_id} not found in provided SpikeTrains.")
        if unit_idx_match.size > 1:
            raise ValueError(f"Unit {unit_id} has multiple matches; expected exactly one.")
        unit_idx = int(unit_idx_match[0])
        base_train = spike_trains.trains[unit_idx]
        base_location = spike_trains.unit_location[unit_idx]
        starts = np.asarray(list(time_event_starts), dtype=float)
        if starts.ndim != 1 or starts.size == 0:
            raise ValueError("time_event_starts must be a non‑empty 1D sequence")
        if not np.all(np.diff(starts) >= 0):
            raise ValueError("time_event_starts must be sorted in non‑decreasing order")
 
        # handle stops / durations
        if time_event_stops is not None:
            stops = np.asarray(list(time_event_stops), dtype=float)
            if stops.shape != starts.shape:
                raise ValueError("time_event_stops must have same length as time_event_starts")
            if not np.all(stops >= starts):
                raise ValueError("All stop times must be >= corresponding start times")
            durations = stops - starts
            longer_mask = durations > t_pad_right
            if np.any(longer_mask):
                warnings.warn("Some event durations exceed t_pad_right; spikes after t_pad_right "
                              "relative to start are excluded.", RuntimeWarning)
        else:
            durations = np.zeros_like(starts)
 
        # build trial spike trains (relative times: event at 0)
        trial_trains: List[np.ndarray] = []
        for t_start_event in starts:
            rel_train = cut_train_at_event(t_start_event, base_train,
                                           t_pad_left, t_pad_right,
                                           warn_on_empty=False)
            trial_trains.append(rel_train.astype(np.float64, copy=False))
 
        n_trials = len(trial_trains)
        units_trials = np.arange(n_trials, dtype=int)
        unit_location_trials = [base_location] * n_trials
 
        # relative window for all trials
        rel_t_start = -float(t_pad_left)
        rel_t_stop = float(t_pad_right)

        obj = cls(
            n_units=n_trials,
            t_start=rel_t_start,
            t_stop=rel_t_stop,
            units=units_trials,
            unit_location=unit_location_trials,
            trains=tuple(trial_trains),
            t_pad_left=t_pad_left,
            t_pad_right=t_pad_right,
            event_times=starts.copy(),
            event_durations=durations.copy()
        )
        # store which original unit these trials correspond to
        obj.unit_id = unit_id
        return obj

    def get_trials_and_relative_times(self,t:float):
        """
        Given an absolute time `t` returns ALL the trials where this time falls within the trial window,
        and all the corresponding relative times within those trials.
        Normally this should be a single point, except when events are less that t_pad_left + t_pad_right apart.
    
        Arguments
        ----------
          - t: float absolute time in seconds
          
        Returns
        ----------
          - trials: np.ndarray of int the trial indices where `t` falls within the trial window
          - rel_times: np.ndarray of float the corresponding relative times within those trials
        """
        if not np.isfinite(t):
            raise ValueError("t must be a finite float")
        if self.event_times is None or self.event_times.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        rel_times = t - self.event_times  # relative to event start (event at 0)
        in_window = (rel_times >= -float(self.t_pad_left)) & (rel_times <= float(self.t_pad_right))
        trial_idxs = np.nonzero(in_window)[0].astype(int, copy=False)
        return trial_idxs, rel_times[in_window].astype(float, copy=False)

    @property
    def n_trials(self) -> int:
        return self.n_units
    @property
    def is_point_event(self) -> bool:
        return np.all(self.event_durations == 0.0)
    @property
    def is_start_stop_event(self) -> bool:
        return np.any(self.event_durations > 0.0)
