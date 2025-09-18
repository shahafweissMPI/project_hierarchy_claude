from __future__ import annotations
from typing import List,Union,Tuple,Optional
import os

import numpy as np
import xarray as xr
import pandas as pd

from dataclasses import dataclass
from collections.abc import Sequence
from numpy.typing import NDArray
from abc import ABC


# simple untility functions

def get_bin_edges_and_centers(
    start_time: float, stop_time: float, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bin edges and centers for a given time range and bin size.
    """
    bin_edges = np.arange(start_time, stop_time, bin_size) # this will not include the last bin edge if it is incomplete
    bin_centers = bin_edges[:-1] + bin_size / 2.0  # Center of each bin
    return bin_edges, bin_centers


def get_bin_edges_and_centers_from_xarray(arr: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bin edges and centers from an xarray DataArray
    Assumes the DataArray has 't_start' , 't_stop' and 'dt' attributes.
    """
    if 't_start' not in arr.attrs or 't_stop' not in arr.attrs or 'dt' not in arr.attrs:
        raise ValueError("DataArray must have 't_start', 't_stop' and 'dt' attributes")
    t_start = arr.attrs['t_start']
    t_stop = arr.attrs['t_stop']
    dt = arr.attrs['dt']
    bin_edges, bin_centers = get_bin_edges_and_centers(t_start, t_stop, dt)
    return bin_edges, bin_centers

# Classes for spike trains and instantaneous firing rates (iFRs)

@dataclass(frozen=True)
class SpikeTrains(Sequence):
    """
    Immutable container for multiple neurons' spike trains.
    Indexing returns the underlying numpy array for that neuron.

    Attributes
    ----------
    n_units: int
        Number of neurons (units).
    t_start: float
        Start time of the analysis window.
    t_stop: float
        End time of the analysis window. If None, defaults to the maximum spike time + 1E-6.
    units: NDArray
        Array of unit indexes for each neuron.
    unit_location: List[str]
        List of locations for each unit
    trains: tuple[NDArray[np.float64], ...]
        Tuple of 1D numpy arrays, each containing spike times for a neuron.
        Empty arrays are allowed for neurons with no spikes.
    
    """
    n_units: int
    t_start: float
    t_stop: float
    units: NDArray
    unit_location: List[str]
    trains: tuple[NDArray[np.float64], ...]  # use tuple for immutability

    @classmethod
    def from_spike_list(cls,
                        spike_times_list: List[np.ndarray],
                        *,
                        units: np.ndarray,
                        unit_location: List[str],
                        t_start: float = 0.0,
                        t_stop: float = None,
                        isi_minimum: float = 0.0) -> SpikeTrains:
        """
        Construct a SpikeTrains object from a list of spike times and unit indexes.
        t_start defaults to 0.0, t_stop defaults to highest spike time + 1E-6.
        All spike times must be between t_start and t_stop.
        """
        if not isinstance(spike_times_list, list):
            raise TypeError(f"spike_times_list must be a list, got instead {type(spike_times_list).__name__}")
        if not isinstance(units, np.ndarray):
            units = np.array(units)
        n_units = len(spike_times_list)
        if len(units) != n_units:
            raise ValueError("units length must match spike_times_list length")
        if len(unit_location) != n_units:
            raise ValueError("unit_location length must match spike_times_list length")
        # if t_stop not provided, use the maximum spike time + 1E-6
        if t_stop is None:
            max_time = max((arr.max() if arr.size > 0 else t_start) for arr in spike_times_list)
            t_stop = max_time + 1E-6
        # Filter spike times to be within time bounds and apply isi_minimum if needed
        filtered_trains = []
        for arr in spike_times_list:
            arr = np.array(arr, dtype=np.float64)
            if arr.size == 0:
                filtered_trains.append(np.array([], dtype=np.float64))
                continue
            # initialize mask to all true
            mask = np.ones_like(arr, dtype=bool)
            mask &= (arr >= t_start) & (arr <= t_stop)
            if (isi_minimum > 0.0) and arr.size > 1:
                # Ensure spikes are at least isi_minimum apart
                isis = np.diff(arr)
                mask[1:] &= (isis >= isi_minimum)
            # Apply mask to filter spikes
            arr = arr[mask]
            filtered_trains.append(arr)
        trains_tuple = tuple(filtered_trains)
        return cls(n_units=n_units, t_start=t_start, t_stop=t_stop, units=units, unit_location=unit_location, trains=trains_tuple)

    def filter_by_time(self, t_start: float, t_stop: float) -> SpikeTrains:
        """
        Creates a NEW SpikeTrains object containing only spikes within the specified time window.
        If the specified time window is invalid, raises ValueError.
        """
        if t_start >= t_stop:
            raise ValueError(f"Invalid time window: {t_start} >= {t_stop}")
        # Filter each neuron's spike train by the time window
        filtered_trains = []
        for arr in self.trains:
            mask = (arr >= t_start) & (arr <= t_stop)
            filtered_trains.append(arr[mask])
        trains_tuple = tuple(filtered_trains)
        return SpikeTrains(n_units=self.n_units, t_start=t_start, t_stop=t_stop,
                           units=self.units, unit_location=self.unit_location,
                           trains=trains_tuple)

    def filter_by_unit_location(self, unit_location_include: str) -> SpikeTrains:
        """
        Creates a NEW SpikeTrains object containing only units with the specified location.
        If the specified location is not found in the original SpikeTrains, raises ValueError.
        
        Parameters
        ----------
        unit_location_include: str
            The location string to filter by (usually "PAG"). Only units with a name that includes this string will be included.
        
        Returns
        -------
        SpikeTrains
            A new SpikeTrains object containing only the units with the specified location.
        """

        # Get indices of units with the specified location
        indices_keep = [i for i, loc in enumerate(self.unit_location) if unit_location_include in loc]
        if not indices_keep:
            raise ValueError(f"No units found with location '{unit_location_include}'")
        # Filter units, locations, and spike trains, make copies of everything!
        filtered_units = self.units[indices_keep].copy()
        filtered_unit_location = [self.unit_location[i] for i in indices_keep]
        filtered_trains = tuple(self.trains[i].copy() for i in indices_keep)
        # Create a new SpikeTrains object with the filtered data
        return SpikeTrains(
            n_units=len(filtered_units),
            t_start=self.t_start,
            t_stop=self.t_stop,
            units=filtered_units,
            unit_location=filtered_unit_location,
            trains=filtered_trains
        )
    
    def filter_by_units(self, units_include: Union[List[int], np.ndarray],
                        *,
                        sort_as_input: bool=True) -> SpikeTrains:
        """
        Creates a NEW SpikeTrains object containing only the specified units.
        If any of the specified units are not found, raises ValueError.
        
        Parameters
        ----------
        units_include: List[int] or np.ndarray
            List or array of unit indexes to include in the new SpikeTrains object.
        sort_as_input: bool, optional, default=True
            If True, the units in the new SpikeTrains will be in the same order 
            as in `units_include`.

        Returns
        -------
        SpikeTrains
            A new SpikeTrains object containing only the specified units.
            If sort_as_input is True, the units will be in the same order as in `units_include`
            Otherwise, the order will be as in the original SpikeTrains object.
        """
        units_include_arr = np.array(units_include)
        # Find indices of units to keep
        indices_keep = []
        for u in units_include_arr:
            matches = np.where(self.units == u)[0]
            if matches.size == 0:
                raise ValueError(f"Unit {u} not found in SpikeTrains object.")
            indices_keep.append(matches[0])
        if sort_as_input:
            # Keep order as in units_include
            ordered_indices = indices_keep
        else:
            # Keep order as in self.units
            ordered_indices = [i for i in range(self.n_units) if self.units[i] in units_include_arr]
        filtered_units = self.units[ordered_indices].copy()
        filtered_unit_location = [self.unit_location[i] for i in ordered_indices]
        filtered_trains = tuple(self.trains[i].copy() for i in ordered_indices)
        return SpikeTrains(
            n_units=len(filtered_units),
            t_start=self.t_start,
            t_stop=self.t_stop,
            units=filtered_units,
            unit_location=filtered_unit_location,
            trains=filtered_trains
        )

    def generate_shuffled_control(self, *, shuffle_neurons: bool = False,
                                  minimum_shift: float = 0.0):
        """
        Generate a new SpikeTrains object with spike times circularly shifted by a random amount.
        The shift is different for each neuron, sampled i.i.d. from a uniform distribution
        over the duration of the recording.
        
        Parameters
        ----------
        shuffle_neurons: bool, optional
            If True, shuffle the order of neurons in the output object. Default is False.
        minimum_shift: float, optional
            Minimum shift amount for circular shifting. Default is 0.0.

        Returns
        -------
        SpikeTrains
            A new SpikeTrains object with the shuffled or shifted spike times.
        """
        # Use the circular_shift_spiketrain_list function
        shifted_trains = circular_shift_spiketrain_list(
            [np.array(arr) for arr in self.trains],
            t_start=self.t_start,
            t_end=self.t_stop,
            shuffle_neurons=shuffle_neurons,
            minimum_shift=minimum_shift
        )
        # If neurons were shuffled, shuffle units and unit_location accordingly
        if shuffle_neurons:
            n = len(shifted_trains)
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            shifted_trains = [shifted_trains[i] for i in idxs]
            new_units = self.units[idxs].copy()
            new_unit_location = [self.unit_location[i] for i in idxs]
        else:
            new_units = self.units.copy()
            new_unit_location = list(self.unit_location)
        return SpikeTrains.from_spike_list(
            shifted_trains,
            units=new_units,
            unit_location=new_unit_location,
            t_start=self.t_start,
            t_stop=self.t_stop
        )

    def generate_sorted_by_rate(self, *, ascending: bool = False) -> SpikeTrains:
        """
        Generate a new SpikeTrains object with units sorted by firing rate.

        Parameters
        ----------
        ascending : bool, optional (default=False)
            If False (default), sort from highest to lowest firing rate.
            If True, sort from lowest to highest firing rate.

        Returns
        -------
        SpikeTrains
            A new SpikeTrains object with units reordered.
        """
        # Spike count is proportional to rate given constant duration
        n_spikes_all = np.array([len(train) for train in self.trains])
        sorted_indices = np.argsort(n_spikes_all)
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        sorted_units = self.units[sorted_indices].copy()
        sorted_unit_location = [self.unit_location[i] for i in sorted_indices]
        sorted_trains = tuple(self.trains[i].copy() for i in sorted_indices)
        return SpikeTrains(
            n_units=self.n_units,
            t_start=self.t_start,
            t_stop=self.t_stop,
            units=sorted_units,
            unit_location=sorted_unit_location,
            trains=sorted_trains
        )
    
    def counts_and_rate_in_intervals(self, intervals: Union[Tuple[float, float], List[Tuple[float, float]]]) -> pd.DataFrame:
        """
        Generates a dataframe with total number of spikes and rate in specified intervals.
        Intervals are considered all together.

        Parameters
        ----------
        intervals: (start, stop) or List[Tuple[float, float]]
            One interval or a list of time intervals (start, end) for which to compute spike counts and rates.
            They are assumed to be expressed in seconds in increasing order and non-overlapping.

        Returns
        -------
        counts_and_intervals_df = pd.DataFrame
            A dataframe with the following columns:
                + unit : The unit (neuron) ID
                + unit_location : The location of the unit
                + n_spikes : The total number of spikes in the intervals
                + rate : The firing rate (spikes per second) in the intervals,
                        corresponding to the number of spikes divided by the total duration.
        """
        # Normalize input to a list of (start, stop) tuples
        if isinstance(intervals, tuple) and len(intervals) == 2:
            interval_list: List[Tuple[float, float]] = [intervals]  # single interval
        elif isinstance(intervals, list):
            interval_list = intervals
        else:
            raise TypeError("intervals must be a (start, stop) tuple or a list of such tuples")

        if len(interval_list) == 0:
            raise ValueError("At least one interval must be provided")

        # Validate, enforce increasing non-overlapping intervals, and clamp to recording bounds
        clipped: List[Tuple[float, float]] = []
        prev_end = -np.inf
        for start, end in interval_list:
            if not (np.isfinite(start) and np.isfinite(end)):
                raise ValueError("Interval bounds must be finite numbers")
            if end <= start:
                raise ValueError(f"Invalid interval with end <= start: ({start}, {end})")
            if start < prev_end:
                raise ValueError("Intervals must be in increasing order and non-overlapping")
            # Clamp to recording bounds
            s = max(float(start), float(self.t_start))
            e = min(float(end), float(self.t_stop))
            if e > s:
                clipped.append((s, e))
            prev_end = end

        total_duration = float(sum(e - s for s, e in clipped))
        counts = np.zeros(self.n_units, dtype=int)

        if not clipped or (total_duration <= 0.0):
            raise ValueError("No valid intervals found")

        # Count spikes in the union of intervals using [start, end) semantics
        for k, tr in enumerate(self.trains):
            if tr.size == 0:
                continue
            cnt = 0
            for s, e in clipped:
                left = np.searchsorted(tr, s, side='left')
                right = np.searchsorted(tr, e, side='left')  # exclude spikes at e
                cnt += (right - left)
            counts[k] = cnt

        rates = counts.astype(float) / total_duration

        return pd.DataFrame(
            {
                "unit": self.units.copy(),
                "unit_location": list(self.unit_location),
                "n_spikes": counts,
                "rate": rates,
            }
        )

    def __post_init__(self):
        # Optional: validate each array is 1D and sorted in increasing value
        for arr in self.trains:
            if arr.ndim != 1:
                raise ValueError("Each spike array must be 1D")
            if not np.all(np.diff(arr) >= 0):
                raise ValueError("Spike times must be sorted in increasing order")

    # Sequence protocol
    def __getitem__(self, idx):  # supports int or slice
        return self.trains[idx]

    def __len__(self):
        return self.n_units

    def __iter__(self):  # optional; Sequence provides default via __len__/__getitem__
        return iter(self.trains)
    
    # ------------------------------------------------------------------
    # Raster‑plot helpers for SpiekeTrains
    # ------------------------------------------------------------------

    @staticmethod
    def _line_segments_onetrain(
        train: Sequence[float],
        neuron_offset: float,
        time_offset: float,
        height_scaling: float,
    ) -> List[Tuple[float, float]]:
        """Return (t, y) pairs for a single neuron's raster ticks."""
        n_spikes = len(train)
        if n_spikes == 0:
            return []
        # Repeat each spike time twice (top and bottom of vertical	line)
        t_rep = np.repeat(np.asarray(train, dtype=float), 2) - time_offset
        y_offsets = np.tile(np.asarray([-0.5 * height_scaling, 0.5 * height_scaling]), n_spikes)
        ys = neuron_offset + y_offsets
        return list(zip(t_rep.tolist(), ys.tolist()))

    def get_line_segments(
        self,
        *,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        height_scaling: float = 0.7,
        neurons: Optional[Sequence[int]] = None,
        neuron_offset: float = 0.0,
        time_offset: Optional[float] = None,
        max_spikes: int = int(1e7),
    ) -> List[Tuple[float, float]]:
        """Generate (t, y) points for raster line segments.

        Parameters mirror the Julia version, but *t_start* / *t_end* are optional
        and default to the object's bounds.
        
        And the first neuron is at coordinate 0 (but in Julia it is 1).
        """
        _t_start = self.t_start if t_start is None else float(t_start)
        _t_end = self.t_stop if t_stop is None else float(t_stop)
        if not (self.t_start <= _t_start <= _t_end <= self.t_stop):
            raise ValueError(f"Requested window outside recording bounds: {_t_start} to {_t_end} "
                             f"but recording is from {self.t_start} to {self.t_stop}.")

        time_off = _t_start if time_offset is None else float(time_offset)
        if neurons is None:
            neurons = range(1, self.n_units + 1)  # 1‑based indexing like Julia
        else:
            for n in neurons:
                if not 1 <= n <= self.n_units:
                    raise ValueError(f"neuron index {n} out of range 1‑{self.n_units}")

        # Collect trains within window
        selected: List[np.ndarray] = []
        for n in neurons:
            tr = self.trains[n - 1]  # convert to 0‑based
            selected.append(tr[(_t_start <= tr) & (tr <= _t_end)])

        tot_spikes = int(sum(len(tr) for tr in selected))
        if tot_spikes > max_spikes:
            mean_rate = float(np.mean(self.numerical_rates(t_start=_t_start, t_end=_t_end)))
            raise RuntimeError(
                f"There are {len(neurons)} neurons with mean rate {mean_rate:.2f} Hz. "
                f"Total spikes {tot_spikes} exceed max_spikes={max_spikes}."
            )

        points: List[Tuple[float, float]] = []
        for k, tr in enumerate(selected):
            points.extend(
                self._line_segments_onetrain(
                    tr, neuron_offset + k, time_off, height_scaling
                )
            )
        return points

    @staticmethod
    def line_segments_to_xynans(
        points: Sequence[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert point list to x/y arrays with NaN separators for plotting."""
        if not points:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = np.array(points, dtype=float).reshape(-1, 2)
        # Insert NaNs after each pair to break segments
        nan_col = np.full((arr.shape[0] // 2, 2), np.nan)
        composed = np.vstack(
            np.column_stack(
                [arr[0::2], arr[1::2], nan_col]  # (t1,y1)(t1,y1')(nan)
            ).reshape(-1, 2)
        )
        ret = tuple([composed[:, 0], composed[:, 1]])
        if len(ret[0]) != len(ret[1]):
            raise AssertionError("Internal error: x/y length mismatch")
        return ret
    
    def get_line_segments_xynans(
    self,
    *,
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
    height_scaling: float = 0.7,
    neurons: Optional[Sequence[int]] = None,
    neuron_offset: float = 0.0,
    time_offset: Optional[float] = None,
    max_spikes: int = int(1e7),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` NumPy arrays with NaN separators for raster plotting.

        Parameters mirror :meth:`get_line_segments`.  Internally calls that
        method and then :py:meth:`line_segments_to_xynans`.
        """
        points = self.get_line_segments(
            t_start=t_start,
            t_stop=t_stop,
            height_scaling=height_scaling,
            neurons=neurons,
            neuron_offset=neuron_offset,
            time_offset=time_offset,
            max_spikes=max_spikes,
        )
        ret = self.line_segments_to_xynans(points) 
        if len(ret[0]) != len(ret[1]):
            raise AssertionError("Internal error: x/y length mismatch")
        return ret



# Now a class for instantaneous firing rates (iFRs)

@dataclass(frozen=True)
class IFRTrains(Sequence):
    """Immutable container for multiple neurons' iFR values with 
    timestaps coincident with the second spike of the ISI.
    Indexing returns the underlying iFR array for that neuron."""
    n_units: int
    unit_location: list[str]
    t_start: float
    t_stop: float
    units: NDArray
    iFRs: tuple[NDArray[np.float64], ...] 
    timestamps: tuple[NDArray[np.float64], ...] 

    @classmethod
    def from_spiketrains(cls, spiketrains: SpikeTrains) -> "IFRTrains":
        """
        Construct an IFRTrains object from a SpikeTrains object.
        For each neuron, computes instantaneous firing rates (iFR = 1/ISI) and timestamps 
        (second spike of each ISI).
        """
        n_units = spiketrains.n_units
        t_start = spiketrains.t_start
        t_stop = spiketrains.t_stop
        units = spiketrains.units
        iFRs = []
        timestamps = []
        for arr in spiketrains.trains:
            arr = np.array(arr, dtype=np.float64)
            if arr.size < 2:
                # Not enough spikes to compute ISIs, fill with empty arrays
                iFRs.append(np.array([], dtype=np.float64))
                timestamps.append(np.array([], dtype=np.float64))
                continue
            isis = np.diff(arr)
            ifrs = np.where(isis > 0, 1.0 / isis, 0.0)
            # Timestamp is the time of the second spike in each ISI
            timestamps.append(arr[1:])
            iFRs.append(ifrs)
        return cls(n_units=n_units, t_start=t_start, t_stop=t_stop, units=units,
                     unit_location=spiketrains.unit_location,
                   iFRs=tuple(iFRs), timestamps=tuple(timestamps))

    def filter_by_unit_location(self, unit_location_include: str) -> IFRTrains:
        """
        Creates a NEW IFRTrains object containing only units with the specified location.
        If the specified location is not found in the original IFRTrains, raises ValueError.
        
        Parameters
        ----------
        unit_location_include: str
            The location string to filter by (usually "PAG"). Only units with a name that includes this string will be included.
        
        Returns
        -------
        IFRTrains
            A new IFRTrains object containing only the units with the specified location.
        """

        # Get indices of units with the specified location
        indices_keep = [i for i, loc in enumerate(self.unit_location) if unit_location_include in loc]
        if not indices_keep:
            raise ValueError(f"No units found with location '{unit_location_include}'")
        # Filter units, locations, and iFRs, timestamps, make copies of everything!
        filtered_units = self.units[indices_keep].copy()
        filtered_unit_location = [self.unit_location[i] for i in indices_keep]
        filtered_iFRs = tuple(self.iFRs[i].copy() for i in indices_keep)
        filtered_timestamps = tuple(self.timestamps[i].copy() for i in indices_keep)
        # Create a new IFRTrains object with the filtered data
        return IFRTrains(
            n_units=len(filtered_units),
            t_start=self.t_start,
            t_stop=self.t_stop,
            units=filtered_units,
            unit_location=filtered_unit_location,
            iFRs=filtered_iFRs,
            timestamps=filtered_timestamps
        )

    def __post_init__(self):
        # Optional: validate each array is 1D and sorted in increasing value
        for (timest,iFR) in zip(self.timestamps, self.iFRs):
            if timest.ndim != 1 or iFR.ndim != 1:
                raise ValueError("Each timestamp and iFR array must be 1D")
            if len(timest) != len(iFR):
                raise ValueError("Timestamps and iFR arrays must have the same length")
            if not np.all(np.diff(timest) >= 0):
                raise ValueError("Timestamps must be sorted in increasing order")
            if not np.all(iFR >= 0):
                raise ValueError("iFR values must be non-negative")

    def get_timestamps(self, idx: int) -> NDArray[np.float64]:
        """Get timestamps for the specified neuron index."""
        if idx < 0 or idx >= self.n_units:
            raise IndexError("Neuron index out of range")
        return self.timestamps[idx]
    # Sequence protocol
    def __getitem__(self, idx): 
        return self.iFRs[idx]
    def __len__(self):
        return self.n_units
    def __iter__(self): 
        return iter(self.iFRs)


# binning functions based on the classes above

class BinningOperation(ABC):
    name: str  # canonical string identifier used in the public API

    # Generic entry point. Calls the specialised handler depending on obj type.
    # obj must be either SpikeTrains or IFRTrains.
    def apply(self, obj, *, dt:float,t_start:float=None,t_stop:float=None, **kwargs):
        # pick t_start and t_stop from obj if not provided
        if t_start is None:
            t_start = obj.t_start
        if t_stop is None:
            t_stop = obj.t_stop
        method_name = self._method_name_for(obj)
        if not hasattr(self, method_name):
            raise NotImplementedError(
                f"Operation '{self.name}' not supported for {type(obj).__name__}."
            )
        return getattr(self, method_name)(obj, dt=dt, t_start=t_start, t_stop=t_stop, **kwargs)

    @staticmethod
    def _method_name_for(obj) -> str:
        # e.g. SpikeTrains -> 'on_spiketrains'
        return 'on_' + type(obj).__name__.lower()


class Count(BinningOperation):
    def on_spiketrains(self, obj: SpikeTrains, *,
                       dt: float, t_start: float, t_stop: float):
        neuron_idxs = obj.units
        time_bin_edges,time_bin_centers = get_bin_edges_and_centers(t_start, t_stop, dt)
        num_neurons = obj.n_units
        num_bins = len(time_bin_centers)

        # Preallocate arrays
        binned_counts = np.zeros((num_bins, num_neurons), dtype=int)

        for (k, spike_times) in enumerate(obj):
            
            # Assign each spike to its corresponding bin
            bin_indices = np.searchsorted(time_bin_edges, spike_times, side='right') - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid indices

            # Use np.bincount to compute sum of firing rates and counts in each bin
            binned_counts[:,k] = np.bincount(bin_indices, minlength=num_bins)

        return xr.DataArray(
            binned_counts,
            dims=["time_bin_center", "neuron"],
            coords={
                "time_bin_center": time_bin_centers,
                "neuron": neuron_idxs,
            },
            attrs={
                "description": "Binned spike counts",
                "dt": dt,
                "t_start": t_start,
                "t_stop": t_stop,
                "time_bin_edges": time_bin_edges
            })

class Rate(BinningOperation):
    # rate is the same as spike count, followed by a division by dt
    def on_spiketrains(self, obj: SpikeTrains, *, dt: float, t_start: float, t_stop: float):
        counts_da = Count().on_spiketrains(obj, dt=dt, t_start=t_start, t_stop=t_stop)
        rates_da = counts_da / dt
        rates_da.attrs["description"] = "Binned firing rates (not iFR!)"
        return rates_da

class Mean(BinningOperation):
    def on_ifrtrains(self, obj: IFRTrains, *,
                     dt: float, t_start: float, t_stop: float):
        neuron_idxs = obj.units
        time_bin_edges, time_bin_centers = get_bin_edges_and_centers(t_start, t_stop, dt)
        t_stop_edges = time_bin_edges[-1]  # last bin edge
        num_neurons = obj.n_units
        num_bins = len(time_bin_centers)
        # Preallocate arrays
        binned_ifr = np.zeros((num_bins, num_neurons), dtype=float)
        for (k, (timestamps, ifrs)) in enumerate(zip(obj.timestamps, obj.iFRs)):
            # Remove timestamps and iFRs in the incomplete last bin
            idx_less_than_stop = timestamps < t_stop_edges
            timestamps_less_than_stop = timestamps[timestamps < t_stop_edges]
            ifrs_less_than_stop = ifrs[idx_less_than_stop]
            # Assign each iFR to its corresponding bin
            bin_indices = np.searchsorted(time_bin_edges, timestamps_less_than_stop, side='right') - 1
            # no need to clip, as long as I did things right
            if not np.all(bin_indices < num_bins):
                print("timestamps_less_than_stop:", timestamps_less_than_stop)
                print("time_bin_edges:", time_bin_edges)
                print("bin_indices:", bin_indices)
                print("num_bins:", num_bins)
                print("max timestamp:", np.max(timestamps_less_than_stop))
                print("last bin edge:", t_stop_edges)
                raise AssertionError("Oh no, the index values are unexpected!")
            # Use np.bincount to compute sum of iFRs in each bin
            bin_sums = np.bincount(bin_indices, weights=ifrs_less_than_stop, minlength=num_bins)
            bin_counts = np.bincount(bin_indices, minlength=num_bins)
            # Compute mean iFR per bin (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                binned_ifr[:, k] = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
        return xr.DataArray(
            binned_ifr,
            dims=["time_bin_center", "neuron"],
            coords={
                "time_bin_center": time_bin_centers,
                "neuron": neuron_idxs,
            },
            attrs={
                "description": "Binned mean iFR",
                "dt": dt,
                "t_start": t_start,
                "t_stop": t_stop,
                "time_bin_edges": time_bin_edges
            })


# Dictionary mapping operation names to their implementations
# names are defined here, and the actual implementations are in the classes above.
OPERATIONS: dict[str, BinningOperation] = {
    'count': Count(),
    'rate': Rate(),
    'mean': Mean(),
    #'max_pooling': None,  # Placeholder for future implementation
    #'gaussian_convolution': None,  # Placeholder for future implementation
}


def do_binning_operation(obj,operation: str,*,
                dt: float,t_start: float=None,t_stop: float=None,**kwargs):
    """
    Perform a binning operation on the given object (SpikeTrains or IFRTrains).
    
    Parameters
    ----------
    obj: SpikeTrains or IFRTrains
        The object to apply the binning operation on.
    operation: str
        The name of the operation to perform. Must be one of the keys in OPERATIONS.
    dt: float
        The bin size in seconds.
    t_start: float, optional
        The start time of the analysis window. If None, uses the object's t_start.
    t_stop: float, optional
        The end time of the analysis window. If None, uses the object's t_stop.
    **kwargs: dict
        Additional keyword arguments to pass to the operation method.
    
    Returns
    -------
    xr.DataArray
        The result of the binning operation as an xarray DataArray.
        In line with scikit-learn conventions, where dimensons should be n_samples x n_features,
        the DataArray will have the following dimensions:
        - time_bin_center: center of each bin
        - neuron: neuron index

        Attributes of the DataArray will include:
        - description: a string describing the operation performed
        - dt: the size of the bins in seconds
        - t_start: the start time of the analysis window
        - t_stop: the end time of the analysis window
        - time_bin_edges: the edges of the bins used for binning
        
    """

    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")
    # Call the appropriate method based on the type of obj    
    return OPERATIONS[operation].apply(obj, dt=dt, t_start=t_start, t_stop=t_stop, **kwargs)

    

def mean_firing_rate_simple(spike_times_list: List[np.ndarray],
                     firing_rates: List[np.ndarray],
                     stop_time: float,
                     start_time: float = 0,
                     bin_size_ms: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute the binned mean firing rate for each neuron, with spike times recast 
    to the nearest time bin.
    

    Parameters
    ----------
    spike_times_list: list of np.arrays
        List containing spike times for each neuron
    firing_rates: list of np.arrays
        List containing instantaneous firing rates (1/ISI) for each neuron
    stop_time: float
        End of the analysis window
    start_time: float, optional
        Start of the analysis window (default is 0)
    bin_size_ms: int, optional
        Bin size in milliseconds (default is 10ms)  
    
    Returns
    -------
    binned_firing_rates: np.ndarray
        2D array of binned firing rates (neurons x bins)
    mean_rates: np.ndarray
        Array of mean firing rates per neuron (ignoring zero bins)
    nan_firing_rates: np.ndarray
        2D array of firing rates with zeros replaced by NaN
    mean_rates_ifr: np.ndarray
        Array of mean firing rates per neuron (ignoring NaN bins)
    time_bins: np.ndarray
        Array of bin edges
    recast_spike_times: list of np.arrays
        List of arrays, where each array contains spike times mapped to the bin centers
    """
    
    # Convert bin size to seconds
    bin_size = bin_size_ms / 1000  # Convert bin size from ms to seconds
    time_bins = np.arange(start_time, stop_time + bin_size, bin_size)  # Bin edges
    bin_centers = time_bins[:-1] + bin_size / 2  # Bin centers
    num_neurons = len(firing_rates)
    num_bins = len(time_bins)  # Number of bins (edges - 1)

    # Preallocate arrays
    binned_firing_rates = np.zeros((num_neurons, num_bins))
    nan_firing_rates = np.full((num_neurons, num_bins), np.nan)  # Initialize with NaN
    recast_spike_times = []
    
    for n in range(num_neurons):
        spike_times = np.array(spike_times_list[n])
        firing_rate = np.array(firing_rates[n])

        # Select spikes within the time window
        valid_mask = (spike_times >= start_time) & (spike_times < stop_time)
        if not np.any(valid_mask):
            recast_spike_times.append([])
            continue  # Skip neurons with no spikes in the window
        
        spike_times = spike_times[valid_mask]
        firing_rate = firing_rate[valid_mask]
        
        # Assign each spike to its corresponding bin
        bin_indices = np.searchsorted(time_bins, spike_times, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid indices

        recast_spike_times.append(bin_centers[bin_indices])  # Recast spike times to bin centers
        
        # Use np.bincount to compute sum of firing rates in each bin
        bin_sums = np.bincount(bin_indices, weights=firing_rate, minlength=num_bins)
        bin_counts = np.bincount(bin_indices, minlength=num_bins)

        # Compute mean firing rate per bin (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_firing_rates[n, :] = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
        
        # Convert zero values to NaN for mean firing rate calculation
        nan_firing_rates[n, :] = np.where(binned_firing_rates[n, :] == 0, np.nan, binned_firing_rates[n, :])
    
    # Compute mean firing rates, ignoring zero and NaN bins
    mean_rates = np.mean(binned_firing_rates, axis=1)
    mean_rates_ifr = np.nanmean(nan_firing_rates, axis=1)
    
    return binned_firing_rates, mean_rates, nan_firing_rates, mean_rates_ifr, time_bins, recast_spike_times


def bin_spiketrain_counts_and_rates_xarray(
    neuron_idxs: np.ndarray,
    spike_times_list: List[np.ndarray],
    stop_time: float,
    start_time: float = 0.0,
    bin_size_ms: float = 25.0,
) -> xr.DataArray:
    """
    Binned spike train counts and rates into a single xarray DataArray.
    Following the convention of scikit-learn, the dimensions are n_samples x n_features.
    So the structure is: time_bins x neurons x rate_or_count.

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List containing spike times for each neuron.
    stop_time : float
        End of the analysis window.
    start_time : float, optional
        Start of the analysis window (default is 0).
    bin_size_ms : float, optional, default is 25 ms
        Bin size in milliseconds

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing binned spike counts and firing rates.
        Dimensions:
            - time_bin_centers: center of each bin
            - neurons: neuron index
            - rate_or_count: 'rate' or 'count'
        Attributes:
            - bin_size_ms: size of the bins in milliseconds
            - start_time: start time of the analysis window
            - stop_time: end time of the analysis window
            - bin_edges: edges of the bins
    """

    bin_size_s = bin_size_ms / 1000.0  # Convert bin size from ms to seconds
    time_bin_edges,time_bin_centers = get_bin_edges_and_centers(start_time, stop_time, bin_size_s)
    num_neurons = len(spike_times_list)
    num_bins = len(time_bin_centers)

    # Preallocate arrays
    binned_rates_and_counts = np.zeros((num_bins, num_neurons, 2), dtype=float)

    for n in range(num_neurons):
        spike_times = np.array(spike_times_list[n])
        # Select spikes within the time window
        valid_mask = (spike_times >= start_time) & (spike_times < stop_time)
        if not np.any(valid_mask):
            continue  # Skip neurons with no spikes in the window
        spike_times = spike_times[valid_mask]
        
        # Assign each spike to its corresponding bin
        bin_indices = np.searchsorted(time_bin_edges, spike_times, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid indices

        # Use np.bincount to compute sum of firing rates and counts in each bin
        binned_rates_and_counts[:, n, 0] = np.bincount(bin_indices, minlength=num_bins)
        binned_rates_and_counts[:, n, 1] = np.bincount(bin_indices, minlength=num_bins)
        # Convert counts to rates (spikes per second)
        binned_rates_and_counts[:, n, 1] /= bin_size_s  # Convert counts to rates (spikes per second)

    return xr.DataArray(
        binned_rates_and_counts,
        dims=["time_bin_centers", "neurons", "rate_or_count"],
        coords={
            "time_bin_centers": time_bin_centers,
            "neurons": neuron_idxs,
            "rate_or_count": ["count", "rate"]
        },
        attrs={
            "description": "Binned spike counts and firing rates",
            "bin_size_ms": bin_size_ms,
            "start_time": start_time,
            "stop_time": stop_time,
            "time_bin_edges": time_bin_edges
        })
    

def circular_shift_event_times_overflowcounter(timestamps: np.ndarray, shift_amount: float, t_start: float, t_end: float):
    """
    Circularly shift event times by shift_amount, wrapping around duration.
    Returns shifted event times and the overflow counter (number of events wrapped).
    """
    duration = t_end - t_start
    if timestamps.size == 0:
        return np.array([], dtype=timestamps.dtype), 0

    # Normalize to [0, duration)
    shifted = timestamps - t_start
    # apply the shift
    shifted += shift_amount

    # Count how many events overflow (wrap around)
    overflow_mask = shifted > duration
    overflow_counter = np.count_nonzero(overflow_mask)

    # Wrap around
    shifted[overflow_mask] -= duration

    # Convert back to absolute time
    shifted += t_start

    # If overflow occurred, circularly shift the array to maintain order
    if overflow_counter > 0:
        shifted = np.roll(shifted, overflow_counter)

    return shifted
    
    
def circular_shift_spiketrain_list(
    spike_times_list: List[np.ndarray],
    t_start: float,
    t_end: float,*,
    shuffle_neurons: bool = False,
    minimum_shift:float = 0.0
) -> List[np.ndarray]:
    """
    This function generates a new list of spike trains by circularly shifting the spike times by
    a random amount sampled i.i.d. from a uniform distribution between 0.0 and t_end - t_start.

    Arguments
    ---------
    spike_times_list: List[np.ndarray]
        List of spike times for each neuron.
    t_start: float
        Start time of the analysis window.
    t_end: float
        End time of the analysis window.
    shuffle_neurons: bool, optional
        If True, shuffle the order of neurons in the output list. Default is False.
    minimum_shift: float, optional
        Minimum shift amount for circular shifting. Default is 0.0.

    Returns
    -------
    List[np.ndarray]
        New list of spike trains with circularly shifted spike times. And optionally shuffled neurons.
    """
    
    duration = t_end - t_start
    n_neurons = len(spike_times_list)

    # check minimum shift
    if minimum_shift < 0:
        raise ValueError("minimum_shift must be non-negative")
    if minimum_shift >= duration:
        raise ValueError("minimum_shift must be less than the duration!")
    
    # Generate random shifts for each neuron
    random_shifts = np.random.uniform(minimum_shift, duration-minimum_shift, n_neurons)
    shifted_spike_trains = []
    for i, (spike_times, shift_amt) in enumerate(zip(spike_times_list, random_shifts)):
        shifted = circular_shift_event_times_overflowcounter(np.array(spike_times), shift_amt, t_start, t_end)
        shifted_spike_trains.append(shifted)
    if shuffle_neurons:
        idxs = np.arange(n_neurons)
        np.random.shuffle(idxs)
        shifted_spike_trains = [shifted_spike_trains[i] for i in idxs]
    return shifted_spike_trains



def get_start_stop_idx(labels_array: np.ndarray, label: str) -> Tuple[int, int]:
    """
    Given a 1D numpy array of integer labels, returns a list of start and stop indices for the specified label.
    So that we can easily access the trials corresponding to a specific behavior.
    
    For exaple 

    Arguments
    ---------
    
    labels_array : np.ndarray
        1D numpy array of integer labels, where each label corresponds to a behavior.
    label : int
        The label for which to find the start and stop indices.

    Returns
    -------

    starts_stops : List[Tuple[int, int]]
        A list of tuples, each containing the start and stop indices for the specified label.
        The length of the list corresponds to the number of trials for that specific label.
    
    Example
    --------
    Given a labels_array of [0, 1, 0, 1, 1] and label 1, the function would return [(1, 2), (3, 5)].

    """
    # Check that labels_array is a 1D integer array
    if not isinstance(labels_array, np.ndarray):
        raise TypeError("labels_array must be a numpy ndarray")
    if labels_array.ndim != 1:
        raise ValueError("labels_array must be a 1D array")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise TypeError("labels_array must be of integer dtype")

    # Check that label is present
    if label not in labels_array:
        raise ValueError(f"Label {label} not found in labels_array")

    starts_stops = []
    in_label = False
    start_idx = None
    for idx, val in enumerate(labels_array):
        if val == label:
            if not in_label:
                start_idx = idx
                in_label = True
        else:
            if in_label:
                stops_idx = idx
                starts_stops.append((start_idx, stops_idx))
                in_label = False
    # Handle case where label runs to the end
    if in_label:
        starts_stops.append((start_idx, len(labels_array)))
    return starts_stops

