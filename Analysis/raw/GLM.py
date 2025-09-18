# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:45:51 2025

@author: su-weisss
"""
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import List, Dict

def calculate_psth(spike_times_list: List[np.ndarray], bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculates the Peri-Stimulus Time Histogram (PSTH).

    Args:
        spike_times_list: A list of numpy arrays, where each array contains
                          spike times for a single trial.
        bin_edges: A numpy array defining the edges of the time bins.

    Returns:
        A numpy array representing the PSTH (firing rate in Hz).
    """
    if not spike_times_list:
        return np.zeros(len(bin_edges) - 1)

    total_spikes = np.zeros(len(bin_edges) - 1)
    num_trials = len(spike_times_list)
    bin_widths = np.diff(bin_edges)
   
    for trial_spikes in spike_times_list:
        for i in trial_spikes:            
                hist, _ = np.histogram(i, bins=bin_edges)
            
        total_spikes += hist

    psth = total_spikes / (num_trials * bin_widths)
    return psth

def classify_neuron_selectivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies the selectivity of each neuron to behavior using a GLM on PSTH data.

    Args:
        df: A pandas DataFrame with columns: "cluster_id", "max_site", "region",
            and a column for each behavior containing a list of spike time arrays.

    Returns:
        A pandas DataFrame with the selectivity classification for each neuron.
    """
    results = []
    behavior_columns = [col for col in df.columns if col not in ["cluster_id", "max_site", "region"]]
    time_window = (-5.0, 5.0)  # Example time window around the event (in seconds)
    bin_width = 0.1  # Example bin width (in seconds)
    bin_edges = np.arange(time_window[0], time_window[1] + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for index, row in df.iterrows():
        cluster_id = row["cluster_id"]
        psth_data = {}
        for behavior in behavior_columns:
            spike_times = row[behavior]
            psth = calculate_psth(spike_times, bin_edges)
            psth_data[behavior] = psth

        # Prepare data for GLM
        glm_df_list = []
        for behavior, psth in psth_data.items():
            for i, firing_rate in enumerate(psth):
                glm_df_list.append({
                    "time": bin_centers[i],
                    "firing_rate": firing_rate,
                    "behavior": behavior
                })
        glm_df = pd.DataFrame(glm_df_list)

        if not glm_df.empty:
            try:
                # Fit a GLM (e.g., Gaussian family)
                model = smf.glm(formula="firing_rate ~ C(behavior) + time + C(behavior):time",
                                data=glm_df,
                                family=sm.families.Gaussian()).fit()

                # Analyze coefficients to determine selectivity
                coefficients = model.params
                p_values = model.pvalues

                # Identify the behavior with the highest significant coefficient (excluding intercept and time)
                selective_behavior = None
                max_coeff = -np.inf
                for behavior in behavior_columns:
                    behavior_coeff_name = f"C(behavior)[T.{behavior}]"
                    if behavior_coeff_name in coefficients and p_values.get(behavior_coeff_name, 1.0) < 0.05:
                        if coefficients[behavior_coeff_name] > max_coeff:
                            max_coeff = coefficients[behavior_coeff_name]
                            selective_behavior = behavior

                results.append({
                    "cluster_id": cluster_id,
                    "selectivity": selective_behavior,
                    "glm_summary": model.summary()
                })

            except Exception as e:
                results.append({
                    "cluster_id": cluster_id,
                    "selectivity": None,
                    "glm_summary": f"GLM fitting failed: {e}"
                })
        else:
            results.append({
                "cluster_id": cluster_id,
                "selectivity": None,
                "glm_summary": "No PSTH data to fit GLM."
            })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Create a sample DataFrame for demonstration
    df= pd.read_pickle('concatanated_PSTHs.pkl')

    # Classify neuron selectivity
    selectivity_results_df = classify_neuron_selectivity(df.copy())
    print(selectivity_results_df)

    # Example of how to access GLM summary for a specific neuron
    if not selectivity_results_df.empty:
        print("\nGLM Summary for cluster_id 1:\n", selectivity_results_df[selectivity_results_df['cluster_id'] == 1]['glm_summary'].iloc[0])