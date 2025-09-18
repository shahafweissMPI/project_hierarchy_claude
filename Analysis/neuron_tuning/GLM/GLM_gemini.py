# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:15:34 2025

@author: su-weisss
"""

# main_script.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from glm_class import GLM
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the GLM analysis on experimental neural data.
    """
    # 1. Load and preprocess experimental data
    # This section is adapted from your GEE.py script to load your specific
    # dataset and prepare it for the GLM analysis.

    print('Step 1: Loading...  ')
    
    # --- Data Loading ---
    # Note: Please ensure these file paths are correct for your system.
    # The script will now directly load your data and will raise an error
    # if the files are not found. The placeholder simulation has been removed.
    try:
        #alt_b=r"D:\GitHub\NewGit\Project_hierarchy_stem\Analysis\neuron_tuning\GLM\behaviour__240522_protocol4.pkl"
#        alt_FR=r"D:\GitHub\NewGit\Project_hierarchy_stem\Analysis\neuron_tuning\GLM\firingrate__240522_protocol4.pkl"
        firing_rates = np.load(f"//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/Figures/20250509/afm16924/concat_PSTHs/firing_rates_240522.npy")
        behaviour_df = pd.read_pickle(f"//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/Figures/20250509/afm16924/concat_PSTHs/behaviors_240522.pkl")
        print('Data loaded successfully.')
    except FileNotFoundError as e:
        print(f"Error: Data file not found at the specified path.")
        print(f"Details: {e}")
        print("Please ensure the file paths are correct and accessible.")
        return # Exit the script if data cannot be loaded

    # --- Data Preprocessing ---
    print('Preprocessing Data...')
    vframerate = 50
    dt = 1.0 / vframerate
    i_neuron = 195  # The specific neuron to analyze
    batch_size=2**8#int(65536) #4096#1024#65536#32768
    # Select the firing rate for the chosen neuron
    firing_rate = firing_rates[i_neuron]
    num_bins = len(firing_rate)
    time_ax = np.linspace(0, num_bins * dt, num_bins)

    # Convert firing rate (Hz) to spike counts for the Poisson model
    # The GLM works with spike counts, which are Poisson distributed.
    spike_train = firing_rate * dt

    # Create binary predictors for each unique behavior from the 'behaviours' column
    unique_behaviours = behaviour_df['behaviours'].unique()
    binary_behaviors = {}
    print(f"Creating binary predictors for behaviors: {unique_behaviours}")

    for beh in unique_behaviours:
        binary_signal = np.zeros(len(time_ax), dtype=int)
        # Filter the dataframe for the current behavior
        subdf = behaviour_df[behaviour_df["behaviours"] == beh].sort_values("frames_s").reset_index(drop=True)
        
        i = 0
        while i < len(subdf):
            row = subdf.iloc[i]
            event_type = row["start_stop"]
            event_time = row["frames_s"]
            
            start_time = event_time
            if event_type == "POINT":
                # For point events, mark a very short duration
                end_time = event_time + 0.001 
                i += 1
            elif event_type == "START":
                # For interval events, find the corresponding STOP
                if i + 1 < len(subdf) and subdf.iloc[i + 1]["start_stop"] == "STOP":
                    end_time = subdf.iloc[i + 1]["frames_s"]
                    i += 2 # Move past the START and STOP pair
                else:
                    # If no STOP is found, treat as a single time-bin event
                    end_time = start_time + dt
                    i += 1
            else: # Unpaired STOP or other event type
                i += 1
                continue

            # Set the binary signal to 1 for the duration of the event
            idx = np.where((time_ax >= start_time) & (time_ax <= end_time))[0]
            binary_signal[idx] = 1
        
        binary_behaviors[beh] = binary_signal

    # --- Build the final design matrix (X) ---
    # Create a DataFrame to hold all variables
    data = {'time': time_ax}
    for b in unique_behaviours:
        data[b] = binary_behaviors[b]
    
    df = pd.DataFrame(data)

    # Standardize the time predictor
    scaler = StandardScaler()
    df['time_scaled'] = scaler.fit_transform(df[['time']])

    # Remove predictors that are constant (all zeros or all ones)
    event_cols = [b for b in unique_behaviours if df[b].nunique() > 1]
    
    # Final list of predictors for the model, taken from your data
    predictor_names = ['time_scaled'] + event_cols
    print(f"\nFinal predictors for model selection: {predictor_names}")
    
    # Create the predictor matrix X
    X = df[predictor_names].values

    # 2. Perform recursive feature elimination
    print("\nStarting feature elimination...")   
    best_subset, best_performance = GLM.combined_feature_selection(X, spike_train, predictor_names, dt=dt, batch_size=batch_size, n_jobs=4)
    
    #print("\nStarting recursive feature elimination...")
#    best_subset, best_performance = GLM.recursive_feature_elimination(
#        X, spike_train, predictor_names, dt=dt, batch_size=batch_size)
    print(f"\nRecursive feature elimination complete.")
    print(f"Best subset of predictors found: {best_subset}")
    print(f"Best model performance (cross-validated NLL): {best_performance:.4f}")
    
    # 3. Fit the final GLM with the best subset of predictors
    best_predictor_indices = [predictor_names.index(p) for p in best_subset]
    final_X = X[:, best_predictor_indices]

    final_glm = GLM(final_X, spike_train, dt=dt)
    
    print("\nFitting final GLM with the best subset...")
    final_glm.fit()
    print("Final GLM fitting complete.")

    # 4. Evaluate and visualize the results
    nll = final_glm.neg_log_likelihood()
    print(f"\nNegative Log-Likelihood of the final model: {nll:.4f}")

    print("\nLearned Weights for the Best Subset of Predictors:")
    for name, weight in zip(best_subset, final_glm.w.numpy().flatten()):
        print(f"{name}: {weight:.4f}")

    predicted_rate = final_glm.predict_rate()

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot 1: Actual vs. Predicted Firing Rate
    axes[0].plot(time_ax, firing_rate, label="Actual Firing Rate", color='black', alpha=0.7)
    axes[0].plot(time_ax, predicted_rate, label="Predicted Firing Rate (GLM)", color='red', linestyle='--')
    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[0].legend()
    axes[0].set_title(f"GLM Performance for Neuron {i_neuron}")

    # Plot 2: Spike Train
    axes[1].stem(time_ax, spike_train, use_line_collection=True, linefmt='grey', markerfmt='|', basefmt=" ")
    axes[1].set_ylabel("Spike Counts (per bin)")
    axes[1].set_ylim(bottom=0)

    # Plot 3: Selected Predictor Variables
    if final_X.shape[1] > 0:
        for i, predictor_name in enumerate(best_subset):
            idx = predictor_names.index(predictor_name)
            # Add offset for better visualization
            axes[2].plot(time_ax, X[:, idx] + i * 1.5, label=predictor_name)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Predictors (offset)")
    axes[2].set_yticks([])
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
