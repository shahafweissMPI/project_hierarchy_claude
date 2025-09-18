# main_script_nemos.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nemos as nmo

def main():
    """
    Main function to run the GLM analysis on experimental neural data using the nemos package.
    The script performs model selection using L1 regularization to identify the minimum
    set of behavioral predictors for a neuron's firing rate.
    """
    # 1. Load and preprocess experimental data
    # This section is adapted from your original script to load your specific
    # dataset and prepare it for the GLM analysis.

    print('Step 1: Loading and Preprocessing Data...')

    # --- Data Loading ---
    # Note: Please ensure these file paths are correct for your system.
    try:
        # These paths point to your data files. The script will fail if they are not found.
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

    # Select the firing rate for the chosen neuron. Ensure i_neuron is a valid index.
    if i_neuron >= firing_rates.shape[0]:
        print(f"Warning: i_neuron index {i_neuron} is out of bounds. Using neuron 0 instead.")
        i_neuron = 0
    firing_rate = firing_rates[i_neuron]
    num_bins = len(firing_rate)
    time_ax = np.linspace(0, num_bins * dt, num_bins)

    # Convert firing rate (Hz) to spike counts for the Poisson model.
    # The nemos GLM works with spike counts, which are modeled as a Poisson process.
    spike_train = (firing_rate * dt).astype(np.int32)

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
                # For point events, mark a very short duration (one time bin)
                end_time = event_time + dt
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
            idx = np.where((time_ax >= start_time) & (time_ax < end_time))[0]
            binary_signal[idx] = 1
        
        binary_behaviors[beh] = binary_signal

    # --- Build the final design matrix (X) and response vector (y) ---
    # Create a DataFrame to hold all variables, then extract for nemos.
    predictor_df = pd.DataFrame(binary_behaviors)
    
    # Design matrix X: contains all predictors (behaviors)
    X = predictor_df.values
    
    # Response vector y: contains the spike counts
    # nemos expects y to be of shape (n_time_bins, 1)
    y = spike_train[:, None]

    print(f"\nStep 2: Building and Fitting GLM with nemos...")
    print(f"Design matrix X shape: {X.shape}")
    print(f"Response vector y shape: {y.shape}")

    # --- Model Selection with L1 Regularization (Lasso) ---
    # The `strength` parameter controls the penalty. Higher values lead
    # to a sparser model. This is a hyperparameter you may want to tune.
    l1_strength = 0.1
    
    print(f"\nUsing Lasso Regularization with strength = {l1_strength}")

    # Instantiate the GLM
    # We use a Poisson observation model, suitable for spike counts.
    # The Lasso regularizer (L1) is used for feature selection.
    glm = nmo.glm.GLM(
        regularizer=nmo.regularizer.Lasso(strength=l1_strength),
        observation_model=nmo.observation_models.Poisson()
    )

    # Fit the model to the data
    # nemos uses jax for automatic differentiation and GPU acceleration.
    # The fit function finds the model parameters that maximize the penalized log-likelihood.
    coefficients, intercept = glm.fit(X, y)
    
    # Convert coefficients to a numpy array for easier handling
    coefficients = np.asarray(coefficients).flatten()
    intercept = np.asarray(intercept).flatten()
    
    print("\nStep 3: Analyzing Results")

    # --- Evaluate Model Fit ---
    # Predict the firing rate based on the fitted model
    predicted_rate = glm.predict(X)
    
    # Calculate pseudo-R^2 to assess goodness-of-fit
    # This metric compares the likelihood of the fitted model to that of a baseline
    # model with only an intercept term.
    score = glm.score(X, y, score_type="pseudo-r2-McFadden")
    print(f"Model Pseudo-R^2 (McFadden): {score:.4f}")

    # --- Identify Selected Predictors ---
    # Find the predictors with non-zero coefficients. These are the behaviors
    # selected by the L1-regularized model as being important.
    selected_predictors_mask = ~np.isclose(coefficients, 0, atol=1e-5)
    selected_behaviors = predictor_df.columns[selected_predictors_mask].tolist()
    selected_coeffs = coefficients[selected_predictors_mask]

    print("\n--- Model Results ---")
    print(f"Intercept (baseline log-firing rate): {intercept[0]:.4f}")
    if selected_behaviors:
        print("Selected behavioral predictors and their weights (coefficients):")
        for beh, coeff in zip(selected_behaviors, selected_coeffs):
            print(f"  - {beh}: {coeff:.4f}")
    else:
        print("No predictors were selected. The model only contains an intercept.")
        print("Consider reducing the L1 regularization strength (`l1_strength`).")


    # --- Step 4: Visualization ---
    print("\nStep 4: Generating Plots...")
    
    # Plot 1: Learned Coefficients
    plt.figure(figsize=(12, 6))
    plt.stem(predictor_df.columns, coefficients, use_line_collection=True)
    plt.title('Learned GLM Coefficients (Lasso Regularized)')
    plt.ylabel('Coefficient (Weight)')
    plt.xlabel('Behavioral Predictor')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='grey', linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2: Actual vs. Predicted Firing Rate
    # We will plot a small segment of the data for clarity.
    plt.figure(figsize=(15, 7))
    plot_duration_sec = 100  # Duration to plot in seconds
    plot_end_idx = int(plot_duration_sec / dt)
    
    # Rescale predicted rate from spikes/bin back to Hz for comparison
    predicted_rate_hz = np.asarray(predicted_rate).flatten() / dt
    
    plt.plot(time_ax[:plot_end_idx], firing_rate[:plot_end_idx], color='black', alpha=0.6, label='Actual Firing Rate')
    plt.plot(time_ax[:plot_end_idx], predicted_rate_hz[:plot_end_idx], color='red', linestyle='-', label='Predicted Firing Rate (GLM)')
    plt.title(f'Actual vs. Predicted Firing Rate (Neuron {i_neuron})')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

