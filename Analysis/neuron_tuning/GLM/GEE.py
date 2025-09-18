# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:36:44 2025

@author: su-weisss
"""

"""
Created on Tue Jun 17 09:29:41 2025

@author: su-weisss
"""
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.tools import add_constant

import matplotlib.pyplot as plt
from scipy.stats import ranksums
from sklearn.model_selection import KFold
# 1. Load data
print('loading test data')
firing_rates=np.load(f"//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/Figures/20250509/afm16924/concat_PSTHs/firing_rates_240522.npy")
behaviour = pd.read_pickle(f"//gpfs.corp.brain.mpg.de/stem/data/project_hierarchy/Figures/20250509/afm16924/concat_PSTHs/behaviors_240522.pkl")

#Step 1: Data Preprocessing
print('Step 1: Data Preprocessing')
vframerate=50
dt = 1.0 / vframerate
i_neuron=195#206#199
firing_rate=firing_rates[i_neuron]
num_bins = np.shape(firing_rates)[1]
time_ax = np.linspace(0, num_bins*dt, num_bins)

# Get unique behaviors
unique_behaviours = behaviour['behaviours'].unique()


# 2. Preprocess: create binary predictors for each behavior
unique_behaviours = behaviour['behaviours'].unique()
binary_behaviors = {}
for beh in unique_behaviours:
    binary_signal = np.zeros(len(time_ax), dtype=int)
    subdf = behaviour[behaviour["behaviours"] == beh].sort_values("frames_s").reset_index(drop=True)
    i = 0
    while i < len(subdf):
        row = subdf.iloc[i]
        event_type = row["start_stop"]
        event_time = row["frames_s"]
        if event_type == "POINT":
            start_time = event_time
            end_time = event_time + 0.001
            idx = np.where((time_ax >= start_time) & (time_ax < end_time))[0]
            binary_signal[idx] = 1
            i += 1
        elif event_type == "START":
            start_time = event_time
            if i + 1 < len(subdf) and subdf.iloc[i + 1]["start_stop"] == "STOP":
                end_time = subdf.iloc[i + 1]["frames_s"]
            else:
                end_time = start_time + dt
            idx = np.where((time_ax >= start_time) & (time_ax <= end_time))[0]
            binary_signal[idx] = 1
            i += 2
        else:
            i += 1
    binary_behaviors[beh] = binary_signal

# Build DataFrame
data = {
    'firing_rate': firing_rate,
    'time': time_ax,
}
for b in unique_behaviours:
    data[b] = binary_behaviors[b]
data['trial'] = np.random.randint(1, 6, len(time_ax))  # Random trial IDs for grouping

df = pd.DataFrame(data)

# Standardize time
scaler = StandardScaler()
df['time_scaled'] = scaler.fit_transform(df[['time']])

# Remove constant predictors
event_cols = [b for b in unique_behaviours if df[b].nunique() > 1]
predictors = ['time_scaled'] + event_cols
print(f"{predictors=}")

print(f"GEE Model selection: trying all non-empty combinations of predictors")
# 3. Model selection: try all non-empty combinations of predictors
results = []
for L in range(1, len(predictors)+1):
    for subset in itertools.combinations(predictors, L):
        X = df[list(subset)]
        X = add_constant(X)
        y = df['firing_rate']
        groups = df['trial']
        try:
            model = GEE(y, X, groups=groups, family=Poisson())
            result = model.fit()
            llf = result.llf
            # Variance explained (pseudo R^2)
            null_model = GEE(y, add_constant(np.ones(len(y))), groups=groups, family=Poisson()).fit()
            r2 = 1 - result.deviance / null_model.deviance
            results.append({
                'predictors': subset,
                'log_likelihood': llf,
                'variance_explained': r2,
                'aic': result.aic
            })
        except Exception as e:
            print(f"Model failed for predictors {subset}: {e}")

# 4. Find the best model(s)
print('finding best model')
results_df = pd.DataFrame(results)

# Sort by variance explained (descending), then by number of predictors (ascending), then by AIC (ascending)
results_df['num_predictors'] = results_df['predictors'].apply(len)
results_df = results_df.sort_values(['variance_explained', 'num_predictors', 'aic'], ascending=[False, True, True])

best = results_df.iloc[0]

print("\nBest minimal model:")
print("Predictors:", best['predictors'])
print("Number of predictors:", best['num_predictors'])
print("Log Likelihood:", best['log_likelihood'])
print("Variance Explained (pseudo R^2):", best['variance_explained'])
print("AIC:", best['aic'])

# Optionally, print the top 10 models
print("\nTop 10 models sorted by variance explained, number of predictors, and AIC:")
print(results_df.head(10))

# Cross-validation parameters
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)
print(f"cross validation {K=}")
cv_results = []

for model_info in results:
    predictors = list(model_info['predictors'])
    X = df[predictors]
    X = add_constant(X)
    y = df['firing_rate']
    groups = df['trial']
    log_likelihoods = []
    variances = []
    pvals = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        try:
            model = GEE(y_train, X_train, groups=groups_train, family=Poisson())
            result = model.fit()
            y_pred = result.predict(X_test)
            # Log likelihood on test set
            # For Poisson: sum(y_test * log(y_pred) - y_pred - log(y_test!))
            # Use np.log(y_pred) safely
            y_pred = np.clip(y_pred, 1e-8, None)
            ll = np.sum(y_test * np.log(y_pred) - y_pred)
            log_likelihoods.append(ll)
            # Pseudo R^2 on test set
            null_pred = np.full_like(y_test, y_train.mean())
            deviance = 2 * np.sum(y_test * (np.log((y_test + 1e-8) / y_pred)) - (y_test - y_pred))
            null_deviance = 2 * np.sum(y_test * (np.log((y_test + 1e-8) / null_pred)) - (y_test - null_pred))
            r2 = 1 - deviance / null_deviance if null_deviance != 0 else np.nan
            variances.append(r2)
            # Ranksum test
            stat, pval = ranksums(y_test, y_pred)
            pvals.append(pval)
        except Exception as e:
            log_likelihoods.append(np.nan)
            variances.append(np.nan)
            pvals.append(np.nan)
    cv_results.append({
        'predictors': tuple(predictors),
        'mean_log_likelihood': np.nanmean(log_likelihoods),
        'mean_variance_explained': np.nanmean(variances),
        'mean_ranksum_pval': np.nanmean(pvals)
    })

cv_df = pd.DataFrame(cv_results)
cv_df['num_predictors'] = cv_df['predictors'].apply(len)
cv_df = cv_df.sort_values(['mean_variance_explained', 'num_predictors', 'mean_log_likelihood'], ascending=[False, True, False])

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Ranksum p-value
axs[0].plot(range(len(cv_df)), cv_df['mean_ranksum_pval'], marker='o')
axs[0].set_ylabel('Mean Ranksum p-value')
axs[0].set_title('Ranksum Test (Actual vs Predicted)')

# Log likelihood
axs[1].plot(range(len(cv_df)), cv_df['mean_log_likelihood'], marker='o', color='orange')
axs[1].set_ylabel('Mean Log Likelihood')
axs[1].set_title('Log Likelihood (CV)')

# Variance explained
axs[2].plot(range(len(cv_df)), cv_df['mean_variance_explained'], marker='o', color='green')
axs[2].set_ylabel('Mean Variance Explained (pseudo R²)')
axs[2].set_xlabel('Model (sorted)')
axs[2].set_title('Variance Explained (CV)')

plt.tight_layout()
plt.show()

# Print the best model from cross-validation
best_cv = cv_df.iloc[0]
print("\nBest minimal model (cross-validated):")
print("Predictors:", best_cv['predictors'])
print("Number of predictors:", best_cv['num_predictors'])
print("Mean Log Likelihood:", best_cv['mean_log_likelihood'])
print("Mean Variance Explained (pseudo R^2):", best_cv['mean_variance_explained'])
print("Mean Ranksum p-value:", best_cv['mean_ranksum_pval'])