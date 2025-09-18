"""
Created by Tom Kern
Last modified on 04/08/24

Computes responsiveness of each neuron to each behaviour, based on shahafs number (s_num) 
-firing change: For each neuron and each behaviour, looks in what proportion of trials of the 
    behaviour the neurons changes its firing above/below threshold as == P+/P-

-threshold: 
    -Takes n random timepoints in baseline period
    -calculates change in firing before and after timepoint
    -std of these changes is threshold of firing

-s_num: P+ - P-, is calculated per neuron and behaviour

-Permutation: recalculates s_num x times, each time with newly shuffled neural data

-significance: if the actual s_num is not really found when you shuffle the neural 
    data, (i.e. in a very low or very high percentile), then we think it is 
    likely that the s_num is not just due to random fluctuations
    
-reference period: 
    -escape behaviours: Before the loom (unless there are hunting behaviours 
                                         before the loom, then before hunting)
    - hunting behaviours: before hunting behaviour, no matter what came before
    - switches/ pullbacks: before hunting behaviour
    
- Results are saved, and can be summarised across sessions with 
    total_tuning_per_area.py and venn3


Important
-When wanting to compare output from different sessions later, it is important 
    that target_bs stays the same across different sessions (at least in the current
                                                            version of the code)

"""

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import helperFunctions as hf
from numpy.random import choice
from time import time
from joblib import Parallel, delayed

# Set parameters
sessions = ['240522', '240524']
n_perms = 1000  # Number of permutations (should be at least 1000 times for stable results, more is better)
target_bs = np.array(['approach', 'pursuit', 'attack', 'switch', 'startle', 'freeze', 'escape', 'pullback', 'eat', 'baseline'])
hunting_bs = ['approach', 'pursuit', 'attack', 'eat']  # Which behaviours are considered hunting? Important for resolution
resolution = 0.01  # In case you want to run things quick, otherwise put to None

def compute_threshold(baseline_data, n_timepoints=100):
    """
    Compute the threshold for firing changes based on baseline data.
    """
    changes = []
    for _ in range(n_timepoints):
        timepoint = choice(baseline_data)
        change = np.std(baseline_data - timepoint)
        changes.append(change)
    return np.std(changes)

def compute_s_num(neuron_data, behaviour_data, threshold):
    """
    Compute the shahafs number (s_num) for a given neuron and behaviour.
    """
    p_plus = np.mean(neuron_data > threshold)
    p_minus = np.mean(neuron_data < -threshold)
    return p_plus - p_minus

def permute_data(neuron_data, behaviour_data, n_perms):
    """
    Perform permutations on the neural data to compute the distribution of s_num.
    """
    permuted_s_nums = []
    for _ in range(n_perms):
        shuffled_data = np.random.permutation(neuron_data)
        threshold = compute_threshold(shuffled_data)
        s_num = compute_s_num(shuffled_data, behaviour_data, threshold)
        permuted_s_nums.append(s_num)
    return permuted_s_nums

def compute_significance(actual_s_num, permuted_s_nums):
    """
    Compute the significance of the actual s_num based on the permuted s_nums.
    """
    lower_percentile = np.percentile(permuted_s_nums, 2.5)
    upper_percentile = np.percentile(permuted_s_nums, 97.5)
    return actual_s_num < lower_percentile or actual_s_num > upper_percentile

def process_session(session):
    """
    Process a single session to compute the responsiveness of each neuron to each behaviour.
    """
    # Load session data (this is a placeholder, replace with actual data loading)
    neuron_data = np.random.rand(100, 1000)  # Placeholder for neuron data
    behaviour_data = np.random.choice(target_bs, size=1000)  # Placeholder for behaviour data

    results = []
    for neuron in neuron_data:
        for behaviour in target_bs:
            behaviour_indices = np.where(behaviour_data == behaviour)[0]
            baseline_data = neuron[behaviour_indices]
            threshold = compute_threshold(baseline_data)
            actual_s_num = compute_s_num(neuron, behaviour_data, threshold)
            permuted_s_nums = permute_data(neuron, behaviour_data, n_perms)
            significance = compute_significance(actual_s_num, permuted_s_nums)
            results.append((neuron, behaviour, actual_s_num, significance))
    return results



def main():
    start_time = time()
    results = Parallel(n_jobs=-1)(delayed(process_session)(session) for session in sessions)
    end_time = time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    savedict={'region_change': change,
              'regions': n_region_index,
              'target_bs': target_bs}
    np.save(rf"{savepath}\s_num_perm_{session}.npy",
            savedict)

 #%% Save results
 
    
    
    

if __name__ == "__main__":
    main()