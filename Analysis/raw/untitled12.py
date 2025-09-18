# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:45:07 2024

@author: su-weisss
"""

# Modified plot_neurons function to handle multiple behaviors
def plot_neurons(neurons_chunk, chunk_num):
    rows, cols, gs, fig = pf.subplots(len(neurons_chunk), gridspec=True)
    fig.set_size_inches(19.2, 12)
    gs.update(hspace=0.5)

    unique_behaviours = behaviour.behaviours.unique()
    n_behaviors = len(unique_behaviours)

    ys = []
    Hz_axs = []
    for i, n in enumerate(neurons_chunk):
        # divide subplot into (2 + n_behaviors) rows: velocity, avg Hz for each behavior
        gs_sub = gridspec.GridSpecFromSubplotSpec(2 + n_behaviors, 1, subplot_spec=gs[i])
        axs = [plt.subplot(gs_sub[j]) for j in range(2 + n_behaviors)]

        # Plot velocity in first subplot
        axs[0].set_title(f"#{neurons_chunk[i]} {n_region_index[n]} {base_mean[i]}Hz")

        current_Hz_axs = []
        for b_idx, b_name in enumerate(unique_behaviours):
            # get behaviour frames
            start_stop = hf.start_stop_array(behaviour, b_name)
            time_window = 10 if b_name == 'eat' else 5
            
            # Make PSTH plot for each behavior
            # Use different subplots for each behavior
            behavior_axs = [axs[0], axs[b_idx + 1], axs[-1]]
            pf.psth(ndata[n], 
                    n_time_index, 
                    start_stop, 
                    velocity,
                    frame_index_s,
                    behavior_axs, 
                    window=time_window, 
                    density_bins=.5)
            
            # Label each behavior
            axs[b_idx + 1].set_ylabel(f'{b_name}\nfiring [Hz]')
            current_Hz_axs.append(axs[b_idx + 1])

        # Set labels
        if i == 0:
            axs[0].set_ylabel('Velocity [cm/s]')
            axs[-1].set_ylabel('trials')
            axs[-1].set_xlabel('time [s]')

        ys.extend([ax.get_ylim()[1] for ax in current_Hz_axs])
        Hz_axs.extend(current_Hz_axs)

    # set ylim for all firing rate subplots
    max_y = max(ys)
    [ax.set_ylim((0, 10)) for ax in Hz_axs]
   
    if not Path.is_dir(save_path):
        Path.mkdir(save_path, exist_ok=True)
    
    plt.savefig(f'{save_path}/{animal}_{session}_all_behaviors_{chunk_num}.png')
    plt.close(fig)

# Main execution
chunk_size = 25
for i in range(0, len(target_n), chunk_size):
    plot_neurons(target_n[i:i + chunk_size], i)
    plt.close('all')
