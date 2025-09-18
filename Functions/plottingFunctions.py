import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib
import math
from matplotlib.gridspec import GridSpec

#%%Plotting

def show_loom_in_video_clip(frame_num, stim_frame, vframerate, loom_loc, ax):
    """
    stim frame: when does loom happen?
    frame: the actual pixel data from that frame
    frame_num: what is th enumber of the current frame?
    loom_loc: x, y
    """
    
    # get the radius of the expanding circle
    loom_length= .63*vframerate #seconds --> frames
    loom_pause= .5*vframerate #seconds --> frames
    loom_size=10
    
    i=0 # loom num
    stim=0
    while not stim and i<5:
        
        stim= ((frame_num - stim_frame) < (1+i)*loom_length+i*loom_pause) * (frame_num >= (stim_frame+i*loom_length+i*loom_pause))
            # frame is < than future loomstart                             # frame is > than past loomstart
        i+=1
    
    if stim:
        i-=1
        past_frame=stim_frame+i*loom_length+i*loom_pause
        radius=loom_size * (frame_num - past_frame) +10

        circle=plt.Circle(loom_loc, radius, edgecolor='k', facecolor='k', alpha=.7)
        ax.add_patch(circle)
    

def make_window(image, centre, window_size):
    """
    Create a fixed-size window around a given point in an image using vectorized operations.

    Parameters:
    - image: 2D numpy array representing the image.
    - point: Tuple (x, y) representing the coordinates of the point.
    - window_size: Size of the window (default is 100x100 pixels).

    Returns:
    - window: 2D numpy array representing the fixed-size window around the point.
    """

    # Extract image dimensions
    img_y, img_x = image.shape

    # Calculate window boundaries
    x_ctr, y_ctr = centre
    half_window = window_size // 2
    
    #are you at an edge?
    if (x_ctr-half_window<0) or (x_ctr+half_window>img_x):
        if x_ctr-half_window<0: #left edge
            x_ctr=half_window
        elif  x_ctr+half_window>img_x: #right_edge
            x_ctr=img_x-half_window
        
        
    if y_ctr-half_window<0 or y_ctr+half_window>img_y:
        if y_ctr-half_window<0: #top edge
            y_ctr=half_window
        elif  y_ctr+half_window>img_y: #bottom edge
            y_ctr=img_y-half_window
    
    x_min = x_ctr - half_window
    x_max = x_ctr + half_window
    y_min = y_ctr + half_window #the top of the image is at y=0
    y_max = y_ctr - half_window
    

    return int(x_min), int(x_max), int(y_min), int(y_max), (int(x_ctr),int(y_ctr))






def psth(neurondata, n_time_index, all_start_stop, velocity, frame_index_s, axs, window=5, density_bins=.5):
    """
    

    Parameters
    ----------
    neurondata : int vector
        row from ndata, containing nuber of spikes per timebin.
    n_time_index : from preprocessing.
    velocity : from preprocessing.
    frame_index_s : from preprocessing
        .
    start_stop : matrix, left are start frames, right are stop frames, for one behaviour
        from hf.start_stop_array.   
    
    axs : List
        contains 3 axes, on which to plot the figure.
    window : float, optional
        how many sec before/ after an event to start/stop plotting.
        The default is 5.
    density_bins : float, optional
        how big should the bins be that average the activity in PSTH.
        The default is .5.


    """
    
    point=False
    
    #Prettify axes
    remove_axes(axs[2])
    remove_axes(axs[0],bottom=True)
    remove_axes(axs[1],bottom=True)
    
    all_spikes=[]
    all_vel=[]
    all_vel_times=[]
    bins=np.arange(-window,window+density_bins, density_bins)

    time_per_bin=0
    i=0 
    
    
    #Test if b is point or state behaviour
    if all_start_stop.shape == (2,):
        all_start_stop=[all_start_stop]
    elif len(all_start_stop.shape) ==1:
        point=True
        
        
    # Go through each trial
    for  startstop in all_start_stop:
        
        # mark where the trial is happening
        if not i: #Exception for first trial
            previous_plotstop=0
            previous_zero=0
            
           
        
        
        # plot centred at starts
        if not point: #State events
            plotzero=startstop[0]
            
            
            
            # shade additional trials (i.e. if distance to next trial is too short, don't give them a separate line but just shade them in the previous trial)
            if (plotzero-window)<previous_plotstop:
                axs[2].barh(i-2, startstop[1]-startstop[0], left = plotzero-previous_zero ,height=1 , color='burlywood', alpha=.5)
                
                continue
            
            # shade trial time
            axs[2].barh(i+1, startstop[1]-startstop[0],height=1 , color='burlywood')
            
        # # plot centred at stops
        # elif plotref==1:
        #     plotzero=startstop[1]
            
        #     # shade trial time
        #     plt.barh(i+1, startstop[0]-startstop[1],height=1 , color='burlywood')
            
        #     # shade additional trials 
        #     if (plotzero-window)<previous_plotstop:
        #         plt.barh(i+1, startstop[0]-startstop[1], left = plotzero-previous_zero ,height=1 , color='burlywood')
        #         continue
            
        
        else: #point events
            plotzero=startstop
       
        #Determine plotting window
        plotstart=plotzero-window
        plotstop=plotzero+window
       
        # Collect spikes in plotting window for this trial
        window_ind=(n_time_index>=plotstart) & (n_time_index<=plotstop)
        spikes_around_stimulus = neurondata[window_ind].copy()
        spikeind=np.where(spikes_around_stimulus>0)[0]
        spiketimes=n_time_index[window_ind][spikeind]
        
        # align to eventtime
        spiketimes-=plotzero

        # Plot the dots for one trial
        axs[2].scatter(spiketimes, np.ones_like(spikeind)*(i+1), c='teal', s=.5)
        
        # Save trial stop, to know if the next trial should get its own line, or just be plotted in the same line
        previous_plotstop=plotstop
        previous_zero=plotzero
        all_spikes.append(spiketimes)
        
        # For calculating firing in Hz, account for multiple spikes in the same 10ms bin
        while np.sum(spikes_around_stimulus>0): 
            spikes_around_stimulus[spikes_around_stimulus>0]-=1
            app_ind=np.where(spikes_around_stimulus>0)[0]
            app_times=n_time_index[window_ind][app_ind]
            all_spikes.append(app_times-plotzero)
        
        # sanity check
        if not i:
            num_timebins=np.sum(window_ind)
        else:
            if (np.sum(window_ind) - num_timebins)>1:
                raise ValueError('not in all trials you cover the same time. You assume this though in Hz calculation')
        
        
        # get veloity for trial
        velind=(frame_index_s>plotstart) & (frame_index_s< plotstop)
        all_vel_times.append(frame_index_s[velind]-plotzero)
        all_vel.append(velocity[velind])
        
        
        
        
        
        i+=1
        time_per_bin+=density_bins

    
    
    axs[2].set_ylim((0,np.nanmax((12,i+4))))
    axs[2].set_yticks(np.hstack((np.arange(0,i,10),[i])))
    axs[2].set_xlim((-window,window))
    
    

    
    #plot avg Hz on top    
    all_spikes=np.hstack(all_spikes)    
    sum_spikes, return_bins=np.histogram  (all_spikes, bins)  
    hz=sum_spikes/time_per_bin
    
    axs[1].set_xticks([])   
    axs[1].bar(bins[:-1],hz, align='edge',width=density_bins, color='grey')
    axs[1].set_xlim((-window,window))
    
    
    # plot velocity
    axs[0].set_xticks([])
    axs[0].set_xlim((-window,window))
    axs[0].set_ylim((0,130))
   
    for plotvel, plotveltime in zip(all_vel, all_vel_times):        
        axs[0].plot(plotveltime, plotvel, lw=.5, c='grey')
      
        
    # get/ plot mean velocity
    velbins=np.arange(-window, window, .1)
    binned_values, _ = np.histogram(np.hstack(all_vel_times), bins=velbins, weights=np.hstack(all_vel))
    binned_counts, _ = np.histogram(np.hstack(all_vel_times), bins=velbins)
    axs[0].plot( velbins[:-1], binned_values/binned_counts, c='orangered')
    
    
    # make dashed line at 0 
    for ax in axs:
        ax.axvline(0,linestyle='--', c='k')#, ymax=len(stimulus_times))
    
   

def box_plotter(behaviour,event, c,p, plotmax=None, ax=None):
    target_b=behaviour[behaviour['behaviours']==event]
    if not len(target_b):
        return
    if p=='axvline':
        for entry in target_b['frames_s']:
            if ax is None:
                plt.axvline(entry, label=event, color=c, lw=1.5)
            else:
                ax.axvline(entry, label=event, color=c, lw=1.5)
                
    elif p=='box':
        starts=target_b[target_b['start_stop']=='START']['frames_s'].to_numpy()
        stops=target_b[target_b['start_stop']=='STOP']['frames_s'].to_numpy()
        #replace starts/stops outside of plotting window with 0/inf
        if target_b['start_stop'].iloc[0]!='START':
            starts=np.insert(starts,0,0)
        if target_b['start_stop'].iloc[-1]!='STOP':
            stops=np.insert(stops,-1,plotmax)
            
        for start, stop in zip(starts, stops):
            if ax is None:
                plt.axvspan(start, stop, label=event, color=c)
            else:
                ax.axvspan(start, stop, label=event, color=c)
        # plt.plot()
        
    else:
        raise ValueError('p variable is unvalid')


def plot_events(behaviour, plotmin=None, plotmax=None, ax=None):
    """
    creates shading in figure according to behaviour. If behaviour is point behaviour, 
    axvline is created
    
    parameters
    -----------
    behaviour: The behaviour output file from preprocessing. It should have following colomns:
        behaviours, frames_s, start_stop
    plotmin: in s, from what time onwards  should plotting of behaviours start
    plotmax: in s, until what time should behaviour be plotted
    --> only works if both plotmin AND plotmax are given
    ax: ax object from plt.subplots()
    """
    
    if plotmin is not None:
        behaviour = behaviour[(behaviour['frames_s'] >= plotmin) & (behaviour['frames_s'] <= plotmax)]
        
        
    #state behaviours
    box_plotter(behaviour, 'approach', 'tan','box', plotmax, ax)
    box_plotter(behaviour, 'pursuit', 'coral','box', plotmax, ax)
    box_plotter(behaviour, 'attack', 'firebrick','box', plotmax, ax)
    box_plotter(behaviour, 'pullback', 'cadetblue','box', plotmax, ax)
    box_plotter(behaviour, 'nesting', 'green','box', plotmax, ax)
    
    box_plotter(behaviour, 'escape', 'steelblue','box', plotmax, ax)
    box_plotter(behaviour, 'freeze', 'aquamarine','box', plotmax, ax)
    box_plotter(behaviour, 'startle', 'mediumaquamarine','box', plotmax, ax)
    box_plotter(behaviour, 'switch', 'violet','box', plotmax, ax)
    
    box_plotter(behaviour, 'eat', 'slategray','box', plotmax, ax)
    
    
    
    # point behaviours
    box_plotter(behaviour, 'loom', 'seagreen','axvline', ax=ax)    
    box_plotter(behaviour, 'introduction', 'm','axvline', ax=ax)
    
    box_plotter(behaviour, 'turn', 'darkblue','axvline', ax=ax) 
    



def get_start_stop(b_names, frames_s, behaviours, start_stop):
    out=[]
    for b in b_names:
        starts=frames_s[(behaviours==b) & (start_stop=='START')]
        stops=frames_s[(behaviours==b) & (start_stop=='STOP')]
        both=np.array((starts,stops)).T # this is now trials * start, stop 
        out.append(both)
    return out


def get_cmap_colors(cmap_name,num_colors):
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors

def region_ticks(n_region_index, ycoords=None, ax=None, yaxis=True, xaxis=False):
    """
    makes ticks on y axis to mark the regions

    Parameters
    ----------
    n_region_index : in order of plotting, which region does each cluster belong to
        
    ycoords :  optional; if yticks shoulld be on different location than just their index
        

    Returns
    -------
    None.

    """
    yticks = [(i, region) for i, region in enumerate(n_region_index) if i == 0 or n_region_index[i-1] != region]
    
    indices, labels = zip(*yticks)
    indices=np.array(indices)
    if ax is None:
        ax=plt.gca()
    if ycoords is not None:
        
        tick_locations=ycoords[indices]
    else:
        tick_locations=indices
    
    if yaxis:
        ax.set_yticks(tick_locations, labels)
    
    if xaxis:
        ax.set_xticks(tick_locations, labels)
    
    return tick_locations







def remove_axes(axis=None, top=True, right=True, bottom=False, left=False, ticks=True, rem_all=False):
    if axis is None:
        axis = plt.gca()  
    if rem_all:
        ticks=False
    
    if isinstance(axis, matplotlib.axes.Axes):
        axs=[axis]
    else:
        axs=axis
        
    for ax in axs:
        if top or rem_all:
            ax.spines['top'].set_visible(False)
        if right or rem_all:
            ax.spines['right'].set_visible(False)
        if bottom or rem_all:
            ax.spines['bottom'].set_visible(False)
            if not ticks:
                ax.set_xticks([])
        if left or rem_all:
            ax.spines['left'].set_visible(False)
            if not ticks:
                ax.set_yticks([])





def make_cmap(colors, values):

    norm = mcolors.Normalize(min(values), max(values))
    tuples = list(zip(map(norm, values), colors))
    cmap = mcolors.LinearSegmentedColormap.from_list("", tuples)
    return cmap





def plot_tuning(tuning_matrix, target_bs,n_region_index, cmap='viridis', vmin=None, vmax=None, lines=True, area_labels=True):    
    plt.figure(figsize=(13,20))
    plt.imshow(tuning_matrix,
               vmin=vmin,
               vmax=vmax,
               aspect='auto',
               cmap=cmap)
    remove_axes()
    plt.xticks(np.arange(len(target_bs)),target_bs)
    cbar=plt.colorbar()
    if lines:
        plt.axvline(2.5, c='k')
        plt.axvline(3.5, c='k',ls='--')
        plt.axvline(6.5, ls='--',c='k')
        plt.axvline(7.5, c='k')
    if area_labels:
        region_ticks(n_region_index)
    return cbar




def logplot(Y):
    plt.plot(np.arange(1,len(Y)+1),Y)
    plt.xscale('log')
    
def subplots(n, rem_axes=True, figsize=None, gridspec=False):
    """
    creates subplots objet that has optimal layout
    also removes top and right axes 

    Parameters
    ----------
    n : int
        total number of subplots.
    rem_axes : bool, optional
        whether to remove top and right axis. The default is True.
    gridspec: whether instead of a new figure, a gridspec object should be returned
        This is useful for subdividing subplots, necessary e.g. for the PSTH function
    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    # Calculate the grid size: find two factors of n that are as close together as possible
    rows = math.floor(math.sqrt(n))
    while n % rows != 0:
      rows -= 1
    cols = n // rows

    # If n is a prime number, make the number of rows and columns as equal as possible
    if rows == 1:
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / rows)

   # Create the subplots
    if gridspec:
       fig=plt.figure(figsize=figsize)
       gs=GridSpec(rows, cols)
       return rows, cols, gs, fig
   
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs=axs.flatten()
    if rem_axes:
        remove_axes(axs)
    
    #remove unused axes
    if rows * cols > n:
        for i in range(n, rows * cols):
            fig.delaxes(axs[i])
   

    return fig, axs