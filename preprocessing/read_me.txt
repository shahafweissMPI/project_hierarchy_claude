
===============================================================================================================================
PREPROCESSING STEPS

0) to add a new recording session, first fill in FILL PATHS.CSV:

-located at "\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\paths.csv"
-One entry per session, with paths to following files
	//Mouse_ID: ID of mouse
	//session: date of session, add _1 / _2 / etc if more than one session per day
	//Video: path to video recording of session
	//csv_timestamps: _timestamps.csv from bonsai, with one entry per camera frame
	//csv_loom: _loomns_timestamp.csv from bonsai, with one entry per loom
	//nidq, lf, ap: .bin files of recording
	//sorting spikes: from IRC clustering, path to csv with one entry per spike (the csv that does not end with _quality)
	//sorting_quality: from IRC clustering, path to csv with one entry per neuron (ends with _quality.csv)
	//boris_labels: path to csv exported from boris labelling (needs to be exported as csv)
	//probe_tracking: path to track.csv from probe alignment to atlas
*	//last_channel_manually: In case there is big difference between electrophysiology and histology, you can give here the 
		last channel that you think is still in the brain. In case you specify this, the alignment to histology will use
		this, rather than the 'depth' metric from histology alignment to determine the number of channels in the brain
	//preprocessed: Where the preprocessed file should be saved (usually in the 'postscript' folder of each session)
	//diode_threshold: Threshold above which the diode signal should be considered a loom. Needs to be fine-tuned for each session
	//framechannel: On which digital channel of the nidq is the frame signal
	//diode_channel: On which analog channel of the nidq is the diode signal
	//frame_loss: what is the difference in frames between nidq and video (relevant for alignment of frame_index_s to velocity variables)
xx	//loom: Boolean; does this session contain looms/ escapes?
xx	//cricket: Boolean; is there crickets/ hunting in this session?
xx	//pups: Boolean; are there pups in the session?
	//mouse_tracking: path to .h5 file from SLEAP, tracking the mouse
xx	//cricket_tracking: path to .h5 file from SLEAP, tracking the cricket
xx	//pup_tracking: path to .h5 file from SLEAP, tracking the pups
xx	//Sorting: not used, don't remember what it was for
xx	//notes: any comments?


*   optional input
xx  not used in preprocessing script





______________________________________________________________
1) GET LOOM TIMES
- run preprocess_all.py until/ including: #%% get diode signal
	--> for this you need to have filled in the paths.csv, but not yet the parts for boris/ spike sorting/ tracking
- use diode_frame_index to annotate looms in boris (using diode_s_index is unreliable, since boris assumes framerate of exactly 50.0 fps)

______________________________________________________________
2) DO BORIS
-best use the project file, located under "\\gpfs.corp.brain.mpg.de\stem\data\project_hierarchy\data\boris.boris"
- use diode_frame_index to annotate looms in boris (using diode_s_index is unreliable, since boris assumes framerate of exactly 50.0 fps)
- mark behaviours:
	//startle: after loom, first visible reaction; ears move up; stop movement
	//approach: carefully getting closer to the cricket, sniffing at it, not yet attacking
	//switch: period between stop of attack and start of escape after a loom; often it is the same time as the 'startle' 
	//loom: first frame of loom, as registered from the preprocess_all.py script
	//introduction: new cricket in arena
	//pursuit: vigorous chase after the cricket/ attack
	//eat: consuming the cricket when it is already dead
	//attack: biting/ grabbing the cricket, during pursuit
	//escape: the running part of escape, first frame is when animal the animal is most elongated in the first leap, 
		last frame when it is properly in the shelter, already sitting
	//freeze: Longer period of immobility directly after loom
	//turn: turning head to initiate escape after loom
	//pullback: animal 'escapes' from cricket even though there is no loom. First frame is when mouse stops attacking, 
		last frame is when it stops running/ turns back to cricket
	//petros_hand: Periods when petros hand is visible; I marked this because sometimes the animal shows strong escape reactions
	//weird_shake: One animal consistently did little shakes, super fast, a bit like a wet dog. Marked it, 
		just because I didn't know what it was

IMPORTANT NOTES:
-Because of how I wrote the scripts now, for one loom there should only be one escape. If the animal escapes,
	stops, and the escapes again, either just count the first part or count the entire duriation until 
	reaching shelter
-Computational thresholding of escapes:
	// if loom is more than 7.5 seconds before escape onset (running, not the startle)
	// NO VELOCITY THRESHOLD, since even the slowest escapes I marked (~30 cm/s) did have some neurons responding to them

______________________________________________________________
3) DO SPIKE SORTING
right now everything expects IRC clustering, but there is also a function in 
preprocess_all.py that can read phy output. 

______________________________________________________________
4) PROBE ALIGNMENT WITH BRAINREG/ BRAIN-SEGMENT
use 1000 tracking points

______________________________________________________________
______________________________________________________________
WHAT SCRIPTS TO RUN NEXT

best look in the README files in project_hierarchy and plotting folder.

Getting a feeling for the data: (project_hierarchy>>plotting )
-entire_session.py: activity of all neurons across the session
-PSTH.py: PSTH for each neuron, for all the behaviours that you specify
-raster_video: have neural activity right next to video


Overview of tuning: (project_hierarchy>>plotting>>neuron_tuning)
- s_num_permutation.py: compute tuning, get overview for this session
- total_tuning_per_area: How does overall area tuning change when taking into account the new one?
- Venn_tuning: How does overall number of neurons tuned to multiple behaviours changed with new session?









===============================================================================================================================
SPECIAL CASES

______________________________________________________________
split sessions

- if sessions were concatenated for spike sorting, you can split them again in the preprocess_all.py script
- Each sub-session needs it's own entry in the paths.csv file
- Steps:
	// add session number to list in line 10
	// set split_sessions to which of the concatenated sessions you want (e.g. if you concatenated 3 
		sessions and want to preprorocess the second one, set split_sessions=1, counting starts from 0)
	// set split values to where sessions were concatenated in seconds. (e.g. if you concatenated 3 sessions, 
		each with a length of 10 seconds, values need to be [0, 10, 20, 30]
	// you get out only the spikes/ behaviour for the sub-session that you indicated


______________________________________________________________
deal with lost frames

-count_lost_frames.py: put desired session number in 'sessions' list, script prints out 
	framerate, unique distances between frames (to check if there whether all frames happen 
	at regular intverals in the nidq), and difference between video frames and nidq frames
	(i.e. positive numbers mean more frames in the video than nidq)
-fill_in_lost_frames.py: if frames were registered on the nidq, but not written in the video BUT can be retrieved
	from the camera timestamps. This script writes a new video in which it duplicates the previous frame, so that 
	now the video is aligned to the nidq. New video is saved at: r"F:\scratch\deleteme.avi"


______________________________________________________________
only_behaviour_preprocessing.py

-If you've done a batch of boris annotations but haven't got the neural data ready yet OR
	if you want to analyse data from animals where you didn't record neural activity
-imports boris information, aligns to frames from nidq, checks diode, gets velocity trace from SLEAP
-You just need to give the session names in 'sessionlist'


______________________________________________________________
preprocess_ses_240212.py

-In this session nidq frame acquisition stopped after 13 min, but video and neural data was still aqcquired. 
-The script extrapolates frame times using the framerate of existing frames. If you want to preprocess that session
	(or another session that has the same issue) you need to first run the file, which puts the necessary variables in memory,
	and then run the preprocess_all.py script from section 'get spiketimes' onwards


______________________________________________________________
view_only_neurons.py

-In case you're done with clustering, but haven't got the boris/ velocity/ histology stuff yet
-Makes a plot that is neurons*time, one time raw, one time zscored







===============================================================================================================================
OUTPUTS FROM PREPROSESSING

______________________________________________________________
np_neural_data.npy

LOADING COMMAND
data=np.load(path, allow_pickle=True).item()

CONTENT DESCRIPTION
neural data, regions, channels, time in s

n_by_t: neural data, neurons*time. Each entry is the number of spikes per time interval. Each time interval covers 10ms (default)
time_index: time index for each entry of n_by_t. In s
cluster_index: What is the clusternumber of each neuron in n_by_t (not used in any of the scripts. E.g. in PSTH I just number neurons from 0-	num_neurons)
cluster_regions: what region does each neuron belong to?
spike_source: phy or ironclust
cluster_channels: What channel is each cluster from?


______________________________________________________________
behaviour.csv

LOADING COMMAND
data=pd.read_csv(path)

CONTENT DESCRIPTION
-behaviours: what is the behaviour that is happening, e.g. loom, approach, etc
-start_stop: is the behaviour starting, ending, or is it a point event?
-boris_frames_s: at what  time is this happening in s



______________________________________________________________
tracking.npy

LOADING COMMAND
data=np.load(path, allow_pickle=True).item()

CONTENT DESCRIPTION
velocity: velocity of the point f_back in cm/s.
locations: frames * tracking_nodes * x,y; gives you the x,y position of each node in each frame
node_names: same order as tracking_nodes, name of each node
frame_index_s: at what time does each frame happen, in s

