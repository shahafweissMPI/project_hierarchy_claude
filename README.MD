=====================================================================================================
# INSTALLATION:
## option 1 (doens't always work)
go to cloned repo folder.

conda env create -f environment.yml

## option 2:
- mamba create -n hierarchy python=3.11.11

- mamba install bzip2=1.0.8 ca-certificates=2024.12.14 libexpat=2.6.4 libffi=3.4.2 liblzma=5.6.3 libsqlite=3.47.2 libzlib=1.3.1 openssl=3.4.0 python=3.11.11 ucrt=10.0.22621.0 vc=14.3 vc14_runtime=14.42.34433 vs2015_runtime=14.42.34433 wheel=0.45.1 numpy=2.2.6

- pip install aiohappyeyeballs==2.4.4 aiohttp==3.11.11 aiosignal==1.3.2 alabaster==1.0.0 altair==5.5.0 annotated-types==0.7.0 anyio==4.8.0 arrow==1.3.0 asciitree==0.3.3 asttokens==3.0.0 attrs==24.3.0 av==14.1.0 babel==2.17.0 backports-tarfile==1.2.0 bidsschematools==0.7.2 blosc2==3.0.0 boto3==1.36.15 botocore==1.36.15 bottleneck==1.4.2 cartopy==0.24.1 cbor2==5.6.5 certifi==2024.12.14 cffi==1.17.1 cftime==1.6.4.post1 charset-normalizer==3.4.1 ci-info==0.3.0 click==8.1.8 click-didyoumean==0.3.1 cloudpickle==3.1.0 colorama==0.4.6 comm==0.2.2 contourpy==1.3.1 cryptography==44.0.0 cuda-python==12.6.2.post1 cupy-cuda11x==13.4.0 cycler==0.12.1 dandi==0.66.5 dandischema==0.11.0 dask==2024.12.1 debugpy==1.8.12 decorator==5.1.1 decord==0.6.0 deprecated==1.2.18 dill==0.3.9 distinctipy==1.3.4 distributed==2024.12.1 dnspython==2.7.0 docstring-parser==0.16 docutils==0.21.2 elephant==1.1.1 email-validator==2.2.0 et-xmlfile==2.0.0 etelemetry==0.3.1 executing==2.1.0 fasteners==0.19 fastrlock==0.8.3 figurl==0.2.22 filelock==3.17.0 flox==0.10.0 fonttools==4.55.3 fqdn==1.5.1 frozenlist==1.5.0 fscacher==0.4.4 fsspec==2024.12.0 futures==3.0.5 h11==0.14.0 h5py==3.12.1 hdf5plugin==5.0.0 hdf5storage==0.1.19 hdmf==3.14.5 hdmf-docutils==0.4.7 hdmf-zarr==0.11.0 httpcore==1.0.7 httpx==0.28.1 huggingface-hub==0.29.2 humanize==4.11.0 idna==3.10 imageio==2.37.0 imageio-ffmpeg==0.6.0 imagesize==1.4.1 importlib-metadata==8.5.0 interleave==0.3.0 ipykernel==6.29.5 ipympl==0.9.6 ipython==8.32.0 ipywidgets==8.1.5 isodate==0.7.2 isoduration==20.11.0 jaraco-classes==3.4.0 jaraco-context==6.0.1 jaraco-functools==4.1.0 jax==0.5.0 jaxlib==0.5.0 jaxopt==0.8.3 jedi==0.19.2 jinja2==3.1.5 jmespath==1.0.1 joblib==1.4.2 jsonpointer==3.0.0 jsonschema==4.23.0 jsonschema-specifications==2024.10.1 jupyter-client==8.6.3 jupyter-core==5.7.2 jupyterlab-widgets==3.0.13 kachery-cloud==0.4.10 kaleido==0.2.1 keyring==25.6.0 keyrings-alt==5.0.2 kiwisolver==1.4.8 lazy-ops==0.2.0 llvmlite==0.43.0 locket==1.0.0 lxml==5.3.0 markdown-it-py==3.0.0 markupsafe==3.0.2 matplotlib==3.10.0 matplotlib-inline==0.1.7 mdurl==0.1.2 mearec==1.9.1 meautility==1.5.2 ml-dtypes==0.5.1 more-itertools==10.6.0 movement==0.0.23 moviepy==2.1.2 msgpack==1.1.0 mtscomp==1.0.2 multidict==6.1.0 narwhals==1.21.1 natsort==8.4.0 nc-time-axis==1.4.1 ndindex==1.9.2 ndx-dandi-icephys==0.4.0 ndx-events opencv-python scikit-learn movement==0.6  scipy==1.15.2

if using spyder:
conda install spyder-kernels=2.5
or
pip install spyder-kernels==2.5.*


packages worth upgrading:
pip install -U spikeinterface 

Set functions path in Spyder:
-open path-manager (blue-yellow python symbol on top of the code)
-Add path to 'Functions' folder in your cloned GitHub repository (e.g. "C:\Users\kernt\GitHub\Project_hierarchy\Functions")

=====================================================================================================
GENERAL FOLDER OVERVIEW

__________________________________________
preprocessing
-Preprocess the raw video/ neural data, mostly using the script preprocess_all.py


__________________________________________
Analysis

------------------------------------------
-Behaviour: Analyse just boris data, i.e. escape probability, peak velocity, etc

	\\all_trajectories: Plots position of animal across entire session on arena with velocity as 
		colormap
*	\\escape_latency: avg time delay between loom and escape onset, also comparison between escape 
		and switch
	\\escape_trajectories: Position and velocity of animal on arena, only during escapes
*	\\escape_trajectories_by_switch: same, but separate plots for regular escape and switch escapes
	\\escape_velocity: Compare peak velocities between escapes and swicthes
	\\Hunting_latency: How much time between cricket introduction and first interaction with 
		cricket?
*	\\loom_reactions: Escape probability vs probability of no reaction
*	\\speed_map_per_loom: Heatmap where each row is an escape, colored by escape velocity (makes 
		this figure that Vanessa has in her papers as well)
	\\strategy_development: Bayesian predictions of escape/ hunting/ pullback probabilities


------------------------------------------
-Communication: How do neurons interact within and across regions? (mainly correlations, but also 	activity predictions using regression)

*	\\comm_video: Creates video of correlation across time in a session.
xx	\\factor_area_comm: Uses SVD factors to predict activity in region B from activity in region A 		(didn't work so well, low reliability)
*	\\pullback_correlation: Shows avg correlation during pullbacks, right before pullbacks, and 
		right after pullbacks. Also summarises this for regions, so that you see avg 
		correlation of region A with all other regions during the mentioned timepoints
xx	\\single_n_area_comm: Tries to predict single neuron activity in region B from single neuron 		activity in region B (also didn't work so well)
*	\\single_n_corr: get correlation of all neurons with each other, either for the whole session, 		or only for hunting periods
	\\single_n_corr_bs: Shows correlation between neurons for specific behaviours, e.g. approach, 		escape, etc


------------------------------------------
-Dimensions: SVD/ PCA analysis
	\\SVD_components: PCA of global activity across entire session during attack, escape, and 		baseline; also calculates neural dimensinality 
	\\SVD_per_area_dim: Same, but separately calculated for each area
	\\svd_per_area_reg: Tuning of each area to each behaviour. Extracts SVD factors from 20 neurons
		 of each area and tries to predict each factor using the marked behaviours. Gives you
		r2 for each factor and beta-weights for the first 4 factors
	\\SVD_regression_analysis: Same, but across areas (i.e. one SVD and regression model for all 
		neurons)


------------------------------------------
-Misc: Different stuff 
	\\escape_by_speed: PSTH of escape for each neuron, loom is marked, trials are sorted by peak 
		velocity
	\\escape_by_speed_scatter: For neurons responsive to escape, plot firing change against escape 
		velocity, to see if their activity scales with velocity
xx	\\lfp: shows lfp signal across time (works only if you preprocessed lfp signal before)
	\\loom_centred_plots: Show cutout of neural activity around each loom
	\\single_neuron: Plot activity of one neuron across time (e.g. if you think the PSTH looks 
		interesting and want to follow up on that)
	\\Velocity_corr: correlation of each neuron with velocity trace and distance to shelter
*	\\warped_switch_activity: For sequence of hunting -- switch -- escape, stretches each event to
		 the same length and averages across events. Doesn't really look like anything at the 
		moment, maybe it gets better when you isolate responsive neurons/ if you do this with 
		the dPAG


------------------------------------------
-neuron_tuning: How does each neuron respond to each behaviour?

	-old_neuron_tuning: Scripts that I don't think are needed anymore but I also didn't want to 
		delete them yet

	-regression: Predicting neural activity from behaviour labels using linear regression. The 
		tuning values are the product of r2 and beta weights

		\\regression_acc_vel_dist: predicting neural activity using velocity, accelaration, and 
			distance to shelter, calculating tuning based on prediction performance (r2) and
			beta weights
*		\\regression_tuning: Same, but predictions are made based on binary behaviour labels

	
	-s_num: Tuning is calculated based on the probability of firing change when behaviour happens
x		\\s_num_example: Shows Each step of the calculation of s_num for example neurons
			---> Outdated, would need updating
*		\\s_num_permutation: Calculates s_num for all neurons in a given session list and uses 
			permutations to find significant changes in firing
*		\\total_tuning_per_area: Summarises tuning from multiple sessions. For each area a 
			percentage is given how many neurons are responsive to each behaviour. If a 
			neuron is responsive to more than 1 behaviour, it will be counted multiple 
			times. 
*		\\venn3: For each region, shows you how many neurons respond to only one or multiple 
			behaviours. Relies on results from s_num_permutation.




------------------------------------------
-raw: Plot raw activity to get a good feeling for data
*	\\entire_session: Plots all neural firing from one session as lineplot, as dots, or as colored 
		plot over time, and marks behaviour
*	\\PSTH: makes one PSTH per neuron for all the behaviours that you indicate. 
	\\PSTH_1_b: Makes one big PSTH for all neurons that you indicate, for only one behaviour
*	\\raster_vid_give_times: Makes video of mouse and video of neural activity next to each other. 
		You can indicate periods of interest by indicating start/ stop times (s)
*	\\raster_video: Same; Automoatically makes videos around each loom






* Important code
xx very unimportant
































































































