## FEATURE:

## Executive Summary
this is python GUI wrapper application for the spikeinterface python package. implemented with NICEGUI.
the puprpose of this app is to help non-programmer neuroscientists use the spikinterface package to build and run spike-sorting pipelines

## CORE FEATURES:
these are implemented as tabs in the GUI
- flexible data loading, using extractors for different data formats
- channel mapping. automatic channel mapping. automatic if neuropixels, using https:\\github.com\jenniferColonell\SGLXMetaToCoords\blob\main\SGLXMetaToCoords.py as reference.
  or using the probeinterface library.
- preprocessing: optional (radio-buttons or toggle buttons), bandpass, phase-shift correction, common average refferencing., detect bad-channels, remove bad channels 
- drift-correction: flexible drift correction using the drift motion \ correction module
- spike-sorting, using Sorters module in spikeinterface
- postprocessing: creating a sorting analyzer object
- summary a visual report of results, and summary of all steps 

with the following file structure:

spike_sorter_app\
├── main.py                     # App entry point (Application, initial MainWindow)
│
├── app_config.ini              # (Generated\managed) Hidden app settings, license info
├── user_config.json            # (Generated\managed) User's paths, pipeline settings
│
├── gui\
│   ├── __init__.py
│   ├── main_window.py            # Main MainWindow, tab management, IPC setup
│   ├── options_window.py         # Dialog for detailed preprocessing options
│   │
│   ├── tabs\                     # Individual QWidget for each tab's content
│   │   ├── __init__.py
│   │   ├── load_tab.py
│   │   ├── probe_tab.py
│   │   ├── preprocess_tab.py
│   │   ├── drift_tab.py
│   │   ├── sort_tab.py
│   │   ├── postprocess_tab.py
│   │   ├── save_tab.py
│   │   └── summary_tab.py
│   │
│   └── widgets\                  # Reusable custom GUI widgets
│       ├── __init__.py
│       ├── file_table_model.py   # (From your helperFunctions)
│       ├── status_indicator.py     # (From your helperFunctions, modified for 3 states)
│       └── result_poller.py        # Thread for polling results from backend
│
├── core\                       # Non-GUI core logic & application settings
│   ├── __init__.py
│   ├── app_settings_manager.py   # Manages app.ini, license checks
│   └── user_settings_manager.py  # Manages user_config.json (CRUD, paths, pipeline state)
│
├── backend\
│   ├── __init__.py
│   ├── worker.py                 # ProcessingWorker(multiprocessing.Process) class
│   └── tasks.py                  # Functions for each pipeline step (using spikeinterface)
│                                 # e.g., run_load_data, run_preprocess, run_sort
│
└── utils\                      # Truly generic helper functions (if any remain)
    └── __init__.py


## Dependencies and Environment
### Python Packages
spikeinterface
spikeinterface-gui
kilosort
nicegui



### System Requirements
Python version: 3.11+



	
## EXAMPLES:
- neuropixels pipeline example: https:\\spikeinterface.readthedocs.io\en\latest\how_to\analyze_neuropixels.html
 this is an example of how to run a whole sorting pipeline on a neuropixels recording
- neuropixel channel mapping: https:\\github.com\jenniferColonell\SGLXMetaToCoords\blob\main\SGLXMetaToCoords.py
- examples\core\plot_4_sorting_analyzer.py  - example for creating a sorting analzyzer for postprocessing
- examples\widgets - examples for vizualisations 
- examples\qualitymetrics\plot_3_quality_metrics.py - example for quality metrics for postprocessing


## DOCUMENTATION:

front end NICEGUI documentation: https:\\nicegui.io\documentation

spikeinterface API: https:\\spikeinterface.readthedocs.io\en\latest\api.html

1. **data loading tab**
   - Description: flexible loading of different recording types
Reference: https:\\spikeinterface.readthedocs.io\en\stable\modules\extractors.html 
2. **probe mapping tab**
   - Description: [the user can select the probe used in the recording from lists of manufacterers, probe models, optionally from a list of headstages\adaptors ].
after choice is made, a plot of the desired probe and channel IDs is rendered to confirm the selection  
   
3. **preprocessing tab**
   - Description: user can choose preprocessing steps. by toggling ON\OFF buttons for:
1) bandpass (with highpass and lowpass parameters) 
2) phase shift correction (available only if a neuropixels probe is selected (manufacturer is IMEC)
3) common median refference (with "global" or "local" options. if local is selected, additional options (inner and outer radius on slider buttons)
4) destripe
5) whiten
6) bad channels handling:
	a) detect bad channels
	b) automatically remove bad channels OR manually manually remove channels. The latter requires the user to input a list of 1-based numbers
reference: https:\\spikeinterface.readthedocs.io\en\stable\modules\preprocessing.html 
3. **Drift correction tab**
   - Description: user can visualize different drift correction methods and their effect on the pipeline.

user can choose drift correction methods steps, by choosing from a dropdown list.
user can optionally choose start and stop time for the drift correction.
the user can then press a button to run the motion correction. Plotting options: “LFP estimation”, “plot peaks”, “plot current drift method”  
4. **spike-sorting tab**
   - Description: user can choose a sorter from a drop down menu. Once selected, the user can optionally show and change sorter parameters.
Once confirmed, the user can then run the sorter
Reference: https:\\spikeinterface.readthedocs.io\en\stable\modules\sorters.html 
5. **postprocessing tab**
   - Description: user can choose postprocessing steps described in the spikeinterface postprocessing module: https:\\spikeinterface.readthedocs.io\en\stable\modules\postprocessing.html#postprocessing-module , https:\\spikeinterface.readthedocs.io\en\stable\modules\qualitymetrics.html 
6. **save data options tab**
   - Description: user can choose what format to save results in: “phy or “zarr”
   Export reference: https:\\spikeinterface.readthedocs.io\en\stable\modules\exporters.html 
 

probeinterface library: https:\\probeinterface.readthedocs.io\en\main\examples\ex_10_get_probe_from_library.html
serena MCP: https:\\github.com\oraios\serena 
https:\\github.com\oraios\serena\blob\main\docs\custom_agent.md

Frontend: nicegui.io\documentation

Pydantic:
Pydantic AI Official Documentation: https:\\ai.pydantic.dev\
Agent Creation Guide: https:\\ai.pydantic.dev\agents\
Tool Integration: https:\\ai.pydantic.dev\tools\
Testing Patterns: https:\\ai.pydantic.dev\testing\
Model Providers: https:\\ai.pydantic.dev\models\


## OTHER CONSIDERATIONS:


Use environment variables for API key configuration instead of hardcoded model strings
Keep agents simple - default to string output unless structured output is specifically needed
Follow the main_agent_reference patterns for configuration and providers
Always include comprehensive testing with TestModel for development






