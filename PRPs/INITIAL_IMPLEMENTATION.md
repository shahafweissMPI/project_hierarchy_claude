name: "NiceGUI Spike Sorting Application - Complete Initial Implementation PRP"
description: |

## Purpose
Comprehensive PRP for implementing a complete NiceGUI-based spike sorting application that wraps SpikeInterface functionality for neuroscientists.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Build a complete spike sorting GUI application using NiceGUI that wraps SpikeInterface functionality, enabling neuroscientists to build and run spike-sorting pipelines through a user-friendly tabbed interface without programming knowledge.

## Why
- **Business value and user impact**: Democratizes spike sorting for neuroscientists without programming skills
- **Integration with existing features**: Leverages proven SpikeInterface library with modern web UI
- **Problems this solves and for whom**: Eliminates coding barriers for researchers who need to analyze electrophysiological data

## What
A multi-tab NiceGUI application with the following user-visible behavior and technical requirements:
- Tab-based navigation through the complete spike sorting pipeline
- File upload and data loading with automatic format detection
- Interactive probe selection and channel mapping with visualization
- Configurable preprocessing options with real-time parameter adjustment
- Spike sorting with multiple algorithm options
- Postprocessing analysis and quality metrics
- Export capabilities in multiple formats
- Progress tracking and error handling throughout the pipeline

- in each tab, reserve the left quarter of the window for text instructions to guide the user.
- in each tab, reserve the right quarter of the window for a canvas or visualization figures
- in each tab, use the center 2 quarters for parameter tuning and user interactions
### Success Criteria
- [ ] Complete tabbed application loads and displays properly
- [ ] Data loading tab successfully imports various recording formats
- [ ] Probe mapping tab displays probe visualizations and handles channel mapping
- [ ] Preprocessing tab provides toggle options with parameter controls
- [ ] Drift correction tab offers method selection and visualization
- [ ] Spike sorting tab runs sorting algorithms with progress tracking
- [ ] Postprocessing tab computes and displays quality metrics
- [ ] Save/export tab provides multiple output format options
- [ ] All pipeline steps maintain state between tabs
- [ ] Error handling provides meaningful user feedback

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://spikeinterface.readthedocs.io/en/latest/api.html
  why: Core API modules for recording extractors, preprocessing, sorters, postprocessing, and exporters
  critical: Understanding of BaseRecording, BaseSorting, SortingAnalyzer workflow
  
- url: https://spikeinterface.readthedocs.io/en/latest/how_to/analyze_neuropixels.html
  why: Complete Neuropixels pipeline example with specific function calls and parameters
  critical: Exact preprocessing sequence, sorting configuration, postprocessing steps
  
- url: https://nicegui.io/documentation
  why: NiceGUI tabbed applications, event handling, file upload, plotting integration
  critical: ui.tabs(), ui.tab_panels(), ui.upload(), reactive data binding patterns
  
- file: examples/core/plot_4_sorting_analyzer.py
  why: SortingAnalyzer creation and extension computation patterns
  critical: create_sorting_analyzer(), compute() method usage, extension dependencies
  
- file: examples/widgets/plot_1_rec_gallery.py
  why: SpikeInterface visualization widget patterns
  critical: sw.plot_traces(), sw.plot_probe_map() integration approaches
  
- url: https://probeinterface.readthedocs.io/en/main/examples/ex_10_get_probe_from_library.html
  why: Probe library access and manufacturer/model selection patterns
  critical: get_probe() function, plot_probe() visualization, probe annotation handling
  
- url: https://github.com/jenniferColonell/SGLXMetaToCoords/blob/main/SGLXMetaToCoords.py
  why: Automatic Neuropixels coordinate extraction from metadata
  critical: geomMapToGeom(), shankMapToGeom() coordinate calculation methods

- docfile: CLAUDE.md
  why: Project conventions, UV package management, vertical slice architecture, testing patterns
  critical: File size limits (500 lines), function limits (50 lines), UV commands, type hints
```

### Current Codebase Tree
```bash
nicegui_app_claude/
├── CLAUDE.md              # Project conventions and guidelines
├── README.md              # Project overview
├── INITIAL.md             # Feature requirements (this file)
├── PRPs/                  # Project Requirement Profiles
│   └── templates/
│       └── prp_base.md
├── examples/              # SpikeInterface example code
│   ├── core/             # Core functionality examples
│   │   ├── plot_1_recording_extractor.py
│   │   ├── plot_4_sorting_analyzer.py
│   │   └── ...
│   ├── widgets/          # Visualization examples
│   │   ├── plot_1_rec_gallery.py
│   │   └── ...
│   ├── extractors/       # Data loading examples
│   ├── forhowto/         # Tutorial examples
│   ├── qualitymetrics/   # Quality assessment examples
│   └── ...
└── .claude/              # Claude Code settings
```

### Desired Codebase Tree with Files to be Added
```bash
nicegui_app_claude/
├── main.py                     # Application entry point with NiceGUI setup
├── pyproject.toml              # UV project configuration with dependencies
├── requirements.txt            # Python dependencies (fallback)
├── app_config.ini              # (Generated/managed) Application settings
├── user_config.json            # (Generated/managed) User pipeline state
├── gui/                        # GUI components and structure
│   ├── __init__.py
│   ├── main_window.py          # Main application window with tab management
│   ├── tabs/                   # Individual tab implementations
│   │   ├── __init__.py
│   │   ├── load_tab.py         # Data loading interface (BaseRecording creation)
│   │   ├── probe_tab.py        # Channel mapping and probe selection
│   │   ├── preprocess_tab.py   # Preprocessing options configuration
│   │   ├── drift_tab.py        # Drift correction methods and visualization
│   │   ├── sort_tab.py         # Spike sorting algorithm selection
│   │   ├── postprocess_tab.py  # SortingAnalyzer and quality metrics
│   │   ├── save_tab.py         # Export format selection
│   │   └── summary_tab.py      # Pipeline summary and results
│   ├── widgets/                # Reusable GUI components
│   │   ├── __init__.py
│   │   ├── file_browser.py     # File selection widget
│   │   ├── progress_bar.py     # Progress tracking widget
│   │   ├── plot_container.py   # Matplotlib/Plotly container widget
│   │   └── parameter_panel.py  # Dynamic parameter configuration widget
├── core/                       # Application logic and state management
│   ├── __init__.py
│   ├── pipeline_state.py       # Pydantic models for pipeline state
│   ├── app_settings.py         # Application configuration management
│   └── validation.py           # Input validation and error handling
├── backend/                    # SpikeInterface integration layer
│   ├── __init__.py
│   ├── recording_manager.py    # Recording loading and management
│   ├── preprocessing_manager.py # Preprocessing pipeline coordination
│   ├── sorting_manager.py      # Spike sorting execution
│   ├── analysis_manager.py     # Postprocessing and quality metrics
│   └── export_manager.py       # Export format handling
├── utils/                      # Utility functions and helpers
│   ├── __init__.py
│   ├── probe_utils.py          # Probe library integration utilities
│   ├── file_utils.py           # File handling and path utilities
│   └── plotting_utils.py       # Visualization helper functions
└── tests/                      # Test suite (following vertical slice pattern)
    ├── __init__.py
    ├── conftest.py             # Pytest configuration and fixtures
    ├── test_gui/               # GUI component tests
    ├── test_core/              # Core logic tests
    ├── test_backend/           # Backend integration tests
    └── test_utils/             # Utility function tests
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: SpikeInterface requires specific import patterns
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

# CRITICAL: NiceGUI async context handling
# All UI updates must happen in proper async context
# Use ui.timer() for background processing updates

# CRITICAL: SpikeInterface SortingAnalyzer extension dependencies
# Extensions have parent/child relationships - recomputing parent deletes children
# Order: random_spikes -> waveforms -> templates -> other extensions

# CRITICAL: ProbeInterface probe assignment
# Probes must be explicitly set to recordings: recording.set_probe(probe)
# Channel indices must be properly mapped: probe.set_device_channel_indices()

# CRITICAL: NiceGUI file upload handling
# Files are temporarily stored, content accessed via event.content.read()
# Need proper cleanup of temporary files

# CRITICAL: UV package management
# NEVER update pyproject.toml directly - always use "uv add packagename"
# Use "uv run python script.py" for execution in virtual environment
```

## Implementation Blueprint

### Data Models and Structure
Create core Pydantic models ensuring type safety and pipeline state consistency:

```python
# core/pipeline_state.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from pathlib import Path

class RecordingState(BaseModel):
    """State for loaded recording data."""
    file_path: Optional[Path] = None
    format_type: Optional[str] = None
    sampling_rate: Optional[float] = None
    num_channels: Optional[int] = None
    duration: Optional[float] = None
    channel_ids: Optional[List[str]] = None

class ProbeState(BaseModel):
    """State for probe configuration."""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    num_channels: Optional[int] = None
    geometry: Optional[Dict[str, Any]] = None
    is_configured: bool = False

class PreprocessingState(BaseModel):
    """State for preprocessing configuration."""
    bandpass_enabled: bool = False
    highpass_freq: float = 400.0
    lowpass_freq: float = 6000.0
    phase_shift_correction: bool = False
    common_reference_enabled: bool = False
    reference_type: str = "global"  # "global" or "local"
    detect_bad_channels: bool = False
    remove_bad_channels: bool = False

class SortingState(BaseModel):
    """State for spike sorting configuration."""
    sorter_name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_complete: bool = False

class PipelineState(BaseModel):
    """Complete pipeline state."""
    recording: RecordingState = Field(default_factory=RecordingState)
    probe: ProbeState = Field(default_factory=ProbeState)
    preprocessing: PreprocessingState = Field(default_factory=PreprocessingState)
    sorting: SortingState = Field(default_factory=SortingState)
    current_tab: str = "load"
```

### List of Tasks to be Completed in Order

```yaml
Task 1: Project Setup and Dependencies
CREATE pyproject.toml:
  - PATTERN: Follow CLAUDE.md UV package management guidelines
  - DEPENDENCIES: nicegui, spikeinterface, spikeinterface-gui, kilosort, pydantic, pytest, ruff, mypy
  - SETUP: Use "uv add" for each dependency, never edit pyproject.toml directly

CREATE main.py:
  - PATTERN: NiceGUI application entry point with tab structure
  - INTEGRATE: ui.tabs() and ui.tab_panels() for main navigation
  - SETUP: FastAPI backend integration for file handling

Task 2: Core State Management
CREATE core/pipeline_state.py:
  - PATTERN: Pydantic v2 models following CLAUDE.md conventions
  - IMPLEMENT: RecordingState, ProbeState, PreprocessingState, SortingState, PipelineState
  - INTEGRATE: JSON serialization for persistence

CREATE core/app_settings.py:
  - PATTERN: pydantic-settings BaseSettings pattern from CLAUDE.md
  - IMPLEMENT: Configuration management with environment variables
  - SETUP: Settings caching with @lru_cache()

Task 3: Backend SpikeInterface Integration
CREATE backend/recording_manager.py:
  - PATTERN: Mirror spikeinterface extractors usage from examples/core/plot_1_recording_extractor.py
  - IMPLEMENT: load_recording() method supporting multiple formats
  - INTEGRATE: se.read_spikeglx(), se.read_openephys(), automatic format detection

CREATE backend/preprocessing_manager.py:
  - PATTERN: Follow examples from spikeinterface.preprocessing module
  - IMPLEMENT: apply_preprocessing() with configurable steps
  - SEQUENCE: bandpass_filter() -> phase_shift() -> common_reference() -> detect_bad_channels()

CREATE backend/sorting_manager.py:
  - PATTERN: Follow Neuropixels example workflow
  - IMPLEMENT: run_sorting() with sorter selection and parameter management
  - INTEGRATE: ss.run_sorter() with Docker image handling

CREATE backend/analysis_manager.py:
  - PATTERN: Mirror examples/core/plot_4_sorting_analyzer.py patterns
  - IMPLEMENT: create_analyzer(), compute_extensions(), calculate_quality_metrics()
  - SEQUENCE: create_sorting_analyzer() -> compute("random_spikes") -> compute("waveforms") -> compute("templates")

Task 4: GUI Widgets and Components
CREATE gui/widgets/file_browser.py:
  - PATTERN: NiceGUI ui.upload() with file filtering
  - IMPLEMENT: FileSelector widget with format detection
  - INTEGRATE: File validation and error handling

CREATE gui/widgets/plot_container.py:
  - PATTERN: NiceGUI matplotlib/plotly integration patterns
  - IMPLEMENT: Dynamic plot updating container
  - INTEGRATE: sw.plot_traces(), sw.plot_probe_map() widget wrappers

CREATE gui/widgets/parameter_panel.py:
  - PATTERN: NiceGUI reactive binding patterns
  - IMPLEMENT: Dynamic parameter form generation
  - INTEGRATE: Pydantic model binding with ui.input(), ui.slider(), ui.toggle()

Task 5: Tab Implementation - Data Loading
CREATE gui/tabs/load_tab.py:
  - PATTERN: File upload and format detection UI
  - IMPLEMENT: Recording loading interface with progress tracking
  - INTEGRATE: backend/recording_manager.py methods

Task 6: Tab Implementation - Probe Configuration
CREATE gui/tabs/probe_tab.py:
  - PATTERN: ProbeInterface library integration with selection dropdowns
  - IMPLEMENT: Manufacturer/model selection with probe visualization
  - INTEGRATE: probeinterface.get_probe(), plot_probe() with NiceGUI

Task 7: Tab Implementation - Preprocessing
CREATE gui/tabs/preprocess_tab.py:
  - PATTERN: Toggle-based preprocessing option selection
  - IMPLEMENT: Parameter configuration with real-time preview
  - INTEGRATE: backend/preprocessing_manager.py methods

Task 8: Tab Implementation - Sorting
CREATE gui/tabs/sort_tab.py:
  - PATTERN: Sorter selection dropdown with parameter customization
  - IMPLEMENT: Sorting execution with progress monitoring
  - INTEGRATE: backend/sorting_manager.py methods

Task 9: Tab Implementation - Postprocessing
CREATE gui/tabs/postprocess_tab.py:
  - PATTERN: Quality metrics computation and visualization
  - IMPLEMENT: Extension selection and metric calculation
  - INTEGRATE: backend/analysis_manager.py methods

Task 10: Tab Implementation - Export
CREATE gui/tabs/save_tab.py:
  - PATTERN: Export format selection with output configuration
  - IMPLEMENT: Multi-format export (Phy, Zarr, etc.)
  - INTEGRATE: spikeinterface.exporters module

Task 11: Application Integration
MODIFY main.py:
  - INTEGRATE: All tab implementations with state management
  - IMPLEMENT: Tab navigation and state persistence
  - SETUP: Error handling and user feedback systems
```

### Per Task Pseudocode

```python
# Task 1: main.py - Application Entry Point
from nicegui import ui, app
from gui.main_window import create_main_window
from core.pipeline_state import PipelineState

def main():
    # PATTERN: NiceGUI application setup with FastAPI integration
    app.add_static_files('/uploads', 'uploads')  # File upload handling
    
    # CRITICAL: Global state management
    pipeline_state = PipelineState()
    
    @ui.page('/')
    def index():
        create_main_window(pipeline_state)
    
    ui.run(port=8080, debug=True)

# Task 3: backend/recording_manager.py
class RecordingManager:
    def __init__(self):
        self.current_recording = None
    
    def load_recording(self, file_path: Path) -> BaseRecording:
        # PATTERN: Format detection and automatic loading
        if file_path.suffix == '.meta':
            # CRITICAL: SpikeGLX format detection
            recording = se.read_spikeglx(file_path.parent)
        elif file_path.suffix == '.continuous':
            recording = se.read_openephys(file_path.parent)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # PATTERN: Validation and metadata extraction
        self.current_recording = recording
        return recording

# Task 6: gui/tabs/probe_tab.py
class ProbeTab:
    def __init__(self, pipeline_state: PipelineState):
        self.state = pipeline_state
        self.setup_ui()
    
    def setup_ui(self):
        # PATTERN: Dropdown selection with reactive updates
        with ui.column():
            self.manufacturer_select = ui.select(
                ['IMEC', 'Neuronexus', 'Cambridge Neurotech'],
                on_change=self.on_manufacturer_change
            )
            
            self.model_select = ui.select([], on_change=self.on_model_change)
            
            # CRITICAL: Probe visualization container
            self.plot_container = ui.matplotlib().classes('w-full h-96')
    
    async def on_model_change(self):
        # PATTERN: ProbeInterface integration
        probe = get_probe(self.manufacturer_select.value, self.model_select.value)
        
        # GOTCHA: Must set device channel indices
        probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
        
        # Update visualization
        self.update_probe_plot(probe)
```

### Integration Points
```yaml
DATABASE:
  - storage: "Use JSON files for pipeline state persistence"
  - pattern: "pipeline_state.json in user directory"
  
CONFIG:
  - add to: core/app_settings.py
  - pattern: "Environment variable configuration with Pydantic BaseSettings"
  
ROUTES:
  - add to: main.py FastAPI integration
  - pattern: "File upload endpoints with temporary storage cleanup"
  
PLOTTING:
  - integration: "NiceGUI matplotlib/plotly containers"
  - pattern: "Wrapper widgets for SpikeInterface visualization functions"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
uv run ruff format .                    # Auto-format code
uv run ruff check . --fix               # Auto-fix linting issues
uv run mypy gui/ core/ backend/ utils/  # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE test files following vertical slice pattern next to each module
# tests/test_core/test_pipeline_state.py
def test_pipeline_state_creation():
    """Test pipeline state model creation and validation."""
    state = PipelineState()
    assert state.current_tab == "load"
    assert not state.recording.file_path

def test_recording_state_validation():
    """Test recording state field validation."""
    with pytest.raises(ValidationError):
        RecordingState(sampling_rate=-1)  # Should fail validation

# tests/test_backend/test_recording_manager.py
def test_spikeglx_loading():
    """Test SpikeGLX format loading."""
    manager = RecordingManager()
    # Use mock data or test files from examples/
    recording = manager.load_recording(Path("test_data.meta"))
    assert recording.get_num_channels() > 0
```

```bash
# Run and iterate until passing:
uv run pytest tests/ -v --cov=gui --cov=core --cov=backend --cov=utils
# Target: 80%+ coverage, all tests passing
```

### Level 3: Integration Test
```bash
# Start the application
uv run python main.py

# Manual testing checklist:
# 1. Application loads without errors
# 2. Tab navigation works correctly
# 3. File upload processes successfully
# 4. Probe selection updates visualization
# 5. Preprocessing toggles work
# 6. State persists between tabs

# Expected: Full pipeline functionality working end-to-end
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check .`
- [ ] No type errors: `uv run mypy .`
- [ ] Application starts successfully: `uv run python main.py`
- [ ] All tabs load and function correctly
- [ ] File upload and processing works
- [ ] Probe visualization displays correctly
- [ ] Preprocessing options are functional
- [ ] Export functionality works
- [ ] Error handling provides user feedback
- [ ] Pipeline state persistence works

---

## Anti-Patterns to Avoid
- ❌ Don't create files longer than 500 lines (per CLAUDE.md)
- ❌ Don't skip SpikeInterface extension dependency order
- ❌ Don't hardcode file paths or parameters
- ❌ Don't mix sync/async patterns inappropriately
- ❌ Don't forget to set probe device channel indices
- ❌ Don't skip UV virtual environment usage
- ❌ Don't create new patterns when SpikeInterface provides them
- ❌ Don't ignore NiceGUI reactive binding patterns
- ❌ Don't skip proper error handling and user feedback