# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
# Core Development Philosophy

### KISS (Keep It Simple, Stupid)

Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)

Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles

- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and module should have one clear purpose.
- **Fail Fast**: Check for potential errors early and raise exceptions immediately when issues occur.

## 🧱 Code Structure & Modularity

### File and Function Limits

- **Never create a file longer than 500 lines of code**. If approaching this limit, refactor by splitting into modules.
- **Functions should be under 50 lines** with a single, clear responsibility.
- **Classes should be under 100 lines** and represent a single concept or entity.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Line lenght should be max 100 characters** ruff rule in pyproject.toml
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.

## 🛠️ Development Environment

### UV Package Management

This project uses UV for blazing-fast Python package and environment management.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```
# Create virtual environment
uv venv

# Sync dependencies
uv sync

# Add a package ***NEVER UPDATE A DEPENDENCY DIRECTLY IN PYPROJECT.toml***
# ALWAYS USE UV ADD
uv add requests

# Add development dependency
uv add --dev pytest ruff mypy

# Remove a package
uv remove requests

# Run commands in the environment
uv run python script.py
uv run pytest
uv run ruff check .

# Install specific Python version
uv python install 3.12

## Environment Setup

This project requires a conda/mamba environment with specific neuroscience packages. Use one of these setup methods:

### Development Commands

```bash
# Run all tests
uv run pytest
```
# Run specific tests with verbose output
uv run pytest tests/test_module.py -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Type checking
uv run mypy src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```
```bash
# Manual setup
mamba create -n hierarchy python=3.11.11
- pip install uv
- uv pip install aiohappyeyeballs==2.4.4 aiohttp==3.11.11 aiosignal==1.3.2 alabaster==1.0.0 altair==5.5.0 annotated-types==0.7.0 anyio==4.8.0 arrow==1.3.0 asciitree==0.3.3 asttokens==3.0.0 attrs==24.3.0 av==14.1.0 babel==2.17.0 backports-tarfile==1.2.0 bidsschematools==0.7.2 blosc2==3.0.0 boto3==1.36.15 botocore==1.36.15 bottleneck==1.4.2 cartopy==0.24.1 cbor2==5.6.5 certifi==2024.12.14 cffi==1.17.1 cftime==1.6.4.post1 charset-normalizer==3.4.1 ci-info==0.3.0 click==8.1.8 click-didyoumean==0.3.1 cloudpickle==3.1.0 colorama==0.4.6 comm==0.2.2 contourpy==1.3.1 cryptography==44.0.0 cuda-python==12.6.2.post1 cupy-cuda11x==13.4.0 cycler==0.12.1 dandi==0.66.5 dandischema==0.11.0 dask==2024.12.1 debugpy==1.8.12 decorator==5.1.1 decord==0.6.0 deprecated==1.2.18 dill==0.3.9 distinctipy==1.3.4 distributed==2024.12.1 dnspython==2.7.0 docstring-parser==0.16 docutils==0.21.2 elephant==1.1.1 email-validator==2.2.0 et-xmlfile==2.0.0 etelemetry==0.3.1 executing==2.1.0 fasteners==0.19 fastrlock==0.8.3 figurl==0.2.22 filelock==3.17.0 flox==0.10.0 fonttools==4.55.3 fqdn==1.5.1 frozenlist==1.5.0 fscacher==0.4.4 fsspec==2024.12.0 futures==3.0.5 h11==0.14.0 h5py==3.12.1 hdf5plugin==5.0.0 hdf5storage==0.1.19 hdmf==3.14.5 hdmf-docutils==0.4.7 hdmf-zarr==0.11.0 httpcore==1.0.7 httpx==0.28.1 huggingface-hub==0.29.2 humanize==4.11.0 idna==3.10 imageio==2.37.0 imageio-ffmpeg==0.6.0 imagesize==1.4.1 importlib-metadata==8.5.0 interleave==0.3.0 ipykernel==6.29.5 ipympl==0.9.6 ipython==8.32.0 ipywidgets==8.1.5 isodate==0.7.2 isoduration==20.11.0 jaraco-classes==3.4.0 jaraco-context==6.0.1 jaraco-functools==4.1.0 jax==0.5.0 jaxlib==0.5.0 jaxopt==0.8.3 jedi==0.19.2 jinja2==3.1.5 jmespath==1.0.1 joblib==1.4.2 jsonpointer==3.0.0 jsonschema==4.23.0 jsonschema-specifications==2024.10.1 jupyter-client==8.6.3 jupyter-core==5.7.2 jupyterlab-widgets==3.0.13 kachery-cloud==0.4.10 kaleido==0.2.1 keyring==25.6.0 keyrings-alt==5.0.2 kiwisolver==1.4.8 lazy-ops==0.2.0 llvmlite==0.43.0 locket==1.0.0 lxml==5.3.0 markdown-it-py==3.0.0 markupsafe==3.0.2 matplotlib==3.10.0 matplotlib-inline==0.1.7 mdurl==0.1.2 mearec==1.9.1 meautility==1.5.2 ml-dtypes==0.5.1 more-itertools==10.6.0 movement==0.0.23 moviepy==2.1.2 msgpack==1.1.0 mtscomp==1.0.2 multidict==6.1.0 narwhals==1.21.1 natsort==8.4.0 nc-time-axis==1.4.1 ndindex==1.9.2 ndx-dandi-icephys==0.4.0 ndx-events opencv-python scikit-learn movement==0.6  scipy==1.15.2 spyder-kernels==2.5.2

## 📋 Style & Conventions

### Python Style Guide

- **Follow PEP8** with these specific choices:
  - Line length: 100 characters (set by Ruff in pyproject.toml)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures
- **Always use type hints** for function signatures and class attributes
- **Format with `ruff format`** (faster alternative to Black)
- **Use `pydantic` v2** for data validation and settings management


## Common Commands

### Preprocessing
- Main preprocessing script: `python preprocessing/preprocess_all.py`
- Modify the `animal` and `sessions` variables at the top of the script for different datasets
- Default spike source is 'ironclust', LFP processing is disabled by default

### Environment Management
- Environment files: `env.yaml` and `project_hir.yaml`/`project_hir.yml`
- Key packages: spikeinterface, opencv-python, numpy, pandas, matplotlib, scipy, scikit-learn

## Project Architecture

### Core Structure
This is a neuroscience analysis pipeline for behavioral and neural data from escape/hunting experiments. The codebase processes electrophysiology data, behavioral tracking, and video data.

### Key Directories

**preprocessing/** - Data preprocessing pipeline
- `preprocess_all.py`: Main preprocessing entry point
- Handles spike sorting integration, behavioral data alignment, and video processing
- Configurable via animal/session parameters

**Functions/** - Shared utility modules
- `helperFunctions_2.py`: Core utility functions (file I/O, data conversion)
- `plottingFunctions.py`: Visualization utilities
- `readSGLX.py`: SpikeGLX data reader

**Analysis/** - Analysis modules organized by type:
- **Behaviour/**: Behavioral analysis (escape latency, trajectories, velocity)
- **Communication/**: Neural correlation and communication analysis
- **Dimensions/**: PCA/SVD dimensionality analysis
- **neuron_tuning/**: Single neuron response characterization
- **raw/**: Raw data visualization and raster plots

### Data Flow
1. Raw electrophysiology (SpikeGLX) and behavioral data input
2. Preprocessing via `preprocess_all.py` (spike sorting, alignment)
3. Analysis scripts operate on preprocessed data
4. Results exported to standard formats

### Configuration
- Data paths configured in CSV file: `paths.csv` or `paths-Copy.csv`
- Session-specific parameters set in preprocessing scripts
- Default spike source: ironclust
- Supports both GPU (CUDA) and CPU processing depending on environment

### Important Notes
- Scripts expect specific directory structure with animal/session organization
- Many analysis scripts are marked with importance levels (* = important, xx = unimportant)
- Functions modules must be in Python path for analysis scripts to work
- Video processing requires cv2 (handled gracefully if missing)


### Docstring Standards

Use Google-style docstrings for all public functions, classes, and modules:

```python
def calculate_discount(
    price: Decimal,
    discount_percent: float,
    min_amount: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate the discounted price for a product.

    Args:
        price: Original price of the product
        discount_percent: Discount percentage (0-100)
        min_amount: Minimum allowed final price

    Returns:
        Final price after applying discount

    Raises:
        ValueError: If discount_percent is not between 0 and 100
        ValueError: If final price would be below min_amount

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal('80.00')
    """
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes/methods**: `_leading_underscore`
- **Type aliases**: `PascalCase`
- **Enum values**: `UPPER_SNAKE_CASE`

## 🧪 Testing Strategy

### Test-Driven Development (TDD)

1. **Write the test first** - Define expected behavior before implementation
2. **Watch it fail** - Ensure the test actually tests something
3. **Write minimal code** - Just enough to make the test pass
4. **Refactor** - Improve code while keeping tests green
5. **Repeat** - One test at a time

### Testing Best Practices

```python
# Always use pytest fixtures for setup
import pytest
from datetime import datetime

@pytest.fixture
def sample_user():
    """Provide a sample user for testing."""
    return User(
        id=123,
        name="Test User",
        email="test@example.com",
        created_at=datetime.now()
    )

# Use descriptive test names
def test_user_can_update_email_when_valid(sample_user):
    """Test that users can update their email with valid input."""
    new_email = "newemail@example.com"
    sample_user.update_email(new_email)
    assert sample_user.email == new_email

# Test edge cases and error conditions
def test_user_update_email_fails_with_invalid_format(sample_user):
    """Test that invalid email formats are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        sample_user.update_email("not-an-email")
    assert "Invalid email format" in str(exc_info.value)
```

### Test Organization

- Unit tests: Test individual functions/methods in isolation
- Integration tests: Test component interactions
- End-to-end tests: Test complete user workflows
- Keep test files next to the code they test
- Use `conftest.py` for shared fixtures
- Aim for 80%+ code coverage, but focus on critical paths

## 🚨 Error Handling

### Exception Best Practices

```python
# Create custom exceptions for your domain
class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account has insufficient funds."""
    def __init__(self, required: Decimal, available: Decimal):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )

# Use specific exception handling
try:
    process_payment(amount)
except InsufficientFundsError as e:
    logger.warning(f"Payment failed: {e}")
    return PaymentResult(success=False, reason="insufficient_funds")
except PaymentError as e:
    logger.error(f"Payment error: {e}")
    return PaymentResult(success=False, reason="payment_error")

# Use context managers for resource management
from contextlib import contextmanager

@contextmanager
def database_transaction():
    """Provide a transactional scope for database operations."""
    conn = get_connection()
    trans = conn.begin_transaction()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()
```

### Logging Strategy

```python
import logging
from functools import wraps

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log function entry/exit for debugging
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

## 🔧 Configuration Management

### Environment Variables and Settings

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    api_key: str
    max_connections: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Usage
settings = get_settings()
```

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test additions or fixes

### Commit Message Format

Never include claude code, or written by claude code in commit messages

```
<type>(<scope>): <subject>

<body>

<footer>
``
Types: feat, fix, docs, style, refactor, test, chore

Example:
```

feat(auth): add two-factor authentication

- Implement TOTP generation and validation
- Add QR code generation for authenticator apps
- Update user model with 2FA fields

Closes #123

````

## 🗄️ Database Naming Standards

### Entity-Specific Primary Keys
All database tables use entity-specific primary keys for clarity and consistency:

```sql
-- ✅ STANDARDIZED: Entity-specific primary keys
sessions.session_id UUID PRIMARY KEY
leads.lead_id UUID PRIMARY KEY
messages.message_id UUID PRIMARY KEY
daily_metrics.daily_metric_id UUID PRIMARY KEY
agencies.agency_id UUID PRIMARY KEY
````

### Field Naming Conventions

```sql
-- Primary keys: {entity}_id
session_id, lead_id, message_id

-- Foreign keys: {referenced_entity}_id
session_id REFERENCES sessions(session_id)
agency_id REFERENCES agencies(agency_id)

-- Timestamps: {action}_at
created_at, updated_at, started_at, expires_at

-- Booleans: is_{state}
is_connected, is_active, is_qualified

-- Counts: {entity}_count
message_count, lead_count, notification_count

-- Durations: {property}_{unit}
duration_seconds, timeout_minutes
```

### Repository Pattern Auto-Derivation

The enhanced BaseRepository automatically derives table names and primary keys:

```python
# ✅ STANDARDIZED: Convention-based repositories
class LeadRepository(BaseRepository[Lead]):
    def __init__(self):
        super().__init__()  # Auto-derives "leads" and "lead_id"

class SessionRepository(BaseRepository[AvatarSession]):
    def __init__(self):
        super().__init__()  # Auto-derives "sessions" and "session_id"
```

**Benefits**:

- ✅ Self-documenting schema
- ✅ Clear foreign key relationships
- ✅ Eliminates repository method overrides
- ✅ Consistent with entity naming patterns
