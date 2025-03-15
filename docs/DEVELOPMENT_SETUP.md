# Development Environment Setup Guide

This guide will help you set up your development environment for the AI Crypto Trading System project.

## Prerequisites

- Python 3.8+ (3.8 or 3.9 recommended)
- Visual Studio Code
- Git

## Environment Setup

### 1. Clone the Repository

```
git clone https://github.com/yourusername/AI-Trading-Agent.git
cd AI-Trading-Agent
```

### 2. Set up Python Virtual Environment

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
pip install -e .
```

> **Note about TA-Lib:** If you encounter issues installing TA-Lib from requirements.txt on Windows, download the appropriate wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib and install it manually with `pip install <downloaded_file>`.

### 4. Install Development Dependencies (Optional)

```
pip install -r requirements-dev.txt
```

## VS Code Configuration

### 1. Recommended Extensions

#### Core Development Extensions
- **Python** (ms-python.python) - Python language support
- **Pylance** (ms-python.vscode-pylance) - Fast, feature-rich language server
- **Python Docstring Generator** (njpwerner.autodocstring) - Generate Python docstrings
- **Jupyter** (ms-toolsai.jupyter) - Jupyter notebook support
- **Black Formatter** (ms-python.black-formatter) - Format code using Black
- **isort** (ms-python.isort) - Sort imports
- **YAML** (redhat.vscode-yaml) - YAML support for configuration files
- **Markdown All in One** (yzhang.markdown-all-in-one) - Better markdown editing
- **Git Graph** (mhutchie.git-graph) - Git repository visualization
- **GitLens** (eamodio.gitlens) - Git capabilities in VS Code

#### Specialized Extensions for AI Trading Projects
- **SQLite Viewer** (alexcvzz.vscode-sqlite) - View SQLite databases for market data storage
- **REST Client** (humao.rest-client) - Test and interact with crypto exchange APIs
- **Error Lens** (usernamehw.errorlens) - Shows errors and warnings inline in your code
- **Better TOML** (bungcip.better-toml) - For Rust-related configuration in the `rust/` directory
- **Python Test Explorer** (littlefoxteam.vscode-python-test-adapter) - Graphical interface for tests
- **vscode-icons** (vscode-icons-team.vscode-icons) - Better file icons for navigating complex project structure
- **Code Spell Checker** (streetsidesoftware.code-spell-checker) - Prevents documentation typos
- **IntelliCode** (visualstudioexptteam.vscodeintellicode) - AI-assisted code completions

To manually install all extensions, open VS Code, press `Ctrl+P`, and run:
```
ext install ms-python.python ms-python.vscode-pylance njpwerner.autodocstring ms-toolsai.jupyter ms-python.black-formatter ms-python.isort redhat.vscode-yaml yzhang.markdown-all-in-one mhutchie.git-graph eamodio.gitlens alexcvzz.vscode-sqlite humao.rest-client usernamehw.errorlens bungcip.better-toml littlefoxteam.vscode-python-test-adapter vscode-icons-team.vscode-icons streetsidesoftware.code-spell-checker visualstudioexptteam.vscodeintellicode
```

### 2. VS Code Settings

A `.vscode/settings.json` file has already been configured with optimal settings for this project. These settings include:

- Type checking with Pylance
- Linting with Flake8 and Pylint
- Formatting with Black
- Automatic import organization
- Pytest test discovery
- Workspace-specific file exclusions

## Configuration Management

The system uses a robust configuration management system that supports:

1. **YAML Configuration Files**: Default settings are in `config/system.yaml`
2. **Environment Variables**: Override config values using uppercase underscore-separated keys
   - Example: `TEST_NESTED_VALUE=true` will override `test.nested.value` in the configuration
   - The system automatically converts values to the appropriate types:
     - Boolean strings (true/false/yes/no/1/0) are converted to boolean values
     - Numeric strings are converted to integers or floats
     - Other values are kept as strings

3. **Runtime Configuration**: Programmatically set configuration values using the Config API

### Configuration Best Practices

1. **Use Default Values**: Always provide default values when getting configuration options
   ```python
   log_level = config.get("system.logging.level", "INFO")
   ```

2. **Type Conversion**: The configuration system will attempt to preserve and convert types properly, but be explicit when needed
   ```python
   # If expecting a boolean, check explicitly
   is_enabled = bool(config.get("feature.enabled", False))
   ```

3. **Environment Variables**: Use environment variables for deployment-specific settings and sensitive information
   ```
   # .env file or environment variables
   EXCHANGE_API_KEY=your-secret-key
   EXCHANGE_API_SECRET=your-secret-value
   ```

## Testing Best Practices

When writing tests for the trading system, follow these guidelines:

1. **Mocking Time**: When testing code that uses datetime functions, properly mock them in both the test and the target module:
   ```python
   # Example of proper datetime mocking
   now = datetime.datetime.utcnow()
   with mock.patch('datetime.datetime') as mock_datetime:
       mock_datetime.utcnow.return_value = now
       # Also patch the module under test
       with mock.patch('src.some_module.datetime') as mock_module_datetime:
           mock_module_datetime.utcnow.return_value = now
           # Call the function that uses datetime
   ```

2. **Configuration Tests**: Be aware that environment variable overrides are converted to their appropriate types:
   - `"true"`, `"yes"`, `"1"`, `"t"`, `"y"` → `True`
   - `"false"`, `"no"`, `"0"`, `"f"`, `"n"` → `False`
   - Integer strings → `int`
   - Floating point strings → `float`

3. **Test Isolation**: Each test should be isolated and not depend on the state created by other tests

4. **Test Coverage**: Aim for high test coverage, particularly for core components and critical business logic

## Testing the Logging System

The logging system includes several advanced features that require specific setup for testing:

### 1. Log Query Language Testing

```python
from src.common.log_query import LogQuery

# Create a query
query = 'level = "error" AND component = "api"'
log_query = LogQuery(query)

# Search logs
results = log_query.search_directory(
    directory="logs",
    pattern="*.log*",
    limit=1000
)
```

### 2. Log Replay Testing

```python
from src.common.log_replay import LogReplay

def handle_error(entry):
    print(f"Error found: {entry['message']}")

# Create replay instance
replay = LogReplay(
    handlers={"error": handle_error},
    filters={"component": "api"}
)

# Replay logs
replay.replay_from_file(
    "logs/app.log",
    speed_factor=2.0  # 2x speed
)
```

### 3. Health Monitoring Testing

```python
from src.common.health_monitoring import (
    HealthCheck,
    HealthMonitor,
    MetricType
)

# Create health check
def check_api():
    return True  # API is healthy

monitor = HealthMonitor()
monitor.add_check(
    HealthCheck(
        name="api_health",
        check_func=check_api,
        interval=30
    )
)

# Add custom metric
monitor.add_metric(
    name="requests.latency",
    metric_type=MetricType.HISTOGRAM,
    thresholds={
        "warning": 1000,
        "critical": 5000
    }
)

# Start monitoring
monitor.start()
```

### Running Logging Tests

To run the logging system tests:

```bash
# Run all logging tests
pytest tests/common/test_log_query.py tests/common/test_log_replay.py tests/common/test_health_monitoring.py

# Run specific test file
pytest tests/common/test_log_query.py

# Run with coverage
pytest --cov=src.common.logging tests/common/
```

### Test Environment Setup

1. **Required Environment Variables**:
```bash
# Windows
set PYTHONPATH=.
set LOG_LEVEL=DEBUG
set LOG_DIR=./logs

# Linux/MacOS
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
export LOG_DIR=./logs
```

2. **Create Log Directory**:
```bash
mkdir logs
```

3. **Test Data Generation**:
```python
from src.common.logging import get_logger
logger = get_logger()

# Generate test logs
logger.info("Test info message")
logger.error("Test error message")
```

### Common Testing Scenarios

1. **Query Language Testing**:
- Test complex queries with multiple conditions
- Test different operators (=, !=, >, >=, <, <=, ~, !~)
- Test with various data types (strings, numbers, timestamps)

2. **Log Replay Testing**:
- Test with compressed and uncompressed logs
- Test time-based replay with different speed factors
- Test filtering by request ID and component

3. **Health Monitoring Testing**:
- Test health check dependencies
- Test metric thresholds and alerts
- Test system metrics collection

### Troubleshooting Tests

1. **Module Not Found Errors**:
- Ensure PYTHONPATH includes project root
- Check import statements use correct paths

2. **Permission Errors**:
- Ensure write access to log directory
- Check file permissions on log files

3. **Test Failures**:
- Check log level configuration
- Verify test data exists
- Check for timing-sensitive tests

## Logging Dashboard

Start the dashboard with:
```bash
python -m src.dashboard.log_dashboard
```

Access at: http://localhost:8050

Features:
- Real-time log analytics
- Query language support
- Historical log replay
- System health monitoring

## Project Structure

See `docs/PROJECT_SUMMARY.md` for a comprehensive overview of the project structure and components.

## Running the Application

### 1. Standard Execution

```
python -m src.main
```

### 2. Running the Dashboard

```
python dashboard.py
```

### 3. Running Tests

```
pytest
```

## Development Workflow

1. **Create a Branch**: For each new feature or bugfix, create a new branch
2. **Implement Changes**: Make your changes following the project coding standards
3. **Run Tests**: Ensure all tests pass
4. **Format Code**: Format your code using Black and isort
5. **Submit Pull Request**: Submit a PR with your changes

## Debugging

1. **Local Debugging**: VS Code's debugging configurations are set up for Python
2. **Logging**: Use the configured logging system with `get_logger()` function

## Common Tasks

### Adding a New Strategy

1. Create a new strategy file in `src/strategy/strategies/`
2. Implement the strategy class inheriting from `BaseStrategy`
3. Register the strategy in `src/strategy/registry.py`

### Adding a New Exchange Connector

1. Create a new connector file in `src/execution/exchanges/`
2. Implement the exchange class inheriting from `BaseExchange`
3. Register the exchange in `src/execution/registry.py`

### Adding a New Sentiment Source

1. Create a new sentiment agent in `src/analysis_agents/sentiment/`
2. Implement the agent class inheriting from `BaseSentimentAgent`
3. Register the agent in `src/analysis_agents/sentiment_analysis_manager.py`

## Documentation Guidelines

- Use Google-style docstrings for all public functions and classes
- Keep documentation up-to-date with code changes
- Add examples for complex functionality

## Performance Considerations

- Use profiling tools for performance-critical code
- Consider moving performance-critical components to Rust
- Use async patterns for I/O-bound operations
