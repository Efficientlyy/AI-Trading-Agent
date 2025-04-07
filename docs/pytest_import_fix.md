# Pytest Import Issue Fix

## Problem
`ModuleNotFoundError: No module named 'trading_engine'` when running pytest, despite correct environment setup.

## Root Cause
The codebase contained two types of problematic imports:

1. **Absolute imports referencing `src`**: Using `from src.xxx import yyy` assumes that `src` is a top-level package in the Python path.
2. **Relative imports beyond the package**: Using `from ..xxx import yyy` in a module inside `trading_engine` tries to go up two levels, which only works if the parent package (`src`) is properly recognized.

When `pytest` tried to import the test modules, it encountered these problematic imports and failed.

## Solution Implemented

1. **Fixed Import Statements**:
   - In `src/trading_engine/base_agent.py`:
     - Changed `from src.data_acquisition.data_service import DataService` to `from data_acquisition.data_service import DataService`
     - Changed `from src.common import logger` to `from common import logger`
   - In `src/trading_engine/order_manager.py`:
     - Changed `from ..common import logger` to `from common import logger`

2. **Installed Missing Dependencies**:
   - `loguru` (for logging)
   - `pyyaml` (for configuration)
   - `python-dotenv` (for environment variables)
   - `pandas` (for data analysis)
   - `ccxt` (for cryptocurrency trading)

## Best Practices for Python Imports

When working with a project that has a `src` directory structure:

1. **For imports within the same package**:
   - Use relative imports: `from .models import Order`

2. **For imports from other packages in the same project**:
   - Use absolute imports without the `src` prefix: `from common import logger`
   - Avoid `from src.common import logger`
   - Avoid relative imports that go beyond the package: `from ..common import logger`

3. **For running tests**:
   - Ensure the `src` directory is in the Python path
   - Use `conftest.py` to add `src` to `sys.path` if needed
   - Consider using an editable install (`pip install -e .`) for development

## Results
After fixing the import issues and installing the required dependencies, the tests now run successfully. There are still some test failures (9 failed, 41 passed), but these are related to implementation details rather than import issues.

## Additional Findings (April 7, 2025)

### Test Structure
- The project has a different test structure than initially assumed
- Tests are located in `tests/unit/` directory (not directly in `tests/`)
- Running with `--cache-clear` revealed 85 tests (63 passing, 22 failing)

### Missing Dependencies
- The `loguru` package was missing and needed to be installed
- Other dependencies might still be missing for specific test cases

### Project Structure Discrepancies
- The test failures initially observed (referencing `portfolio_manager.py`) were likely from cached results
- There is no actual `portfolio_manager.py` file or `test_portfolio_manager.py` in the current project
- The current failing tests are related to implementation details in:
  - Data acquisition (CCXT provider and data service)
  - Data processing (feature engineering and indicators)
  - Trading engine (models and order manager)

### Next Steps
1. Fix the remaining failing tests by addressing implementation issues
2. Ensure all dependencies are properly documented in `requirements.txt`
3. Standardize import practices across the entire codebase
4. Consider adding a `conftest.py` file to handle test-specific setup
