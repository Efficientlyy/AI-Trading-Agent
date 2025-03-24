# Rust Optimization Implementation Guide

This guide provides detailed instructions for developers who want to add new Rust-optimized functions to the trading system.

## Prerequisites

- Basic knowledge of Rust and Python
- Familiarity with PyO3 for Python bindings
- Understanding of the trading system architecture

## Development Environment Setup

1. **Install Rust:**
   ```
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install Python development tools:**
   ```
   pip install maturin pytest numpy pandas
   ```

3. **Configure IDE:** We recommend VS Code with the following extensions:
   - Rust Analyzer
   - Python
   - Better TOML

## Implementation Workflow

### 1. Identify Bottlenecks

Before writing any Rust code, profile the Python implementation to identify true bottlenecks:

```python
import cProfile
cProfile.run('my_slow_function(args)', 'output.prof')

# Analyze results
import pstats
from pstats import SortKey
p = pstats.Stats('output.prof')
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
```

Focus on functions that:
- Process large volumes of data
- Perform mathematical operations
- Execute repetitive tasks
- Would benefit from parallelization

### 2. Create Rust Implementation

#### 2.1 Create Module Structure

Add your new module in the appropriate location within `rust/src/`:

```
rust/src/
  └── your_module/
      ├── mod.rs           # Module declarations
      └── your_feature.rs  # Implementation
```

Update `rust/src/lib.rs` to include your module:

```rust
pub mod your_module;
```

#### 2.2 Implement Function

In `your_feature.rs`, implement your optimized function:

```rust
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

#[pyfunction]
pub fn optimized_function(
    py: Python,
    input_data: &PyDict,
    param1: f64,
    param2: u32,
) -> PyResult<PyObject> {
    // Convert Python objects to Rust
    let rust_data = extract_data_from_python(input_data)?;
    
    // Process data (potentially in parallel)
    let results = process_data_in_parallel(rust_data, param1, param2);
    
    // Convert results back to Python
    convert_results_to_python(py, results)
}

fn process_data_in_parallel(data: Vec<Item>, param1: f64, param2: u32) -> Vec<Result> {
    // Use Rayon for parallel processing
    data.par_iter()
        .map(|item| process_item(item, param1, param2))
        .collect()
}
```

#### 2.3 Add Python Module Interface

In your module's `mod.rs`, create the Python module:

```rust
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod your_feature;
use your_feature::optimized_function;

/// Python module for your feature
#[pymodule]
pub fn your_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimized_function, m)?)?;
    Ok(())
}
```

### 3. Update Python Bridge

Create a bridge file at `src/rust_bridge/your_module_py.py`:

```python
"""
Python wrapper for the Rust implementation of your feature.
"""

import logging
from typing import Dict, List, Any, Optional

# Try to import the Rust module
try:
    from crypto_trading_engine.your_module import optimized_function as _optimized_function
    RUST_AVAILABLE = True
except ImportError:
    logging.warning("Rust module not available, falling back to Python implementation")
    RUST_AVAILABLE = False

def optimized_function(input_data: Dict[str, Any], param1: float, param2: int) -> Any:
    """
    Python interface to the Rust optimized function.
    
    Args:
        input_data: Input data dictionary
        param1: First parameter
        param2: Second parameter
        
    Returns:
        Processing results
    """
    if not RUST_AVAILABLE:
        # Fall back to Python implementation
        return _optimized_function_py(input_data, param1, param2)
        
    # Call the Rust implementation
    return _optimized_function(input_data, param1, param2)

def _optimized_function_py(input_data: Dict[str, Any], param1: float, param2: int) -> Any:
    """
    Python fallback implementation.
    """
    # Implement a fallback Python version for when Rust is unavailable
    # This should be functionally equivalent, though slower
    ...
```

Update `src/rust_bridge/__init__.py` to expose your function:

```python
# Import functions
from .your_module_py import optimized_function

# Add to __all__
__all__ = [..., 'optimized_function']
```

### 4. Update Rust Lib Interface

Update `rust/src/python/mod.rs` to include your module:

```rust
// Re-export the bindings modules
pub mod market_data;
pub mod technical;
pub mod backtesting;
pub mod your_module;  // Add this line

// ...

/// Python module for the crypto trading engine
#[pymodule]
pub fn crypto_trading_engine(py: Python, m: &PyModule) -> PyResult<()> {
    // ...
    
    // Register your module
    let your_module = PyModule::new(py, "your_module")?;
    your_module::your_module(py, your_module)?;
    m.add_submodule(your_module)?;
    
    // ...
}
```

### 5. Update Cargo.toml

Ensure you have all required dependencies in `rust/Cargo.toml`:

```toml
[dependencies]
# Core dependencies (already there)
# ...

# Add any new dependencies
your_new_dependency = "0.1.0"
```

### 6. Write Tests

#### 6.1 Rust Tests

Create tests in your Rust module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_data_in_parallel() {
        let test_data = vec![...];
        let results = process_data_in_parallel(test_data, 0.5, 100);
        assert_eq!(results.len(), test_data.len());
        // More assertions...
    }
}
```

#### 6.2 Python Tests

Create Python tests for your bridge:

```python
import pytest
from src.rust_bridge import optimized_function

def test_optimized_function():
    # Test data
    input_data = {...}
    
    # Run function
    result = optimized_function(input_data, 0.5, 100)
    
    # Verify results
    assert 'expected_field' in result
    assert result['expected_value'] == expected_value
```

#### 6.3 Performance Benchmarks

Add performance tests:

```python
import time

def test_performance():
    # Generate large test data
    large_input = generate_test_data(size=1000)
    
    # Time Rust implementation
    start = time.time()
    rust_result = optimized_function(large_input, 0.5, 100)
    rust_time = time.time() - start
    
    # Time Python implementation (force it to use Python)
    start = time.time()
    python_result = _optimized_function_py(large_input, 0.5, 100)
    python_time = time.time() - start
    
    # Verify speedup
    print(f"Python: {python_time:.4f}s, Rust: {rust_time:.4f}s, Speedup: {python_time/rust_time:.2f}x")
    assert rust_time < python_time / 2  # Should be at least 2x faster
```

### 7. Build and Test

Build the Rust library:

```bash
cd rust
cargo build --release
```

Generate Python wheels:

```bash
cd rust
maturin build --release
```

Install the wheel:

```bash
pip install ./target/wheels/crypto_trading_engine-*.whl --force-reinstall
```

Run tests:

```bash
pytest tests/your_module_tests
```

## Best Practices

### Data Conversion

When converting between Python and Rust:

1. **Extract Early, Return Late:** Convert Python objects to Rust structures as early as possible, and convert back to Python objects as late as possible.

2. **Minimize Copies:** Use references where possible to avoid unnecessary copies.

3. **Batch Extraction:** Extract collections (lists, dicts) in a single operation rather than item by item.

### Parallelization

When using Rayon for parallelization:

1. **Overhead Awareness:** Parallelization has overhead; only use for computations that take enough time to benefit.

2. **Adjust Chunk Size:** For very large collections, control chunk size to balance load:
   ```rust
   collection.par_chunks(chunk_size).map(|chunk| process_chunk(chunk)).collect()
   ```

3. **Avoid Locks:** Design your parallel code to avoid locks and shared mutable state.

### Error Handling

1. **Graceful Fallback:** Always ensure your Python bridge gracefully falls back to Python implementation.

2. **Detailed Error Messages:** Provide detailed error messages that help diagnose issues.

3. **Validation:** Validate all inputs before processing to avoid unexpected behavior.

## Troubleshooting

### Common Issues

1. **Segmentation Faults:** Often caused by improper memory management. Check for:
   - Invalid pointers
   - Use after free
   - Out of bounds access

2. **Memory Leaks:** Use Rust's memory profiling tools:
   ```
   cargo install cargo-valgrind
   cargo valgrind --release
   ```

3. **Type Conversion Errors:** Ensure Python-to-Rust and Rust-to-Python type conversions are correctly handled.

4. **Thread Safety:** Ensure all Rust code is thread-safe when using `par_iter()`.

### Debugging Tips

1. **Print Statements:** Add debug prints in Rust:
   ```rust
   eprintln!("Debug: {:?}", variable);
   ```

2. **GDB/LLDB:** For complex issues, use a debugger:
   ```
   rust-gdb target/release/libcrypto_trading_engine.so
   ```

3. **Python Traceback:** For PyO3 issues, Python's traceback shows where the error originates.

## Example: Complete Module

### Rust Code (rust/src/example/mod.rs)

```rust
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Expose module functions
pub fn example(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_calculation, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn fast_calculation(py: Python, data: Vec<f64>) -> PyResult<PyObject> {
    // Process data in parallel
    let result = data.par_iter()
        .map(|x| x.powi(2) + x * 2.0 + 1.0)
        .sum::<f64>();
    
    // Return as Python float
    Ok(result.to_object(py))
}
```

### Python Bridge (src/rust_bridge/example_py.py)

```python
import logging
import numpy as np
from typing import List, Union

try:
    from crypto_trading_engine.example import fast_calculation as _fast_calculation
    RUST_AVAILABLE = True
except ImportError:
    logging.warning("Rust example module not available")
    RUST_AVAILABLE = False

def fast_calculation(data: Union[List[float], np.ndarray]) -> float:
    """Fast calculation using Rust optimization."""
    if not RUST_AVAILABLE:
        return _fast_calculation_py(data)
    
    # Convert numpy array to list if needed
    if isinstance(data, np.ndarray):
        data = data.tolist()
    
    return _fast_calculation(data)

def _fast_calculation_py(data: Union[List[float], np.ndarray]) -> float:
    """Python fallback implementation."""
    return sum(x**2 + x * 2 + 1 for x in data)
```

## Conclusion

By following this guide, you can effectively add Rust optimizations to performance-critical parts of the trading system. Remember to:

1. Focus on real bottlenecks
2. Use parallelism effectively
3. Provide proper fallbacks
4. Write comprehensive tests

This approach allows us to maintain the flexibility of Python with the performance of Rust where it matters most.