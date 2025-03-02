# Rust Integration Guide

This guide provides detailed information on how to work with the Rust components in the AI Crypto Trading System.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Building Rust Components](#building-rust-components)
5. [Testing Rust Components](#testing-rust-components)
6. [Extending with New Components](#extending-with-new-components)
7. [Troubleshooting](#troubleshooting)

## Overview

The AI Crypto Trading System uses Rust for performance-critical components to significantly improve execution speed while maintaining Python's flexibility for the overall architecture. This hybrid approach provides the best of both worlds:

- **Python**: Used for high-level application logic, strategy development, and data processing
- **Rust**: Used for performance-critical operations that require low-level optimization

## Prerequisites

To work with the Rust components, you need:

- **Rust**: Latest stable version (install via [rustup](https://rustup.rs/))
- **Python**: 3.8+ with development headers
- **Cargo**: The Rust package manager (included with rustup)
- **maturin** or **setuptools-rust**: For building Python extensions (optional)

## Architecture

The Rust integration architecture has these key components:

1. **Rust Core Library**: Contains all Rust implementations in `rust/src/`
   - `market_data/`: Market data structures and functions
   - `technical/`: Technical indicators and analysis tools
   - `backtesting/`: High-performance backtesting engine
   - `execution/`: Order execution components

2. **PyO3 Bindings**: Python bindings for the Rust code in `rust/src/python/`
   - Exposes Rust functions and types to Python
   - Handles type conversions between Rust and Python

3. **Python Bridge**: Python code that provides a clean API in `src/rust_bridge/`
   - Handles importing the Rust library
   - Provides fallback to pure Python implementations when Rust is unavailable

## Building Rust Components

### Initial Setup

Clone the repository and install development dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-crypto-trading-system.git
cd ai-crypto-trading-system

# Install Python dependencies
pip install -r requirements-dev.txt

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Building the Rust Library

```bash
# Navigate to the Rust directory
cd rust

# Build in debug mode
cargo build

# Build in release mode (recommended for production)
cargo build --release

# Copy the built library to the Python bridge directory
cp target/release/libcrypto_trading_engine.* ../src/rust_bridge/
```

## Testing Rust Components

### Running Unit Tests

The Rust library includes comprehensive unit tests:

```bash
# Run all Rust tests
cd rust
cargo test

# Run tests for a specific module
cargo test --package crypto_trading_engine --lib technical::indicators
```

### Running Integration Tests

To test the Python-Rust integration:

```bash
# Run the integration test script
python tests/test_rust_integration.py

# Run the performance benchmark
python examples/rust_performance_demo.py
```

### Verifying Rust Installation

To verify that the Rust components are properly installed and accessible from Python:

```python
from src.rust_bridge import is_rust_available, version

# Check if Rust is available
if is_rust_available():
    print(f"Rust components available, version: {version()}")
else:
    print("Rust components not available, using Python fallbacks")
```

## Extending with New Components

To add new Rust-accelerated functionality:

### 1. Implement the Component in Rust

Create a new Rust module in the appropriate directory:

```rust
// rust/src/technical/volatility.rs

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Calculate the Average True Range (ATR)
pub fn calculate_atr(
    high_prices: &[Decimal], 
    low_prices: &[Decimal], 
    close_prices: &[Decimal], 
    period: usize
) -> Vec<Decimal> {
    // Implementation goes here
}
```

### 2. Add it to the Module Exports

Update the module's `mod.rs` file:

```rust
// rust/src/technical/mod.rs

pub mod indicators;
pub mod volatility;  // Add the new module

// Re-export public items
pub use indicators::{MovingAverage, calculate_sma, calculate_ema};
pub use volatility::calculate_atr;  // Re-export the new function
```

### 3. Create PyO3 Bindings

Add Python bindings in the appropriate Python module:

```rust
// rust/src/python/technical.rs

// In the init_module function:
m.add_function(wrap_pyfunction!(calc_atr, m)?)?;

// Add the binding function
#[pyfunction]
fn calc_atr(
    high_prices: Vec<f64>,
    low_prices: Vec<f64>,
    close_prices: Vec<f64>,
    period: usize
) -> PyResult<Vec<f64>> {
    // Convert Python types to Rust types and call the Rust function
}
```

### 4. Create a Python Wrapper

Create a Python wrapper in the appropriate module:

```python
# src/analysis_agents/technical/volatility_rust.py

from src.rust_bridge import Technical, is_rust_available

def atr(high_prices, low_prices, close_prices, period):
    """
    Calculate Average True Range (ATR)
    
    Uses Rust implementation when available,
    with automatic fallback to Python.
    """
    return Technical.atr(high_prices, low_prices, close_prices, period)
```

## Troubleshooting

### Common Issues

#### Rust Library Not Found

```
ImportError: cannot import name 'crypto_trading_engine' from 'src.rust_bridge'
```

**Solution**: Ensure the library is built and copied to the correct location:

```bash
cd rust
cargo build --release
cp target/release/libcrypto_trading_engine.* ../src/rust_bridge/
```

#### Type Conversion Errors

```
TypeError: incompatible function arguments
```

**Solution**: Check that your data types match what the Rust functions expect. Common issues include:

- Passing integers when floats are expected
- Using NumPy arrays instead of Python lists
- Not handling None/null values properly

#### Build Errors

```
error[E0308]: mismatched types
```

**Solution**: Rust has a strict type system. Ensure your Rust code has the correct types and properly handles all edge cases.

### Getting Help

- Check the logs for detailed error messages
- Look at the Rust documentation comments for function signatures
- Review the PyO3 documentation for Python binding issues
- File an issue on the project repository if you encounter persistent problems

## Performance Optimization Tips

To get the best performance from Rust components:

1. **Batch Processing**: When possible, process data in batches rather than element by element
2. **Avoid Frequent Type Conversions**: Minimize conversions between Python and Rust
3. **Use Release Builds**: Always use `--release` for production builds
4. **Benchmark Your Code**: Use the benchmarking tools to verify performance improvements 