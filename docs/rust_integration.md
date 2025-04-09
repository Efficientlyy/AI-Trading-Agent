# Rust Integration Guide

## Overview

This document provides a comprehensive guide to the Rust integration in the AI Trading Agent project. We use Rust to accelerate performance-critical components, particularly in data processing and feature engineering, while maintaining a Python-friendly interface through PyO3 bindings.

## Why Rust?

Rust offers several advantages for our performance-critical components:

1. **Speed**: Rust provides near-C performance, making it ideal for computationally intensive tasks like feature engineering and backtesting.
2. **Memory Safety**: Rust's ownership model eliminates common bugs like null pointer dereferences and data races without a garbage collector.
3. **Concurrency**: Rust's thread safety guarantees make it easier to write parallel code.
4. **Interoperability**: Through PyO3, Rust code can be called seamlessly from Python.

## Setup Instructions

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.8+ with development headers
- A C compiler (e.g., gcc, clang, or MSVC)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Trading-Agent
   ```

2. **Build the Rust extensions**:
   ```bash
   cd rust_extensions
   maturin develop
   ```

   This will compile the Rust code and install the resulting Python package in development mode.

3. **Verify installation**:
   ```python
   python -c "import ai_trading_agent_rs; print('Rust extensions available')"
   ```

## Available Functions

The Rust extension provides the following functions, all accessible through the `src.rust_integration.features` module:

### Technical Indicators

- `calculate_sma_rs(series, window)`: Simple Moving Average
- `calculate_ema_rs(series, window)`: Exponential Moving Average
- `calculate_macd_rs(series, fast_period, slow_period, signal_period)`: MACD
- `calculate_rsi_rs(series, period)`: Relative Strength Index

### Feature Engineering

- `create_lag_features_rs(series, lags)`: Generate lag features for a time series
- `create_diff_features_rs(series, periods)`: Calculate differences over specified periods
- `create_pct_change_features_rs(series, periods)`: Calculate percentage changes
- `create_rolling_window_features_rs(series, windows, function)`: Calculate rolling window statistics

## Python Fallback Mechanism

A key feature of our implementation is the robust fallback mechanism that automatically switches to Python implementations when Rust extensions are unavailable. This ensures:

1. **Development Flexibility**: Contributors can work on the codebase without needing to compile Rust code.
2. **Cross-Platform Compatibility**: The system works even on platforms where Rust compilation might be challenging.
3. **Graceful Degradation**: Performance is reduced but functionality is maintained when Rust extensions aren't available.

### How It Works

The fallback mechanism is implemented in `src/rust_integration/features.py`:

```python
try:
    import ai_trading_agent_rs
    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust extensions not available. Falling back to Python implementations. Error: {e}")

def create_lag_features(series, lags):
    """
    Create lag features from a time series.
    
    Args:
        series: 1D array-like object containing the time series
        lags: List of lag periods
        
    Returns:
        2D array with each column representing a lag feature
    """
    if RUST_AVAILABLE:
        try:
            return ai_trading_agent_rs.create_lag_features_rs(np.array(series, dtype=np.float64), lags)
        except Exception as e:
            logger.error(f"Error using Rust implementation for lag features: {e}")
            logger.info("Falling back to Python implementation")
    
    # Python fallback implementation
    series_np = np.array(series)
    result = np.zeros((len(series_np), len(lags)))
    
    for i, lag in enumerate(lags):
        result[lag:, i] = series_np[:-lag] if lag > 0 else series_np
    
    return result
```

## Performance Comparison

Benchmarks comparing Rust vs Python implementations show significant performance improvements, especially for larger datasets:

| Function | Dataset Size | Python Time (ms) | Rust Time (ms) | Speedup |
|----------|--------------|------------------|----------------|---------|
| Lag Features | 10,000 | 12.5 | 0.8 | 15.6x |
| Diff Features | 10,000 | 14.2 | 0.9 | 15.8x |
| Rolling Window | 10,000 | 28.7 | 1.2 | 23.9x |
| RSI | 10,000 | 18.3 | 1.1 | 16.6x |

## Testing

We maintain comprehensive tests for both Rust and Python implementations to ensure correctness:

```bash
# Run Python tests (including Rust integration tests if available)
pytest tests/test_features.py

# Run Rust-specific tests
cd rust_extensions
cargo test
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'ai_trading_agent_rs'**
   - Ensure you've built the Rust extensions with `maturin develop`
   - Check that your Python environment is the same one where you installed the extensions

2. **Rust compilation errors**
   - Make sure you have the latest Rust toolchain: `rustup update`
   - Check that all dependencies in `Cargo.toml` are available

3. **Performance issues**
   - Verify that you're actually using the Rust implementations by checking logs
   - Ensure you're passing NumPy arrays with the correct data type (float64)

## Future Enhancements

We plan to expand our Rust integration to include:

1. **Backtesting Core Loop**: Accelerate the main simulation loop for faster backtesting
2. **Additional Technical Indicators**: Implement more indicators in Rust
3. **Parallel Processing**: Leverage Rust's concurrency features for multi-asset analysis
4. **Custom ML Model Inference**: Implement fast inference for trained models

## Contributing

When contributing to the Rust extensions:

1. Ensure your code has both Rust and Python implementations
2. Add comprehensive tests for both implementations
3. Update this documentation with any new functions or changes
4. Benchmark your changes to demonstrate performance improvements
