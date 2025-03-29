/*
 * Python bindings for the Rust trading engine
 */

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;

// Re-export the bindings modules
pub mod market_data;
pub mod technical;
pub mod backtesting;
pub mod sentiment;

// Version information
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python module for the crypto trading engine
#[pymodule]
pub fn crypto_trading_engine(py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize logging
    pyo3_log::init();
    
    // Register market data module
    let market_data_module = PyModule::new(py, "market_data")?;
    market_data::init_module(market_data_module)?;
    m.add_submodule(market_data_module)?;
    
    // Register technical analysis module
    let technical_module = PyModule::new(py, "technical")?;
    technical::init_module(technical_module)?;
    m.add_submodule(technical_module)?;
    
    // Register backtesting module
    let backtesting_module = PyModule::new(py, "backtesting")?;
    backtesting::init_module(backtesting_module)?;
    m.add_submodule(backtesting_module)?;
    
    // Register sentiment analysis module
    let sentiment_module = PyModule::new(py, "sentiment")?;
    sentiment::init_module(sentiment_module)?;
    m.add_submodule(sentiment_module)?;
    
    // Register version info
    m.add("__version__", VERSION)?;
    
    // Register top-level functions
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    
    Ok(())
}

/// Get the engine version
#[pyfunction]
fn version() -> String {
    VERSION.to_string()
}

/// Initialize the engine
#[pyfunction]
fn initialize() -> PyResult<bool> {
    // Perform any global initialization that might be needed
    // For now, we don't need any special initialization
    
    log::info!("Crypto trading engine initialized");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert_eq!(version(), VERSION);
    }
} 