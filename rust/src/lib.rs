/*
 * Main entry point for the Rust trading engine
 */

// Set up logging
#[macro_use]
extern crate log;

// Re-export modules
pub mod market_data;
pub mod technical;
pub mod backtesting;
pub mod execution;
pub mod sentiment;
pub mod python;

// Library information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Get library version information
pub fn version() -> String {
    VERSION.to_string()
}

/// Initialize the Rust engine
pub fn initialize() -> Result<(), String> {
    info!("Initializing Rust crypto trading engine v{}", VERSION);
    // Future: Initialize global resources like thread pools
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
} 