use pyo3::prelude::*;

// Import feature engineering modules
mod features;
mod moving_averages;
mod advanced_features;
// mod patterns;
mod backtesting;
mod c_api;
mod lag_features;

pub use features::{register as register_features, create_lag_features_rs, create_diff_features_rs, create_pct_change_features_rs, create_rolling_window_features_rs};
pub use moving_averages::{register as register_moving_averages, create_ema_features_rs, calculate_sma_rust_direct};
pub use advanced_features::{register as register_advanced_features, create_bollinger_bands_rs, create_rsi_features_rs, create_macd_features_rs, calculate_fibonacci_retracement_rs, calculate_pivot_points_rs, calculate_volume_profile_rs, create_stochastic_oscillator_rs};
// pub use patterns::{register as register_patterns};
// pub use backtesting::{register as register_backtesting};

/// Python module for rust extensions
#[pymodule]
fn ai_trading_agent_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register functions/classes from submodules
    features::register(m)?;
    moving_averages::register(m)?;
    advanced_features::register(m)?;
    // patterns::register(_py, m)?;
    // backtesting::register(_py, m)?;

    // Add version (optional, but good practice)
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
