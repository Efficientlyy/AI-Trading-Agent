use pyo3::prelude::*;
use ta::indicators::{SimpleMovingAverage, ExponentialMovingAverage, RelativeStrengthIndex};
use ta::Next;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, IntoPyArray};
use ndarray::{Array1, Array2};

// Import backtesting module
pub mod backtesting;

/// Calculate Simple Moving Average (SMA) using the ta-rs library.
///
/// Args:
///     data (list): List of price data.
///     period (int): The window period for the SMA.
///
/// Returns:
///     list: List of SMA values, padded with None at the beginning.
#[pyfunction]
fn calculate_sma_rs(_py: Python<'_>, data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    // Create SMA indicator
    let mut sma_indicator = SimpleMovingAverage::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create SMA indicator: {}", e)))?;

    // Calculate SMA values
    let mut results: Vec<Option<f64>> = Vec::with_capacity(data.len());
    
    // Pad initial results with None, as SMA requires `period` data points
    for _ in 0..period.saturating_sub(1) {
        results.push(None);
    }

    // Calculate SMA for the rest of the data
    let data_len = data.len();
    for val in data {
        let sma_val = sma_indicator.next(val);
        if results.len() < data_len {
            results.push(Some(sma_val));
        }
    }

    Ok(results)
}

/// Calculate Exponential Moving Average (EMA) using the ta-rs library.
///
/// Args:
///     data (list): List of price data.
///     period (int): The window period for the EMA.
///
/// Returns:
///     list: List of EMA values, padded with None at the beginning.
#[pyfunction]
fn calculate_ema_rs(_py: Python<'_>, data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    // Create EMA indicator
    let mut ema_indicator = ExponentialMovingAverage::new(period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create EMA indicator: {}", e)))?;

    // Calculate EMA values
    let mut results: Vec<Option<f64>> = Vec::with_capacity(data.len());
    
    // Pad initial results with None, as EMA requires `period` data points
    for _ in 0..period.saturating_sub(1) {
        results.push(None);
    }

    // Calculate EMA for the rest of the data
    let data_len = data.len();
    for val in data {
        let ema_val = ema_indicator.next(val);
        if results.len() < data_len {
            results.push(Some(ema_val));
        }
    }

    Ok(results)
}

/// Calculate Moving Average Convergence Divergence (MACD) using EMA calculations.
///
/// Args:
///     data (list): List of price data.
///     fast_period (int): The fast EMA period (default: 12).
///     slow_period (int): The slow EMA period (default: 26).
///     signal_period (int): The signal EMA period (default: 9).
///
/// Returns:
///     tuple: Tuple containing (macd_line, signal_line, histogram), each as a list with None values for padding.
#[pyfunction]
fn calculate_macd_rs(
    _py: Python<'_>,
    data: Vec<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> PyResult<(Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>)> {
    // Validate parameters
    if slow_period <= fast_period {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "slow_period must be greater than fast_period",
        ));
    }

    // Create EMA indicators
    let mut fast_ema = ExponentialMovingAverage::new(fast_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create fast EMA: {}", e)))?;
    
    let mut slow_ema = ExponentialMovingAverage::new(slow_period)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create slow EMA: {}", e)))?;

    // Calculate fast and slow EMAs
    let mut fast_ema_values: Vec<Option<f64>> = vec![None; data.len()];
    let mut slow_ema_values: Vec<Option<f64>> = vec![None; data.len()];
    
    for (i, &val) in data.iter().enumerate() {
        let fast_val = fast_ema.next(val);
        let slow_val = slow_ema.next(val);
        
        if i >= fast_period - 1 {
            fast_ema_values[i] = Some(fast_val);
        }
        
        if i >= slow_period - 1 {
            slow_ema_values[i] = Some(slow_val);
        }
    }
    
    // Calculate MACD line (fast_ema - slow_ema)
    let mut macd_line: Vec<Option<f64>> = vec![None; data.len()];
    for i in 0..data.len() {
        if let (Some(fast), Some(slow)) = (fast_ema_values[i], slow_ema_values[i]) {
            macd_line[i] = Some(fast - slow);
        }
    }
    
    // Calculate signal line (EMA of MACD line)
    let mut signal_line: Vec<Option<f64>> = vec![None; data.len()];
    
    // We need to create a vector of f64 values from the MACD line for the signal EMA calculation
    // First, find the first valid MACD value
    let mut first_valid_idx = 0;
    while first_valid_idx < macd_line.len() && macd_line[first_valid_idx].is_none() {
        first_valid_idx += 1;
    }
    
    if first_valid_idx < macd_line.len() {
        // Extract valid MACD values
        let valid_macd: Vec<f64> = macd_line[first_valid_idx..]
            .iter()
            .filter_map(|&x| x)
            .collect();
            
        // Calculate signal line EMA
        if !valid_macd.is_empty() {
            let mut signal_ema = ExponentialMovingAverage::new(signal_period)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create signal EMA: {}", e)))?;
                
            let mut signal_values: Vec<Option<f64>> = vec![None; valid_macd.len()];
            
            for (i, &val) in valid_macd.iter().enumerate() {
                let signal_val = signal_ema.next(val);
                if i >= signal_period - 1 {
                    signal_values[i] = Some(signal_val);
                }
            }
            
            // Copy signal values to the correct positions in the signal_line
            let signal_offset = first_valid_idx + signal_period - 1;
            for (i, val) in signal_values.iter().enumerate().skip(signal_period - 1) {
                if signal_offset + i - (signal_period - 1) < signal_line.len() {
                    signal_line[signal_offset + i - (signal_period - 1)] = *val;
                }
            }
        }
    }
    
    // Calculate histogram (MACD line - signal line)
    let mut histogram: Vec<Option<f64>> = vec![None; data.len()];
    for i in 0..data.len() {
        if let (Some(macd), Some(signal)) = (macd_line[i], signal_line[i]) {
            histogram[i] = Some(macd - signal);
        }
    }
    
    Ok((macd_line, signal_line, histogram))
}

/// Calculate Relative Strength Index (RSI) using a custom implementation
/// that matches pandas' behavior with ewm(span=window, adjust=False).
///
/// Args:
///     data (list): List of price data.
///     period (int): The window period for the RSI (default: 14).
///
/// Returns:
///     list: List of RSI values, padded with None at the beginning.
#[pyfunction]
fn calculate_rsi_rs(_py: Python<'_>, data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    if data.len() <= period {
        // Not enough data to calculate RSI
        return Ok(vec![None; data.len()]);
    }

    // Calculate price changes
    let mut changes: Vec<f64> = Vec::with_capacity(data.len());
    changes.push(0.0); // First change is undefined, set to 0
    
    for i in 1..data.len() {
        changes.push(data[i] - data[i-1]);
    }
    
    // Separate gains and losses
    let mut gains: Vec<f64> = changes.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    let mut losses: Vec<f64> = changes.iter().map(|&x| if x < 0.0 { -x } else { 0.0 }).collect();
    
    // Calculate average gains and losses using EMA
    // This matches pandas' ewm(span=window, adjust=False) behavior
    let alpha = 2.0 / (period as f64 + 1.0);
    
    let mut avg_gains: Vec<Option<f64>> = vec![None; data.len()];
    let mut avg_losses: Vec<Option<f64>> = vec![None; data.len()];
    
    // Initialize with SMA for the first period values
    let first_gain_sum: f64 = gains[1..=period].iter().sum();
    let first_loss_sum: f64 = losses[1..=period].iter().sum();
    
    avg_gains[period] = Some(first_gain_sum / period as f64);
    avg_losses[period] = Some(first_loss_sum / period as f64);
    
    // Calculate smoothed averages for the rest of the data
    for i in period+1..data.len() {
        if let Some(prev_gain) = avg_gains[i-1] {
            avg_gains[i] = Some(alpha * gains[i] + (1.0 - alpha) * prev_gain);
        }
        
        if let Some(prev_loss) = avg_losses[i-1] {
            avg_losses[i] = Some(alpha * losses[i] + (1.0 - alpha) * prev_loss);
        }
    }
    
    // Calculate RSI
    let mut results: Vec<Option<f64>> = vec![None; data.len()];
    
    for i in period..data.len() {
        if let (Some(avg_gain), Some(avg_loss)) = (avg_gains[i], avg_losses[i]) {
            if avg_loss == 0.0 {
                // If no losses, RSI is 100
                results[i] = Some(100.0);
            } else {
                let rs = avg_gain / avg_loss;
                results[i] = Some(100.0 - (100.0 / (1.0 + rs)));
            }
        }
    }
    
    Ok(results)
}

/// Calculate lag features from a time series.
///
/// Args:
///     series (array): Input time series as a 1D array.
///     lags (list): List of lag periods.
///
/// Returns:
///     array: 2D array with each column representing a lag feature.
#[pyfunction]
fn create_lag_features_rs(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    lags: Vec<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert input to Rust array
    let series_array = series.as_array();
    let n_samples = series_array.len();
    
    // Create output array with shape (n_samples, n_lags)
    let n_lags = lags.len();
    let mut result = Array2::<f64>::from_elem((n_samples, n_lags), f64::NAN);
    
    // Fill the output array with lag values
    for (i, &lag) in lags.iter().enumerate() {
        for j in 0..n_samples {
            if j >= lag {
                result[[j, i]] = series_array[j - lag];
            }
        }
    }
    
    // Convert to Python array
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate difference features from a time series.
///
/// Args:
///     series (array): Input time series as a 1D array.
///     periods (list): List of periods for calculating differences.
///
/// Returns:
///     array: 2D array with each column representing a difference feature.
#[pyfunction]
fn create_diff_features_rs(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    periods: Vec<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert input to Rust array
    let series_array = series.as_array();
    let n_samples = series_array.len();
    
    // Create output array with shape (n_samples, n_periods)
    let n_periods = periods.len();
    let mut result = Array2::<f64>::from_elem((n_samples, n_periods), f64::NAN);
    
    // Fill the output array with difference values
    for (i, &period) in periods.iter().enumerate() {
        for j in 0..n_samples {
            if j >= period {
                result[[j, i]] = series_array[j] - series_array[j - period];
            }
        }
    }
    
    // Convert to Python array
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate percentage change features from a time series.
///
/// Args:
///     series (array): Input time series as a 1D array.
///     periods (list): List of periods for calculating percentage changes.
///
/// Returns:
///     array: 2D array with each column representing a percentage change feature.
#[pyfunction]
fn create_pct_change_features_rs(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    periods: Vec<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert input to Rust array
    let series_array = series.as_array();
    let n_samples = series_array.len();
    
    // Create output array with shape (n_samples, n_periods)
    let n_periods = periods.len();
    let mut result = Array2::<f64>::from_elem((n_samples, n_periods), f64::NAN);
    
    // Fill the output array with percentage change values
    for (i, &period) in periods.iter().enumerate() {
        for j in 0..n_samples {
            if j >= period && series_array[j - period] != 0.0 {
                result[[j, i]] = (series_array[j] - series_array[j - period]) / series_array[j - period];
            }
        }
    }
    
    // Convert to Python array
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate rolling window features from a time series.
///
/// Args:
///     series (array): Input time series as a 1D array.
///     window_sizes (list): List of window sizes for rolling calculations.
///     feature_type (str): Type of feature to calculate ('min', 'max', 'mean', 'std', 'sum').
///
/// Returns:
///     array: 2D array with each column representing a rolling window feature.
#[pyfunction]
fn create_rolling_window_features_rs(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    window_sizes: Vec<usize>,
    feature_type: String,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert input to Rust array
    let series_array = series.as_array();
    let n_samples = series_array.len();
    
    // Create output array with shape (n_samples, n_windows)
    let n_windows = window_sizes.len();
    let mut result = Array2::<f64>::from_elem((n_samples, n_windows), f64::NAN);
    
    // Fill the output array with rolling window values
    for (i, &window) in window_sizes.iter().enumerate() {
        for j in 0..n_samples {
            if j >= window - 1 {
                // Get the window of values
                let window_start = j - window + 1;
                let window_end = j + 1;
                let window_values = &series_array.slice(ndarray::s![window_start..window_end]);
                
                // Calculate the feature based on the type
                match feature_type.as_str() {
                    "min" => {
                        let min_val = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        result[[j, i]] = min_val;
                    },
                    "max" => {
                        let max_val = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        result[[j, i]] = max_val;
                    },
                    "mean" => {
                        let sum: f64 = window_values.iter().sum();
                        result[[j, i]] = sum / (window as f64);
                    },
                    "sum" => {
                        let sum: f64 = window_values.iter().sum();
                        result[[j, i]] = sum;
                    },
                    "std" => {
                        let sum: f64 = window_values.iter().sum();
                        let mean = sum / (window as f64);
                        let variance: f64 = window_values.iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>() / (window as f64);
                        result[[j, i]] = variance.sqrt();
                    },
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Unsupported feature type: {}", feature_type)
                        ));
                    }
                }
            }
        }
    }
    
    // Convert to Python array
    Ok(result.into_pyarray(py).to_owned())
}

/// AI Trading Agent Rust extensions module.
#[pymodule]
fn rust_extensions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_sma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ema_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_macd_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rsi_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_lag_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_diff_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_pct_change_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rolling_window_features_rs, m)?)?;
    
    // Add backtesting function
    m.add_function(wrap_pyfunction!(backtesting::run_backtest_rs, m)?)?;
    
    Ok(())
}