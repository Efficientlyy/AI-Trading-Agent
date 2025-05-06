use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::collections::VecDeque;

/// Create Bollinger Bands features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     windows: List of window sizes for calculating Bollinger Bands
///     num_std: Number of standard deviations for the bands (default: 2.0)
///
/// Returns:
///     List of lists: Each inner list contains [middle_band, upper_band, lower_band] for each window size
#[pyfunction]
pub fn create_bollinger_bands_rs(
    py: Python,
    series: &PyList,
    windows: Vec<i32>,
    num_std: Option<f64>,
) -> PyResult<Py<PyList>> {
    // Input validation
    if windows.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "windows must be a non-empty list of integers",
        ));
    }

    let num_std_val = num_std.unwrap_or(2.0);

    // Convert Python list to Rust vector
    let mut series_vec = Vec::new();
    for item in series.iter() {
        let value = item.extract::<f64>()?;
        series_vec.push(value);
    }

    if series_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    let n_samples = series_vec.len();
    let n_windows = windows.len();

    // Create a 3D vector to store the result (n_samples x n_windows x 3)
    // For each sample and window, we store [middle_band, upper_band, lower_band]
    let mut result = vec![vec![vec![f64::NAN; 3]; n_windows]; n_samples];

    // Calculate Bollinger Bands for each window size in parallel
    windows.par_iter().enumerate().for_each(|(i, &window)| {
        if window <= 0 {
            // Skip invalid windows (will be handled by error checking below)
            return;
        }

        let window_usize = window as usize;
        
        // For each window size, calculate the Bollinger Bands for each time point
        for j in window_usize - 1..n_samples {
            // Get the window data
            let window_data = &series_vec[j - (window_usize - 1)..=j];
            
            // Calculate mean (middle band)
            let sum: f64 = window_data.iter().sum();
            let mean = sum / window_usize as f64;
            
            // Calculate standard deviation
            let variance: f64 = window_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window_usize as f64;
            let std_dev = variance.sqrt();
            
            // Calculate bands
            let middle_band = mean;
            let upper_band = mean + num_std_val * std_dev;
            let lower_band = mean - num_std_val * std_dev;
            
            // Store results
            result[j][i][0] = middle_band;
            result[j][i][1] = upper_band;
            result[j][i][2] = lower_band;
        }
    });

    // Check for invalid window sizes
    for &window in &windows {
        if window <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Window sizes must be positive integers, got {}", window),
            ));
        }
    }

    // Convert result to Python list of lists of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, &[]);
        
        for bands in row.iter() {
            let py_bands = PyList::new(py, bands);
            py_row.append(py_bands)?;
        }
        
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Create Relative Strength Index (RSI) features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     periods: List of periods for calculating RSI
///
/// Returns:
///     List of lists: Each inner list represents an RSI feature for a specific period
#[pyfunction]
pub fn create_rsi_features_rs(
    py: Python,
    series: &PyList,
    periods: Vec<i32>,
) -> PyResult<Py<PyList>> {
    // Input validation
    if periods.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "periods must be a non-empty list of integers",
        ));
    }

    // Convert Python list to Rust vector
    let mut series_vec = Vec::new();
    for item in series.iter() {
        let value = item.extract::<f64>()?;
        series_vec.push(value);
    }

    if series_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    let n_samples = series_vec.len();
    let n_features = periods.len();

    // Create a 2D vector to store the result
    let mut result = vec![vec![f64::NAN; n_features]; n_samples];

    // Calculate RSI for each period
    for (i, &period) in periods.iter().enumerate() {
        if period <= 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("RSI periods must be greater than 1, got {}", period),
            ));
        }

        let period_usize = period as usize;
        
        // Calculate price changes
        let mut gains = vec![0.0; n_samples];
        let mut losses = vec![0.0; n_samples];
        
        for j in 1..n_samples {
            let change = series_vec[j] - series_vec[j - 1];
            if change > 0.0 {
                gains[j] = change;
            } else {
                losses[j] = -change;  // Convert to positive value
            }
        }
        
        // Calculate initial average gain and loss
        if n_samples > period_usize {
            let mut avg_gain = 0.0;
            let mut avg_loss = 0.0;
            
            for j in 1..=period_usize {
                avg_gain += gains[j];
                avg_loss += losses[j];
            }
            
            avg_gain /= period as f64;
            avg_loss /= period as f64;
            
            // Calculate first RSI
            if avg_loss == 0.0 {
                result[period_usize][i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[period_usize][i] = 100.0 - (100.0 / (1.0 + rs));
            }
            
            // Calculate subsequent RSIs using smoothed averages
            for j in period_usize + 1..n_samples {
                // Update average gain and loss using smoothing
                avg_gain = ((avg_gain * (period as f64 - 1.0)) + gains[j]) / period as f64;
                avg_loss = ((avg_loss * (period as f64 - 1.0)) + losses[j]) / period as f64;
                
                // Calculate RSI
                if avg_loss == 0.0 {
                    result[j][i] = 100.0;
                } else {
                    let rs = avg_gain / avg_loss;
                    result[j][i] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, row);
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Create Moving Average Convergence Divergence (MACD) features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     fast_periods: List of fast EMA periods
///     slow_periods: List of slow EMA periods
///     signal_periods: Number of periods for the signal line
///
/// Returns:
///     List of lists: Each inner list contains [macd_line, signal_line, histogram] for each fast-slow period combination
#[pyfunction]
pub fn create_macd_features_rs(
    py: Python,
    series: &PyList,
    fast_periods: Vec<i32>,
    slow_periods: Vec<i32>,
    signal_periods: i32,
) -> PyResult<Py<PyList>> {
    // Input validation
    if fast_periods.is_empty() || slow_periods.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fast_periods and slow_periods must be non-empty lists of integers",
        ));
    }

    if signal_periods <= 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("signal_periods must be a positive integer, got {}", signal_periods),
        ));
    }

    // Convert Python list to Rust vector
    let mut series_vec = Vec::new();
    for item in series.iter() {
        let value = item.extract::<f64>()?;
        series_vec.push(value);
    }

    if series_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    let n_samples = series_vec.len();
    let n_combinations = fast_periods.len() * slow_periods.len();

    // Create a 3D vector to store the result (n_samples x n_combinations x 3)
    // For each sample and combination, we store [macd_line, signal_line, histogram]
    let mut result = vec![vec![vec![f64::NAN; 3]; n_combinations]; n_samples];

    // Helper function to calculate EMA
    let calculate_ema = |data: &[f64], period: i32| -> Vec<f64> {
        let period_f64 = period as f64;
        let alpha = 2.0 / (period_f64 + 1.0);
        let mut ema = vec![f64::NAN; data.len()];
        
        // Initialize EMA with the first value
        ema[0] = data[0];
        
        // Calculate EMA for each point
        for i in 1..data.len() {
            ema[i] = data[i] * alpha + ema[i-1] * (1.0 - alpha);
        }
        
        ema
    };

    // Calculate MACD for each combination of fast and slow periods
    let mut combination_idx = 0;
    
    for &fast_period in &fast_periods {
        if fast_period <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Fast periods must be positive integers, got {}", fast_period),
            ));
        }
        
        for &slow_period in &slow_periods {
            if slow_period <= 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Slow periods must be positive integers, got {}", slow_period),
                ));
            }
            
            if fast_period >= slow_period {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Fast period ({}) must be less than slow period ({})", fast_period, slow_period),
                ));
            }
            
            // Calculate fast and slow EMAs
            let fast_ema = calculate_ema(&series_vec, fast_period);
            let slow_ema = calculate_ema(&series_vec, slow_period);
            
            // Calculate MACD line (fast EMA - slow EMA)
            let mut macd_line = vec![f64::NAN; n_samples];
            for i in 0..n_samples {
                if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
                    macd_line[i] = fast_ema[i] - slow_ema[i];
                }
            }
            
            // Calculate signal line (EMA of MACD line)
            let signal_line = calculate_ema(&macd_line, signal_periods);
            
            // Calculate histogram (MACD line - signal line)
            let mut histogram = vec![f64::NAN; n_samples];
            for i in 0..n_samples {
                if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                    histogram[i] = macd_line[i] - signal_line[i];
                }
            }
            
            // Store results
            for i in 0..n_samples {
                result[i][combination_idx][0] = macd_line[i];
                result[i][combination_idx][1] = signal_line[i];
                result[i][combination_idx][2] = histogram[i];
            }
            
            combination_idx += 1;
        }
    }

    // Convert result to Python list of lists of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, &[]);
        
        for components in row.iter() {
            let py_components = PyList::new(py, components);
            py_row.append(py_components)?;
        }
        
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Calculate Fibonacci retracement levels from high and low points.
///
/// Args:
///     high_prices: List of high prices
///     low_prices: List of low prices
///     is_uptrend: Whether the trend is up (true) or down (false)
///     levels: List of Fibonacci levels to calculate (default: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
///
/// Returns:
///     List of retracement levels for each price point
#[pyfunction]
pub fn calculate_fibonacci_retracement_rs(
    py: Python,
    high_prices: &PyList,
    low_prices: &PyList,
    is_uptrend: bool,
    levels: Option<Vec<f64>>,
) -> PyResult<Py<PyList>> {
    // Input validation
    if high_prices.len() != low_prices.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "high_prices and low_prices must have the same length",
        ));
    }

    // Default Fibonacci levels if not provided
    let fib_levels = levels.unwrap_or_else(|| vec![0.236, 0.382, 0.5, 0.618, 0.786, 1.0]);
    
    // Convert Python lists to Rust vectors
    let mut high_vec = Vec::new();
    let mut low_vec = Vec::new();
    
    for (high_item, low_item) in high_prices.iter().zip(low_prices.iter()) {
        let high_value = high_item.extract::<f64>()?;
        let low_value = low_item.extract::<f64>()?;
        high_vec.push(high_value);
        low_vec.push(low_value);
    }

    if high_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "price lists must be non-empty",
        ));
    }

    let n_samples = high_vec.len();
    let n_levels = fib_levels.len();

    // Create a 2D vector to store the result (n_samples x n_levels)
    let mut result = vec![vec![f64::NAN; n_levels]; n_samples];

    // Find the highest high and lowest low in the entire dataset
    let mut highest_high = f64::NEG_INFINITY;
    let mut lowest_low = f64::INFINITY;
    
    for i in 0..n_samples {
        if high_vec[i] > highest_high {
            highest_high = high_vec[i];
        }
        if low_vec[i] < lowest_low {
            lowest_low = low_vec[i];
        }
    }

    // Calculate the range
    let range = highest_high - lowest_low;

    // Calculate retracement levels based on trend direction
    for i in 0..n_samples {
        for (j, &level) in fib_levels.iter().enumerate() {
            if is_uptrend {
                // For uptrend: high - (range * level)
                result[i][j] = highest_high - (range * level);
            } else {
                // For downtrend: low + (range * level)
                result[i][j] = lowest_low + (range * level);
            }
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, row);
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Calculate pivot points for price data.
///
/// Args:
///     high_prices: List of high prices
///     low_prices: List of low prices
///     close_prices: List of closing prices
///     pivot_type: Type of pivot points to calculate ("standard", "fibonacci", "camarilla", "woodie")
///
/// Returns:
///     List of lists: Each inner list contains pivot points for a specific type
#[pyfunction]
pub fn calculate_pivot_points_rs(
    py: Python,
    high_prices: &PyList,
    low_prices: &PyList,
    close_prices: &PyList,
    pivot_type: &str,
) -> PyResult<Py<PyList>> {
    // Input validation
    if high_prices.len() != low_prices.len() || high_prices.len() != close_prices.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "high_prices, low_prices, and close_prices must have the same length",
        ));
    }

    // Convert Python lists to Rust vectors
    let mut high_vec = Vec::new();
    let mut low_vec = Vec::new();
    let mut close_vec = Vec::new();
    
    for ((high_item, low_item), close_item) in high_prices.iter().zip(low_prices.iter()).zip(close_prices.iter()) {
        let high_value = high_item.extract::<f64>()?;
        let low_value = low_item.extract::<f64>()?;
        let close_value = close_item.extract::<f64>()?;
        high_vec.push(high_value);
        low_vec.push(low_value);
        close_vec.push(close_value);
    }

    if high_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "price lists must be non-empty",
        ));
    }

    let n_samples = high_vec.len();
    let mut result_size = 5; // Default for standard (PP, S1, S2, R1, R2)
    
    // Determine the number of pivot points based on type
    match pivot_type {
        "standard" => result_size = 5, // PP, S1, S2, R1, R2
        "fibonacci" => result_size = 7, // PP, S1, S2, S3, R1, R2, R3
        "camarilla" => result_size = 9, // PP, S1, S2, S3, S4, R1, R2, R3, R4
        "woodie" => result_size = 5,    // PP, S1, S2, R1, R2
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown pivot_type: {}", pivot_type),
        )),
    }

    // Create a 2D vector to store the result (n_samples x result_size)
    let mut result = vec![vec![f64::NAN; result_size]; n_samples];

    // Calculate pivot points for each sample
    for i in 1..n_samples {
        let high = high_vec[i-1];
        let low = low_vec[i-1];
        let close = close_vec[i-1];
        
        match pivot_type {
            "standard" => {
                // Standard pivot points
                let pp = (high + low + close) / 3.0;
                let r1 = (2.0 * pp) - low;
                let r2 = pp + (high - low);
                let s1 = (2.0 * pp) - high;
                let s2 = pp - (high - low);
                
                result[i][0] = pp;
                result[i][1] = s1;
                result[i][2] = s2;
                result[i][3] = r1;
                result[i][4] = r2;
            },
            "fibonacci" => {
                // Fibonacci pivot points
                let pp = (high + low + close) / 3.0;
                let r1 = pp + 0.382 * (high - low);
                let r2 = pp + 0.618 * (high - low);
                let r3 = pp + 1.0 * (high - low);
                let s1 = pp - 0.382 * (high - low);
                let s2 = pp - 0.618 * (high - low);
                let s3 = pp - 1.0 * (high - low);
                
                result[i][0] = pp;
                result[i][1] = s1;
                result[i][2] = s2;
                result[i][3] = s3;
                result[i][4] = r1;
                result[i][5] = r2;
                result[i][6] = r3;
            },
            "camarilla" => {
                // Camarilla pivot points
                let pp = (high + low + close) / 3.0;
                let r1 = close + 1.1 / 12.0 * (high - low);
                let r2 = close + 1.1 / 6.0 * (high - low);
                let r3 = close + 1.1 / 4.0 * (high - low);
                let r4 = close + 1.1 / 2.0 * (high - low);
                let s1 = close - 1.1 / 12.0 * (high - low);
                let s2 = close - 1.1 / 6.0 * (high - low);
                let s3 = close - 1.1 / 4.0 * (high - low);
                let s4 = close - 1.1 / 2.0 * (high - low);
                
                result[i][0] = pp;
                result[i][1] = s1;
                result[i][2] = s2;
                result[i][3] = s3;
                result[i][4] = s4;
                result[i][5] = r1;
                result[i][6] = r2;
                result[i][7] = r3;
                result[i][8] = r4;
            },
            "woodie" => {
                // Woodie's pivot points
                let pp = (high + low + (2.0 * close)) / 4.0;
                let r1 = (2.0 * pp) - low;
                let r2 = pp + (high - low);
                let s1 = (2.0 * pp) - high;
                let s2 = pp - (high - low);
                
                result[i][0] = pp;
                result[i][1] = s1;
                result[i][2] = s2;
                result[i][3] = r1;
                result[i][4] = r2;
            },
            _ => {}, // Already handled in validation
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, row);
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Calculate volume profile for price data.
///
/// Args:
///     prices: List of price values
///     volumes: List of volume values
///     n_bins: Number of price bins to use
///
/// Returns:
///     List of lists: Each inner list contains [price_level, volume_at_level]
#[pyfunction]
pub fn calculate_volume_profile_rs(
    py: Python,
    prices: &PyList,
    volumes: &PyList,
    n_bins: i32,
) -> PyResult<Py<PyList>> {
    // Input validation
    if prices.len() != volumes.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prices and volumes must have the same length",
        ));
    }

    if n_bins <= 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be a positive integer",
        ));
    }

    // Convert Python lists to Rust vectors
    let mut price_vec = Vec::new();
    let mut volume_vec = Vec::new();
    
    for (price_item, volume_item) in prices.iter().zip(volumes.iter()) {
        let price_value = price_item.extract::<f64>()?;
        let volume_value = volume_item.extract::<f64>()?;
        price_vec.push(price_value);
        volume_vec.push(volume_value);
    }

    if price_vec.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "price and volume lists must be non-empty",
        ));
    }

    // Find the min and max prices
    let min_price = *price_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let max_price = *price_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    
    // Calculate the bin size
    let bin_size = (max_price - min_price) / n_bins as f64;
    
    // Create bins for the volume profile
    let mut bins = vec![0.0; n_bins as usize];
    let mut bin_prices = vec![0.0; n_bins as usize];
    
    // Calculate the center price for each bin
    for i in 0..n_bins as usize {
        bin_prices[i] = min_price + (i as f64 + 0.5) * bin_size;
    }
    
    // Assign volumes to bins
    for i in 0..price_vec.len() {
        let bin_index = ((price_vec[i] - min_price) / bin_size).floor() as usize;
        if bin_index < bins.len() {
            bins[bin_index] += volume_vec[i];
        }
    }
    
    // Create the result as a list of [price_level, volume_at_level] pairs
    let mut result = Vec::new();
    for i in 0..n_bins as usize {
        result.push(vec![bin_prices[i], bins[i]]);
    }
    
    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, row);
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Register Python module
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_bollinger_bands_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rsi_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_macd_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fibonacci_retracement_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pivot_points_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_volume_profile_rs, m)?)?;
    Ok(())
}
