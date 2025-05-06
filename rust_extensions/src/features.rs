use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use std::collections::VecDeque;
use rayon::prelude::*;

/// Create lag features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     lags: List of lag periods
///
/// Returns:
///     List of lists: Each inner list represents a lag feature
#[pyfunction]
pub fn create_lag_features_rs(
    py: Python,
    series: &PyList,
    lags: Vec<i32>,
) -> PyResult<Py<PyList>> {
    // Input validation
    if lags.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lags must be a non-empty list of integers",
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
    let n_features = lags.len();

    // Create a 2D vector to store the result
    let mut result = vec![vec![f64::NAN; n_features]; n_samples];

    // Calculate lag features
    for (i, &lag) in lags.iter().enumerate() {
        if lag <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Lag periods must be positive integers, got {}", lag),
            ));
        }

        let lag_usize = lag as usize;
        for j in lag_usize..n_samples {
            result[j][i] = series_vec[j - lag_usize];
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

/// Create difference features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     periods: List of periods for calculating differences
///
/// Returns:
///     List of lists: Each inner list represents a difference feature
#[pyfunction]
pub fn create_diff_features_rs(
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

    // Calculate difference features
    for (i, &period) in periods.iter().enumerate() {
        if period <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Periods must be positive integers, got {}", period),
            ));
        }

        let period_usize = period as usize;
        for j in period_usize..n_samples {
            result[j][i] = series_vec[j] - series_vec[j - period_usize];
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

/// Create percentage change features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     periods: List of periods for calculating percentage changes
///
/// Returns:
///     List of lists: Each inner list represents a percentage change feature
#[pyfunction]
pub fn create_pct_change_features_rs(
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

    // Calculate percentage change features
    for (i, &period) in periods.iter().enumerate() {
        if period <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Periods must be positive integers, got {}", period),
            ));
        }

        let period_usize = period as usize;
        for j in period_usize..n_samples {
            let previous_value = series_vec[j - period_usize];
            if previous_value != 0.0 {
                result[j][i] = (series_vec[j] - previous_value) / previous_value;
            } else {
                // Handle division by zero
                result[j][i] = f64::NAN;
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

/// Create rolling window features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     windows: List of window sizes
///     feature_type: Type of feature to calculate (0: mean, 1: std, 2: min, 3: max, 4: sum)
///
/// Returns:
///     List of lists: Each inner list represents a rolling window feature
#[pyfunction]
pub fn create_rolling_window_features_rs(
    py: Python,
    series: &PyList,
    windows: Vec<i32>,
    feature_type: i32,
) -> PyResult<Py<PyList>> {
    // Input validation
    if windows.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "windows must be a non-empty list of integers",
        ));
    }

    // Validate feature type
    if feature_type < 0 || feature_type > 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("feature_type must be between 0 and 4, got {}", feature_type),
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
    let n_features = windows.len();

    // Create a 2D vector to store the result
    let mut result = vec![vec![f64::NAN; n_features]; n_samples];

    // Calculate rolling window features in parallel for each window size
    windows.par_iter().enumerate().for_each(|(i, &window)| {
        if window <= 0 {
            // Skip invalid windows (will be handled by error checking below)
            return;
        }

        let window_usize = window as usize;
        
        // For each window size, calculate the rolling window feature for each time point
        for j in window_usize - 1..n_samples {
            // Get the window data
            let window_data = &series_vec[j - (window_usize - 1)..=j];
            
            // Calculate the feature based on the feature type
            match feature_type {
                0 => { // Mean
                    let sum: f64 = window_data.iter().sum();
                    result[j][i] = sum / window_usize as f64;
                },
                1 => { // Standard deviation
                    let mean: f64 = window_data.iter().sum::<f64>() / window_usize as f64;
                    let variance: f64 = window_data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / window_usize as f64;
                    result[j][i] = variance.sqrt();
                },
                2 => { // Min
                    result[j][i] = window_data.iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                },
                3 => { // Max
                    result[j][i] = window_data.iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                },
                4 => { // Sum
                    result[j][i] = window_data.iter().sum();
                },
                _ => unreachable!(), // We already checked for invalid feature types
            }
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
    m.add_function(wrap_pyfunction!(create_lag_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_diff_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_pct_change_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rolling_window_features_rs, m)?)?;
    Ok(())
}
