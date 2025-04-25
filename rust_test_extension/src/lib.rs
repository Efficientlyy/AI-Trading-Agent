use pyo3::prelude::*;

/// A simple function that adds two numbers
#[pyfunction]
fn add_numbers(a: i64, b: i64) -> PyResult<i64> {
    Ok(a + b)
}

/// Create lag feature from a time series.
///
/// Args:
///     series: Input time series as a list of floats
///     lag: Lag period
///
/// Returns:
///     List of lagged values
#[pyfunction]
fn create_lag_feature(series: Vec<f64>, lag: usize) -> PyResult<Vec<f64>> {
    // Input validation
    if series.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    if lag == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lag must be greater than 0",
        ));
    }

    let n_samples = series.len();
    let mut result = vec![f64::NAN; n_samples];

    // Calculate lag feature
    for j in lag..n_samples {
        result[j] = series[j - lag];
    }

    Ok(result)
}

/// Create difference feature from a time series.
///
/// Args:
///     series: Input time series as a list of floats
///     period: Period for calculating difference
///
/// Returns:
///     List of difference values
#[pyfunction]
fn create_diff_feature(series: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    // Input validation
    if series.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    if period == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "period must be greater than 0",
        ));
    }

    let n_samples = series.len();
    let mut result = vec![f64::NAN; n_samples];

    // Calculate difference feature
    for j in period..n_samples {
        result[j] = series[j] - series[j - period];
    }

    Ok(result)
}

/// Create percentage change feature from a time series.
///
/// Args:
///     series: Input time series as a list of floats
///     period: Period for calculating percentage change
///
/// Returns:
///     List of percentage change values
#[pyfunction]
fn create_pct_change_feature(series: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    // Input validation
    if series.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "series must be non-empty",
        ));
    }

    if period == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "period must be greater than 0",
        ));
    }

    let n_samples = series.len();
    let mut result = vec![f64::NAN; n_samples];

    // Calculate percentage change feature
    for j in period..n_samples {
        let previous_value = series[j - period];
        if previous_value != 0.0 {
            result[j] = (series[j] - previous_value) / previous_value;
        }
        // If previous value is zero, result remains NaN
    }

    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_test_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(create_lag_feature, m)?)?;
    m.add_function(wrap_pyfunction!(create_diff_feature, m)?)?;
    m.add_function(wrap_pyfunction!(create_pct_change_feature, m)?)?;
    Ok(())
}
