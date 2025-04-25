use pyo3::prelude::*;
use pyo3::types::{PyList, PyFloat};

/// Create lag features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     lags: List of lag periods
///
/// Returns:
///     List of lists: Each inner list represents a lag feature
#[pyfunction]
fn create_lag_features(py: Python, series: &PyList, lags: Vec<usize>) -> PyResult<Py<PyList>> {
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
        if lag == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Lag periods must be positive integers, got 0",
            ));
        }

        for j in lag..n_samples {
            result[j][i] = series_vec[j - lag];
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, &[]);
        for &val in row.iter() {
            if val.is_nan() {
                py_row.append(py.None())?;
            } else {
                py_row.append(val)?;
            }
        }
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
fn create_diff_features(py: Python, series: &PyList, periods: Vec<usize>) -> PyResult<Py<PyList>> {
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
        if period == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Periods must be positive integers, got 0",
            ));
        }

        for j in period..n_samples {
            result[j][i] = series_vec[j] - series_vec[j - period];
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, &[]);
        for &val in row.iter() {
            if val.is_nan() {
                py_row.append(py.None())?;
            } else {
                py_row.append(val)?;
            }
        }
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
fn create_pct_change_features(py: Python, series: &PyList, periods: Vec<usize>) -> PyResult<Py<PyList>> {
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
        if period == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Periods must be positive integers, got 0",
            ));
        }

        for j in period..n_samples {
            let previous_value = series_vec[j - period];
            if previous_value != 0.0 {
                result[j][i] = (series_vec[j] - previous_value) / previous_value;
            }
            // NaN is already the default for division by zero
        }
    }

    // Convert result to Python list of lists
    let py_result = PyList::new(py, &[]);
    
    for row in result.iter() {
        let py_row = PyList::new(py, &[]);
        for &val in row.iter() {
            if val.is_nan() {
                py_row.append(py.None())?;
            } else {
                py_row.append(val)?;
            }
        }
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_lag_features(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_lag_features, m)?)?;
    m.add_function(wrap_pyfunction!(create_diff_features, m)?)?;
    m.add_function(wrap_pyfunction!(create_pct_change_features, m)?)?;
    Ok(())
}
