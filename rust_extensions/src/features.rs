use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
// use rayon::prelude::*;

/// Create lag features from a time series.
///
/// Args:
///     series: Input time series as a Rust vector
///     lags: List of lag periods
///
/// Returns:
///     List of lists: Each inner list represents a lag feature
#[pyfunction]
pub fn create_lag_features_rs(
    series: Vec<f64>,
    lags: Vec<i32>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    for &lag_val in &lags {
        if lag_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Lag values must be positive."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &lag_val in &lags {
            let lag = lag_val as usize;
            let mut single_lag_series_data: Vec<Option<f64>> = vec![None; series.len()];

            if lag <= series.len() { // lag > 0 is already checked
                for i in lag..series.len() {
                    single_lag_series_data[i] = Some(series[i - lag]);
                }
            }
            // If lag > series.len(), all values remain None, which is correct.

            let py_single_lag_series = PyList::new_bound(py, &single_lag_series_data);
            main_result_list.append(&py_single_lag_series)?;
        }
        Ok(main_result_list.into())
    })
}

/// Create difference features from a time series.
///
/// Args:
///     series: Input time series as a Rust vector
///     periods: List of periods for calculating differences
///
/// Returns:
///     List of lists: Each inner list represents a difference feature
#[pyfunction]
pub fn create_diff_features_rs(
    series: Vec<f64>,
    periods: Vec<i32>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    for &period_val in &periods {
        if period_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Period values must be positive."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &period_val in &periods {
            let period = period_val as usize;
            let mut single_diff_series_data: Vec<Option<f64>> = vec![None; series.len()];

            if period <= series.len() { // period > 0 is already checked
                for i in period..series.len() {
                    single_diff_series_data[i] = Some(series[i] - series[i - period]);
                }
            }
            // If period > series.len(), all values remain None.

            let py_single_diff_series = PyList::new_bound(py, &single_diff_series_data);
            main_result_list.append(&py_single_diff_series)?;
        }
        Ok(main_result_list.into())
    })
}

/// Create percentage change features from a time series.
///
/// Args:
///     series: Input time series as a Rust vector
///     periods: List of periods for calculating percentage changes
///
/// Returns:
///     List of lists: Each inner list represents a percentage change feature
#[pyfunction]
pub fn create_pct_change_features_rs(
    series: Vec<f64>,
    periods: Vec<i32>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    for &period_val in &periods {
        if period_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Period values must be positive."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &period_val in &periods {
            let period = period_val as usize;
            let mut single_pct_change_series_data: Vec<Option<f64>> = vec![None; series.len()];

            if period <= series.len() { // period > 0 is already checked
                for i in period..series.len() {
                    let prev_val = series[i - period];
                    if prev_val != 0.0 {
                        single_pct_change_series_data[i] = Some((series[i] - prev_val) / prev_val);
                    } else {
                        single_pct_change_series_data[i] = Some(f64::NAN); // Or None, depending on desired behavior for division by zero
                    }
                }
            }
            let py_single_pct_change_series = PyList::new_bound(py, &single_pct_change_series_data);
            main_result_list.append(&py_single_pct_change_series)?;
        }
        Ok(main_result_list.into())
    })
}

/// Create rolling window features from a time series.
///
/// Args:
///     series: Input time series as a Rust vector
///     windows: List of window sizes
///     feature_type: Type of feature to calculate (0: mean, 1: std, 2: min, 3: max, 4: sum)
///
/// Returns:
///     List of lists: Each inner list represents a rolling window feature
#[pyfunction]
pub fn create_rolling_window_features_rs(
    series: Vec<f64>,
    windows: Vec<i32>,
    feature_type: String, // "mean", "std", "min", "max", "sum"
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    for &window_val in &windows {
        if window_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Window values must be positive."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &window_val in &windows {
            let window = window_val as usize;
            let mut single_rolling_series_data: Vec<Option<f64>> = vec![None; series.len()];

            if window > 0 && window <= series.len() {
                for i in (window - 1)..series.len() {
                    let window_slice = &series[(i - window + 1)..=i];
                    let val = match feature_type.as_str() {
                        "mean" => Some(window_slice.iter().sum::<f64>() / window as f64),
                        "std" => {
                            let mean = window_slice.iter().sum::<f64>() / window as f64;
                            let variance = window_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                            Some(variance.sqrt())
                        }
                        "min" => Some(window_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
                        "max" => Some(window_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
                        "sum" => Some(window_slice.iter().sum::<f64>()),
                        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Invalid feature_type: {}", feature_type)
                        )),
                    };
                    single_rolling_series_data[i] = val;
                }
            }
            let py_single_rolling_series = PyList::new_bound(py, &single_rolling_series_data);
            main_result_list.append(&py_single_rolling_series)?;
        }
        Ok(main_result_list.into())
    })
}

/// Register Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_lag_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_diff_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_pct_change_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rolling_window_features_rs, m)?)?;
    Ok(())
}
