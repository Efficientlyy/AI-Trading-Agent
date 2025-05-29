use pyo3::prelude::*;
use pyo3::types::PyList;
// use rayon::prelude::*;

// Helper function to calculate EMA for a single series and span
fn calculate_ema_single(
    series: &[f64],
    span: i32,
    alpha_opt: Option<f64>,
) -> Vec<Option<f64>> {
    let n = series.len();
    if n == 0 || span <= 0 {
        return vec![None; n]; // Return vector of Nones matching series length
    }

    let mut ema_values: Vec<Option<f64>> = vec![None; n];
    let alpha = alpha_opt.unwrap_or_else(|| 2.0 / (span as f64 + 1.0));

    if n > 0 {
        ema_values[0] = Some(series[0]); // First EMA is the first data point
    }

    for i in 1..n {
        if let Some(prev_ema) = ema_values[i - 1] {
            ema_values[i] = Some(alpha * series[i] + (1.0 - alpha) * prev_ema);
        } else {
            // If previous EMA was None (e.g., due to insufficient data at the very start for a complex init),
            // we might restart EMA or continue with None. Here, we restart with current value.
            ema_values[i] = Some(series[i]); 
        }
    }
    ema_values
}

/// Create Exponential Moving Average (EMA) features from a time series.
/// 
/// Args:
///     series (Vec<f64>): Input time series.
///     spans (Vec<i32>): List of EMA window spans.
///     alpha (Option<f64>): Optional smoothing factor alpha. If None, it's calculated as 2.0 / (span + 1.0).
/// 
/// Returns:
///     PyResult<Py<PyList>>: A Python list where each inner list contains the EMA series for a corresponding span.
#[pyfunction]
#[pyo3(signature = (series, spans, alpha = None))]
pub fn create_ema_features_rs(
    series: Vec<f64>,
    spans: Vec<i32>,
    alpha: Option<f64>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }

    for &span_val in &spans {
        if span_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Span values must be positive."
            ));
        }
    }

    if let Some(a_val) = alpha {
        if a_val <= 0.0 || a_val >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Alpha must be between 0 and 1 exclusive if provided."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        let results_from_spans: Vec<Vec<Option<f64>>> = spans
            // .par_iter() // Example: Using Rayon for parallel processing if desired
            .iter()
            .map(|&current_span| {
                calculate_ema_single(&series, current_span, alpha)
            })
            .collect();

        for single_ema_series_data in results_from_spans {
            let py_single_ema_series = PyList::new_bound(py, &single_ema_series_data);
            main_result_list.append(py_single_ema_series)?;
        }

        Ok(main_result_list.into())
    })
}

// Helper function to calculate SMA for a single series and window
fn calculate_sma_single(series: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = series.len();
    if n == 0 || window == 0 || window > n {
        // If window is 0, or larger than series, or series is empty, all results are None
        // or you could return an empty vec or error based on desired behavior.
        // For consistency with indicators often padding initial values, vec of Nones is chosen.
        return vec![None; n];
    }

    let mut sma_values: Vec<Option<f64>> = vec![None; n];
    let mut current_sum: f64 = 0.0;

    for i in 0..n {
        current_sum += series[i];
        if i >= window {
            current_sum -= series[i - window]; // Subtract the element that's sliding out of the window
            sma_values[i] = Some(current_sum / window as f64);
        } else if i == window - 1 {
            // First point where a full window is available
            sma_values[i] = Some(current_sum / window as f64);
        } else {
            // Not enough data points yet for a full window
            sma_values[i] = None;
        }
    }
    sma_values
}

/// Create Simple Moving Average (SMA) features from a time series.
///
/// Args:
///     series (Vec<f64>): Input time series.
///     windows (Vec<i32>): List of SMA window sizes.
///
/// Returns:
///     PyResult<Py<PyList>>: A Python list where each inner list contains the SMA series for a corresponding window.
#[pyfunction]
pub fn calculate_sma_rust_direct(
    series: Vec<f64>,
    windows: Vec<i32>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }

    for &window_val in &windows {
        if window_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Window sizes must be positive."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        let results_from_windows: Vec<Vec<Option<f64>>> = windows
            .iter()
            .map(|&current_window| {
                calculate_sma_single(&series, current_window as usize)
            })
            .collect();

        for single_sma_series_data in results_from_windows {
            let py_single_sma_series = PyList::new_bound(py, &single_sma_series_data);
            main_result_list.append(py_single_sma_series)?;
        }

        Ok(main_result_list.into())
    })
}

// Helper function to register all moving average functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_ema_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sma_rust_direct, m)?)?;
    Ok(())
}
