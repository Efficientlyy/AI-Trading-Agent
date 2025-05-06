use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use rayon::prelude::*;

/// Create exponential moving average (EMA) features from a time series.
///
/// Args:
///     series: Input time series as a Python list
///     spans: List of EMA spans (periods)
///     alpha: Optional smoothing factor (if None, alpha = 2/(span+1))
///
/// Returns:
///     List of lists: Each inner list represents an EMA feature
#[pyfunction]
pub fn create_ema_features_rs(
    py: Python,
    series: &PyList,
    spans: Vec<i32>,
    alpha: Option<f64>,
) -> PyResult<Py<PyList>> {
    // Input validation
    if spans.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "spans must be a non-empty list of integers",
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
    let n_features = spans.len();

    // Create a 2D vector to store the result
    let mut result = vec![vec![f64::NAN; n_features]; n_samples];

    // Calculate EMA features
    for (i, &span) in spans.iter().enumerate() {
        if span <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Span periods must be positive integers, got {}", span),
            ));
        }

        let span_usize = span as usize;
        let alpha_value = alpha.unwrap_or(2.0 / (span as f64 + 1.0));
        
        if alpha_value <= 0.0 || alpha_value > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Alpha must be in (0, 1], got {}", alpha_value),
            ));
        }

        // Initialize EMA with the first value
        result[0][i] = series_vec[0];
        
        // Calculate EMA for each time point
        for j in 1..n_samples {
            result[j][i] = alpha_value * series_vec[j] + (1.0 - alpha_value) * result[j-1][i];
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

/// Create C-compatible function for calculating EMA features
#[no_mangle]
pub extern "C" fn create_ema_features_c(
    data_ptr: *const f64,
    data_len: usize,
    spans_ptr: *const i32,
    spans_len: usize,
    alpha_ptr: *const f64,  // Can be null for default alpha
    result_ptr: *mut f64
) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || spans_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let spans = unsafe { std::slice::from_raw_parts(spans_ptr, spans_len) };
    let result = unsafe { std::slice::from_raw_parts_mut(result_ptr, data_len * spans_len) };
    
    // Get alpha value if provided
    let alpha = if alpha_ptr.is_null() {
        None
    } else {
        Some(unsafe { *alpha_ptr })
    };
    
    // Check for invalid spans
    for &span in spans {
        if span <= 0 {
            return -2;
        }
    }
    
    // Check alpha value if provided
    if let Some(a) = alpha {
        if a <= 0.0 || a > 1.0 {
            return -3;
        }
    }
    
    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }
    
    // Calculate EMA features in parallel for each span
    spans.par_iter().enumerate().for_each(|(i, &span)| {
        let alpha_value = alpha.unwrap_or(2.0 / (span as f64 + 1.0));
        
        // Initialize EMA with the first value
        let result_idx = 0 * spans_len + i;
        result[result_idx] = data[0];
        
        // Calculate EMA for each time point
        for j in 1..data_len {
            let prev_idx = (j-1) * spans_len + i;
            let curr_idx = j * spans_len + i;
            result[curr_idx] = alpha_value * data[j] + (1.0 - alpha_value) * result[prev_idx];
        }
    });
    
    0 // Success
}

/// Register Python module
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_ema_features_rs, m)?)?;
    Ok(())
}
