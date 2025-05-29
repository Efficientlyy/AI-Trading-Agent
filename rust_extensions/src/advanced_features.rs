use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::collections::HashMap;

// --- Helper function for SMA (used in Bollinger Bands and MACD signal line) ---
fn sma(data: &[f64], window: usize) -> Vec<Option<f64>> {
    if window == 0 || data.is_empty() {
        return vec![None; data.len()];
    }
    let mut result = vec![None; data.len()];
    if window > data.len() {
        return result; // Not enough data for even one window
    }
    for i in (window - 1)..data.len() {
        let window_slice = &data[(i + 1 - window)..=i];
        result[i] = Some(window_slice.iter().sum::<f64>() / window as f64);
    }
    result
}

// --- Helper function for EMA (used in MACD) ---
fn ema(data: &[f64], window: usize, alpha_opt: Option<f64>) -> Vec<Option<f64>> {
    let n = data.len();
    if n == 0 || window == 0 {
        return vec![None; n];
    }
    let mut ema_values: Vec<Option<f64>> = vec![None; n];
    let alpha = alpha_opt.unwrap_or_else(|| 2.0 / (window as f64 + 1.0));

    if n > 0 {
        ema_values[0] = Some(data[0]);
    }
    for i in 1..n {
        if let Some(prev_ema) = ema_values[i - 1] {
            ema_values[i] = Some(alpha * data[i] + (1.0 - alpha) * prev_ema);
        } else {
            ema_values[i] = Some(data[i]); // Fallback if previous EMA is None
        }
    }
    ema_values
}

/// Create Bollinger Bands features from a time series.
#[pyfunction]
#[pyo3(signature = (series, windows, num_std_dev = None))]
pub fn create_bollinger_bands_rs(
    series: Vec<f64>,
    windows: Vec<i32>,
    num_std_dev: Option<f64>,
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
    let num_std = num_std_dev.unwrap_or(2.0);
    if num_std <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of standard deviations must be positive."
        ));
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &window_val in &windows {
            let window = window_val as usize;
            let middle_band_vec = sma(&series, window);
            
            let mut upper_band_vec: Vec<Option<f64>> = vec![None; series.len()];
            let mut lower_band_vec: Vec<Option<f64>> = vec![None; series.len()];

            if window > 0 && window <= series.len() {
                for i in (window - 1)..series.len() {
                    if middle_band_vec[i].is_some() {
                        let window_slice = &series[(i + 1 - window)..=i];
                        let mean = middle_band_vec[i].unwrap();
                        let variance = window_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                        let std_dev_val = variance.sqrt();
                        upper_band_vec[i] = Some(mean + num_std * std_dev_val);
                        lower_band_vec[i] = Some(mean - num_std * std_dev_val);
                    }
                }
            }

            let py_upper_band = PyList::new_bound(py, upper_band_vec.iter().map(|&x| x.to_object(py)));
            let py_middle_band = PyList::new_bound(py, middle_band_vec.iter().map(|&x| x.to_object(py)));
            let py_lower_band = PyList::new_bound(py, lower_band_vec.iter().map(|&x| x.to_object(py)));
            
            let bands_for_window = PyList::empty_bound(py);
            bands_for_window.append(&py_upper_band)?;
            bands_for_window.append(&py_middle_band)?;
            bands_for_window.append(&py_lower_band)?;
            main_result_list.append(&bands_for_window)?;
        }
        Ok(main_result_list.into())
    })
}

/// Create Relative Strength Index (RSI) features from a time series.
#[pyfunction]
pub fn create_rsi_features_rs(
    series: Vec<f64>,
    windows: Vec<i32>,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    for &window_val in &windows {
        if window_val <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Window values must be positive and typically > 1 for RSI."
            ));
        }
    }

    Python::with_gil(|py| {
        let main_result_list = PyList::empty_bound(py);

        for &window_val in &windows {
            let window = window_val as usize;
            let mut rsi_values: Vec<Option<f64>> = vec![None; series.len()];

            if window > 0 && series.len() >= window {
                let mut gains = 0.0;
                let mut losses = 0.0;

                // Calculate initial average gain and loss for the first window
                for i in 1..=window {
                    let diff = series[i] - series[i-1];
                    if diff > 0.0 {
                        gains += diff;
                    } else {
                        losses -= diff; // losses are positive values
                    }
                }
                
                let mut avg_gain = gains / window as f64;
                let mut avg_loss = losses / window as f64;

                if avg_loss == 0.0 {
                    rsi_values[window] = Some(100.0);
                } else {
                    let rs = avg_gain / avg_loss;
                    rsi_values[window] = Some(100.0 - (100.0 / (1.0 + rs)));
                }

                // Calculate subsequent RSI values
                for i in (window + 1)..series.len() {
                    let diff = series[i] - series[i-1];
                    let current_gain = if diff > 0.0 { diff } else { 0.0 };
                    let current_loss = if diff < 0.0 { -diff } else { 0.0 };

                    avg_gain = (avg_gain * (window - 1) as f64 + current_gain) / window as f64;
                    avg_loss = (avg_loss * (window - 1) as f64 + current_loss) / window as f64;

                    if avg_loss == 0.0 {
                        rsi_values[i] = Some(100.0);
                    } else {
                        let rs = avg_gain / avg_loss;
                        rsi_values[i] = Some(100.0 - (100.0 / (1.0 + rs)));
                    }
                }
            }
            let py_rsi_series = PyList::new_bound(py, rsi_values.iter().map(|&x| x.to_object(py)));
            main_result_list.append(&py_rsi_series)?;
        }
        Ok(main_result_list.into())
    })
}

/// Create Moving Average Convergence Divergence (MACD) features.
#[pyfunction]
pub fn create_macd_features_rs(
    series: Vec<f64>,
    short_window: i32,
    long_window: i32,
    signal_window: i32,
) -> PyResult<Py<PyList>> {
    if series.is_empty() {
        return Python::with_gil(|py| Ok(PyList::empty_bound(py).into()));
    }
    if short_window <= 0 || long_window <= 0 || signal_window <= 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window values must be positive."
        ));
    }
    if short_window >= long_window {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Short window must be less than long window for MACD."
        ));
    }

    Python::with_gil(|py| {
        let ema_short = ema(&series, short_window as usize, None);
        let ema_long = ema(&series, long_window as usize, None);

        let mut macd_line: Vec<Option<f64>> = vec![None; series.len()];
        for i in 0..series.len() {
            if let (Some(es), Some(el)) = (ema_short[i], ema_long[i]) {
                macd_line[i] = Some(es - el);
            }
        }

        // Convert Vec<Option<f64>> to Vec<f64> for sma input, handling None as appropriate
        // For SMA, it's better to skip None values or use a version of SMA that handles Option<f64>
        // For simplicity here, we'll create a new series for SMA input, replacing None with a value that won't affect sum if possible, or filter them.
        // A more robust SMA would handle Option directly. Here, we filter out Nones before passing to SMA.
        let macd_line_for_signal_calc: Vec<f64> = macd_line.iter().filter_map(|&x| x).collect();
        let signal_line_calculated_on_filtered = sma(&macd_line_for_signal_calc, signal_window as usize);
        
        // Align signal_line_calculated_on_filtered back to the original series length
        let mut signal_line: Vec<Option<f64>> = vec![None; series.len()];
        let mut current_signal_idx = 0;
        for i in 0..series.len() {
            if macd_line[i].is_some() {
                if current_signal_idx < signal_line_calculated_on_filtered.len() {
                    signal_line[i] = signal_line_calculated_on_filtered[current_signal_idx];
                    current_signal_idx += 1;
                } else {
                    // Not enough data in macd_line_for_signal_calc to produce more signal points
                    break;
                }
            }
        }

        let mut histogram: Vec<Option<f64>> = vec![None; series.len()];
        for i in 0..series.len() {
            if let (Some(m_val), Some(s_val)) = (macd_line[i], signal_line[i]) {
                histogram[i] = Some(m_val - s_val);
            }
        }

        let py_macd_line = PyList::new_bound(py, macd_line.iter().map(|&x| x.to_object(py)));
        let py_signal_line = PyList::new_bound(py, signal_line.iter().map(|&x| x.to_object(py)));
        let py_histogram = PyList::new_bound(py, histogram.iter().map(|&x| x.to_object(py)));

        let result_list = PyList::empty_bound(py);
        result_list.append(&py_macd_line)?;
        result_list.append(&py_signal_line)?;
        result_list.append(&py_histogram)?;

        Ok(result_list.into())
    })
}

/// Calculate Fibonacci Retracement levels.
#[pyfunction]
pub fn calculate_fibonacci_retracement_rs(
    high_prices: Vec<f64>,
    low_prices: Vec<f64>,
) -> PyResult<PyObject> { 
    if high_prices.is_empty() || low_prices.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Price series cannot be empty."
        ));
    }
    if high_prices.len() != low_prices.len() {
         return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "High and low price series must have the same length."
        ));
    }

    // For simplicity, using the overall high and low of the provided period.
    // A more complex version might identify significant swing highs/lows.
    let overall_high = high_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let overall_low = low_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if overall_high == f64::NEG_INFINITY || overall_low == f64::INFINITY {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Could not determine overall high/low from price series (e.g., all NaN or empty after filter)."
        ));
    }
    
    let diff = overall_high - overall_low;
    if diff < 0.0 { // Should not happen if overall_high >= overall_low
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Overall high cannot be less than overall low."
        ));
    }

    Python::with_gil(|py| {
        let levels = PyDict::new_bound(py);
        levels.set_item("high", overall_high)?;
        levels.set_item("low", overall_low)?;
        levels.set_item("0.236", overall_high - diff * 0.236)?;
        levels.set_item("0.382", overall_high - diff * 0.382)?;
        levels.set_item("0.5", overall_high - diff * 0.5)?;
        levels.set_item("0.618", overall_high - diff * 0.618)?;
        // levels.set_item("0.786", overall_high - diff * 0.786)?; // Optional
        Ok(levels.into_py(py))
    })
}

/// Calculate Pivot Points (daily).
#[pyfunction]
pub fn calculate_pivot_points_rs(
    high: f64,
    low: f64,
    close: f64,
) -> PyResult<PyObject> { 
    Python::with_gil(|py| {
        let pp = (high + low + close) / 3.0;
        let r1 = (2.0 * pp) - low;
        let s1 = (2.0 * pp) - high;
        let r2 = pp + (high - low);
        let s2 = pp - (high - low);
        let r3 = high + 2.0 * (pp - low);
        let s3 = low - 2.0 * (high - pp);

        let points = PyDict::new_bound(py);
        points.set_item("PP", pp)?;
        points.set_item("R1", r1)?;
        points.set_item("S1", s1)?;
        points.set_item("R2", r2)?;
        points.set_item("S2", s2)?;
        points.set_item("R3", r3)?;
        points.set_item("S3", s3)?;
        Ok(points.into_py(py))
    })
}

/// Calculate Volume Profile.
#[pyfunction]
pub fn calculate_volume_profile_rs(
    prices: Vec<f64>,
    volumes: Vec<f64>,
    num_bins: usize,
) -> PyResult<PyObject> { 
    if prices.len() != volumes.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Prices and volumes series must have the same length."
        ));
    }
    if prices.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Price series cannot be empty for volume profile."
        ));
    }
    if num_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of bins must be positive."
        ));
    }

    Python::with_gil(|py| {
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_price == f64::INFINITY || max_price == f64::NEG_INFINITY {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Could not determine min/max price (e.g., all NaN or empty after filter)."
            ));
        }
        if min_price == max_price { // All prices are the same, or only one price point
            let profile = PyDict::new_bound(py);
            let bin_label = format!("{}", min_price);
            let total_volume: f64 = volumes.iter().sum();
            profile.set_item(bin_label, total_volume)?;
            return Ok(profile.into_py(py));
        }

        let bin_size = (max_price - min_price) / num_bins as f64;
        let mut bins: HashMap<usize, f64> = HashMap::new();

        for (price, volume) in prices.iter().zip(volumes.iter()) {
            if *price < min_price || *price > max_price { continue; } // Should not happen if min/max are correct
            let mut bin_index = ((price - min_price) / bin_size) as usize;
            if bin_index >= num_bins { // Handle price == max_price case
                bin_index = num_bins - 1;
            }
            *bins.entry(bin_index).or_insert(0.0) += volume;
        }

        let profile = PyDict::new_bound(py);
        for i in 0..num_bins {
            let bin_start = min_price + (i as f64 * bin_size);
            let bin_end = bin_start + bin_size;
            // Using middle of the bin or range as key
            let bin_label = format!("{:.2}-{:.2}", bin_start, bin_end);
            profile.set_item(bin_label, bins.get(&i).unwrap_or(&0.0).to_object(py))?;
        }
        Ok(profile.into_py(py))
    })
}

// --- Stochastic Oscillator --- 
fn calculate_stochastic_oscillator_raw(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    k_period: usize,
    d_period: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = highs.len();
    if n == 0 || k_period == 0 || d_period == 0 || k_period > n {
        return (vec![None; n], vec![None; n]);
    }

    let mut percent_k_values = vec![None; n];

    for i in (k_period - 1)..n {
        let high_window = &highs[(i + 1 - k_period)..=i];
        let low_window = &lows[(i + 1 - k_period)..=i];
        let current_close = closes[i];

        let highest_high_k = high_window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low_k = low_window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if highest_high_k == f64::NEG_INFINITY || lowest_low_k == f64::INFINITY {
            percent_k_values[i] = None; // Should not happen with valid price data
            continue;
        }

        let denominator = highest_high_k - lowest_low_k;

        if denominator < 1e-9 { // Denominator is zero or very close to it
            percent_k_values[i] = Some(100.0); 
        } else {
            percent_k_values[i] = Some(((current_close - lowest_low_k) / denominator) * 100.0);
        }
    }

    // Prepare %K values for %D calculation (SMA)
    // Collect only the valid (Some) %K values that will be used as input for SMA for %D
    let mut k_for_sma_input: Vec<f64> = Vec::new();
    // The SMA for %D will start aligning from the first valid %K value.
    // So, we collect all %K values from the first valid one.
    for i in (k_period - 1)..n {
        if let Some(val) = percent_k_values[i] {
            k_for_sma_input.push(val);
        } else {
            // If a %K value is None (e.g. due to bad input slice), 
            // it should still occupy a slot for SMA calculation if SMA handles Option<f64> directly.
            // However, our current SMA takes &[f64]. So, we must decide how to handle intermediate Nones.
            // For simplicity, if %K is None, the SMA over it will also be None for that window.
            // This implies that k_for_sma_input should only contain actual numbers.
            // The SMA function will then produce Nones if its window encounters insufficient data points.
            // This means if percent_k has a None in the middle of its valid range, k_for_sma_input will be shorter.
            // This is generally not expected for %K. Let's assume percent_k values are valid after k_period-1.
        }
    }
    
    let mut percent_d_values = vec![None; n];
    if !k_for_sma_input.is_empty() {
        let sma_of_k = sma(&k_for_sma_input, d_period); // sma helper returns Vec<Option<f64>>
        
        // Align SMA results back to the original data length
        // The first valid %K is at index (k_period - 1)
        // The sma_of_k is calculated on k_values_for_sma, which starts effectively at original index (k_period - 1)
        // So, sma_of_k[j] corresponds to original index (k_period - 1 + j)
        for j in 0..sma_of_k.len() {
            if (k_period - 1 + j) < n { // Ensure we don't write out of bounds for percent_d_values
                 percent_d_values[k_period - 1 + j] = sma_of_k[j];
            }
        }
    }

    (percent_k_values, percent_d_values)
}

#[pyfunction]
#[pyo3(signature = (highs, lows, closes, k_period = 14, d_period = 3))]
pub fn create_stochastic_oscillator_rs(
    gil_token: Python, 
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    k_period: usize,
    d_period: usize,
) -> PyResult<Py<PyDict>> { 
    
    use pyo3::types::{PyDict, PyList}; 

    if highs.len() != lows.len() || highs.len() != closes.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same length."
        ));
    }
    if k_period == 0 || d_period == 0 { 
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k_period and d_period must be positive."
        ));
    }

    let (percent_k, percent_d) = calculate_stochastic_oscillator_raw(&highs, &lows, &closes, k_period, d_period);

    let mut k_objects: Vec<PyObject> = Vec::with_capacity(percent_k.len());
    for item_opt in percent_k.into_iter() { 
        let py_obj = match item_opt { 
            Some(actual_val) => actual_val.to_object(gil_token),
            None => gil_token.None().into(),
        };
        k_objects.push(py_obj);
    }
    let py_percent_k = PyList::new_bound(gil_token, k_objects);

    let mut d_objects: Vec<PyObject> = Vec::with_capacity(percent_d.len());
    for item_opt in percent_d.into_iter() { 
        let py_obj = match item_opt { 
            Some(actual_val) => actual_val.to_object(gil_token),
            None => gil_token.None().into(),
        };
        d_objects.push(py_obj);
    }
    let py_percent_d = PyList::new_bound(gil_token, d_objects);

    let result_dict = PyDict::new_bound(gil_token);
    result_dict.set_item("percent_k", py_percent_k)?;
    result_dict.set_item("percent_d", py_percent_d)?;

    Ok(result_dict.into())
}

// Helper function to register all advanced feature functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_bollinger_bands_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rsi_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_macd_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fibonacci_retracement_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pivot_points_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_volume_profile_rs, m)?)?;
    m.add_function(wrap_pyfunction!(create_stochastic_oscillator_rs, m)?)?; 
    Ok(())
}
