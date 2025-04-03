/*
 * Python bindings for the sentiment analysis optimizations
 */

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::sentiment::continuous_improvement;

/// Initialize the sentiment module
pub fn init_module(m: &PyModule) -> PyResult<()> {
    // Register continuous improvement submodule
    let ci_module = PyModule::new(m.py(), "continuous_improvement")?;
    continuous_improvement::continuous_improvement(m.py(), ci_module)?;
    m.add_submodule(ci_module)?;

    // Register top-level functions (if any)
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_init() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let m = PyModule::new(py, "test_sentiment").unwrap();
            assert!(init_module(m).is_ok());
        });
    }
}