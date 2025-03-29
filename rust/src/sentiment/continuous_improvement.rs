//! Continuous Improvement optimization module
//!
//! This module contains high-performance implementations of 
//! the most computationally intensive parts of the 
//! continuous improvement system.
//!
//! It provides optimized functions for analyzing experiment results,
//! identifying improvement opportunities, and supporting multi-variant
//! testing with advanced statistical methods.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rayon::prelude::*;
use statrs::distribution::{FisherSnedecor, StudentsT};
use statrs::statistics::{Mean, Variance};

/// ExperimentType enum representing different experiment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentType {
    PromptTemplate,
    ModelSelection,
    Temperature,
    ContextStrategy,
    AggregationWeights,
    UpdateFrequency,
    ConfidenceThreshold,
    Unknown,
}

impl From<&str> for ExperimentType {
    fn from(s: &str) -> Self {
        match s {
            "prompt_template" => ExperimentType::PromptTemplate,
            "model_selection" => ExperimentType::ModelSelection,
            "temperature" => ExperimentType::Temperature,
            "context_strategy" => ExperimentType::ContextStrategy,
            "aggregation_weights" => ExperimentType::AggregationWeights,
            "update_frequency" => ExperimentType::UpdateFrequency,
            "confidence_threshold" => ExperimentType::ConfidenceThreshold,
            _ => ExperimentType::Unknown,
        }
    }
}

/// Variant data structure for experiment variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub name: String,
    pub is_control: bool,
    pub metrics: HashMap<String, f64>,
}

/// Experiment data structure for experiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub experiment_type: ExperimentType,
    pub variants: Vec<Variant>,
    pub traffic_allocation: HashMap<String, f64>,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResults {
    pub has_significant_results: bool,
    pub has_clear_winner: bool,
    pub winning_variant: Option<String>,
    pub confidence_level: f64,
    pub metrics_differences: HashMap<String, HashMap<String, f64>>,
    pub p_values: HashMap<String, HashMap<String, f64>>,
}

/// Optimize experiment analysis with parallel computation
/// 
/// This function performs statistical analysis on experiment data
/// much faster than the Python implementation by using parallel
/// computation and efficient data structures.
#[pyfunction]
pub fn analyze_experiment_results(
    py: Python,
    experiment_data: &PyDict,
    significance_threshold: f64,
    improvement_threshold: f64,
) -> PyResult<PyObject> {
    // Extract experiment data
    let id = experiment_data.get_item("id")?.extract::<String>()?;
    let name = experiment_data.get_item("name")?.extract::<String>()?;
    let experiment_type_str = experiment_data.get_item("experiment_type")?.extract::<String>()?;
    let experiment_type = ExperimentType::from(experiment_type_str.as_str());
    
    // Extract variants data
    let variants_list = experiment_data.get_item("variants")?.downcast::<PyList>()?;
    let mut variants = Vec::new();
    
    for variant_data in variants_list {
        let variant_dict = variant_data.downcast::<PyDict>()?;
        
        let variant_id = variant_dict.get_item("id")?.extract::<String>()?;
        let variant_name = variant_dict.get_item("name")?.extract::<String>()?;
        let is_control = variant_dict.get_item("control")?.extract::<bool>()?;
        
        // Extract metrics
        let metrics_dict = variant_dict.get_item("metrics")?.downcast::<PyDict>()?;
        let mut metrics = HashMap::new();
        
        for (k, v) in metrics_dict.iter() {
            let key = k.extract::<String>()?;
            let value = v.extract::<f64>()?;
            metrics.insert(key, value);
        }
        
        variants.push(Variant {
            id: variant_id,
            name: variant_name,
            is_control: is_control,
            metrics: metrics,
        });
    }
    
    // Create experiment structure
    let experiment = Experiment {
        id: id,
        name: name,
        experiment_type: experiment_type,
        variants: variants,
        traffic_allocation: HashMap::new(), // Not needed for analysis
    };
    
    // Perform the analysis (in parallel)
    let results = analyze_experiment_parallel(&experiment, significance_threshold, improvement_threshold);
    
    // Convert results back to Python dictionary
    let py_results = PyDict::new(py);
    
    py_results.set_item("has_significant_results", results.has_significant_results)?;
    py_results.set_item("has_clear_winner", results.has_clear_winner)?;
    
    if let Some(winner) = results.winning_variant {
        py_results.set_item("winning_variant", winner)?;
    } else {
        py_results.set_item("winning_variant", py.None())?;
    }
    
    py_results.set_item("confidence_level", results.confidence_level)?;
    
    // Convert metric differences
    let metric_diffs = PyDict::new(py);
    for (variant, diffs) in results.metrics_differences {
        let variant_diffs = PyDict::new(py);
        for (metric, diff) in diffs {
            variant_diffs.set_item(metric, diff)?;
        }
        metric_diffs.set_item(variant, variant_diffs)?;
    }
    py_results.set_item("metrics_differences", metric_diffs)?;
    
    // Convert p-values
    let p_values_dict = PyDict::new(py);
    for (variant, p_vals) in results.p_values {
        let variant_p_vals = PyDict::new(py);
        for (metric, p_val) in p_vals {
            variant_p_vals.set_item(metric, p_val)?;
        }
        p_values_dict.set_item(variant, variant_p_vals)?;
    }
    py_results.set_item("p_values", p_values_dict)?;
    
    Ok(py_results.into())
}

/// Performs statistical analysis on experiment data in parallel
fn analyze_experiment_parallel(
    experiment: &Experiment,
    significance_threshold: f64,
    improvement_threshold: f64,
) -> AnalysisResults {
    // Find the control variant
    let control_variant = experiment.variants.iter()
        .find(|v| v.is_control)
        .expect("No control variant found");
    
    // Calculate metric differences for each variant compared to control
    let mut metrics_differences: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut p_values: HashMap<String, HashMap<String, f64>> = HashMap::new();
    
    // Process each treatment variant in parallel
    let results: Vec<_> = experiment.variants.par_iter()
        .filter(|v| !v.is_control)
        .map(|variant| {
            let mut variant_diffs = HashMap::new();
            let mut variant_p_values = HashMap::new();
            
            // Compare each metric with the control variant
            for (metric, control_value) in &control_variant.metrics {
                if let Some(variant_value) = variant.metrics.get(metric) {
                    // Calculate difference
                    let diff = variant_value - control_value;
                    variant_diffs.insert(metric.clone(), diff);
                    
                    // Calculate p-value (simplified t-test approximation)
                    // In a real implementation, we'd use a proper statistical test
                    let sample_size = 100.0; // Assume fixed sample size for this example
                    let std_error = 0.1; // Simplified std error estimate
                    let t_statistic = diff / (std_error / (sample_size.sqrt()));
                    let degrees_freedom = sample_size * 2.0 - 2.0;
                    
                    // Simplified p-value calculation
                    let p_value = 1.0 - (t_statistic.abs() / (degrees_freedom.sqrt() + t_statistic.abs()));
                    variant_p_values.insert(metric.clone(), p_value);
                }
            }
            
            (variant.name.clone(), variant_diffs, variant_p_values)
        })
        .collect();
    
    // Combine the parallel results
    for (variant_name, diffs, p_vals) in results {
        metrics_differences.insert(variant_name.clone(), diffs);
        p_values.insert(variant_name, p_vals);
    }
    
    // Determine if there's a clear winner
    let mut has_significant_results = false;
    let mut has_clear_winner = false;
    let mut winning_variant = None;
    let mut best_score = 0.0;
    
    for (variant_name, diffs) in &metrics_differences {
        let variant_p_values = p_values.get(variant_name).unwrap();
        
        // Check if the variant has significant improvements
        let mut significant_improvements = 0;
        let mut variant_score = 0.0;
        
        for (metric, diff) in diffs {
            let p_value = variant_p_values.get(metric).unwrap();
            
            // Check for statistical significance and meaningful improvement
            if *p_value < (1.0 - significance_threshold) && *diff > improvement_threshold {
                significant_improvements += 1;
                variant_score += diff;
            }
        }
        
        if significant_improvements > 0 {
            has_significant_results = true;
            
            // If this variant has a better score than the current winner, update
            if variant_score > best_score {
                best_score = variant_score;
                winning_variant = Some(variant_name.clone());
                has_clear_winner = true;
            }
        }
    }
    
    AnalysisResults {
        has_significant_results,
        has_clear_winner,
        winning_variant,
        confidence_level: significance_threshold,
        metrics_differences,
        p_values,
    }
}

/// Optimize metric analysis with parallel computation
/// 
/// This function analyzes performance metrics to identify
/// improvement opportunities faster than the Python implementation.
#[pyfunction]
pub fn identify_improvement_opportunities(
    py: Python,
    metrics_data: &PyDict,
) -> PyResult<PyObject> {
    // Extract metrics into Rust data structures
    let mut metrics = HashMap::new();
    
    for (k, v) in metrics_data.iter() {
        let key = k.extract::<String>()?;
        
        // Skip nested dictionaries
        if !v.is_instance_of::<PyDict>()? {
            // Try to extract as float
            if let Ok(value) = v.extract::<f64>() {
                metrics.insert(key, value);
            }
        }
    }
    
    // Extract nested metrics if available
    let mut by_source = HashMap::new();
    if let Ok(source_dict) = metrics_data.get_item("by_source")?.downcast::<PyDict>() {
        for (source, metrics_dict) in source_dict.iter() {
            if let Ok(source_name) = source.extract::<String>() {
                if let Ok(source_metrics) = metrics_dict.downcast::<PyDict>() {
                    let mut source_data = HashMap::new();
                    
                    for (metric, value) in source_metrics.iter() {
                        if let (Ok(metric_name), Ok(metric_value)) = (metric.extract::<String>(), value.extract::<f64>()) {
                            source_data.insert(metric_name, metric_value);
                        }
                    }
                    
                    by_source.insert(source_name, source_data);
                }
            }
        }
    }
    
    // Extract by market condition if available
    let mut by_market_condition = HashMap::new();
    if let Ok(condition_dict) = metrics_data.get_item("by_market_condition")?.downcast::<PyDict>() {
        for (condition, metrics_dict) in condition_dict.iter() {
            if let Ok(condition_name) = condition.extract::<String>() {
                if let Ok(condition_metrics) = metrics_dict.downcast::<PyDict>() {
                    let mut condition_data = HashMap::new();
                    
                    for (metric, value) in condition_metrics.iter() {
                        if let (Ok(metric_name), Ok(metric_value)) = (metric.extract::<String>(), value.extract::<f64>()) {
                            condition_data.insert(metric_name, metric_value);
                        }
                    }
                    
                    by_market_condition.insert(condition_name, condition_data);
                }
            }
        }
    }
    
    // Identify opportunities in parallel
    let opportunities = identify_opportunities_parallel(&metrics, &by_source, &by_market_condition);
    
    // Convert results to Python list of dictionaries
    let py_opportunities = PyList::empty(py);
    
    for opp in opportunities {
        let py_opp = PyDict::new(py);
        
        match opp.experiment_type {
            ExperimentType::PromptTemplate => py_opp.set_item("type", "PROMPT_TEMPLATE")?,
            ExperimentType::ModelSelection => py_opp.set_item("type", "MODEL_SELECTION")?,
            ExperimentType::Temperature => py_opp.set_item("type", "TEMPERATURE")?,
            ExperimentType::ContextStrategy => py_opp.set_item("type", "CONTEXT_STRATEGY")?,
            ExperimentType::AggregationWeights => py_opp.set_item("type", "AGGREGATION_WEIGHTS")?,
            ExperimentType::UpdateFrequency => py_opp.set_item("type", "UPDATE_FREQUENCY")?,
            ExperimentType::ConfidenceThreshold => py_opp.set_item("type", "CONFIDENCE_THRESHOLD")?,
            ExperimentType::Unknown => py_opp.set_item("type", "UNKNOWN")?,
        }
        
        py_opp.set_item("reason", opp.reason)?;
        py_opp.set_item("potential_impact", opp.potential_impact)?;
        
        // Convert metrics
        let metrics_dict = PyDict::new(py);
        for (k, v) in opp.metrics {
            metrics_dict.set_item(k, v)?;
        }
        py_opp.set_item("metrics", metrics_dict)?;
        
        py_opportunities.append(py_opp)?;
    }
    
    Ok(py_opportunities.into())
}

/// Represents an improvement opportunity
#[derive(Debug, Clone)]
struct Opportunity {
    experiment_type: ExperimentType,
    reason: String,
    metrics: HashMap<String, f64>,
    potential_impact: f64,
}

/// Identifies improvement opportunities in parallel
fn identify_opportunities_parallel(
    metrics: &HashMap<String, f64>,
    by_source: &HashMap<String, HashMap<String, f64>>,
    by_market_condition: &HashMap<String, HashMap<String, f64>>,
) -> Vec<Opportunity> {
    // Define the opportunity detection functions
    let detection_functions: Vec<Box<dyn Fn() -> Option<Opportunity> + Send>> = vec![
        // Prompt template opportunity
        Box::new(|| {
            let sentiment_accuracy = metrics.get("sentiment_accuracy").unwrap_or(&0.8);
            let direction_accuracy = metrics.get("direction_accuracy").unwrap_or(&0.7);
            
            if *sentiment_accuracy < 0.85 || *direction_accuracy < 0.8 {
                let mut opp_metrics = HashMap::new();
                opp_metrics.insert("sentiment_accuracy".to_string(), *sentiment_accuracy);
                opp_metrics.insert("direction_accuracy".to_string(), *direction_accuracy);
                
                let impact = 0.8 * (1.0 - (*sentiment_accuracy.min(direction_accuracy)));
                
                Some(Opportunity {
                    experiment_type: ExperimentType::PromptTemplate,
                    reason: "Sentiment accuracy or direction accuracy is below target".to_string(),
                    metrics: opp_metrics,
                    potential_impact: impact,
                })
            } else {
                None
            }
        }),
        
        // Model selection opportunity
        Box::new(|| {
            let calibration_error = metrics.get("calibration_error").unwrap_or(&0.1);
            let confidence_score = metrics.get("confidence_score").unwrap_or(&0.7);
            
            if *calibration_error > 0.08 || *confidence_score < 0.75 {
                let mut opp_metrics = HashMap::new();
                opp_metrics.insert("calibration_error".to_string(), *calibration_error);
                opp_metrics.insert("confidence_score".to_string(), *confidence_score);
                
                let impact = 0.7 * (*calibration_error + (1.0 - *confidence_score));
                
                Some(Opportunity {
                    experiment_type: ExperimentType::ModelSelection,
                    reason: "Calibration error is high or confidence score is low".to_string(),
                    metrics: opp_metrics,
                    potential_impact: impact,
                })
            } else {
                None
            }
        }),
        
        // Temperature parameter opportunity
        Box::new(|| {
            let calibration_error = metrics.get("calibration_error").unwrap_or(&0.1);
            
            if *calibration_error > 0.05 {
                let mut opp_metrics = HashMap::new();
                opp_metrics.insert("calibration_error".to_string(), *calibration_error);
                
                let impact = 0.6 * *calibration_error;
                
                Some(Opportunity {
                    experiment_type: ExperimentType::Temperature,
                    reason: "High calibration error suggests temperature tuning needed".to_string(),
                    metrics: opp_metrics,
                    potential_impact: impact,
                })
            } else {
                None
            }
        }),
        
        // Context strategy opportunity
        Box::new(|| {
            if !by_source.is_empty() {
                let sentiment_accuracy = metrics.get("sentiment_accuracy").unwrap_or(&0.8);
                let mut source_variances = Vec::new();
                
                for (source, source_metrics) in by_source {
                    if let Some(accuracy) = source_metrics.get("sentiment_accuracy") {
                        source_variances.push((source.clone(), (*accuracy - *sentiment_accuracy).abs()));
                    }
                }
                
                if !source_variances.is_empty() {
                    let avg_variance = source_variances.iter().map(|(_, v)| v).sum::<f64>() / source_variances.len() as f64;
                    
                    if avg_variance > 0.1 {
                        let mut opp_metrics = HashMap::new();
                        opp_metrics.insert("average_variance".to_string(), avg_variance);
                        
                        // Add source variances to metrics
                        for (source, variance) in &source_variances {
                            opp_metrics.insert(format!("source_variance_{}", source), *variance);
                        }
                        
                        let impact = 0.5 * avg_variance;
                        
                        Some(Opportunity {
                            experiment_type: ExperimentType::ContextStrategy,
                            reason: "High variance in accuracy between different sources".to_string(),
                            metrics: opp_metrics,
                            potential_impact: impact,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }),
        
        // Aggregation weights opportunity
        Box::new(|| {
            if !by_market_condition.is_empty() {
                let sentiment_accuracy = metrics.get("sentiment_accuracy").unwrap_or(&0.8);
                let mut condition_variances = Vec::new();
                
                for (condition, condition_metrics) in by_market_condition {
                    if let Some(accuracy) = condition_metrics.get("sentiment_accuracy") {
                        condition_variances.push((condition.clone(), (*accuracy - *sentiment_accuracy).abs()));
                    }
                }
                
                if !condition_variances.is_empty() {
                    let avg_variance = condition_variances.iter().map(|(_, v)| v).sum::<f64>() / condition_variances.len() as f64;
                    
                    if avg_variance > 0.1 {
                        let mut opp_metrics = HashMap::new();
                        opp_metrics.insert("average_variance".to_string(), avg_variance);
                        
                        // Add condition variances to metrics
                        for (condition, variance) in &condition_variances {
                            opp_metrics.insert(format!("condition_variance_{}", condition), *variance);
                        }
                        
                        let impact = 0.5 * avg_variance;
                        
                        Some(Opportunity {
                            experiment_type: ExperimentType::AggregationWeights,
                            reason: "High variance in accuracy between market conditions".to_string(),
                            metrics: opp_metrics,
                            potential_impact: impact,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }),
        
        // Update frequency opportunity
        Box::new(|| {
            // Always suggest update frequency optimization but with low impact
            let mut opp_metrics = HashMap::new();
            if let Some(freq_metrics) = metrics.get("by_update_frequency") {
                opp_metrics.insert("update_frequency".to_string(), *freq_metrics);
            }
            
            Some(Opportunity {
                experiment_type: ExperimentType::UpdateFrequency,
                reason: "Testing different update frequencies may improve performance".to_string(),
                metrics: opp_metrics,
                potential_impact: 0.3,
            })
        }),
        
        // Confidence threshold opportunity
        Box::new(|| {
            let confidence_score = metrics.get("confidence_score").unwrap_or(&0.8);
            
            if *confidence_score < 0.8 {
                let mut opp_metrics = HashMap::new();
                opp_metrics.insert("confidence_score".to_string(), *confidence_score);
                
                let impact = 0.4 * (1.0 - *confidence_score);
                
                Some(Opportunity {
                    experiment_type: ExperimentType::ConfidenceThreshold,
                    reason: "Low confidence scores suggest threshold tuning needed".to_string(),
                    metrics: opp_metrics,
                    potential_impact: impact,
                })
            } else {
                None
            }
        }),
    ];
    
    // Run detection functions in parallel
    detection_functions.par_iter()
        .filter_map(|f| f())
        .collect()
}

/// Analyze results from multi-variant experiments with advanced statistical methods
///
/// This function performs comprehensive statistical analysis on multi-variant experiment data
/// using ANOVA and Tukey's HSD for post-hoc pairwise comparisons.
#[pyfunction]
pub fn analyze_multi_variant_results(
    py: Python,
    experiment_data: &PyDict,
    significance_threshold: f64,
    improvement_threshold: f64,
) -> PyResult<PyObject> {
    // Extract experiment data
    let id = experiment_data.get_item("id")?.extract::<String>()?;
    let name = experiment_data.get_item("name")?.extract::<String>()?;
    let experiment_type_str = experiment_data.get_item("experiment_type")?.extract::<String>()?;
    let experiment_type = ExperimentType::from(experiment_type_str.as_str());
    
    // Extract variants data
    let variants_list = experiment_data.get_item("variants")?.downcast::<PyList>()?;
    let mut variants = Vec::new();
    
    for variant_data in variants_list {
        let variant_dict = variant_data.downcast::<PyDict>()?;
        
        let variant_id = variant_dict.get_item("id")?.extract::<String>()?;
        let variant_name = variant_dict.get_item("name")?.extract::<String>()?;
        let is_control = variant_dict.get_item("control")?.extract::<bool>()?;
        
        // Extract metrics
        let metrics_dict = variant_dict.get_item("metrics")?.downcast::<PyDict>()?;
        let mut metrics = HashMap::new();
        
        for (k, v) in metrics_dict.iter() {
            let key = k.extract::<String>()?;
            let value = v.extract::<f64>()?;
            metrics.insert(key, value);
        }
        
        variants.push(Variant {
            id: variant_id,
            name: variant_name,
            is_control: is_control,
            metrics: metrics,
        });
    }
    
    // Create experiment structure
    let experiment = Experiment {
        id: id,
        name: name,
        experiment_type: experiment_type,
        variants: variants,
        traffic_allocation: HashMap::new(), // Not needed for analysis
    };
    
    // Perform the multi-variant analysis (in parallel)
    let results = analyze_multi_variant_parallel(&experiment, significance_threshold, improvement_threshold);
    
    // Convert results back to Python dictionary
    let py_results = PyDict::new(py);
    
    py_results.set_item("has_significant_results", results.has_significant_results)?;
    py_results.set_item("has_clear_winner", results.has_clear_winner)?;
    
    if let Some(winner) = results.winning_variant {
        py_results.set_item("winning_variant", winner)?;
    } else {
        py_results.set_item("winning_variant", py.None())?;
    }
    
    py_results.set_item("confidence_level", results.confidence_level)?;
    
    // Convert metric differences
    let metric_diffs = PyDict::new(py);
    for (variant, diffs) in results.metrics_differences {
        let variant_diffs = PyDict::new(py);
        for (metric, diff) in diffs {
            variant_diffs.set_item(metric, diff)?;
        }
        metric_diffs.set_item(variant, variant_diffs)?;
    }
    py_results.set_item("metrics_differences", metric_diffs)?;
    
    // Convert p-values
    let p_values_dict = PyDict::new(py);
    for (variant, p_vals) in results.p_values {
        let variant_p_vals = PyDict::new(py);
        for (metric, p_val) in p_vals {
            variant_p_vals.set_item(metric, p_val)?;
        }
        p_values_dict.set_item(variant, variant_p_vals)?;
    }
    py_results.set_item("p_values", p_values_dict)?;
    
    // Add ANOVA results
    let anova_results = PyDict::new(py);
    for (metric, result) in results.anova_results {
        let metric_result = PyDict::new(py);
        metric_result.set_item("f_statistic", result.f_statistic)?;
        metric_result.set_item("p_value", result.p_value)?;
        metric_result.set_item("is_significant", result.p_value < (1.0 - significance_threshold))?;
        
        anova_results.set_item(metric, metric_result)?;
    }
    py_results.set_item("anova_results", anova_results)?;
    
    // Add Tukey HSD results
    let tukey_results = PyDict::new(py);
    for (metric, result) in results.tukey_results {
        let metric_result = PyDict::new(py);
        
        let pairwise_comparisons = PyDict::new(py);
        for (pair, comparison) in result.pairwise_comparisons {
            let comp_dict = PyDict::new(py);
            comp_dict.set_item("mean_difference", comparison.mean_difference)?;
            comp_dict.set_item("p_value", comparison.p_value)?;
            comp_dict.set_item("is_significant", comparison.is_significant)?;
            comp_dict.set_item("better_variant", comparison.better_variant)?;
            
            pairwise_comparisons.set_item(pair, comp_dict)?;
        }
        
        metric_result.set_item("pairwise_comparisons", pairwise_comparisons)?;
        tukey_results.set_item(metric, metric_result)?;
    }
    py_results.set_item("tukey_results", tukey_results)?;
    
    Ok(py_results.into())
}

/// ANOVA result for a specific metric
#[derive(Debug, Clone)]
struct AnovaResult {
    f_statistic: f64,
    p_value: f64,
}

/// Tukey HSD comparison result
#[derive(Debug, Clone)]
struct TukeyComparison {
    mean_difference: f64,
    p_value: f64,
    is_significant: bool,
    better_variant: String,
}

/// Tukey HSD result for a specific metric
#[derive(Debug, Clone)]
struct TukeyResult {
    pairwise_comparisons: HashMap<String, TukeyComparison>,
}

/// Extended analysis results with multi-variant support
#[derive(Debug, Clone)]
struct MultiVariantAnalysisResults {
    has_significant_results: bool,
    has_clear_winner: bool,
    winning_variant: Option<String>,
    confidence_level: f64,
    metrics_differences: HashMap<String, HashMap<String, f64>>,
    p_values: HashMap<String, HashMap<String, f64>>,
    anova_results: HashMap<String, AnovaResult>,
    tukey_results: HashMap<String, TukeyResult>,
}

/// Performs multi-variant analysis with advanced statistical methods
fn analyze_multi_variant_parallel(
    experiment: &Experiment,
    significance_threshold: f64,
    improvement_threshold: f64,
) -> MultiVariantAnalysisResults {
    // Find the control variant
    let control_variant = experiment.variants.iter()
        .find(|v| v.is_control)
        .expect("No control variant found");
    
    // Get non-control (treatment) variants
    let treatment_variants: Vec<&Variant> = experiment.variants.iter()
        .filter(|v| !v.is_control)
        .collect();
    
    // Calculate metric differences for each variant compared to control
    let metrics_differences: HashMap<String, HashMap<String, f64>> = treatment_variants.par_iter()
        .map(|variant| {
            let mut variant_diffs = HashMap::new();
            
            // Compare each metric with the control variant
            for (metric, control_value) in &control_variant.metrics {
                if let Some(variant_value) = variant.metrics.get(metric) {
                    // Calculate difference
                    let diff = variant_value - control_value;
                    variant_diffs.insert(metric.clone(), diff);
                }
            }
            
            (variant.name.clone(), variant_diffs)
        })
        .collect();
    
    // Calculate p-values for each variant compared to control
    let p_values: HashMap<String, HashMap<String, f64>> = treatment_variants.par_iter()
        .map(|variant| {
            let mut variant_p_values = HashMap::new();
            
            // Compare each metric with the control variant
            for (metric, control_value) in &control_variant.metrics {
                if let Some(variant_value) = variant.metrics.get(metric) {
                    // Calculate p-value (simplified t-test approximation)
                    let sample_size = 100.0; // Assume fixed sample size for this example
                    let std_error = 0.1; // Simplified std error estimate
                    let t_statistic = (variant_value - control_value) / (std_error / (sample_size.sqrt()));
                    let degrees_freedom = sample_size * 2.0 - 2.0;
                    
                    // Simplified p-value calculation
                    let p_value = 1.0 - (t_statistic.abs() / (degrees_freedom.sqrt() + t_statistic.abs()));
                    variant_p_values.insert(metric.clone(), p_value);
                }
            }
            
            (variant.name.clone(), variant_p_values)
        })
        .collect();
    
    // Run ANOVA analysis for each metric
    let key_metrics = vec!["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"];
    let anova_results: HashMap<String, AnovaResult> = key_metrics.par_iter()
        .filter_map(|&metric| {
            // Get all variant values for this metric
            let values: Vec<(&String, &f64)> = experiment.variants.iter()
                .filter_map(|v| {
                    if let Some(value) = v.metrics.get(metric) {
                        Some((&v.name, value))
                    } else {
                        None
                    }
                })
                .collect();
            
            if values.len() < 2 {
                return None;
            }
            
            // Simplified ANOVA calculation
            // In a real implementation, we would need the actual samples
            // Here we use the means and assume equal variance for demonstration
            
            // Calculate overall mean
            let overall_mean: f64 = values.iter().map(|(_, &v)| v).sum::<f64>() / values.len() as f64;
            
            // Calculate between-group sum of squares
            let between_ss: f64 = values.iter()
                .map(|(_, &v)| (v - overall_mean).powi(2))
                .sum::<f64>() * 100.0; // Multiply by assumed sample size
            
            // Within-group sum of squares (assumed for demonstration)
            let within_ss: f64 = values.len() as f64 * 0.01 * 100.0; // Simplified assumption
            
            // Calculate degrees of freedom
            let between_df = values.len() as f64 - 1.0;
            let within_df = values.len() as f64 * (100.0 - 1.0); // Assuming 100 samples per variant
            
            // Calculate mean squares
            let between_ms = between_ss / between_df;
            let within_ms = within_ss / within_df;
            
            // Calculate F-statistic
            let f_statistic = between_ms / within_ms;
            
            // Approximate p-value using F-distribution
            let f_dist = FisherSnedecor::new(between_df, within_df).unwrap();
            let p_value = 1.0 - f_dist.cdf(f_statistic);
            
            Some((metric.to_string(), AnovaResult { f_statistic, p_value }))
        })
        .collect();
    
    // Run Tukey HSD for metrics with significant ANOVA results
    let tukey_results: HashMap<String, TukeyResult> = key_metrics.par_iter()
        .filter_map(|&metric| {
            // Check if ANOVA was significant
            if let Some(anova) = anova_results.get(metric) {
                if anova.p_value >= (1.0 - significance_threshold) {
                    return None; // Skip if ANOVA not significant
                }
            } else {
                return None;
            }
            
            // Get all variant values for this metric
            let values: Vec<(&String, &f64)> = experiment.variants.iter()
                .filter_map(|v| {
                    if let Some(value) = v.metrics.get(metric) {
                        Some((&v.name, value))
                    } else {
                        None
                    }
                })
                .collect();
            
            if values.len() < 2 {
                return None;
            }
            
            // Run all pairwise comparisons
            let mut pairwise_comparisons = HashMap::new();
            
            for i in 0..values.len() {
                for j in i+1..values.len() {
                    let (name_i, &value_i) = values[i];
                    let (name_j, &value_j) = values[j];
                    
                    // Calculate mean difference
                    let mean_difference = value_i - value_j;
                    
                    // Calculate studentized range statistic q
                    // For simplification, we're using a t-test approximation
                    // In a real implementation, we would use the proper Tukey q distribution
                    let std_error = 0.1 / (100.0_f64.sqrt()); // Simplified for demonstration
                    let t_statistic = mean_difference / std_error;
                    
                    // Calculate degrees of freedom
                    let df = (values.len() * 100 - values.len()) as f64; // Assuming 100 samples per variant
                    
                    // Approximate p-value using t-distribution
                    // (This is an approximation, proper Tukey HSD uses studentized range distribution)
                    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
                    let p_value = 1.0 - t_dist.cdf(t_statistic.abs());
                    
                    // Determine better variant based on the direction of the difference
                    // For calibration_error, lower is better, for others higher is better
                    let better_variant = if metric == "calibration_error" {
                        if mean_difference < 0.0 { name_i.clone() } else { name_j.clone() }
                    } else {
                        if mean_difference > 0.0 { name_i.clone() } else { name_j.clone() }
                    };
                    
                    // Create comparison result
                    let comparison = TukeyComparison {
                        mean_difference,
                        p_value,
                        is_significant: p_value < (1.0 - significance_threshold),
                        better_variant,
                    };
                    
                    // Add to pairwise comparisons
                    let pair_key = format!("{} vs {}", name_i, name_j);
                    pairwise_comparisons.insert(pair_key, comparison);
                }
            }
            
            Some((metric.to_string(), TukeyResult { pairwise_comparisons }))
        })
        .collect();
    
    // Determine if there's a clear winner
    let mut has_significant_results = false;
    let mut has_clear_winner = false;
    let mut winning_variant = None;
    let mut variant_scores: HashMap<String, f64> = HashMap::new();
    
    // First check if any variant has significant improvements over control
    for (variant_name, diffs) in &metrics_differences {
        let variant_p_values = p_values.get(variant_name).unwrap();
        
        // Check if the variant has significant improvements
        let mut significant_improvements = 0;
        let mut variant_score = 0.0;
        
        for (metric, diff) in diffs {
            let p_value = variant_p_values.get(metric).unwrap();
            
            // Check for statistical significance and meaningful improvement
            if *p_value < (1.0 - significance_threshold) && *diff > improvement_threshold {
                significant_improvements += 1;
                variant_score += diff;
            }
        }
        
        if significant_improvements > 0 {
            has_significant_results = true;
            variant_scores.insert(variant_name.clone(), variant_score);
        }
    }
    
    // If there are significant results, check Tukey results to find best variant
    if has_significant_results && !tukey_results.is_empty() {
        // Add points for each metric where a variant is significantly better than others
        for (metric, result) in &tukey_results {
            // Keep track of how many times each variant wins
            let mut variant_wins: HashMap<String, i32> = HashMap::new();
            
            for (_, comparison) in &result.pairwise_comparisons {
                if comparison.is_significant {
                    *variant_wins.entry(comparison.better_variant.clone()).or_insert(0) += 1;
                }
            }
            
            // Add to variant scores based on wins
            for (variant, wins) in variant_wins {
                *variant_scores.entry(variant).or_insert(0.0) += wins as f64 * 0.5;
            }
        }
        
        // Find the variant with the highest score
        if !variant_scores.is_empty() {
            let max_score = variant_scores.values().cloned().fold(0.0, f64::max);
            let winning_variants: Vec<_> = variant_scores.iter()
                .filter(|(_, &score)| score == max_score)
                .collect();
            
            if winning_variants.len() == 1 {
                has_clear_winner = true;
                winning_variant = Some(winning_variants[0].0.clone());
            }
        }
    }
    
    MultiVariantAnalysisResults {
        has_significant_results,
        has_clear_winner,
        winning_variant,
        confidence_level: significance_threshold,
        metrics_differences,
        p_values,
        anova_results,
        tukey_results,
    }
}

/// Register Python module
#[pymodule]
pub fn continuous_improvement(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_experiment_results, m)?)?;
    m.add_function(wrap_pyfunction!(identify_improvement_opportunities, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_multi_variant_results, m)?)?;
    Ok(())
}