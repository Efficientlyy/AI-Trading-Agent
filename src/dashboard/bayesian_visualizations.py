"""
Bayesian analysis visualization utilities for the continuous improvement system.

This module provides specialized visualization components for Bayesian analysis results
and multi-variant experiments, offering rich insights into experiment performance.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from scipy import stats

from src.analysis_agents.sentiment.ab_testing import (
    Experiment, ExperimentVariant, ExperimentMetrics, ExperimentStatus
)
from src.analysis_agents.sentiment.continuous_improvement.bayesian_analysis import (
    BayesianAnalysisResults, BayesianAnalyzer
)
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    stopping_criteria_manager
)


def create_posterior_distribution_plot(
    posterior_samples: Dict[str, np.ndarray],
    metric_name: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a plot showing posterior distributions for different variants.
    
    Args:
        posterior_samples: Dictionary of posterior samples for each variant
        metric_name: Name of the metric being visualized
        title: Optional custom title
        
    Returns:
        Plotly figure object
    """
    if not posterior_samples:
        fig = go.Figure()
        fig.add_annotation(text="No posterior samples available", showarrow=False, font=dict(size=16))
        return fig
    
    fig = go.Figure()
    
    # Determine appropriate range for x-axis
    all_samples = np.concatenate(list(posterior_samples.values()))
    x_min = np.min(all_samples) - 0.05
    x_max = np.max(all_samples) + 0.05
    
    # Generate kernel density estimation for each variant
    x = np.linspace(x_min, x_max, 1000)
    
    for variant_name, samples in posterior_samples.items():
        # Use KDE to get smooth density
        kde = stats.gaussian_kde(samples)
        y = kde(x)
        
        # Add trace for this variant
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=variant_name,
            fill='tozeroy',
            fillcolor=f'rgba(0, 0, 0, 0.1)',
            line=dict(width=2)
        ))
        
        # Add vertical line for mean
        mean_value = np.mean(samples)
        fig.add_trace(go.Scatter(
            x=[mean_value, mean_value],
            y=[0, max(kde(mean_value), 0.001)],
            mode='lines',
            name=f'{variant_name} mean',
            line=dict(color=fig.data[-1].line.color, width=2, dash='dash'),
            showlegend=False
        ))
    
    # Set title
    plot_title = title or f"Posterior Distributions for {metric_name.replace('_', ' ').title()}"
    
    # Handle calibration_error specially as lower is better
    if metric_name == "calibration_error":
        plot_title += " (Lower is Better)"
    
    fig.update_layout(
        title=plot_title,
        xaxis_title=metric_name.replace('_', ' ').title(),
        yaxis_title="Density",
        hovermode="x unified",
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_winning_probability_chart(
    winning_probability: Dict[str, Dict[str, float]],
    experiment_name: str
) -> go.Figure:
    """
    Create a bar chart showing winning probabilities for each variant.
    
    Args:
        winning_probability: Dictionary of winning probabilities by metric and variant
        experiment_name: Name of the experiment
        
    Returns:
        Plotly figure object
    """
    if not winning_probability:
        fig = go.Figure()
        fig.add_annotation(text="No winning probability data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Prepare data for visualization
    data = []
    
    for metric_name, probabilities in winning_probability.items():
        for variant_name, probability in probabilities.items():
            data.append({
                'metric': metric_name.replace('_', ' ').title(),
                'variant': variant_name,
                'probability': probability,
                'percentage': probability * 100
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create visualization
    fig = px.bar(
        df,
        x='variant',
        y='percentage',
        color='metric',
        barmode='group',
        title=f"Probability of Being Best for '{experiment_name}'",
        labels={
            'variant': 'Variant',
            'percentage': 'Probability (%)',
            'metric': 'Metric'
        },
        text='percentage',
        text_auto='.0f'
    )
    
    # Add threshold line at 95%
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(df['variant'].unique()) - 0.5,
        y0=95,
        y1=95,
        line=dict(color="red", width=2, dash="dash"),
        name="95% Threshold"
    )
    
    # Add annotation for threshold
    fig.add_annotation(
        x=len(df['variant'].unique()) - 0.5,
        y=95,
        text="95% Threshold",
        showarrow=False,
        yshift=10,
        xshift=-5,
        font=dict(color="red")
    )
    
    fig.update_layout(
        yaxis_range=[0, 105],
        yaxis_title="Probability of Being Best (%)",
        xaxis_title="Variant",
        legend_title="Metric",
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig


def create_lift_estimation_chart(
    lift_estimation: Dict[str, Dict[str, Dict[str, float]]],
    experiment_name: str
) -> go.Figure:
    """
    Create an error bar chart showing estimated lift with confidence intervals.
    
    Args:
        lift_estimation: Dictionary of lift estimations by metric and variant
        experiment_name: Name of the experiment
        
    Returns:
        Plotly figure object
    """
    if not lift_estimation:
        fig = go.Figure()
        fig.add_annotation(text="No lift estimation data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Prepare data for visualization
    data = []
    
    for metric_name, variants in lift_estimation.items():
        for variant_name, estimates in variants.items():
            if all(k in estimates for k in ["mean", "credible_interval"]):
                mean = estimates["mean"] * 100  # Convert to percentage
                ci_low, ci_high = estimates["credible_interval"]
                ci_low *= 100
                ci_high *= 100
                
                data.append({
                    'metric': metric_name.replace('_', ' ').title(),
                    'variant': variant_name,
                    'mean': mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_width': ci_high - ci_low
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by metric and mean lift
    df = df.sort_values(['metric', 'mean'], ascending=[True, False])
    
    # Create figure
    fig = go.Figure()
    
    # Get unique metrics
    metrics = df['metric'].unique()
    
    # Calculate offset for grouped bars
    num_metrics = len(metrics)
    offsets = np.linspace(-0.3, 0.3, num_metrics)
    
    # Get unique variants
    variants = df['variant'].unique()
    
    # Add shapes for each metric group
    for i, variant in enumerate(variants):
        # Add vertical gridlines between variants
        if i > 0:
            fig.add_shape(
                type="line",
                x0=i - 0.5,
                x1=i - 0.5,
                y0=df['ci_low'].min() - 5,
                y1=df['ci_high'].max() + 5,
                line=dict(color="lightgrey", width=1)
            )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(variants) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1)
    )
    
    # Add bars and error bars for each metric
    for i, metric in enumerate(metrics):
        metric_data = df[df['metric'] == metric]
        
        # Create mapping of variant to x-position
        variant_positions = {variant: j + offsets[i] for j, variant in enumerate(variants)}
        
        # Add bar for each variant
        for _, row in metric_data.iterrows():
            variant = row['variant']
            x_pos = variant_positions.get(variant, 0)
            
            # Add bar
            fig.add_trace(go.Bar(
                x=[x_pos],
                y=[row['mean']],
                name=f"{metric} - {variant}",
                legendgroup=metric,
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                width=0.8 / num_metrics,
                text=[f"{row['mean']:.1f}%"],
                textposition='outside'
            ))
            
            # Add error bars
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[row['ci_low'], row['ci_high']],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                showlegend=False,
                legendgroup=metric
            ))
            
            # Add caps to error bars
            cap_width = 0.4 / num_metrics
            
            # Lower cap
            fig.add_trace(go.Scatter(
                x=[x_pos - cap_width/2, x_pos + cap_width/2],
                y=[row['ci_low'], row['ci_low']],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                showlegend=False,
                legendgroup=metric
            ))
            
            # Upper cap
            fig.add_trace(go.Scatter(
                x=[x_pos - cap_width/2, x_pos + cap_width/2],
                y=[row['ci_high'], row['ci_high']],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                showlegend=False,
                legendgroup=metric
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Estimated Lift Over Control for '{experiment_name}'",
        xaxis=dict(
            title="Variant",
            tickmode='array',
            tickvals=list(range(len(variants))),
            ticktext=variants
        ),
        yaxis=dict(
            title="Lift (%)",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        barmode='group',
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            groupclick="toggleitem"
        )
    )
    
    return fig


def create_experiment_progress_chart(
    experiment: Experiment,
    stopping_criteria_results: Dict[str, Any]
) -> go.Figure:
    """
    Create a chart showing experiment progress toward stopping criteria.
    
    Args:
        experiment: Experiment object
        stopping_criteria_results: Results from evaluating stopping criteria
        
    Returns:
        Plotly figure object
    """
    if not experiment or not stopping_criteria_results:
        fig = go.Figure()
        fig.add_annotation(text="No experiment data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sample size information
    variant_requests = {}
    
    for variant in experiment.variants:
        metrics = experiment.variant_metrics[variant.id]
        variant_requests[variant.name] = metrics.requests
    
    # Add bar chart for current sample sizes
    fig.add_trace(
        go.Bar(
            x=list(variant_requests.keys()),
            y=list(variant_requests.values()),
            name="Current Samples",
            text=list(variant_requests.values()),
            textposition='outside'
        )
    )
    
    # Get target sample size if defined
    target_sample_size = experiment.sample_size
    
    # If target sample size exists, add a line
    if target_sample_size:
        fig.add_trace(
            go.Scatter(
                x=list(variant_requests.keys()),
                y=[target_sample_size] * len(variant_requests),
                mode='lines',
                name="Target Sample Size",
                line=dict(color='red', width=2, dash='dash')
            )
        )
    
    # Add information about stopping criteria
    criteria_results = stopping_criteria_results.get("criteria_results", {})
    
    # Create data for secondary y-axis with probabilities
    probability_data = []
    
    for criterion_name, result in criteria_results.items():
        if "bayesian_probability" in criterion_name:
            # Extract the current probability from the reason string
            reason = result.get("reason", "")
            probability = 0.0
            
            # Try to parse the probability from the reason
            import re
            probability_match = re.search(r'(\d+\.\d+)%', reason)
            if probability_match:
                try:
                    probability = float(probability_match.group(1)) / 100.0
                except ValueError:
                    pass
            
            # If the criterion says to stop, use 1.0 as the probability
            if result.get("should_stop", False):
                probability = 1.0
            
            probability_data.append({
                "criterion": criterion_name,
                "probability": probability,
                "should_stop": result.get("should_stop", False)
            })
    
    # If we have probability data, add it to the secondary y-axis
    if probability_data:
        df_prob = pd.DataFrame(probability_data)
        
        # Add trace for each criterion
        for _, row in df_prob.iterrows():
            color = "green" if row["should_stop"] else "orange"
            
            fig.add_trace(
                go.Scatter(
                    x=[variant_requests.keys()[-1]],
                    y=[row["probability"]],
                    mode='markers',
                    name=f"{row['criterion']} ({row['probability']:.2f})",
                    marker=dict(size=15, color=color),
                    text=[f"{row['probability']:.2f}"],
                    textposition="top center"
                ),
                secondary_y=True
            )
        
        # Add threshold line at 0.95
        fig.add_trace(
            go.Scatter(
                x=list(variant_requests.keys()),
                y=[0.95] * len(variant_requests),
                mode='lines',
                name="Probability Threshold (0.95)",
                line=dict(color='green', width=2, dash='dash')
            ),
            secondary_y=True
        )
    
    # Set titles
    fig.update_layout(
        title=f"Experiment Progress: {experiment.name}",
        xaxis_title="Variant",
        yaxis_title="Sample Size",
        barmode='group',
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Update yaxis properties
    fig.update_yaxes(title_text="Sample Size", secondary_y=False)
    fig.update_yaxes(title_text="Probability", secondary_y=True, range=[0, 1.05])
    
    return fig


def create_expected_loss_chart(
    expected_loss: Dict[str, Dict[str, float]],
    experiment_name: str
) -> go.Figure:
    """
    Create a chart showing expected loss (regret) for each variant.
    
    Args:
        expected_loss: Dictionary of expected loss by metric and variant
        experiment_name: Name of the experiment
        
    Returns:
        Plotly figure object
    """
    if not expected_loss:
        fig = go.Figure()
        fig.add_annotation(text="No expected loss data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Prepare data for visualization
    data = []
    
    for metric_name, losses in expected_loss.items():
        for variant_name, loss in losses.items():
            data.append({
                'metric': metric_name.replace('_', ' ').title(),
                'variant': variant_name,
                'loss': loss,
                'loss_percentage': loss * 100
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by metric and loss (ascending)
    df = df.sort_values(['metric', 'loss'], ascending=[True, True])
    
    # Create visualization
    fig = px.bar(
        df,
        x='variant',
        y='loss',
        color='metric',
        barmode='group',
        title=f"Expected Loss (Regret) for '{experiment_name}'",
        labels={
            'variant': 'Variant',
            'loss': 'Expected Loss',
            'metric': 'Metric'
        },
        text='loss',
        text_auto='.4f'
    )
    
    # Add threshold line at 0.005 (common threshold)
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(df['variant'].unique()) - 0.5,
        y0=0.005,
        y1=0.005,
        line=dict(color="red", width=2, dash="dash"),
        name="Loss Threshold"
    )
    
    # Add annotation for threshold
    fig.add_annotation(
        x=len(df['variant'].unique()) - 0.5,
        y=0.005,
        text="Loss Threshold (0.005)",
        showarrow=False,
        yshift=10,
        xshift=-5,
        font=dict(color="red")
    )
    
    fig.update_layout(
        yaxis_title="Expected Loss (Regret)",
        xaxis_title="Variant",
        legend_title="Metric",
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig


def create_credible_interval_chart(
    credible_intervals: Dict[str, Dict[str, Dict[str, List[float]]]],
    experiment_name: str
) -> go.Figure:
    """
    Create a chart showing credible intervals for each variant.
    
    Args:
        credible_intervals: Dictionary of credible intervals by metric, variant, and interval type
        experiment_name: Name of the experiment
        
    Returns:
        Plotly figure object
    """
    if not credible_intervals:
        fig = go.Figure()
        fig.add_annotation(text="No credible interval data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Prepare data for visualization
    data = []
    
    for metric_name, variants in credible_intervals.items():
        for variant_name, intervals in variants.items():
            if "95%" in intervals:
                ci_low, ci_high = intervals["95%"]
                ci_width = ci_high - ci_low
                
                data.append({
                    'metric': metric_name.replace('_', ' ').title(),
                    'variant': variant_name,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_width': ci_width
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    fig = go.Figure()
    
    # Get unique metrics and variants
    metrics = df['metric'].unique()
    variants = df['variant'].unique()
    
    # Create groups by metric
    for i, metric in enumerate(metrics):
        color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        metric_data = df[df['metric'] == metric]
        
        # Add a trace for each variant's credible interval
        for j, (_, row) in enumerate(metric_data.iterrows()):
            variant = row['variant']
            
            # Horizontal position calculation
            x_pos = list(variants).index(variant)
            
            # Add point for mean (average of CI bounds for simplicity)
            mean_value = (row['ci_low'] + row['ci_high']) / 2
            
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[mean_value],
                mode='markers',
                marker=dict(color=color, size=10),
                name=f"{metric} - {variant}",
                legendgroup=metric,
                text=[f"Mean: {mean_value:.4f}"],
                hoverinfo='text'
            ))
            
            # Add error bars
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[row['ci_low'], row['ci_high']],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=metric,
                text=[f"Lower: {row['ci_low']:.4f}", f"Upper: {row['ci_high']:.4f}"],
                hoverinfo='text'
            ))
            
            # Add caps to error bars
            cap_width = 0.2
            
            # Lower cap
            fig.add_trace(go.Scatter(
                x=[x_pos - cap_width/2, x_pos + cap_width/2],
                y=[row['ci_low'], row['ci_low']],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=metric
            ))
            
            # Upper cap
            fig.add_trace(go.Scatter(
                x=[x_pos - cap_width/2, x_pos + cap_width/2],
                y=[row['ci_high'], row['ci_high']],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=metric
            ))
            
            # Add width annotation
            fig.add_annotation(
                x=x_pos,
                y=row['ci_high'],
                text=f"Width: {row['ci_width']:.4f}",
                showarrow=False,
                yshift=15,
                font=dict(size=10)
            )
    
    # Update layout
    fig.update_layout(
        title=f"95% Credible Intervals for '{experiment_name}'",
        xaxis=dict(
            title="Variant",
            tickmode='array',
            tickvals=list(range(len(variants))),
            ticktext=variants
        ),
        yaxis=dict(
            title="Value",
            zeroline=True
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            groupclick="toggleitem"
        )
    )
    
    return fig


def create_multi_variant_comparison_chart(
    experiment: Experiment,
    analysis_results: Dict[str, Any]
) -> go.Figure:
    """
    Create a comprehensive comparison chart for multi-variant experiments.
    
    Args:
        experiment: Experiment object
        analysis_results: Analysis results (including ANOVA and Tukey HSD)
        
    Returns:
        Plotly figure object
    """
    if not experiment or not analysis_results:
        fig = go.Figure()
        fig.add_annotation(text="No multi-variant analysis data available", showarrow=False, font=dict(size=16))
        return fig
    
    # Extract ANOVA and Tukey HSD results
    anova_results = analysis_results.get("anova_results", {})
    tukey_results = analysis_results.get("tukey_results", {})
    
    # Prepare data for visualization
    data = []
    
    # Add variant metrics
    for variant in experiment.variants:
        metrics = experiment.variant_metrics[variant.id].to_dict()
        
        for metric_name, value in metrics.items():
            # Skip non-numeric metrics
            if not isinstance(value, (int, float)):
                continue
                
            # Skip metrics like requests, successes, etc.
            if metric_name in ["requests", "successes", "errors", "total_latency", "user_overrides"]:
                continue
            
            data.append({
                'metric': metric_name.replace('_', ' ').title(),
                'variant': variant.name,
                'value': value,
                'is_control': variant.control
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create figure with multiple subplots (one for each metric)
    metrics = df['metric'].unique()
    num_metrics = len(metrics)
    
    fig = make_subplots(
        rows=num_metrics, 
        cols=1,
        subplot_titles=[m for m in metrics],
        vertical_spacing=0.1
    )
    
    # Create a plot for each metric
    for i, metric in enumerate(metrics):
        metric_data = df[df['metric'] == metric]
        
        # Sort by control status first, then by value
        if metric == "Calibration Error":  # Lower is better
            metric_data = metric_data.sort_values(['is_control', 'value'], ascending=[False, True])
        else:  # Higher is better
            metric_data = metric_data.sort_values(['is_control', 'value'], ascending=[False, False])
        
        # Add bar chart
        colors = ['rgba(99, 110, 250, 0.7)' if not is_control else 'rgba(239, 85, 59, 0.7)' 
                 for is_control in metric_data['is_control']]
        
        fig.add_trace(
            go.Bar(
                x=metric_data['variant'],
                y=metric_data['value'],
                marker_color=colors,
                text=metric_data['value'].round(4),
                textposition='outside',
                name=metric,
                showlegend=(i == 0)  # Only show legend for the first metric
            ),
            row=i+1, col=1
        )
        
        # Add ANOVA results if available
        if metric.lower().replace(' ', '_') in anova_results:
            anova_metric = anova_results[metric.lower().replace(' ', '_')]
            
            if "p_value" in anova_metric and "f_statistic" in anova_metric:
                p_value = anova_metric["p_value"]
                f_stat = anova_metric["f_statistic"]
                is_significant = anova_metric.get("is_significant", False)
                
                annotation_color = "green" if is_significant else "red"
                significance_text = "Significant" if is_significant else "Not Significant"
                
                fig.add_annotation(
                    x=0.02, y=0.95,
                    text=f"ANOVA: F={f_stat:.2f}, p={p_value:.4f} ({significance_text})",
                    xref=f"x{i+1} domain", yref=f"y{i+1} domain",
                    showarrow=False,
                    font=dict(color=annotation_color, size=10),
                    align="left"
                )
                
                # Add Tukey results if significant and available
                if is_significant and metric.lower().replace(' ', '_') in tukey_results:
                    tukey_metric = tukey_results[metric.lower().replace(' ', '_')]
                    pairwise_results = tukey_metric.get("pairwise_comparisons", {})
                    
                    # Find significant pairwise differences
                    significant_pairs = []
                    for pair, pair_results in pairwise_results.items():
                        if pair_results.get("is_significant", False):
                            better_variant = pair_results.get("better_variant", "")
                            significant_pairs.append(f"{pair}: {better_variant} better")
                    
                    # Add annotations for significant pairs (max 3)
                    for j, pair_text in enumerate(significant_pairs[:3]):
                        fig.add_annotation(
                            x=0.02, y=0.85 - j*0.06,
                            text=f"Tukey: {pair_text}",
                            xref=f"x{i+1} domain", yref=f"y{i+1} domain",
                            showarrow=False,
                            font=dict(color="green", size=8),
                            align="left"
                        )
    
    # Update layout
    fig.update_layout(
        title=f"Multi-Variant Comparison: {experiment.name}",
        height=300 * num_metrics,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=True,
        legend=dict(
            title="Variant Type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def generate_experiment_visualizations(
    experiment: Experiment,
    stopping_criteria_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate multiple visualizations for an experiment.
    
    Args:
        experiment: Experiment object
        stopping_criteria_results: Results from evaluating stopping criteria
        
    Returns:
        Dictionary of visualization figures
    """
    visualizations = {}
    
    # Run Bayesian analysis
    bayesian_analyzer = BayesianAnalyzer()
    bayesian_results = bayesian_analyzer.analyze_experiment(experiment)
    
    # Generate visualizations based on analysis results
    if bayesian_results:
        # Add winning probability chart
        if bayesian_results.winning_probability:
            visualizations["winning_probability"] = create_winning_probability_chart(
                bayesian_results.winning_probability,
                experiment.name
            )
        
        # Add lift estimation chart
        if bayesian_results.lift_estimation:
            visualizations["lift_estimation"] = create_lift_estimation_chart(
                bayesian_results.lift_estimation,
                experiment.name
            )
        
        # Add posterior distribution plots for key metrics
        for metric in ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]:
            if metric in bayesian_results.posterior_samples:
                visualizations[f"posterior_{metric}"] = create_posterior_distribution_plot(
                    bayesian_results.posterior_samples[metric],
                    metric
                )
        
        # Add expected loss chart
        if bayesian_results.expected_loss:
            visualizations["expected_loss"] = create_expected_loss_chart(
                bayesian_results.expected_loss,
                experiment.name
            )
        
        # Add credible interval chart
        if bayesian_results.credible_intervals:
            visualizations["credible_intervals"] = create_credible_interval_chart(
                bayesian_results.credible_intervals,
                experiment.name
            )
    
    # Add experiment progress visualization if stopping criteria results are provided
    if stopping_criteria_results:
        visualizations["experiment_progress"] = create_experiment_progress_chart(
            experiment,
            stopping_criteria_results
        )
    
    # Add multi-variant comparison if there are analysis results in the experiment
    if hasattr(experiment, 'results') and experiment.results:
        visualizations["multi_variant_comparison"] = create_multi_variant_comparison_chart(
            experiment,
            experiment.results
        )
    
    return visualizations