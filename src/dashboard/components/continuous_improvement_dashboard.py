"""Continuous Improvement dashboard component.

This module provides a dashboard for monitoring and managing the
automated improvement system for sentiment analysis.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.analysis_agents.sentiment.continuous_improvement import continuous_improvement_manager
from src.analysis_agents.sentiment.ab_testing import ab_testing_framework, ExperimentType
from src.common.logging import get_logger


# Initialize logger
logger = get_logger("dashboard", "continuous_improvement")


def create_layout():
    """Create the continuous improvement dashboard layout.
    
    Returns:
        Dash layout
    """
    return html.Div([
        html.H2("Continuous Improvement System", className="mt-4 mb-4"),
        
        # Controls and refresh
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Refresh Data", id="refresh-ci-data", color="primary", className="mr-2"),
                    dcc.Interval(id="ci-refresh-interval", interval=30000, n_intervals=0),  # 30 seconds
                ], width=3),
                dbc.Col([
                    dbc.Button("Generate Experiments", id="generate-experiments-btn", color="success", className="mr-2"),
                    dbc.Button("Run Maintenance", id="run-maintenance-btn", color="warning")
                ], width=4),
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Auto-Implementation:"),
                        dbc.Select(
                            id="auto-implement-toggle",
                            options=[
                                {"label": "Enabled", "value": "enabled"},
                                {"label": "Disabled", "value": "disabled"}
                            ],
                            value="disabled"
                        )
                    ], className="mb-0")
                ], width=3)
            ])
        ], className="mb-4"),
        
        # System status card
        dbc.Card([
            dbc.CardHeader("System Status"),
            dbc.CardBody(id="ci-system-status")
        ], className="mb-4"),
        
        # Tab navigation
        dbc.Tabs([
            # Improvement History Tab
            dbc.Tab([
                html.Div([
                    html.H4("Improvement History", className="mt-3"),
                    html.Div(id="improvement-history-container")
                ])
            ], label="Improvement History", tab_id="history-tab"),
            
            # Active Experiments Tab
            dbc.Tab([
                html.Div([
                    html.H4("Auto-Generated Experiments", className="mt-3"),
                    html.Div(id="auto-experiments-container")
                ])
            ], label="Auto-Generated Experiments", tab_id="experiments-tab"),
            
            # Metrics Improvement Tab
            dbc.Tab([
                html.Div([
                    html.H4("Performance Metrics", className="mt-3"),
                    html.Div(id="performance-metrics-container")
                ])
            ], label="Performance Metrics", tab_id="metrics-tab"),
            
            # Configuration Tab
            dbc.Tab([
                html.Div([
                    html.H4("System Configuration", className="mt-3"),
                    html.Div(id="ci-configuration-container")
                ])
            ], label="Configuration", tab_id="config-tab")
        ], id="ci-tabs", active_tab="history-tab"),
        
        # Stores
        dcc.Store(id="ci-data-store")
    ])


@callback(
    Output("ci-data-store", "data"),
    [Input("refresh-ci-data", "n_clicks"),
     Input("ci-refresh-interval", "n_intervals")]
)
def update_ci_data(n_clicks, n_intervals):
    """Update continuous improvement data store.
    
    Args:
        n_clicks: Button click count
        n_intervals: Interval refresh count
        
    Returns:
        Continuous improvement data
    """
    try:
        # Get system status
        status = continuous_improvement_manager.get_status()
        
        # Get improvement history
        history = continuous_improvement_manager.get_improvement_history()
        
        # Get active experiments
        active_experiments = []
        for exp_id in ab_testing_framework.active_experiment_ids:
            exp = ab_testing_framework.get_experiment(exp_id)
            if exp and exp.metadata.get("auto_generated", False):
                active_experiments.append(exp.to_dict())
        
        # Get implemented improvements
        implemented_improvements = continuous_improvement_manager.results_history
        
        return {
            "status": status,
            "history": history,
            "active_experiments": active_experiments,
            "implemented_improvements": implemented_improvements,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating continuous improvement data: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@callback(
    Output("ci-system-status", "children"),
    [Input("ci-data-store", "data")]
)
def update_system_status(data):
    """Update system status display.
    
    Args:
        data: Continuous improvement data
        
    Returns:
        System status content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    status = data.get("status", {})
    
    # Format dates
    last_check = format_date(status.get("last_check", ""))
    last_gen = format_date(status.get("last_experiment_generation", ""))
    
    status_color = "success" if status.get("enabled", False) else "warning"
    auto_implement_color = "success" if status.get("auto_implement", False) else "warning"
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("System State"),
                html.P([
                    html.Strong("Status: "),
                    html.Span(
                        "Enabled" if status.get("enabled", False) else "Disabled", 
                        className=f"text-{status_color}"
                    )
                ]),
                html.P([
                    html.Strong("Auto-Implementation: "),
                    html.Span(
                        "Enabled" if status.get("auto_implement", False) else "Disabled",
                        className=f"text-{auto_implement_color}"
                    )
                ])
            ])
        ], md=4),
        dbc.Col([
            html.Div([
                html.H5("Activity"),
                html.P([html.Strong("Last Check: "), last_check]),
                html.P([html.Strong("Last Experiment Generation: "), last_gen]),
                html.P([html.Strong("Active Experiments: "), str(status.get("active_experiments", 0))])
            ])
        ], md=4),
        dbc.Col([
            html.Div([
                html.H5("Improvements"),
                html.P([html.Strong("Total Improvements: "), str(status.get("improvements_count", 0))]),
                html.P([
                    html.Strong("System Health: "),
                    html.Span(
                        "Good" if status.get("active_experiments", 0) > 0 else "Needs Attention",
                        className=f"text-{'success' if status.get('active_experiments', 0) > 0 else 'warning'}"
                    )
                ])
            ])
        ], md=4)
    ])


@callback(
    Output("improvement-history-container", "children"),
    [Input("ci-data-store", "data")]
)
def update_improvement_history(data):
    """Update improvement history display.
    
    Args:
        data: Continuous improvement data
        
    Returns:
        Improvement history content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    improvements = data.get("implemented_improvements", [])
    
    if not improvements:
        return html.Div("No improvements have been implemented yet", className="alert alert-info")
    
    # Create a timeline of improvements
    timeline_items = []
    
    for i, improvement in enumerate(reversed(improvements)):  # Show newest first
        timestamp = format_date(improvement.get("timestamp", ""))
        experiment_name = improvement.get("experiment_name", "Unknown experiment")
        experiment_type = improvement.get("experiment_type", "unknown").replace("_", " ").title()
        winning_variant = improvement.get("winning_variant", "Unknown variant")
        
        # Get metrics improvement if available
        metrics_improvement = improvement.get("metrics_improvement", {})
        metrics_text = []
        
        for metric, value in metrics_improvement.items():
            if isinstance(value, float):
                metrics_text.append(f"{metric.replace('_', ' ').title()}: {value:.2f}%")
            else:
                metrics_text.append(f"{metric.replace('_', ' ').title()}: {value}")
        
        # Create timeline item
        item = dbc.Card([
            dbc.CardHeader(
                html.H5(f"Improvement #{len(improvements) - i}: {experiment_name}", className="mb-0")
            ),
            dbc.CardBody([
                html.P([html.Strong("Implemented: "), timestamp]),
                html.P([html.Strong("Type: "), experiment_type]),
                html.P([html.Strong("Winning Variant: "), winning_variant]),
                
                # Show metrics improvement if available
                html.Div([
                    html.Strong("Metrics Improvement:"),
                    html.Ul([html.Li(metric) for metric in metrics_text]) if metrics_text else html.P("No metrics data available")
                ]) if metrics_improvement else None,
                
                # Action button to view details
                dbc.Button(
                    "View Details", 
                    id={"type": "improvement-details-btn", "index": i},
                    color="primary", 
                    size="sm",
                    className="mt-2"
                )
            ])
        ], className="mb-3")
        
        timeline_items.append(item)
    
    # Create a graph showing improvements over time
    if len(improvements) > 1:
        improvement_graph = create_improvement_graph(improvements)
        graph_card = dbc.Card([
            dbc.CardHeader("Improvement Metrics Over Time"),
            dbc.CardBody([
                dcc.Graph(
                    id="improvement-metrics-graph",
                    figure=improvement_graph
                )
            ])
        ], className="mb-4")
    else:
        graph_card = None
    
    return html.Div([
        # Improvement graph if available
        graph_card,
        
        # Timeline items
        html.Div(timeline_items)
    ])


@callback(
    Output("auto-experiments-container", "children"),
    [Input("ci-data-store", "data")]
)
def update_auto_experiments(data):
    """Update auto-generated experiments display.
    
    Args:
        data: Continuous improvement data
        
    Returns:
        Auto-generated experiments content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    active_experiments = data.get("active_experiments", [])
    
    if not active_experiments:
        return html.Div("No auto-generated experiments are currently active", className="alert alert-info")
    
    # Create cards for each active experiment
    experiment_cards = []
    
    for experiment in active_experiments:
        # Extract experiment data
        exp_id = experiment.get("id", "")
        name = experiment.get("name", "Unknown experiment")
        status = experiment.get("status", "unknown")
        exp_type = experiment.get("experiment_type", "unknown").replace("_", " ").title()
        total_traffic = sum(metrics.get("requests", 0) for metrics in experiment.get("variant_metrics", {}).values())
        variants = experiment.get("variants", [])
        
        # Create variant details
        variant_items = []
        for variant in variants:
            variant_id = variant.get("id", "")
            variant_name = variant.get("name", "Unknown variant")
            is_control = variant.get("control", False)
            
            # Get metrics for this variant
            metrics = experiment.get("variant_metrics", {}).get(variant_id, {})
            requests = metrics.get("requests", 0)
            success_rate = metrics.get("success_rate", 0) * 100  # Convert to percentage
            
            variant_items.append(
                dbc.ListGroupItem([
                    html.Strong(f"{variant_name} ({'Control' if is_control else 'Treatment'})"),
                    html.Br(),
                    f"Traffic: {requests} requests ({requests / max(1, total_traffic) * 100:.1f}%)",
                    html.Br(),
                    f"Success Rate: {success_rate:.1f}%"
                ])
            )
        
        # Create card
        card = dbc.Card([
            dbc.CardHeader(html.H5(name, className="mb-0")),
            dbc.CardBody([
                html.P([html.Strong("Status: "), status.title()]),
                html.P([html.Strong("Type: "), exp_type]),
                html.P([html.Strong("Traffic: "), f"{total_traffic} requests"]),
                
                html.H6("Variants:"),
                dbc.ListGroup(variant_items, className="mb-3"),
                
                # Action buttons
                dbc.ButtonGroup([
                    dbc.Button(
                        "View Details", 
                        id={"type": "view-auto-experiment-btn", "index": exp_id},
                        color="primary", 
                        size="sm",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Pause", 
                        id={"type": "pause-auto-experiment-btn", "index": exp_id},
                        color="warning", 
                        size="sm",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Complete", 
                        id={"type": "complete-auto-experiment-btn", "index": exp_id},
                        color="danger", 
                        size="sm"
                    )
                ])
            ])
        ], className="mb-3")
        
        experiment_cards.append(dbc.Col(card, md=6))
    
    # Arrange cards in rows
    rows = []
    for i in range(0, len(experiment_cards), 2):
        rows.append(dbc.Row(experiment_cards[i:i+2], className="mb-4"))
    
    return html.Div([
        html.P("These experiments were automatically generated by the continuous improvement system:", className="mb-3"),
        html.Div(rows)
    ])


@callback(
    Output("performance-metrics-container", "children"),
    [Input("ci-data-store", "data")]
)
def update_performance_metrics(data):
    """Update performance metrics display.
    
    Args:
        data: Continuous improvement data
        
    Returns:
        Performance metrics content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    improvements = data.get("implemented_improvements", [])
    
    if not improvements:
        return html.Div("No improvements have been implemented yet to show performance metrics", className="alert alert-info")
    
    # Create a summary of metrics over time
    metrics_over_time = {}
    dates = []
    
    for improvement in improvements:
        timestamp = improvement.get("timestamp", "")
        if not timestamp:
            continue
            
        # Try to parse the timestamp
        try:
            date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
            dates.append(date)
        except:
            continue
            
        # Extract metrics
        metrics_improvement = improvement.get("metrics_improvement", {})
        for metric, value in metrics_improvement.items():
            if metric not in metrics_over_time:
                metrics_over_time[metric] = []
            
            if isinstance(value, (int, float)):
                metrics_over_time[metric].append(value)
            else:
                # Try to convert to float if possible
                try:
                    metrics_over_time[metric].append(float(value))
                except:
                    metrics_over_time[metric].append(0)
    
    # Create the metrics visualization
    metrics_graphs = []
    
    for metric, values in metrics_over_time.items():
        if len(values) < 2:
            continue
            
        # Create a line graph for this metric
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates[:len(values)],
            y=values,
            mode='lines+markers',
            name=metric.replace("_", " ").title()
        ))
        
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            height=400
        )
        
        metrics_graphs.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(metric.replace("_", " ").title()),
                    dbc.CardBody([
                        dcc.Graph(
                            id=f"metric-graph-{metric}",
                            figure=fig
                        )
                    ])
                ]),
                md=6,
                className="mb-4"
            )
        )
    
    # Create cumulative improvement visualization
    cumulative_improvement = {}
    
    for improvement in improvements:
        metrics_improvement = improvement.get("metrics_improvement", {})
        
        for metric, value in metrics_improvement.items():
            if isinstance(value, (int, float)):
                if metric not in cumulative_improvement:
                    cumulative_improvement[metric] = 0
                
                cumulative_improvement[metric] += value
    
    # Create the cumulative improvement bar chart
    if cumulative_improvement:
        metrics = list(cumulative_improvement.keys())
        values = list(cumulative_improvement.values())
        
        cum_fig = go.Figure(go.Bar(
            x=metrics,
            y=values,
            text=[f"{v:.2f}%" for v in values],
            textposition="auto"
        ))
        
        cum_fig.update_layout(
            title="Cumulative Improvement by Metric",
            xaxis_title="Metric",
            yaxis_title="Cumulative Improvement (%)",
            height=400
        )
        
        cumulative_card = dbc.Card([
            dbc.CardHeader("Cumulative Improvement"),
            dbc.CardBody([
                dcc.Graph(
                    id="cumulative-improvement-graph",
                    figure=cum_fig
                )
            ])
        ], className="mb-4")
    else:
        cumulative_card = None
    
    # Arrange graphs in rows
    metric_rows = []
    for i in range(0, len(metrics_graphs), 2):
        metric_rows.append(dbc.Row(metrics_graphs[i:i+2], className="mb-4"))
    
    return html.Div([
        # Cumulative improvement chart
        cumulative_card,
        
        # Individual metric charts
        html.Div(metric_rows)
    ])


@callback(
    Output("ci-configuration-container", "children"),
    [Input("ci-data-store", "data")]
)
def update_configuration(data):
    """Update configuration display.
    
    Args:
        data: Continuous improvement data
        
    Returns:
        Configuration content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    status = data.get("status", {})
    
    # Create configuration form
    return dbc.Form([
        dbc.FormGroup([
            dbc.Label("System Enabled:"),
            dbc.Select(
                id="system-enabled-config",
                options=[
                    {"label": "Enabled", "value": "enabled"},
                    {"label": "Disabled", "value": "disabled"}
                ],
                value="enabled" if status.get("enabled", False) else "disabled"
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Auto-Implementation:"),
            dbc.Select(
                id="auto-implement-config",
                options=[
                    {"label": "Enabled", "value": "enabled"},
                    {"label": "Disabled", "value": "disabled"}
                ],
                value="enabled" if status.get("auto_implement", False) else "disabled"
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Check Interval (seconds):"),
            dbc.Input(
                id="check-interval-config",
                type="number",
                value=3600,  # 1 hour
                min=300,  # 5 minutes
                max=86400  # 1 day
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Experiment Generation Interval (seconds):"),
            dbc.Input(
                id="generation-interval-config",
                type="number",
                value=86400,  # 1 day
                min=3600,  # 1 hour
                max=604800  # 1 week
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Maximum Concurrent Experiments:"),
            dbc.Input(
                id="max-concurrent-config",
                type="number",
                value=3,
                min=1,
                max=10
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Significance Threshold:"),
            dbc.Input(
                id="significance-threshold-config",
                type="number",
                value=0.95,
                min=0.9,
                max=0.99,
                step=0.01
            )
        ]),
        
        dbc.FormGroup([
            dbc.Label("Improvement Threshold (%):"),
            dbc.Input(
                id="improvement-threshold-config",
                type="number",
                value=5,
                min=1,
                max=20
            )
        ]),
        
        # Submit button
        dbc.Button(
            "Save Configuration", 
            id="save-config-btn",
            color="primary"
        ),
        
        html.Div(id="save-config-result", className="mt-3")
    ])


@callback(
    Output("save-config-result", "children"),
    [Input("save-config-btn", "n_clicks")],
    [State("system-enabled-config", "value"),
     State("auto-implement-config", "value"),
     State("check-interval-config", "value"),
     State("generation-interval-config", "value"),
     State("max-concurrent-config", "value"),
     State("significance-threshold-config", "value"),
     State("improvement-threshold-config", "value")]
)
def save_configuration(
    n_clicks, system_enabled, auto_implement, check_interval,
    generation_interval, max_concurrent, significance_threshold, improvement_threshold
):
    """Save configuration changes.
    
    Args:
        n_clicks: Button click count
        system_enabled: System enabled value
        auto_implement: Auto implement value
        check_interval: Check interval value
        generation_interval: Generation interval value
        max_concurrent: Max concurrent experiments value
        significance_threshold: Significance threshold value
        improvement_threshold: Improvement threshold value
        
    Returns:
        Save result message
    """
    if not n_clicks:
        return ""
    
    try:
        # Convert values to appropriate types
        enabled = system_enabled == "enabled"
        auto_impl = auto_implement == "enabled"
        check_int = int(check_interval)
        gen_int = int(generation_interval)
        max_conc = int(max_concurrent)
        sig_thresh = float(significance_threshold)
        imp_thresh = float(improvement_threshold) / 100.0  # Convert from percentage
        
        # Create a dictionary of updates
        updates = {
            "sentiment_analysis.continuous_improvement.enabled": enabled,
            "sentiment_analysis.continuous_improvement.auto_implement": auto_impl,
            "sentiment_analysis.continuous_improvement.check_interval": check_int,
            "sentiment_analysis.continuous_improvement.experiment_generation_interval": gen_int,
            "sentiment_analysis.continuous_improvement.max_concurrent_experiments": max_conc,
            "sentiment_analysis.continuous_improvement.significance_threshold": sig_thresh,
            "sentiment_analysis.continuous_improvement.improvement_threshold": imp_thresh
        }
        
        # Apply updates to the continuous improvement manager
        continuous_improvement_manager._update_config(updates)
        
        return html.Div("Configuration saved successfully!", className="alert alert-success")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return html.Div(f"Error saving configuration: {str(e)}", className="alert alert-danger")


@callback(
    Output("ci-data-store", "data", allow_duplicate=True),
    [Input("generate-experiments-btn", "n_clicks")],
    prevent_initial_call=True
)
def generate_experiments(n_clicks):
    """Generate new experiments.
    
    Args:
        n_clicks: Button click count
        
    Returns:
        Updated data
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Run the experiment generation
        asyncio.create_task(continuous_improvement_manager.generate_experiments())
        
        # Return updated data (after a slight delay to allow generation to complete)
        time.sleep(1)
        return update_ci_data(None, None)
        
    except Exception as e:
        logger.error(f"Error generating experiments: {str(e)}")
        raise dash.exceptions.PreventUpdate


@callback(
    Output("ci-data-store", "data", allow_duplicate=True),
    [Input("run-maintenance-btn", "n_clicks")],
    prevent_initial_call=True
)
def run_maintenance(n_clicks):
    """Run maintenance task.
    
    Args:
        n_clicks: Button click count
        
    Returns:
        Updated data
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Run the maintenance task
        asyncio.create_task(continuous_improvement_manager.run_maintenance())
        
        # Return updated data (after a slight delay to allow maintenance to complete)
        time.sleep(1)
        return update_ci_data(None, None)
        
    except Exception as e:
        logger.error(f"Error running maintenance: {str(e)}")
        raise dash.exceptions.PreventUpdate


@callback(
    Output("ci-data-store", "data", allow_duplicate=True),
    [Input("auto-implement-toggle", "value")],
    prevent_initial_call=True
)
def toggle_auto_implement(value):
    """Toggle auto-implementation.
    
    Args:
        value: Toggle value
        
    Returns:
        Updated data
    """
    try:
        # Set auto-implement based on value
        enabled = value == "enabled"
        continuous_improvement_manager._update_config({
            "sentiment_analysis.continuous_improvement.auto_implement": enabled
        })
        
        # Return updated data
        return update_ci_data(None, None)
        
    except Exception as e:
        logger.error(f"Error toggling auto-implement: {str(e)}")
        raise dash.exceptions.PreventUpdate


# Event handlers for experiment actions
for action in ["pause", "complete"]:
    @callback(
        Output("ci-data-store", "data", allow_duplicate=True),
        [Input({"type": f"{action}-auto-experiment-btn", "index": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_experiment_action(btn_clicks):
        """Handle experiment actions.
        
        Args:
            btn_clicks: Button click counts
            
        Returns:
            Updated data
        """
        ctx = dash.callback_context
        
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"]
        
        if not trigger_id or "n_clicks" not in trigger_id:
            raise dash.exceptions.PreventUpdate
        
        try:
            # Extract experiment ID from the trigger ID
            trigger_dict = json.loads(trigger_id.split(".")[0])
            experiment_id = trigger_dict["index"]
            action_type = trigger_dict["type"].split("-")[0]
            
            # Perform the requested action
            if action_type == "pause":
                success = ab_testing_framework.pause_experiment(experiment_id)
            elif action_type == "complete":
                success = ab_testing_framework.complete_experiment(experiment_id)
            else:
                success = False
            
            if not success:
                logger.warning(f"Failed to {action_type} experiment {experiment_id}")
            
            # Return updated data
            time.sleep(0.5)  # Small delay to allow action to complete
            return update_ci_data(None, None)
            
        except Exception as e:
            logger.error(f"Error performing experiment action: {str(e)}")
            raise dash.exceptions.PreventUpdate


def format_date(date_str):
    """Format a date string.
    
    Args:
        date_str: ISO date string
        
    Returns:
        Formatted date string
    """
    if not date_str:
        return ""
        
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str


def create_improvement_graph(improvements):
    """Create a graph showing improvements over time.
    
    Args:
        improvements: List of improvement records
        
    Returns:
        Plotly figure
    """
    # Extract dates and metrics
    dates = []
    metrics = {}
    
    for improvement in improvements:
        timestamp = improvement.get("timestamp", "")
        if not timestamp:
            continue
            
        try:
            date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
            dates.append(date)
        except:
            continue
            
        metrics_improvement = improvement.get("metrics_improvement", {})
        for metric, value in metrics_improvement.items():
            if metric not in metrics:
                metrics[metric] = []
            
            if isinstance(value, (int, float)):
                metrics[metric].append(value)
            else:
                try:
                    metrics[metric].append(float(value))
                except:
                    metrics[metric].append(0)
    
    # Create the figure
    fig = go.Figure()
    
    # Add a trace for each metric
    for metric, values in metrics.items():
        if len(values) > 1:
            fig.add_trace(go.Scatter(
                x=dates[:len(values)],
                y=values,
                mode='lines+markers',
                name=metric.replace("_", " ").title()
            ))
    
    # Update layout
    fig.update_layout(
        title="Improvement Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Improvement (%)",
        legend_title="Metrics",
        height=400
    )
    
    return fig