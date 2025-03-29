"""A/B Testing dashboard component.

This module provides a dashboard for creating, managing, and visualizing
A/B testing experiments for the sentiment analysis system.
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

from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus,
    TargetingCriteria, VariantAssignmentStrategy
)
from src.common.logging import get_logger


# Initialize logger
logger = get_logger("dashboard", "ab_testing")


def create_layout():
    """Create the A/B testing dashboard layout.
    
    Returns:
        Dash layout
    """
    return html.Div([
        html.H2("A/B Testing Dashboard", className="mt-4 mb-4"),
        
        # Controls and refresh
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Refresh Data", id="refresh-ab-test-data", color="primary", className="mr-2"),
                    dcc.Interval(id="ab-test-refresh-interval", interval=30000, n_intervals=0),  # 30 seconds
                ], width=3),
                dbc.Col([
                    dbc.Button("Create New Experiment", id="create-experiment-btn", color="success")
                ], width=3)
            ])
        ], className="mb-4"),
        
        # Tab navigation
        dbc.Tabs([
            # Active Experiments Tab
            dbc.Tab([
                html.Div([
                    html.H4("Active Experiments", className="mt-3"),
                    html.Div(id="active-experiments-container")
                ])
            ], label="Active Experiments", tab_id="active-tab"),
            
            # All Experiments Tab
            dbc.Tab([
                html.Div([
                    html.H4("All Experiments", className="mt-3"),
                    
                    # Filters
                    dbc.Row([
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Status:"),
                                dbc.Select(
                                    id="experiment-status-filter",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Draft", "value": "draft"},
                                        {"label": "Active", "value": "active"},
                                        {"label": "Paused", "value": "paused"},
                                        {"label": "Completed", "value": "completed"},
                                        {"label": "Analyzed", "value": "analyzed"},
                                        {"label": "Implemented", "value": "implemented"},
                                        {"label": "Archived", "value": "archived"}
                                    ],
                                    value="all"
                                )
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label("Type:"),
                                dbc.Select(
                                    id="experiment-type-filter",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Prompt Template", "value": "prompt_template"},
                                        {"label": "Model Selection", "value": "model_selection"},
                                        {"label": "Temperature", "value": "temperature"},
                                        {"label": "Context Strategy", "value": "context_strategy"},
                                        {"label": "Aggregation Weights", "value": "aggregation_weights"},
                                        {"label": "Update Frequency", "value": "update_frequency"},
                                        {"label": "Confidence Threshold", "value": "confidence_threshold"}
                                    ],
                                    value="all"
                                )
                            ])
                        ], width=3)
                    ], className="mb-3"),
                    
                    # Experiments table
                    dash_table.DataTable(
                        id="experiments-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Status", "id": "status"},
                            {"name": "Type", "id": "type"},
                            {"name": "Variants", "id": "variants"},
                            {"name": "Traffic", "id": "total_traffic"},
                            {"name": "Start Date", "id": "start_time"},
                            {"name": "Has Results", "id": "has_results"},
                            {"name": "Has Winner", "id": "has_winner"}
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "minWidth": "100px", 
                            "maxWidth": "300px",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis"
                        },
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold"
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{status} = 'active'"},
                                "backgroundColor": "rgb(220, 240, 220)"
                            },
                            {
                                "if": {"filter_query": "{status} = 'paused'"},
                                "backgroundColor": "rgb(240, 240, 220)"
                            },
                            {
                                "if": {"filter_query": "{has_winner} = 'Yes'"},
                                "color": "green"
                            }
                        ],
                        row_selectable="single",
                        selected_rows=[],
                        page_action="native",
                        page_size=10
                    )
                ])
            ], label="All Experiments", tab_id="all-tab"),
            
            # Experiment Details Tab
            dbc.Tab([
                html.Div(id="experiment-details-container", className="mt-3")
            ], label="Experiment Details", tab_id="details-tab", disabled=True),
            
            # Create Experiment Tab
            dbc.Tab([
                html.Div([
                    html.H4("Create New Experiment", className="mt-3"),
                    
                    # Basic information form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Name:"),
                                    dbc.Input(id="new-experiment-name", type="text", placeholder="Experiment name")
                                ])
                            ], width=6),
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Experiment Type:"),
                                    dbc.Select(
                                        id="new-experiment-type",
                                        options=[
                                            {"label": "Prompt Template", "value": "prompt_template"},
                                            {"label": "Model Selection", "value": "model_selection"},
                                            {"label": "Temperature", "value": "temperature"},
                                            {"label": "Context Strategy", "value": "context_strategy"},
                                            {"label": "Aggregation Weights", "value": "aggregation_weights"},
                                            {"label": "Update Frequency", "value": "update_frequency"},
                                            {"label": "Confidence Threshold", "value": "confidence_threshold"}
                                        ],
                                        value="prompt_template"
                                    )
                                ])
                            ], width=6)
                        ]),
                        
                        dbc.FormGroup([
                            dbc.Label("Description:"),
                            dbc.Textarea(
                                id="new-experiment-description", 
                                placeholder="Experiment description",
                                style={"height": "100px"}
                            )
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Sample Size (optional):"),
                                    dbc.Input(id="new-experiment-sample-size", type="number", placeholder="Sample size")
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Minimum Confidence:"),
                                    dbc.Input(
                                        id="new-experiment-min-confidence", 
                                        type="number", 
                                        value=0.95,
                                        min=0.8,
                                        max=0.99,
                                        step=0.01
                                    )
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.FormGroup([
                                    dbc.Label("Assignment Strategy:"),
                                    dbc.Select(
                                        id="new-experiment-assignment-strategy",
                                        options=[
                                            {"label": "Random", "value": "random"},
                                            {"label": "Round Robin", "value": "round_robin"},
                                            {"label": "Session Sticky", "value": "session_sticky"},
                                            {"label": "Symbol Hash", "value": "symbol_hash"},
                                            {"label": "Time Based", "value": "time_based"},
                                            {"label": "Context Based", "value": "context_based"}
                                        ],
                                        value="random"
                                    )
                                ])
                            ], width=4)
                        ]),
                        
                        dbc.FormGroup([
                            dbc.Label("Targeting Criteria:"),
                            dbc.Checklist(
                                id="new-experiment-targeting",
                                options=[
                                    {"label": "All Traffic", "value": "all_traffic"},
                                    {"label": "Symbol Specific", "value": "symbol_specific"},
                                    {"label": "Source Specific", "value": "source_specific"},
                                    {"label": "Time Specific", "value": "time_specific"},
                                    {"label": "Random Assignment", "value": "random_assignment"},
                                    {"label": "Market Condition", "value": "market_condition"}
                                ],
                                value=["all_traffic"],
                                inline=True
                            )
                        ]),
                        
                        html.Hr(),
                        
                        # Variants
                        html.H5("Variants"),
                        
                        # Control variant
                        dbc.Card([
                            dbc.CardHeader("Control Variant"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.FormGroup([
                                            dbc.Label("Name:"),
                                            dbc.Input(
                                                id="control-variant-name", 
                                                type="text", 
                                                value="Control",
                                                placeholder="Control variant name"
                                            )
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        dbc.FormGroup([
                                            dbc.Label("Weight:"),
                                            dbc.Input(
                                                id="control-variant-weight", 
                                                type="number", 
                                                value=0.5,
                                                min=0.1,
                                                max=0.9,
                                                step=0.05
                                            )
                                        ])
                                    ], width=6)
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Description:"),
                                    dbc.Input(
                                        id="control-variant-description", 
                                        type="text", 
                                        value="Control variant with default settings",
                                        placeholder="Control variant description"
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Configuration (JSON):"),
                                    dbc.Textarea(
                                        id="control-variant-config", 
                                        value="{}",
                                        style={"height": "100px", "fontFamily": "monospace"}
                                    )
                                ])
                            ])
                        ], className="mb-3"),
                        
                        # Treatment variant
                        dbc.Card([
                            dbc.CardHeader("Treatment Variant"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.FormGroup([
                                            dbc.Label("Name:"),
                                            dbc.Input(
                                                id="treatment-variant-name", 
                                                type="text", 
                                                value="Treatment",
                                                placeholder="Treatment variant name"
                                            )
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        dbc.FormGroup([
                                            dbc.Label("Weight:"),
                                            dbc.Input(
                                                id="treatment-variant-weight", 
                                                type="number", 
                                                value=0.5,
                                                min=0.1,
                                                max=0.9,
                                                step=0.05
                                            )
                                        ])
                                    ], width=6)
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Description:"),
                                    dbc.Input(
                                        id="treatment-variant-description", 
                                        type="text", 
                                        value="Treatment variant with experimental settings",
                                        placeholder="Treatment variant description"
                                    )
                                ]),
                                dbc.FormGroup([
                                    dbc.Label("Configuration (JSON):"),
                                    dbc.Textarea(
                                        id="treatment-variant-config", 
                                        value="{}",
                                        style={"height": "100px", "fontFamily": "monospace"}
                                    )
                                ])
                            ])
                        ], className="mb-3"),
                        
                        # Add variant button (disabled for simplicity in this version)
                        dbc.Button(
                            "Add Another Variant", 
                            id="add-variant-btn", 
                            color="secondary",
                            className="mb-3",
                            disabled=True
                        ),
                        
                        # Submit button
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Create Experiment", 
                                    id="submit-experiment-btn",
                                    color="primary",
                                    size="lg",
                                    className="mt-3"
                                ),
                                html.Div(id="create-experiment-result", className="mt-2")
                            ], width={"size": 6, "offset": 3}, className="text-center")
                        ])
                    ])
                ])
            ], label="Create Experiment", tab_id="create-tab")
        ], id="ab-testing-tabs", active_tab="active-tab"),
        
        # Stores
        dcc.Store(id="ab-testing-data-store"),
        dcc.Store(id="selected-experiment-id-store")
    ])


@callback(
    Output("ab-testing-data-store", "data"),
    [Input("refresh-ab-test-data", "n_clicks"),
     Input("ab-test-refresh-interval", "n_intervals")]
)
def update_ab_testing_data(n_clicks, n_intervals):
    """Update A/B testing data store.
    
    Args:
        n_clicks: Button click count
        n_intervals: Interval refresh count
        
    Returns:
        A/B testing data
    """
    try:
        # Get list of experiments
        experiments = ab_testing_framework.list_experiments()
        
        return {
            "experiments": experiments,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating A/B testing data: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@callback(
    Output("active-experiments-container", "children"),
    [Input("ab-testing-data-store", "data")]
)
def update_active_experiments(data):
    """Update active experiments display.
    
    Args:
        data: A/B testing data
        
    Returns:
        Active experiments content
    """
    if not data or "error" in data:
        return html.Div("No data available", className="alert alert-warning")
    
    experiments = data.get("experiments", [])
    
    # Filter active experiments
    active_experiments = ["exp for exp in experiments if exp["status""] = = "active"]
    
    if not active_experiments:
        return html.Div("No active experiments", className="alert alert-info")
    
    # Create cards for each active experiment
    experiment_cards = []
    
    for experiment in active_experiments:
        # Create card
        card = dbc.Card([
            dbc.CardHeader(html.H5(experiment["name"])),
            dbc.CardBody([
                html.P(f"Type: {experiment['type'].replace('_', ' ').title()}"),
                html.P(f"Traffic: {experiment['total_traffic']} requests"),
                html.P(f"Started: {format_date(experiment.get('start_time'))}"),
                
                # Action buttons
                dbc.ButtonGroup([
                    dbc.Button(
                        "View Details", 
                        id={"type": "view-experiment-btn", "index": experiment["id"]},
                        color="primary", 
                        size="sm",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Pause", 
                        id={"type": "pause-experiment-btn", "index": experiment["id"]},
                        color="warning", 
                        size="sm",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Complete", 
                        id={"type": "complete-experiment-btn", "index": experiment["id"]},
                        color="danger", 
                        size="sm"
                    )
                ])
            ])
        ], className="mb-3")
        
        experiment_cards.append(dbc.Col(card, md=6, lg=4))
    
    # Arrange cards in rows
    rows = []
    for i in range(0, len(experiment_cards), 3):
        rows.append(dbc.Row(experiment_cards[i:i+3], className="mb-4"))
    
    return html.Div(rows)


@callback(
    Output("experiments-table", "data"),
    Output("experiments-table", "selected_rows"),
    [Input("ab-testing-data-store", "data"),
     Input("experiment-status-filter", "value"),
     Input("experiment-type-filter", "value")]
)
def update_experiments_table(data, status_filter, type_filter):
    """Update experiments table.
    
    Args:
        data: A/B testing data
        status_filter: Status filter value
        type_filter: Type filter value
        
    Returns:
        Table data and selected rows
    """
    if not data or "error" in data:
        return [], []
    
    experiments = data.get("experiments", [])
    
    # Apply filters
    filtered_experiments = []
    
    for exp in experiments:
        # Apply status filter
        if status_filter != "all" and exp["status"] != status_filter:
            continue
            
        # Apply type filter
        if type_filter != "all" and exp["type"] != type_filter:
            continue
            
        # Add formatted experiment to table
        filtered_experiments.append({
            "id": exp["id"],
            "name": exp["name"],
            "status": exp["status"],
            "type": exp["type"].replace("_", " ").title(),
            "variants": exp["variants"],
            "total_traffic": exp["total_traffic"],
            "start_time": format_date(exp.get("start_time")),
            "has_results": "Yes" if exp["has_results"] else "No",
            "has_winner": "Yes" if exp["has_winner"] else "No"
        })
    
    return filtered_experiments, []


@callback(
    Output("selected-experiment-id-store", "data"),
    [Input("experiments-table", "selected_rows"),
     Input("experiments-table", "data"),
     Input({"type": "view-experiment-btn", "index": ALL}, "n_clicks")]
)
def store_selected_experiment_id(selected_rows, table_data, view_btn_clicks):
    """Store the selected experiment ID.
    
    Args:
        selected_rows: Selected table rows
        table_data: Table data
        view_btn_clicks: View button click counts
        
    Returns:
        Selected experiment ID
    """
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # Handle table selection
    if "experiments-table.selected_rows" in trigger_id:
        if not selected_rows or not table_data:
            return None
        
        selected_index = selected_rows[0]
        if selected_index >= len(table_data):
            return None
        
        return table_data[selected_index]["id"]
    
    # Handle view button clicks
    elif "view-experiment-btn" in trigger_id:
        # Extract experiment ID from the trigger ID
        try:
            trigger_dict = json.loads(trigger_id.split(".")[0])
            return trigger_dict["index"]
        except:
            return None
    
    return None


@callback(
    Output("ab-testing-tabs", "active_tab"),
    [Input("selected-experiment-id-store", "data"),
     Input("create-experiment-btn", "n_clicks")]
)
def switch_to_detail_tab(experiment_id, create_btn_clicks):
    """Switch to the experiment details tab when an experiment is selected.
    
    Args:
        experiment_id: Selected experiment ID
        create_btn_clicks: Create button click count
        
    Returns:
        Active tab ID
    """
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "active-tab"
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    if "selected-experiment-id-store" in trigger_id and experiment_id:
        return "details-tab"
    
    if "create-experiment-btn" in trigger_id and create_btn_clicks:
        return "create-tab"
    
    return "active-tab"


@callback(
    Output("experiment-details-container", "children"),
    [Input("selected-experiment-id-store", "data")]
)
def update_experiment_details(experiment_id):
    """Update experiment details display.
    
    Args:
        experiment_id: Selected experiment ID
        
    Returns:
        Experiment details content
    """
    if not experiment_id:
        return html.Div("Select an experiment to view details", className="alert alert-info")
    
    try:
        # Get experiment report
        experiment = ab_testing_framework.get_experiment(experiment_id)
        if not experiment:
            return html.Div("Experiment not found", className="alert alert-danger")
            
        report = ab_testing_framework.create_experiment_report(experiment_id)
        visualization_data = ab_testing_framework.generate_visualization_data(experiment_id)
        
        # Create experiment details display
        return html.Div([
            # Header
            html.H3(report["name"]),
            html.P(report["description"]),
            
            # Basic info
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Experiment Information"),
                        dbc.CardBody([
                            html.P([html.Strong("Status: "), report["status"].title()]),
                            html.P([html.Strong("Type: "), report["type"].replace("_", " ").title()]),
                            html.P([html.Strong("Created: "), format_date(report["created_at"])]),
                            html.P([html.Strong("Started: "), format_date(report["start_time"]) or "Not started"]),
                            html.P([html.Strong("Ended: "), format_date(report["end_time"]) or "Not ended"]),
                            html.P([html.Strong("Duration: "), 
                                   f"{report['duration_hours']:.1f} hours" if report.get('duration_hours') else "N/A"]),
                            html.P([html.Strong("Total Traffic: "), 
                                   f"{report['total_traffic']} requests"]),
                            html.P([html.Strong("Owner: "), report["owner"]])
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Experiment Status"),
                        dbc.CardBody([
                            html.P([html.Strong("Is Conclusive: "), 
                                   "Yes" if report.get("is_conclusive") else "No"]),
                            html.P([html.Strong("Needs More Data: "), 
                                   "Yes" if report.get("needs_more_data") else "No"]),
                            html.P([html.Strong("Has Winner: "), 
                                   "Yes" if report["results"].get("has_clear_winner") else "No"]),
                            html.P([html.Strong("Winning Variant: "), 
                                   report["results"].get("winning_variant") or "None yet"]),
                            html.Hr(),
                            html.P([html.Strong("Recommendation: ")]),
                            html.P(report["recommendation"])
                        ])
                    ])
                ], md=6)
            ], className="mb-4"),
            
            # Variant comparison chart
            dbc.Card([
                dbc.CardHeader("Variant Performance"),
                dbc.CardBody([
                    dcc.Graph(
                        id={"type": "variant-performance-graph", "index": experiment_id},
                        figure=create_variant_comparison_figure(visualization_data)
                    )
                ])
            ], className="mb-4"),
            
            # Detailed metrics charts
            dbc.Card([
                dbc.CardHeader("Detailed Metrics"),
                dbc.CardBody([
                    dcc.Graph(
                        id={"type": "detailed-metrics-graph", "index": experiment_id},
                        figure=create_detailed_metrics_figure(visualization_data)
                    )
                ])
            ], className="mb-4"),
            
            # Variant details table
            dbc.Card([
                dbc.CardHeader("Variant Details"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id={"type": "variant-details-table", "index": experiment_id},
                        columns=[
                            {"name": "Variant", "id": "name"},
                            {"name": "Type", "id": "type"},
                            {"name": "Traffic", "id": "traffic"},
                            {"name": "Success Rate", "id": "success_rate"},
                            {"name": "Avg Latency (ms)", "id": "avg_latency"},
                            {"name": "Accuracy", "id": "accuracy"},
                            {"name": "Direction Accuracy", "id": "direction_accuracy"},
                            {"name": "Calibration Error", "id": "calibration_error"},
                            {"name": "Confidence Score", "id": "confidence_score"}
                        ],
                        data=[
                            {
                                "name": variant["name"],
                                "type": "Control" if variant["is_control"] else "Treatment",
                                "traffic": variant["metrics"]["requests"],
                                "success_rate": f"{variant['metrics']['success_rate'] * 100:.1f}%",
                                "avg_latency": f"{variant['metrics']['average_latency']:.1f}",
                                "accuracy": f"{variant['metrics']['sentiment_accuracy']:.3f}",
                                "direction_accuracy": f"{variant['metrics']['direction_accuracy']:.3f}",
                                "calibration_error": f"{variant['metrics']['calibration_error']:.3f}",
                                "confidence_score": f"{variant['metrics']['confidence_score']:.3f}"
                            }
                            for variant in report["variants"]
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "minWidth": "100px"
                        },
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold"
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{type} = 'Control'"},
                                "backgroundColor": "rgb(240, 240, 240)"
                            }
                        ]
                    )
                ])
            ], className="mb-4"),
            
            # Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "Start", 
                            id={"type": "start-experiment-btn", "index": experiment_id},
                            color="success", 
                            className="mr-2",
                            disabled=experiment.status != ExperimentStatus.DRAFT and 
                                     experiment.status != ExperimentStatus.PAUSED
                        ),
                        dbc.Button(
                            "Pause", 
                            id={"type": "pause-detail-btn", "index": experiment_id},
                            color="warning", 
                            className="mr-2",
                            disabled=experiment.status != ExperimentStatus.ACTIVE
                        ),
                        dbc.Button(
                            "Complete", 
                            id={"type": "complete-detail-btn", "index": experiment_id},
                            color="danger", 
                            className="mr-2",
                            disabled=experiment.status != ExperimentStatus.ACTIVE and 
                                     experiment.status != ExperimentStatus.PAUSED
                        ),
                        dbc.Button(
                            "Analyze", 
                            id={"type": "analyze-experiment-btn", "index": experiment_id},
                            color="primary", 
                            className="mr-2",
                            disabled=experiment.status != ExperimentStatus.COMPLETED
                        ),
                        dbc.Button(
                            "Implement", 
                            id={"type": "implement-experiment-btn", "index": experiment_id},
                            color="info", 
                            className="mr-2",
                            disabled=experiment.status != ExperimentStatus.ANALYZED or 
                                     not report["results"].get("has_clear_winner")
                        ),
                        dbc.Button(
                            "Archive", 
                            id={"type": "archive-experiment-btn", "index": experiment_id},
                            color="secondary",
                            disabled=experiment.status == ExperimentStatus.DRAFT or 
                                     experiment.status == ExperimentStatus.ACTIVE
                        )
                    ])
                ], className="text-center")
            ])
        ])
    except Exception as e:
        logger.error(f"Error updating experiment details: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")


@callback(
    Output("create-experiment-result", "children"),
    [Input("submit-experiment-btn", "n_clicks")],
    [State("new-experiment-name", "value"),
     State("new-experiment-description", "value"),
     State("new-experiment-type", "value"),
     State("new-experiment-sample-size", "value"),
     State("new-experiment-min-confidence", "value"),
     State("new-experiment-assignment-strategy", "value"),
     State("new-experiment-targeting", "value"),
     State("control-variant-name", "value"),
     State("control-variant-description", "value"),
     State("control-variant-weight", "value"),
     State("control-variant-config", "value"),
     State("treatment-variant-name", "value"),
     State("treatment-variant-description", "value"),
     State("treatment-variant-weight", "value"),
     State("treatment-variant-config", "value")]
)
def create_new_experiment(
    n_clicks, name, description, exp_type, sample_size, min_confidence, 
    assignment_strategy, targeting, control_name, control_desc, control_weight,
    control_config, treatment_name, treatment_desc, treatment_weight, treatment_config
):
    """Create a new experiment.
    
    Args:
        n_clicks: Submit button click count
        name: Experiment name
        description: Experiment description
        exp_type: Experiment type
        sample_size: Target sample size
        min_confidence: Minimum confidence level
        assignment_strategy: Assignment strategy
        targeting: Targeting criteria
        control_name: Control variant name
        control_desc: Control variant description
        control_weight: Control variant weight
        control_config: Control variant configuration
        treatment_name: Treatment variant name
        treatment_desc: Treatment variant description
        treatment_weight: Treatment variant weight
        treatment_config: Treatment variant configuration
        
    Returns:
        Result message
    """
    if not n_clicks or not name:
        return ""
    
    try:
        # Parse variant configurations
        try:
            control_config_dict = json.loads(control_config)
            treatment_config_dict = json.loads(treatment_config)
        except json.JSONDecodeError:
            return html.Div("Invalid JSON in variant configuration", className="alert alert-danger")
        
        # Create variants list
        variants = [
            {
                "name": control_name,
                "description": control_desc,
                "weight": control_weight,
                "config": control_config_dict,
                "control": True
            },
            {
                "name": treatment_name,
                "description": treatment_desc,
                "weight": treatment_weight,
                "config": treatment_config_dict,
                "control": False
            }
        ]
        
        # Create the experiment
        experiment = ab_testing_framework.create_experiment(
            name=name,
            description=description,
            experiment_type=ExperimentType(exp_type),
            variants=variants,
            targeting=[TargetingCriteria(t) for t in targeting],
            assignment_strategy=VariantAssignmentStrategy(assignment_strategy),
            sample_size=int(sample_size) if sample_size else None,
            min_confidence=float(min_confidence),
            owner="dashboard",
            metadata={}
        )
        
        return html.Div([
            html.P(f"Experiment '{name}' created successfully!", className="text-success"),
            dbc.Button(
                "View Experiment", 
                id={"type": "view-new-experiment-btn", "index": experiment.id},
                color="primary"
            )
        ])
        
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")


# Event handlers for various experiment actions
for action in ["start", "pause", "complete", "analyze", "implement", "archive"]:
    @callback(
        Output("ab-testing-data-store", "data", allow_duplicate=True),
        [Input({"type": f"{action}-experiment-btn", "index": ALL}, "n_clicks"),
         Input({"type": f"{action}-detail-btn", "index": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_experiment_action(btn_clicks, detail_btn_clicks):
        """Handle experiment actions.
        
        Args:
            btn_clicks: Button click counts
            detail_btn_clicks: Detail button click counts
            
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
            if action_type == "start":
                success = ab_testing_framework.start_experiment(experiment_id)
            elif action_type == "pause":
                success = ab_testing_framework.pause_experiment(experiment_id)
            elif action_type == "complete":
                success = ab_testing_framework.complete_experiment(experiment_id)
            elif action_type == "analyze":
                # Get the experiment and analyze it
                experiment = ab_testing_framework.get_experiment(experiment_id)
                if experiment:
                    experiment.analyze()
                    success = True
                else:
                    success = False
            elif action_type == "implement":
                # Get the experiment and implement it
                experiment = ab_testing_framework.get_experiment(experiment_id)
                if experiment:
                    experiment.implement()
                    success = True
                else:
                    success = False
            elif action_type == "archive":
                success = ab_testing_framework.archive_experiment(experiment_id)
            else:
                success = False
            
            if not success:
                logger.warning(f"Failed to {action_type} experiment {experiment_id}")
            
            # Return updated data
            return update_ab_testing_data(None, None)
            
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


def create_variant_comparison_figure(data):
    """Create a figure for comparing variants.
    
    Args:
        data: Visualization data
        
    Returns:
        Plotly figure
    """
    # Extract variant data
    variants = data.get("variants", [])
    
    if not variants:
        return go.Figure()
    
    # Create a grouped bar chart
    fig = go.Figure()
    
    # Add bars for each metric for each variant
    for i, variant in enumerate(variants):
        for metric_name in data.get("metrics", []):
            if metric_name in variant.get("metrics", {}):
                # Skip latency for now (different scale)
                if metric_name == "average_latency":
                    continue
                
                # Get value and format label
                value = variant["metrics"][metric_name]
                if isinstance(value, float):
                    if metric_name == "success_rate":
                        value = value * 100  # Convert to percentage
                        label = f"{value:.1f}%"
                    else:
                        label = f"{value:.3f}"
                else:
                    label = str(value)
                
                # Determine if this is significant (if analysis data available)
                is_significant = False
                percent_change = None
                
                if (data.get("analysis") and variant["name"] in data["analysis"] and
                        not variant["is_control"] and metric_name in data["analysis"][variant["name"]]["metric_details"]):
                    metric_details = data["analysis"][variant["name"]]["metric_details"][metric_name]
                    is_significant = metric_details.get("is_significant", False)
                    percent_change = metric_details.get("percent_change")
                
                # Determine color based on significance
                if is_significant:
                    # For calibration error, lower is better
                    if metric_name == "calibration_error":
                        color = "green" if percent_change and percent_change < 0 else "red"
                    else:
                        color = "green" if percent_change and percent_change > 0 else "red"
                    
                    # Add marker
                    marker = {"color": color, "line": {"width": 2, "color": "black"}}
                else:
                    # Use variant status to determine color
                    color = "gray" if variant["is_control"] else "blue"
                    marker = {"color": color}
                
                # Add bar
                fig.add_trace(go.Bar(
                    name=f"{variant['name']} - {metric_name.replace('_', ' ').title()}",
                    x=[metric_name.replace("_", " ").title()],
                    y=[value],
                    text=[label],
                    textposition="auto",
                    marker=marker,
                    showlegend=True,
                    hovertemplate=f"{variant['name']}<br>{metric_name.replace('_', ' ').title()}: {label}"
                ))
    
    # Update layout
    fig.update_layout(
        title="Variant Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        legend_title="Variant - Metric",
        height=500
    )
    
    return fig


def create_detailed_metrics_figure(data):
    """Create a figure with detailed metrics.
    
    Args:
        data: Visualization data
        
    Returns:
        Plotly figure
    """
    # Extract variant data
    variants = data.get("variants", [])
    
    if not variants:
        return go.Figure()
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            "Success Rate", 
            "Sentiment Accuracy",
            "Direction Accuracy",
            "Confidence Score"
        )
    )
    
    # Colors for variants
    colors = ["blue", "green", "red", "orange", "purple"]
    
    # Add metrics for each variant
    for i, variant in enumerate(variants):
        # Get metrics
        success_rate = variant["metrics"].get("success_rate", 0) * 100  # Percentage
        sentiment_accuracy = variant["metrics"].get("sentiment_accuracy", 0)
        direction_accuracy = variant["metrics"].get("direction_accuracy", 0)
        confidence_score = variant["metrics"].get("confidence_score", 0)
        
        # Use variant type to determine color
        color = "gray" if variant["is_control"] else colors[i % len(colors)]
        
        # Add traces
        fig.add_trace(
            go.Bar(
                name=variant["name"],
                x=[variant["name"]],
                y=[success_rate],
                text=[f"{success_rate:.1f}%"],
                textposition="auto",
                marker_color=color,
                showlegend=True,
                hovertemplate=f"{variant['name']}<br>Success Rate: {success_rate:.1f}%"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name=variant["name"],
                x=[variant["name"]],
                y=[sentiment_accuracy],
                text=[f"{sentiment_accuracy:.3f}"],
                textposition="auto",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"{variant['name']}<br>Sentiment Accuracy: {sentiment_accuracy:.3f}"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                name=variant["name"],
                x=[variant["name"]],
                y=[direction_accuracy],
                text=[f"{direction_accuracy:.3f}"],
                textposition="auto",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"{variant['name']}<br>Direction Accuracy: {direction_accuracy:.3f}"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name=variant["name"],
                x=[variant["name"]],
                y=[confidence_score],
                text=[f"{confidence_score:.3f}"],
                textposition="auto",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"{variant['name']}<br>Confidence Score: {confidence_score:.3f}"
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Detailed Metrics Comparison",
        legend_title="Variants"
    )
    
    # Update y-axis ranges
    fig.update_yaxes(range=[0, 100], title_text="Percentage", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text="Score", row=1, col=2)
    fig.update_yaxes(range=[0, 1], title_text="Score", row=2, col=1)
    fig.update_yaxes(range=[0, 1], title_text="Score", row=2, col=2)
    
    return fig