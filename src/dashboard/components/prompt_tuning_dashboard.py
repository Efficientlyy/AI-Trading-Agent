"""Prompt tuning dashboard component.

This module provides a dashboard for visualizing and managing LLM prompt templates
and their performance metrics.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.analysis_agents.sentiment.prompt_tuning import prompt_tuning_system
from src.common.logging import get_logger


# Initialize logger
logger = get_logger("dashboard", "prompt_tuning")


def create_layout():
    """Create the prompt tuning dashboard layout.
    
    Returns:
        Dash layout
    """
    return html.Div([
        html.H2("Prompt Tuning Dashboard", className="mt-4 mb-4"),
        
        # Controls and refresh
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Refresh Data", id="refresh-prompt-data", color="primary", className="mr-2"),
                    dcc.Interval(id="prompt-refresh-interval", interval=30000, n_intervals=0),  # 30 seconds
                ], width=3),
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Prompt Type:"),
                        dbc.Select(
                            id="prompt-type-select",
                            options=[
                                {"label": "Sentiment Analysis", "value": "sentiment_analysis"},
                                {"label": "Event Detection", "value": "event_detection"},
                                {"label": "Impact Assessment", "value": "impact_assessment"}
                            ],
                            value="sentiment_analysis"
                        )
                    ])
                ], width=3),
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Tuning Controls:"),
                        dbc.Checklist(
                            id="tuning-controls",
                            options=[
                                {"label": "Enable Tuning", "value": "enable_tuning"},
                                {"label": "Auto-Optimize", "value": "auto_optimize"},
                                {"label": "Run Experiments", "value": "run_experiments"}
                            ],
                            value=["enable_tuning", "auto_optimize"],
                            inline=True,
                            switch=True
                        )
                    ])
                ], width=6)
            ])
        ], className="mb-4"),
        
        # Summary Cards
        dbc.Row(id="prompt-summary-cards", className="mb-4"),
        
        # Performance graph
        dbc.Card([
            dbc.CardHeader(html.H5("Prompt Performance Metrics")),
            dbc.CardBody([
                dcc.Graph(id="prompt-performance-graph")
            ])
        ], className="mb-4"),
        
        # Prompt Version Table
        dbc.Card([
            dbc.CardHeader(html.H5("Prompt Versions")),
            dbc.CardBody([
                dash_table.DataTable(
                    id="prompt-versions-table",
                    columns=[
                        {"name": "Version", "id": "version"},
                        {"name": "Active", "id": "is_active"},
                        {"name": "Description", "id": "description"},
                        {"name": "Performance", "id": "performance_score"},
                        {"name": "Usage", "id": "usage_count"},
                        {"name": "Success Rate", "id": "success_rate"},
                        {"name": "Accuracy", "id": "accuracy"},
                        {"name": "Created", "id": "created_at"}
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
                            "if": {"filter_query": "{is_active} = 'Yes'"},
                            "backgroundColor": "rgb(220, 240, 220)"
                        }
                    ],
                    row_selectable="single",
                    selected_rows=[],
                    page_action="native",
                    page_size=5
                )
            ])
        ], className="mb-4"),
        
        # Selected Prompt Details
        dbc.Card([
            dbc.CardHeader(html.H5("Prompt Template")),
            dbc.CardBody([
                html.Div(id="selected-prompt-details"),
                dcc.Textarea(
                    id="prompt-template-display",
                    style={"width": "100%", "height": "300px", "fontFamily": "monospace"},
                    readOnly=True
                )
            ])
        ], className="mb-4"),
        
        # Create Variation Form
        dbc.Card([
            dbc.CardHeader(html.H5("Create Prompt Variation")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Variation Type:"),
                            dbc.Select(
                                id="variation-type-select",
                                options=[
                                    {"label": "Instruction Adjustment", "value": "instruction_adjustment"},
                                    {"label": "Format Adjustment", "value": "format_adjustment"},
                                    {"label": "System Role Adjustment", "value": "system_role_adjustment"},
                                    {"label": "Example Addition", "value": "example_addition"},
                                    {"label": "Context Enrichment", "value": "context_enrichment"},
                                    {"label": "Temperature Adjustment", "value": "temperature_adjustment"}
                                ],
                                value="instruction_adjustment"
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Button("Generate Variation", id="generate-variation-btn", color="success")
                    ], width=6, className="d-flex align-items-center")
                ]),
                html.Div(id="variation-result", className="mt-3")
            ])
        ]),
        
        # Store for prompt data
        dcc.Store(id="prompt-data-store"),
        dcc.Store(id="selected-prompt-id-store")
    ])


@callback(
    Output("prompt-data-store", "data"),
    [Input("refresh-prompt-data", "n_clicks"),
     Input("prompt-refresh-interval", "n_intervals")]
)
def update_prompt_data(n_clicks, n_intervals):
    """Update prompt data store.
    
    Args:
        n_clicks: Button click count
        n_intervals: Interval refresh count
        
    Returns:
        Prompt data
    """
    try:
        # Get prompt data from the tuning system
        all_prompts = prompt_tuning_system.get_all_prompts()
        prompt_report = prompt_tuning_system.generate_prompt_report()
        
        return {
            "all_prompts": all_prompts,
            "report": prompt_report,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating prompt data: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@callback(
    Output("prompt-summary-cards", "children"),
    [Input("prompt-data-store", "data"),
     Input("prompt-type-select", "value")]
)
def update_summary_cards(data, prompt_type):
    """Update summary cards.
    
    Args:
        data: Prompt data
        prompt_type: Selected prompt type
        
    Returns:
        Summary cards
    """
    if not data or "error" in data:
        return [html.Div("No prompt data available", className="alert alert-warning")]
    
    report = data.get("report", {})
    type_data = report.get("prompt_types", {}).get(prompt_type, {})
    
    # Create cards
    cards = []
    
    # Total versions card
    version_count = type_data.get("version_count", 0)
    cards.append(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Total Versions"),
                dbc.CardBody([
                    html.H3(version_count),
                    html.P("Prompt templates")
                ])
            ], className="text-center"),
            width=12, sm=6, md=3
        )
    )
    
    # Active version card
    active_version = type_data.get("active_version", {})
    if active_version:
        active_text = f"Version {active_version.get('version', 'N/A')}"
        score = active_version.get("performance_score", 0)
        score_text = f"Score: {score:.2f}" if score else "No data"
    else:
        active_text = "None"
        score_text = "N/A"
    
    cards.append(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Active Version"),
                dbc.CardBody([
                    html.H3(active_text),
                    html.P(score_text)
                ])
            ], className="text-center"),
            width=12, sm=6, md=3
        )
    )
    
    # Best version card
    best_version = type_data.get("best_version", {})
    if best_version:
        best_text = f"Version {best_version.get('version', 'N/A')}"
        score = best_version.get("performance_score", 0)
        score_text = f"Score: {score:.2f}" if score else "No data"
    else:
        best_text = "None"
        score_text = "N/A"
    
    cards.append(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Best Version"),
                dbc.CardBody([
                    html.H3(best_text),
                    html.P(score_text)
                ])
            ], className="text-center"),
            width=12, sm=6, md=3
        )
    )
    
    # System status card
    tuning_enabled = report.get("tuning_enabled", False)
    auto_optimize = report.get("auto_optimize", False)
    
    status_color = "success" if tuning_enabled else "secondary"
    status_text = "Enabled" if tuning_enabled else "Disabled"
    
    auto_status = "Auto-optimization ON" if auto_optimize else "Manual optimization"
    
    cards.append(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("System Status"),
                dbc.CardBody([
                    html.H3(status_text, className=f"text-{status_color}"),
                    html.P(auto_status)
                ])
            ], className="text-center"),
            width=12, sm=6, md=3
        )
    )
    
    return cards


@callback(
    Output("prompt-versions-table", "data"),
    Output("prompt-versions-table", "selected_rows"),
    [Input("prompt-data-store", "data"),
     Input("prompt-type-select", "value")]
)
def update_versions_table(data, prompt_type):
    """Update versions table.
    
    Args:
        data: Prompt data
        prompt_type: Selected prompt type
        
    Returns:
        Table data and selected rows
    """
    if not data or "error" in data:
        return [], []
    
    all_prompts = data.get("all_prompts", {})
    versions = all_prompts.get(prompt_type, [])
    
    # Convert to table data
    table_data = []
    for version in versions:
        created_at = version.get("created_at", "")
        try:
            if created_at:
                created_date = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
            else:
                created_date = "Unknown"
        except:
            created_date = created_at
            
        # Format scores
        performance_score = version.get("performance_score", 0)
        if isinstance(performance_score, (int, float)):
            performance_formatted = f"{performance_score:.3f}"
        else:
            performance_formatted = "N/A"
            
        success_rate = version.get("success_rate", 0)
        if isinstance(success_rate, (int, float)):
            success_formatted = f"{success_rate * 100:.1f}%"
        else:
            success_formatted = "N/A"
            
        accuracy = version.get("accuracy", 0)
        if isinstance(accuracy, (int, float)):
            accuracy_formatted = f"{accuracy:.3f}"
        else:
            accuracy_formatted = "N/A"
        
        table_data.append({
            "version": version.get("version", ""),
            "is_active": "Yes" if version.get("is_active", False) else "No",
            "description": version.get("description", "")[:50] + ("..." if len(version.get("description", "")) > 50 else ""),
            "performance_score": performance_formatted,
            "usage_count": version.get("usage_count", 0),
            "success_rate": success_formatted,
            "accuracy": accuracy_formatted,
            "created_at": created_date,
            "prompt_id": version.get("prompt_id", "")  # Hidden column for reference
        })
    
    # Sort by version number
    table_data.sort(key=lambda x: x["version"])
    
    # Set active version as selected by default
    selected_rows = []
    for i, row in enumerate(table_data):
        if row["is_active"] = = "Yes":
            selected_rows = [i]
            break
    
    return table_data, selected_rows


@callback(
    Output("prompt-performance-graph", "figure"),
    [Input("prompt-data-store", "data"),
     Input("prompt-type-select", "value")]
)
def update_performance_graph(data, prompt_type):
    """Update performance graph.
    
    Args:
        data: Prompt data
        prompt_type: Selected prompt type
        
    Returns:
        Performance graph figure
    """
    if not data or "error" in data:
        return go.Figure()
    
    all_prompts = data.get("all_prompts", {})
    versions = all_prompts.get(prompt_type, [])
    
    if not versions:
        return go.Figure()
    
    # Extract performance metrics
    version_numbers = []
    performance_scores = []
    success_rates = []
    accuracies = []
    usage_counts = []
    is_active = []
    
    for version in versions:
        version_numbers.append(version.get("version", 0))
        performance_scores.append(version.get("performance_score", 0))
        
        # Calculate success rate
        success_count = version.get("success_count", 0)
        usage_count = version.get("usage_count", 0)
        success_rate = success_count / usage_count if usage_count > 0 else 0
        success_rates.append(success_rate)
        
        accuracies.append(version.get("accuracy", 0))
        usage_counts.append(usage_count)
        is_active.append("Active" if version.get("is_active", False) else "Inactive")
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Performance score bars
    fig.add_trace(go.Bar(
        x=version_numbers,
        y=performance_scores,
        name="Performance Score",
        marker_color=["green" if active == "Active" else "lightgrey" for active in is_active],
        hovertemplate="Version %{x}<br>Score: %{y:.3f}<br>Status: %{text}",
        text=is_active
    ))
    
    # Success rate line
    fig.add_trace(go.Scatter(
        x=version_numbers,
        y=success_rates,
        mode="lines+markers",
        name="Success Rate",
        yaxis="y2",
        marker=dict(color="blue"),
        hovertemplate="Version %{x}<br>Success Rate: %{y:.1%}"
    ))
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=version_numbers,
        y=accuracies,
        mode="lines+markers",
        name="Accuracy",
        yaxis="y2",
        marker=dict(color="purple"),
        hovertemplate="Version %{x}<br>Accuracy: %{y:.3f}"
    ))
    
    # Usage bubble size
    normalized_usage = [max(5, min(50, count / 5)) for count in usage_counts]
    fig.add_trace(go.Scatter(
        x=version_numbers,
        y=[0.1] * len(version_numbers),  # Position at bottom
        mode="markers",
        marker=dict(
            size=normalized_usage,
            color="rgba(150, 150, 150, 0.7)"
        ),
        name="Usage",
        hovertemplate="Version %{x}<br>Usage: %{customdata}",
        customdata=usage_counts,
        yaxis="y3"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Performance Metrics for {prompt_type.replace('_', ' ').title()} Prompts",
        xaxis=dict(title="Version"),
        yaxis=dict(title="Performance Score", range=[0, 1]),
        yaxis2=dict(
            title="Rate/Accuracy",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False
        ),
        yaxis3=dict(
            range=[0, 0.2],
            overlaying="y",
            showticklabels=False,
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
        barmode="group",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


@callback(
    Output("selected-prompt-id-store", "data"),
    [Input("prompt-versions-table", "selected_rows"),
     Input("prompt-versions-table", "data")]
)
def store_selected_prompt_id(selected_rows, table_data):
    """Store the selected prompt ID.
    
    Args:
        selected_rows: Selected table rows
        table_data: Table data
        
    Returns:
        Selected prompt ID
    """
    if not selected_rows or not table_data:
        return None
    
    selected_index = selected_rows[0]
    if selected_index >= len(table_data):
        return None
    
    # Get the prompt ID
    return table_data[selected_index]["prompt_id"]


@callback(
    Output("selected-prompt-details", "children"),
    Output("prompt-template-display", "value"),
    [Input("selected-prompt-id-store", "data"),
     Input("prompt-data-store", "data"),
     Input("prompt-type-select", "value")]
)
def update_selected_prompt_details(selected_id, data, prompt_type):
    """Update selected prompt details.
    
    Args:
        selected_id: Selected prompt ID
        data: Prompt data
        prompt_type: Selected prompt type
        
    Returns:
        Prompt details and template
    """
    if not selected_id or not data or "error" in data:
        return "No prompt selected", ""
    
    all_prompts = data.get("all_prompts", {})
    versions = all_prompts.get(prompt_type, [])
    
    # Find the selected prompt
    selected_prompt = None
    for version in versions:
        if version.get("prompt_id") == selected_id:
            selected_prompt = version
            break
    
    if not selected_prompt:
        return "Prompt not found", ""
    
    # Create prompt details display
    details = html.Div([
        html.P([
            html.Strong("Version: "), 
            f"{selected_prompt.get('version', 'Unknown')}"
        ]),
        html.P([
            html.Strong("Status: "), 
            html.Span(
                "Active", 
                className="badge badge-success"
            ) if selected_prompt.get("is_active", False) else html.Span(
                "Inactive", 
                className="badge badge-secondary"
            )
        ]),
        html.P([
            html.Strong("Description: "), 
            selected_prompt.get("description", "No description")
        ]),
        html.P([
            html.Strong("Variation Type: "), 
            selected_prompt.get("variation_type", "None")
        ]),
        html.P([
            html.Strong("Performance: "), 
            f"{selected_prompt.get('performance_score', 0):.3f}"
        ]),
        html.P([
            html.Strong("Created: "), 
            datetime.fromisoformat(selected_prompt.get("created_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M")
        ])
    ])
    
    # Get the template
    template = selected_prompt.get("template", "")
    
    return details, template


@callback(
    Output("variation-result", "children"),
    [Input("generate-variation-btn", "n_clicks")],
    [State("prompt-type-select", "value"),
     State("variation-type-select", "value"),
     State("selected-prompt-id-store", "data")]
)
async def generate_variation(n_clicks, prompt_type, variation_type, selected_id):
    """Generate a prompt variation.
    
    Args:
        n_clicks: Button click count
        prompt_type: Selected prompt type
        variation_type: Selected variation type
        selected_id: Selected prompt ID
        
    Returns:
        Result message
    """
    if not n_clicks or not selected_id:
        return ""
    
    try:
        # Find base prompt
        base_prompt = None
        for versions in prompt_tuning_system.prompt_versions.values():
            for version in versions:
                if version.prompt_id == selected_id:
                    base_prompt = version
                    break
            if base_prompt:
                break
        
        if not base_prompt:
            return html.Div("Selected prompt not found", className="alert alert-danger")
        
        # Convert variation type
        from src.analysis_agents.sentiment.prompt_tuning import PromptVariationType
        variation_enum = PromptVariationType(variation_type)
        
        # Generate variation
        variations = await prompt_tuning_system.generate_prompt_variations(
            prompt_type,
            base_prompt,
            [variation_enum]
        )
        
        if not variations:
            return html.Div(
                "Could not generate variation. Check logs for details.",
                className="alert alert-warning"
            )
        
        # Show success message
        return html.Div([
            html.P(f"Successfully created variation (Version {variations[0].version})!", className="alert alert-success"),
            html.P(f"Description: {variations[0].description}")
        ])
        
    except Exception as e:
        logger.error(f"Error generating variation: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")


@callback(
    Output("tuning-controls", "value"),
    [Input("tuning-controls", "value")],
    [State("tuning-controls", "value")]
)
def update_tuning_controls(new_values, current_values):
    """Update tuning controls in the configuration.
    
    Args:
        new_values: New control values
        current_values: Current control values
        
    Returns:
        Updated control values
    """
    if new_values == current_values:
        return new_values
    
    try:
        # Update config based on changes
        enable_tuning = "enable_tuning" in new_values
        auto_optimize = "auto_optimize" in new_values
        run_experiments = "run_experiments" in new_values
        
        # Save settings to config if needed
        # In a real implementation, this would update the config
        
        return new_values
        
    except Exception as e:
        logger.error(f"Error updating tuning controls: {str(e)}")
        return current_values