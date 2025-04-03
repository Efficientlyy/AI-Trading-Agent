"""Provider health dashboard component.

This module provides a dashboard component for visualizing LLM provider health status
and failover metrics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.analysis_agents.sentiment.provider_failover import provider_failover_manager
from src.analysis_agents.sentiment.llm_service import LLMService
from src.common.logging import get_logger


# Initialize logger
logger = get_logger("dashboard", "provider_health")


def create_layout():
    """Create the provider health dashboard layout.
    
    Returns:
        Dash layout
    """
    return html.Div([
        html.H2("LLM Provider Health Dashboard", className="mt-4 mb-4"),
        
        # Refresh button and interval
        html.Div([
            dbc.Button("Refresh", id="refresh-provider-button", color="primary", className="mr-2"),
            dcc.Interval(id="provider-refresh-interval", interval=30 * 1000, n_intervals=0),  # 30 seconds
        ], className="mb-4"),
        
        # Provider status cards
        html.Div(id="provider-status-cards", className="row"),
        
        # Provider details
        html.Div([
            html.H4("Provider Details", className="mt-4 mb-3"),
            html.Div(id="provider-details-content"),
        ]),
        
        # Metrics graphs
        html.Div([
            html.H4("Provider Metrics", className="mt-4 mb-3"),
            dcc.Tabs([
                dcc.Tab(label="Success Rate", children=[
                    dcc.Graph(id="success-rate-graph")
                ]),
                dcc.Tab(label="Latency", children=[
                    dcc.Graph(id="latency-graph")
                ]),
                dcc.Tab(label="Usage", children=[
                    dcc.Graph(id="usage-graph")
                ]),
            ]),
        ]),
        
        # Store components for data
        dcc.Store(id="provider-health-data"),
        dcc.Store(id="provider-metrics-history"),
    ])


@callback(
    Output("provider-health-data", "data"),
    [Input("refresh-provider-button", "n_clicks"),
     Input("provider-refresh-interval", "n_intervals")]
)
def update_provider_data(n_clicks, n_intervals):
    """Update provider health data.
    
    Args:
        n_clicks: Button click count
        n_intervals: Interval refresh count
        
    Returns:
        Provider health data
    """
    # Create LLMService instance to access provider status
    llm_service = LLMService()
    
    try:
        # Get provider health status
        provider_status = llm_service.get_provider_health_status()
        
        # Add timestamp
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "providers": provider_status
        }
        
        return data
    except Exception as e:
        logger.error(f"Error updating provider data: {str(e)}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {},
            "error": str(e)
        }


@callback(
    Output("provider-status-cards", "children"),
    [Input("provider-health-data", "data")]
)
def update_provider_cards(data):
    """Update provider status cards.
    
    Args:
        data: Provider health data
        
    Returns:
        Provider status cards
    """
    if not data or "providers" not in data:
        return [html.Div("No provider data available", className="alert alert-warning")]
    
    cards = []
    
    for provider, details in data["providers"].items():
        status = details.get("status", "unknown")
        
        # Determine card color based on status
        color = "success"  # Default to green for healthy
        if status == "degraded":
            color = "warning"
        elif status == "unhealthy":
            color = "danger"
        
        # Create card content
        stats = details.get("stats", {})
        success_rate = stats.get("success_rate", 1.0) * 100
        avg_latency = stats.get("average_latency_ms", 0)
        requests = stats.get("request_count", 0)
        errors = stats.get("error_count", 0)
        
        card = dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.H5(f"{provider.capitalize()}", className="card-title")
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span("Status: ", className="font-weight-bold"),
                        html.Span(
                            status.capitalize(), 
                            className=f"badge badge-{color}"
                        )
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Success Rate: ", className="font-weight-bold"),
                        html.Span(f"{success_rate:.1f}%")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Avg Latency: ", className="font-weight-bold"),
                        html.Span(f"{avg_latency:.0f}ms")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Requests: ", className="font-weight-bold"),
                        html.Span(f"{requests} ({errors} errors)")
                    ], className="mb-2"),
                    dbc.Button(
                        "Details", 
                        id=f"details-btn-{provider}", 
                        color="primary", 
                        size="sm", 
                        className="mt-2"
                    ),
                ])
            ], className="mb-4"),
            width=12, sm=6, md=4, lg=3
        )
        
        cards.append(card)
    
    return cards


@callback(
    Output("provider-details-content", "children"),
    [Input("provider-health-data", "data")],
    [State("provider-details-content", "children")]
)
def update_provider_details(data, current_content):
    """Update provider details section.
    
    Args:
        data: Provider health data
        current_content: Current content
        
    Returns:
        Provider details content
    """
    if not data or "providers" not in data:
        return html.Div("No provider details available", className="alert alert-warning")
    
    details = []
    
    for provider, provider_data in data["providers"].items():
        status = provider_data.get("status", "unknown")
        stats = provider_data.get("stats", {})
        
        # Format last success and error times
        last_success = stats.get("last_success_time")
        if last_success:
            try:
                last_success = datetime.fromisoformat(last_success).strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
                
        last_error = stats.get("last_error_time")
        if last_error:
            try:
                last_error = datetime.fromisoformat(last_error).strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        # Circuit breaker reset time
        circuit_breaker_time = provider_data.get("circuit_breaker_reset_time")
        if circuit_breaker_time:
            try:
                circuit_breaker_time = datetime.fromisoformat(circuit_breaker_time).strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        # Create table
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Metric"), 
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([html.Td("Status"), html.Td(status.capitalize())]),
                html.Tr([html.Td("Success Rate"), html.Td(f"{stats.get('success_rate', 1.0) * 100:.1f}%")]),
                html.Tr([html.Td("Average Latency"), html.Td(f"{stats.get('average_latency_ms', 0):.0f}ms")]),
                html.Tr([html.Td("Total Requests"), html.Td(stats.get("request_count", 0))]),
                html.Tr([html.Td("Successful Requests"), html.Td(stats.get("success_count", 0))]),
                html.Tr([html.Td("Failed Requests"), html.Td(stats.get("error_count", 0))]),
                html.Tr([html.Td("Consecutive Errors"), html.Td(stats.get("consecutive_errors", 0))]),
                html.Tr([html.Td("Tokens Processed"), html.Td(stats.get("tokens_processed", 0))]),
                html.Tr([html.Td("Last Success"), html.Td(last_success or "N/A")]),
                html.Tr([html.Td("Last Error"), html.Td(last_error or "N/A")]),
                html.Tr([html.Td("Last Error Message"), html.Td(stats.get("last_error_message", "N/A"))]),
                html.Tr([html.Td("Circuit Breaker Reset"), html.Td(circuit_breaker_time or "N/A")]),
            ])
        ], bordered=True, striped=True, hover=True, size="sm")
        
        provider_detail = html.Div([
            html.H5(f"{provider.capitalize()} Details", className="mt-3 mb-2"),
            table,
            html.Hr()
        ], className="mb-4")
        
        details.append(provider_detail)
    
    return details


@callback(
    Output("provider-metrics-history", "data"),
    [Input("provider-health-data", "data")],
    [State("provider-metrics-history", "data")]
)
def update_metrics_history(new_data, history_data):
    """Update provider metrics history.
    
    Args:
        new_data: New provider health data
        history_data: Existing metrics history
        
    Returns:
        Updated metrics history
    """
    if not new_data or "providers" not in new_data:
        return history_data or {"providers": {}, "timestamps": []}
    
    # Initialize history if needed
    if not history_data:
        history_data = {
            "providers": {},
            "timestamps": []
        }
    
    # Add timestamp
    current_time = datetime.utcnow().isoformat()
    history_data["timestamps"].append(current_time)
    
    # Limit to last 100 data points
    if len(history_data["timestamps"]) > 100:
        history_data["timestamps"] = history_data["timestamps"][-100:]
    
    # Update metrics for each provider
    for provider, details in new_data["providers"].items():
        if provider not in history_data["providers"]:
            history_data["providers"][provider] = {
                "success_rates": [],
                "latencies": [],
                "request_counts": [],
                "statuses": []
            }
        
        provider_history = history_data["providers"][provider]
        stats = details.get("stats", {})
        
        # Add metrics
        provider_history["success_rates"].append(stats.get("success_rate", 1.0) * 100)
        provider_history["latencies"].append(stats.get("average_latency_ms", 0))
        provider_history["request_counts"].append(stats.get("request_count", 0))
        provider_history["statuses"].append(details.get("status", "unknown"))
        
        # Limit to last 100 data points
        provider_history["success_rates"] = provider_history["success_rates"][-100:]
        provider_history["latencies"] = provider_history["latencies"][-100:]
        provider_history["request_counts"] = provider_history["request_counts"][-100:]
        provider_history["statuses"] = provider_history["statuses"][-100:]
    
    return history_data


@callback(
    Output("success-rate-graph", "figure"),
    [Input("provider-metrics-history", "data")]
)
def update_success_rate_graph(history_data):
    """Update success rate graph.
    
    Args:
        history_data: Provider metrics history
        
    Returns:
        Success rate graph figure
    """
    if not history_data or "providers" not in history_data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Convert timestamps to human-readable format
    timestamps = history_data["timestamps"]
    x_values = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            x_values.append(dt)
        except:
            x_values.append(ts)
    
    # Add trace for each provider
    for provider, metrics in history_data["providers"].items():
        success_rates = metrics.get("success_rates", [])
        if success_rates:
            # Ensure arrays have same length
            y_values = success_rates[-len(x_values):]
            x_subset = x_values[-len(y_values):]
            
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=y_values,
                mode="lines+markers",
                name=provider.capitalize(),
                hovertemplate="Time: %{x}<br>Success Rate: %{y:.1f}%"
            ))
    
    fig.update_layout(
        title="Provider Success Rate Over Time",
        xaxis_title="Time",
        yaxis_title="Success Rate (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )
    
    return fig


@callback(
    Output("latency-graph", "figure"),
    [Input("provider-metrics-history", "data")]
)
def update_latency_graph(history_data):
    """Update latency graph.
    
    Args:
        history_data: Provider metrics history
        
    Returns:
        Latency graph figure
    """
    if not history_data or "providers" not in history_data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Convert timestamps to human-readable format
    timestamps = history_data["timestamps"]
    x_values = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            x_values.append(dt)
        except:
            x_values.append(ts)
    
    # Add trace for each provider
    for provider, metrics in history_data["providers"].items():
        latencies = metrics.get("latencies", [])
        if latencies:
            # Ensure arrays have same length
            y_values = latencies[-len(x_values):]
            x_subset = x_values[-len(y_values):]
            
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=y_values,
                mode="lines+markers",
                name=provider.capitalize(),
                hovertemplate="Time: %{x}<br>Latency: %{y:.0f}ms"
            ))
    
    fig.update_layout(
        title="Provider Latency Over Time",
        xaxis_title="Time",
        yaxis_title="Average Latency (ms)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )
    
    return fig


@callback(
    Output("usage-graph", "figure"),
    [Input("provider-metrics-history", "data")]
)
def update_usage_graph(history_data):
    """Update usage graph.
    
    Args:
        history_data: Provider metrics history
        
    Returns:
        Usage graph figure
    """
    if not history_data or "providers" not in history_data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Convert timestamps to human-readable format
    timestamps = history_data["timestamps"]
    x_values = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            x_values.append(dt)
        except:
            x_values.append(ts)
    
    # Add trace for each provider
    for provider, metrics in history_data["providers"].items():
        request_counts = metrics.get("request_counts", [])
        if request_counts:
            # Ensure arrays have same length
            y_values = request_counts[-len(x_values):]
            x_subset = x_values[-len(y_values):]
            
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=y_values,
                mode="lines+markers",
                name=provider.capitalize(),
                hovertemplate="Time: %{x}<br>Requests: %{y}"
            ))
    
    fig.update_layout(
        title="Provider Request Volume Over Time",
        xaxis_title="Time",
        yaxis_title="Total Requests",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )
    
    return fig