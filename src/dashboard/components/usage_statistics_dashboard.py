"""Usage Statistics Dashboard Component.

This module provides visualization and analysis of LLM API usage statistics
including costs, token counts, and optimization recommendations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc

from src.analysis_agents.sentiment.usage_statistics import usage_tracker
from src.analysis_agents.sentiment.provider_failover import provider_failover_manager
from src.common.config import config


# Initialize logging
logger = logging.getLogger(__name__)


def create_usage_statistics_layout():
    """Create the layout for the usage statistics dashboard component.
    
    Returns:
        Dash component with the layout
    """
    return html.Div([
        html.H2("LLM API Usage Statistics", className="mt-4 mb-3"),
        
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="tab-overview", children=[
                html.Div([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Cost Summary (Last 30 Days)"), className="bg-primary text-white"),
                            dbc.CardBody(id="usage-stats-cost-summary"),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                    
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Provider Health Status"), className="bg-info text-white"),
                            dbc.CardBody(id="usage-stats-health-status"),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                ], className="row"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Daily Cost Breakdown"), className="bg-primary text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Range:"),
                                    dcc.Dropdown(
                                        id="usage-stats-time-range",
                                        options=[
                                            {"label": "Last 7 days", "value": 7},
                                            {"label": "Last 30 days", "value": 30},
                                            {"label": "Last 90 days", "value": 90},
                                        ],
                                        value=30,
                                        clearable=False,
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Display:"),
                                    dcc.Dropdown(
                                        id="usage-stats-metric",
                                        options=[
                                            {"label": "Cost ($)", "value": "cost"},
                                            {"label": "Tokens", "value": "tokens"},
                                            {"label": "Requests", "value": "requests"},
                                        ],
                                        value="cost",
                                        clearable=False,
                                    ),
                                ], width=3),
                            ], className="mb-3"),
                            dcc.Graph(id="usage-stats-daily-chart"),
                        ]),
                    ], className="mb-4"),
                ], className="row"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Optimization Recommendations"), className="bg-warning text-white"),
                        dbc.CardBody(id="usage-stats-recommendations"),
                    ], className="mb-4"),
                ], className="row"),
            ]),
            
            dbc.Tab(label="Cost Analysis", tab_id="tab-cost-analysis", children=[
                html.Div([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Provider Cost Breakdown"), className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id="usage-stats-provider-cost-chart"),
                            ]),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                    
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Model Cost Breakdown"), className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id="usage-stats-model-cost-chart"),
                            ]),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                ], className="row"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Operation Type Cost Breakdown"), className="bg-primary text-white"),
                        dbc.CardBody([
                            dcc.Graph(id="usage-stats-operation-cost-chart"),
                        ]),
                    ], className="mb-4"),
                ], className="row"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Cost Projection"), className="bg-info text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Projection Period:"),
                                    dcc.Dropdown(
                                        id="usage-stats-projection-period",
                                        options=[
                                            {"label": "Next 30 days", "value": 30},
                                            {"label": "Next 90 days", "value": 90},
                                            {"label": "Next 180 days", "value": 180},
                                            {"label": "Next 365 days", "value": 365},
                                        ],
                                        value=30,
                                        clearable=False,
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Growth Rate (%):"),
                                    dcc.Input(
                                        id="usage-stats-growth-rate",
                                        type="number",
                                        min=-50,
                                        max=100,
                                        value=0,
                                        className="form-control",
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Base Period:"),
                                    dcc.Dropdown(
                                        id="usage-stats-base-period",
                                        options=[
                                            {"label": "Last 7 days", "value": 7},
                                            {"label": "Last 30 days", "value": 30},
                                            {"label": "Last 90 days", "value": 90},
                                        ],
                                        value=30,
                                        clearable=False,
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Button(
                                        "Generate Projection",
                                        id="usage-stats-generate-projection",
                                        className="btn btn-primary mt-4",
                                    ),
                                ], width=3),
                            ], className="mb-3"),
                            html.Div(id="usage-stats-cost-projection"),
                        ]),
                    ], className="mb-4"),
                ], className="row"),
            ]),
            
            dbc.Tab(label="Usage Patterns", tab_id="tab-usage-patterns", children=[
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Hourly Usage Pattern"), className="bg-primary text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Date:"),
                                    dcc.DatePickerSingle(
                                        id="usage-stats-date-picker",
                                        min_date_allowed=datetime.now().date() - timedelta(days=90),
                                        max_date_allowed=datetime.now().date(),
                                        initial_visible_month=datetime.now().date(),
                                        date=datetime.now().date(),
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Metric:"),
                                    dcc.Dropdown(
                                        id="usage-stats-hourly-metric",
                                        options=[
                                            {"label": "Cost ($)", "value": "cost"},
                                            {"label": "Tokens", "value": "tokens"},
                                            {"label": "Requests", "value": "requests"},
                                        ],
                                        value="requests",
                                        clearable=False,
                                    ),
                                ], width=3),
                            ], className="mb-3"),
                            dcc.Graph(id="usage-stats-hourly-chart"),
                        ]),
                    ], className="mb-4"),
                ], className="row"),
                
                html.Div([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Success Rate Analysis"), className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id="usage-stats-success-rate-chart"),
                            ]),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                    
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Latency Analysis"), className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id="usage-stats-latency-chart"),
                            ]),
                        ], className="mb-4"),
                    ], className="col-md-6"),
                ], className="row"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Token Efficiency Analysis"), className="bg-primary text-white"),
                        dbc.CardBody([
                            dcc.Graph(id="usage-stats-token-efficiency-chart"),
                        ]),
                    ], className="mb-4"),
                ], className="row"),
            ]),
        ], id="usage-stats-tabs", active_tab="tab-overview"),
        
        # Update interval
        dcc.Interval(
            id="usage-stats-update-interval",
            interval=5 * 60 * 1000,  # update every 5 minutes
            n_intervals=0
        ),
    ])


@callback(
    Output("usage-stats-cost-summary", "children"),
    [Input("usage-stats-update-interval", "n_intervals"),
     Input("usage-stats-time-range", "value")]
)
def update_cost_summary(n_intervals, days):
    """Update the cost summary panel.
    
    Args:
        n_intervals: Interval trigger
        days: Number of days to look back
        
    Returns:
        Cost summary HTML components
    """
    days = days or 30
    cost_summary = usage_tracker.get_cost_summary(days)
    
    return html.Div([
        html.Div([
            html.Div([
                html.H2(f"${cost_summary['total_cost']:.2f}"),
                html.P("Total Cost", className="text-muted"),
            ], className="col-md-3 text-center border-end"),
            
            html.Div([
                html.H2(f"${cost_summary['daily_average']:.2f}"),
                html.P("Daily Average", className="text-muted"),
            ], className="col-md-3 text-center border-end"),
            
            html.Div([
                html.H2(f"{cost_summary['total_tokens']:,}"),
                html.P("Total Tokens", className="text-muted"),
            ], className="col-md-3 text-center border-end"),
            
            html.Div([
                html.H2(f"{cost_summary['total_requests']:,}"),
                html.P("Total Requests", className="text-muted"),
            ], className="col-md-3 text-center"),
        ], className="row mb-4"),
        
        html.H5("Provider Breakdown"),
        html.Div([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Provider"),
                        html.Th("Cost"),
                        html.Th("% of Total"),
                        html.Th("Tokens"),
                        html.Th("Requests")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(provider["provider"]),
                        html.Td(f"${provider['cost']:.2f}"),
                        html.Td(f"{provider['percent']:.1f}%"),
                        html.Td(f"{provider['tokens']:,}"),
                        html.Td(f"{provider['requests']:,}")
                    ]) for provider in cost_summary["providers"]
                ])
            ], bordered=True, hover=True, striped=True, size="sm")
        ])
    ])


@callback(
    Output("usage-stats-health-status", "children"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_health_status(n_intervals):
    """Update the provider health status panel.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Health status HTML components
    """
    health_status = provider_failover_manager.get_provider_health_status()
    
    status_badges = {
        "healthy": html.Span("Healthy", className="badge bg-success"),
        "degraded": html.Span("Degraded", className="badge bg-warning"),
        "unhealthy": html.Span("Unhealthy", className="badge bg-danger")
    }
    
    return html.Div([
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Provider"),
                    html.Th("Status"),
                    html.Th("Success Rate"),
                    html.Th("Avg. Latency"),
                    html.Th("Last Error")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(provider),
                    html.Td(status_badges.get(status_info["status"], status_badges["healthy"])),
                    html.Td(f"{status_info['stats']['success_rate'] * 100:.1f}%"),
                    html.Td(f"{status_info['stats']['average_latency_ms']:.0f} ms"),
                    html.Td(status_info['stats']['last_error_message'] or "None", 
                            className="text-truncate", style={"max-width": "200px"})
                ]) for provider, status_info in health_status.items()
            ])
        ], bordered=True, hover=True, striped=True, size="sm")
    ])


@callback(
    Output("usage-stats-daily-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals"),
     Input("usage-stats-time-range", "value"),
     Input("usage-stats-metric", "value")]
)
def update_daily_chart(n_intervals, days, metric):
    """Update the daily usage chart.
    
    Args:
        n_intervals: Interval trigger
        days: Number of days to look back
        metric: Metric to display (cost, tokens, or requests)
        
    Returns:
        Plotly figure
    """
    days = days or 30
    metric = metric or "cost"
    
    daily_usage = usage_tracker.get_daily_usage(days)
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_usage)
    
    # Extract provider data
    providers = set()
    for day in daily_usage:
        providers.update(day["providers"].keys())
    
    # Prepare data for stacked bar chart
    provider_data = {}
    for provider in providers:
        provider_data[provider] = []
        
        for day in daily_usage:
            if provider in day["providers"]:
                provider_stats = day["providers"][provider]
                if metric == "cost":
                    value = provider_stats.get("costs", {}).get("total", 0)
                elif metric == "tokens":
                    value = provider_stats.get("tokens", {}).get("total", 0)
                else:  # requests
                    value = provider_stats.get("requests", {}).get("total", 0)
            else:
                value = 0
                
            provider_data[provider].append(value)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    for provider in providers:
        fig.add_trace(go.Bar(
            name=provider,
            x=df["date"],
            y=provider_data[provider],
            hovertemplate=(
                "%{x}<br>" +
                f"{provider}: %{{y:,.2f}}" + (" tokens" if metric == "tokens" else (" requests" if metric == "requests" else " USD")) +
                "<extra></extra>"
            )
        ))
    
    title_metric = "Cost ($)" if metric == "cost" else ("Token Count" if metric == "tokens" else "Request Count")
    
    fig.update_layout(
        title=f"Daily {title_metric} by Provider",
        barmode="stack",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=title_metric,
            tickformat=",.2f" if metric == "cost" else ",d",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


@callback(
    Output("usage-stats-recommendations", "children"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_recommendations(n_intervals):
    """Update the optimization recommendations panel.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Recommendations HTML components
    """
    recommendations = usage_tracker.get_usage_optimization_recommendations()
    
    if not recommendations:
        return html.P("No optimization recommendations at this time.")
    
    # Map priority to badge class
    priority_badges = {
        "high": "bg-danger",
        "medium": "bg-warning",
        "low": "bg-info"
    }
    
    return html.Div([
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Recommendation"),
                    html.Th("Target"),
                    html.Th("Potential Savings"),
                    html.Th("Priority")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(rec["message"]),
                    html.Td(rec.get("model") or rec.get("operation") or "System"),
                    html.Td(rec["potential_savings"]),
                    html.Td(html.Span(
                        rec["priority"].capitalize(),
                        className=f"badge {priority_badges.get(rec['priority'], 'bg-primary')}"
                    ))
                ]) for rec in recommendations
            ])
        ], bordered=True, hover=True, striped=True, size="sm")
    ])


@callback(
    Output("usage-stats-provider-cost-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals"),
     Input("usage-stats-time-range", "value")]
)
def update_provider_cost_chart(n_intervals, days):
    """Update the provider cost breakdown chart.
    
    Args:
        n_intervals: Interval trigger
        days: Number of days to look back
        
    Returns:
        Plotly figure
    """
    days = days or 30
    
    # Get cost summary
    cost_summary = usage_tracker.get_cost_summary(days)
    providers = cost_summary["providers"]
    
    # Sort by cost
    providers.sort(key=lambda x: x["cost"], reverse=True)
    
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(go.Pie(
        labels=[p["provider"] for p in providers],
        values=[p["cost"] for p in providers],
        hoverinfo="label+percent+value",
        hovertemplate="%{label}<br>%{value:.2f} USD<br>%{percent}<extra></extra>",
        textinfo="label+percent",
    ))
    
    fig.update_layout(
        title=f"Provider Cost Distribution (Last {days} Days)",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


@callback(
    Output("usage-stats-model-cost-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_model_cost_chart(n_intervals):
    """Update the model cost breakdown chart.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Plotly figure
    """
    model_stats = usage_tracker.get_model_usage()
    
    # Convert to list for sorting
    models = []
    for model, stats in model_stats.items():
        cost = stats.get("costs", {}).get("total", 0)
        tokens = stats.get("tokens", {}).get("total", 0)
        requests = stats.get("requests", {}).get("total", 0)
        
        if cost > 0:
            models.append({
                "model": model,
                "cost": cost,
                "tokens": tokens,
                "requests": requests,
            })
    
    # Sort by cost
    models.sort(key=lambda x: x["cost"], reverse=True)
    
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(go.Pie(
        labels=[m["model"] for m in models],
        values=[m["cost"] for m in models],
        hoverinfo="label+percent+value",
        hovertemplate="%{label}<br>%{value:.2f} USD<br>%{percent}<extra></extra>",
        textinfo="label+percent",
    ))
    
    fig.update_layout(
        title="Model Cost Distribution",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


@callback(
    Output("usage-stats-operation-cost-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_operation_cost_chart(n_intervals):
    """Update the operation cost breakdown chart.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Plotly figure
    """
    operation_stats = usage_tracker.get_operation_usage()
    
    # Convert to list for sorting
    operations = []
    for operation, stats in operation_stats.items():
        cost = stats.get("costs", {}).get("total", 0)
        tokens = stats.get("tokens", {}).get("total", 0)
        requests = stats.get("requests", {}).get("total", 0)
        
        if cost > 0:
            operations.append({
                "operation": operation,
                "cost": cost,
                "tokens": tokens,
                "requests": requests,
            })
    
    # Sort by cost
    operations.sort(key=lambda x: x["cost"], reverse=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Cost Distribution by Operation", "Average Cost per Request")
    )
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=[o["operation"] for o in operations],
            values=[o["cost"] for o in operations],
            hoverinfo="label+percent+value",
            hovertemplate="%{label}<br>%{value:.2f} USD<br>%{percent}<extra></extra>",
            textinfo="label+percent",
        ),
        row=1, col=1
    )
    
    # Calculate cost per request for bar chart
    cost_per_request = []
    for op in operations:
        if op["requests"] > 0:
            cost_per_request.append({
                "operation": op["operation"],
                "cost_per_request": op["cost"] / op["requests"]
            })
    
    # Sort by cost per request
    cost_per_request.sort(key=lambda x: x["cost_per_request"], reverse=True)
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=[o["operation"] for o in cost_per_request],
            y=[o["cost_per_request"] for o in cost_per_request],
            hovertemplate="%{x}<br>%{y:.4f} USD per request<extra></extra>",
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Operation Type Cost Analysis",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    fig.update_yaxes(title_text="USD per Request", row=1, col=2)
    
    return fig


@callback(
    Output("usage-stats-cost-projection", "children"),
    [Input("usage-stats-generate-projection", "n_clicks")],
    [State("usage-stats-projection-period", "value"),
     State("usage-stats-growth-rate", "value"),
     State("usage-stats-base-period", "value")]
)
def update_cost_projection(n_clicks, projection_days, growth_rate, base_days):
    """Update the cost projection.
    
    Args:
        n_clicks: Button click trigger
        projection_days: Number of days to project
        growth_rate: Growth rate percentage
        base_days: Number of days to use as base
        
    Returns:
        Cost projection HTML components
    """
    if n_clicks is None:
        return html.P("Click 'Generate Projection' to calculate cost projections.")
    
    projection_days = projection_days or 30
    growth_rate = growth_rate or 0
    base_days = base_days or 30
    
    # Get cost summary for base period
    cost_summary = usage_tracker.get_cost_summary(base_days)
    
    # Calculate daily average
    daily_average = cost_summary["daily_average"]
    
    # Calculate projections
    monthly_growth_rate = growth_rate / 100
    
    # Project for each month
    months = projection_days // 30
    if months < 1:
        months = 1
    
    monthly_costs = []
    cumulative_cost = 0
    current_daily_rate = daily_average
    
    for month in range(months):
        days_in_month = min(30, projection_days - month * 30)
        if days_in_month <= 0:
            break
            
        monthly_cost = current_daily_rate * days_in_month
        cumulative_cost += monthly_cost
        monthly_costs.append({
            "month": month + 1,
            "days": days_in_month,
            "monthly_cost": monthly_cost,
            "cumulative_cost": cumulative_cost,
            "daily_rate": current_daily_rate
        })
        
        # Apply growth rate for next month
        current_daily_rate *= (1 + monthly_growth_rate)
    
    # Create projections chart
    projection_df = pd.DataFrame(monthly_costs)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=("Monthly Cost Projection", "Cumulative Cost Projection")
    )
    
    # Add monthly bar chart
    fig.add_trace(
        go.Bar(
            x=projection_df["month"].apply(lambda x: f"Month {x}"),
            y=projection_df["monthly_cost"],
            hovertemplate="Month %{x}<br>Cost: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )
    
    # Add cumulative line chart
    fig.add_trace(
        go.Scatter(
            x=projection_df["month"].apply(lambda x: f"Month {x}"),
            y=projection_df["cumulative_cost"],
            mode="lines+markers",
            hovertemplate="Month %{x}<br>Cumulative Cost: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"{projection_days}-Day Cost Projection (Growth Rate: {growth_rate}%)",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    fig.update_yaxes(title_text="Monthly Cost (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Cost (USD)", row=1, col=2)
    
    # Summary table
    summary_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Metric"),
                html.Th("Value"),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td("Base Daily Average"),
                html.Td(f"${daily_average:.2f}"),
            ]),
            html.Tr([
                html.Td("Monthly Growth Rate"),
                html.Td(f"{growth_rate}%"),
            ]),
            html.Tr([
                html.Td("Projection Period"),
                html.Td(f"{projection_days} days"),
            ]),
            html.Tr([
                html.Td("Total Projected Cost"),
                html.Td(f"${cumulative_cost:.2f}"),
            ]),
            html.Tr([
                html.Td("Final Daily Rate"),
                html.Td(f"${current_daily_rate:.2f}"),
            ]),
        ])
    ], bordered=True, hover=True, striped=True, size="sm")
    
    return html.Div([
        dcc.Graph(figure=fig, className="mb-3"),
        summary_table
    ])


@callback(
    Output("usage-stats-hourly-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals"),
     Input("usage-stats-date-picker", "date"),
     Input("usage-stats-hourly-metric", "value")]
)
def update_hourly_chart(n_intervals, date, metric):
    """Update the hourly usage chart.
    
    Args:
        n_intervals: Interval trigger
        date: Selected date
        metric: Metric to display (cost, tokens, or requests)
        
    Returns:
        Plotly figure
    """
    date = date or datetime.now().date().isoformat()
    metric = metric or "requests"
    
    hourly_usage = usage_tracker.get_hourly_usage(date)
    
    # Convert to DataFrame
    df = pd.DataFrame(hourly_usage)
    
    # Extract provider data
    providers = set()
    for hour_data in hourly_usage:
        providers.update(hour_data["providers"].keys())
    
    # Prepare data for stacked bar chart
    provider_data = {}
    for provider in providers:
        provider_data[provider] = []
        
        for hour_data in hourly_usage:
            if provider in hour_data["providers"]:
                provider_stats = hour_data["providers"][provider]
                if metric == "cost":
                    value = provider_stats.get("costs", {}).get("total", 0)
                elif metric == "tokens":
                    value = provider_stats.get("tokens", {}).get("total", 0)
                else:  # requests
                    value = provider_stats.get("requests", {}).get("total", 0)
            else:
                value = 0
                
            provider_data[provider].append(value)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    for provider in providers:
        fig.add_trace(go.Bar(
            name=provider,
            x=df["hour"],
            y=provider_data[provider],
            hovertemplate=(
                "Hour %{x}:00<br>" +
                f"{provider}: %{{y:,.2f}}" + (" tokens" if metric == "tokens" else (" requests" if metric == "requests" else " USD")) +
                "<extra></extra>"
            )
        ))
    
    title_metric = "Cost ($)" if metric == "cost" else ("Token Count" if metric == "tokens" else "Request Count")
    
    fig.update_layout(
        title=f"Hourly {title_metric} by Provider for {date}",
        barmode="stack",
        xaxis=dict(
            title="Hour of Day",
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)]
        ),
        yaxis=dict(
            title=title_metric,
            tickformat=",.2f" if metric == "cost" else ",d",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


@callback(
    Output("usage-stats-success-rate-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_success_rate_chart(n_intervals):
    """Update the success rate chart.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Plotly figure
    """
    provider_stats = usage_tracker.get_provider_usage()
    
    # Calculate success rates
    providers = []
    for provider, stats in provider_stats.items():
        total = stats.get("requests", {}).get("total", 0)
        success = stats.get("requests", {}).get("success", 0)
        
        if total > 0:
            success_rate = success / total * 100
            requests = total
            providers.append({
                "provider": provider,
                "success_rate": success_rate,
                "requests": requests
            })
    
    # Sort by number of requests
    providers.sort(key=lambda x: x["requests"], reverse=True)
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=[p["provider"] for p in providers],
        y=[p["success_rate"] for p in providers],
        text=[f"{p['success_rate']:.1f}%" for p in providers],
        textposition="auto",
        hovertemplate="%{x}<br>Success Rate: %{y:.1f}%<br>Requests: %{customdata:,d}<extra></extra>",
        customdata=[[p["requests"]] for p in providers],
        marker_color=["#28a745" if p["success_rate"] >= 95 else 
                      "#ffc107" if p["success_rate"] >= 90 else 
                      "#dc3545" for p in providers]
    ))
    
    fig.update_layout(
        title="API Success Rate by Provider",
        xaxis=dict(title="Provider"),
        yaxis=dict(
            title="Success Rate (%)",
            range=[0, 100],
            ticksuffix="%"
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


@callback(
    Output("usage-stats-latency-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_latency_chart(n_intervals):
    """Update the latency chart.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Plotly figure
    """
    provider_stats = usage_tracker.get_provider_usage()
    model_stats = usage_tracker.get_model_usage()
    
    # Calculate average latency by provider
    providers = []
    for provider, stats in provider_stats.items():
        latency = stats.get("latency", {}).get("average_ms", 0)
        count = stats.get("latency", {}).get("count", 0)
        
        if count > 0:
            providers.append({
                "provider": provider,
                "latency": latency,
                "requests": count
            })
    
    # Calculate average latency by model
    models = []
    for model, stats in model_stats.items():
        latency = stats.get("latency", {}).get("average_ms", 0)
        count = stats.get("latency", {}).get("count", 0)
        
        if count > 10:  # Only include models with sufficient data
            models.append({
                "model": model,
                "latency": latency,
                "requests": count
            })
    
    # Sort by latency
    providers.sort(key=lambda x: x["latency"])
    models.sort(key=lambda x: x["latency"])
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("Average Latency by Provider", "Average Latency by Model")
    )
    
    # Add provider latency chart
    fig.add_trace(
        go.Bar(
            x=[p["provider"] for p in providers],
            y=[p["latency"] for p in providers],
            text=[f"{p['latency']:.0f} ms" for p in providers],
            textposition="auto",
            hovertemplate="%{x}<br>Latency: %{y:.0f} ms<br>Requests: %{customdata:,d}<extra></extra>",
            customdata=[[p["requests"]] for p in providers],
        ),
        row=1, col=1
    )
    
    # Add model latency chart
    fig.add_trace(
        go.Bar(
            x=[m["model"] for m in models],
            y=[m["latency"] for m in models],
            text=[f"{m['latency']:.0f} ms" for m in models],
            textposition="auto",
            hovertemplate="%{x}<br>Latency: %{y:.0f} ms<br>Requests: %{customdata:,d}<extra></extra>",
            customdata=[[m["requests"]] for m in models],
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="API Latency Analysis",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
    
    return fig


@callback(
    Output("usage-stats-token-efficiency-chart", "figure"),
    [Input("usage-stats-update-interval", "n_intervals")]
)
def update_token_efficiency_chart(n_intervals):
    """Update the token efficiency chart.
    
    Args:
        n_intervals: Interval trigger
        
    Returns:
        Plotly figure
    """
    operation_stats = usage_tracker.get_operation_usage()
    
    # Calculate tokens per request
    operations = []
    for operation, stats in operation_stats.items():
        tokens = stats.get("tokens", {}).get("total", 0)
        requests = stats.get("requests", {}).get("total", 0)
        
        if requests > 10:  # Only include operations with sufficient data
            tokens_per_request = tokens / requests
            operations.append({
                "operation": operation,
                "tokens_per_request": tokens_per_request,
                "requests": requests,
                "tokens": tokens
            })
    
    # Sort by tokens per request
    operations.sort(key=lambda x: x["tokens_per_request"], reverse=True)
    
    fig = go.Figure()
    
    # Add scatter plot with bubble size based on request count
    fig.add_trace(go.Scatter(
        x=[o["operation"] for o in operations],
        y=[o["tokens_per_request"] for o in operations],
        mode="markers",
        marker=dict(
            size=[min(max(10, o["requests"] / 10), 50) for o in operations],
            sizemode="area",
            sizeref=2. * max([o["requests"] for o in operations]) / (50.**2),
            sizemin=4
        ),
        text=[f"{o['operation']}<br>{o['tokens_per_request']:.0f} tokens per request<br>{o['requests']} requests" for o in operations],
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title="Token Efficiency by Operation Type",
        xaxis=dict(title="Operation Type"),
        yaxis=dict(title="Tokens per Request"),
        margin=dict(l=40, r=40, t=60, b=100),
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig