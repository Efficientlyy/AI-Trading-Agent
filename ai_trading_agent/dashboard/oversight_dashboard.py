"""
Dashboard for monitoring LLM oversight decisions and performance.

This module provides dashboard components for visualizing LLM oversight metrics,
decision outcomes, and performance trends.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from ai_trading_agent.oversight.evaluation import OversightEvaluator, MetricType, DecisionOutcome

# Set up logging
logger = logging.getLogger(__name__)


class OversightDashboard:
    """
    Dashboard for monitoring LLM oversight performance.
    
    This class provides visualizations and metrics for monitoring
    the performance of the LLM oversight system.
    """
    
    def __init__(self, evaluator: OversightEvaluator):
        """
        Initialize the dashboard with an evaluator.
        
        Args:
            evaluator: OversightEvaluator instance with decision history
        """
        self.evaluator = evaluator
        
    def create_metrics_cards(self) -> List[dbc.Card]:
        """
        Create metric cards for the dashboard.
        
        Returns:
            List of Dash Bootstrap cards with key metrics
        """
        # Get the latest metrics
        metrics = self.evaluator.calculate_metrics()
        
        # Define the metrics to display
        display_metrics = [
            {
                "title": "Decision Accuracy",
                "value": f"{metrics[MetricType.ACCURACY.value] * 100:.1f}%",
                "icon": "fas fa-bullseye",
                "color": "primary",
                "description": "Percentage of correct oversight decisions"
            },
            {
                "title": "Precision",
                "value": f"{metrics[MetricType.PRECISION.value] * 100:.1f}%",
                "icon": "fas fa-check-circle",
                "color": "success",
                "description": "Correct approvals / all approvals"
            },
            {
                "title": "Recall",
                "value": f"{metrics[MetricType.RECALL.value] * 100:.1f}%",
                "icon": "fas fa-search",
                "color": "info",
                "description": "Percentage of profitable opportunities captured"
            },
            {
                "title": "False Positives",
                "value": f"{metrics[MetricType.FALSE_POSITIVES.value] * 100:.1f}%",
                "icon": "fas fa-exclamation-triangle",
                "color": "warning",
                "description": "Percentage of incorrect approvals"
            },
            {
                "title": "PnL Impact",
                "value": f"${metrics[MetricType.PNL_IMPACT.value]:.2f}",
                "icon": "fas fa-chart-line",
                "color": "success" if metrics[MetricType.PNL_IMPACT.value] > 0 else "danger",
                "description": "Net profit from oversight decisions"
            },
            {
                "title": "Risk Reduction",
                "value": f"{metrics[MetricType.RISK_REDUCTION.value]:.2f}",
                "icon": "fas fa-shield-alt",
                "color": "info",
                "description": "Portfolio risk reduction from oversight"
            }
        ]
        
        # Create cards for each metric
        cards = []
        for metric in display_metrics:
            card = dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{metric['icon']} fa-2x me-2", 
                               style={"color": f"var(--bs-{metric['color']})"})
                    ], className="float-end"),
                    html.H5(metric["title"], className="card-title"),
                    html.H3(metric["value"], className=f"text-{metric['color']}"),
                    html.P(metric["description"], className="card-text text-muted small")
                ]),
                className="mb-4 shadow-sm"
            )
            cards.append(card)
            
        return cards
    
    def create_decision_history_graph(self, days: int = 30) -> dcc.Graph:
        """
        Create a graph of decision history.
        
        Args:
            days: Number of days to include
            
        Returns:
            Dash graph component
        """
        # Get decision history
        df = self.evaluator.get_decision_history_df(days=days)
        
        if df.empty:
            # Return empty placeholder
            fig = go.Figure()
            fig.update_layout(
                title="No Decision History Available",
                xaxis_title="Date",
                yaxis_title="Count",
                height=400
            )
            return dcc.Graph(figure=fig)
        
        # Convert timestamp to date for grouping
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Group by date and oversight action
        action_counts = df.groupby(['date', 'oversight_action']).size().reset_index(name='count')
        
        # Create figure
        fig = px.bar(
            action_counts,
            x='date',
            y='count',
            color='oversight_action',
            barmode='stack',
            color_discrete_map={
                'approve': '#28a745',  # green
                'reject': '#dc3545',   # red
                'modify': '#ffc107'    # yellow
            },
            title="Oversight Decisions Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Decisions",
            legend_title="Oversight Action",
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_outcome_distribution_graph(self) -> dcc.Graph:
        """
        Create a graph showing the distribution of decision outcomes.
        
        Returns:
            Dash graph component
        """
        # Get decision history with outcomes
        df = self.evaluator.get_decision_history_df(include_incomplete=False)
        
        if df.empty:
            # Return empty placeholder
            fig = go.Figure()
            fig.update_layout(
                title="No Outcome Data Available",
                height=400
            )
            return dcc.Graph(figure=fig)
        
        # Create a pivot table of outcomes by action
        pivot = pd.crosstab(df['oversight_action'], df['outcome'])
        
        # Create a stacked bar chart
        fig = go.Figure()
        
        outcomes = pivot.columns.tolist()
        colors = {
            DecisionOutcome.PROFITABLE.value: '#28a745',  # green
            DecisionOutcome.LOSS.value: '#dc3545',        # red
            DecisionOutcome.NEUTRAL.value: '#6c757d',     # gray
            DecisionOutcome.UNKNOWN.value: '#17a2b8'      # blue
        }
        
        for outcome in outcomes:
            fig.add_trace(go.Bar(
                x=pivot.index,
                y=pivot[outcome],
                name=outcome,
                marker_color=colors.get(outcome, '#17a2b8')
            ))
        
        fig.update_layout(
            title="Decision Outcomes by Oversight Action",
            xaxis_title="Oversight Action",
            yaxis_title="Count",
            barmode='stack',
            legend_title="Outcome",
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_metrics_trend_graph(self, days: int = 30) -> dcc.Graph:
        """
        Create a graph showing trends of key metrics.
        
        Args:
            days: Number of days to include
            
        Returns:
            Dash graph component
        """
        # Get metric trends for key metrics
        metrics_to_plot = [
            MetricType.ACCURACY,
            MetricType.PRECISION,
            MetricType.RECALL
        ]
        
        trend_data = {}
        for metric in metrics_to_plot:
            trend_data[metric.value] = self.evaluator.get_metrics_trend(
                metric, days=days, interval_hours=24
            )
        
        # If no data, return empty placeholder
        if not any(trend_data.values()):
            fig = go.Figure()
            fig.update_layout(
                title="No Metric Trend Data Available",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            return dcc.Graph(figure=fig)
        
        # Create dataframe from trend data
        trend_rows = []
        for metric_name, trend_points in trend_data.items():
            for point in trend_points:
                trend_rows.append({
                    'timestamp': pd.to_datetime(point['timestamp']),
                    'metric': metric_name,
                    'value': point['value']
                })
        
        trend_df = pd.DataFrame(trend_rows)
        
        # Create line chart
        fig = px.line(
            trend_df,
            x='timestamp',
            y='value',
            color='metric',
            title="Oversight Metric Trends",
            color_discrete_map={
                'accuracy': '#4e73df',  # blue
                'precision': '#1cc88a',  # green
                'recall': '#f6c23e'      # yellow
            }
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            yaxis=dict(tickformat='.0%'),
            legend_title="Metric",
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_performance_comparison_graph(self) -> dcc.Graph:
        """
        Create a graph comparing performance of different decision types.
        
        Returns:
            Dash graph component
        """
        # Get performance comparison data
        performance = self.evaluator.compare_performance()
        
        # Transform into dataframe for plotting
        performance_df = pd.DataFrame.from_dict(performance, orient='index')
        
        if performance_df.empty or performance_df['count'].sum() == 0:
            # Return empty placeholder
            fig = go.Figure()
            fig.update_layout(
                title="No Performance Comparison Data Available",
                height=400
            )
            return dcc.Graph(figure=fig)
        
        # Create subplot with 2 metrics
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Avg PnL per Decision", "Win Rate"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bars for avg_pnl
        fig.add_trace(
            go.Bar(
                x=performance_df.index,
                y=performance_df['avg_pnl'],
                marker_color='#4e73df',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add bars for win_rate
        fig.add_trace(
            go.Bar(
                x=performance_df.index,
                y=performance_df['win_rate'],
                marker_color='#1cc88a',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Performance Comparison by Decision Type",
            height=400
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Average PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate", tickformat='.0%', row=1, col=2)
        
        return dcc.Graph(figure=fig)
    
    def create_decision_details_table(self, max_rows: int = 10) -> html.Div:
        """
        Create a table with recent decision details.
        
        Args:
            max_rows: Maximum number of rows to display
            
        Returns:
            Dash HTML div containing the table
        """
        # Get recent decisions
        df = self.evaluator.get_decision_history_df(days=7)
        
        if df.empty:
            return html.Div("No recent decisions", className="text-center my-4")
        
        # Sort by timestamp (most recent first) and limit rows
        df = df.sort_values('timestamp', ascending=False).head(max_rows)
        
        # Format the table data
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df['pnl_impact'] = df['pnl_impact'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
        df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
        
        # Create the table
        table = dbc.Table.from_dataframe(
            df[[
                'timestamp', 'symbol', 'action', 
                'oversight_action', 'confidence', 
                'outcome', 'pnl_impact'
            ]],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mb-0"
        )
        
        return html.Div([
            html.H5("Recent Oversight Decisions"),
            table
        ])
    
    def create_dashboard_layout(self) -> html.Div:
        """
        Create the complete dashboard layout.
        
        Returns:
            Dash HTML layout
        """
        # Create metric cards
        metric_cards = self.create_metrics_cards()
        
        # Create metric card row (2 rows with 3 cards each)
        metric_row_1 = dbc.Row([
            dbc.Col(metric_cards[i], md=4) for i in range(3)
        ])
        
        metric_row_2 = dbc.Row([
            dbc.Col(metric_cards[i], md=4) for i in range(3, 6)
        ])
        
        # Create charts
        decision_history = self.create_decision_history_graph()
        metrics_trend = self.create_metrics_trend_graph()
        outcome_distribution = self.create_outcome_distribution_graph()
        performance_comparison = self.create_performance_comparison_graph()
        decision_details = self.create_decision_details_table()
        
        # Build layout
        layout = html.Div([
            html.H1("LLM Oversight Dashboard", className="text-primary mb-4"),
            html.Hr(),
            
            html.H4("Key Performance Metrics", className="mb-3"),
            metric_row_1,
            metric_row_2,
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(decision_history),
                        className="shadow-sm mb-4"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(metrics_trend),
                        className="shadow-sm mb-4"
                    )
                ], md=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(outcome_distribution),
                        className="shadow-sm mb-4"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(performance_comparison),
                        className="shadow-sm mb-4"
                    )
                ], md=6)
            ]),
            
            dbc.Card(
                dbc.CardBody(decision_details),
                className="shadow-sm mb-4"
            )
        ], className="container-fluid py-4")
        
        return layout
    
    def create_dash_app(self, server=None, url_base_pathname="/oversight/"):
        """
        Create a Dash app for the oversight dashboard.
        
        Args:
            server: Optional Flask server to attach the dashboard to
            url_base_pathname: URL base path for the dashboard
            
        Returns:
            Dash app instance
        """
        # Create Dash app
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname=url_base_pathname,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
            ]
        )
        
        # Set layout
        app.layout = self.create_dashboard_layout()
        
        # Add callback for refreshing data
        @app.callback(
            Output("dashboard-container", "children"),
            Input("refresh-interval", "n_intervals")
        )
        def refresh_dashboard(n):
            """Refresh dashboard data."""
            return self.create_dashboard_layout()
        
        return app


def create_standalone_dashboard(evaluator_path: str, host: str = "0.0.0.0", port: int = 8050):
    """
    Create and run a standalone dashboard application.
    
    Args:
        evaluator_path: Path to saved evaluator data
        host: Host to run the server on
        port: Port to run the server on
    """
    # Load evaluator data
    evaluator = OversightEvaluator.load_from_file(evaluator_path)
    
    # Create dashboard
    dashboard = OversightDashboard(evaluator)
    app = dashboard.create_dash_app()
    
    # Run the app
    app.run_server(debug=True, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the LLM Oversight Dashboard")
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to the evaluator data file"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8050,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    create_standalone_dashboard(args.data, args.host, args.port)
