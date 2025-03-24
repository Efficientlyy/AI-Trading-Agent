"""Performance monitoring dashboard for sentiment analysis system.

This module provides a dashboard for visualizing performance metrics
of the sentiment analysis system, including model accuracy, confidence
calibration, and event detection statistics.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.sentiment.performance_tracker import performance_tracker


class PerformanceDashboard:
    """Dashboard for visualizing sentiment system performance metrics."""
    
    def __init__(self):
        """Initialize the performance dashboard."""
        self.logger = get_logger("dashboard", "performance_dashboard")
        
        # Configuration
        self.update_interval = config.get("dashboard.performance_dashboard.update_interval", 30)  # seconds
        self.max_history = config.get("dashboard.performance_dashboard.max_history", 7)  # days
        self.theme = config.get("dashboard.performance_dashboard.theme", "light")
        
        # Initialize data
        self.performance_data = None
        self.historical_data = []
        self.last_update = datetime.now() - timedelta(seconds=self.update_interval * 2)
        
        # Load historical data if available
        self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical performance data."""
        history_file = "data/performance/performance_history.json"
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                # Filter to only keep data from the last max_history days
                cutoff_date = datetime.now() - timedelta(days=self.max_history)
                self.historical_data = [
                    entry for entry in data
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
                ]
                
                self.logger.info(f"Loaded {len(self.historical_data)} historical performance records")
                
            except Exception as e:
                self.logger.error(f"Error loading historical performance data: {str(e)}")
                self.historical_data = []
    
    def _save_historical_data(self) -> None:
        """Save historical performance data."""
        history_file = "data/performance/performance_history.json"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            # Save data
            with open(history_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.historical_data)} historical performance records")
            
        except Exception as e:
            self.logger.error(f"Error saving historical performance data: {str(e)}")
    
    async def update_data(self) -> None:
        """Update performance data."""
        # Check if it's time to update
        now = datetime.now()
        if (now - self.last_update).total_seconds() < self.update_interval:
            return
        
        self.last_update = now
        
        try:
            # Get current performance report
            current_report = performance_tracker.generate_performance_report()
            
            # Store data for dashboard
            self.performance_data = current_report
            
            # Add to historical data with timestamp
            self.historical_data.append(current_report)
            
            # Trim historical data to max_history days
            cutoff_date = now - timedelta(days=self.max_history)
            self.historical_data = [
                entry for entry in self.historical_data
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]
            
            # Save updated historical data
            self._save_historical_data()
            
            self.logger.info("Updated performance dashboard data")
            
        except Exception as e:
            self.logger.error(f"Error updating performance data: {str(e)}")
    
    def create_app(self) -> dash.Dash:
        """Create the Dash application for the performance dashboard.
        
        Returns:
            The Dash application
        """
        # Create Dash app
        app = dash.Dash(
            __name__,
            title="Sentiment Analysis Performance Dashboard",
            update_title=None,
            suppress_callback_exceptions=True,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        # App layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Sentiment Analysis Performance Dashboard", className="header-title"),
                html.P(id="last-update-time", className="header-update-time"),
                html.Button("Refresh Data", id="refresh-button", className="refresh-button"),
            ], className="header"),
            
            # Main content
            html.Div([
                # Summary statistics
                html.Div([
                    html.Div([
                        html.H3("Overall Accuracy"),
                        html.Div(id="overall-accuracy", className="metric-value")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("Calibration Error"),
                        html.Div(id="calibration-error", className="metric-value")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("Total Predictions"),
                        html.Div(id="total-predictions", className="metric-value")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("Active Symbols"),
                        html.Div(id="active-symbols", className="metric-value")
                    ], className="metric-card"),
                ], className="metrics-container"),
                
                # Charts
                html.Div([
                    # Accuracy by source
                    html.Div([
                        html.H3("Accuracy by Source"),
                        dcc.Dropdown(
                            id="symbol-dropdown",
                            placeholder="Select a symbol",
                            className="dropdown"
                        ),
                        dcc.Graph(id="accuracy-chart")
                    ], className="chart-container"),
                    
                    # Accuracy trend
                    html.Div([
                        html.H3("Accuracy Trend"),
                        dcc.Dropdown(
                            id="source-dropdown",
                            placeholder="Select source(s)",
                            multi=True,
                            className="dropdown"
                        ),
                        dcc.Graph(id="accuracy-trend-chart")
                    ], className="chart-container"),
                ], className="charts-row"),
                
                # Additional charts
                html.Div([
                    # Confidence calibration
                    html.Div([
                        html.H3("Confidence Calibration"),
                        dcc.Dropdown(
                            id="calibration-symbol-dropdown",
                            placeholder="Select a symbol",
                            className="dropdown"
                        ),
                        dcc.Graph(id="calibration-chart")
                    ], className="chart-container"),
                    
                    # Source weights
                    html.Div([
                        html.H3("Optimal Source Weights"),
                        dcc.Dropdown(
                            id="weights-symbol-dropdown",
                            placeholder="Select a symbol",
                            className="dropdown"
                        ),
                        dcc.Graph(id="weights-chart")
                    ], className="chart-container"),
                ], className="charts-row"),
                
                # Detailed stats
                html.Div([
                    html.H3("Detailed Statistics by Symbol and Source"),
                    html.Div(id="detailed-stats")
                ], className="detailed-stats-container"),
                
                # Hidden div for storing data
                html.Div(id="performance-data-store", style={"display": "none"}),
                
                # Interval for periodic updates
                dcc.Interval(
                    id="update-interval",
                    interval=self.update_interval * 1000,  # convert to milliseconds
                    n_intervals=0
                ),
            ], className="main-content"),
            
            # Footer
            html.Div([
                html.P("Sentiment Analysis System Performance Dashboard"),
                html.P(id="current-time", className="footer-time")
            ], className="footer"),
        ], className=f"app-container {self.theme}-theme")
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register Dash callbacks for interactivity.
        
        Args:
            app: The Dash application
        """
        # Update data store periodically
        @app.callback(
            Output("performance-data-store", "children"),
            Input("update-interval", "n_intervals"),
            Input("refresh-button", "n_clicks"),
        )
        async def update_data_store(n_intervals, n_clicks):
            # Update data
            self.update_data()
            
            # Return serialized data
            if self.performance_data:
                return json.dumps(self.performance_data)
            return "null"
        
        # Update last update time
        @app.callback(
            Output("last-update-time", "children"),
            Input("performance-data-store", "children"),
        )
        def update_last_update_time(data_json):
            if data_json and data_json != "null":
                data = json.loads(data_json)
                timestamp = datetime.fromisoformat(data["timestamp"])
                return f"Last Updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            return "Last Updated: N/A"
        
        # Update current time
        @app.callback(
            Output("current-time", "children"),
            Input("update-interval", "n_intervals"),
        )
        def update_current_time(n_intervals):
            return f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Update symbol dropdown options
        @app.callback(
            [Output("symbol-dropdown", "options"),
             Output("calibration-symbol-dropdown", "options"),
             Output("weights-symbol-dropdown", "options")],
            Input("performance-data-store", "children"),
        )
        def update_symbol_dropdown(data_json):
            if data_json and data_json != "null":
                data = json.loads(data_json)
                symbols = list(data.get("symbols", {}).keys())
                options = [{"label": symbol, "value": symbol} for symbol in symbols]
                return options, options, options
            return [], [], []
        
        # Update source dropdown options
        @app.callback(
            Output("source-dropdown", "options"),
            Input("performance-data-store", "children"),
        )
        def update_source_dropdown(data_json):
            if data_json and data_json != "null":
                data = json.loads(data_json)
                all_sources = set()
                for symbol_data in data.get("symbols", {}).values():
                    sources = symbol_data.get("direction_accuracy", {}).keys()
                    all_sources.update(sources)
                
                options = [{"label": source, "value": source} for source in sorted(all_sources)]
                return options
            return []
        
        # Update summary metrics
        @app.callback(
            [Output("overall-accuracy", "children"),
             Output("calibration-error", "children"),
             Output("total-predictions", "children"),
             Output("active-symbols", "children")],
            Input("performance-data-store", "children"),
        )
        def update_summary_metrics(data_json):
            if data_json and data_json != "null":
                data = json.loads(data_json)
                summary = data.get("summary", {})
                
                # Format accuracy as percentage
                accuracy = summary.get("avg_direction_accuracy", 0) * 100
                
                # Format calibration error
                calibration = summary.get("avg_calibration_error", 0) * 100
                
                # Count total predictions
                total_predictions = 0
                for symbol_data in data.get("symbols", {}).values():
                    for source_count in symbol_data.get("total_predictions", {}).values():
                        total_predictions += source_count
                
                # Count active symbols
                active_symbols = len(data.get("symbols", {}))
                
                return (
                    f"{accuracy:.1f}%",
                    f"{calibration:.1f}%",
                    f"{total_predictions:,}",
                    f"{active_symbols}"
                )
            
            return "N/A", "N/A", "N/A", "N/A"
        
        # Update accuracy chart
        @app.callback(
            Output("accuracy-chart", "figure"),
            [Input("performance-data-store", "children"),
             Input("symbol-dropdown", "value")],
        )
        def update_accuracy_chart(data_json, symbol):
            if data_json and data_json != "null" and symbol:
                data = json.loads(data_json)
                symbol_data = data.get("symbols", {}).get(symbol, {})
                
                accuracy_data = symbol_data.get("direction_accuracy", {})
                if not accuracy_data:
                    return go.Figure().update_layout(title="No data available for this symbol")
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {"Source": source, "Accuracy": value * 100}
                    for source, value in accuracy_data.items()
                ])
                
                # Create bar chart
                fig = px.bar(
                    df, 
                    x="Source", 
                    y="Accuracy",
                    title=f"Direction Accuracy by Source for {symbol}",
                    labels={"Accuracy": "Accuracy (%)"},
                    color="Source",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Add a horizontal line at 50% (random guessing)
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    y0=50,
                    x1=len(accuracy_data) - 0.5,
                    y1=50,
                    line=dict(
                        color='red',
                        width=2,
                        dash='dash',
                    )
                )
                
                # Add text annotation for the 50% line
                fig.add_annotation(
                    x=len(accuracy_data) - 1,
                    y=50,
                    text="Random",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="red")
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Source",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[0, 100]),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            return go.Figure().update_layout(title="Select a symbol to view accuracy metrics")
        
        # Update accuracy trend chart
        @app.callback(
            Output("accuracy-trend-chart", "figure"),
            [Input("performance-data-store", "children"),
             Input("source-dropdown", "value")],
        )
        def update_accuracy_trend_chart(data_json, sources):
            if data_json and data_json != "null" and sources:
                data = json.loads(data_json)
                
                # Convert historical data to DataFrame
                trend_data = []
                
                for entry in self.historical_data:
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    
                    for symbol, symbol_data in entry.get("symbols", {}).items():
                        accuracy_data = symbol_data.get("direction_accuracy", {})
                        
                        for source, accuracy in accuracy_data.items():
                            if source in sources:
                                trend_data.append({
                                    "Timestamp": timestamp,
                                    "Symbol": symbol,
                                    "Source": source,
                                    "Accuracy": accuracy * 100
                                })
                
                if not trend_data:
                    return go.Figure().update_layout(title="No historical data available for selected sources")
                
                df = pd.DataFrame(trend_data)
                
                # Create line chart
                fig = px.line(
                    df,
                    x="Timestamp",
                    y="Accuracy",
                    color="Source",
                    facet_row="Symbol",
                    labels={"Accuracy": "Accuracy (%)"},
                    title="Accuracy Trend by Source and Symbol",
                    markers=True
                )
                
                # Add a horizontal line at 50% (random guessing)
                for i, symbol in enumerate(df["Symbol"].unique()):
                    fig.add_shape(
                        type='line',
                        x0=df["Timestamp"].min(),
                        y0=50,
                        x1=df["Timestamp"].max(),
                        y1=50,
                        line=dict(
                            color='red',
                            width=2,
                            dash='dash',
                        ),
                        row=i+1,
                        col=1
                    )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Accuracy (%)",
                    margin=dict(l=40, r=40, t=60, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Update y-axis range for all subplots
                for annotation in fig.layout.annotations:
                    annotation.update(x=-0.07)
                
                for i in range(len(df["Symbol"].unique())):
                    fig.update_yaxes(range=[0, 100], row=i+1, col=1)
                
                return fig
            
            return go.Figure().update_layout(title="Select one or more sources to view accuracy trends")
        
        # Update confidence calibration chart
        @app.callback(
            Output("calibration-chart", "figure"),
            [Input("performance-data-store", "children"),
             Input("calibration-symbol-dropdown", "value")],
        )
        def update_calibration_chart(data_json, symbol):
            if data_json and data_json != "null" and symbol:
                data = json.loads(data_json)
                symbol_data = data.get("symbols", {}).get(symbol, {})
                
                accuracy_data = symbol_data.get("direction_accuracy", {})
                if not accuracy_data:
                    return go.Figure().update_layout(title="No data available for this symbol")
                
                # Create a DataFrame for calibration plot
                df = pd.DataFrame([
                    {"Source": source, "Confidence": 0.7, "Accuracy": value * 100}  # Using fixed confidence for illustration
                    for source, value in accuracy_data.items()
                ])
                
                # Create scatter plot
                fig = px.scatter(
                    df,
                    x="Confidence",
                    y="Accuracy",
                    color="Source",
                    text="Source",
                    title=f"Confidence vs. Accuracy for {symbol}",
                    labels={"Confidence": "Average Confidence (%)", "Accuracy": "Actual Accuracy (%)"},
                    range_x=[0, 100],
                    range_y=[0, 100]
                )
                
                # Add diagonal line (perfect calibration)
                fig.add_shape(
                    type='line',
                    x0=0,
                    y0=0,
                    x1=100,
                    y1=100,
                    line=dict(
                        color='black',
                        width=2,
                        dash='dash',
                    )
                )
                
                # Add text annotation for perfect calibration
                fig.add_annotation(
                    x=80,
                    y=85,
                    text="Perfect Calibration",
                    showarrow=False,
                    textangle=-45,
                    font=dict(color="black")
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Average Confidence (%)",
                    yaxis_title="Actual Accuracy (%)",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Configure text display
                fig.update_traces(
                    textposition="top center",
                    marker=dict(size=12)
                )
                
                return fig
            
            return go.Figure().update_layout(title="Select a symbol to view calibration metrics")
        
        # Update weights chart
        @app.callback(
            Output("weights-chart", "figure"),
            [Input("performance-data-store", "children"),
             Input("weights-symbol-dropdown", "value")],
        )
        def update_weights_chart(data_json, symbol):
            if data_json and data_json != "null" and symbol:
                data = json.loads(data_json)
                symbol_data = data.get("symbols", {}).get(symbol, {})
                
                weights_data = symbol_data.get("optimal_weights", {})
                if not weights_data:
                    return go.Figure().update_layout(title="No weights data available for this symbol")
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {"Source": source, "Weight": weight * 100}
                    for source, weight in weights_data.items()
                ])
                
                # Create pie chart
                fig = px.pie(
                    df,
                    values="Weight",
                    names="Source",
                    title=f"Optimal Source Weights for {symbol}",
                    color="Source",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout
                fig.update_layout(
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Update text format
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=12
                )
                
                return fig
            
            return go.Figure().update_layout(title="Select a symbol to view optimal weights")
        
        # Update detailed stats table
        @app.callback(
            Output("detailed-stats", "children"),
            Input("performance-data-store", "children"),
        )
        def update_detailed_stats(data_json):
            if data_json and data_json != "null":
                data = json.loads(data_json)
                symbols_data = data.get("symbols", {})
                
                if not symbols_data:
                    return html.P("No detailed statistics available")
                
                # Create a detailed statistics table
                tables = []
                
                for symbol, symbol_data in symbols_data.items():
                    # Collect data for this symbol
                    table_data = []
                    
                    # Get sources for this symbol
                    sources = set()
                    for metric_type in ["direction_accuracy", "value_accuracy", "calibration_error", "total_predictions"]:
                        sources.update(symbol_data.get(metric_type, {}).keys())
                    
                    # Create header row
                    headers = ["Source", "Direction Accuracy", "Value Accuracy", "Calibration Error", "Total Predictions"]
                    
                    # Create data rows
                    for source in sorted(sources):
                        row = [
                            source,
                            f"{symbol_data.get('direction_accuracy', {}).get(source, 0) * 100:.1f}%",
                            f"{symbol_data.get('value_accuracy', {}).get(source, 0) * 100:.1f}%",
                            f"{symbol_data.get('calibration_error', {}).get(source, 0) * 100:.1f}%",
                            f"{symbol_data.get('total_predictions', {}).get(source, 0):,}"
                        ]
                        table_data.append(row)
                    
                    # Create table
                    table = html.Table(
                        [html.Tr([html.Th(cell) for cell in headers])] +
                        [html.Tr([html.Td(cell) for cell in row]) for row in table_data],
                        className="stats-table"
                    )
                    
                    # Create section for this symbol
                    symbol_section = html.Div([
                        html.H4(f"Statistics for {symbol}"),
                        table
                    ], className="symbol-stats")
                    
                    tables.append(symbol_section)
                
                return html.Div(tables, className="all-stats-tables")
            
            return html.P("No detailed statistics available")
    
    async def run_server(self, port: int = 8051, debug: bool = False) -> None:
        """Run the dashboard server.
        
        Args:
            port: Port to run the server on
            debug: Whether to run in debug mode
        """
        # Create app
        app = self.create_app()
        
        # Run server
        self.logger.info(f"Starting performance dashboard on port {port}")
        app.run_server(debug=debug, port=port)


async def main():
    """Main function to run the dashboard."""
    # Create and initialize performance tracker
    performance_tracker.initialize()
    
    # Create dashboard
    dashboard = PerformanceDashboard()
    
    # Run dashboard server
    await dashboard.run_server(port=8051, debug=True)


if __name__ == "__main__":
    asyncio.run(main())