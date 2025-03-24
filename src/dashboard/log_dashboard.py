"""Log analytics dashboard using Dash.

This provides a web-based UI for visualizing and analyzing log data.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import json
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Check if Dash is available
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Import the actual log query and replay modules
from src.common.config import config
from src.common.logging import DEFAULT_LOG_FILE, LOG_DIR, get_logger
from src.common.log_query import LogQuery
from src.common.log_replay import LogReplay
from src.common.health_monitoring import HealthMonitor
from src.analytics.anomaly_detection import AnomalyDetector
from src.dashboard.visualizations import (
    create_time_heatmap, 
    create_log_patterns_chart,
    create_correlation_heatmap,
    create_log_volume_comparison,
    create_error_distribution_chart
)

# Initialize logger
logger = get_logger("dashboard")

# Dashboard configuration
REFRESH_INTERVAL = config.get("system.logging.dashboard.refresh_interval", 10)  # seconds
MAX_LOGS_TO_ANALYZE = config.get("system.logging.dashboard.max_logs", 10000)
PORT = config.get("system.logging.dashboard.port", 8050)


class LogAnalyzer:
    """Analyze log data and generate statistics."""
    
    def __init__(self, log_dir=None):
        """Initialize log analyzer with log directory."""
        self.log_dir = log_dir or LOG_DIR
        self.log_query = LogQuery()
        self.anomaly_detector = AnomalyDetector(sensitivity=0.95)
        self.anomaly_detector_trained = False
        
    def get_logs(self, query=None, limit=1000):
        """
        Get logs from files, optionally filtered by query.
        
        Args:
            query: Optional query string to filter logs
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        try:
            if query:
                self.log_query.compile(query)
                return self.log_query.search_directory(
                    directory=str(self.log_dir),
                    limit=limit
                )
            else:
                return self.log_query.search_directory(
                    directory=str(self.log_dir),
                    limit=limit
                )
        except Exception as e:
            logger.error("Error getting logs", error=str(e))
            return []
    
    def detect_anomalies(self, limit=1000):
        """
        Detect anomalies in the log data.
        
        Args:
            limit: Maximum number of logs to analyze
            
        Returns:
            List of anomalies detected
        """
        try:
            # Get logs for training if not trained
            if not self.anomaly_detector_trained:
                training_logs = self.get_logs(limit=5000)
                if training_logs:
                    self.anomaly_detector.train(training_logs)
                    self.anomaly_detector_trained = True
                    logger.info("Anomaly detector trained on {} logs".format(len(training_logs)))
            
            # Get recent logs for anomaly detection
            recent_logs = self.get_logs(limit=limit)
            
            if not recent_logs:
                return []
                
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(recent_logs)
            
            # Sort anomalies by score
            anomalies.sort(key=lambda x: x.get('anomaly_score', 0), reverse=True)
            
            return anomalies
            
        except Exception as e:
            logger.error("Error detecting anomalies", error=str(e))
            return []
    
    def get_log_summary(self, logs=None, limit=1000):
        """
        Generate summary statistics from log data.
        
        Args:
            logs: Optional pre-fetched logs to analyze
            limit: Maximum number of logs to fetch if logs not provided
            
        Returns:
            Dictionary with log summary statistics
        """
        if logs is None:
            logs = self.get_logs(limit=limit)
            
        if not logs:
            return {
                "total_logs": 0,
                "levels": {},
                "components": {},
                "timeline": [],
                "recent_errors": []
            }
            
        df = pd.DataFrame(logs)
        
        # Add timestamp as datetime
        if 'timestamp' in df.columns:
            df["datetime"] = pd.to_datetime(df['timestamp'])
            
        # Count by level
        level_counts = df.get('level', pd.Series()).value_counts().to_dict()
        
        # Count by component
        component_counts = df.get('component', pd.Series()).value_counts().to_dict()
        
        # Timeline data
        timeline_data = []
        if 'datetime' in df.columns:
            timeline = df.groupby([pd.Grouper(key='datetime', freq='1H'), 'level']).size().reset_index(name='count')
            timeline_data = timeline.to_dict(orient='records')
        
        # Recent errors
        recent_errors = []
        if 'level' in df.columns:
            errors = df[df['level'].isin(['error', 'critical'])]
            if not errors.empty and 'datetime' in errors.columns:
                errors = errors.sort_values('datetime', ascending=False)
                recent_errors = errors.head(10).to_dict(orient='records')
        
        return {
            "total_logs": len(logs),
            "levels": level_counts,
            "components": component_counts,
            "timeline": timeline_data,
            "recent_errors": recent_errors
        }


def create_dashboard():
    """
    Create and configure the Dash app for the log dashboard.
    """
    # Constants
    REFRESH_INTERVAL = 10  # seconds
    
    # Create analyzer
    analyzer = LogAnalyzer()
    
    # Create Dash app with theme store
    app = dash.Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Store the current theme
    THEME = dbc.themes.BOOTSTRAP
    
    # Helper functions for enhanced visualizations
    def create_gauge_chart(title, value, max_value=100, color_thresholds=None):
        if color_thresholds is None:
            color_thresholds = [(0, "green"), (70, "yellow"), (90, "red")]
        
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, color_thresholds[0][0]], 'color': color_thresholds[0][1]},
                    {'range': [color_thresholds[0][0], color_thresholds[1][0]], 'color': color_thresholds[1][1]},
                    {'range': [color_thresholds[1][0], max_value], 'color': color_thresholds[2][1]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
    
    # Define layout
    app.layout = html.Div([
        # Store for theme state
        dcc.Store(id='theme-store', data={"theme": "light"}),
        
        # Stylesheet for theme switching
        html.Link(
            id="_stylesheet", 
            rel="stylesheet",
            href=dbc.themes.BOOTSTRAP
        ),
        
        # Main content
        html.Div(
            id="main-content",
            children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H1("Log Analytics Dashboard", className="text-center mb-4 mt-4"),
                            html.P(id="update-time", className="text-center text-muted"),
                        ], width=12)
                    ]),
                    
                    dbc.Tabs([
                        # Existing Overview Tab
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id='level-pie-chart'), md=4),
                                    dbc.Col(dcc.Graph(id='component-bar-chart'), md=8)
                                ]),
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id='timeline-chart'), width=12)
                                ])
                            ]),
                            label='Overview'
                        ),
                        # Log Query Tab with Enhanced Filtering
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        # Advanced filtering options
                                        dbc.Card([
                                            dbc.CardHeader("Advanced Filters"),
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Label("Log Level"),
                                                        dcc.Dropdown(
                                                            id='level-filter',
                                                            options=[
                                                                {'label': 'Error', 'value': 'error'},
                                                                {'label': 'Warning', 'value': 'warning'},
                                                                {'label': 'Info', 'value': 'info'},
                                                                {'label': 'Debug', 'value': 'debug'}
                                                            ],
                                                            multi=True
                                                        )
                                                    ], md=6),
                                                    dbc.Col([
                                                        html.Label("Component"),
                                                        dcc.Dropdown(
                                                            id='component-filter',
                                                            options=[
                                                                {'label': 'System', 'value': 'system'},
                                                                {'label': 'API', 'value': 'api'},
                                                                {'label': 'Database', 'value': 'database'},
                                                                {'label': 'Example', 'value': 'example'}
                                                            ],
                                                            multi=True
                                                        )
                                                    ], md=6)
                                                ]),
                                                html.Hr(),
                                                dcc.Textarea(
                                                    id='log-query-input',
                                                    placeholder='Enter query (e.g. level:"error" AND component:"api")',
                                                    style={'height': 100, 'width': '100%'}
                                                ),
                                                dbc.Button('Execute Query', 
                                                    id='query-submit', 
                                                    color='primary',
                                                    className='mt-2'
                                                )
                                            ])
                                        ])
                                    ], md=4),
                                    dbc.Col([
                                        html.Div([
                                            html.H4("Query Results"),
                                            html.Div(id='query-results',
                                                style={'height': '400px', 'overflowY': 'scroll'}
                                            ),
                                            html.Div([
                                                dbc.Button("Export CSV", id="export-csv", color="secondary", size="sm", className="me-1"),
                                                dbc.Button("Export JSON", id="export-json", color="secondary", size="sm")
                                            ], className="mt-2 d-flex justify-content-end")
                                        ], className="border p-3 rounded")
                                    ], md=8)
                                ])
                            ]),
                            label='Query Explorer'
                        ),
                        # Health Monitoring Tab with Gauge Charts
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("System Health Metrics", className="mb-3"),
                                        dbc.Row([
                                            dbc.Col(dcc.Graph(id='cpu-gauge'), md=6),
                                            dbc.Col(dcc.Graph(id='memory-gauge'), md=6)
                                        ]),
                                        dbc.Row([
                                            dbc.Col(dcc.Graph(id='disk-gauge'), md=6),
                                            dbc.Col(dcc.Graph(id='error-gauge'), md=6)
                                        ]),
                                        html.Div(id='health-alerts', className="mt-3")
                                    ], md=6),
                                    dbc.Col([
                                        html.H4("Performance Trends", className="mb-3"),
                                        dcc.Graph(id='health-graph', style={"height": "400px"}),
                                        dcc.Interval(
                                            id='health-update-interval',
                                            interval=10*1000
                                        )
                                    ], md=6)
                                ])
                            ]),
                            label='Health Monitor'
                        ),
                        # Log Replay Tab with Enhanced UI
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardHeader("Replay Configuration"),
                                            dbc.CardBody([
                                                html.Label("Component"),
                                                dcc.Dropdown(
                                                    id='replay-component',
                                                    options=[
                                                        {'label': 'System', 'value': 'system'},
                                                        {'label': 'API', 'value': 'api'},
                                                        {'label': 'Database', 'value': 'database'},
                                                        {'label': 'Example', 'value': 'example'}
                                                    ],
                                                    placeholder='Select Component'
                                                ),
                                                html.Label("Request ID", className="mt-2"),
                                                dcc.Input(
                                                    id='replay-request-id',
                                                    placeholder='Request ID',
                                                    className="form-control"
                                                ),
                                                dbc.Button('Start Replay',
                                                    id='replay-start',
                                                    color='success',
                                                    className='mt-3 w-100'
                                                )
                                            ])
                                        ])
                                    ], md=4),
                                    dbc.Col([
                                        html.Div([
                                            html.H4("Replay Results"),
                                            html.Div(id='replay-output',
                                                className="border p-3 bg-light",
                                                style={'height': '500px', 'overflowY': 'scroll', 'fontFamily': 'monospace'}
                                            )
                                        ])
                                    ], md=8)
                                ])
                            ]),
                            label='Log Replay'
                        ),
                        # Anomaly Detection Tab
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Anomaly Detection"),
                                        html.P("Automatically detect unusual patterns and potential issues in logs."),
                                        dbc.Card([
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Label("Analysis Window"),
                                                        dcc.Dropdown(
                                                            id="anomaly-window",
                                                            options=[
                                                                {"label": "Last hour", "value": "1h"},
                                                                {"label": "Last 12 hours", "value": "12h"},
                                                                {"label": "Last day", "value": "24h"},
                                                                {"label": "Last week", "value": "7d"},
                                                                {"label": "All logs", "value": "all"}
                                                            ],
                                                            value="24h"
                                                        ),
                                                    ], width=6),
                                                    dbc.Col([
                                                        html.Label("Sensitivity"),
                                                        dcc.Slider(
                                                            id="anomaly-sensitivity",
                                                            min=0.5,
                                                            max=1.0,
                                                            step=0.05,
                                                            value=0.95,
                                                            marks={
                                                                0.5: "Low",
                                                                0.75: "Medium",
                                                                1.0: "High"
                                                            }
                                                        ),
                                                    ], width=6),
                                                ]),
                                                html.Br(),
                                                dbc.Button("Detect Anomalies", id="detect-anomalies-btn", color="primary"),
                                                html.Br(),
                                                html.Div(id="anomalies-loading", children=[
                                                    dcc.Loading(
                                                        id="anomalies-loading-spinner",
                                                        type="circle",
                                                        children=html.Div(id="anomalies-content")
                                                    )
                                                ])
                                            ])
                                        ])
                                    ])
                                ])
                            ]),
                            label='Anomaly Detection'
                        ),
                        
                        # Advanced Analytics Tab
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Advanced Analytics"),
                                        html.P("Explore log patterns with advanced visualizations and custom date ranges.")
                                    ], width=12)
                                ]),
                                
                                # Date Range Selector
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H5("Custom Date Range", className="card-title"),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Label("Start Date"),
                                                        dcc.DatePickerSingle(
                                                            id='analytics-start-date',
                                                            date=datetime.now().date() - timedelta(days=7),
                                                            display_format='YYYY-MM-DD'
                                                        ),
                                                    ], width=6),
                                                    dbc.Col([
                                                        html.Label("End Date"),
                                                        dcc.DatePickerSingle(
                                                            id='analytics-end-date',
                                                            date=datetime.now().date(),
                                                            display_format='YYYY-MM-DD'
                                                        ),
                                                    ], width=6),
                                                ]),
                                                html.Br(),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Label("Visualization Type"),
                                                        dcc.Dropdown(
                                                            id='analytics-viz-type',
                                                            options=[
                                                                {'label': 'Time Heatmap', 'value': 'time_heatmap'},
                                                                {'label': 'Log Patterns', 'value': 'log_patterns'},
                                                                {'label': 'Field Correlations', 'value': 'correlations'},
                                                                {'label': 'Volume Comparison', 'value': 'volume_comparison'},
                                                                {'label': 'Error Distribution', 'value': 'error_distribution'}
                                                            ],
                                                            value='time_heatmap'
                                                        ),
                                                    ], width=6),
                                                    dbc.Col([
                                                        html.Label("Field to Analyze"),
                                                        dcc.Dropdown(
                                                            id='analytics-field',
                                                            options=[
                                                                {'label': 'Log Level', 'value': 'level'},
                                                                {'label': 'Component', 'value': 'component'},
                                                                {'label': 'Error Type', 'value': 'error_type'},
                                                                {'label': 'User ID', 'value': 'user_id'},
                                                                {'label': 'Request ID', 'value': 'request_id'}
                                                            ],
                                                            value='level'
                                                        ),
                                                    ], width=6),
                                                ]),
                                                html.Br(),
                                                dbc.Button("Generate Visualization", id="generate-viz-btn", color="primary"),
                                                html.Br(),
                                                html.Div([
                                                    html.Label("Export Format"),
                                                    dcc.Dropdown(
                                                        id='export-format',
                                                        options=[
                                                            {'label': 'CSV', 'value': 'csv'},
                                                            {'label': 'JSON', 'value': 'json'},
                                                            {'label': 'Excel', 'value': 'excel'},
                                                            {'label': 'Image (PNG)', 'value': 'png'}
                                                        ],
                                                        value='csv',
                                                        style={'width': '200px'}
                                                    ),
                                                    dbc.Button("Export Data", id="export-data-btn", color="secondary", className="ml-2"),
                                                    dcc.Download(id="download-data")
                                                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '10px'})
                                            ])
                                        ])
                                    ], width=12)
                                ]),
                                
                                # Visualization Output
                                dbc.Row([
                                    dbc.Col([
                                        html.Br(),
                                        html.Div(id="advanced-viz-loading", children=[
                                            dcc.Loading(
                                                id="advanced-viz-loading-spinner",
                                                type="circle",
                                                children=html.Div(id="advanced-viz-content")
                                            )
                                        ])
                                    ], width=12)
                                ])
                            ]),
                            label='Advanced Analytics'
                        )
                    ], id="tabs", className='mb-4'),
                    
                    dcc.Interval(
                        id="interval-component",
                        interval=REFRESH_INTERVAL * 1000,  # Convert to milliseconds
                        n_intervals=0
                    )
                ], fluid=True)
            ]
        ),
        
        # Theme toggle button in top-right corner
        html.Div([
            dbc.Button(
                html.I(className="bi bi-sun", id="theme-icon"),
                id="theme-toggle",
                className="rounded-circle",
                style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 1000}
            )
        ]),
        
        # Anomaly alert toast
        dbc.Toast(
            id="anomaly-alert",
            header="System Anomaly Detected",
            icon="danger",
            dismissable=True,
            is_open=False,
            style={"position": "fixed", "top": 70, "right": 10, "width": 350, "zIndex": 1000}
        )
    ])
    
    # Theme toggle callback - working version
    @app.callback(
        [Output("theme-store", "data"),
         Output("theme-icon", "className"),
         Output("_stylesheet", "href")],
        [Input("theme-toggle", "n_clicks")],
        [State("theme-store", "data")]
    )
    def toggle_theme(n_clicks, data):
        if n_clicks is None:
            # Initialize with light theme
            return {"theme": "light"}, "bi bi-moon", dbc.themes.BOOTSTRAP
        
        if data["theme"] = = "light":
            # Switch to dark theme
            return {"theme": "dark"}, "bi bi-sun", dbc.themes.DARKLY
        else:
            # Switch to light theme
            return {"theme": "light"}, "bi bi-moon", dbc.themes.BOOTSTRAP
    
    # Callback for updating overview charts
    @app.callback(
        [
            Output("update-time", "children"),
            Output("level-pie-chart", "figure"),
            Output("component-bar-chart", "figure"),
            Output("timeline-chart", "figure")
        ],
        Input("interval-component", "n_intervals")
    )
    def update_overview_charts(n):
        """Update the overview charts with fresh log data."""
        # Get real log data
        logs = analyzer.get_logs(limit=1000)
        summary = analyzer.get_log_summary(logs)
        
        # Create timestamp
        timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create level pie chart
        if summary["levels"]:
            level_df = pd.DataFrame(list(summary["levels"].items()), columns=["level", "count"])
            level_pie = px.pie(
                level_df, 
                values="count", 
                names="level", 
                title="Logs by Level",
                color="level",
                color_discrete_map={
                    "error": "#FF4136",
                    "warning": "#FF851B",
                    "info": "#0074D9",
                    "debug": "#2ECC40"
                }
            )
            level_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        else:
            level_pie = go.Figure(go.Pie(
                labels=["No Data"],
                values=[1],
                textinfo="label"
            ))
            level_pie.update_layout(title_text="Logs by Level (No Data)")
        
        # Create component bar chart
        if summary["components"]:
            component_df = pd.DataFrame(list(summary["components"].items()), columns=["component", "count"])
            component_df = component_df.sort_values("count", ascending=True)
            component_bar = px.bar(
                component_df, 
                x="count", 
                y="component", 
                title="Logs by Component",
                orientation="h"
            )
            component_bar.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        else:
            component_bar = go.Figure(go.Bar(
                x=[0],
                y=["No Data"],
                orientation="h"
            ))
            component_bar.update_layout(title_text="Logs by Component (No Data)")
        
        # Create timeline chart
        if summary["timeline"]:
            timeline_df = pd.DataFrame(summary["timeline"])
            timeline_df["datetime"] = pd.to_datetime(timeline_df["datetime"])
            timeline_chart = px.line(
                timeline_df,
                x="datetime",
                y="count",
                color="level",
                title="Log Volume Over Time",
                color_discrete_map={
                    "error": "#FF4136",
                    "warning": "#FF851B",
                    "info": "#0074D9",
                    "debug": "#2ECC40"
                }
            )
            timeline_chart.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        else:
            timeline_chart = go.Figure()
            timeline_chart.add_annotation(
                text="No timeline data available",
                showarrow=False,
                font=dict(size=16)
            )
            timeline_chart.update_layout(title_text="Log Volume Over Time")
        
        return timestamp, level_pie, component_bar, timeline_chart
    
    # Callback for executing log queries
    @app.callback(
        Output("query-results", "children"),
        [Input("query-submit", "n_clicks")],
        [State("log-query-input", "value"),
         State("level-filter", "value"),
         State("component-filter", "value")]
    )
    def execute_log_query(n_clicks, query_text, level_filter, component_filter):
        """Execute a log query and return the results."""
        if n_clicks is None:
            return html.Div("Enter a query and click 'Execute Query' to search logs")
        
        # Build query from text and filters
        query_parts = []
        if query_text:
            query_parts.append(f"({query_text})")
        
        if level_filter:
            level_conditions = " OR ".join([f'level = "{level}"' for level in level_filter])
            query_parts.append(f"({level_conditions})")
        
        if component_filter:
            component_conditions = " OR ".join([f'component = "{comp}"' for comp in component_filter])
            query_parts.append(f"({component_conditions})")
        
        final_query = " AND ".join(query_parts) if query_parts else None
        
        # Execute query
        try:
            logs = analyzer.get_logs(query=final_query, limit=100)
            
            if not logs:
                return html.Div("No logs matching your query were found", className="alert alert-warning")
            
            # Format results as a table
            table_rows = []
            for log in logs:
                timestamp = log.get("timestamp", "N/A")
                level = log.get("level", "N/A")
                component = log.get("component", "N/A")
                message = log.get("message", "N/A")
                
                level_class = {
                    "error": "text-danger",
                    "warning": "text-warning",
                    "info": "text-info",
                    "debug": "text-muted"
                }.get(level, "")
                
                row = html.Tr([
                    html.Td(timestamp),
                    html.Td(level, className=level_class),
                    html.Td(component),
                    html.Td(message)
                ])
                table_rows.append(row)
            
            table = dbc.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Timestamp"),
                            html.Th("Level"),
                            html.Th("Component"),
                            html.Th("Message")
                        ])
                    ),
                    html.Tbody(table_rows)
                ],
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
            
            return html.Div([
                html.P(f"Found {len(logs)} matching logs"),
                table
            ])
        
        except Exception as e:
            logger.error("Error executing query", error=str(e), query=final_query)
            return html.Div(f"Error executing query: {str(e)}", className="alert alert-danger")
    
    # Add callbacks for gauge charts with real system health data
    @app.callback(
        [Output('cpu-gauge', 'figure'), 
         Output('memory-gauge', 'figure'),
         Output('disk-gauge', 'figure'),
         Output('error-gauge', 'figure')],
        Input('health-update-interval', 'n_intervals')
    )
    def update_gauges(n):
        # Get real system health metrics
        try:
            # Get real metrics from HealthMonitor
            health_monitor = HealthMonitor()
            cpu_usage = health_monitor.get_cpu_usage()
            memory_usage = health_monitor.get_memory_usage()
            disk_usage = health_monitor.get_disk_usage()
            
            # Get error rate from logs (last hour)
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            error_query = 'level = "error" AND timestamp > "{}"'.format(
                one_hour_ago.strftime("%Y-%m-%d %H:%M:%S")
            )
            error_logs = analyzer.get_logs(query=error_query)
            error_rate = len(error_logs)  # Count of errors in the last hour
        except Exception as e:
            logger.error("Error fetching health metrics", error=str(e))
            # Use fallback values if real metrics are unavailable
            cpu_usage = 50
            memory_usage = 40
            disk_usage = 60
            error_rate = 5
        
        cpu_gauge = create_gauge_chart('CPU Usage (%)', cpu_usage)
        memory_gauge = create_gauge_chart('Memory Usage (%)', memory_usage)
        disk_gauge = create_gauge_chart('Disk Usage (%)', disk_usage)
        error_gauge = create_gauge_chart('Error Rate (last hour)', error_rate, max_value=50, 
                                      color_thresholds=[(0, "green"), (10, "yellow"), (30, "red")])
        
        return cpu_gauge, memory_gauge, disk_gauge, error_gauge
    
    # Callback for health metrics chart with real data
    @app.callback(
        Output('health-graph', 'figure'),
        Input('health-update-interval', 'n_intervals')
    )
    def update_health_metrics(n):
        try:
            # Get error logs for the past 24 hours grouped by hour
            current_time = datetime.now()
            one_day_ago = current_time - timedelta(days=1)
            error_query = 'level = "error" AND timestamp > "{}"'.format(
                one_day_ago.strftime("%Y-%m-%d %H:%M:%S")
            )
            error_logs = analyzer.get_logs(query=error_query)
            
            if not error_logs:
                # Create empty placeholder graph if no data
                fig = go.Figure()
                fig.add_annotation(
                    text="No health metrics data available",
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title_text="System Health Metrics (24h)")
                return fig
            
            # Convert to dataframe and group by hour
            df = pd.DataFrame(error_logs)
            if 'timestamp' in df.columns:
                df["datetime"] = pd.to_datetime(df['timestamp'])
                hourly_errors = df.groupby(pd.Grouper(key='datetime', freq='1H')).size().reset_index(name='count')
                
                # Create the figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_errors['datetime'],
                    y=hourly_errors['count'],
                    mode='lines+markers',
                    name='Errors',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title='Error Rate (24h)',
                    xaxis_title='Time',
                    yaxis_title='Error Count',
                    template='plotly_white'
                )
                return fig
            else:
                # Create empty placeholder graph if timestamp data is missing
                fig = go.Figure()
                fig.add_annotation(
                    text="No timestamp data in logs",
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title_text="System Health Metrics (24h)")
                return fig
                
        except Exception as e:
            logger.error("Error updating health metrics", error=str(e))
            # Create error placeholder graph
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error retrieving health metrics: {str(e)}",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title_text="System Health Metrics (24h)")
            return fig
    
    # Callback for log replay functionality
    @app.callback(
        Output('replay-output', 'children'),
        [Input('replay-start', 'n_clicks')],
        [State('replay-component', 'value'), 
         State('replay-request-id', 'value')]
    )
    def start_replay(n_clicks, component, request_id):
        if n_clicks is None:
            return "Select a component or enter a request ID, then click 'Start Replay'"
        
        if not component and not request_id:
            return "Please provide either a component or request ID for replay"
        
        try:
            # Create filter for the replay
            filters = {}
            if component:
                filters["component"] = component
            if request_id:
                filters["request_id"] = request_id
            
            # Create a log replay handler that captures output
            replay_output = []
            
            def log_handler(entry):
                """Handler that captures log entries for display."""
                replay_output.append(entry)
                return True
            
            # Create replay session
            replayer = LogReplay(
                log_dir=str(LOG_DIR),
                filters=filters,
                handlers={"*": log_handler}
            )
            
            # Run the replay
            replayer.replay()
            
            if not replay_output:
                return "No matching logs found for replay"
            
            # Format the output
            formatted_logs = []
            for entry in replay_output:
                timestamp = entry.get('timestamp', '')
                level = entry.get('level', '')
                component_name = entry.get('component', '')
                message = entry.get('message', '')
                
                level_class = {
                    "error": "text-danger fw-bold",
                    "warning": "text-warning",
                    "info": "text-info",
                    "debug": "text-muted"
                }.get(level, "")
                
                log_line = html.Div([
                    html.Span(f"{timestamp} ", className="text-secondary"),
                    html.Span(f"[{level.upper()}] ", className=level_class),
                    html.Span(f"{component_name}: ", className="fw-bold"),
                    html.Span(message)
                ], className="mb-1")
                
                formatted_logs.append(log_line)
            
            return html.Div([
                html.P(f"Replayed {len(formatted_logs)} log entries", className="alert alert-success"),
                html.Div(formatted_logs)
            ])
            
        except Exception as e:
            logger.error("Error in log replay", error=str(e))
            return html.Div(f"Error during replay: {str(e)}", className="alert alert-danger")
    
    # Add callback for anomaly detection
    @app.callback(
        Output("anomalies-content", "children"),
        [Input("detect-anomalies-btn", "n_clicks")],
        [State("anomaly-window", "value"),
         State("anomaly-sensitivity", "value")]
    )
    def detect_anomalies(n_clicks, window, sensitivity):
        """Detect anomalies in logs based on specified parameters."""
        if n_clicks is None:
            return html.Div("Click the 'Detect Anomalies' button to start anomaly detection.")
        
        # Update anomaly detector sensitivity
        analyzer.anomaly_detector.sensitivity = sensitivity
        
        # Convert window to limit based on selection
        limits = {
            "1h": 500,    # Approximately logs from last hour
            "12h": 2000,  # Approximately logs from last 12 hours
            "24h": 5000,  # Approximately logs from last day
            "7d": 10000,  # Approximately logs from last week
            "all": 20000  # All available logs
        }
        limit = limits.get(window, 5000)
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(limit=limit)
        
        if not anomalies:
            return html.Div([
                html.Br(),
                html.P("No anomalies detected.", className="text-success")
            ])
        
        # Format anomalies into a table
        headers = ["Timestamp", "Level", "Component", "Message", "Anomaly Type", "Score", "Explanation"]
        rows = []
        
        for anomaly in anomalies[:20]:  # Limit to top 20 anomalies
            timestamp = anomaly.get("timestamp", "")
            level = anomaly.get("level", "")
            component = anomaly.get("component", "")
            message = anomaly.get("event", "")
            anomaly_type = anomaly.get("anomaly_type", "")
            score = anomaly.get("anomaly_score", 0)
            explanation = anomaly.get("anomaly_explanation", "")
            
            # Format score
            score_formatted = f"{score:.2f}" if score else ""
            
            # Add row with color based on anomaly score
            row_color = "table-danger" if score > 3 else "table-warning"
            
            rows.append(html.Tr([
                html.Td(timestamp),
                html.Td(level),
                html.Td(component),
                html.Td(message),
                html.Td(anomaly_type),
                html.Td(score_formatted),
                html.Td(explanation)
            ], className=row_color))
        
        return html.Div([
            html.H5(f"Detected {len(anomalies)} anomalies"),
            html.P("Showing top 20 anomalies by score"),
            dbc.Table([
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody(rows)
            ], striped=True, bordered=True, hover=True, responsive=True),
            # Add a scatter plot of anomalies by time and severity
            dcc.Graph(
                figure=create_anomaly_scatter_plot(anomalies),
                style={"height": "400px"}
            )
        ])
    
    def create_anomaly_scatter_plot(anomalies):
        """Create a scatter plot of anomalies by time and score."""
        try:
            # Extract data for plotting
            data = []
            for anomaly in anomalies:
                try:
                    timestamp = anomaly.get("timestamp", "")
                    if not timestamp:
                        continue
                        
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                        
                    data.append({
                        "timestamp": dt,
                        "score": anomaly.get("anomaly_score", 0),
                        "type": anomaly.get("anomaly_type", "unknown"),
                        "component": anomaly.get("component", "unknown"),
                        "message": anomaly.get("event", "")[:50] + "..." if len(anomaly.get("event", "")) > 50 else anomaly.get("event", "")
                    })
                except Exception:
                    continue
            
            if not data:
                return go.Figure()
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Create scatter plot
            fig = px.scatter(
                df, 
                x="timestamp", 
                y="score", 
                color="type",
                hover_name="message",
                hover_data=["component"],
                size="score",
                title="Anomalies by Time and Severity"
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                legend_title="Anomaly Type"
            )
            
            return fig
        except Exception as e:
            logger.error("Error creating anomaly scatter plot", error=str(e))
            return go.Figure()
    
    # Add callback for advanced analytics visualizations
    @app.callback(
        Output("advanced-viz-content", "children"),
        [Input("generate-viz-btn", "n_clicks")],
        [State("analytics-start-date", "date"),
         State("analytics-end-date", "date"),
         State("analytics-viz-type", "value"),
         State("analytics-field", "value")]
    )
    def generate_advanced_visualization(n_clicks, start_date, end_date, viz_type, field):
        """Generate advanced visualizations based on user selections."""
        if n_clicks is None:
            return html.Div("Select parameters and click 'Generate Visualization' to create a visualization.")
        
        try:
            # Convert string dates to datetime objects
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                # Set to end of day
                end_date = end_date.replace(hour=23, minute=59, second=59)
            
            # Query logs within date range
            query_params = {
                "start_time": start_date,
                "end_time": end_date,
                "limit": 10000  # Reasonable limit for visualization
            }
            logs = analyzer.query_logs(**query_params)
            
            if not logs:
                return html.Div([
                    html.Br(),
                    html.P("No logs found in the selected date range.", className="text-warning")
                ])
            
            # Generate visualization based on type
            if viz_type == 'time_heatmap':
                fig = create_time_heatmap(logs, value_field=field, title=f"Log Activity Heatmap by {field.capitalize()}")
                return dcc.Graph(figure=fig)
                
            elif viz_type == 'log_patterns':
                fig = create_log_patterns_chart(logs, pattern_field=field, title=f"Log Patterns by {field.capitalize()}")
                return dcc.Graph(figure=fig)
                
            elif viz_type == 'correlations':
                # For correlations, we need to specify multiple fields
                fields = ['level', 'component', 'error_type', 'user_id', 'request_id']
                fig = create_correlation_heatmap(logs, fields=fields, title="Log Field Correlations")
                return dcc.Graph(figure=fig)
                
            elif viz_type == 'volume_comparison':
                fig = create_log_volume_comparison(logs, compare_field=field, title=f"Log Volume by {field.capitalize()}")
                return dcc.Graph(figure=fig)
                
            elif viz_type == 'error_distribution':
                fig = create_error_distribution_chart(logs, component_field=field if field != 'error_type' else 'component')
                return dcc.Graph(figure=fig)
                
            else:
                return html.Div("Visualization type not recognized.")
                
        except Exception as e:
            logger.error("Error generating advanced visualization", error=str(e))
            return html.Div([
                html.Br(),
                html.P(f"Error generating visualization: {str(e)}", className="text-danger")
            ])
    
    # Add callback for exporting data
    @app.callback(
        Output("download-data", "data"),
        [Input("export-data-btn", "n_clicks")],
        [State("analytics-start-date", "date"),
         State("analytics-end-date", "date"),
         State("analytics-field", "value"),
         State("export-format", "value")]
    )
    def export_data(n_clicks, start_date, end_date, field, export_format):
        """Export log data in various formats."""
        if n_clicks is None:
            return None
        
        try:
            # Convert string dates to datetime objects
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                # Set to end of day
                end_date = end_date.replace(hour=23, minute=59, second=59)
            
            # Query logs within date range
            query_params = {
                "start_time": start_date,
                "end_time": end_date,
                "limit": 10000
            }
            logs = analyzer.query_logs(**query_params)
            
            if not logs:
                return None
            
            # Create a DataFrame from logs
            df = pd.DataFrame(logs)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_data_{timestamp}"
            
            # Export based on format
            if export_format == 'csv':
                return dcc.send_data_frame(df.to_csv, f"{filename}.csv", index=False)
            elif export_format == 'json':
                return dict(content=df.to_json(orient='records'), filename=f"{filename}.json")
            elif export_format == 'excel':
                return dcc.send_data_frame(df.to_excel, f"{filename}.xlsx", index=False)
            elif export_format == 'png':
                # For PNG, we need to create a visualization first
                if field == 'level':
                    fig = create_log_patterns_chart(logs, pattern_field=field)
                else:
                    fig = create_time_heatmap(logs, value_field=field)
                
                # Convert figure to image
                img_bytes = fig.to_image(format="png")
                return dict(content=img_bytes, filename=f"{filename}.png", type="image/png")
            else:
                return None
                
        except Exception as e:
            logger.error("Error exporting data", error=str(e))
            return None
    
    return app


def get_level_color(level: str) -> str:
    """Get color for log level."""
    level = level.lower()
    if level == 'error':
        return 'red'
    elif level == 'warning':
        return 'orange'
    elif level == 'info':
        return 'blue'
    elif level == 'debug':
        return 'green'
    return 'black'


def run_dashboard(debug: bool = False):
    """Run the dashboard server."""
    app = create_dashboard()
    if app:
        print(f"Starting Log Analytics Dashboard on http://localhost:{PORT}")
        app.run_server(debug=debug, port=PORT)
    else:
        print("Failed to create dashboard. Check if Dash is installed.")


if __name__ == "__main__":
    run_dashboard(debug=True)
