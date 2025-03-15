"""
Advanced visualization utilities for the log dashboard.

This module provides reusable visualization components and functions
for creating rich, interactive data visualizations for log analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

def create_time_heatmap(logs: List[Dict[str, Any]], 
                       time_unit: str = 'hour',
                       value_field: str = 'level',
                       title: str = 'Log Activity Heatmap'):
    """
    Create a heatmap showing log activity over time.
    
    Args:
        logs: List of log entries
        time_unit: Time unit for grouping ('hour', 'day', 'weekday', 'hour_of_day')
        value_field: Field to use for coloring (e.g., 'level', 'component')
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not logs:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Extract timestamps and convert to datetime
    data = []
    for log in logs:
        if 'timestamp' in log:
            try:
                timestamp = log['timestamp']
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                value = log.get(value_field, 'unknown')
                
                # Extract time components based on time_unit
                if time_unit == 'hour':
                    time_key = dt.strftime('%Y-%m-%d %H:00')
                elif time_unit == 'day':
                    time_key = dt.strftime('%Y-%m-%d')
                elif time_unit == 'weekday':
                    time_key = dt.strftime('%A')  # Day name
                elif time_unit == 'hour_of_day':
                    time_key = dt.strftime('%H:00')  # Hour of day
                else:
                    time_key = dt.strftime('%Y-%m-%d %H:00')
                
                data.append({
                    'time': time_key,
                    'value': value,
                    'count': 1
                })
            except (ValueError, TypeError):
                continue
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No valid timestamp data", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Create DataFrame and pivot for heatmap
    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(
        index='time', 
        columns='value', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Sort index appropriately
    if time_unit == 'weekday':
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df.reindex(weekday_order)
    elif time_unit == 'hour_of_day':
        hour_order = [f'{h:02d}:00' for h in range(24)]
        pivot_df = pivot_df.reindex(hour_order)
    else:
        pivot_df = pivot_df.sort_index()
    
    # Create heatmap
    fig = px.imshow(
        pivot_df.T,
        labels=dict(x="Time", y=value_field.capitalize(), color="Count"),
        title=title,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis_nticks=20,
        height=400
    )
    
    # Rotate x-axis labels if needed
    if time_unit in ['hour', 'day']:
        fig.update_xaxes(tickangle=45)
    
    return fig

def create_log_patterns_chart(logs: List[Dict[str, Any]], 
                             pattern_field: str = 'component',
                             time_field: str = 'timestamp',
                             title: str = 'Log Patterns Over Time'):
    """
    Create a stacked area chart showing log patterns over time.
    
    Args:
        logs: List of log entries
        pattern_field: Field to use for patterns (e.g., 'component', 'level')
        time_field: Field containing timestamp
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not logs:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Extract data
    data = []
    for log in logs:
        if time_field in log and pattern_field in log:
            try:
                timestamp = log[time_field]
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                pattern_value = log.get(pattern_field, 'unknown')
                
                # Round to nearest hour for grouping
                hour_rounded = dt.replace(minute=0, second=0, microsecond=0)
                
                data.append({
                    'time': hour_rounded,
                    'pattern': pattern_value
                })
            except (ValueError, TypeError):
                continue
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No valid data for patterns", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Count occurrences by time and pattern
    pattern_counts = df.groupby(['time', 'pattern']).size().reset_index(name='count')
    
    # Pivot for plotting
    pivot_df = pattern_counts.pivot(index='time', columns='pattern', values='count').fillna(0)
    
    # Sort by time
    pivot_df = pivot_df.sort_index()
    
    # Create stacked area chart
    fig = px.area(
        pivot_df, 
        title=title,
        labels={'value': 'Count', 'variable': pattern_field.capitalize()}
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Count",
        legend_title=pattern_field.capitalize(),
        height=400
    )
    
    return fig

def create_correlation_heatmap(logs: List[Dict[str, Any]], 
                              fields: List[str] = None,
                              title: str = 'Log Field Correlations'):
    """
    Create a correlation heatmap between different log fields.
    
    Args:
        logs: List of log entries
        fields: List of fields to correlate (defaults to common fields)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not logs:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Default fields if none provided
    if fields is None:
        fields = ['level', 'component', 'request_id', 'user_id', 'error_type']
    
    # Extract field presence as binary features
    data = []
    for log in logs:
        entry = {}
        for field in fields:
            # Check if field exists and has a value
            entry[field] = 1 if field in log and log[field] else 0
        data.append(entry)
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No valid field data", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Field", y="Field", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        title=title,
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        height=500,
        width=600
    )
    
    return fig

def create_log_volume_comparison(logs: List[Dict[str, Any]], 
                                compare_field: str = 'component',
                                time_field: str = 'timestamp',
                                interval: str = 'day',
                                title: str = 'Log Volume Comparison'):
    """
    Create a comparison chart showing log volumes for different field values over time.
    
    Args:
        logs: List of log entries
        compare_field: Field to compare (e.g., 'component', 'level')
        time_field: Field containing timestamp
        interval: Time interval for grouping ('hour', 'day', 'week')
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not logs:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Extract data
    data = []
    for log in logs:
        if time_field in log and compare_field in log:
            try:
                timestamp = log[time_field]
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                field_value = log.get(compare_field, 'unknown')
                
                # Format time based on interval
                if interval == 'hour':
                    time_key = dt.replace(minute=0, second=0, microsecond=0)
                elif interval == 'day':
                    time_key = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                elif interval == 'week':
                    # Start of the week (Monday)
                    time_key = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    time_key = dt.replace(minute=0, second=0, microsecond=0)
                
                data.append({
                    'time': time_key,
                    'value': field_value
                })
            except (ValueError, TypeError):
                continue
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No valid timestamp data", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Count occurrences by time and value
    counts = df.groupby(['time', 'value']).size().reset_index(name='count')
    
    # Create comparison chart
    fig = px.line(
        counts, 
        x='time', 
        y='count', 
        color='value',
        title=title,
        labels={
            'time': 'Time', 
            'count': 'Log Count', 
            'value': compare_field.capitalize()
        }
    )
    
    fig.update_layout(
        xaxis_title=f"Time ({interval})",
        yaxis_title="Log Count",
        legend_title=compare_field.capitalize(),
        height=400
    )
    
    return fig

def create_error_distribution_chart(logs: List[Dict[str, Any]], 
                                   error_field: str = 'error_type',
                                   component_field: str = 'component',
                                   title: str = 'Error Distribution by Component'):
    """
    Create a grouped bar chart showing error distributions by component.
    
    Args:
        logs: List of log entries
        error_field: Field containing error type
        component_field: Field containing component name
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Filter for error logs
    error_logs = [log for log in logs if log.get('level') in ['error', 'critical']]
    
    if not error_logs:
        fig = go.Figure()
        fig.add_annotation(text="No error logs available", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Extract data
    data = []
    for log in error_logs:
        if error_field in log or component_field in log:
            error_type = log.get(error_field, 'Unknown Error')
            component = log.get(component_field, 'Unknown Component')
            
            data.append({
                'error_type': error_type,
                'component': component
            })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No valid error data", showarrow=False, font=dict(size=16))
        fig.update_layout(title_text=title)
        return fig
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Count errors by type and component
    error_counts = df.groupby(['component', 'error_type']).size().reset_index(name='count')
    
    # Create grouped bar chart
    fig = px.bar(
        error_counts, 
        x='component', 
        y='count', 
        color='error_type',
        title=title,
        barmode='group',
        labels={
            'component': 'Component', 
            'count': 'Error Count', 
            'error_type': 'Error Type'
        }
    )
    
    fig.update_layout(
        xaxis_title="Component",
        yaxis_title="Error Count",
        legend_title="Error Type",
        height=400
    )
    
    return fig
