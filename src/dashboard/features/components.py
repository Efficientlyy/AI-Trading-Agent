"""
Dashboard Components Module

This module provides reusable UI components for the dashboard.
"""

import logging
from typing import Dict, Any, List, Optional
from flask import render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard_components")

def render_chart_component(chart_id: str, title: str, data_url: str, chart_type: str = 'candlestick', height: int = 400) -> str:
    """
    Render a chart component.
    
    Args:
        chart_id: Unique ID for the chart
        title: Chart title
        data_url: URL to fetch chart data
        chart_type: Type of chart (candlestick, line, bar)
        height: Chart height in pixels
        
    Returns:
        Rendered chart component HTML
    """
    return render_template('components/chart.html',
                          chart_id=chart_id,
                          title=title,
                          data_url=data_url,
                          chart_type=chart_type,
                          height=height)

def render_data_table(table_id: str, title: str, data_url: str, columns: List[Dict[str, Any]], page_size: int = 10) -> str:
    """
    Render a data table component.
    
    Args:
        table_id: Unique ID for the table
        title: Table title
        data_url: URL to fetch table data
        columns: List of column definitions
        page_size: Number of rows per page
        
    Returns:
        Rendered data table component HTML
    """
    return render_template('components/data_table.html',
                          table_id=table_id,
                          title=title,
                          data_url=data_url,
                          columns=columns,
                          page_size=page_size)

def render_metric_card(metric_id: str, title: str, value: Any, unit: Optional[str] = None, 
                      change: Optional[float] = None, icon: Optional[str] = None) -> str:
    """
    Render a metric card component.
    
    Args:
        metric_id: Unique ID for the metric
        title: Metric title
        value: Metric value
        unit: Unit of measurement (optional)
        change: Percentage change (optional)
        icon: Icon name (optional)
        
    Returns:
        Rendered metric card component HTML
    """
    return render_template('components/metric_card.html',
                          metric_id=metric_id,
                          title=title,
                          value=value,
                          unit=unit,
                          change=change,
                          icon=icon)

def render_alert_list(alert_id: str, title: str, data_url: str, max_items: int = 5) -> str:
    """
    Render an alert list component.
    
    Args:
        alert_id: Unique ID for the alert list
        title: Alert list title
        data_url: URL to fetch alert data
        max_items: Maximum number of alerts to display
        
    Returns:
        Rendered alert list component HTML
    """
    return render_template('components/alert_list.html',
                          alert_id=alert_id,
                          title=title,
                          data_url=data_url,
                          max_items=max_items)

def render_control_panel(panel_id: str, title: str, controls: List[Dict[str, Any]]) -> str:
    """
    Render a control panel component.
    
    Args:
        panel_id: Unique ID for the control panel
        title: Control panel title
        controls: List of control definitions
        
    Returns:
        Rendered control panel component HTML
    """
    return render_template('components/control_panel.html',
                          panel_id=panel_id,
                          title=title,
                          controls=controls)

def render_sentiment_gauge(gauge_id: str, title: str, value: float, min_value: float = -1.0, max_value: float = 1.0) -> str:
    """
    Render a sentiment gauge component.
    
    Args:
        gauge_id: Unique ID for the gauge
        title: Gauge title
        value: Sentiment value (-1 to 1)
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Rendered sentiment gauge component HTML
    """
    return render_template('components/sentiment_gauge.html',
                          gauge_id=gauge_id,
                          title=title,
                          value=value,
                          min_value=min_value,
                          max_value=max_value)

def render_performance_chart(chart_id: str, title: str, data_url: str, height: int = 300) -> str:
    """
    Render a performance chart component.
    
    Args:
        chart_id: Unique ID for the chart
        title: Chart title
        data_url: URL to fetch performance data
        height: Chart height in pixels
        
    Returns:
        Rendered performance chart component HTML
    """
    return render_template('components/performance_chart.html',
                          chart_id=chart_id,
                          title=title,
                          data_url=data_url,
                          height=height)

def render_system_status(status_id: str, system_state: str, trading_state: str, system_mode: str, data_source: str) -> str:
    """
    Render a system status component.
    
    Args:
        status_id: Unique ID for the status component
        system_state: Current system state
        trading_state: Current trading state
        system_mode: Current system mode
        data_source: Current data source
        
    Returns:
        Rendered system status component HTML
    """
    return render_template('components/system_status.html',
                          status_id=status_id,
                          system_state=system_state,
                          trading_state=trading_state,
                          system_mode=system_mode,
                          data_source=data_source)
