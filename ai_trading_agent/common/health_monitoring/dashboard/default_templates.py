"""
Default Templates for Health Monitoring Dashboard.

This module provides default template content for the dashboard when
templates do not exist in the filesystem.
"""

import logging

# Set up logger
logger = logging.getLogger(__name__)


def get_default_html_template(template_name: str) -> str:
    """
    Get default HTML template content by template name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template content as string
    """
    template_name = template_name.lower()
    
    if template_name == "index":
        return _get_index_html()
    elif template_name == "components":
        return _get_components_html()
    elif template_name == "metrics":
        return _get_metrics_html()
    elif template_name == "alerts":
        return _get_alerts_html()
    else:
        logger.warning(f"No default HTML template for {template_name}, returning empty template")
        return "<!-- Empty template -->"


def get_default_js_template(template_name: str) -> str:
    """
    Get default JavaScript template content by template name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template content as string
    """
    template_name = template_name.lower()
    
    if template_name == "dashboard":
        return "// This is just a placeholder. The actual dashboard.js file should be used."
    elif template_name == "components":
        return "// This is just a placeholder. The actual components.js file should be used."
    elif template_name == "metrics":
        return "// This is just a placeholder. The actual metrics.js file should be used."
    elif template_name == "alerts":
        return "// This is just a placeholder. The actual alerts.js file should be used."
    else:
        logger.warning(f"No default JS template for {template_name}, returning empty template")
        return "// Empty template"


def get_default_css_template(template_name: str) -> str:
    """
    Get default CSS template content by template name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template content as string
    """
    template_name = template_name.lower()
    
    if template_name == "dashboard":
        return "/* This is just a placeholder. The actual dashboard.css file should be used. */"
    else:
        logger.warning(f"No default CSS template for {template_name}, returning empty template")
        return "/* Empty template */"


def _get_index_html() -> str:
    """Get default index.html content."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System Health Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/dashboard.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">Trading System Health Dashboard</a>
            <span class="navbar-text" id="system-status-indicator">
                <span class="status-badge unknown">Unknown</span>
            </span>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link active" href="#overview">Overview</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#components">Components</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#metrics">Metrics</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#alerts">Alerts</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="alert alert-info" role="alert">
                    Last updated: <span id="last-update-time">Never</span>
                    <button id="refresh-btn" class="btn btn-sm btn-primary float-end">Refresh Now</button>
                </div>
            </div>
        </div>

        <section id="overview"></section>
        <section id="components"></section>
        <section id="metrics"></section>
        <section id="alerts"></section>
    </div>

    <footer class="footer mt-auto py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">AI Trading Agent Health Dashboard</span>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="/static/js/dashboard.js"></script>
    <script src="/static/js/components.js"></script>
    <script src="/static/js/metrics.js"></script>
    <script src="/static/js/alerts.js"></script>
</body>
</html>"""


def _get_components_html() -> str:
    """Get default components.html content."""
    return """<!-- Components view template -->
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5>Component Status</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover" id="components-table">
                        <thead>
                            <tr>
                                <th>Component ID</th>
                                <th>Description</th>
                                <th>Status</th>
                                <th>Last Heartbeat</th>
                                <th>Uptime</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="components-tbody">
                            <tr>
                                <td colspan="6" class="text-center">Loading components...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>"""


def _get_metrics_html() -> str:
    """Get default metrics.html content."""
    return """<!-- Metrics view template -->
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5>Performance Metrics</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <div class="form-group mb-3">
                            <label for="component-select">Component:</label>
                            <select class="form-control" id="component-select">
                                <option value="all">All Components</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group mb-3">
                            <label for="metric-select">Metric:</label>
                            <select class="form-control" id="metric-select">
                                <option value="all">All Metrics</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group mb-3">
                            <label for="time-range">Time Range:</label>
                            <select class="form-control" id="time-range">
                                <option value="1h">Last Hour</option>
                                <option value="6h">Last 6 Hours</option>
                                <option value="24h" selected>Last 24 Hours</option>
                                <option value="7d">Last 7 Days</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div id="metrics-charts" class="mt-4">
                    <div class="text-center">
                        <p>Select a component and metric to view charts</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>"""


def _get_alerts_html() -> str:
    """Get default alerts.html content."""
    return """<!-- Alerts view template -->
<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5>Active Alerts</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover" id="alerts-table">
                        <thead>
                            <tr>
                                <th>Alert ID</th>
                                <th>Component</th>
                                <th>Severity</th>
                                <th>Message</th>
                                <th>Time</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="alerts-tbody">
                            <tr>
                                <td colspan="6" class="text-center">No active alerts</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>"""
