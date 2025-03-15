# Log Dashboard Usage Instructions

This document provides instructions for running and using the log analytics dashboard.

## Prerequisites

Before running the dashboard, ensure you have the following dependencies installed:

```bash
pip install dash dash-bootstrap-components pandas plotly structlog psutil
```

## Generating Test Logs

We've created a script to generate test log data for the dashboard. To generate test logs:

1. Run the test log generator:
   ```bash
   python tests/generate_test_logs.py
   ```

This will create log entries in the `logs/crypto_trading.log` file with various log levels, components, and messages.

## Running the Dashboard

1. Use the provided helper script:
   ```bash
   python run_dashboard.py
   ```

2. The dashboard will start and be available at:
   ```
   http://127.0.0.1:8050/
   ```

3. Open this URL in your web browser to access the dashboard.

## Dashboard Features

The log dashboard offers the following features:

1. **Log Overview** - Shows log distribution by levels, components, and time

2. **Log Query** - Allows you to search and filter logs based on:
   - Time range
   - Component
   - Log level
   - Custom search terms

3. **Health Monitor** - Displays system health metrics including:
   - CPU usage
   - Memory usage
   - Disk usage
   - Error rates

4. **Log Replay** - Allows you to replay logs in sequence for debugging and analysis

5. **Anomaly Detection** - Automatically identifies unusual patterns in your logs:
   - Detects abnormal frequencies of log levels (e.g., sudden spike in errors)
   - Identifies unusual component activity
   - Spots temporal anomalies (activity at unusual times)
   - Detects error bursts
   - Provides visualizations of anomalies over time

6. **Advanced Analytics** - Provides sophisticated visualizations and custom date range analysis:
   - Custom date range selection for precise time period analysis
   - Multiple visualization types:
     - Time Heatmaps: View log activity patterns across time periods
     - Log Patterns: Analyze how log patterns evolve over time
     - Field Correlations: Discover relationships between different log fields
     - Volume Comparison: Compare log volumes across different components or levels
     - Error Distribution: Analyze error types across different components

## Using the Anomaly Detection Feature

The Anomaly Detection tab provides advanced analytics to automatically identify potential issues:

1. **Analysis Window** - Select the time range for analysis:
   - Last hour (for recent issues)
   - Last 12 hours
   - Last day
   - Last week
   - All logs

2. **Sensitivity** - Adjust the detection sensitivity:
   - Low: Only detect major anomalies
   - Medium: Balanced detection
   - High: Detect even minor anomalies

3. Click "Detect Anomalies" to analyze your logs.

4. Results are displayed in a table showing:
   - Anomaly type
   - Severity score
   - Detailed explanation

5. A scatter plot visualization shows anomalies by time and severity.

## Using the Advanced Analytics Feature

The Advanced Analytics tab provides sophisticated visualization tools for deeper log analysis:

1. **Custom Date Range** - Select precise start and end dates for your analysis:
   - Start Date: Choose the beginning of your analysis period
   - End Date: Choose the end of your analysis period

2. **Visualization Type** - Select from multiple visualization options:
   - Time Heatmap: Shows log activity patterns across different time periods
   - Log Patterns: Displays how log patterns evolve over time
   - Field Correlations: Reveals relationships between different log fields
   - Volume Comparison: Compares log volumes across different components or levels
   - Error Distribution: Analyzes error types across different components

3. **Field to Analyze** - Choose which log field to focus on:
   - Log Level: Analyze by severity level (info, warning, error, etc.)
   - Component: Analyze by system component
   - Error Type: Focus on specific error categories
   - User ID: Analyze logs by user
   - Request ID: Analyze logs by request

4. Click "Generate Visualization" to create your custom visualization.

5. **Export Data** - Save your analysis results in various formats:
   - CSV: For spreadsheet analysis
   - JSON: For programmatic processing
   - Excel: For detailed spreadsheet analysis with multiple sheets
   - PNG: For including visualizations in reports

6. Use these visualizations to:
   - Identify usage patterns and peak activity times
   - Discover correlations between errors and specific components
   - Track system behavior changes over time
   - Optimize system performance based on usage patterns
   - Troubleshoot complex issues by visualizing relationships

## Troubleshooting

If you encounter issues running the dashboard:

1. **Module Not Found Errors**:
   - Ensure you're running the dashboard with the correct Python path using `run_dashboard.py`
   
2. **No Data Displayed**:
   - Make sure you've generated test logs using `tests/generate_test_logs.py`
   - Check that the logs directory and log file exist: `logs/crypto_trading.log`

3. **Dashboard Not Loading**:
   - Verify that all required dependencies are installed
   - Check the terminal output for any error messages

## Next Steps

Future dashboard enhancements could include:

1. User-customizable dashboards with saved views
2. Real-time log streaming
3. Advanced anomaly detection with machine learning
4. Integration with alerting systems
5. ~~Export capabilities for log analysis results~~
6. Scheduled reports and automated analysis
7. Integration with external monitoring systems
8. Mobile-friendly dashboard views
9. Collaborative sharing and annotation of insights
