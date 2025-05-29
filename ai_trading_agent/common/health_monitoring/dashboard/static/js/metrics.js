/**
 * Metrics JavaScript
 * 
 * Handles performance metrics visualization and charts
 */

// Store chart references to update them later
const chartInstances = {};

// Initialize metrics module
document.addEventListener('DOMContentLoaded', () => {
    console.log('Metrics module initializing...');
    
    // Set up event listeners for metric controls
    setupMetricControls();
});

/**
 * Set up event listeners for metric control elements
 */
function setupMetricControls() {
    // Component select change
    const componentSelect = document.getElementById('component-select');
    if (componentSelect) {
        componentSelect.addEventListener('change', () => {
            updateMetricSelect();
            updateMetricsCharts();
        });
    }
    
    // Metric select change
    const metricSelect = document.getElementById('metric-select');
    if (metricSelect) {
        metricSelect.addEventListener('change', () => {
            updateMetricsCharts();
        });
    }
    
    // Time range select change
    const timeRangeSelect = document.getElementById('time-range');
    if (timeRangeSelect) {
        timeRangeSelect.addEventListener('change', () => {
            updateMetricsCharts();
        });
    }
}

/**
 * Update metrics charts with the latest data
 * 
 * @param {Object} metricsData - Metrics data from API
 */
function updateMetricsCharts(metricsData) {
    if (!metricsData) {
        console.log('No metrics data provided, skipping chart updates');
        return;
    }
    
    console.log('Updating metrics charts...');
    
    // Get selected values
    const componentSelect = document.getElementById('component-select');
    const metricSelect = document.getElementById('metric-select');
    const timeRangeSelect = document.getElementById('time-range');
    
    if (!componentSelect || !metricSelect || !timeRangeSelect) {
        console.error('Metrics control elements not found');
        return;
    }
    
    const selectedComponent = componentSelect.value;
    const selectedMetric = metricSelect.value;
    const selectedTimeRange = timeRangeSelect.value;
    
    // Populate component select if empty
    if (componentSelect.options.length <= 1) {
        populateComponentSelect(metricsData);
    }
    
    // Update available metrics based on selected component
    updateMetricSelect(metricsData, selectedComponent);
    
    // Get metrics chart container
    const chartsContainer = document.getElementById('metrics-charts');
    if (!chartsContainer) {
        console.error('Metrics charts container not found');
        return;
    }
    
    // Clear existing charts
    chartsContainer.innerHTML = '';
    
    // Filter metrics data based on selections
    let metricsToDisplay = [];
    
    if (selectedComponent === 'all') {
        // Show all components' metrics
        if (selectedMetric === 'all') {
            // Show all metrics for all components
            Object.entries(metricsData).forEach(([componentId, componentMetrics]) => {
                Object.entries(componentMetrics).forEach(([metricName, metricData]) => {
                    metricsToDisplay.push({
                        componentId,
                        metricName,
                        metricData,
                        displayName: `${componentId} - ${metricName}`
                    });
                });
            });
        } else {
            // Show specific metric for all components
            Object.entries(metricsData).forEach(([componentId, componentMetrics]) => {
                if (componentMetrics[selectedMetric]) {
                    metricsToDisplay.push({
                        componentId,
                        metricName: selectedMetric,
                        metricData: componentMetrics[selectedMetric],
                        displayName: `${componentId} - ${selectedMetric}`
                    });
                }
            });
        }
    } else {
        // Show specific component's metrics
        const componentMetrics = metricsData[selectedComponent];
        if (componentMetrics) {
            if (selectedMetric === 'all') {
                // Show all metrics for selected component
                Object.entries(componentMetrics).forEach(([metricName, metricData]) => {
                    metricsToDisplay.push({
                        componentId: selectedComponent,
                        metricName,
                        metricData,
                        displayName: metricName
                    });
                });
            } else {
                // Show specific metric for selected component
                if (componentMetrics[selectedMetric]) {
                    metricsToDisplay.push({
                        componentId: selectedComponent,
                        metricName: selectedMetric,
                        metricData: componentMetrics[selectedMetric],
                        displayName: selectedMetric
                    });
                }
            }
        }
    }
    
    // Filter metrics by time range
    metricsToDisplay = filterMetricsByTimeRange(metricsToDisplay, selectedTimeRange);
    
    // Check if we have any metrics to display
    if (metricsToDisplay.length === 0) {
        chartsContainer.innerHTML = '<div class="text-center"><p>No metrics data available for the selected filters</p></div>';
        return;
    }
    
    // Create charts
    metricsToDisplay.forEach((metric, index) => {
        createMetricChart(chartsContainer, metric, index);
    });
}

/**
 * Populate the component select dropdown with available components
 * 
 * @param {Object} metricsData - Metrics data from API
 */
function populateComponentSelect(metricsData) {
    const componentSelect = document.getElementById('component-select');
    if (!componentSelect) {
        console.error('Component select element not found');
        return;
    }
    
    // Keep the "All Components" option
    componentSelect.innerHTML = '<option value="all">All Components</option>';
    
    // Add options for each component
    Object.keys(metricsData).forEach(componentId => {
        const option = document.createElement('option');
        option.value = componentId;
        option.textContent = componentId;
        componentSelect.appendChild(option);
    });
}

/**
 * Update the metric select dropdown based on selected component
 * 
 * @param {Object} metricsData - Metrics data from API
 * @param {string} selectedComponent - Selected component ID
 */
function updateMetricSelect(metricsData, selectedComponent) {
    if (!metricsData) {
        return;
    }
    
    const metricSelect = document.getElementById('metric-select');
    if (!metricSelect) {
        console.error('Metric select element not found');
        return;
    }
    
    // Keep the "All Metrics" option
    metricSelect.innerHTML = '<option value="all">All Metrics</option>';
    
    // Get unique metric names based on selected component
    const metricNames = new Set();
    
    if (selectedComponent === 'all') {
        // Add all metric names from all components
        Object.values(metricsData).forEach(componentMetrics => {
            Object.keys(componentMetrics).forEach(metricName => {
                metricNames.add(metricName);
            });
        });
    } else {
        // Add metric names for selected component only
        const componentMetrics = metricsData[selectedComponent];
        if (componentMetrics) {
            Object.keys(componentMetrics).forEach(metricName => {
                metricNames.add(metricName);
            });
        }
    }
    
    // Add options for each metric
    Array.from(metricNames).sort().forEach(metricName => {
        const option = document.createElement('option');
        option.value = metricName;
        option.textContent = metricName;
        metricSelect.appendChild(option);
    });
}

/**
 * Filter metrics data by the selected time range
 * 
 * @param {Array} metricsToDisplay - Array of metrics to display
 * @param {string} timeRange - Selected time range (1h, 6h, 24h, 7d)
 * @returns {Array} Filtered metrics data
 */
function filterMetricsByTimeRange(metricsToDisplay, timeRange) {
    const now = Math.floor(Date.now() / 1000);
    let cutoffTime;
    
    switch (timeRange) {
        case '1h':
            cutoffTime = now - 3600;
            break;
        case '6h':
            cutoffTime = now - 21600;
            break;
        case '24h':
            cutoffTime = now - 86400;
            break;
        case '7d':
            cutoffTime = now - 604800;
            break;
        default:
            cutoffTime = now - 86400;  // Default to 24h
            break;
    }
    
    return metricsToDisplay.map(metric => {
        // Clone the metric object
        const filteredMetric = { ...metric };
        
        // Filter data points to the selected time range
        if (metric.metricData && metric.metricData.history) {
            filteredMetric.metricData = {
                ...metric.metricData,
                history: metric.metricData.history.filter(point => point.timestamp >= cutoffTime)
            };
        }
        
        return filteredMetric;
    });
}

/**
 * Create a chart for a metric
 * 
 * @param {HTMLElement} container - Container element for the chart
 * @param {Object} metric - Metric data to display
 * @param {number} index - Index of the metric in the display list
 */
function createMetricChart(container, metric, index) {
    // Create card for this chart
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card metric-card mb-4';
    
    // Create card header
    const cardHeader = document.createElement('div');
    cardHeader.className = 'card-header';
    cardHeader.innerHTML = `<h6>${metric.displayName}</h6>`;
    
    // Create card body
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // Create canvas for the chart
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${metric.componentId}-${metric.metricName}-${index}`;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    
    // Add elements to the container
    cardBody.appendChild(canvas);
    cardDiv.appendChild(cardHeader);
    cardDiv.appendChild(cardBody);
    container.appendChild(cardDiv);
    
    // Create the chart
    createChart(canvas.id, metric);
}

/**
 * Create a Chart.js chart for a metric
 * 
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} metric - Metric data
 */
function createChart(canvasId, metric) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element not found: ${canvasId}`);
        return;
    }
    
    // Get metric history data
    const history = metric.metricData.history || [];
    
    // Prepare data for chart
    const labels = [];
    const values = [];
    const thresholdValues = [];
    
    // Get threshold if available
    let threshold = null;
    if (metric.metricData.threshold) {
        threshold = metric.metricData.threshold.value;
    }
    
    // Process history data
    history.forEach(point => {
        // Format timestamp as time
        const date = new Date(point.timestamp * 1000);
        const timeString = date.toLocaleTimeString();
        
        labels.push(timeString);
        values.push(point.value);
        
        // Add threshold value for each point if available
        if (threshold !== null) {
            thresholdValues.push(threshold);
        }
    });
    
    // Create chart config
    const chartConfig = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: metric.metricName,
                    data: values,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    };
    
    // Add threshold line if available
    if (threshold !== null) {
        chartConfig.data.datasets.push({
            label: 'Threshold',
            data: thresholdValues,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        });
    }
    
    // Check if chart already exists
    if (chartInstances[canvasId]) {
        // Update existing chart
        chartInstances[canvasId].data = chartConfig.data;
        chartInstances[canvasId].update();
    } else {
        // Create new chart
        chartInstances[canvasId] = new Chart(canvas, chartConfig);
    }
}
