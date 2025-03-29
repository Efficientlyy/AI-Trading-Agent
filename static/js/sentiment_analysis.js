/**
 * Sentiment Analysis Dashboard
 * 
 * This module provides functionality for the Sentiment Analysis Dashboard component
 * including sentiment spike detection, sentiment trend visualization, and source filtering.
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize the sentiment analysis dashboard
    initializeSentimentDashboard();
});

function initializeSentimentDashboard() {
    console.log('Initializing sentiment analysis dashboard');

    // Get DOM elements
    const symbolSelector = document.getElementById('sentiment-symbol');
    const timeframeSelector = document.getElementById('sentiment-timeframe');
    const sourceSelector = document.getElementById('sentiment-source');
    const thresholdSlider = document.getElementById('sentiment-threshold');
    const alertsToggle = document.getElementById('sentiment-alerts');
    const realtimeToggle = document.getElementById('sentiment-realtime');

    // Set up event listeners for controls
    if (symbolSelector) {
        symbolSelector.addEventListener('change', updateSentimentData);
    }

    if (timeframeSelector) {
        timeframeSelector.addEventListener('change', updateSentimentData);
    }

    if (sourceSelector) {
        sourceSelector.addEventListener('change', updateSentimentData);
    }

    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', updateThreshold);
    }

    if (alertsToggle) {
        alertsToggle.addEventListener('change', toggleAlerts);
    }

    if (realtimeToggle) {
        realtimeToggle.addEventListener('change', toggleRealtime);
    }

    // Initial data load
    loadMockSentimentData();
}

function updateSentimentData() {
    console.log('Updating sentiment data');
    // In a real implementation, this would fetch data from an API
    // For now, we'll just reload the mock data
    loadMockSentimentData();
}

function updateThreshold(event) {
    const value = event.target.value;
    console.log(`Updating sentiment threshold to ${value}`);

    // Update threshold value display if it exists
    const thresholdValue = document.getElementById('threshold-value');
    if (thresholdValue) {
        thresholdValue.textContent = value;
    }

    // In a real implementation, this would update the visualization
}

function toggleAlerts(event) {
    const enabled = event.target.checked;
    console.log(`Sentiment alerts ${enabled ? 'enabled' : 'disabled'}`);

    // In a real implementation, this would enable/disable alerts
}

function toggleRealtime(event) {
    const enabled = event.target.checked;
    console.log(`Realtime updates ${enabled ? 'enabled' : 'disabled'}`);

    // In a real implementation, this would start/stop realtime updates
}

function loadMockSentimentData() {
    console.log('Loading mock sentiment data');

    // In a real implementation, this would fetch data from an API
    // For now, we'll just simulate a delay

    // Show loading indicators
    const chartContainers = document.querySelectorAll('.sentiment-chart');
    chartContainers.forEach(container => {
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.textContent = 'Loading...';
        }
    });

    // Simulate API delay
    setTimeout(() => {
        // Hide loading indicators
        chartContainers.forEach(container => {
            const overlay = container.querySelector('.chart-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        });

        // Render charts with mock data
        renderSentimentCharts();
    }, 1000);
}

function renderSentimentCharts() {
    console.log('Rendering sentiment charts');

    // In a real implementation, this would render charts with actual data
    // For now, we'll just log a message

    // Check if Plotly is available
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not available');
        return;
    }

    // Render sentiment trend chart
    renderSentimentTrendChart();

    // Render sentiment distribution chart
    renderSentimentDistributionChart();

    // Render sentiment source breakdown chart
    renderSentimentSourceChart();

    // Render sentiment spike table
    renderSentimentSpikeTable();
}

function renderSentimentTrendChart() {
    const element = document.getElementById('sentiment-trend-chart');
    if (!element) return;

    // Generate mock data
    const dates = [];
    const sentimentValues = [];
    const priceValues = [];

    const now = new Date();
    for (let i = 30; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        dates.push(date);

        // Generate sentiment values with some pattern
        const baseValue = 0.5 + 0.3 * Math.sin(i / 5);
        const randomNoise = (Math.random() - 0.5) * 0.2;
        sentimentValues.push(baseValue + randomNoise);

        // Generate price values with correlation to sentiment
        const basePrice = 50000 + 10000 * Math.sin(i / 7);
        const priceNoise = (Math.random() - 0.5) * 1000;
        const sentimentEffect = (baseValue - 0.5) * 5000;
        priceValues.push(basePrice + priceNoise + sentimentEffect);
    }

    // Create traces
    const sentimentTrace = {
        x: dates,
        y: sentimentValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Sentiment',
        line: {
            color: 'rgba(66, 153, 225, 0.8)',
            width: 2
        },
        yaxis: 'y'
    };

    const priceTrace = {
        x: dates,
        y: priceValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Price',
        line: {
            color: 'rgba(72, 187, 120, 0.8)',
            width: 2
        },
        yaxis: 'y2'
    };

    // Layout configuration
    const layout = {
        title: 'Sentiment Trend vs Price',
        xaxis: {
            title: 'Date',
            showgrid: false,
            type: 'date'
        },
        yaxis: {
            title: 'Sentiment Score',
            range: [0, 1],
            tickformat: '.1f',
            side: 'left'
        },
        yaxis2: {
            title: 'Price (USD)',
            overlaying: 'y',
            side: 'right'
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        margin: { l: 50, r: 50, t: 40, b: 80 },
        paper_bgcolor: 'var(--card-bg)',
        plot_bgcolor: 'var(--card-bg)',
        font: {
            color: 'var(--text)',
            size: 10
        }
    };

    // Render with Plotly
    Plotly.newPlot(element, [sentimentTrace, priceTrace], layout, { responsive: true });
}

function renderSentimentDistributionChart() {
    const element = document.getElementById('sentiment-distribution-chart');
    if (!element) return;

    // Generate mock data
    const values = [];
    for (let i = 0; i < 100; i++) {
        // Generate a distribution skewed toward positive
        const baseValue = 0.6 + 0.2 * (Math.random() + Math.random() - 1);
        values.push(Math.max(0, Math.min(1, baseValue)));
    }

    // Create trace
    const trace = {
        x: values,
        type: 'histogram',
        marker: {
            color: 'rgba(66, 153, 225, 0.7)',
            line: {
                color: 'rgba(66, 153, 225, 1)',
                width: 1
            }
        },
        nbinsx: 20
    };

    // Layout configuration
    const layout = {
        title: 'Sentiment Distribution',
        xaxis: {
            title: 'Sentiment Score',
            range: [0, 1],
            tickformat: '.1f'
        },
        yaxis: {
            title: 'Frequency'
        },
        margin: { l: 50, r: 20, t: 40, b: 50 },
        paper_bgcolor: 'var(--card-bg)',
        plot_bgcolor: 'var(--card-bg)',
        font: {
            color: 'var(--text)',
            size: 10
        }
    };

    // Render with Plotly
    Plotly.newPlot(element, [trace], layout, { responsive: true });
}

function renderSentimentSourceChart() {
    const element = document.getElementById('sentiment-source-chart');
    if (!element) return;

    // Mock data for sources
    const sources = ['Twitter', 'Reddit', 'News', 'Blogs', 'Forums'];
    const values = [35, 25, 20, 15, 5];
    const colors = [
        'rgba(66, 153, 225, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(72, 187, 120, 0.8)',
        'rgba(246, 173, 85, 0.8)',
        'rgba(160, 174, 192, 0.8)'
    ];

    // Create trace
    const trace = {
        labels: sources,
        values: values,
        type: 'pie',
        marker: {
            colors: colors
        },
        textinfo: 'percent',
        hoverinfo: 'label+percent',
        hole: 0.4
    };

    // Layout configuration
    const layout = {
        title: 'Sentiment by Source',
        margin: { l: 20, r: 20, t: 40, b: 20 },
        paper_bgcolor: 'var(--card-bg)',
        plot_bgcolor: 'var(--card-bg)',
        font: {
            color: 'var(--text)',
            size: 10
        }
    };

    // Render with Plotly
    Plotly.newPlot(element, [trace], layout, { responsive: true });
}

function renderSentimentSpikeTable() {
    const tableBody = document.querySelector('#sentiment-spike-table tbody');
    if (!tableBody) return;

    // Clear existing rows
    tableBody.innerHTML = '';

    // Mock data for sentiment spikes
    const spikes = [
        {
            timestamp: '2025-03-29 14:15:22',
            source: 'Twitter',
            magnitude: 0.85,
            impact: 'High',
            keywords: 'regulation, SEC, approval'
        },
        {
            timestamp: '2025-03-29 12:30:45',
            source: 'News',
            magnitude: 0.78,
            impact: 'Medium',
            keywords: 'partnership, enterprise, adoption'
        },
        {
            timestamp: '2025-03-29 10:05:18',
            source: 'Reddit',
            magnitude: 0.92,
            impact: 'High',
            keywords: 'hack, security, vulnerability'
        },
        {
            timestamp: '2025-03-29 08:45:33',
            source: 'Blogs',
            magnitude: 0.65,
            impact: 'Low',
            keywords: 'technology, upgrade, roadmap'
        },
        {
            timestamp: '2025-03-28 22:12:07',
            source: 'Twitter',
            magnitude: 0.88,
            impact: 'High',
            keywords: 'whale, transaction, accumulation'
        }
    ];

    // Add rows to table
    spikes.forEach(spike => {
        const row = document.createElement('tr');

        // Timestamp cell
        const timestampCell = document.createElement('td');
        timestampCell.textContent = spike.timestamp;
        row.appendChild(timestampCell);

        // Source cell
        const sourceCell = document.createElement('td');
        sourceCell.textContent = spike.source;
        row.appendChild(sourceCell);

        // Magnitude cell
        const magnitudeCell = document.createElement('td');
        const magnitudeValue = document.createElement('div');
        magnitudeValue.className = 'progress';
        magnitudeValue.style.width = `${spike.magnitude * 100}%`;
        magnitudeValue.style.backgroundColor = getColorForMagnitude(spike.magnitude);
        magnitudeCell.appendChild(magnitudeValue);
        row.appendChild(magnitudeCell);

        // Impact cell
        const impactCell = document.createElement('td');
        const impactBadge = document.createElement('span');
        impactBadge.className = `badge ${getClassForImpact(spike.impact)}`;
        impactBadge.textContent = spike.impact;
        impactCell.appendChild(impactBadge);
        row.appendChild(impactCell);

        // Keywords cell
        const keywordsCell = document.createElement('td');
        keywordsCell.textContent = spike.keywords;
        row.appendChild(keywordsCell);

        // Actions cell
        const actionsCell = document.createElement('td');
        const detailsButton = document.createElement('button');
        detailsButton.className = 'btn btn-sm btn-outline-secondary';
        detailsButton.textContent = 'Details';
        detailsButton.addEventListener('click', () => showSpikeDetails(spike));
        actionsCell.appendChild(detailsButton);
        row.appendChild(actionsCell);

        tableBody.appendChild(row);
    });
}

function getColorForMagnitude(magnitude) {
    if (magnitude >= 0.8) return 'var(--danger)';
    if (magnitude >= 0.6) return 'var(--warning)';
    return 'var(--success)';
}

function getClassForImpact(impact) {
    switch (impact) {
        case 'High': return 'bg-danger';
        case 'Medium': return 'bg-warning';
        case 'Low': return 'bg-success';
        default: return 'bg-secondary';
    }
}

function showSpikeDetails(spike) {
    console.log('Showing details for spike:', spike);
    // In a real implementation, this would show a modal with details
    alert(`Sentiment Spike Details:\nTime: ${spike.timestamp}\nSource: ${spike.source}\nMagnitude: ${spike.magnitude}\nImpact: ${spike.impact}\nKeywords: ${spike.keywords}`);
}
