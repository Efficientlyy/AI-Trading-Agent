/**
 * Real-time Market Anomaly Detection
 * 
 * This module provides visualization and detection of unusual market activity,
 * historical pattern analysis, anomaly classification, and severity scoring.
 */

class AnomalyDetection {
    constructor(options = {}) {
        this.options = Object.assign({
            realtimeChartElementId: 'realtime-anomaly-chart',
            patternLibraryElementId: 'pattern-library-chart',
            classificationElementId: 'anomaly-classification-chart',
            severityElementId: 'anomaly-severity-chart',
            updateInterval: 10000, // milliseconds
            historyLength: 100,    // data points to show
            anomalyThreshold: 2.5, // standard deviations
            colorScale: [
                [0, 'rgba(59, 130, 246, 0.5)'],   // Normal (blue)
                [0.33, 'rgba(234, 179, 8, 0.5)'],  // Warning (yellow)
                [0.66, 'rgba(239, 68, 68, 0.5)']   // Alert (red)
            ],
            defaultSymbol: 'BTC-USD',
            defaultMetric: 'price'
        }, options);

        // State management
        this.symbol = document.getElementById('anomaly-symbol')?.value || this.options.defaultSymbol;
        this.metric = document.getElementById('anomaly-metric')?.value || this.options.defaultMetric;
        this.sensitivity = document.getElementById('anomaly-sensitivity')?.value || this.options.anomalyThreshold;
        this.timeWindow = document.getElementById('time-window')?.value || '1h';
        this.alertsEnabled = document.getElementById('alerts-enabled')?.checked || true;

        // Data containers
        this.priceData = [];
        this.volumeData = [];
        this.volatilityData = [];
        this.anomalyHistory = [];
        this.patternLibrary = [];
        this.recentAnomalies = [];
        this.severityHistory = [];

        // Chart objects
        this.realtimeChart = null;
        this.patternLibraryChart = null;
        this.classificationChart = null;
        this.severityChart = null;

        // Flags
        this.isInitialized = false;
        this.isRealtime = true;
        this.updateTimer = null;

        // Initialize
        this.initialize();
    }

    initialize() {
        // Fetch initial data and set up charts
        this.fetchData()
            .then(() => {
                this.detectAnomalies();
                this.initializeCharts();
                this.setupEventListeners();
                this.startRealtimeUpdates();
                this.isInitialized = true;

                // Initialize feather icons if available
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            })
            .catch(error => {
                console.error('Error initializing Anomaly Detection:', error);
                this.showError('Failed to initialize anomaly detection');
            });
    }

    fetchData() {
        // In a real implementation, this would fetch from an API
        return Promise.all([
            this.fetchPriceData(),
            this.fetchVolumeData(),
            this.fetchVolatilityData(),
            this.fetchPatternLibrary(),
            this.fetchAnomalyHistory()
        ]);
    }

    fetchPriceData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock price data
                const priceData = this.generateMockPriceData();
                this.priceData = priceData;
                resolve(priceData);
            }, 300);
        });
    }

    fetchVolumeData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock volume data
                const volumeData = this.generateMockVolumeData();
                this.volumeData = volumeData;
                resolve(volumeData);
            }, 250);
        });
    }

    fetchVolatilityData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock volatility data
                const volatilityData = this.generateMockVolatilityData();
                this.volatilityData = volatilityData;
                resolve(volatilityData);
            }, 200);
        });
    }

    fetchPatternLibrary() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock pattern library
                const patternLibrary = this.generateMockPatternLibrary();
                this.patternLibrary = patternLibrary;
                resolve(patternLibrary);
            }, 350);
        });
    }

    fetchAnomalyHistory() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock anomaly history
                const anomalyHistory = this.generateMockAnomalyHistory();
                this.anomalyHistory = anomalyHistory;
                resolve(anomalyHistory);
            }, 300);
        });
    }

    detectAnomalies() {
        // Detect anomalies in the data
        this.recentAnomalies = [];
        this.severityHistory = [];

        // Get the data based on the selected metric
        let data;
        if (this.metric === 'price') {
            data = this.priceData;
        } else if (this.metric === 'volume') {
            data = this.volumeData;
        } else if (this.metric === 'volatility') {
            data = this.volatilityData;
        } else {
            // Use price data by default
            data = this.priceData;
        }

        // Calculate moving average and standard deviation
        const windowSize = 10;
        const values = data.map(d => d.value);
        
        for (let i = windowSize; i < values.length; i++) {
            // Get the window of data
            const window = values.slice(i - windowSize, i);
            
            // Calculate mean and standard deviation
            const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
            const stdDev = Math.sqrt(
                window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length
            );
            
            // Get the current value
            const currentValue = values[i];
            
            // Calculate z-score
            const zScore = stdDev > 0 ? Math.abs((currentValue - mean) / stdDev) : 0;
            
            // Check if anomaly
            const isAnomaly = zScore > this.sensitivity;
            const severity = this.calculateSeverity(zScore);
            
            // Add to anomaly list if detected
            if (isAnomaly) {
                this.recentAnomalies.push({
                    timestamp: data[i].timestamp,
                    metric: this.metric,
                    value: currentValue,
                    expectedValue: mean,
                    zScore: zScore,
                    severity: severity,
                    type: this.classifyAnomaly(zScore, currentValue > mean)
                });
            }
            
            // Add to severity history
            this.severityHistory.push({
                timestamp: data[i].timestamp,
                zScore: zScore,
                severity: severity,
                isAnomaly: isAnomaly
            });
        }
        
        // Match patterns from the library
        this.matchPatterns();
    }

    calculateSeverity(zScore) {
        // Calculate severity based on z-score
        if (zScore < this.sensitivity) {
            return 0; // Normal
        } else if (zScore < this.sensitivity * 1.5) {
            return 1; // Warning
        } else if (zScore < this.sensitivity * 2) {
            return 2; // Alert
        } else {
            return 3; // Critical
        }
    }

    classifyAnomaly(zScore, isPositive) {
        // Classify anomaly based on z-score and direction
        if (zScore < this.sensitivity * 1.5) {
            return isPositive ? 'price_spike' : 'price_drop';
        } else if (zScore < this.sensitivity * 2) {
            return isPositive ? 'volatility_surge' : 'liquidity_gap';
        } else {
            return isPositive ? 'flash_crash' : 'market_manipulation';
        }
    }

    matchPatterns() {
        // Match detected anomalies with pattern library
        this.recentAnomalies.forEach(anomaly => {
            // Simple pattern matching logic
            const matchedPatterns = this.patternLibrary.filter(pattern => {
                return pattern.type === anomaly.type && 
                       Math.abs(pattern.zScore - anomaly.zScore) < 0.5;
            });
            
            if (matchedPatterns.length > 0) {
                // Add the most similar pattern
                anomaly.matchedPattern = matchedPatterns[0].id;
                anomaly.patternConfidence = 0.7 + (Math.random() * 0.3);
            }
        });
    }

    initializeCharts() {
        this.renderRealtimeChart();
        this.renderPatternLibraryChart();
        this.renderClassificationChart();
        this.renderSeverityChart();
        this.updateAnomalyTable();
    }

    setupEventListeners() {
        // Symbol selector change
        const symbolSelector = document.getElementById('anomaly-symbol');
        if (symbolSelector) {
            symbolSelector.addEventListener('change', () => {
                this.symbol = symbolSelector.value;
                this.refreshData();
            });
        }

        // Metric selector change
        const metricSelector = document.getElementById('anomaly-metric');
        if (metricSelector) {
            metricSelector.addEventListener('change', () => {
                this.metric = metricSelector.value;
                this.detectAnomalies();
                this.renderRealtimeChart();
                this.renderSeverityChart();
                this.updateAnomalyTable();
            });
        }

        // Sensitivity slider change
        const sensitivitySlider = document.getElementById('anomaly-sensitivity');
        if (sensitivitySlider) {
            sensitivitySlider.addEventListener('input', () => {
                this.sensitivity = parseFloat(sensitivitySlider.value);
                this.detectAnomalies();
                this.renderRealtimeChart();
                this.renderSeverityChart();
                this.updateAnomalyTable();
                
                // Update the sensitivity display
                const sensitivityDisplay = document.getElementById('sensitivity-value');
                if (sensitivityDisplay) {
                    sensitivityDisplay.textContent = this.sensitivity.toFixed(1);
                }
            });
        }

        // Time window change
        const timeWindowSelector = document.getElementById('time-window');
        if (timeWindowSelector) {
            timeWindowSelector.addEventListener('change', () => {
                this.timeWindow = timeWindowSelector.value;
                this.refreshData();
            });
        }

        // Alerts toggle
        const alertsToggle = document.getElementById('alerts-enabled');
        if (alertsToggle) {
            alertsToggle.addEventListener('change', () => {
                this.alertsEnabled = alertsToggle.checked;
            });
        }

        // Realtime toggle
        const realtimeToggle = document.getElementById('realtime-toggle');
        if (realtimeToggle) {
            realtimeToggle.addEventListener('change', () => {
                this.isRealtime = realtimeToggle.checked;
                if (this.isRealtime) {
                    this.startRealtimeUpdates();
                } else {
                    this.stopRealtimeUpdates();
                }
            });
        }

        // Refresh button
        const refreshButton = document.getElementById('refresh-anomaly');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshData();
            });
        }

        // Export data button
        const exportButton = document.getElementById('export-anomaly-data');
        if (exportButton) {
            exportButton.addEventListener('click', () => {
                this.exportData();
            });
        }

        // Pattern library items
        document.querySelectorAll('.pattern-library-item').forEach(item => {
            item.addEventListener('click', () => {
                const patternId = item.getAttribute('data-pattern-id');
                this.selectPattern(patternId);
            });
        });
    }

    startRealtimeUpdates() {
        // Clear any existing timers
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }

        // Set up timer for real-time updates
        this.updateTimer = setInterval(() => {
            this.updateRealtimeData();
        }, this.options.updateInterval);
    }

    stopRealtimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    updateRealtimeData() {
        // Update price data
        this.addNewDataPoint();
        
        // Detect anomalies
        this.detectAnomalies();
        
        // Update visualizations
        this.renderRealtimeChart();
        this.renderSeverityChart();
        this.updateAnomalyTable();
        
        // Show notification for new anomalies
        if (this.alertsEnabled && this.recentAnomalies.length > 0) {
            const latestAnomaly = this.recentAnomalies[this.recentAnomalies.length - 1];
            if (latestAnomaly.timestamp > Date.now() - 60000) { // Within the last minute
                this.showNotification(latestAnomaly);
            }
        }
    }

    refreshData() {
        // Show loading state
        this.showLoading();
        
        // Fetch new data
        this.fetchData()
            .then(() => {
                this.detectAnomalies();
                this.renderRealtimeChart();
                this.renderPatternLibraryChart();
                this.renderClassificationChart();
                this.renderSeverityChart();
                this.updateAnomalyTable();
                
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                this.showError('Failed to refresh anomaly detection data');
            });
    }

    renderRealtimeChart() {
        const element = document.getElementById(this.options.realtimeChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Get data based on selected metric
        let data;
        if (this.metric === 'price') {
            data = this.priceData;
        } else if (this.metric === 'volume') {
            data = this.volumeData;
        } else if (this.metric === 'volatility') {
            data = this.volatilityData;
        } else {
            data = this.priceData;
        }

        // Extract timestamps and values
        const timestamps = data.map(d => d.timestamp);
        const values = data.map(d => d.value);
        
        // Create the main data trace
        const mainTrace = {
            x: timestamps,
            y: values,
            type: 'scatter',
            mode: 'lines',
            name: this.capitalizeFirst(this.metric),
            line: {
                color: 'rgba(59, 130, 246, 0.8)',
                width: 2
            },
            hovertemplate: '%{x}<br>' + this.capitalizeFirst(this.metric) + ': %{y:.4f}<extra></extra>'
        };
        
        // Add anomaly points
        const anomalyPoints = this.recentAnomalies.map(a => {
            return {
                x: a.timestamp,
                y: a.value,
                severity: a.severity
            };
        });
        
        // Create anomaly trace
        const anomalyTrace = {
            x: anomalyPoints.map(p => p.x),
            y: anomalyPoints.map(p => p.y),
            type: 'scatter',
            mode: 'markers',
            name: 'Anomalies',
            marker: {
                color: anomalyPoints.map(p => this.getSeverityColor(p.severity)),
                size: anomalyPoints.map(p => 8 + (p.severity * 3)),
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '%{x}<br>Value: %{y:.4f}<br>Anomaly Detected<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: `Real-time ${this.capitalizeFirst(this.metric)} Anomaly Detection (${this.symbol})`,
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: this.capitalizeFirst(this.metric),
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                y: 1.15,
                xanchor: 'center'
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Anomaly Threshold: ${this.sensitivity.toFixed(1)}σ`,
                    showarrow: false,
                    xanchor: 'right',
                    yanchor: 'top',
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
                    },
                    bgcolor: 'rgba(var(--card-bg-rgb), 0.7)',
                    borderpad: 2
                }
            ]
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.realtimeChartElementId, [mainTrace, anomalyTrace], layout, config);
        this.realtimeChart = document.getElementById(this.options.realtimeChartElementId);
    }

    renderPatternLibraryChart() {
        const element = document.getElementById(this.options.patternLibraryElementId);
        if (!element || !this.patternLibrary) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Extract unique pattern types
        const patternTypes = [...new Set(this.patternLibrary.map(p => p.type))];
        
        // Create traces for each pattern type
        const traces = [];
        
        patternTypes.forEach((type, index) => {
            // Filter patterns by type
            const typePatterns = this.patternLibrary.filter(p => p.type === type);
            
            // Create a trace for each pattern
            typePatterns.forEach(pattern => {
                traces.push({
                    x: pattern.data.map(d => d.x),
                    y: pattern.data.map(d => d.y),
                    type: 'scatter',
                    mode: 'lines',
                    name: `${this.formatPatternType(type)} (ID: ${pattern.id})`,
                    line: {
                        color: this.getPatternColor(type, 0.8),
                        width: 2
                    },
                    visible: index === 0 ? true : 'legendonly', // Only show first pattern type by default
                    hovertemplate: 'Pattern: ' + this.formatPatternType(type) + '<br>Value: %{y:.4f}<extra></extra>'
                });
            });
        });
        
        // Layout configuration
        const layout = {
            title: 'Historical Anomaly Pattern Library',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Relative Time',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Normalized Value',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                y: 1.15,
                xanchor: 'center'
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.patternLibraryElementId, traces, layout, config);
        this.patternLibraryChart = document.getElementById(this.options.patternLibraryElementId);
    }

    renderClassificationChart() {
        const element = document.getElementById(this.options.classificationElementId);
        if (!element || !this.anomalyHistory) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Count anomalies by type
        const typeCounts = {};
        this.anomalyHistory.forEach(anomaly => {
            typeCounts[anomaly.type] = (typeCounts[anomaly.type] || 0) + 1;
        });
        
        // Extract labels and values
        const labels = Object.keys(typeCounts).map(type => this.formatPatternType(type));
        const values = Object.values(typeCounts);
        
        // Create pie chart trace
        const trace = {
            type: 'pie',
            labels: labels,
            values: values,
            textinfo: 'label+percent',
            textposition: 'inside',
            insidetextorientation: 'radial',
            marker: {
                colors: Object.keys(typeCounts).map(type => this.getPatternColor(type, 0.7))
            },
            hovertemplate: '%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Anomaly Classification Distribution',
            margin: { l: 20, r: 20, t: 40, b: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            showlegend: false
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.classificationElementId, [trace], layout, config);
        this.classificationChart = document.getElementById(this.options.classificationElementId);
    }

    renderSeverityChart() {
        const element = document.getElementById(this.options.severityElementId);
        if (!element || !this.severityHistory) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Extract data for the chart
        const timestamps = this.severityHistory.map(d => d.timestamp);
        const zScores = this.severityHistory.map(d => d.zScore);
        const isAnomalyPoints = this.severityHistory.filter(d => d.isAnomaly);
        
        // Z-score trace
        const zScoreTrace = {
            x: timestamps,
            y: zScores,
            type: 'scatter',
            mode: 'lines',
            name: 'Anomaly Score (Z-Score)',
            line: {
                color: 'rgba(59, 130, 246, 0.8)',
                width: 2
            },
            hovertemplate: '%{x}<br>Z-Score: %{y:.2f}<extra></extra>'
        };
        
        // Anomaly threshold line
        const thresholdTrace = {
            x: [timestamps[0], timestamps[timestamps.length - 1]],
            y: [this.sensitivity, this.sensitivity],
            type: 'scatter',
            mode: 'lines',
            name: 'Anomaly Threshold',
            line: {
                color: 'rgba(234, 179, 8, 0.8)',
                width: 2,
                dash: 'dash'
            },
            hoverinfo: 'skip'
        };
        
        // Anomaly points
        const anomalyTrace = {
            x: isAnomalyPoints.map(d => d.timestamp),
            y: isAnomalyPoints.map(d => d.zScore),
            type: 'scatter',
            mode: 'markers',
            name: 'Detected Anomalies',
            marker: {
                color: isAnomalyPoints.map(d => this.getSeverityColor(d.severity)),
                size: isAnomalyPoints.map(d => 8 + (d.severity * 2)),
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '%{x}<br>Z-Score: %{y:.2f}<br>Severity: %{marker.color}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Anomaly Severity Scoring',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Z-Score',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                y: 1.15,
                xanchor: 'center'
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.severityElementId, [zScoreTrace, thresholdTrace, anomalyTrace], layout, config);
        this.severityChart = document.getElementById(this.options.severityElementId);
    }

    updateAnomalyTable() {
        const tableBody = document.getElementById('anomaly-table-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Sort anomalies by timestamp (most recent first)
        const sortedAnomalies = [...this.recentAnomalies]
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, 10); // Only show the 10 most recent
        
        // Add rows for each anomaly
        sortedAnomalies.forEach(anomaly => {
            const row = document.createElement('tr');
            
            // Format timestamp
            const timestampCell = document.createElement('td');
            const date = new Date(anomaly.timestamp);
            timestampCell.textContent = date.toLocaleTimeString() + ' ' + date.toLocaleDateString();
            
            // Format metric
            const metricCell = document.createElement('td');
            metricCell.textContent = this.capitalizeFirst(anomaly.metric);
            
            // Format type
            const typeCell = document.createElement('td');
            typeCell.textContent = this.formatPatternType(anomaly.type);
            
            // Format severity
            const severityCell = document.createElement('td');
            const severityText = this.getSeverityText(anomaly.severity);
            severityCell.textContent = severityText;
            severityCell.className = `severity-${severityText.toLowerCase()}`;
            
            // Format z-score
            const zScoreCell = document.createElement('td');
            zScoreCell.textContent = anomaly.zScore.toFixed(2);
            
            // Format value and expected
            const valueCell = document.createElement('td');
            valueCell.textContent = anomaly.value.toFixed(4);
            
            const expectedCell = document.createElement('td');
            expectedCell.textContent = anomaly.expectedValue.toFixed(4);
            
            // Add pattern match if available
            const matchCell = document.createElement('td');
            if (anomaly.matchedPattern) {
                matchCell.innerHTML = `Pattern #${anomaly.matchedPattern} <span class="confidence">(${(anomaly.patternConfidence * 100).toFixed(0)}%)</span>`;
            } else {
                matchCell.textContent = 'None';
            }
            
            // Add all cells to the row
            row.appendChild(timestampCell);
            row.appendChild(metricCell);
            row.appendChild(typeCell);
            row.appendChild(severityCell);
            row.appendChild(zScoreCell);
            row.appendChild(valueCell);
            row.appendChild(expectedCell);
            row.appendChild(matchCell);
            
            // Add row to the table
            tableBody.appendChild(row);
        });
        
        // Update total count
        const totalCount = document.getElementById('anomaly-count');
        if (totalCount) {
            totalCount.textContent = this.recentAnomalies.length;
        }
        
        // Update severity counts
        const criticalCount = document.getElementById('critical-count');
        const alertCount = document.getElementById('alert-count');
        const warningCount = document.getElementById('warning-count');
        
        if (criticalCount) {
            criticalCount.textContent = this.recentAnomalies.filter(a => a.severity === 3).length;
        }
        
        if (alertCount) {
            alertCount.textContent = this.recentAnomalies.filter(a => a.severity === 2).length;
        }
        
        if (warningCount) {
            warningCount.textContent = this.recentAnomalies.filter(a => a.severity === 1).length;
        }
    }

    selectPattern(patternId) {
        // Find the pattern
        const pattern = this.patternLibrary.find(p => p.id === parseInt(patternId));
        if (!pattern) return;
        
        // Update the selected pattern details
        const patternName = document.getElementById('selected-pattern-name');
        const patternDesc = document.getElementById('selected-pattern-desc');
        const patternFreq = document.getElementById('selected-pattern-frequency');
        const patternImplications = document.getElementById('selected-pattern-implications');
        
        if (patternName) {
            patternName.textContent = this.formatPatternType(pattern.type);
        }
        
        if (patternDesc) {
            patternDesc.textContent = pattern.description;
        }
        
        if (patternFreq) {
            patternFreq.textContent = pattern.frequency;
        }
        
        if (patternImplications) {
            patternImplications.textContent = pattern.implications;
        }
        
        // Show the pattern details panel
        const patternDetails = document.querySelector('.pattern-details');
        if (patternDetails) {
            patternDetails.classList.add('visible');
        }
    }

    showNotification(anomaly) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `anomaly-notification severity-${this.getSeverityText(anomaly.severity).toLowerCase()}`;
        
        // Create notification content
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">${this.formatPatternType(anomaly.type)} Detected</span>
                <span class="notification-close">&times;</span>
            </div>
            <div class="notification-body">
                <div class="notification-metric">${this.capitalizeFirst(anomaly.metric)}: ${anomaly.value.toFixed(4)}</div>
                <div class="notification-info">Expected: ${anomaly.expectedValue.toFixed(4)} | Z-Score: ${anomaly.zScore.toFixed(2)}</div>
                <div class="notification-time">${new Date(anomaly.timestamp).toLocaleTimeString()}</div>
            </div>
        `;
        
        // Add to notifications area
        const notificationsArea = document.querySelector('.anomaly-notifications');
        if (notificationsArea) {
            notificationsArea.appendChild(notification);
            
            // Add close handler
            const closeButton = notification.querySelector('.notification-close');
            if (closeButton) {
                closeButton.addEventListener('click', () => {
                    notification.remove();
                });
            }
            
            // Auto-remove after 10 seconds
            setTimeout(() => {
                notification.classList.add('fade-out');
                setTimeout(() => {
                    notification.remove();
                }, 500);
            }, 10000);
        }
    }

    // Data generation and update functions
    generateMockPriceData() {
        // Generate realistic price data with occasional anomalies
        const dataPoints = 200;
        const data = [];
        
        // Base price varies by symbol
        let basePrice;
        switch (this.symbol) {
            case 'BTC-USD':
                basePrice = 30000;
                break;
            case 'ETH-USD':
                basePrice = 2000;
                break;
            case 'SOL-USD':
                basePrice = 100;
                break;
            case 'BNB-USD':
                basePrice = 300;
                break;
            default:
                basePrice = 1000;
        }
        
        // Start at the base price
        let price = basePrice;
        
        // Generate data points
        const now = new Date();
        for (let i = 0; i < dataPoints; i++) {
            // Basic random walk
            const randomMove = (Math.random() - 0.5) * (basePrice * 0.005);
            
            // Add some trends
            const trend = Math.sin(i / 20) * (basePrice * 0.01);
            
            // Add occasional jumps (anomalies)
            let jump = 0;
            if (Math.random() < 0.03) {
                jump = (Math.random() > 0.5 ? 1 : -1) * (basePrice * (0.02 + Math.random() * 0.05));
            }
            
            // Update price
            price = price + randomMove + trend + jump;
            
            // Ensure price doesn't go negative
            price = Math.max(price, basePrice * 0.5);
            
            // Calculate timestamp (going backward from now)
            const timestamp = new Date(now.getTime() - ((dataPoints - i) * 60000));
            
            // Add to data array
            data.push({
                timestamp: timestamp,
                value: price
            });
        }
        
        return data;
    }

    generateMockVolumeData() {
        // Generate realistic volume data with occasional spikes
        const dataPoints = 200;
        const data = [];
        
        // Base volume varies by symbol
        let baseVolume;
        switch (this.symbol) {
            case 'BTC-USD':
                baseVolume = 1000;
                break;
            case 'ETH-USD':
                baseVolume = 5000;
                break;
            case 'SOL-USD':
                baseVolume = 20000;
                break;
            case 'BNB-USD':
                baseVolume = 3000;
                break;
            default:
                baseVolume = 2000;
        }
        
        // Generate data points
        const now = new Date();
        for (let i = 0; i < dataPoints; i++) {
            // Basic random volume
            const volumeVariation = (Math.random() * 0.4 + 0.8);
            let volume = baseVolume * volumeVariation;
            
            // Add daily cycle
            const hourOfDay = (now.getHours() + (24 - (dataPoints - i) / 60)) % 24;
            const dayCycle = Math.sin((hourOfDay - 10) * Math.PI / 12) * 0.3 + 0.7;
            volume *= dayCycle;
            
            // Add occasional volume spikes
            if (Math.random() < 0.02) {
                volume *= (2 + Math.random() * 5);
            }
            
            // Calculate timestamp (going backward from now)
            const timestamp = new Date(now.getTime() - ((dataPoints - i) * 60000));
            
            // Add to data array
            data.push({
                timestamp: timestamp,
                value: volume
            });
        }
        
        return data;
    }

    generateMockVolatilityData() {
        // Generate realistic volatility data
        const dataPoints = 200;
        const data = [];
        
        // Base volatility varies by symbol
        let baseVolatility;
        switch (this.symbol) {
            case 'BTC-USD':
                baseVolatility = 0.02;
                break;
            case 'ETH-USD':
                baseVolatility = 0.03;
                break;
            case 'SOL-USD':
                baseVolatility = 0.04;
                break;
            case 'BNB-USD':
                baseVolatility = 0.025;
                break;
            default:
                baseVolatility = 0.01;
        }
        
        // Start with base volatility
        let volatility = baseVolatility;
        
        // Generate data points
        const now = new Date();
        for (let i = 0; i < dataPoints; i++) {
            // Random walk for volatility
            const randomChange = (Math.random() - 0.5) * 0.005;
            volatility = Math.max(0.001, volatility + randomChange);
            
            // Add occasional volatility spikes
            if (Math.random() < 0.03) {
                volatility += baseVolatility * (1 + Math.random() * 2);
            }
            
            // Mean reversion
            volatility = volatility * 0.95 + baseVolatility * 0.05;
            
            // Calculate timestamp (going backward from now)
            const timestamp = new Date(now.getTime() - ((dataPoints - i) * 60000));
            
            // Add to data array
            data.push({
                timestamp: timestamp,
                value: volatility
            });
        }
        
        return data;
    }

    generateMockPatternLibrary() {
        // Generate a library of anomaly patterns
        const patternTypes = [
            'price_spike', 'price_drop', 'volatility_surge', 
            'liquidity_gap', 'flash_crash', 'market_manipulation'
        ];
        
        const library = [];
        
        // Create patterns for each type
        patternTypes.forEach((type, index) => {
            // Create 1-3 patterns per type
            const patternCount = 1 + Math.floor(Math.random() * 3);
            
            for (let i = 0; i < patternCount; i++) {
                // Generate pattern data (normalized -1 to 1)
                const patternData = [];
                const dataPoints = 20 + Math.floor(Math.random() * 10);
                
                // Starting point
                let value = 0;
                
                // Generate data points
                for (let j = 0; j < dataPoints; j++) {
                    // Base pattern varies by type
                    if (type === 'price_spike') {
                        // Gradual rise then sharp up
                        if (j < dataPoints * 0.7) {
                            value += 0.05 * Math.random();
                        } else {
                            value += 0.15 + 0.1 * Math.random();
                        }
                    } else if (type === 'price_drop') {
                        // Gradual decline then sharp down
                        if (j < dataPoints * 0.7) {
                            value -= 0.05 * Math.random();
                        } else {
                            value -= 0.15 + 0.1 * Math.random();
                        }
                    } else if (type === 'volatility_surge') {
                        // Increasing oscillations
                        const oscillation = Math.sin(j * 0.5) * (j / dataPoints);
                        value = oscillation;
                    } else if (type === 'liquidity_gap') {
                        // Sudden drop then recovery
                        if (j === Math.floor(dataPoints * 0.5)) {
                            value -= 0.5;
                        } else if (j > Math.floor(dataPoints * 0.5)) {
                            value += 0.1;
                        } else {
                            value += (Math.random() - 0.5) * 0.1;
                        }
                    } else if (type === 'flash_crash') {
                        // Sharp drop then quick recovery
                        if (j === Math.floor(dataPoints * 0.5)) {
                            value -= 0.8;
                        } else if (j > Math.floor(dataPoints * 0.5)) {
                            value += 0.2;
                        } else {
                            value += (Math.random() - 0.5) * 0.05;
                        }
                    } else if (type === 'market_manipulation') {
                        // Pump and dump pattern
                        if (j < dataPoints * 0.4) {
                            value += 0.1;
                        } else if (j < dataPoints * 0.6) {
                            value += 0.2;
                        } else {
                            value -= 0.15;
                        }
                    }
                    
                    // Add some noise
                    value += (Math.random() - 0.5) * 0.05;
                    
                    // Add to pattern data
                    patternData.push({
                        x: j,
                        y: value
                    });
                }
                
                // Create pattern object
                const pattern = {
                    id: library.length + 1,
                    type: type,
                    data: patternData,
                    description: this.getPatternDescription(type),
                    frequency: this.getPatternFrequency(type),
                    implications: this.getPatternImplications(type),
                    zScore: 2.5 + Math.random() * 2, // Typical z-score for this pattern
                    confidence: 0.7 + Math.random() * 0.3 // Pattern detection confidence
                };
                
                library.push(pattern);
            }
        });
        
        return library;
    }

    generateMockAnomalyHistory() {
        // Generate historical anomaly records
        const anomalyCount = 50 + Math.floor(Math.random() * 50);
        const history = [];
        
        // Pattern types
        const patternTypes = [
            'price_spike', 'price_drop', 'volatility_surge', 
            'liquidity_gap', 'flash_crash', 'market_manipulation'
        ];
        
        // Metrics
        const metrics = ['price', 'volume', 'volatility'];
        
        // Generate anomalies
        const now = new Date();
        for (let i = 0; i < anomalyCount; i++) {
            // Random timestamp within the last 30 days
            const daysAgo = Math.random() * 30;
            const timestamp = new Date(now.getTime() - (daysAgo * 24 * 60 * 60 * 1000));
            
            // Random pattern type
            const typeIndex = Math.floor(Math.random() * patternTypes.length);
            const type = patternTypes[typeIndex];
            
            // Random metric
            const metricIndex = Math.floor(Math.random() * metrics.length);
            const metric = metrics[metricIndex];
            
            // Random z-score (above threshold)
            const zScore = this.sensitivity + Math.random() * 5;
            
            // Random values
            const expectedValue = this.getBaseValue(metric);
            const deviation = expectedValue * (zScore / 10) * (Math.random() > 0.5 ? 1 : -1);
            const value = expectedValue + deviation;
            
            // Calculate severity
            const severity = this.calculateSeverity(zScore);
            
            // Create anomaly object
            const anomaly = {
                timestamp: timestamp,
                type: type,
                metric: metric,
                value: value,
                expectedValue: expectedValue,
                zScore: zScore,
                severity: severity
            };
            
            history.push(anomaly);
        }
        
        // Sort by timestamp (oldest first)
        history.sort((a, b) => a.timestamp - b.timestamp);
        
        return history;
    }

    addNewDataPoint() {
        // Add a new data point to price, volume, and volatility data
        
        // Price data
        if (this.priceData.length > 0) {
            const lastPrice = this.priceData[this.priceData.length - 1].value;
            const randomMove = (Math.random() - 0.5) * (lastPrice * 0.005);
            
            // Add occasional jumps (anomalies)
            let jump = 0;
            if (Math.random() < 0.05) {
                jump = (Math.random() > 0.5 ? 1 : -1) * (lastPrice * (0.01 + Math.random() * 0.03));
            }
            
            // New price and timestamp
            const newPrice = lastPrice + randomMove + jump;
            const newTimestamp = new Date();
            
            // Add to price data
            this.priceData.push({
                timestamp: newTimestamp,
                value: newPrice
            });
            
            // Remove oldest point if needed
            if (this.priceData.length > this.options.historyLength) {
                this.priceData.shift();
            }
        }
        
        // Volume data
        if (this.volumeData.length > 0) {
            const baseVolume = this.volumeData.reduce((sum, point) => sum + point.value, 0) / this.volumeData.length;
            
            // Basic random volume
            const volumeVariation = (Math.random() * 0.4 + 0.8);
            let newVolume = baseVolume * volumeVariation;
            
            // Add occasional volume spikes
            if (Math.random() < 0.05) {
                newVolume *= (1.5 + Math.random() * 3);
            }
            
            // New timestamp
            const newTimestamp = new Date();
            
            // Add to volume data
            this.volumeData.push({
                timestamp: newTimestamp,
                value: newVolume
            });
            
            // Remove oldest point if needed
            if (this.volumeData.length > this.options.historyLength) {
                this.volumeData.shift();
            }
        }
        
        // Volatility data
        if (this.volatilityData.length > 0) {
            const lastVolatility = this.volatilityData[this.volatilityData.length - 1].value;
            
            // Random walk for volatility
            const randomChange = (Math.random() - 0.5) * 0.002;
            let newVolatility = Math.max(0.001, lastVolatility + randomChange);
            
            // Add occasional volatility spikes
            if (Math.random() < 0.05) {
                newVolatility += lastVolatility * (Math.random() * 2);
            }
            
            // Mean reversion
            const baseVolatility = 0.02;
            newVolatility = newVolatility * 0.95 + baseVolatility * 0.05;
            
            // New timestamp
            const newTimestamp = new Date();
            
            // Add to volatility data
            this.volatilityData.push({
                timestamp: newTimestamp,
                value: newVolatility
            });
            
            // Remove oldest point if needed
            if (this.volatilityData.length > this.options.historyLength) {
                this.volatilityData.shift();
            }
        }
    }

    // Helper functions
    getBaseValue(metric) {
        // Get base value for a metric based on symbol
        if (metric === 'price') {
            switch (this.symbol) {
                case 'BTC-USD': return 30000;
                case 'ETH-USD': return 2000;
                case 'SOL-USD': return 100;
                case 'BNB-USD': return 300;
                default: return 1000;
            }
        } else if (metric === 'volume') {
            switch (this.symbol) {
                case 'BTC-USD': return 1000;
                case 'ETH-USD': return 5000;
                case 'SOL-USD': return 20000;
                case 'BNB-USD': return 3000;
                default: return 2000;
            }
        } else if (metric === 'volatility') {
            switch (this.symbol) {
                case 'BTC-USD': return 0.02;
                case 'ETH-USD': return 0.03;
                case 'SOL-USD': return 0.04;
                case 'BNB-USD': return 0.025;
                default: return 0.01;
            }
        }
        
        return 1; // Default
    }

    getPatternDescription(type) {
        // Get description text for pattern type
        switch (type) {
            case 'price_spike':
                return 'Sudden upward price movement often driven by significant buying pressure or positive news events.';
            case 'price_drop':
                return 'Rapid price decline typically caused by sell-offs, liquidations, or negative market news.';
            case 'volatility_surge':
                return 'Dramatic increase in price volatility, often preceding major market moves or uncertainty.';
            case 'liquidity_gap':
                return 'Sudden reduction in market liquidity, causing slippage and price gaps between trades.';
            case 'flash_crash':
                return 'Extreme price drop followed by quick recovery, often triggered by cascading liquidations.';
            case 'market_manipulation':
                return 'Artificial price movement pattern suggesting coordinated buying or selling activity.';
            default:
                return 'Unknown pattern type.';
        }
    }

    getPatternFrequency(type) {
        // Get frequency text for pattern type
        switch (type) {
            case 'price_spike':
                return 'Common (15-20 times per month)';
            case 'price_drop':
                return 'Common (12-18 times per month)';
            case 'volatility_surge':
                return 'Moderate (8-12 times per month)';
            case 'liquidity_gap':
                return 'Uncommon (3-6 times per month)';
            case 'flash_crash':
                return 'Rare (1-2 times per month)';
            case 'market_manipulation':
                return 'Uncommon (4-8 times per month)';
            default:
                return 'Unknown frequency';
        }
    }

    getPatternImplications(type) {
        // Get market implications for pattern type
        switch (type) {
            case 'price_spike':
                return 'Often indicates strong momentum but may lead to a short-term reversal. Can trigger stop orders and liquidations in short positions.';
            case 'price_drop':
                return 'May indicate trend reversal or continuation of downtrend. Often causes cascading liquidations and opportunity for dip buyers.';
            case 'volatility_surge':
                return 'Suggests market uncertainty and potential for large moves in either direction. Option premiums typically increase.';
            case 'liquidity_gap':
                return 'Indicates thin order books and potential for erratic price movement. Increased slippage risk for market orders.';
            case 'flash_crash':
                return 'Creates opportunities for limit orders below market price. Often followed by a period of increased volatility.';
            case 'market_manipulation':
                return 'Typically followed by price reversal once manipulation ends. May indicate accumulation or distribution by large players.';
            default:
                return 'Unknown implications';
        }
    }

    formatPatternType(type) {
        // Format pattern type for display
        return type.split('_').map(word => this.capitalizeFirst(word)).join(' ');
    }

    getSeverityText(severity) {
        // Get text for severity level
        switch (severity) {
            case 0: return 'Normal';
            case 1: return 'Warning';
            case 2: return 'Alert';
            case 3: return 'Critical';
            default: return 'Unknown';
        }
    }

    getSeverityColor(severity) {
        // Get color for severity level
        switch (severity) {
            case 0: return 'rgba(59, 130, 246, 0.7)'; // Blue
            case 1: return 'rgba(234, 179, 8, 0.7)';  // Yellow
            case 2: return 'rgba(249, 115, 22, 0.7)'; // Orange
            case 3: return 'rgba(239, 68, 68, 0.7)';  // Red
            default: return 'rgba(156, 163, 175, 0.7)'; // Gray
        }
    }

    getPatternColor(type, alpha = 1) {
        // Get color for pattern type
        switch (type) {
            case 'price_spike':
                return `rgba(16, 185, 129, ${alpha})`;  // Green
            case 'price_drop':
                return `rgba(239, 68, 68, ${alpha})`;   // Red
            case 'volatility_surge':
                return `rgba(234, 179, 8, ${alpha})`;   // Yellow
            case 'liquidity_gap':
                return `rgba(124, 58, 237, ${alpha})`;  // Purple
            case 'flash_crash':
                return `rgba(239, 68, 68, ${alpha})`;   // Red
            case 'market_manipulation':
                return `rgba(59, 130, 246, ${alpha})`;  // Blue
            default:
                return `rgba(156, 163, 175, ${alpha})`;  // Gray
        }
    }

    capitalizeFirst(string) {
        // Capitalize first letter of string
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    showLoading(element) {
        // Show loading overlay on charts
        const elements = element ? [element] : [
            document.getElementById(this.options.realtimeChartElementId),
            document.getElementById(this.options.patternLibraryElementId),
            document.getElementById(this.options.classificationElementId),
            document.getElementById(this.options.severityElementId)
        ];
        
        elements.forEach(el => {
            if (!el) return;
            
            let overlay = el.querySelector('.chart-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'chart-overlay';
                el.appendChild(overlay);
            }
            
            overlay.textContent = 'Loading...';
            overlay.style.display = 'flex';
        });
    }

    hideLoading(element) {
        // Hide loading overlay
        const elements = element ? [element] : [
            document.getElementById(this.options.realtimeChartElementId),
            document.getElementById(this.options.patternLibraryElementId),
            document.getElementById(this.options.classificationElementId),
            document.getElementById(this.options.severityElementId)
        ];
        
        elements.forEach(el => {
            if (!el) return;
            
            const overlay = el.querySelector('.chart-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        });
    }

    showError(message, element) {
        // Show error message on charts
        const elements = element ? [element] : [
            document.getElementById(this.options.realtimeChartElementId),
            document.getElementById(this.options.patternLibraryElementId),
            document.getElementById(this.options.classificationElementId),
            document.getElementById(this.options.severityElementId)
        ];
        
        elements.forEach(el => {
            if (!el) return;
            
            let overlay = el.querySelector('.chart-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'chart-overlay';
                el.appendChild(overlay);
            }
            
            overlay.textContent = message || 'Error loading data';
            overlay.style.display = 'flex';
        });
    }

    exportData() {
        // Export the current data as JSON
        const data = {
            symbol: this.symbol,
            metric: this.metric,
            sensitivity: this.sensitivity,
            timestamp: new Date(),
            priceData: this.priceData,
            volumeData: this.volumeData,
            volatilityData: this.volatilityData,
            anomalies: this.recentAnomalies,
            patternLibrary: this.patternLibrary
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `anomaly_detection_${this.symbol}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of AnomalyDetection
    const anomalyDetection = new AnomalyDetection();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});