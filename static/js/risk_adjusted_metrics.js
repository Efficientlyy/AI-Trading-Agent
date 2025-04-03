/**
 * Risk-Adjusted Performance Metrics
 * 
 * This module provides visualizations of various risk-adjusted performance metrics
 * for portfolio and trading strategy evaluation.
 */

class RiskAdjustedMetrics {
    constructor(options = {}) {
        this.options = Object.assign({
            ratioChartElementId: 'ratio-chart',
            evolutionChartElementId: 'evolution-chart',
            surfaceChartElementId: 'surface-chart',
            drawdownChartElementId: 'drawdown-evolution-chart',
            updateInterval: 60000, // milliseconds
            colorScalePositive: [
                [0, 'rgba(16, 185, 129, 0.1)'],  // Light green
                [1, 'rgba(16, 185, 129, 0.9)']   // Dark green
            ],
            colorScaleNegative: [
                [0, 'rgba(239, 68, 68, 0.1)'],   // Light red
                [1, 'rgba(239, 68, 68, 0.9)']    // Dark red
            ],
            defaultTimeframe: '1m',
            enableOptimizations: true
        }, options);

        // State management
        this.timeframe = document.getElementById('risk-timeframe')?.value || this.options.defaultTimeframe;
        this.ratioType = document.getElementById('ratio-type')?.value || 'sharpe';
        this.surfaceMetric = document.getElementById('surface-metric')?.value || 'sharpe';
        
        // Data containers
        this.performanceData = [];
        this.ratioData = {};
        this.riskRewardData = {};
        
        // Chart objects
        this.ratioChart = null;
        this.evolutionChart = null;
        this.surfaceChart = null;
        this.drawdownChart = null;
        
        // Flags
        this.isInitialized = false;
        this.updateTimer = null;
        
        // Performance optimization
        this.optimizer = null;
        if (this.options.enableOptimizations && typeof window.DashboardOptimizer !== 'undefined') {
            this.optimizer = new DashboardOptimizer();
        }
        
        // Initialize
        this.initialize();
    }

    initialize() {
        // Fetch initial data and set up charts
        this.fetchData()
            .then(() => {
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
                console.error('Error initializing Risk-Adjusted Metrics:', error);
                this.showError('Failed to initialize risk-adjusted metrics');
            });
    }

    fetchData() {
        // If optimizer is available, use it for efficient data fetching
        if (this.optimizer) {
            return Promise.all([
                this.fetchPerformanceDataOptimized(),
                this.fetchRatioDataOptimized(),
                this.fetchRiskRewardDataOptimized()
            ]);
        } else {
            // Fall back to original implementation
            return Promise.all([
                this.fetchPerformanceData(),
                this.fetchRatioData(),
                this.fetchRiskRewardData()
            ]);
        }
    }

    // Optimized data fetching methods
    fetchPerformanceDataOptimized() {
        return this.optimizer.getData('performance-data', { 
            timeframe: this.timeframe 
        }).then(data => {
            this.performanceData = data;
            return data;
        });
    }

    fetchRatioDataOptimized() {
        return Promise.all([
            this.optimizer.getData('risk-metrics', { type: 'sharpe', timeframe: this.timeframe }),
            this.optimizer.getData('risk-metrics', { type: 'sortino', timeframe: this.timeframe }),
            this.optimizer.getData('risk-metrics', { type: 'calmar', timeframe: this.timeframe }),
            this.optimizer.getData('risk-metrics', { type: 'mar', timeframe: this.timeframe }),
            this.optimizer.getData('risk-metrics', { type: 'treynor', timeframe: this.timeframe })
        ]).then(([sharpe, sortino, calmar, mar, treynor]) => {
            const data = {
                sharpe: sharpe,
                sortino: sortino,
                calmar: calmar,
                mar: mar,
                treynor: treynor
            };
            this.ratioData = data;
            return data;
        });
    }

    fetchRiskRewardDataOptimized() {
        return this.optimizer.getData('risk-reward', { 
            metric: this.surfaceMetric,
            timeframe: this.timeframe 
        }).then(data => {
            this.riskRewardData = data;
            return data;
        });
    }

    // Original data fetching methods (fallback)
    fetchPerformanceData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock performance data
                const data = this.generateMockPerformanceData();
                this.performanceData = data;
                resolve(data);
            }, 300);
        });
    }

    fetchRatioData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock ratio data
                const data = {
                    sharpe: this.generateMockRatioData('sharpe'),
                    sortino: this.generateMockRatioData('sortino'),
                    calmar: this.generateMockRatioData('calmar'),
                    mar: this.generateMockRatioData('mar'),
                    treynor: this.generateMockRatioData('treynor')
                };
                this.ratioData = data;
                resolve(data);
            }, 250);
        });
    }

    fetchRiskRewardData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock risk-reward surface data
                const data = this.generateMockRiskRewardData();
                this.riskRewardData = data;
                resolve(data);
            }, 350);
        });
    }

    initializeCharts() {
        this.renderRatioChart();
        this.renderEvolutionChart();
        this.renderSurfaceChart();
        this.renderDrawdownChart();
        this.updateMetricsTable();
        this.updateStrategyTable();
    }

    setupEventListeners() {
        // Timeframe selector change
        const timeframeSelector = document.getElementById('risk-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }

        // Ratio type selector change
        const ratioTypeSelector = document.getElementById('ratio-type');
        if (ratioTypeSelector) {
            ratioTypeSelector.addEventListener('change', () => {
                this.ratioType = ratioTypeSelector.value;
                this.renderRatioChart();
                this.renderEvolutionChart();
            });
        }

        // Surface metric selector change
        const surfaceMetricSelector = document.getElementById('surface-metric');
        if (surfaceMetricSelector) {
            surfaceMetricSelector.addEventListener('change', () => {
                this.surfaceMetric = surfaceMetricSelector.value;
                this.renderSurfaceChart();
            });
        }

        // Refresh button
        const refreshButton = document.getElementById('refresh-risk-metrics');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshData();
            });
        }

        // Export data button
        const exportButton = document.getElementById('export-risk-data');
        if (exportButton) {
            exportButton.addEventListener('click', () => {
                this.exportData();
            });
        }
    }

    startRealtimeUpdates() {
        // Clear any existing timers
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }

        // Set up timer for real-time updates
        this.updateTimer = setInterval(() => {
            this.updateData();
        }, this.options.updateInterval);
    }

    stopRealtimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    refreshData() {
        // Show loading state
        this.showLoading();
        
        // Fetch new data
        this.fetchData()
            .then(() => {
                this.renderRatioChart();
                this.renderEvolutionChart();
                this.renderSurfaceChart();
                this.renderDrawdownChart();
                this.updateMetricsTable();
                this.updateStrategyTable();
                
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                this.showError('Failed to refresh risk-adjusted metrics data');
            });
    }

    updateData() {
        // Update data with slight changes
        this.updatePerformanceData();
        this.updateRatioData();
        
        // Re-render visualizations
        this.renderRatioChart();
        this.renderEvolutionChart();
        this.renderSurfaceChart();
        this.renderDrawdownChart();
        this.updateMetricsTable();
        this.updateStrategyTable();
    }

    renderRatioChart() {
        const element = document.getElementById(this.options.ratioChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Get relevant data based on selected ratio type
        const ratioData = this.ratioData[this.ratioType];
        if (!ratioData || !ratioData.strategies) return;
        
        // Prepare data for bar chart
        const strategies = Object.keys(ratioData.strategies);
        const ratioValues = strategies.map(strategy => ratioData.strategies[strategy].current);
        
        // Color based on values (positive or negative)
        const colors = ratioValues.map(value => 
            value >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'
        );
        
        // Create trace for bar chart
        const trace = {
            type: 'bar',
            x: strategies,
            y: ratioValues,
            marker: {
                color: colors
            },
            text: ratioValues.map(v => v.toFixed(2)),
            textposition: 'auto',
            hovertemplate: '%{x}<br>' + this.formatRatioName(this.ratioType) + ': %{y:.2f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: this.formatRatioName(this.ratioType) + ' Ratio by Strategy',
            margin: { l: 60, r: 20, t: 40, b: 80 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Strategy',
                tickangle: -45,
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: this.formatRatioName(this.ratioType) + ' Ratio',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Timeframe: ${this.formatTimeframe(this.timeframe)}`,
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
        Plotly.newPlot(this.options.ratioChartElementId, [trace], layout, config);
        this.ratioChart = document.getElementById(this.options.ratioChartElementId);
    }

    renderEvolutionChart() {
        const element = document.getElementById(this.options.evolutionChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Get relevant data based on selected ratio type
        const ratioData = this.ratioData[this.ratioType];
        if (!ratioData || !ratioData.strategies) return;
        
        // Prepare data for line chart
        const traces = [];
        
        // Create a trace for each strategy
        Object.entries(ratioData.strategies).forEach(([strategy, data]) => {
            if (!data.history) return;
            
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: strategy,
                x: data.history.map(p => p.date),
                y: data.history.map(p => p.value),
                line: {
                    width: 2
                },
                hovertemplate: '%{x|%b %d, %Y}<br>' + this.formatRatioName(this.ratioType) + ': %{y:.2f}<extra>' + strategy + '</extra>'
            });
        });
        
        // Layout configuration
        const layout = {
            title: this.formatRatioName(this.ratioType) + ' Ratio Evolution',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Date',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: this.formatRatioName(this.ratioType) + ' Ratio',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1
            },
            legend: {
                orientation: 'h',
                y: 1.12,
                x: 0.5,
                xanchor: 'center'
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Timeframe: ${this.formatTimeframe(this.timeframe)}`,
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
        Plotly.newPlot(this.options.evolutionChartElementId, traces, layout, config);
        this.evolutionChart = document.getElementById(this.options.evolutionChartElementId);
    }

    renderSurfaceChart() {
        const element = document.getElementById(this.options.surfaceChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Get relevant data for the surface chart
        const surfaceData = this.riskRewardData[this.surfaceMetric];
        if (!surfaceData) return;
        
        // Determine color scale based on metric
        let colorscale;
        if (this.surfaceMetric === 'sharpe' || this.surfaceMetric === 'sortino') {
            colorscale = [
                [0, 'rgba(239, 68, 68, 0.7)'],      // Red for low values
                [0.5, 'rgba(234, 179, 8, 0.7)'],    // Yellow for mid values
                [1, 'rgba(16, 185, 129, 0.7)']      // Green for high values
            ];
        } else {
            colorscale = [
                [0, 'rgba(59, 130, 246, 0.7)'],     // Blue for low values
                [0.5, 'rgba(124, 58, 237, 0.7)'],   // Purple for mid values
                [1, 'rgba(16, 185, 129, 0.7)']      // Green for high values
            ];
        }
        
        // Create surface plot
        const trace = {
            type: 'surface',
            x: surfaceData.riskLevels,
            y: surfaceData.returnLevels,
            z: surfaceData.values,
            colorscale: colorscale,
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "rgba(255,255,255,0.5)",
                    project: {z: true}
                }
            },
            hovertemplate: 'Risk: %{x:.2f}<br>Return: %{y:.2f}<br>' + this.formatRatioName(this.surfaceMetric) + ': %{z:.2f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Risk/Reward Optimization Surface',
            margin: { l: 40, r: 40, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            scene: {
                xaxis: {
                    title: 'Risk',
                    backgroundcolor: 'transparent',
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    showbackground: false
                },
                yaxis: {
                    title: 'Return',
                    backgroundcolor: 'transparent',
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    showbackground: false
                },
                zaxis: {
                    title: this.formatRatioName(this.surfaceMetric),
                    backgroundcolor: 'transparent',
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    showbackground: false
                },
                camera: {
                    eye: { x: 1.5, y: -1.5, z: 1 }
                }
            },
            annotations: [
                {
                    x: 0,
                    y: 0,
                    z: surfaceData.optimalValue,
                    text: 'Optimal',
                    showarrow: true,
                    arrowhead: 3,
                    ax: 30,
                    ay: 30,
                    font: {
                        color: 'rgba(255, 255, 255, 0.9)',
                        size: 12
                    }
                }
            ]
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select2d', 'lasso2d']
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.surfaceChartElementId, [trace], layout, config);
        this.surfaceChart = document.getElementById(this.options.surfaceChartElementId);
    }

    renderDrawdownChart() {
        const element = document.getElementById(this.options.drawdownChartElementId);
        if (!element || !this.performanceData || !this.performanceData.strategies) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Prepare data for drawdown chart
        const traces = [];
        
        // Create a trace for each strategy
        Object.entries(this.performanceData.strategies).forEach(([strategy, data]) => {
            if (!data.drawdowns) return;
            
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: strategy,
                x: data.drawdowns.map(p => p.date),
                y: data.drawdowns.map(p => p.value),
                fill: 'tozeroy',
                fillcolor: 'rgba(239, 68, 68, 0.1)',
                line: {
                    color: 'rgba(239, 68, 68, 0.8)',
                    width: 2
                },
                hovertemplate: '%{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra>' + strategy + '</extra>'
            });
        });
        
        // Layout configuration
        const layout = {
            title: 'Maximum Drawdown Evolution',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Date',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Drawdown (%)',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1,
                autorange: 'reversed' // Inverted so drawdowns go down
            },
            legend: {
                orientation: 'h',
                y: 1.12,
                x: 0.5,
                xanchor: 'center'
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.drawdownChartElementId, traces, layout, config);
        this.drawdownChart = document.getElementById(this.options.drawdownChartElementId);
    }

    updateMetricsTable() {
        const tableBody = document.getElementById('metrics-table-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Define metrics to display
        const metrics = ['sharpe', 'sortino', 'calmar', 'mar', 'treynor'];
        
        // If optimizer is available, use chunked rendering for better performance
        if (this.optimizer) {
            const renderMetricRow = (metric) => {
                const row = document.createElement('tr');
                
                // Metric name
                const nameCell = document.createElement('td');
                nameCell.textContent = this.formatRatioName(metric);
                
                // Target value
                const targetCell = document.createElement('td');
                const targetValue = this.getMetricTarget(metric);
                targetCell.textContent = targetValue;
                
                // Current value
                const currentCell = document.createElement('td');
                if (this.ratioData[metric] && this.ratioData[metric].portfolio) {
                    const value = this.ratioData[metric].portfolio.current;
                    currentCell.textContent = value.toFixed(2);
                    currentCell.className = value >= targetValue ? 'positive' : 'negative';
                } else {
                    currentCell.textContent = 'N/A';
                }
                
                // Best value
                const bestCell = document.createElement('td');
                if (this.ratioData[metric] && this.ratioData[metric].portfolio && this.ratioData[metric].portfolio.history) {
                    const history = this.ratioData[metric].portfolio.history;
                    const bestValue = Math.max(...history.map(p => p.value));
                    bestCell.textContent = bestValue.toFixed(2);
                } else {
                    bestCell.textContent = 'N/A';
                }
                
                // Description
                const descCell = document.createElement('td');
                descCell.textContent = this.getMetricDescription(metric);
                
                // Add cells to row
                row.appendChild(nameCell);
                row.appendChild(targetCell);
                row.appendChild(currentCell);
                row.appendChild(bestCell);
                row.appendChild(descCell);
                
                return row;
            };
            
            // Use efficient chunked rendering
            this.optimizer.renderChunked('metrics-table-body', metrics, renderMetricRow);
        } else {
            // Fallback to traditional method
            metrics.forEach(metric => {
                const row = document.createElement('tr');
                
                // Metric name
                const nameCell = document.createElement('td');
                nameCell.textContent = this.formatRatioName(metric);
                
                // Target value
                const targetCell = document.createElement('td');
                const targetValue = this.getMetricTarget(metric);
                targetCell.textContent = targetValue;
                
                // Current value
                const currentCell = document.createElement('td');
                if (this.ratioData[metric] && this.ratioData[metric].portfolio) {
                    const value = this.ratioData[metric].portfolio.current;
                    currentCell.textContent = value.toFixed(2);
                    currentCell.className = value >= targetValue ? 'positive' : 'negative';
                } else {
                    currentCell.textContent = 'N/A';
                }
                
                // Best value
                const bestCell = document.createElement('td');
                if (this.ratioData[metric] && this.ratioData[metric].portfolio && this.ratioData[metric].portfolio.history) {
                    const history = this.ratioData[metric].portfolio.history;
                    const bestValue = Math.max(...history.map(p => p.value));
                    bestCell.textContent = bestValue.toFixed(2);
                } else {
                    bestCell.textContent = 'N/A';
                }
                
                // Description
                const descCell = document.createElement('td');
                descCell.textContent = this.getMetricDescription(metric);
                
                // Add cells to row
                row.appendChild(nameCell);
                row.appendChild(targetCell);
                row.appendChild(currentCell);
                row.appendChild(bestCell);
                row.appendChild(descCell);
                
                // Add row to table
                tableBody.appendChild(row);
            });
        }
    }

    updateStrategyTable() {
        const tableBody = document.getElementById('strategy-table-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Ensure ratio data exists
        if (!this.ratioData.sharpe || !this.ratioData.sharpe.strategies) return;
        
        // Get strategies
        const strategies = Object.keys(this.ratioData.sharpe.strategies);
        
        // If optimizer is available, use chunked rendering for better performance
        if (this.optimizer) {
            const renderStrategyRow = (strategy) => {
                const row = document.createElement('tr');
                
                // Strategy name
                const nameCell = document.createElement('td');
                nameCell.textContent = strategy;
                
                // Sharpe ratio
                const sharpeCell = document.createElement('td');
                if (this.ratioData.sharpe.strategies[strategy]) {
                    const value = this.ratioData.sharpe.strategies[strategy].current;
                    sharpeCell.textContent = value.toFixed(2);
                    sharpeCell.className = value >= 1 ? 'positive' : 'negative';
                } else {
                    sharpeCell.textContent = 'N/A';
                }
                
                // Sortino ratio
                const sortinoCell = document.createElement('td');
                if (this.ratioData.sortino.strategies[strategy]) {
                    const value = this.ratioData.sortino.strategies[strategy].current;
                    sortinoCell.textContent = value.toFixed(2);
                    sortinoCell.className = value >= 1.5 ? 'positive' : 'negative';
                } else {
                    sortinoCell.textContent = 'N/A';
                }
                
                // Calmar ratio
                const calmarCell = document.createElement('td');
                if (this.ratioData.calmar.strategies[strategy]) {
                    const value = this.ratioData.calmar.strategies[strategy].current;
                    calmarCell.textContent = value.toFixed(2);
                    calmarCell.className = value >= 0.5 ? 'positive' : 'negative';
                } else {
                    calmarCell.textContent = 'N/A';
                }
                
                // Max drawdown
                const drawdownCell = document.createElement('td');
                if (this.performanceData.strategies[strategy]) {
                    const currentDrawdown = this.performanceData.strategies[strategy].currentDrawdown;
                    const maxDrawdown = this.performanceData.strategies[strategy].maxDrawdown;
                    drawdownCell.textContent = `${currentDrawdown.toFixed(2)}% / ${maxDrawdown.toFixed(2)}%`;
                    drawdownCell.className = currentDrawdown < 10 ? 'positive' : 
                                          currentDrawdown < 20 ? 'neutral' : 'negative';
                } else {
                    drawdownCell.textContent = 'N/A';
                }
                
                // Add cells to row
                row.appendChild(nameCell);
                row.appendChild(sharpeCell);
                row.appendChild(sortinoCell);
                row.appendChild(calmarCell);
                row.appendChild(drawdownCell);
                
                return row;
            };
            
            // Use efficient chunked rendering
            this.optimizer.renderChunked('strategy-table-body', strategies, renderStrategyRow);
        } else {
            // Fallback to traditional method
            strategies.forEach(strategy => {
                const row = document.createElement('tr');
                
                // Strategy name
                const nameCell = document.createElement('td');
                nameCell.textContent = strategy;
                
                // Sharpe ratio
                const sharpeCell = document.createElement('td');
                if (this.ratioData.sharpe.strategies[strategy]) {
                    const value = this.ratioData.sharpe.strategies[strategy].current;
                    sharpeCell.textContent = value.toFixed(2);
                    sharpeCell.className = value >= 1 ? 'positive' : 'negative';
                } else {
                    sharpeCell.textContent = 'N/A';
                }
                
                // Sortino ratio
                const sortinoCell = document.createElement('td');
                if (this.ratioData.sortino.strategies[strategy]) {
                    const value = this.ratioData.sortino.strategies[strategy].current;
                    sortinoCell.textContent = value.toFixed(2);
                    sortinoCell.className = value >= 1.5 ? 'positive' : 'negative';
                } else {
                    sortinoCell.textContent = 'N/A';
                }
                
                // Calmar ratio
                const calmarCell = document.createElement('td');
                if (this.ratioData.calmar.strategies[strategy]) {
                    const value = this.ratioData.calmar.strategies[strategy].current;
                    calmarCell.textContent = value.toFixed(2);
                    calmarCell.className = value >= 0.5 ? 'positive' : 'negative';
                } else {
                    calmarCell.textContent = 'N/A';
                }
                
                // Max drawdown
                const drawdownCell = document.createElement('td');
                if (this.performanceData.strategies[strategy]) {
                    const currentDrawdown = this.performanceData.strategies[strategy].currentDrawdown;
                    const maxDrawdown = this.performanceData.strategies[strategy].maxDrawdown;
                    drawdownCell.textContent = `${currentDrawdown.toFixed(2)}% / ${maxDrawdown.toFixed(2)}%`;
                    drawdownCell.className = currentDrawdown < 10 ? 'positive' : 
                                          currentDrawdown < 20 ? 'neutral' : 'negative';
                } else {
                    drawdownCell.textContent = 'N/A';
                }
                
                // Add cells to row
                row.appendChild(nameCell);
                row.appendChild(sharpeCell);
                row.appendChild(sortinoCell);
                row.appendChild(calmarCell);
                row.appendChild(drawdownCell);
                
                // Add row to table
                tableBody.appendChild(row);
            });
        }
    }

    // Data generation and update functions
    generateMockPerformanceData() {
        // Generate mock performance data
        const data = {
            portfolio: {
                returns: [],
                drawdowns: [],
                currentDrawdown: 5.2,
                maxDrawdown: 12.8
            },
            strategies: {
                'Trend Following': {
                    returns: [],
                    drawdowns: [],
                    currentDrawdown: 3.5,
                    maxDrawdown: 15.4
                },
                'Mean Reversion': {
                    returns: [],
                    drawdowns: [],
                    currentDrawdown: 7.8,
                    maxDrawdown: 10.2
                },
                'Statistical Arbitrage': {
                    returns: [],
                    drawdowns: [],
                    currentDrawdown: 2.1,
                    maxDrawdown: 8.7
                },
                'Sentiment Based': {
                    returns: [],
                    drawdowns: [],
                    currentDrawdown: 12.5,
                    maxDrawdown: 24.6
                }
            }
        };
        
        // Generate dates
        const now = new Date();
        const days = this.getTimeframeDays();
        
        // Generate daily data points
        for (let i = 0; i <= days; i++) {
            const date = new Date(now.getTime() - ((days - i) * 24 * 60 * 60 * 1000));
            
            // Portfolio data
            data.portfolio.returns.push({
                date: date,
                value: this.generateRandomReturn(i, 0.1)
            });
            
            data.portfolio.drawdowns.push({
                date: date,
                value: this.generateRandomDrawdown(i, 0.08)
            });
            
            // Strategy data
            Object.keys(data.strategies).forEach(strategy => {
                // Different biases for different strategies
                let returnBias = 0;
                let drawdownBias = 0;
                
                switch (strategy) {
                    case 'Trend Following':
                        returnBias = 0.15;
                        drawdownBias = -0.02;
                        break;
                    case 'Mean Reversion':
                        returnBias = 0.1;
                        drawdownBias = 0.03;
                        break;
                    case 'Statistical Arbitrage':
                        returnBias = 0.2;
                        drawdownBias = -0.05;
                        break;
                    case 'Sentiment Based':
                        returnBias = 0.05;
                        drawdownBias = 0.08;
                        break;
                }
                
                data.strategies[strategy].returns.push({
                    date: date,
                    value: this.generateRandomReturn(i, 0.15, returnBias)
                });
                
                data.strategies[strategy].drawdowns.push({
                    date: date,
                    value: this.generateRandomDrawdown(i, 0.12, drawdownBias)
                });
            });
        }
        
        return data;
    }

    generateMockRatioData(ratioType) {
        // Generate mock ratio data
        const data = {
            portfolio: {
                current: this.getRandomRatioValue(ratioType),
                history: []
            },
            strategies: {
                'Trend Following': {
                    current: this.getRandomRatioValue(ratioType, 0.2),
                    history: []
                },
                'Mean Reversion': {
                    current: this.getRandomRatioValue(ratioType, 0.1),
                    history: []
                },
                'Statistical Arbitrage': {
                    current: this.getRandomRatioValue(ratioType, 0.3),
                    history: []
                },
                'Sentiment Based': {
                    current: this.getRandomRatioValue(ratioType, -0.1),
                    history: []
                }
            }
        };
        
        // Generate dates
        const now = new Date();
        const days = this.getTimeframeDays();
        
        // Generate daily data points
        for (let i = 0; i <= days; i++) {
            const date = new Date(now.getTime() - ((days - i) * 24 * 60 * 60 * 1000));
            
            // Portfolio history
            data.portfolio.history.push({
                date: date,
                value: this.generateRandomRatioValue(i, ratioType)
            });
            
            // Strategy history
            Object.keys(data.strategies).forEach(strategy => {
                // Different biases for different strategies
                let bias = 0;
                
                switch (strategy) {
                    case 'Trend Following':
                        bias = 0.2;
                        break;
                    case 'Mean Reversion':
                        bias = 0.1;
                        break;
                    case 'Statistical Arbitrage':
                        bias = 0.3;
                        break;
                    case 'Sentiment Based':
                        bias = -0.1;
                        break;
                }
                
                data.strategies[strategy].history.push({
                    date: date,
                    value: this.generateRandomRatioValue(i, ratioType, bias)
                });
            });
        }
        
        return data;
    }

    generateMockRiskRewardData() {
        // Generate mock risk-reward surface data
        const data = {};
        
        // Generate data for each metric
        ['sharpe', 'sortino', 'calmar'].forEach(metric => {
            // Risk levels
            const riskLevels = Array.from({length: 10}, (_, i) => i * 0.5 + 1);
            
            // Return levels
            const returnLevels = Array.from({length: 10}, (_, i) => i * 1 + 2);
            
            // Values matrix
            const values = [];
            let maxValue = 0;
            let optimalRisk = 0;
            let optimalReturn = 0;
            
            // Each row
            riskLevels.forEach((risk, rIdx) => {
                const row = [];
                
                // Each column
                returnLevels.forEach((ret, cIdx) => {
                    // Calculate metric value based on risk and return
                    let value;
                    
                    if (metric === 'sharpe') {
                        // Sharpe = (return - risk-free) / risk
                        const riskFree = 0.5;
                        value = (ret - riskFree) / risk;
                    } else if (metric === 'sortino') {
                        // Sortino = (return - risk-free) / downside-risk
                        const riskFree = 0.5;
                        const downsideRisk = risk * 0.7; // Assuming 70% of risk is downside
                        value = (ret - riskFree) / downsideRisk;
                    } else if (metric === 'calmar') {
                        // Calmar = annualized return / max drawdown
                        const maxDrawdown = risk * 1.5; // Assuming max drawdown is 150% of risk
                        value = ret / maxDrawdown;
                    }
                    
                    // Add some randomness
                    value *= (0.8 + Math.random() * 0.4);
                    
                    // Track maximum value
                    if (value > maxValue) {
                        maxValue = value;
                        optimalRisk = risk;
                        optimalReturn = ret;
                    }
                    
                    row.push(value);
                });
                
                values.push(row);
            });
            
            data[metric] = {
                riskLevels: riskLevels,
                returnLevels: returnLevels,
                values: values,
                optimalRisk: optimalRisk,
                optimalReturn: optimalReturn,
                optimalValue: maxValue
            };
        });
        
        return data;
    }

    generateRandomReturn(day, volatility = 0.1, bias = 0) {
        // Base pattern (sine wave)
        const base = Math.sin(day / 20) * 5;
        
        // Add some randomness
        const random = (Math.random() - 0.5) * volatility * 30;
        
        // Add bias and return
        return base + random + (bias * 15);
    }

    generateRandomDrawdown(day, volatility = 0.08, bias = 0) {
        // Base pattern for drawdown (lower values are better)
        const base = Math.abs(Math.sin(day / 30)) * 8;
        
        // Add some randomness
        const random = Math.random() * volatility * 20;
        
        // Add bias and return (always positive)
        return Math.max(0, base + random + (bias * 10));
    }

    generateRandomRatioValue(day, ratioType, bias = 0) {
        // Base ratio depends on type
        let base;
        let volatility;
        
        switch (ratioType) {
            case 'sharpe':
                base = 1.2;
                volatility = 0.4;
                break;
            case 'sortino':
                base = 1.5;
                volatility = 0.5;
                break;
            case 'calmar':
                base = 0.8;
                volatility = 0.3;
                break;
            case 'mar':
                base = 0.5;
                volatility = 0.2;
                break;
            case 'treynor':
                base = 0.6;
                volatility = 0.25;
                break;
            default:
                base = 1.0;
                volatility = 0.3;
        }
        
        // Add some seasonal pattern
        const seasonal = Math.sin(day / 30) * 0.3;
        
        // Add some randomness
        const random = (Math.random() - 0.5) * volatility;
        
        // Add bias and return
        return base + seasonal + random + bias;
    }

    updatePerformanceData() {
        // Update performance data with new points
        const now = new Date();
        
        // Update portfolio data
        if (this.performanceData.portfolio) {
            // Get last data point
            const lastReturn = this.performanceData.portfolio.returns[this.performanceData.portfolio.returns.length - 1];
            const lastDrawdown = this.performanceData.portfolio.drawdowns[this.performanceData.portfolio.drawdowns.length - 1];
            
            // Add new data points
            this.performanceData.portfolio.returns.push({
                date: now,
                value: lastReturn.value + (Math.random() - 0.5) * 2
            });
            
            this.performanceData.portfolio.drawdowns.push({
                date: now,
                value: Math.max(0, lastDrawdown.value + (Math.random() - 0.5) * 1)
            });
            
            // Remove oldest points
            this.performanceData.portfolio.returns.shift();
            this.performanceData.portfolio.drawdowns.shift();
            
            // Update current drawdown
            this.performanceData.portfolio.currentDrawdown = 
                this.performanceData.portfolio.drawdowns[this.performanceData.portfolio.drawdowns.length - 1].value;
        }
        
        // Update strategy data
        if (this.performanceData.strategies) {
            Object.keys(this.performanceData.strategies).forEach(strategy => {
                // Get last data points
                const lastReturn = this.performanceData.strategies[strategy].returns[this.performanceData.strategies[strategy].returns.length - 1];
                const lastDrawdown = this.performanceData.strategies[strategy].drawdowns[this.performanceData.strategies[strategy].drawdowns.length - 1];
                
                // Add new data points
                this.performanceData.strategies[strategy].returns.push({
                    date: now,
                    value: lastReturn.value + (Math.random() - 0.5) * 3
                });
                
                this.performanceData.strategies[strategy].drawdowns.push({
                    date: now,
                    value: Math.max(0, lastDrawdown.value + (Math.random() - 0.5) * 1.5)
                });
                
                // Remove oldest points
                this.performanceData.strategies[strategy].returns.shift();
                this.performanceData.strategies[strategy].drawdowns.shift();
                
                // Update current drawdown
                this.performanceData.strategies[strategy].currentDrawdown = 
                    this.performanceData.strategies[strategy].drawdowns[this.performanceData.strategies[strategy].drawdowns.length - 1].value;
            });
        }
    }

    updateRatioData() {
        // Update ratio data with new values
        const now = new Date();
        
        // Update each ratio type
        Object.keys(this.ratioData).forEach(ratioType => {
            // Update portfolio data
            if (this.ratioData[ratioType].portfolio) {
                // Get last data point
                const lastPoint = this.ratioData[ratioType].portfolio.history[this.ratioData[ratioType].portfolio.history.length - 1];
                
                // Add new data point
                const newValue = Math.max(0, lastPoint.value + (Math.random() - 0.5) * 0.2);
                this.ratioData[ratioType].portfolio.history.push({
                    date: now,
                    value: newValue
                });
                
                // Remove oldest point
                this.ratioData[ratioType].portfolio.history.shift();
                
                // Update current value
                this.ratioData[ratioType].portfolio.current = newValue;
            }
            
            // Update strategy data
            if (this.ratioData[ratioType].strategies) {
                Object.keys(this.ratioData[ratioType].strategies).forEach(strategy => {
                    // Get last data point
                    const lastPoint = this.ratioData[ratioType].strategies[strategy].history[this.ratioData[ratioType].strategies[strategy].history.length - 1];
                    
                    // Add new data point
                    const newValue = Math.max(0, lastPoint.value + (Math.random() - 0.5) * 0.3);
                    this.ratioData[ratioType].strategies[strategy].history.push({
                        date: now,
                        value: newValue
                    });
                    
                    // Remove oldest point
                    this.ratioData[ratioType].strategies[strategy].history.shift();
                    
                    // Update current value
                    this.ratioData[ratioType].strategies[strategy].current = newValue;
                });
            }
        });
    }

    // Helper functions
    getTimeframeDays() {
        // Convert timeframe to days
        switch (this.timeframe) {
            case '1w': return 7;
            case '1m': return 30;
            case '3m': return 90;
            case '6m': return 180;
            case '1y': return 365;
            default: return 30;
        }
    }

    formatTimeframe(timeframe) {
        // Format timeframe for display
        switch (timeframe) {
            case '1w': return '1 Week';
            case '1m': return '1 Month';
            case '3m': return '3 Months';
            case '6m': return '6 Months';
            case '1y': return '1 Year';
            default: return timeframe;
        }
    }

    formatRatioName(ratio) {
        // Format ratio name for display
        switch (ratio) {
            case 'sharpe': return 'Sharpe';
            case 'sortino': return 'Sortino';
            case 'calmar': return 'Calmar';
            case 'mar': return 'MAR';
            case 'treynor': return 'Treynor';
            default: return ratio.charAt(0).toUpperCase() + ratio.slice(1);
        }
    }

    getMetricTarget(metric) {
        // Get target value for each metric
        switch (metric) {
            case 'sharpe': return 1.0;
            case 'sortino': return 1.5;
            case 'calmar': return 0.5;
            case 'mar': return 0.3;
            case 'treynor': return 0.5;
            default: return 1.0;
        }
    }

    getMetricDescription(metric) {
        // Get description for each metric
        switch (metric) {
            case 'sharpe':
                return 'Measures excess return per unit of risk (volatility). Higher is better.';
            case 'sortino':
                return 'Similar to Sharpe, but only considers downside risk. Higher is better.';
            case 'calmar':
                return 'Measures return relative to maximum drawdown. Higher is better.';
            case 'mar':
                return 'Annualized return divided by Maximum Drawdown. Higher is better.';
            case 'treynor':
                return 'Measures excess return per unit of systematic risk (beta). Higher is better.';
            default:
                return 'Risk-adjusted performance metric.';
        }
    }

    showLoading(element) {
        // Show loading overlay on charts
        const elements = element ? [element] : [
            document.getElementById(this.options.ratioChartElementId),
            document.getElementById(this.options.evolutionChartElementId),
            document.getElementById(this.options.surfaceChartElementId),
            document.getElementById(this.options.drawdownChartElementId)
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
            document.getElementById(this.options.ratioChartElementId),
            document.getElementById(this.options.evolutionChartElementId),
            document.getElementById(this.options.surfaceChartElementId),
            document.getElementById(this.options.drawdownChartElementId)
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
            document.getElementById(this.options.ratioChartElementId),
            document.getElementById(this.options.evolutionChartElementId),
            document.getElementById(this.options.surfaceChartElementId),
            document.getElementById(this.options.drawdownChartElementId)
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
            timeframe: this.timeframe,
            ratioType: this.ratioType,
            surfaceMetric: this.surfaceMetric,
            timestamp: new Date(),
            performanceData: this.performanceData,
            ratioData: this.ratioData,
            riskRewardData: this.riskRewardData
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `risk_adjusted_metrics_${new Date().toISOString().split('T')[0]}.json`;
        
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
    // Create instance of RiskAdjustedMetrics
    const riskMetrics = new RiskAdjustedMetrics();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});