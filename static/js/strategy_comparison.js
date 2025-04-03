/**
 * Strategy Performance Comparison Module
 * 
 * This module provides functionality for comparing multiple trading strategies,
 * analyzing their performance metrics, and visualizing performance attribution.
 */

class StrategyComparison {
    constructor(options = {}) {
        this.options = Object.assign({
            // Default options
            defaultPeriod: '1y',
            defaultMetric: 'returns',
            defaultBenchmark: 'BTC-USD',
            chartContainer: 'performance-chart',
            attributionContainer: 'attribution-chart',
            distributionContainer: 'distribution-chart',
            strategies: [
                { id: 'strategy1', name: 'MA Crossover', color: 'var(--primary)', active: true },
                { id: 'strategy2', name: 'Sentiment-based', color: 'var(--success)', active: true }
            ],
            showBenchmark: true,
            apiEndpoint: '/api/strategy-performance'
        }, options);
        
        // State management
        this.state = {
            period: this.options.defaultPeriod,
            metric: this.options.defaultMetric,
            benchmark: this.options.defaultBenchmark,
            strategies: [...this.options.strategies],
            showBenchmark: this.options.showBenchmark,
            chartsInitialized: false
        };
        
        // Chart instances
        this.charts = {
            performance: null,
            attribution: null,
            distribution: null
        };
    }
    
    /**
     * Initialize the strategy comparison module
     */
    init() {
        console.log('Initializing strategy comparison module');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initCharts();
        
        return this;
    }
    
    /**
     * Set up event listeners for controls
     */
    setupEventListeners() {
        // Period selector
        const periodSelect = document.getElementById('comparison-period');
        if (periodSelect) {
            periodSelect.value = this.state.period;
            periodSelect.addEventListener('change', (e) => {
                this.state.period = e.target.value;
                this.updatePerformanceChart();
            });
        }
        
        // Metric selector
        const metricSelect = document.getElementById('comparison-metric');
        if (metricSelect) {
            metricSelect.value = this.state.metric;
            metricSelect.addEventListener('change', (e) => {
                this.state.metric = e.target.value;
                this.updatePerformanceChart();
            });
        }
        
        // Benchmark selector
        const benchmarkSelect = document.getElementById('comparison-benchmark');
        if (benchmarkSelect) {
            benchmarkSelect.value = this.state.benchmark;
            benchmarkSelect.addEventListener('change', (e) => {
                this.state.benchmark = e.target.value;
                this.updateBenchmark();
            });
        }
        
        // Add strategy buttons
        const addStrategyButtons = [
            document.getElementById('add-strategy-btn'),
            document.getElementById('add-strategy-inline')
        ];
        
        addStrategyButtons.forEach(button => {
            if (button) {
                button.addEventListener('click', () => {
                    this.showStrategyModal();
                });
            }
        });
        
        // Add strategy confirmation
        const addStrategyConfirm = document.getElementById('add-strategy-confirm');
        if (addStrategyConfirm) {
            addStrategyConfirm.addEventListener('click', () => {
                this.addStrategy();
            });
        }
        
        // Close modal buttons
        const closeModalButtons = document.querySelectorAll('[data-dismiss="modal"]');
        closeModalButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.hideStrategyModal();
            });
        });
        
        // Strategy badges
        this.setupStrategyToggleListeners();
        
        // Remove strategy buttons
        this.setupRemoveStrategyListeners();
        
        // Benchmark toggle
        const benchmarkToggleBtn = document.getElementById('benchmark-toggle-btn');
        if (benchmarkToggleBtn) {
            benchmarkToggleBtn.addEventListener('click', () => {
                this.toggleBenchmark();
            });
        }
        
        // Expand panel button
        const expandButton = document.getElementById('expand-comparison-btn');
        if (expandButton) {
            expandButton.addEventListener('click', () => {
                this.toggleFullscreen();
            });
        }
    }
    
    /**
     * Set up strategy toggle listeners
     */
    setupStrategyToggleListeners() {
        const strategyBadges = document.querySelectorAll('.strategy-badge');
        strategyBadges.forEach(badge => {
            badge.addEventListener('click', () => {
                const strategyId = badge.dataset.strategy;
                this.toggleStrategy(strategyId);
            });
        });
    }
    
    /**
     * Set up remove strategy listeners
     */
    setupRemoveStrategyListeners() {
        const removeButtons = document.querySelectorAll('.remove-strategy');
        removeButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const strategyRow = button.closest('.strategy-row');
                if (strategyRow) {
                    this.removeStrategy(strategyRow.dataset.strategy);
                }
            });
        });
    }
    
    /**
     * Initialize charts
     */
    initCharts() {
        if (this.state.chartsInitialized) {
            return;
        }
        
        this.renderPerformanceChart();
        this.renderAttributionChart();
        this.renderDistributionChart();
        
        this.state.chartsInitialized = true;
    }
    
    /**
     * Render the main performance comparison chart
     */
    renderPerformanceChart() {
        const container = document.getElementById(this.options.chartContainer);
        if (!container) {
            console.error(`Chart container ${this.options.chartContainer} not found`);
            return;
        }
        
        // Show loading
        this.showChartLoading(this.options.chartContainer);
        
        // Fetch data
        this.fetchPerformanceData(this.state.period, this.state.metric)
            .then(data => {
                // Hide loading
                this.hideChartLoading(this.options.chartContainer);
                
                // Create traces for each active strategy
                const traces = [];
                
                this.state.strategies.forEach(strategy => {
                    if (strategy.active) {
                        traces.push({
                            type: 'scatter',
                            mode: 'lines',
                            name: strategy.name,
                            x: data.dates,
                            y: data[strategy.id],
                            line: {
                                color: strategy.color,
                                width: 2
                            }
                        });
                    }
                });
                
                // Add benchmark if enabled
                if (this.state.showBenchmark && this.state.benchmark !== 'NONE') {
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        name: `${this.state.benchmark} (Benchmark)`,
                        x: data.dates,
                        y: data.benchmark,
                        line: {
                            color: 'var(--text-light)',
                            width: 1.5,
                            dash: 'dot'
                        }
                    });
                }
                
                // Create layout
                const layout = this.createPerformanceChartLayout();
                
                // Create config
                const config = {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d', 'toggleSpikelines'],
                    displaylogo: false
                };
                
                // Render chart
                if (typeof Plotly !== 'undefined') {
                    Plotly.newPlot(this.options.chartContainer, traces, layout, config);
                    this.charts.performance = this.options.chartContainer;
                } else {
                    console.error('Plotly is not available');
                }
            })
            .catch(error => {
                console.error(`Error fetching performance data: ${error}`);
                this.showChartError(this.options.chartContainer, 'Failed to load performance data');
            });
    }
    
    /**
     * Create layout for performance chart
     */
    createPerformanceChartLayout() {
        return {
            title: {
                text: this.getChartTitle(),
                font: {
                    size: 14,
                    color: 'var(--text-light)'
                }
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.2
            },
            xaxis: {
                title: '',
                showgrid: true,
                gridcolor: 'var(--border-light)',
                gridwidth: 1,
                autorange: true
            },
            yaxis: {
                title: this.getYAxisTitle(),
                showgrid: true,
                gridcolor: 'var(--border-light)',
                gridwidth: 1,
                autorange: true,
                tickformat: this.getTickFormat()
            },
            margin: {
                l: 50,
                r: 20,
                t: 40,
                b: 40
            },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 11
            },
            hovermode: 'x unified',
            hoverlabel: {
                bgcolor: 'var(--tooltip-bg)',
                bordercolor: 'var(--tooltip-border)',
                font: {
                    color: 'var(--tooltip-text)',
                    size: 11
                }
            }
        };
    }
    
    /**
     * Get chart title based on current selections
     */
    getChartTitle() {
        const periodTexts = {
            '1w': '1 Week',
            '1m': '1 Month',
            '3m': '3 Months',
            '6m': '6 Months',
            '1y': '1 Year',
            'ytd': 'Year to Date',
            'all': 'All Time'
        };
        
        const metricTexts = {
            'returns': 'Performance Comparison',
            'drawdown': 'Drawdown Comparison',
            'sharpe': 'Sharpe Ratio Comparison',
            'sortino': 'Sortino Ratio Comparison',
            'volatility': 'Volatility Comparison',
            'win-rate': 'Win Rate Comparison',
            'profit-factor': 'Profit Factor Comparison'
        };
        
        return `${metricTexts[this.state.metric] || 'Performance'} - ${periodTexts[this.state.period] || this.state.period}`;
    }
    
    /**
     * Get Y-axis title based on selected metric
     */
    getYAxisTitle() {
        const titles = {
            'returns': 'Cumulative Returns',
            'drawdown': 'Drawdown',
            'sharpe': 'Sharpe Ratio',
            'sortino': 'Sortino Ratio',
            'volatility': 'Volatility',
            'win-rate': 'Win Rate',
            'profit-factor': 'Profit Factor'
        };
        
        return titles[this.state.metric] || '';
    }
    
    /**
     * Get tick format based on selected metric
     */
    getTickFormat() {
        if (this.state.metric === 'returns' || this.state.metric === 'drawdown' || this.state.metric === 'volatility' || this.state.metric === 'win-rate') {
            return '.1%'; // Percentage format
        }
        
        return '.2f'; // Default decimal format
    }
    
    /**
     * Render the attribution chart
     */
    renderAttributionChart() {
        const container = document.getElementById(this.options.attributionContainer);
        if (!container) {
            console.error(`Attribution chart container ${this.options.attributionContainer} not found`);
            return;
        }
        
        // Show loading
        this.showChartLoading(this.options.attributionContainer);
        
        // Fetch attribution data
        this.fetchAttributionData()
            .then(data => {
                // Hide loading
                this.hideChartLoading(this.options.attributionContainer);
                
                // Create trace
                const trace = {
                    type: 'bar',
                    x: data.factors,
                    y: data.contributions,
                    marker: {
                        color: data.colors || [
                            'rgba(var(--primary-rgb), 0.7)',
                            'rgba(var(--success-rgb), 0.7)',
                            'rgba(var(--info-rgb), 0.7)',
                            'rgba(var(--warning-rgb), 0.7)',
                            'rgba(var(--danger-rgb), 0.7)',
                            'rgba(var(--purple-rgb), 0.7)',
                            'rgba(var(--secondary-rgb), 0.7)'
                        ]
                    },
                    name: 'Contribution (%)'
                };
                
                // Create layout
                const layout = {
                    autosize: true,
                    margin: { t: 10, l: 50, r: 10, b: 60 },
                    paper_bgcolor: 'var(--card-bg)',
                    plot_bgcolor: 'var(--card-bg)',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    xaxis: {
                        tickangle: -45
                    },
                    yaxis: {
                        title: 'Contribution (%)',
                        ticksuffix: '%'
                    },
                    showlegend: false
                };
                
                // Create config
                const config = {
                    responsive: true,
                    displayModeBar: false
                };
                
                // Render chart
                if (typeof Plotly !== 'undefined') {
                    Plotly.newPlot(this.options.attributionContainer, [trace], layout, config);
                    this.charts.attribution = this.options.attributionContainer;
                } else {
                    console.error('Plotly is not available');
                }
            })
            .catch(error => {
                console.error(`Error fetching attribution data: ${error}`);
                this.showChartError(this.options.attributionContainer, 'Failed to load attribution data');
            });
    }
    
    /**
     * Render the distribution chart
     */
    renderDistributionChart() {
        const container = document.getElementById(this.options.distributionContainer);
        if (!container) {
            console.error(`Distribution chart container ${this.options.distributionContainer} not found`);
            return;
        }
        
        // Show loading
        this.showChartLoading(this.options.distributionContainer);
        
        // Fetch distribution data
        this.fetchDistributionData()
            .then(data => {
                // Hide loading
                this.hideChartLoading(this.options.distributionContainer);
                
                // Create traces
                const traces = [
                    {
                        x: data.profitable,
                        type: 'histogram',
                        opacity: 0.7,
                        marker: {
                            color: 'rgba(var(--success-rgb), 0.7)'
                        },
                        name: 'Profitable Trades'
                    },
                    {
                        x: data.losses,
                        type: 'histogram',
                        opacity: 0.7,
                        marker: {
                            color: 'rgba(var(--danger-rgb), 0.7)'
                        },
                        name: 'Loss-making Trades'
                    }
                ];
                
                // Create layout
                const layout = {
                    autosize: true,
                    margin: { t: 10, l: 50, r: 10, b: 40 },
                    paper_bgcolor: 'var(--card-bg)',
                    plot_bgcolor: 'var(--card-bg)',
                    barmode: 'overlay',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    legend: {
                        orientation: 'h',
                        y: -0.2
                    },
                    xaxis: {
                        title: 'Return (%)',
                        tickformat: '.0%'
                    },
                    yaxis: {
                        title: 'Frequency'
                    }
                };
                
                // Create config
                const config = {
                    responsive: true,
                    displayModeBar: false
                };
                
                // Render chart
                if (typeof Plotly !== 'undefined') {
                    Plotly.newPlot(this.options.distributionContainer, traces, layout, config);
                    this.charts.distribution = this.options.distributionContainer;
                } else {
                    console.error('Plotly is not available');
                }
            })
            .catch(error => {
                console.error(`Error fetching distribution data: ${error}`);
                this.showChartError(this.options.distributionContainer, 'Failed to load distribution data');
            });
    }
    
    /**
     * Fetch performance data
     * @param {string} period - Time period to fetch data for
     * @param {string} metric - Metric to display
     * @returns {Promise} - Promise resolving to performance data
     */
    fetchPerformanceData(period, metric) {
        // In a production environment, this would make an API call
        // For now, we'll use mock data
        return new Promise((resolve) => {
            setTimeout(() => {
                const data = this.generatePerformanceData(period, metric);
                resolve(data);
            }, 500);
        });
    }
    
    /**
     * Fetch attribution data
     * @returns {Promise} - Promise resolving to attribution data
     */
    fetchAttributionData() {
        // In a production environment, this would make an API call
        // For now, we'll use mock data
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    factors: ['Market', 'Technical', 'Sentiment', 'Momentum', 'Volatility', 'Timing', 'Alpha'],
                    contributions: [12.5, 8.3, 7.2, 6.8, 4.5, 3.2, 2.5],
                    colors: [
                        'rgba(var(--primary-rgb), 0.7)',
                        'rgba(var(--success-rgb), 0.7)',
                        'rgba(var(--info-rgb), 0.7)',
                        'rgba(var(--warning-rgb), 0.7)',
                        'rgba(var(--danger-rgb), 0.7)',
                        'rgba(var(--purple-rgb), 0.7)',
                        'rgba(var(--secondary-rgb), 0.7)'
                    ]
                });
            }, 500);
        });
    }
    
    /**
     * Fetch distribution data
     * @returns {Promise} - Promise resolving to distribution data
     */
    fetchDistributionData() {
        // In a production environment, this would make an API call
        // For now, we'll use mock data
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    profitable: this.generateDistributionData(0.5, 0.2, 100),
                    losses: this.generateDistributionData(-0.3, 0.15, 50)
                });
            }, 500);
        });
    }
    
    /**
     * Generate mock performance data
     */
    generatePerformanceData(period, metric) {
        // Get date range based on period
        const end = new Date();
        let start = new Date();
        let points = 0;
        
        switch (period) {
            case '1w':
                start.setDate(end.getDate() - 7);
                points = 7;
                break;
            case '1m':
                start.setMonth(end.getMonth() - 1);
                points = 30;
                break;
            case '3m':
                start.setMonth(end.getMonth() - 3);
                points = 90;
                break;
            case '6m':
                start.setMonth(end.getMonth() - 6);
                points = 180;
                break;
            case '1y':
                start.setFullYear(end.getFullYear() - 1);
                points = 365;
                break;
            case 'ytd':
                start = new Date(end.getFullYear(), 0, 1);
                points = Math.floor((end - start) / (1000 * 60 * 60 * 24));
                break;
            case 'all':
                start.setFullYear(end.getFullYear() - 3);
                points = 1095;
                break;
            default:
                start.setFullYear(end.getFullYear() - 1);
                points = 365;
        }
        
        // Generate dates
        const dates = [];
        const msPerDay = 24 * 60 * 60 * 1000;
        for (let i = 0; i < points; i++) {
            dates.push(new Date(start.getTime() + i * msPerDay));
        }
        
        // Performance is different based on metric
        if (metric === 'returns') {
            // Generate strategy and benchmark cumulative returns
            const strategy1 = this.generateCumulativeReturns(1.0, points, 0.08, 0.15);
            const strategy2 = this.generateCumulativeReturns(1.0, points, 0.06, 0.12);
            const benchmark = this.generateCumulativeReturns(1.0, points, 0.05, 0.20);
            
            return {
                dates,
                strategy1,
                strategy2,
                benchmark
            };
        } else if (metric === 'drawdown') {
            // Generate drawdown data
            const strategy1 = this.generateDrawdownData(points, 0.15);
            const strategy2 = this.generateDrawdownData(points, 0.20);
            const benchmark = this.generateDrawdownData(points, 0.25);
            
            return {
                dates,
                strategy1,
                strategy2,
                benchmark
            };
        } else {
            // Default to returns
            const strategy1 = this.generateCumulativeReturns(1.0, points, 0.08, 0.15);
            const strategy2 = this.generateCumulativeReturns(1.0, points, 0.06, 0.12);
            const benchmark = this.generateCumulativeReturns(1.0, points, 0.05, 0.20);
            
            return {
                dates,
                strategy1,
                strategy2,
                benchmark
            };
        }
    }
    
    /**
     * Generate cumulative returns data
     */
    generateCumulativeReturns(initial, points, meanReturn, volatility) {
        const values = [initial];
        let current = initial;
        
        for (let i = 1; i < points; i++) {
            // Random daily return with specified mean and volatility
            const dailyReturn = (Math.random() - 0.5) * volatility + meanReturn / 365;
            current = current * (1 + dailyReturn);
            values.push(current);
        }
        
        return values;
    }
    
    /**
     * Generate drawdown data
     */
    generateDrawdownData(points, maxDrawdown) {
        const values = [0];
        let current = 0;
        
        for (let i = 1; i < points; i++) {
            // Random walk with slight mean reversion, bounded by 0 and maxDrawdown
            const change = (Math.random() - 0.5) * 0.01 - 0.0002 * (current / maxDrawdown);
            current = Math.max(Math.min(current + change, 0), -maxDrawdown);
            values.push(current);
        }
        
        return values;
    }
    
    /**
     * Generate distribution data
     */
    generateDistributionData(mean, stdDev, count) {
        const values = [];
        
        for (let i = 0; i < count; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            
            // Apply mean and standard deviation
            const value = mean + z0 * stdDev;
            values.push(value);
        }
        
        return values;
    }
    
    /**
     * Show loading indicator for a chart
     */
    showChartLoading(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        let overlay = container.querySelector('.chart-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'chart-overlay';
            overlay.textContent = 'Loading...';
            container.appendChild(overlay);
        } else {
            overlay.textContent = 'Loading...';
            overlay.style.display = 'flex';
        }
    }
    
    /**
     * Hide loading indicator for a chart
     */
    hideChartLoading(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    /**
     * Show error message for a chart
     */
    showChartError(containerId, message) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        let overlay = container.querySelector('.chart-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'chart-overlay';
            container.appendChild(overlay);
        }
        
        overlay.textContent = message;
        overlay.style.display = 'flex';
        overlay.classList.add('error');
    }
    
    /**
     * Show the add strategy modal
     */
    showStrategyModal() {
        const modal = document.getElementById('strategy-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }
    
    /**
     * Hide the add strategy modal
     */
    hideStrategyModal() {
        const modal = document.getElementById('strategy-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    /**
     * Add a new strategy for comparison
     */
    addStrategy() {
        const strategySelect = document.getElementById('strategy-type');
        const strategyName = strategySelect.options[strategySelect.selectedIndex].text;
        const strategyColor = document.getElementById('strategy-color').value;
        
        // Generate a unique ID
        const strategyId = 'strategy' + Date.now();
        
        // Add to state
        this.state.strategies.push({
            id: strategyId,
            name: strategyName,
            color: strategyColor,
            active: true
        });
        
        // Generate mock return
        const mockReturn = (Math.random() * 30 + 10).toFixed(1);
        
        // Add to UI
        this.addStrategyToUI(strategyId, strategyName, strategyColor, mockReturn);
        
        // Update charts
        this.updatePerformanceChart();
        
        // Hide modal
        this.hideStrategyModal();
    }
    
    /**
     * Add strategy to UI elements
     */
    addStrategyToUI(strategyId, strategyName, strategyColor, returnValue) {
        // Add to badges
        const badgesContainer = document.querySelector('.strategy-badges');
        if (badgesContainer) {
            const badge = document.createElement('span');
            badge.className = 'strategy-badge active';
            badge.dataset.strategy = strategyId;
            badge.innerHTML = `
                ${strategyName}
                <span class="badge positive">+${returnValue}%</span>
            `;
            
            // Add before add-strategy button
            const addButton = document.getElementById('add-strategy-inline');
            if (addButton) {
                badgesContainer.insertBefore(badge, addButton);
            } else {
                badgesContainer.appendChild(badge);
            }
            
            // Add event listener
            badge.addEventListener('click', () => {
                this.toggleStrategy(strategyId);
            });
        }
        
        // Add to table
        const metricsTable = document.querySelector('#metrics-table tbody');
        if (metricsTable) {
            const row = document.createElement('tr');
            row.className = 'strategy-row';
            row.dataset.strategy = strategyId;
            row.innerHTML = `
                <td class="strategy-name">
                    <span class="color-dot" style="background-color: ${strategyColor};"></span>
                    ${strategyName}
                </td>
                <td class="returns positive">+${returnValue}%</td>
                <td class="drawdown">-${(Math.random() * 20 + 5).toFixed(1)}%</td>
                <td class="sharpe">${(Math.random() * 1 + 0.8).toFixed(2)}</td>
                <td class="win-rate">${Math.floor(Math.random() * 20 + 50)}%</td>
                <td class="trades">${Math.floor(Math.random() * 100 + 50)}</td>
                <td class="actions">
                    <button class="btn btn-icon btn-sm" data-tooltip="View details" data-position="left">
                        <i data-feather="eye"></i>
                    </button>
                    <button class="btn btn-icon btn-sm remove-strategy" data-tooltip="Remove" data-position="left">
                        <i data-feather="x"></i>
                    </button>
                </td>
            `;
            
            // Add to table
            metricsTable.appendChild(row);
            
            // Initialize Feather icons
            if (typeof feather !== 'undefined') {
                feather.replace();
            }
            
            // Add remove event listener
            const removeButton = row.querySelector('.remove-strategy');
            if (removeButton) {
                removeButton.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.removeStrategy(strategyId);
                });
            }
        }
    }
    
    /**
     * Remove a strategy from comparison
     */
    removeStrategy(strategyId) {
        // Remove from state
        const index = this.state.strategies.findIndex(s => s.id === strategyId);
        if (index !== -1) {
            this.state.strategies.splice(index, 1);
        }
        
        // Remove from UI
        const badge = document.querySelector(`.strategy-badge[data-strategy="${strategyId}"]`);
        if (badge) {
            badge.remove();
        }
        
        const row = document.querySelector(`.strategy-row[data-strategy="${strategyId}"]`);
        if (row) {
            row.remove();
        }
        
        // Update chart
        this.updatePerformanceChart();
    }
    
    /**
     * Toggle a strategy's visibility
     */
    toggleStrategy(strategyId) {
        // Update state
        const strategy = this.state.strategies.find(s => s.id === strategyId);
        if (strategy) {
            strategy.active = !strategy.active;
        }
        
        // Update UI
        const badge = document.querySelector(`.strategy-badge[data-strategy="${strategyId}"]`);
        if (badge) {
            badge.classList.toggle('active');
        }
        
        const row = document.querySelector(`.strategy-row[data-strategy="${strategyId}"]`);
        if (row) {
            if (row.style.opacity === '0.5') {
                row.style.opacity = '1';
            } else {
                row.style.opacity = '0.5';
            }
        }
        
        // Update chart
        this.updatePerformanceChart();
    }
    
    /**
     * Toggle benchmark visibility
     */
    toggleBenchmark() {
        // Update state
        this.state.showBenchmark = !this.state.showBenchmark;
        
        // Update UI
        const badge = document.querySelector('.strategy-badge.benchmark');
        if (badge) {
            badge.classList.toggle('active');
        }
        
        const row = document.querySelector('.strategy-row.benchmark');
        if (row) {
            if (row.style.opacity === '0.5') {
                row.style.opacity = '1';
            } else {
                row.style.opacity = '0.5';
            }
        }
        
        // Update chart
        this.updatePerformanceChart();
    }
    
    /**
     * Update benchmark selection
     */
    updateBenchmark() {
        // Update UI
        const badge = document.querySelector('.strategy-badge.benchmark');
        const row = document.querySelector('.strategy-row.benchmark');
        
        if (this.state.benchmark === 'NONE') {
            // Hide benchmark
            if (badge) badge.style.display = 'none';
            if (row) row.style.display = 'none';
        } else {
            // Show and update benchmark
            if (badge) {
                badge.style.display = '';
                badge.textContent = this.state.benchmark + ' (Benchmark)';
            }
            
            if (row) {
                row.style.display = '';
                const nameTd = row.querySelector('.strategy-name');
                if (nameTd) {
                    nameTd.innerHTML = `
                        <span class="color-dot" style="background-color: var(--text-light);"></span>
                        ${this.state.benchmark} (Benchmark)
                    `;
                }
            }
        }
        
        // Update chart
        this.updatePerformanceChart();
    }
    
    /**
     * Update the performance chart (re-render)
     */
    updatePerformanceChart() {
        this.renderPerformanceChart();
    }
    
    /**
     * Toggle fullscreen mode for the panel
     */
    toggleFullscreen() {
        const panel = document.querySelector('.strategy-comparison-panel');
        if (!panel) return;
        
        if (!document.fullscreenElement) {
            panel.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable full-screen mode: ${err.message}`);
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if needed containers exist
    const performanceChart = document.getElementById('performance-chart');
    if (!performanceChart) {
        console.warn('Performance chart container not found, skipping strategy comparison initialization');
        return;
    }
    
    // Initialize the strategy comparison module
    const strategyComparison = new StrategyComparison();
    strategyComparison.init();
    
    // Make it available globally for debugging
    window.strategyComparison = strategyComparison;
});