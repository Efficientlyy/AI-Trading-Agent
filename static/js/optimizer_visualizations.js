/**
 * Optimizer Visualizations
 * 
 * This module provides visualization tools for optimization processes and results,
 * including parameter sensitivity analysis, performance surfaces, and optimization 
 * comparisons.
 */

// Configuration
const optimizerConfig = {
    chartColors: {
        primary: getComputedStyle(document.documentElement).getPropertyValue('--primary').trim(),
        secondary: getComputedStyle(document.documentElement).getPropertyValue('--secondary').trim(),
        success: getComputedStyle(document.documentElement).getPropertyValue('--success').trim(),
        danger: getComputedStyle(document.documentElement).getPropertyValue('--danger').trim(),
        warning: getComputedStyle(document.documentElement).getPropertyValue('--warning').trim(),
        info: getComputedStyle(document.documentElement).getPropertyValue('--info').trim(),
        light: getComputedStyle(document.documentElement).getPropertyValue('--text-light').trim()
    },
    animation: {
        enabled: true,
        duration: 800,
        easing: 'easeOutQuart'
    },
    defaultGradient: ['rgba(115,103,240,0.7)', 'rgba(115,103,240,0.1)'],
    chartDefaults: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    boxWidth: 12,
                    padding: 15,
                    usePointStyle: true,
                    pointStyle: 'circle'
                }
            },
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleFont: { size: 12 },
                bodyFont: { size: 12 },
                padding: 10,
                displayColors: true,
                usePointStyle: true
            }
        }
    }
};

// Data cache to avoid redundant API calls
const dataCache = {
    optimizationProcess: null,
    parameterSensitivity: null,
    performanceSurface: null,
    resultsComparison: null
};

// Utility functions
const utils = {
    // Generate a color array from a base color with decreasing opacity
    generateGradientColors: (baseColor, count) => {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const opacity = 1 - (i / count * 0.7);
            colors.push(utils.setOpacity(baseColor, opacity));
        }
        return colors;
    },

    // Set opacity for a color
    setOpacity: (color, opacity) => {
        if (color.startsWith('rgb')) {
            return color.replace('rgb', 'rgba').replace(')', `, ${opacity})`);
        } else if (color.startsWith('#')) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${opacity})`;
        }
        return color;
    },

    // Format number with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    // Format percentage
    formatPercent: (num) => {
        return `${num >= 0 ? '+' : ''}${num}%`;
    },

    // Convert dash-case to camelCase
    toCamelCase: (str) => {
        return str.replace(/-([a-z])/g, function (g) { return g[1].toUpperCase(); });
    },

    // Deep merge objects
    mergeDeep: (target, ...sources) => {
        if (!sources.length) return target;
        const source = sources.shift();
        
        if (utils.isObject(target) && utils.isObject(source)) {
            for (const key in source) {
                if (utils.isObject(source[key])) {
                    if (!target[key]) Object.assign(target, { [key]: {} });
                    utils.mergeDeep(target[key], source[key]);
                } else {
                    Object.assign(target, { [key]: source[key] });
                }
            }
        }
        
        return utils.mergeDeep(target, ...sources);
    },
    
    isObject: (item) => {
        return (item && typeof item === 'object' && !Array.isArray(item));
    }
};

/**
 * Optimization Process Visualizations
 * Shows the progression of optimization iterations, convergence paths, and population distributions
 */
const optimizationProcess = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // View type selector
        document.getElementById('process-view').addEventListener('change', () => {
            this.render();
        });
        
        // Metric selector
        document.getElementById('process-metric').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if data is already cached
        if (dataCache.optimizationProcess) {
            return dataCache.optimizationProcess;
        }
        
        try {
            // In a real implementation, this would be an API call
            // For now, we'll generate mock data
            const data = this.generateMockData();
            dataCache.optimizationProcess = data;
            return data;
        } catch (error) {
            console.error('Error fetching optimization process data:', error);
            return null;
        }
    },
    
    generateMockData: function() {
        // Generate mock optimization process data
        const iterations = 30;
        const populationSize = 20;
        
        const metrics = {
            sharpe: {
                mean: Array(iterations).fill().map((_, i) => 0.8 + (i / iterations) * 0.7 + Math.random() * 0.2 - 0.1),
                best: Array(iterations).fill().map((_, i) => 1.0 + (i / iterations) * 0.8 + Math.random() * 0.1),
                worst: Array(iterations).fill().map((_, i) => 0.6 + (i / iterations) * 0.5 + Math.random() * 0.3 - 0.2)
            },
            returns: {
                mean: Array(iterations).fill().map((_, i) => 10 + (i / iterations) * 25 + Math.random() * 5 - 2.5),
                best: Array(iterations).fill().map((_, i) => 15 + (i / iterations) * 30 + Math.random() * 3),
                worst: Array(iterations).fill().map((_, i) => 5 + (i / iterations) * 20 + Math.random() * 8 - 5)
            },
            drawdown: {
                mean: Array(iterations).fill().map((_, i) => 30 - (i / iterations) * 12 + Math.random() * 4 - 2),
                best: Array(iterations).fill().map((_, i) => 25 - (i / iterations) * 15 + Math.random() * 2 - 1),
                worst: Array(iterations).fill().map((_, i) => 35 - (i / iterations) * 8 + Math.random() * 5 - 2.5)
            },
            calmar: {
                mean: Array(iterations).fill().map((_, i) => 0.3 + (i / iterations) * 0.5 + Math.random() * 0.1 - 0.05),
                best: Array(iterations).fill().map((_, i) => 0.4 + (i / iterations) * 0.6 + Math.random() * 0.05),
                worst: Array(iterations).fill().map((_, i) => 0.2 + (i / iterations) * 0.4 + Math.random() * 0.15 - 0.1)
            }
        };
        
        // Generate convergence data (parameter values over iterations)
        const parameters = {
            sentiment_threshold: Array(iterations).fill().map((_, i) => 0.5 + (Math.random() * 0.05 - 0.025) + (i / iterations) * 0.15),
            position_size: Array(iterations).fill().map((_, i) => 0.1 + (Math.random() * 0.02 - 0.01) + (i / iterations) * 0.05),
            exit_threshold: Array(iterations).fill().map((_, i) => -0.25 - (Math.random() * 0.03 - 0.015) - (i / iterations) * 0.05),
            lookback_period: Array(iterations).fill().map((_, i) => 14 - Math.floor((i / iterations) * 4))
        };
        
        // Generate population distribution data for each iteration
        const populationData = [];
        for (let i = 0; i < iterations; i++) {
            const individualFitness = [];
            const baseValue = metrics.sharpe.mean[i];
            
            for (let j = 0; j < populationSize; j++) {
                individualFitness.push(baseValue - 0.3 + Math.random() * 0.6);
            }
            
            populationData.push({
                iteration: i + 1,
                individuals: individualFitness.sort((a, b) => b - a)
            });
        }
        
        return {
            iterations: Array.from({length: iterations}, (_, i) => i + 1),
            metrics,
            parameters,
            populationData
        };
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('optimization-process-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get selected view and metric
        const viewType = document.getElementById('process-view').value;
        const metricType = document.getElementById('process-metric').value;
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Create chart based on view type
        switch (viewType) {
            case 'iterations':
                this.renderIterationsChart(data, metricType);
                break;
            case 'convergence':
                this.renderConvergenceChart(data, metricType);
                break;
            case 'population':
                this.renderPopulationChart(data, metricType);
                break;
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    renderIterationsChart: function(data, metricType) {
        const ctx = document.getElementById('optimization-process-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get metric data
        const metricData = data.metrics[metricType];
        
        // Configure chart
        const config = {
            type: 'line',
            data: {
                labels: data.iterations,
                datasets: [
                    {
                        label: 'Best Individual',
                        data: metricData.best,
                        borderColor: optimizerConfig.chartColors.success,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.success, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Population Mean',
                        data: metricData.mean,
                        borderColor: optimizerConfig.chartColors.primary,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.primary, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Worst Individual',
                        data: metricData.worst,
                        borderColor: optimizerConfig.chartColors.danger,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.danger, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: this.getMetricLabel(metricType)
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    renderConvergenceChart: function(data, metricType) {
        const ctx = document.getElementById('optimization-process-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Configure chart
        const config = {
            type: 'line',
            data: {
                labels: data.iterations,
                datasets: [
                    {
                        label: 'Sentiment Threshold',
                        data: data.parameters.sentiment_threshold,
                        borderColor: optimizerConfig.chartColors.primary,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.primary, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Position Size',
                        data: data.parameters.position_size,
                        borderColor: optimizerConfig.chartColors.success,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.success, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Exit Threshold',
                        data: data.parameters.exit_threshold,
                        borderColor: optimizerConfig.chartColors.danger,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.danger, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Lookback Period',
                        data: data.parameters.lookback_period,
                        borderColor: optimizerConfig.chartColors.warning,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.warning, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Parameter Value'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Lookback Period'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    renderPopulationChart: function(data, metricType) {
        const ctx = document.getElementById('optimization-process-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get population data for selected iterations
        // We'll take a subset of iterations to avoid overcrowding
        const populationIterations = [0, 4, 9, 14, 19, 29];
        const datasets = populationIterations.map((iterationIndex, index) => {
            const iterationData = data.populationData[iterationIndex];
            const color = utils.generateGradientColors(optimizerConfig.chartColors.primary, populationIterations.length)[index];
            
            return {
                label: `Iteration ${iterationData.iteration}`,
                data: iterationData.individuals,
                backgroundColor: color,
                borderColor: color,
                borderWidth: 1
            };
        });
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                labels: Array.from({length: data.populationData[0].individuals.length}, (_, i) => i + 1),
                datasets: datasets
            },
            options: {
                ...optimizerConfig.chartDefaults,
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return `Individual ${context[0].label}`;
                            },
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Individual (Ranked)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: this.getMetricLabel(metricType)
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    getMetricLabel: function(metricType) {
        switch (metricType) {
            case 'sharpe':
                return 'Sharpe Ratio';
            case 'returns':
                return 'Total Returns (%)';
            case 'drawdown':
                return 'Maximum Drawdown (%)';
            case 'calmar':
                return 'Calmar Ratio';
            default:
                return 'Value';
        }
    }
};

/**
 * Parameter Sensitivity Analysis
 * Shows how parameter changes affect performance metrics
 */
const parameterSensitivity = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Parameter selector
        document.getElementById('sensitivity-parameter').addEventListener('change', () => {
            this.render();
        });
        
        // View type selector
        document.getElementById('sensitivity-view').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if data is already cached
        if (dataCache.parameterSensitivity) {
            return dataCache.parameterSensitivity;
        }
        
        try {
            // In a real implementation, this would be an API call
            // For now, we'll generate mock data
            const data = this.generateMockData();
            dataCache.parameterSensitivity = data;
            return data;
        } catch (error) {
            console.error('Error fetching parameter sensitivity data:', error);
            return null;
        }
    },
    
    generateMockData: function() {
        // Generate mock parameter sensitivity data
        
        // Parameter ranges
        const parameters = {
            sentiment_threshold: {
                range: Array.from({length: 20}, (_, i) => 0.3 + i * 0.05),
                sharpe: [],
                returns: [],
                drawdown: [],
                calmar: []
            },
            position_size: {
                range: Array.from({length: 15}, (_, i) => 0.05 + i * 0.02),
                sharpe: [],
                returns: [],
                drawdown: [],
                calmar: []
            },
            exit_threshold: {
                range: Array.from({length: 20}, (_, i) => -0.5 + i * 0.03),
                sharpe: [],
                returns: [],
                drawdown: [],
                calmar: []
            },
            lookback_period: {
                range: Array.from({length: 15}, (_, i) => 5 + i),
                sharpe: [],
                returns: [],
                drawdown: [],
                calmar: []
            }
        };
        
        // Generate metrics for each parameter value
        for (const param in parameters) {
            const paramData = parameters[param];
            
            // Create a "bell curve" like pattern for Sharpe ratio with some randomness
            const midpoint = Math.floor(paramData.range.length / 2);
            
            paramData.sharpe = paramData.range.map((_, i) => {
                const base = 1.2 - Math.abs(i - midpoint) * (0.8 / midpoint);
                return base + (Math.random() * 0.2 - 0.1);
            });
            
            // Create returns that generally increase with parameter value with diminishing returns
            paramData.returns = paramData.range.map((_, i) => {
                const base = 15 + (i / paramData.range.length) * 25;
                return base * (0.8 + Math.random() * 0.4);
            });
            
            // Create drawdown that initially decreases but then increases again
            paramData.drawdown = paramData.range.map((_, i) => {
                const normalizedPos = i / paramData.range.length;
                const base = 35 - 20 * normalizedPos + 15 * Math.pow(normalizedPos, 2);
                return base * (0.8 + Math.random() * 0.4);
            });
            
            // Create calmar ratio (combination of returns and drawdown)
            paramData.calmar = paramData.range.map((_, i) => {
                return paramData.returns[i] / paramData.drawdown[i];
            });
        }
        
        // Generate bivariate data for parameter interactions
        const bivariate = {
            sentiment_threshold_position_size: this.generateBivariateData(
                parameters.sentiment_threshold.range, 
                parameters.position_size.range
            ),
            sentiment_threshold_exit_threshold: this.generateBivariateData(
                parameters.sentiment_threshold.range, 
                parameters.exit_threshold.range
            ),
            position_size_exit_threshold: this.generateBivariateData(
                parameters.position_size.range, 
                parameters.exit_threshold.range
            )
        };
        
        // Generate impact contribution data
        const impactContribution = {
            sharpe: [
                { parameter: 'Sentiment Threshold', contribution: 42 },
                { parameter: 'Position Size', contribution: 28 },
                { parameter: 'Exit Threshold', contribution: 18 },
                { parameter: 'Lookback Period', contribution: 12 }
            ],
            returns: [
                { parameter: 'Position Size', contribution: 45 },
                { parameter: 'Sentiment Threshold', contribution: 25 },
                { parameter: 'Exit Threshold', contribution: 20 },
                { parameter: 'Lookback Period', contribution: 10 }
            ],
            drawdown: [
                { parameter: 'Position Size', contribution: 50 },
                { parameter: 'Exit Threshold', contribution: 30 },
                { parameter: 'Sentiment Threshold', contribution: 15 },
                { parameter: 'Lookback Period', contribution: 5 }
            ]
        };
        
        return {
            parameters,
            bivariate,
            impactContribution
        };
    },
    
    generateBivariateData: function(xRange, yRange) {
        const data = [];
        
        for (let i = 0; i < xRange.length; i++) {
            for (let j = 0; j < yRange.length; j++) {
                // Create a pattern with an optimal zone in the center
                const xNorm = i / xRange.length;
                const yNorm = j / yRange.length;
                
                // Distance from the center (0.6, 0.4)
                const distance = Math.sqrt(Math.pow(xNorm - 0.6, 2) + Math.pow(yNorm - 0.4, 2));
                
                // Create a bell curve function
                const value = 1.5 * Math.exp(-5 * distance) + 0.3 + (Math.random() * 0.2 - 0.1);
                
                data.push({
                    x: xRange[i],
                    y: yRange[j],
                    z: value
                });
            }
        }
        
        return data;
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('parameter-sensitivity-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get selected parameter and view
        const parameter = document.getElementById('sensitivity-parameter').value;
        const viewType = document.getElementById('sensitivity-view').value;
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Create chart based on view type
        switch (viewType) {
            case 'univariate':
                this.renderUnivariateChart(data, parameter);
                break;
            case 'bivariate':
                this.renderBivariateChart(data, parameter);
                break;
            case 'contribution':
                this.renderContributionChart(data);
                break;
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    renderUnivariateChart: function(data, parameterName) {
        const ctx = document.getElementById('parameter-sensitivity-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get parameter data
        const paramData = data.parameters[parameterName];
        
        // Configure chart
        const config = {
            type: 'line',
            data: {
                labels: paramData.range,
                datasets: [
                    {
                        label: 'Sharpe Ratio',
                        data: paramData.sharpe,
                        borderColor: optimizerConfig.chartColors.primary,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.primary, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Returns (%)',
                        data: paramData.returns,
                        borderColor: optimizerConfig.chartColors.success,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.success, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Max Drawdown (%)',
                        data: paramData.drawdown,
                        borderColor: optimizerConfig.chartColors.danger,
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.danger, 0.1),
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: this.getParameterLabel(parameterName)
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Sharpe Ratio'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Returns / Drawdown (%)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    annotation: {
                        annotations: {
                            optimalLine: {
                                type: 'line',
                                xMin: paramData.range[paramData.sharpe.indexOf(Math.max(...paramData.sharpe))],
                                xMax: paramData.range[paramData.sharpe.indexOf(Math.max(...paramData.sharpe))],
                                borderColor: optimizerConfig.chartColors.info,
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Optimal',
                                    position: 'top'
                                }
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    renderBivariateChart: function(data, parameter) {
        const ctx = document.getElementById('parameter-sensitivity-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Determine which parameter pair to show based on selected parameter
        let bivariateKey;
        let xLabel, yLabel;
        
        switch (parameter) {
            case 'sentiment_threshold':
                bivariateKey = 'sentiment_threshold_position_size';
                xLabel = 'Sentiment Threshold';
                yLabel = 'Position Size';
                break;
            case 'position_size':
                bivariateKey = 'position_size_exit_threshold';
                xLabel = 'Position Size';
                yLabel = 'Exit Threshold';
                break;
            case 'exit_threshold':
                bivariateKey = 'sentiment_threshold_exit_threshold';
                xLabel = 'Sentiment Threshold';
                yLabel = 'Exit Threshold';
                break;
            default:
                bivariateKey = 'sentiment_threshold_position_size';
                xLabel = 'Sentiment Threshold';
                yLabel = 'Position Size';
        }
        
        // Get bivariate data
        const bivariateData = data.bivariate[bivariateKey];
        
        // Transform data for heatmap
        const uniqueX = [...new Set(bivariateData.map(item => item.x))];
        const uniqueY = [...new Set(bivariateData.map(item => item.y))];
        
        // Create the heatmap data
        const heatmapData = uniqueY.map((y, yIndex) => {
            return uniqueX.map((x, xIndex) => {
                const dataPoint = bivariateData.find(item => item.x === x && item.y === y);
                return dataPoint ? dataPoint.z : 0;
            });
        });
        
        // Configure chart
        const config = {
            type: 'heatmap',
            data: {
                datasets: [{
                    label: 'Sharpe Ratio',
                    data: bivariateData.map(item => ({
                        x: item.x,
                        y: item.y,
                        v: item.z
                    })),
                    borderWidth: 1,
                    borderColor: '#ffffff',
                    backgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex].v;
                        const maxValue = Math.max(...bivariateData.map(item => item.z));
                        const minValue = Math.min(...bivariateData.map(item => item.z));
                        const normalizedValue = (value - minValue) / (maxValue - minValue);
                        
                        // Color gradient from red (low) to green (high)
                        if (normalizedValue < 0.3) {
                            return `rgba(255, ${Math.round(normalizedValue * 255 * 3)}, 0, 0.8)`;
                        } else if (normalizedValue < 0.7) {
                            return `rgba(${Math.round(255 - (normalizedValue - 0.3) * 255 * 2.5)}, 255, 0, 0.8)`;
                        } else {
                            return `rgba(0, 255, ${Math.round((normalizedValue - 0.7) * 255 * 3)}, 0.8)`;
                        }
                    }
                }]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return '';
                            },
                            label: function(context) {
                                const data = context.dataset.data[context.dataIndex];
                                return [
                                    `${xLabel}: ${data.x}`,
                                    `${yLabel}: ${data.y}`,
                                    `Sharpe Ratio: ${data.v.toFixed(2)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: xLabel
                        }
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: yLabel
                        }
                    }
                }
            }
        };
        
        // Create chart - use fallback if heatmap is not available
        try {
            this.chart = new Chart(ctx, config);
        } catch (error) {
            console.error('Heatmap chart type not available, using scatter instead', error);
            
            // Fallback to scatter plot
            config.type = 'scatter';
            config.options.plugins.tooltip = {
                callbacks: {
                    label: function(context) {
                        const data = context.dataset.data[context.dataIndex];
                        return [
                            `${xLabel}: ${data.x}`,
                            `${yLabel}: ${data.y}`,
                            `Sharpe Ratio: ${data.v.toFixed(2)}`
                        ];
                    }
                }
            };
            
            this.chart = new Chart(ctx, config);
        }
    },
    
    renderContributionChart: function(data) {
        const ctx = document.getElementById('parameter-sensitivity-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get impact contribution data for Sharpe ratio
        const contributionData = data.impactContribution.sharpe;
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                labels: contributionData.map(item => item.parameter),
                datasets: [{
                    label: 'Impact Contribution (%)',
                    data: contributionData.map(item => item.contribution),
                    backgroundColor: [
                        optimizerConfig.chartColors.primary,
                        optimizerConfig.chartColors.success,
                        optimizerConfig.chartColors.warning,
                        optimizerConfig.chartColors.info
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Contribution to Performance (%)'
                        }
                    }
                },
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Contribution: ${context.parsed.y}%`;
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    getParameterLabel: function(parameterName) {
        switch (parameterName) {
            case 'sentiment_threshold':
                return 'Sentiment Threshold';
            case 'position_size':
                return 'Position Size';
            case 'exit_threshold':
                return 'Exit Threshold';
            case 'lookback_period':
                return 'Lookback Period (Days)';
            default:
                return 'Parameter Value';
        }
    }
};

/**
 * Performance Surface Visualization
 * Shows how two parameters interact to affect performance metrics
 */
const performanceSurface = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // X-axis parameter selector
        document.getElementById('surface-param-x').addEventListener('change', () => {
            this.render();
        });
        
        // Y-axis parameter selector
        document.getElementById('surface-param-y').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if data is already cached
        if (dataCache.performanceSurface) {
            return dataCache.performanceSurface;
        }
        
        try {
            // In a real implementation, this would be an API call
            // For now, we'll generate mock data
            const data = await this.generateMockData();
            dataCache.performanceSurface = data;
            return data;
        } catch (error) {
            console.error('Error fetching performance surface data:', error);
            return null;
        }
    },
    
    generateMockData: async function() {
        // We'll reuse the bivariate data from the parameter sensitivity chart
        const sensitivityData = await parameterSensitivity.fetchData();
        return sensitivityData.bivariate;
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('performance-surface-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get selected parameters
        const xParam = document.getElementById('surface-param-x').value;
        const yParam = document.getElementById('surface-param-y').value;
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Determine which performance surface to show
        let surfaceKey;
        
        if ((xParam === 'sentiment_threshold' && yParam === 'position_size') ||
            (xParam === 'position_size' && yParam === 'sentiment_threshold')) {
            surfaceKey = 'sentiment_threshold_position_size';
        } else if ((xParam === 'sentiment_threshold' && yParam === 'exit_threshold') ||
                   (xParam === 'exit_threshold' && yParam === 'sentiment_threshold')) {
            surfaceKey = 'sentiment_threshold_exit_threshold';
        } else if ((xParam === 'position_size' && yParam === 'exit_threshold') ||
                   (xParam === 'exit_threshold' && yParam === 'position_size')) {
            surfaceKey = 'position_size_exit_threshold';
        } else {
            // Default to first one if no match
            surfaceKey = 'sentiment_threshold_position_size';
        }
        
        // Get surface data
        const surfaceData = data[surfaceKey];
        
        // Transform data for contour plot
        this.renderContourPlot(surfaceData, xParam, yParam);
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    renderContourPlot: function(surfaceData, xParam, yParam) {
        const ctx = document.getElementById('performance-surface-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Extract unique x and y values and z values
        const uniqueX = [...new Set(surfaceData.map(item => item.x))].sort((a, b) => a - b);
        const uniqueY = [...new Set(surfaceData.map(item => item.y))].sort((a, b) => a - b);
        
        // Prepare data for contour plot using scatter
        const scatterData = surfaceData.map(item => ({
            x: item.x,
            y: item.y,
            v: item.z
        }));
        
        // Find max and min values for z
        const maxZ = Math.max(...surfaceData.map(item => item.z));
        const minZ = Math.min(...surfaceData.map(item => item.z));
        
        // Find optimal point (highest z value)
        const optimalPoint = surfaceData.reduce((prev, current) => {
            return (prev.z > current.z) ? prev : current;
        });
        
        // Configure chart
        const config = {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Sharpe Ratio',
                    data: scatterData,
                    pointBackgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex].v;
                        const normalizedValue = (value - minZ) / (maxZ - minZ);
                        
                        // Color gradient from red (low) to green (high)
                        if (normalizedValue < 0.3) {
                            return `rgba(255, ${Math.round(normalizedValue * 255 * 3)}, 0, 0.8)`;
                        } else if (normalizedValue < 0.7) {
                            return `rgba(${Math.round(255 - (normalizedValue - 0.3) * 255 * 2.5)}, 255, 0, 0.8)`;
                        } else {
                            return `rgba(0, 255, ${Math.round((normalizedValue - 0.7) * 255 * 3)}, 0.8)`;
                        }
                    },
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    showLine: false
                }]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: this.getParameterLabel(xParam)
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: this.getParameterLabel(yParam)
                        }
                    }
                },
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const data = context.dataset.data[context.dataIndex];
                                return [
                                    `${scatterData[0].x === undefined ? 'X' : parameterSensitivity.getParameterLabel(xParam)}: ${data.x}`,
                                    `${scatterData[0].y === undefined ? 'Y' : parameterSensitivity.getParameterLabel(yParam)}: ${data.y}`,
                                    `Sharpe Ratio: ${data.v.toFixed(2)}`
                                ];
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            optimalPoint: {
                                type: 'point',
                                xValue: optimalPoint.x,
                                yValue: optimalPoint.y,
                                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                                borderColor: '#000',
                                borderWidth: 2,
                                radius: 8,
                                label: {
                                    display: true,
                                    content: 'Optimal',
                                    position: 'top'
                                }
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    getParameterLabel: function(parameterName) {
        // Reuse the function from the parameter sensitivity module
        return parameterSensitivity.getParameterLabel(parameterName);
    }
};

/**
 * Optimization Results Comparison
 * Compares optimization results across different runs
 */
const resultsComparison = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // View type selector
        document.getElementById('results-view').addEventListener('change', () => {
            this.render();
        });
        
        // Comparison selector
        document.getElementById('compare-with').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if data is already cached
        if (dataCache.resultsComparison) {
            return dataCache.resultsComparison;
        }
        
        try {
            // In a real implementation, this would be an API call
            // For now, we'll generate mock data
            const data = this.generateMockData();
            dataCache.resultsComparison = data;
            return data;
        } catch (error) {
            console.error('Error fetching results comparison data:', error);
            return null;
        }
    },
    
    generateMockData: function() {
        // Generate mock comparison data
        
        // Parameter values for different optimization runs
        const parameters = {
            baseline: {
                sentiment_threshold: 0.50,
                position_size: 0.10,
                exit_threshold: -0.25,
                lookback_period: 14
            },
            optimized: {
                sentiment_threshold: 0.65,
                position_size: 0.15,
                exit_threshold: -0.30,
                lookback_period: 10
            },
            previous: {
                sentiment_threshold: 0.60,
                position_size: 0.12,
                exit_threshold: -0.28,
                lookback_period: 12
            },
            best_historical: {
                sentiment_threshold: 0.70,
                position_size: 0.18,
                exit_threshold: -0.35,
                lookback_period: 8
            }
        };
        
        // Performance metrics for different optimization runs
        const metrics = {
            baseline: {
                sharpe: 1.05,
                sortino: 1.40,
                returns: 22.5,
                drawdown: 15.8,
                volatility: 18.5,
                winRate: 58,
                profitFactor: 1.35,
                calmar: 1.42
            },
            optimized: {
                sharpe: 1.47,
                sortino: 1.85,
                returns: 29.8,
                drawdown: 12.5,
                volatility: 16.2,
                winRate: 64,
                profitFactor: 1.72,
                calmar: 2.38
            },
            previous: {
                sharpe: 1.32,
                sortino: 1.65,
                returns: 26.2,
                drawdown: 14.3,
                volatility: 17.0,
                winRate: 61,
                profitFactor: 1.55,
                calmar: 1.83
            },
            best_historical: {
                sharpe: 1.52,
                sortino: 1.92,
                returns: 33.5,
                drawdown: 14.0,
                volatility: 17.8,
                winRate: 67,
                profitFactor: 1.88,
                calmar: 2.39
            }
        };
        
        // Robustness analysis across different market conditions
        const robustness = {
            baseline: {
                bull: 1.35,
                bear: 0.72,
                sideways: 0.95,
                volatile: 0.65
            },
            optimized: {
                bull: 1.65,
                bear: 0.98,
                sideways: 1.25,
                volatile: 0.92
            },
            previous: {
                bull: 1.55,
                bear: 0.85,
                sideways: 1.12,
                volatile: 0.75
            },
            best_historical: {
                bull: 1.75,
                bear: 0.92,
                sideways: 1.32,
                volatile: 0.88
            }
        };
        
        return {
            parameters,
            metrics,
            robustness
        };
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('results-comparison-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get selected view and comparison
        const viewType = document.getElementById('results-view').value;
        const compareWith = document.getElementById('compare-with').value;
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Create chart based on view type
        switch (viewType) {
            case 'parameters':
                this.renderParametersChart(data, compareWith);
                break;
            case 'metrics':
                this.renderMetricsChart(data, compareWith);
                break;
            case 'robustness':
                this.renderRobustnessChart(data, compareWith);
                break;
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    renderParametersChart: function(data, compareWith) {
        const ctx = document.getElementById('results-comparison-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get parameter data for comparison
        const optimizedParams = data.parameters.optimized;
        const comparisonParams = data.parameters[compareWith];
        
        // Calculate percent change for parameters
        const parameterLabels = ['Sentiment Threshold', 'Position Size', 'Exit Threshold', 'Lookback Period'];
        const parameterKeys = ['sentiment_threshold', 'position_size', 'exit_threshold', 'lookback_period'];
        
        const paramChanges = parameterKeys.map(key => {
            // Calculate percentage change
            const baseline = comparisonParams[key];
            const optimized = optimizedParams[key];
            
            // For numerical comparison
            let changePercent = ((optimized - baseline) / Math.abs(baseline)) * 100;
            
            // Special handling for negative values (like exit threshold)
            if (baseline < 0 && optimized < 0) {
                changePercent = ((Math.abs(optimized) - Math.abs(baseline)) / Math.abs(baseline)) * 100;
                if (Math.abs(optimized) > Math.abs(baseline)) {
                    changePercent = -changePercent; // More negative is "increase" in magnitude
                }
            }
            
            return changePercent;
        });
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                labels: parameterLabels,
                datasets: [{
                    label: 'Parameter Change (%)',
                    data: paramChanges,
                    backgroundColor: paramChanges.map(value => 
                        value >= 0 ? optimizerConfig.chartColors.success : optimizerConfig.chartColors.danger
                    ),
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Change (%)'
                        }
                    }
                },
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const index = context.dataIndex;
                                const key = parameterKeys[index];
                                const baseline = comparisonParams[key];
                                const optimized = optimizedParams[key];
                                
                                return [
                                    `Change: ${context.parsed.y.toFixed(1)}%`,
                                    `${compareWith.charAt(0).toUpperCase() + compareWith.slice(1)}: ${baseline}`,
                                    `Optimized: ${optimized}`
                                ];
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    renderMetricsChart: function(data, compareWith) {
        const ctx = document.getElementById('results-comparison-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get metrics data for comparison
        const optimizedMetrics = data.metrics.optimized;
        const comparisonMetrics = data.metrics[compareWith];
        
        // Calculate percent change for metrics
        const metricLabels = ['Sharpe', 'Sortino', 'Returns', 'Drawdown', 'Volatility', 'Win Rate', 'Profit Factor', 'Calmar'];
        const metricKeys = ['sharpe', 'sortino', 'returns', 'drawdown', 'volatility', 'winRate', 'profitFactor', 'calmar'];
        
        const metricChanges = metricKeys.map(key => {
            // Calculate percentage change
            const baseline = comparisonMetrics[key];
            const optimized = optimizedMetrics[key];
            const changePercent = ((optimized - baseline) / baseline) * 100;
            
            // For drawdown and volatility, a decrease is good (positive change)
            if (key === 'drawdown' || key === 'volatility') {
                return -changePercent;
            }
            
            return changePercent;
        });
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                labels: metricLabels,
                datasets: [{
                    label: 'Metric Improvement (%)',
                    data: metricChanges,
                    backgroundColor: metricChanges.map(value => 
                        value >= 0 ? optimizerConfig.chartColors.success : optimizerConfig.chartColors.danger
                    ),
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Improvement (%)'
                        }
                    }
                },
                plugins: {
                    ...optimizerConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const index = context.dataIndex;
                                const key = metricKeys[index];
                                const baseline = comparisonMetrics[key];
                                const optimized = optimizedMetrics[key];
                                
                                // For drawdown and volatility, show actual change directly
                                if (key === 'drawdown' || key === 'volatility') {
                                    return [
                                        `Improvement: ${context.parsed.y.toFixed(1)}%`,
                                        `${compareWith.charAt(0).toUpperCase() + compareWith.slice(1)}: ${baseline.toFixed(1)}%`,
                                        `Optimized: ${optimized.toFixed(1)}%`
                                    ];
                                }
                                
                                return [
                                    `Improvement: ${context.parsed.y.toFixed(1)}%`,
                                    `${compareWith.charAt(0).toUpperCase() + compareWith.slice(1)}: ${baseline.toFixed(2)}`,
                                    `Optimized: ${optimized.toFixed(2)}`
                                ];
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    },
    
    renderRobustnessChart: function(data, compareWith) {
        const ctx = document.getElementById('results-comparison-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get robustness data
        const marketConditions = ['Bull Market', 'Bear Market', 'Sideways Market', 'Volatile Market'];
        const marketKeys = ['bull', 'bear', 'sideways', 'volatile'];
        
        // Configure chart
        const config = {
            type: 'radar',
            data: {
                labels: marketConditions,
                datasets: [
                    {
                        label: 'Optimized',
                        data: marketKeys.map(key => data.robustness.optimized[key]),
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.success, 0.2),
                        borderColor: optimizerConfig.chartColors.success,
                        pointBackgroundColor: optimizerConfig.chartColors.success,
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: optimizerConfig.chartColors.success
                    },
                    {
                        label: compareWith.charAt(0).toUpperCase() + compareWith.slice(1),
                        data: marketKeys.map(key => data.robustness[compareWith][key]),
                        backgroundColor: utils.setOpacity(optimizerConfig.chartColors.primary, 0.2),
                        borderColor: optimizerConfig.chartColors.primary,
                        pointBackgroundColor: optimizerConfig.chartColors.primary,
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: optimizerConfig.chartColors.primary
                    }
                ]
            },
            options: {
                ...optimizerConfig.chartDefaults,
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 2
                    }
                },
                elements: {
                    line: {
                        tension: 0.2
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    }
};

/**
 * Initialize all visualizations when the DOM is ready
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Chart.js plugins (if present)
    if (Chart.register) {
        // Register necessary plugins if they exist
        if (window.ChartDataLabels) {
            Chart.register(ChartDataLabels);
        }
        if (window.annotationPlugin) {
            Chart.register(annotationPlugin);
        }
    }
    
    // Initialize all modules
    optimizationProcess.init();
    parameterSensitivity.init();
    performanceSurface.init();
    resultsComparison.init();
    
    // Set up global event listeners
    
    // Strategy selector
    document.getElementById('optimizer-strategy').addEventListener('change', function() {
        // Reset all visualizations when strategy changes
        Object.keys(dataCache).forEach(key => {
            dataCache[key] = null;
        });
        
        // Re-render all visualizations
        optimizationProcess.render();
        parameterSensitivity.render();
        performanceSurface.render();
        resultsComparison.render();
        
        // Update summary information
        updateSummaryInformation();
    });
    
    // Asset selector
    document.getElementById('optimizer-asset').addEventListener('change', function() {
        // Similar reset and re-render
        Object.keys(dataCache).forEach(key => {
            dataCache[key] = null;
        });
        
        optimizationProcess.render();
        parameterSensitivity.render();
        performanceSurface.render();
        resultsComparison.render();
        
        updateSummaryInformation();
    });
    
    // Time range selector
    document.getElementById('optimizer-timerange').addEventListener('change', function() {
        // Similar reset and re-render
        Object.keys(dataCache).forEach(key => {
            dataCache[key] = null;
        });
        
        optimizationProcess.render();
        parameterSensitivity.render();
        performanceSurface.render();
        resultsComparison.render();
        
        updateSummaryInformation();
    });
    
    // Optimization run selector
    document.getElementById('optimization-run').addEventListener('change', function() {
        // Similar reset and re-render
        Object.keys(dataCache).forEach(key => {
            dataCache[key] = null;
        });
        
        optimizationProcess.render();
        parameterSensitivity.render();
        performanceSurface.render();
        resultsComparison.render();
        
        updateSummaryInformation();
    });
    
    // Modal settings
    document.getElementById('optimizer-settings-btn').addEventListener('click', function() {
        const modal = document.getElementById('optimizer-settings-modal');
        modal.style.display = 'block';
    });
    
    document.querySelector('#optimizer-settings-modal .btn-close').addEventListener('click', function() {
        const modal = document.getElementById('optimizer-settings-modal');
        modal.style.display = 'none';
    });
    
    document.getElementById('save-optimizer-settings').addEventListener('click', function() {
        const modal = document.getElementById('optimizer-settings-modal');
        modal.style.display = 'none';
        
        // In a real implementation, you would save the settings here
        // For now, we'll just show a success message
        alert('Optimizer settings applied successfully');
    });
    
    // Download data button
    document.getElementById('download-optimizer-data-btn').addEventListener('click', function() {
        // In a real implementation, you would generate and download data here
        // For now, we'll just show a success message
        alert('Optimizer data downloaded successfully');
    });
    
    // Learn more button
    document.getElementById('optimizer-learn-more').addEventListener('click', function() {
        // In a real implementation, you would open a documentation page
        alert('Documentation for optimizer visualizations would open here');
    });
    
    // Export report button
    document.getElementById('export-optimizer-report').addEventListener('click', function() {
        // In a real implementation, you would generate and export a report
        alert('Optimization report exported successfully');
    });
    
    // Initialize the UI
    updateSummaryInformation();
    initializeFeatherIcons();
});

/**
 * Updates the summary information in the UI
 */
function updateSummaryInformation() {
    // In a real implementation, this would update with actual data
    // For now, we'll use static values
    document.getElementById('iterations-count').textContent = '128';
    document.getElementById('convergence-status').textContent = 'Achieved';
    document.getElementById('performance-gain').textContent = '+18.7%';
    document.getElementById('sharpe-improvement').textContent = '+0.42';
    document.getElementById('sensitive-parameter').textContent = 'Sentiment Threshold';
    document.getElementById('optimal-value').textContent = '0.65';
    
    document.getElementById('optimization-time').textContent = '18 minutes';
    document.getElementById('parameter-combinations').textContent = '2,048';
    document.getElementById('cv-score').textContent = '0.82';
    document.getElementById('optimizer-last-updated').textContent = '3 days ago';
}

/**
 * Initializes Feather icons if available
 */
function initializeFeatherIcons() {
    if (typeof feather !== 'undefined' && feather.replace) {
        feather.replace();
    }
}