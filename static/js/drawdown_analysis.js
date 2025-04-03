/**
 * Drawdown Analysis Module
 * 
 * This module provides visualization and analysis tools for portfolio drawdowns,
 * including underwater charts, drawdown distribution, recovery path analysis,
 * and maximum drawdown projection.
 */

// Configuration
const drawdownConfig = {
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
    regimeColors: {
        bear: 'rgba(239, 68, 68, 0.15)',       // Red (danger)
        volatile: 'rgba(249, 115, 22, 0.15)',  // Orange (warning)
        correction: 'rgba(59, 130, 246, 0.15)', // Blue (info)
        consolidation: 'rgba(107, 114, 128, 0.15)', // Gray (secondary)
        bull: 'rgba(16, 185, 129, 0.15)'       // Green (success)
    },
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
                align: 'start',
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
    underwaterChart: null,
    distributionChart: null,
    recoveryChart: null,
    projectionChart: null,
    drawdownEvents: null
};

// Utility functions
const utils = {
    // Format date as MM/DD/YYYY
    formatDate: (date) => {
        if (!(date instanceof Date)) {
            date = new Date(date);
        }
        return `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()}`;
    },
    
    // Format number with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },
    
    // Format percentage
    formatPercent: (num, decimals = 2) => {
        return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
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
    
    // Generates a lighter or darker version of a color
    adjustColor: (color, percent) => {
        let R, G, B, A;
        
        if (color.startsWith('rgba')) {
            const parts = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/);
            R = parseInt(parts[1]);
            G = parseInt(parts[2]);
            B = parseInt(parts[3]);
            A = parseFloat(parts[4]);
        } else if (color.startsWith('rgb')) {
            const parts = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            R = parseInt(parts[1]);
            G = parseInt(parts[2]);
            B = parseInt(parts[3]);
            A = 1;
        } else if (color.startsWith('#')) {
            R = parseInt(color.slice(1, 3), 16);
            G = parseInt(color.slice(3, 5), 16);
            B = parseInt(color.slice(5, 7), 16);
            A = 1;
        } else {
            return color;
        }
        
        // Adjust color - positive percent brightens, negative darkens
        R = Math.min(255, Math.max(0, Math.round(R + (percent * R / 100))));
        G = Math.min(255, Math.max(0, Math.round(G + (percent * G / 100))));
        B = Math.min(255, Math.max(0, Math.round(B + (percent * B / 100))));
        
        return `rgba(${R}, ${G}, ${B}, ${A})`;
    },
    
    // Calculates drawdown value from price series
    calculateDrawdowns: (prices) => {
        if (!prices || prices.length === 0) return [];
        
        const drawdowns = [];
        let peak = prices[0];
        let peakDate = prices[0].date;
        let trough = peak;
        let troughDate = peakDate;
        let inDrawdown = false;
        let drawdownStart = null;
        let currentDrawdown = 0;
        
        for (let i = 0; i < prices.length; i++) {
            const price = prices[i];
            
            // New peak
            if (price.value > peak) {
                // If we were in a drawdown, record it
                if (inDrawdown && currentDrawdown < -0.01) {  // Only record drawdowns > 1%
                    drawdowns.push({
                        startDate: drawdownStart,
                        endDate: troughDate,
                        recoveryDate: price.date,
                        magnitude: currentDrawdown,
                        duration: (troughDate - drawdownStart) / (1000 * 60 * 60 * 24), // days
                        recoveryTime: (price.date - troughDate) / (1000 * 60 * 60 * 24) // days
                    });
                }
                
                // Reset peak
                peak = price.value;
                peakDate = price.date;
                trough = peak;
                troughDate = peakDate;
                inDrawdown = false;
                currentDrawdown = 0;
            } 
            // New trough
            else if (price.value < trough) {
                trough = price.value;
                troughDate = price.date;
                const drawdown = (trough - peak) / peak * 100;
                
                if (!inDrawdown) {
                    inDrawdown = true;
                    drawdownStart = peakDate;
                }
                
                currentDrawdown = drawdown;
            }
            
            // Always calculate current drawdown for underwaterChart
            const underwaterValue = (price.value - peak) / peak * 100;
            price.drawdown = underwaterValue;
        }
        
        // If we're still in a drawdown at the end, record it as ongoing
        if (inDrawdown && currentDrawdown < -0.01) {
            drawdowns.push({
                startDate: drawdownStart,
                endDate: troughDate,
                recoveryDate: null, // No recovery yet
                magnitude: currentDrawdown,
                duration: (troughDate - drawdownStart) / (1000 * 60 * 60 * 24), // days
                recoveryTime: null // No recovery yet
            });
        }
        
        return drawdowns;
    },
    
    // Estimate recovery time based on historical recovery rates
    estimateRecoveryTime: (magnitude, drawdowns) => {
        if (!drawdowns || drawdowns.length === 0) return null;
        
        // Filter for completed drawdowns with valid recovery times
        const completedDrawdowns = drawdowns.filter(d => d.recoveryTime !== null);
        if (completedDrawdowns.length === 0) return null;
        
        // Calculate average recovery rate (percent per day)
        const recoveryRates = completedDrawdowns.map(d => Math.abs(d.magnitude) / d.recoveryTime);
        const avgRecoveryRate = recoveryRates.reduce((sum, rate) => sum + rate, 0) / recoveryRates.length;
        
        // Estimate days to recover
        return Math.abs(magnitude) / avgRecoveryRate;
    }
};

/**
 * Underwater Chart
 * Shows the percentage drawdown over time
 */
const underwaterChart = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Portfolio selector
        document.getElementById('drawdown-portfolio').addEventListener('change', () => {
            this.render();
        });
        
        // Period selector
        document.getElementById('drawdown-period').addEventListener('change', () => {
            this.render();
        });
        
        // Market regime toggle
        document.getElementById('show-regimes-toggle').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if data is already cached
        if (dataCache.underwaterChart) {
            return dataCache.underwaterChart;
        }
        
        try {
            // In a real implementation, this would be an API call
            // For now, we'll generate mock data
            const data = this.generateMockData();
            dataCache.underwaterChart = data;
            return data;
        } catch (error) {
            console.error('Error fetching underwater chart data:', error);
            return null;
        }
    },
    
    generateMockData: function() {
        // Generate mock price data - simulate several years of daily prices
        const daysToGenerate = 1095; // 3 years of data
        const prices = [];
        let currentValue = 10000; // Starting portfolio value
        let currentDate = new Date();
        currentDate.setDate(currentDate.getDate() - daysToGenerate);
        
        // Regime periods (for visualization)
        const regimePeriods = [
            { start: 0, end: 180, type: 'bull' },
            { start: 181, end: 310, type: 'correction' },
            { start: 311, end: 420, type: 'consolidation' },
            { start: 421, end: 600, type: 'bull' },
            { start: 601, end: 750, type: 'volatile' },
            { start: 751, end: 950, type: 'bear' },
            { start: 951, end: 1095, type: 'bull' }
        ];
        
        // Generate prices with realistic market behavior
        for (let i = 0; i < daysToGenerate; i++) {
            // Find current regime
            const regime = regimePeriods.find(r => i >= r.start && i <= r.end);
            let dailyReturn;
            
            // Adjust volatility and bias based on market regime
            switch (regime.type) {
                case 'bull':
                    dailyReturn = (Math.random() * 0.02) - 0.005; // Positive bias
                    break;
                case 'bear':
                    dailyReturn = (Math.random() * 0.025) - 0.018; // Negative bias
                    break;
                case 'correction':
                    dailyReturn = (Math.random() * 0.022) - 0.014; // Moderate negative bias
                    break;
                case 'volatile':
                    dailyReturn = (Math.random() * 0.03) - 0.015; // Higher volatility
                    break;
                case 'consolidation':
                    dailyReturn = (Math.random() * 0.01) - 0.005; // Lower volatility
                    break;
                default:
                    dailyReturn = (Math.random() * 0.015) - 0.0075; // Neutral
            }
            
            // Add some auto-correlation (momentum)
            if (i > 0 && Math.random() > 0.3) {
                const prevReturn = (prices[i-1].value / (i > 1 ? prices[i-2].value : prices[i-1].value)) - 1;
                dailyReturn = 0.7 * dailyReturn + 0.3 * prevReturn;
            }
            
            // Update portfolio value
            currentValue *= (1 + dailyReturn);
            
            // Add to price series
            const date = new Date(currentDate);
            prices.push({
                date: date,
                value: currentValue,
                regime: regime.type
            });
            
            // Move to next day
            currentDate.setDate(currentDate.getDate() + 1);
        }
        
        // Calculate drawdowns
        const drawdowns = utils.calculateDrawdowns(prices);
        
        return {
            prices: prices,
            drawdowns: drawdowns,
            regimePeriods: regimePeriods
        };
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('underwater-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get filter values
        const portfolioType = document.getElementById('drawdown-portfolio').value;
        const periodValue = document.getElementById('drawdown-period').value;
        const showRegimes = document.getElementById('show-regimes-toggle').checked;
        
        // Map period selection to days
        let daysToShow;
        switch (periodValue) {
            case '6m': daysToShow = 180; break;
            case '1y': daysToShow = 365; break;
            case '3y': daysToShow = 1095; break;
            case '5y': daysToShow = 1825; break;
            case 'max': daysToShow = 9999; break;
            default: daysToShow = 365;
        }
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Filter data based on selected time period
        const filteredPrices = data.prices.slice(-daysToShow);
        
        // Prepare data for chart
        const chartData = {
            labels: filteredPrices.map(price => price.date),
            drawdowns: filteredPrices.map(price => price.drawdown || 0)
        };
        
        // Create chart
        this.createUnderwaterChart(chartData, data.regimePeriods, showRegimes);
        
        // Update current drawdown indicator
        const currentDrawdown = document.getElementById('current-drawdown');
        if (currentDrawdown) {
            // Use the most recent drawdown value
            const latestDrawdown = chartData.drawdowns[chartData.drawdowns.length - 1];
            currentDrawdown.textContent = utils.formatPercent(latestDrawdown);
            
            // Apply color based on severity
            if (latestDrawdown < -20) {
                currentDrawdown.style.color = drawdownConfig.chartColors.danger;
            } else if (latestDrawdown < -10) {
                currentDrawdown.style.color = drawdownConfig.chartColors.warning;
            } else if (latestDrawdown < -5) {
                currentDrawdown.style.color = drawdownConfig.chartColors.info;
            } else {
                currentDrawdown.style.color = '';
            }
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    createUnderwaterChart: function(data, regimePeriods, showRegimes) {
        const ctx = document.getElementById('underwater-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Prepare annotations for market regimes if enabled
        const annotations = {};
        
        if (showRegimes) {
            // Convert regime periods to annotation boxes
            regimePeriods.forEach((regime, index) => {
                // Only include regimes that overlap with the displayed data range
                if (regime.end >= data.labels.length - data.labels.length / 3) {
                    const startIndex = Math.max(0, regime.start - (data.labels.length - data.labels.length / 3));
                    const endIndex = Math.min(data.labels.length - 1, regime.end - (data.labels.length - data.labels.length / 3));
                    
                    if (startIndex < endIndex) {
                        annotations[`regime-${index}`] = {
                            type: 'box',
                            xMin: data.labels[startIndex],
                            xMax: data.labels[endIndex],
                            yMin: 'min',
                            yMax: 'max',
                            backgroundColor: drawdownConfig.regimeColors[regime.type] || 'rgba(0, 0, 0, 0.05)',
                            borderWidth: 0
                        };
                    }
                }
            });
        }
        
        // Configure chart
        const config = {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Underwater Chart (Drawdown %)',
                    data: data.drawdowns,
                    fill: 'start',
                    backgroundColor: (context) => {
                        const ctx = context.chart.ctx;
                        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                        gradient.addColorStop(0, utils.setOpacity(drawdownConfig.chartColors.danger, 0.8));
                        gradient.addColorStop(0.5, utils.setOpacity(drawdownConfig.chartColors.warning, 0.4));
                        gradient.addColorStop(1, utils.setOpacity(drawdownConfig.chartColors.info, 0.1));
                        return gradient;
                    },
                    borderColor: drawdownConfig.chartColors.danger,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.4
                }]
            },
            options: {
                ...drawdownConfig.chartDefaults,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM YYYY'
                            }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Drawdown (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        // Invert y-axis to show drawdowns going down
                        max: 5,
                        min: -40
                    }
                },
                plugins: {
                    ...drawdownConfig.chartDefaults.plugins,
                    annotation: {
                        annotations: annotations
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Drawdown: ${context.parsed.y.toFixed(2)}%`;
                            },
                            title: function(tooltipItems) {
                                return utils.formatDate(tooltipItems[0].parsed.x);
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    }
};

/**
 * Drawdown Distribution
 * Shows the distribution of drawdowns by magnitude
 */
const distributionChart = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Portfolio selector
        document.getElementById('drawdown-portfolio').addEventListener('change', () => {
            this.render();
        });
        
        // Period selector
        document.getElementById('drawdown-period').addEventListener('change', () => {
            this.render();
        });
        
        // Threshold selector
        document.getElementById('drawdown-threshold').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if cached data exists
        if (dataCache.distributionChart) {
            return dataCache.distributionChart;
        }
        
        try {
            // Fetch underwater chart data which contains drawdowns
            const underwaterData = await underwaterChart.fetchData();
            if (!underwaterData) {
                return null;
            }
            
            // Process drawdown data for distribution
            const data = this.processDrawdownData(underwaterData.drawdowns);
            dataCache.distributionChart = data;
            return data;
        } catch (error) {
            console.error('Error fetching drawdown distribution data:', error);
            return null;
        }
    },
    
    processDrawdownData: function(drawdowns) {
        if (!drawdowns || drawdowns.length === 0) {
            return null;
        }
        
        // Create distribution buckets
        const buckets = [
            { min: 0, max: 5, label: '0-5%', count: 0 },
            { min: 5, max: 10, label: '5-10%', count: 0 },
            { min: 10, max: 15, label: '10-15%', count: 0 },
            { min: 15, max: 20, label: '15-20%', count: 0 },
            { min: 20, max: 25, label: '20-25%', count: 0 },
            { min: 25, max: 30, label: '25-30%', count: 0 },
            { min: 30, max: 35, label: '30-35%', count: 0 },
            { min: 35, max: 40, label: '35-40%', count: 0 },
            { min: 40, max: 100, label: '40%+', count: 0 }
        ];
        
        // Count drawdowns in each bucket
        drawdowns.forEach(drawdown => {
            const magnitude = Math.abs(drawdown.magnitude);
            const bucket = buckets.find(b => magnitude >= b.min && magnitude < b.max);
            if (bucket) {
                bucket.count++;
            }
        });
        
        // Calculate average drawdown
        const totalDrawdown = drawdowns.reduce((sum, d) => sum + Math.abs(d.magnitude), 0);
        const avgDrawdown = totalDrawdown / drawdowns.length;
        
        return {
            distribution: buckets,
            averageDrawdown: avgDrawdown,
            count: drawdowns.length,
            maxDrawdown: Math.max(...drawdowns.map(d => Math.abs(d.magnitude)))
        };
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('distribution-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get filter values
        const thresholdValue = parseInt(document.getElementById('drawdown-threshold').value);
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Filter data based on threshold
        const filteredDistribution = data.distribution.filter(bucket => bucket.min >= thresholdValue);
        
        // Create chart
        this.createDistributionChart(filteredDistribution);
        
        // Update average drawdown indicator
        const avgDrawdown = document.getElementById('avg-drawdown');
        if (avgDrawdown) {
            avgDrawdown.textContent = utils.formatPercent(-data.averageDrawdown);
            
            // Apply color based on severity
            if (data.averageDrawdown > 20) {
                avgDrawdown.style.color = drawdownConfig.chartColors.danger;
            } else if (data.averageDrawdown > 10) {
                avgDrawdown.style.color = drawdownConfig.chartColors.warning;
            } else {
                avgDrawdown.style.color = '';
            }
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    createDistributionChart: function(distribution) {
        const ctx = document.getElementById('distribution-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Generate color gradient based on drawdown severity
        const colors = distribution.map(bucket => {
            const severity = bucket.min / 40; // Max 40% bucket
            return utils.adjustColor(
                utils.setOpacity(drawdownConfig.chartColors.danger, 0.7), 
                (1 - severity) * 100 - 50
            );
        });
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                labels: distribution.map(bucket => bucket.label),
                datasets: [{
                    label: 'Drawdown Frequency',
                    data: distribution.map(bucket => bucket.count),
                    backgroundColor: colors,
                    borderColor: 'rgba(255, 255, 255, 0.5)',
                    borderWidth: 1
                }]
            },
            options: {
                ...drawdownConfig.chartDefaults,
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Drawdown Magnitude'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        },
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    ...drawdownConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Count: ${context.parsed.y}`;
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    }
};

/**
 * Recovery Analysis
 * Shows recovery time vs. drawdown magnitude
 */
const recoveryChart = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Portfolio selector
        document.getElementById('drawdown-portfolio').addEventListener('change', () => {
            this.render();
        });
        
        // Period selector
        document.getElementById('drawdown-period').addEventListener('change', () => {
            this.render();
        });
        
        // Threshold selector
        document.getElementById('drawdown-threshold').addEventListener('change', () => {
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if cached data exists
        if (dataCache.recoveryChart) {
            return dataCache.recoveryChart;
        }
        
        try {
            // Fetch underwater chart data which contains drawdowns
            const underwaterData = await underwaterChart.fetchData();
            if (!underwaterData) {
                return null;
            }
            
            // Process drawdown data for recovery analysis
            const data = this.processRecoveryData(underwaterData.drawdowns);
            dataCache.recoveryChart = data;
            return data;
        } catch (error) {
            console.error('Error fetching recovery analysis data:', error);
            return null;
        }
    },
    
    processRecoveryData: function(drawdowns) {
        if (!drawdowns || drawdowns.length === 0) {
            return null;
        }
        
        // Filter for completed drawdowns (with recovery)
        const completedDrawdowns = drawdowns.filter(d => d.recoveryTime !== null);
        
        // Calculate average recovery time
        const totalRecoveryTime = completedDrawdowns.reduce((sum, d) => sum + d.recoveryTime, 0);
        const avgRecoveryTime = completedDrawdowns.length > 0 ? totalRecoveryTime / completedDrawdowns.length : 0;
        
        // Calculate recovery statistics by magnitude
        const recoveryByMagnitude = [];
        
        // Group by magnitude buckets (5% intervals)
        const magnitudeBuckets = [
            { min: 0, max: 5, label: '0-5%' },
            { min: 5, max: 10, label: '5-10%' },
            { min: 10, max: 15, label: '10-15%' },
            { min: 15, max: 20, label: '15-20%' },
            { min: 20, max: 25, label: '20-25%' },
            { min: 25, max: 30, label: '25-30%' },
            { min: 30, max: 35, label: '30-35%' },
            { min: 35, max: 40, label: '35-40%' },
            { min: 40, max: 100, label: '40%+' }
        ];
        
        magnitudeBuckets.forEach(bucket => {
            const bucketDrawdowns = completedDrawdowns.filter(d => {
                const magnitude = Math.abs(d.magnitude);
                return magnitude >= bucket.min && magnitude < bucket.max;
            });
            
            if (bucketDrawdowns.length > 0) {
                const avgTime = bucketDrawdowns.reduce((sum, d) => sum + d.recoveryTime, 0) / bucketDrawdowns.length;
                recoveryByMagnitude.push({
                    ...bucket,
                    avgRecoveryTime: avgTime,
                    count: bucketDrawdowns.length,
                    maxRecoveryTime: Math.max(...bucketDrawdowns.map(d => d.recoveryTime)),
                    minRecoveryTime: Math.min(...bucketDrawdowns.map(d => d.recoveryTime))
                });
            }
        });
        
        // In an actual implementation, we'd also analyze recovery by market regime
        const recoveryByRegime = [
            { regime: 'bull', avgRecoveryTime: 12, count: 5 },
            { regime: 'correction', avgRecoveryTime: 28, count: 8 },
            { regime: 'consolidation', avgRecoveryTime: 24, count: 6 },
            { regime: 'volatile', avgRecoveryTime: 42, count: 7 },
            { regime: 'bear', avgRecoveryTime: 65, count: 4 }
        ];
        
        // Scatter plot data of magnitude vs. recovery time
        const scatterData = completedDrawdowns.map(drawdown => ({
            x: Math.abs(drawdown.magnitude),
            y: drawdown.recoveryTime,
            startDate: drawdown.startDate,
            endDate: drawdown.endDate,
            recoveryDate: drawdown.recoveryDate,
            duration: drawdown.duration
        }));
        
        return {
            averageRecoveryTime: avgRecoveryTime,
            recoveryByMagnitude: recoveryByMagnitude,
            recoveryByRegime: recoveryByRegime,
            scatterData: scatterData,
            maxRecoveryTime: completedDrawdowns.length > 0 ? 
                Math.max(...completedDrawdowns.map(d => d.recoveryTime)) : 0
        };
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('recovery-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get filter values
        const thresholdValue = parseInt(document.getElementById('drawdown-threshold').value);
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Filter data based on threshold
        const filteredScatterData = data.scatterData.filter(point => point.x >= thresholdValue);
        
        // Create chart
        this.createRecoveryChart(filteredScatterData, data);
        
        // Update average recovery indicator
        const avgRecovery = document.getElementById('avg-recovery');
        if (avgRecovery) {
            avgRecovery.textContent = `${Math.round(data.averageRecoveryTime)} days`;
            
            // Apply color based on recovery time
            if (data.averageRecoveryTime > 60) {
                avgRecovery.style.color = drawdownConfig.chartColors.danger;
            } else if (data.averageRecoveryTime > 30) {
                avgRecovery.style.color = drawdownConfig.chartColors.warning;
            } else {
                avgRecovery.style.color = '';
            }
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    createRecoveryChart: function(scatterData, data) {
        const ctx = document.getElementById('recovery-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Generate trend line data using simple linear regression
        let trendLine = [];
        if (scatterData.length > 1) {
            // Simple linear regression
            const n = scatterData.length;
            const sumX = scatterData.reduce((sum, point) => sum + point.x, 0);
            const sumY = scatterData.reduce((sum, point) => sum + point.y, 0);
            const sumXY = scatterData.reduce((sum, point) => sum + (point.x * point.y), 0);
            const sumX2 = scatterData.reduce((sum, point) => sum + (point.x * point.x), 0);
            
            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            const intercept = (sumY - slope * sumX) / n;
            
            // Generate points for trend line
            const minX = Math.min(...scatterData.map(point => point.x));
            const maxX = Math.max(...scatterData.map(point => point.x));
            
            trendLine = [
                { x: minX, y: minX * slope + intercept },
                { x: maxX, y: maxX * slope + intercept }
            ];
        }
        
        // Define point colors based on efficiency of recovery
        const pointColors = scatterData.map(point => {
            // Recovery efficiency = magnitude / recovery time
            // Higher is better (faster recovery relative to drawdown size)
            const efficiency = point.x / point.y;
            
            if (efficiency > 0.7) {
                return drawdownConfig.chartColors.success;
            } else if (efficiency > 0.4) {
                return drawdownConfig.chartColors.warning;
            } else {
                return drawdownConfig.chartColors.danger;
            }
        });
        
        // Configure chart
        const config = {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Drawdown Recovery',
                    data: scatterData,
                    backgroundColor: pointColors,
                    borderColor: '#ffffff',
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 8
                },
                {
                    label: 'Trend Line',
                    data: trendLine,
                    backgroundColor: 'transparent',
                    borderColor: drawdownConfig.chartColors.secondary,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    showLine: true
                }]
            },
            options: {
                ...drawdownConfig.chartDefaults,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Drawdown Magnitude (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Recovery Time (Days)'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    ...drawdownConfig.chartDefaults.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return [
                                    `Drawdown: ${point.x.toFixed(2)}%`,
                                    `Recovery: ${Math.round(point.y)} days`,
                                    `Period: ${utils.formatDate(point.startDate)} - ${utils.formatDate(point.recoveryDate)}`
                                ];
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    }
};

/**
 * Maximum Drawdown Projection
 * Projects drawdown using Monte Carlo simulation
 */
const projectionChart = {
    chart: null,
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Portfolio selector
        document.getElementById('drawdown-portfolio').addEventListener('change', () => {
            this.render();
        });
        
        // Settings button
        document.getElementById('drawdown-settings-btn').addEventListener('click', () => {
            // Show settings modal
            const modal = document.getElementById('drawdown-settings-modal');
            if (modal) {
                modal.style.display = 'block';
            }
        });
        
        // Settings save button
        document.getElementById('save-drawdown-settings').addEventListener('click', () => {
            // Hide modal
            const modal = document.getElementById('drawdown-settings-modal');
            if (modal) {
                modal.style.display = 'none';
            }
            
            // Re-render projection with new settings
            this.render();
        });
    },
    
    fetchData: async function() {
        // Check if cached data exists
        if (dataCache.projectionChart) {
            return dataCache.projectionChart;
        }
        
        try {
            // In a real implementation, this would be calculated server-side
            // For now, generate mock projection data
            const data = this.generateMockProjectionData();
            dataCache.projectionChart = data;
            return data;
        } catch (error) {
            console.error('Error fetching drawdown projection data:', error);
            return null;
        }
    },
    
    generateMockProjectionData: function() {
        // Generate mock projection data using Monte Carlo simulation
        const simulationCount = 1000;
        const timeHorizon = 252; // trading days (1 year)
        
        // Portfolio parameters (these would come from actual portfolio in real implementation)
        const annualReturn = 0.12; // 12%
        const annualVolatility = 0.18; // 18%
        const dailyReturn = annualReturn / 252;
        const dailyVolatility = annualVolatility / Math.sqrt(252);
        
        // Run Monte Carlo simulations
        const simulations = [];
        for (let i = 0; i < simulationCount; i++) {
            const prices = [100]; // Start at 100
            let peak = 100;
            let maxDrawdown = 0;
            
            for (let day = 1; day < timeHorizon; day++) {
                // Generate daily return with normal distribution
                const dailyRandomReturn = this.generateNormalRandom(dailyReturn, dailyVolatility);
                const price = prices[day - 1] * (1 + dailyRandomReturn);
                prices.push(price);
                
                // Track peak and calculate drawdown
                if (price > peak) {
                    peak = price;
                } else {
                    const drawdown = (price - peak) / peak * 100;
                    maxDrawdown = Math.min(maxDrawdown, drawdown);
                }
            }
            
            simulations.push({
                prices: prices,
                maxDrawdown: maxDrawdown
            });
        }
        
        // Sort simulations by max drawdown severity
        simulations.sort((a, b) => a.maxDrawdown - b.maxDrawdown);
        
        // Calculate drawdown distribution at different confidence levels
        const confidenceLevels = [50, 75, 90, 95, 99];
        const drawdownAtConfidence = {};
        
        confidenceLevels.forEach(level => {
            const index = Math.min(simulationCount - 1, Math.floor((level / 100) * simulationCount));
            drawdownAtConfidence[level] = simulations[index].maxDrawdown;
        });
        
        // Get historical drawdowns for comparison
        // In a real implementation, this would come from actual portfolio data
        const historicalDrawdowns = [
            { year: '2018', maxDrawdown: -33.7 },
            { year: '2019', maxDrawdown: -15.2 },
            { year: '2020', maxDrawdown: -37.2 },
            { year: '2021', maxDrawdown: -21.5 },
            { year: '2022', maxDrawdown: -30.1 },
            { year: '2023', maxDrawdown: -14.8 },
            { year: '2024 YTD', maxDrawdown: -18.3 }
        ];
        
        return {
            simulations: simulations,
            drawdownAtConfidence: drawdownAtConfidence,
            historicalDrawdowns: historicalDrawdowns,
            simulationParams: {
                annualReturn: annualReturn,
                annualVolatility: annualVolatility,
                simulationCount: simulationCount,
                timeHorizon: timeHorizon
            }
        };
    },
    
    generateNormalRandom: function(mean, stdDev) {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        
        return z0 * stdDev + mean;
    },
    
    render: async function() {
        // Hide previous chart if it exists
        const chartContainer = document.getElementById('projection-chart');
        chartContainer.querySelector('.chart-overlay').style.display = 'flex';
        
        // Get confidence setting
        const confidenceSlider = document.getElementById('projection-confidence');
        const confidenceLevel = confidenceSlider ? parseInt(confidenceSlider.value) : 95;
        
        // Fetch data
        const data = await this.fetchData();
        if (!data) {
            chartContainer.querySelector('.chart-overlay').textContent = 'Error loading data';
            return;
        }
        
        // Create chart
        this.createProjectionChart(data, confidenceLevel);
        
        // Update projection model indicator
        const projectionModel = document.getElementById('projection-model');
        if (projectionModel) {
            const useMonteCarlo = document.getElementById('use-monte-carlo');
            projectionModel.textContent = useMonteCarlo && useMonteCarlo.checked ? 
                'Monte Carlo' : 'Historical';
        }
        
        // Hide loading overlay
        chartContainer.querySelector('.chart-overlay').style.display = 'none';
    },
    
    createProjectionChart: function(data, confidenceLevel) {
        const ctx = document.getElementById('projection-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Get closest confidence level
        const closestLevel = [50, 75, 90, 95, 99].reduce((prev, curr) => {
            return (Math.abs(curr - confidenceLevel) < Math.abs(prev - confidenceLevel)) ? curr : prev;
        });
        
        // Histogram data - group drawdowns into buckets
        const bucketSize = 5; // 5% buckets
        const buckets = {};
        
        // Initialize buckets
        for (let i = 0; i <= 60; i += bucketSize) {
            buckets[i] = 0;
        }
        
        // Count drawdowns in each bucket
        data.simulations.forEach(sim => {
            const drawdown = Math.abs(sim.maxDrawdown);
            const bucketKey = Math.floor(drawdown / bucketSize) * bucketSize;
            buckets[bucketKey] = (buckets[bucketKey] || 0) + 1;
        });
        
        // Convert to array format for chart
        const histogramData = Object.keys(buckets).map(key => ({
            x: parseInt(key),
            y: buckets[key] / data.simulations.length * 100 // Convert to percentage
        })).filter(b => b.y > 0);
        
        // Highlight projected max drawdown
        const projectedDrawdown = Math.abs(data.drawdownAtConfidence[closestLevel]);
        
        // Configure chart
        const config = {
            type: 'bar',
            data: {
                datasets: [{
                    label: 'Maximum Drawdown Probability',
                    data: histogramData,
                    backgroundColor: (context) => {
                        const value = context.parsed.x;
                        
                        // Gradient color based on drawdown severity
                        if (value < 15) {
                            return utils.setOpacity(drawdownConfig.chartColors.info, 0.7);
                        } else if (value < 25) {
                            return utils.setOpacity(drawdownConfig.chartColors.warning, 0.7);
                        } else {
                            return utils.setOpacity(drawdownConfig.chartColors.danger, 0.7);
                        }
                    },
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                ...drawdownConfig.chartDefaults,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Maximum Drawdown (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Probability (%)'
                        },
                        beginAtZero: true,
                        max: 30 // Max 30% probability for better visualization
                    }
                },
                plugins: {
                    ...drawdownConfig.chartDefaults.plugins,
                    annotation: {
                        annotations: {
                            projected: {
                                type: 'line',
                                xMin: projectedDrawdown,
                                xMax: projectedDrawdown,
                                borderColor: drawdownConfig.chartColors.danger,
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: `${confidenceLevel}% Confidence: ${projectedDrawdown.toFixed(1)}%`,
                                    position: 'top'
                                }
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${context.parsed.y.toFixed(1)}%`;
                            },
                            title: function(tooltipItems) {
                                const value = tooltipItems[0].parsed.x;
                                return `${value}% - ${value + bucketSize}% Drawdown`;
                            }
                        }
                    }
                }
            }
        };
        
        // Create chart
        this.chart = new Chart(ctx, config);
    }
};

/**
 * Drawdown Events Table
 * Manages the drawdown events table and pagination
 */
const drawdownTable = {
    currentPage: 1,
    itemsPerPage: 5,
    sortBy: 'date-desc',
    searchTerm: '',
    
    init: function() {
        this.setupEventListeners();
        this.render();
    },
    
    setupEventListeners: function() {
        // Pagination controls
        document.getElementById('prev-page').addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.render();
            }
        });
        
        document.getElementById('next-page').addEventListener('click', () => {
            this.currentPage++;
            this.render();
        });
        
        // Sort control
        document.getElementById('drawdown-sort').addEventListener('change', (e) => {
            this.sortBy = e.target.value;
            this.currentPage = 1; // Reset to first page
            this.render();
        });
        
        // Search input
        document.getElementById('drawdown-search').addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.currentPage = 1; // Reset to first page
            this.render();
        });
        
        // Portfolio and period selectors
        document.getElementById('drawdown-portfolio').addEventListener('change', () => {
            this.currentPage = 1; // Reset to first page
            this.render();
        });
        
        document.getElementById('drawdown-period').addEventListener('change', () => {
            this.currentPage = 1; // Reset to first page
            this.render();
        });
        
        // Threshold selector
        document.getElementById('drawdown-threshold').addEventListener('change', () => {
            this.currentPage = 1; // Reset to first page
            this.render();
        });
        
        // Analyze buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('analyze-btn')) {
                const drawdownId = e.target.getAttribute('data-drawdown-id');
                this.analyzeDrawdown(drawdownId);
            }
        });
    },
    
    fetchData: async function() {
        // Check if cached data exists
        if (dataCache.drawdownEvents) {
            return dataCache.drawdownEvents;
        }
        
        try {
            // Fetch underwater chart data which contains drawdowns
            const underwaterData = await underwaterChart.fetchData();
            if (!underwaterData) {
                return null;
            }
            
            // Process drawdown data for table
            const data = this.processDrawdownEventsData(underwaterData.drawdowns);
            dataCache.drawdownEvents = data;
            return data;
        } catch (error) {
            console.error('Error fetching drawdown events data:', error);
            return null;
        }
    },
    
    processDrawdownEventsData: function(drawdowns) {
        if (!drawdowns || drawdowns.length === 0) {
            return [];
        }
        
        // Map drawdowns to table format
        return drawdowns.map((drawdown, index) => {
            // For a real implementation, we'd get market regime from actual data
            // For now, assign regimes based on magnitude and duration
            let regime;
            if (Math.abs(drawdown.magnitude) > 30) {
                regime = 'bear';
            } else if (Math.abs(drawdown.magnitude) > 20) {
                regime = 'volatile';
            } else if (Math.abs(drawdown.magnitude) > 10) {
                regime = 'correction';
            } else {
                regime = 'consolidation';
            }
            
            // Assign severity class
            let severityClass;
            if (Math.abs(drawdown.magnitude) > 30) {
                severityClass = 'severe-drawdown';
            } else if (Math.abs(drawdown.magnitude) > 20) {
                severityClass = 'major-drawdown';
            } else if (Math.abs(drawdown.magnitude) > 10) {
                severityClass = 'moderate-drawdown';
            } else {
                severityClass = 'minor-drawdown';
            }
            
            return {
                id: `d${index + 1}`,
                startDate: drawdown.startDate,
                endDate: drawdown.endDate,
                recoveryDate: drawdown.recoveryDate,
                magnitude: drawdown.magnitude,
                duration: drawdown.duration,
                recoveryTime: drawdown.recoveryTime,
                regime: regime,
                severityClass: severityClass
            };
        });
    },
    
    sortDrawdowns: function(drawdowns, sortBy) {
        if (!drawdowns || drawdowns.length === 0) {
            return [];
        }
        
        const sorted = [...drawdowns];
        
        switch (sortBy) {
            case 'date-desc':
                sorted.sort((a, b) => b.startDate - a.startDate);
                break;
            case 'date-asc':
                sorted.sort((a, b) => a.startDate - b.startDate);
                break;
            case 'magnitude-desc':
                sorted.sort((a, b) => a.magnitude - b.magnitude); // More negative first
                break;
            case 'duration-desc':
                sorted.sort((a, b) => b.duration - a.duration);
                break;
            case 'recovery-desc':
                // Sort by recovery time, handling null values (ongoing drawdowns)
                sorted.sort((a, b) => {
                    if (a.recoveryTime === null && b.recoveryTime === null) return 0;
                    if (a.recoveryTime === null) return -1; // Null (ongoing) first
                    if (b.recoveryTime === null) return 1;
                    return b.recoveryTime - a.recoveryTime;
                });
                break;
            default:
                sorted.sort((a, b) => b.startDate - a.startDate);
        }
        
        return sorted;
    },
    
    filterDrawdowns: function(drawdowns) {
        if (!drawdowns || drawdowns.length === 0) {
            return [];
        }
        
        // Get filter values
        const thresholdValue = parseInt(document.getElementById('drawdown-threshold').value);
        const searchTerm = this.searchTerm.toLowerCase();
        
        // Filter by threshold and search term
        return drawdowns.filter(drawdown => {
            // Filter by threshold
            if (Math.abs(drawdown.magnitude) < thresholdValue) {
                return false;
            }
            
            // Filter by search term
            if (searchTerm) {
                // Match against date, magnitude, or regime
                const startDateStr = utils.formatDate(drawdown.startDate);
                const endDateStr = utils.formatDate(drawdown.endDate);
                const magnitudeStr = drawdown.magnitude.toFixed(2);
                
                return startDateStr.includes(searchTerm) || 
                       endDateStr.includes(searchTerm) || 
                       magnitudeStr.includes(searchTerm) || 
                       drawdown.regime.includes(searchTerm);
            }
            
            return true;
        });
    },
    
    render: async function() {
        // Get table body element
        const tableBody = document.getElementById('drawdown-table-body');
        if (!tableBody) return;
        
        // Clear table
        tableBody.innerHTML = '';
        
        // Show loading indicator
        tableBody.innerHTML = '<tr><td colspan="7" class="text-center">Loading drawdown events...</td></tr>';
        
        // Fetch data
        const data = await this.fetchData();
        if (!data || data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No drawdown events found</td></tr>';
            return;
        }
        
        // Filter and sort data
        const filteredData = this.filterDrawdowns(data);
        const sortedData = this.sortDrawdowns(filteredData, this.sortBy);
        
        // Calculate pagination
        const totalPages = Math.ceil(sortedData.length / this.itemsPerPage);
        if (this.currentPage > totalPages) {
            this.currentPage = totalPages > 0 ? totalPages : 1;
        }
        
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, sortedData.length);
        const pageData = sortedData.slice(startIndex, endIndex);
        
        // Clear table
        tableBody.innerHTML = '';
        
        // Add rows
        if (pageData.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No matching drawdown events</td></tr>';
        } else {
            pageData.forEach(drawdown => {
                const row = document.createElement('tr');
                row.className = drawdown.severityClass;
                
                row.innerHTML = `
                    <td>${utils.formatDate(drawdown.startDate)}</td>
                    <td>${utils.formatDate(drawdown.endDate)}</td>
                    <td>${utils.formatPercent(drawdown.magnitude, 2)}</td>
                    <td>${Math.round(drawdown.duration)} days</td>
                    <td>${drawdown.recoveryTime !== null ? Math.round(drawdown.recoveryTime) + ' days' : 'Ongoing'}</td>
                    <td><span class="${drawdown.regime}-regime">${this.getRegimeLabel(drawdown.regime)}</span></td>
                    <td><button class="btn btn-sm analyze-btn" data-drawdown-id="${drawdown.id}">Analyze</button></td>
                `;
                
                tableBody.appendChild(row);
            });
        }
        
        // Update pagination controls
        document.getElementById('current-page').textContent = this.currentPage;
        document.getElementById('total-pages').textContent = totalPages;
        document.getElementById('prev-page').disabled = this.currentPage <= 1;
        document.getElementById('next-page').disabled = this.currentPage >= totalPages;
    },
    
    getRegimeLabel: function(regime) {
        switch (regime) {
            case 'bear': return 'Bear Market';
            case 'volatile': return 'Volatile Market';
            case 'correction': return 'Correction';
            case 'consolidation': return 'Consolidation';
            case 'bull': return 'Bull Market';
            default: return regime;
        }
    },
    
    analyzeDrawdown: function(drawdownId) {
        // In a real implementation, this would open a detailed analysis modal
        // For now, just log to console
        console.log(`Analyzing drawdown ${drawdownId}`);
        alert(`Detailed analysis for drawdown ${drawdownId} would appear here.`);
    }
};

/**
 * Drawdown Metrics
 * Updates summary metrics for drawdowns
 */
const drawdownMetrics = {
    init: function() {
        this.render();
    },
    
    async render() {
        try {
            // Fetch data from underwater chart (contains drawdowns)
            const underwaterData = await underwaterChart.fetchData();
            if (!underwaterData || !underwaterData.drawdowns) {
                return;
            }
            
            // Calculate metrics
            const metrics = this.calculateMetrics(underwaterData.drawdowns);
            
            // Update UI elements
            this.updateMetricsUI(metrics);
            
        } catch (error) {
            console.error('Error rendering drawdown metrics:', error);
        }
    },
    
    calculateMetrics: function(drawdowns) {
        if (!drawdowns || drawdowns.length === 0) {
            return {};
        }
        
        // Calculate drawdown statistics
        const magnitudes = drawdowns.map(d => Math.abs(d.magnitude));
        const durations = drawdowns.map(d => d.duration);
        const recoveryTimes = drawdowns.filter(d => d.recoveryTime !== null).map(d => d.recoveryTime);
        
        // Find maximum drawdown
        const maxDrawdown = Math.max(...magnitudes);
        const maxDrawdownIndex = magnitudes.indexOf(maxDrawdown);
        
        // Calculate averages
        const avgDrawdown = magnitudes.reduce((sum, val) => sum + val, 0) / magnitudes.length;
        const avgDuration = durations.reduce((sum, val) => sum + val, 0) / durations.length;
        const avgRecoveryTime = recoveryTimes.length > 0 ? 
            recoveryTimes.reduce((sum, val) => sum + val, 0) / recoveryTimes.length : 0;
        
        // Find longest recovery
        const maxRecoveryTime = recoveryTimes.length > 0 ? Math.max(...recoveryTimes) : 0;
        
        // Calculate Calmar Ratio (annualized return / max drawdown)
        // In a real implementation, we'd use actual portfolio returns
        const annualizedReturn = 0.15; // 15% annual return (mock data)
        const calmarRatio = annualizedReturn / (maxDrawdown / 100);
        
        // Calculate Ulcer Index
        // Ulcer Index = sqrt(sum of squared drawdowns / number of periods)
        // For mock data, we'll just use a realistic value
        const ulcerIndex = 4.23;
        
        // Calculate recovery factor
        // Recovery Factor = cumulative return / max drawdown
        // For mock data, we'll just use a realistic value
        const recoveryFactor = 3.12;
        
        // Calculate pain index and ratio
        // For mock data, we'll just use realistic values
        const painIndex = 3.46;
        const painRatio = 2.15;
        
        // Calculate recovery efficiency
        // Recovery Efficiency = average recovery rate
        const recoveryEfficiency = 1.73;
        
        // Calculate time in drawdown (percentage of time spent in drawdown state)
        // For mock data, we'll just use a realistic value
        const timeInDrawdown = 32.5; // 32.5% of time spent in drawdown
        
        return {
            maxDrawdown: maxDrawdown,
            avgDrawdown: avgDrawdown,
            drawdownCount: drawdowns.length,
            avgDrawdownDuration: avgDuration,
            calmarRatio: calmarRatio,
            ulcerIndex: ulcerIndex,
            maxRecoveryTime: maxRecoveryTime,
            avgRecoveryTime: avgRecoveryTime,
            recoveryFactor: recoveryFactor,
            painIndex: painIndex,
            painRatio: painRatio,
            recoveryEfficiency: recoveryEfficiency,
            timeInDrawdown: timeInDrawdown
        };
    },
    
    updateMetricsUI: function(metrics) {
        // Update drawdown statistics
        document.getElementById('max-drawdown').textContent = utils.formatPercent(-metrics.maxDrawdown);
        document.getElementById('avg-drawdown-stats').textContent = utils.formatPercent(-metrics.avgDrawdown);
        document.getElementById('drawdown-count').textContent = metrics.drawdownCount;
        document.getElementById('avg-drawdown-duration').textContent = `${Math.round(metrics.avgDrawdownDuration)} days`;
        document.getElementById('calmar-ratio').textContent = metrics.calmarRatio.toFixed(2);
        document.getElementById('ulcer-index').textContent = metrics.ulcerIndex.toFixed(2);
        
        // Update recovery analysis
        document.getElementById('max-recovery-time').textContent = `${Math.round(metrics.maxRecoveryTime)} days`;
        document.getElementById('avg-recovery-time').textContent = `${Math.round(metrics.avgRecoveryTime)} days`;
        document.getElementById('recovery-factor').textContent = metrics.recoveryFactor.toFixed(2);
        document.getElementById('pain-index').textContent = metrics.painIndex.toFixed(2);
        document.getElementById('pain-ratio').textContent = metrics.painRatio.toFixed(2);
        document.getElementById('recovery-efficiency').textContent = metrics.recoveryEfficiency.toFixed(2);
        
        // Update footer summary
        document.getElementById('recovery-progress').textContent = '62.5%'; // Mock data
        document.getElementById('projected-recovery').textContent = 'April 12, 2025'; // Mock data
        document.getElementById('time-in-drawdown').textContent = `${metrics.timeInDrawdown}%`;
        
        // Update last updated timestamp
        const now = new Date();
        const minutes = Math.floor(Math.random() * 30) + 1; // Random minutes for demo
        document.getElementById('drawdown-last-updated').textContent = `${minutes} minutes ago`;
    }
};

/**
 * Settings and Export Functionality
 */
const drawdownSettings = {
    init: function() {
        this.setupEventListeners();
    },
    
    setupEventListeners: function() {
        // Modal close button
        const modalCloseBtn = document.querySelector('#drawdown-settings-modal .btn-close');
        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', () => {
                document.getElementById('drawdown-settings-modal').style.display = 'none';
            });
        }
        
        // Range sliders with live value update
        const rangeSliders = [
            'minor-drawdown-threshold',
            'moderate-drawdown-threshold',
            'major-drawdown-threshold',
            'severe-drawdown-threshold',
            'projection-confidence',
            'simulation-count'
        ];
        
        rangeSliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.addEventListener('input', () => {
                    const valueDisplay = slider.nextElementSibling;
                    if (valueDisplay) {
                        if (sliderId === 'simulation-count') {
                            valueDisplay.textContent = utils.formatNumber(slider.value);
                        } else {
                            valueDisplay.textContent = `${slider.value}%`;
                        }
                    }
                });
            }
        });
        
        // Export report button
        document.getElementById('export-drawdown-report').addEventListener('click', () => {
            this.exportReport();
        });
        
        // Stress test button
        document.getElementById('drawdown-stress-test').addEventListener('click', () => {
            this.runStressTest();
        });
        
        // Download data button
        document.getElementById('download-drawdown-data-btn').addEventListener('click', () => {
            this.downloadData();
        });
        
        // Expand panel button
        document.getElementById('expand-drawdown-panel-btn').addEventListener('click', () => {
            this.expandPanel();
        });
    },
    
    exportReport: function() {
        // In a real implementation, this would generate and download a PDF report
        console.log('Exporting drawdown analysis report');
        alert('Drawdown analysis report would be exported as PDF');
    },
    
    runStressTest: function() {
        // In a real implementation, this would run a stress test simulation
        console.log('Running drawdown stress test');
        alert('Drawdown stress test would be executed');
    },
    
    downloadData: function() {
        // In a real implementation, this would download data as CSV
        console.log('Downloading drawdown data');
        alert('Drawdown data would be downloaded as CSV');
    },
    
    expandPanel: function() {
        // In a real implementation, this would expand the panel to full screen
        console.log('Expanding drawdown panel');
        
        const panel = document.querySelector('.drawdown-analysis-panel');
        panel.classList.toggle('expanded');
        
        const expandBtn = document.getElementById('expand-drawdown-panel-btn');
        if (panel.classList.contains('expanded')) {
            expandBtn.innerHTML = '<i data-feather="minimize-2"></i>';
        } else {
            expandBtn.innerHTML = '<i data-feather="maximize-2"></i>';
        }
        
        // Re-render charts to fit new size
        underwaterChart.render();
        distributionChart.render();
        recoveryChart.render();
        projectionChart.render();
        
        // Refresh Feather icons
        if (typeof feather !== 'undefined' && feather.replace) {
            feather.replace();
        }
    }
};

/**
 * Initialize all components when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if drawdown analysis panel exists
    if (!document.querySelector('.drawdown-analysis-panel')) {
        return;
    }
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined' && feather.replace) {
        feather.replace();
    }
    
    // Initialize all drawdown analysis components
    underwaterChart.init();
    distributionChart.init();
    recoveryChart.init();
    projectionChart.init();
    drawdownTable.init();
    drawdownMetrics.init();
    drawdownSettings.init();
    
    console.log('Drawdown analysis panel initialized');
});