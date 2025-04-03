/**
 * Trade Entry/Exit Quality Analysis
 * 
 * This module provides functionality for the Trade Entry/Exit Quality Analysis component,
 * showing how trade entries and exits perform, analyzing slippage across market conditions,
 * revealing missed opportunities, and simulating what-if scenarios.
 */

class TradeQualityAnalysis {
    constructor(options = {}) {
        this.options = Object.assign({
            timingEfficiencyElementId: 'timing-efficiency-chart',
            slippageChartElementId: 'slippage-chart',
            missedOpportunityElementId: 'missed-opportunity-chart',
            whatIfElementId: 'what-if-chart',
            qualityTrendElementId: 'quality-trend-chart',
            colorScaleEfficiency: [
                [0, 'rgba(239, 68, 68, 0.7)'],     // Poor efficiency (red)
                [0.5, 'rgba(234, 179, 8, 0.7)'],   // Medium efficiency (yellow)
                [1, 'rgba(16, 185, 129, 0.7)']     // Good efficiency (green)
            ],
            colorScaleSlippage: [
                [0, 'rgba(16, 185, 129, 0.7)'],    // Low slippage (green)
                [0.5, 'rgba(234, 179, 8, 0.7)'],   // Medium slippage (yellow)
                [1, 'rgba(239, 68, 68, 0.7)']      // High slippage (red)
            ]
        }, options);
        
        // Initialize from UI control values
        this.strategy = document.getElementById('trade-quality-strategy')?.value || 'all';
        this.asset = document.getElementById('trade-quality-asset')?.value || 'all';
        this.timeRange = document.getElementById('trade-quality-timerange')?.value || '1m';
        this.tradeType = document.getElementById('trade-quality-type')?.value || 'all';
        
        // Initialize chart-specific settings
        this.efficiencyMetric = document.getElementById('efficiency-metric')?.value || 'percent';
        this.tradeCount = document.getElementById('trade-count')?.value || '50';
        this.slippageView = document.getElementById('slippage-view')?.value || 'heatmap';
        this.marketCondition = document.getElementById('market-condition')?.value || 'volatility';
        this.opportunityType = document.getElementById('opportunity-type')?.value || 'signals';
        this.opportunityThreshold = document.getElementById('opportunity-threshold')?.value || 'medium';
        this.scenarioType = document.getElementById('scenario-type')?.value || 'optimal-entry';
        this.impactMetric = document.getElementById('impact-metric')?.value || 'pnl';
        
        // Initialize data containers
        this.timingEfficiencyData = [];
        this.slippageData = [];
        this.missedOpportunityData = [];
        this.whatIfData = [];
        this.summaryMetrics = {};
        this.recommendationData = [];
        
        // Initialize chart objects
        this.timingEfficiencyChart = null;
        this.slippageChart = null;
        this.missedOpportunityChart = null;
        this.whatIfChart = null;
        this.qualityTrendChart = null;
        
        this.initialize();
    }
    
    initialize() {
        // Fetch data and initialize visualizations
        this.fetchData()
            .then(() => {
                this.initializeVisualizations();
                this.setupEventListeners();
                
                // Update feather icons if available
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            })
            .catch(error => {
                console.error('Error initializing Trade Quality Analysis:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchTimingEfficiencyData(),
            this.fetchSlippageData(),
            this.fetchMissedOpportunityData(),
            this.fetchWhatIfData(),
            this.fetchSummaryMetrics(),
            this.fetchRecommendations()
        ]);
    }
    
    fetchTimingEfficiencyData() {
        // Mock data for timing efficiency
        return new Promise(resolve => {
            setTimeout(() => {
                const assets = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'];
                const tradeTypes = ['entry', 'exit'];
                
                let data = [];
                let tradeId = 1;
                
                // Generate trades for the selected assets
                assets.forEach(asset => {
                    if (this.asset !== 'all' && this.asset !== asset) return;
                    
                    // Generate both entry and exit trades
                    tradeTypes.forEach(type => {
                        if (this.tradeType !== 'all' && this.tradeType !== type && 
                            !(this.tradeType === 'profitable' || this.tradeType === 'losing')) return;
                        
                        // Generate a random number of trades for this asset/type
                        const tradeCount = Math.floor(Math.random() * 20) + 10;
                        
                        for (let i = 0; i < tradeCount; i++) {
                            // Generate trade date within the selected time range
                            const daysAgo = this.getRandomDaysAgo(this.timeRange);
                            const tradeDate = new Date();
                            tradeDate.setDate(tradeDate.getDate() - daysAgo);
                            
                            // Efficiency between 0.4 and 1.0 (40% to 100%)
                            const efficiency = 0.4 + (Math.random() * 0.6);
                            
                            // Determine if trade was profitable (65% chance)
                            const profitable = Math.random() < 0.65;
                            
                            // Skip if filtering by profitable/losing
                            if ((this.tradeType === 'profitable' && !profitable) || 
                                (this.tradeType === 'losing' && profitable)) continue;
                            
                            // Calculate metrics based on efficiency
                            const pipsFromOptimal = Math.round((1 - efficiency) * 100 * (Math.random() + 0.5));
                            const timeDeviation = Math.round((1 - efficiency) * 60 * (Math.random() + 0.5)); // minutes
                            
                            data.push({
                                id: tradeId++,
                                asset: asset,
                                type: type,
                                date: tradeDate,
                                efficiency: efficiency,
                                pipsFromOptimal: pipsFromOptimal,
                                timeDeviation: timeDeviation, // minutes
                                profitable: profitable,
                                strategy: this.getRandomStrategy()
                            });
                        }
                    });
                });
                
                // Sort trades by date (newest first)
                data.sort((a, b) => b.date - a.date);
                
                // Limit to requested number of trades
                if (this.tradeCount !== 'all') {
                    data = data.slice(0, parseInt(this.tradeCount));
                }
                
                this.timingEfficiencyData = data;
                resolve(data);
            }, 300);
        });
    }
    
    fetchSlippageData() {
        // Mock data for slippage analysis
        return new Promise(resolve => {
            setTimeout(() => {
                let data = {
                    heatmap: {
                        x: [], // Market condition values
                        y: [], // Assets
                        z: []  // Slippage values
                    },
                    scatter: [],
                    bar: {
                        categories: [],
                        values: []
                    }
                };
                
                // Define condition metrics based on selected market condition
                const conditions = this.generateMarketConditions();
                
                // Define assets to include
                const assets = this.asset === 'all' ? 
                    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'] : 
                    [this.asset];
                
                // Generate heatmap data
                data.heatmap.x = conditions;
                data.heatmap.y = assets;
                
                const zValues = [];
                assets.forEach(asset => {
                    const assetRow = [];
                    conditions.forEach(condition => {
                        // Generate slippage based on condition
                        // Higher condition value generally means higher slippage
                        let baseSlippage;
                        if (asset === 'BTC-USD') baseSlippage = 0.05 + (Math.random() * 0.1);
                        else if (asset === 'ETH-USD') baseSlippage = 0.1 + (Math.random() * 0.15);
                        else if (asset === 'SOL-USD') baseSlippage = 0.15 + (Math.random() * 0.2);
                        else baseSlippage = 0.12 + (Math.random() * 0.18);
                        
                        // Adjust based on condition value (higher condition, higher slippage)
                        const conditionIndex = conditions.indexOf(condition);
                        const conditionFactor = conditionIndex / (conditions.length - 1);
                        const slippage = baseSlippage * (0.5 + conditionFactor);
                        
                        assetRow.push(slippage);
                        
                        // Also add to scatter data
                        data.scatter.push({
                            asset: asset,
                            condition: condition,
                            slippage: slippage,
                            volume: 50 + Math.random() * 200 // Trade volume for bubble size
                        });
                    });
                    zValues.push(assetRow);
                });
                data.heatmap.z = zValues;
                
                // Generate bar chart data
                if (this.marketCondition === 'timeofday') {
                    data.bar.categories = conditions;
                    
                    const values = [];
                    conditions.forEach((time, index) => {
                        // Calculate average slippage across assets for this time
                        let sum = 0;
                        assets.forEach((asset, assetIndex) => {
                            sum += zValues[assetIndex][index];
                        });
                        const avgSlippage = sum / assets.length;
                        values.push(avgSlippage);
                    });
                    data.bar.values = values;
                } else {
                    // For other conditions, bin into categories
                    const categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
                    data.bar.categories = categories;
                    
                    const binValues = [0, 0, 0, 0, 0];
                    const binCounts = [0, 0, 0, 0, 0];
                    
                    data.scatter.forEach(point => {
                        const normalizedCondition = conditions.indexOf(point.condition) / (conditions.length - 1);
                        const binIndex = Math.min(Math.floor(normalizedCondition * 5), 4);
                        binValues[binIndex] += point.slippage;
                        binCounts[binIndex]++;
                    });
                    
                    // Calculate averages
                    data.bar.values = binValues.map((val, idx) => 
                        binCounts[idx] > 0 ? val / binCounts[idx] : 0
                    );
                }
                
                this.slippageData = data;
                resolve(data);
            }, 350);
        });
    }
    
    fetchMissedOpportunityData() {
        // Mock data for missed opportunity analysis
        return new Promise(resolve => {
            setTimeout(() => {
                const data = {
                    trades: [],
                    stats: {
                        totalMissed: 0,
                        potentialValue: 0,
                        categories: {}
                    }
                };
                
                // Define assets to include
                const assets = this.asset === 'all' ? 
                    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'] : 
                    [this.asset];
                
                // Generate missed trades
                const tradeCount = 30 + Math.floor(Math.random() * 20);
                let totalValue = 0;
                
                for (let i = 0; i < tradeCount; i++) {
                    const asset = assets[Math.floor(Math.random() * assets.length)];
                    const daysAgo = this.getRandomDaysAgo(this.timeRange);
                    const date = new Date();
                    date.setDate(date.getDate() - daysAgo);
                    
                    // Generate missed opportunity details
                    let type, reason, impact, confidence;
                    
                    // Based on selected opportunity type
                    if (this.opportunityType === 'signals') {
                        type = 'missed_signal';
                        reason = this.getRandomElement([
                            'Signal threshold too high', 
                            'Signal delayed processing', 
                            'Conflicting indicators', 
                            'Risk limits prevented execution'
                        ]);
                        impact = 0.5 + (Math.random() * 1.5); // 0.5% to 2.0%
                        confidence = 0.6 + (Math.random() * 0.3); // 60% to 90%
                    } else if (this.opportunityType === 'entries') {
                        type = 'late_entry';
                        reason = this.getRandomElement([
                            'Delayed execution', 
                            'Slippage too high', 
                            'Order queued behind others', 
                            'Network latency'
                        ]);
                        impact = 0.3 + (Math.random() * 1.2); // 0.3% to 1.5%
                        confidence = 0.7 + (Math.random() * 0.25); // 70% to 95%
                    } else { // exits
                        type = 'early_exit';
                        reason = this.getRandomElement([
                            'Stop loss too tight', 
                            'Premature profit taking', 
                            'Overreaction to volatility', 
                            'Risk management override'
                        ]);
                        impact = 0.4 + (Math.random() * 1.3); // 0.4% to 1.7%
                        confidence = 0.65 + (Math.random() * 0.3); // 65% to 95%
                    }
                    
                    // Filter based on threshold
                    let severityLevel;
                    if (impact < 0.7) severityLevel = 'low';
                    else if (impact < 1.3) severityLevel = 'medium';
                    else severityLevel = 'high';
                    
                    if (this.opportunityThreshold !== 'low' && severityLevel === 'low') continue;
                    if (this.opportunityThreshold === 'high' && severityLevel !== 'high') continue;
                    
                    // Add to the dataset
                    data.trades.push({
                        id: i + 1,
                        asset: asset,
                        date: date,
                        type: type,
                        reason: reason,
                        impact: impact,
                        confidence: confidence,
                        severity: severityLevel,
                        strategy: this.getRandomStrategy()
                    });
                    
                    // Update totals
                    totalValue += impact;
                    
                    // Update categories
                    data.stats.categories[type] = (data.stats.categories[type] || 0) + 1;
                }
                
                // Update summary stats
                data.stats.totalMissed = data.trades.length;
                data.stats.potentialValue = totalValue.toFixed(2);
                
                // Sort by impact (highest first)
                data.trades.sort((a, b) => b.impact - a.impact);
                
                this.missedOpportunityData = data;
                resolve(data);
            }, 400);
        });
    }
    
    fetchWhatIfData() {
        // Mock data for what-if scenario simulation
        return new Promise(resolve => {
            setTimeout(() => {
                // Define the time range for the simulation
                const days = this.getTimeRangeDays(this.timeRange);
                const dataPoints = days + 1;
                
                const data = {
                    dates: [],
                    actual: [],
                    scenario: [],
                    difference: [], // Percentage difference
                    scenarioType: this.scenarioType,
                    metric: this.impactMetric,
                    summary: {
                        improvement: 0,
                        riskChange: 0,
                        tradeCount: 0,
                        confidenceLevel: 0
                    }
                };
                
                // Generate dates
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                for (let i = 0; i < dataPoints; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    data.dates.push(date);
                }
                
                // Generate baseline performance
                let cumValue = 100; // Start with $100
                
                if (this.impactMetric === 'pnl') {
                    // P&L curve
                    for (let i = 0; i < dataPoints; i++) {
                        data.actual.push(cumValue);
                        // Daily change between -2% and +3%
                        const dailyChange = -0.02 + (Math.random() * 0.05);
                        cumValue *= (1 + dailyChange);
                    }
                } else if (this.impactMetric === 'roi') {
                    // ROI over time (cumulative)
                    let cumRoi = 0;
                    for (let i = 0; i < dataPoints; i++) {
                        data.actual.push(cumRoi);
                        // Daily ROI between -1% and +1.5%
                        const dailyRoi = -0.01 + (Math.random() * 0.025);
                        cumRoi += dailyRoi;
                    }
                } else if (this.impactMetric === 'drawdown') {
                    // Max drawdown
                    let peak = 0;
                    let value = 100;
                    for (let i = 0; i < dataPoints; i++) {
                        // Calculate daily change
                        const dailyChange = -0.015 + (Math.random() * 0.03);
                        value *= (1 + dailyChange);
                        
                        // Update peak
                        if (value > peak) peak = value;
                        
                        // Calculate drawdown as percentage from peak
                        const drawdown = peak > 0 ? ((peak - value) / peak) * 100 : 0;
                        data.actual.push(drawdown);
                    }
                }
                
                // Generate scenario performance based on scenario type
                let improvementFactor;
                switch (this.scenarioType) {
                    case 'optimal-entry':
                        improvementFactor = 0.15; // 15% better
                        break;
                    case 'optimal-exit':
                        improvementFactor = 0.12; // 12% better
                        break;
                    case 'zero-slippage':
                        improvementFactor = 0.08; // 8% better
                        break;
                    case 'all-signals':
                        improvementFactor = 0.25; // 25% better
                        break;
                    default:
                        improvementFactor = 0.1;
                }
                
                // Apply improvement factor based on metric
                if (this.impactMetric === 'pnl' || this.impactMetric === 'roi') {
                    // For P&L and ROI, higher is better
                    for (let i = 0; i < dataPoints; i++) {
                        const improvement = data.actual[i] * improvementFactor;
                        data.scenario.push(data.actual[i] + improvement);
                        
                        // Calculate percentage difference
                        const pctDiff = (improvement / data.actual[i]) * 100;
                        data.difference.push(pctDiff);
                    }
                } else if (this.impactMetric === 'drawdown') {
                    // For drawdown, lower is better
                    for (let i = 0; i < dataPoints; i++) {
                        const improvement = data.actual[i] * improvementFactor;
                        data.scenario.push(Math.max(0, data.actual[i] - improvement));
                        
                        // Calculate percentage difference (negative is good for drawdown)
                        const pctDiff = -1 * (improvement / (data.actual[i] || 0.0001)) * 100;
                        data.difference.push(pctDiff);
                    }
                }
                
                // Calculate summary statistics
                data.summary.improvement = (improvementFactor * 100).toFixed(1) + '%';
                data.summary.riskChange = this.impactMetric === 'drawdown' ? 
                    'reduced by ' + (improvementFactor * 100).toFixed(1) + '%' : 
                    'unchanged';
                data.summary.tradeCount = Math.floor(days / 2) + Math.floor(Math.random() * 10);
                data.summary.confidenceLevel = (0.7 + (Math.random() * 0.2)).toFixed(2);
                
                this.whatIfData = data;
                resolve(data);
            }, 350);
        });
    }
    
    fetchSummaryMetrics() {
        // Mock data for summary metrics
        return new Promise(resolve => {
            setTimeout(() => {
                const metrics = {
                    avgEntryEfficiency: (0.6 + Math.random() * 0.2).toFixed(2) * 100 + '%',
                    avgExitEfficiency: (0.6 + Math.random() * 0.2).toFixed(2) * 100 + '%',
                    avgSlippage: (0.15 + Math.random() * 0.15).toFixed(2) + '%',
                    executionRate: (0.7 + Math.random() * 0.25).toFixed(2) * 100 + '%',
                    improvementPotential: '+' + (8 + Math.random() * 8).toFixed(1) + '%',
                    tradeCount: 80 + Math.floor(Math.random() * 100),
                    bestAsset: this.getRandomElement(['BTC/USD', 'ETH/USD', 'SOL/USD']) + 
                               ' (' + (75 + Math.floor(Math.random() * 15)) + '% efficiency)',
                    worstCondition: this.getRandomElement([
                        'High Vol. + Low Liquidity',
                        'Market Opens',
                        'News Events',
                        'Thin Order Books'
                    ]),
                    bestStrategy: this.getRandomElement(['Meta Strategy', 'ML Strategy', 'Sentiment Strategy']) + 
                                 ' (' + (70 + Math.floor(Math.random() * 15)) + '%)',
                    lastUpdated: Math.floor(Math.random() * 10) + ' minutes ago',
                    
                    // Quality trend data (for mini chart)
                    qualityTrend: []
                };
                
                // Generate trend data (30 days)
                for (let i = 0; i < 30; i++) {
                    metrics.qualityTrend.push({
                        date: new Date(new Date().setDate(new Date().getDate() - 29 + i)),
                        value: 0.6 + Math.sin(i / 5) * 0.2 + (Math.random() * 0.1) // Oscillating pattern with noise
                    });
                }
                
                this.summaryMetrics = metrics;
                resolve(metrics);
            }, 250);
        });
    }
    
    fetchRecommendations() {
        // Mock data for recommendations
        return new Promise(resolve => {
            setTimeout(() => {
                const recommendations = [
                    {
                        title: 'Reduce Slippage During High Volatility',
                        priority: 'high',
                        description: 'Slippage is 3.2x higher during volatility spikes. Consider using limit orders instead of market orders when volatility exceeds 2 standard deviations from the mean.'
                    },
                    {
                        title: 'Improve Exit Timing for BTC Trades',
                        priority: 'medium',
                        description: 'BTC exits are on average 15.3% away from optimal. Using trailing stops at 5% instead of fixed stops may improve exit efficiency.'
                    },
                    {
                        title: 'Increase Signal Response Time',
                        priority: 'low',
                        description: 'Average delay between signal generation and trade execution is 1.2 minutes. Optimize API connection or use WebSocket for faster execution.'
                    }
                ];
                
                // Add more recommendations based on settings
                if (this.strategy === 'sentiment_strategy') {
                    recommendations.push({
                        title: 'Adjust Sentiment Signal Thresholds',
                        priority: 'medium',
                        description: 'Sentiment-based entries have 12% lower efficiency than technical entries. Consider increasing the sentiment score threshold from 0.65 to 0.75 to improve signal quality.'
                    });
                } else if (this.strategy === 'ml_strategy') {
                    recommendations.push({
                        title: 'Re-train ML Model with Recent Data',
                        priority: 'medium',
                        description: 'ML strategy efficiency has declined 8% in the last 14 days. Last model training was 31 days ago. Consider retraining with the latest market data.'
                    });
                }
                
                if (this.asset === 'SOL-USD') {
                    recommendations.push({
                        title: 'Reduce Position Size for SOL-USD',
                        priority: 'high',
                        description: 'SOL-USD shows 2.4x higher slippage than other assets. Consider reducing position sizes by 15% or splitting orders to minimize market impact.'
                    });
                }
                
                this.recommendationData = recommendations;
                resolve(recommendations);
            }, 300);
        });
    }
    
    initializeVisualizations() {
        this.renderTimingEfficiency();
        this.renderSlippageChart();
        this.renderMissedOpportunity();
        this.renderWhatIfScenario();
        this.renderQualityTrend();
        this.updateSummaryMetrics();
        this.updateRecommendations();
    }
    
    renderTimingEfficiency() {
        const element = document.getElementById(this.options.timingEfficiencyElementId);
        if (!element || !this.timingEfficiencyData || this.timingEfficiencyData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data based on selected metric
        const tradeIds = this.timingEfficiencyData.map(d => d.id);
        
        let values, tooltips;
        if (this.efficiencyMetric === 'percent') {
            values = this.timingEfficiencyData.map(d => d.efficiency * 100);
            tooltips = this.timingEfficiencyData.map(d => 
                `${d.asset}<br>` +
                `Type: ${d.type.charAt(0).toUpperCase() + d.type.slice(1)}<br>` +
                `Efficiency: ${(d.efficiency * 100).toFixed(1)}%<br>` +
                `Date: ${d.date.toLocaleDateString()}<br>` +
                `${d.profitable ? 'Profitable' : 'Unprofitable'}`
            );
        } else if (this.efficiencyMetric === 'pips') {
            values = this.timingEfficiencyData.map(d => d.pipsFromOptimal);
            tooltips = this.timingEfficiencyData.map(d => 
                `${d.asset}<br>` +
                `Type: ${d.type.charAt(0).toUpperCase() + d.type.slice(1)}<br>` +
                `Pips from optimal: ${d.pipsFromOptimal}<br>` +
                `Date: ${d.date.toLocaleDateString()}<br>` +
                `${d.profitable ? 'Profitable' : 'Unprofitable'}`
            );
        } else if (this.efficiencyMetric === 'time') {
            values = this.timingEfficiencyData.map(d => d.timeDeviation);
            tooltips = this.timingEfficiencyData.map(d => 
                `${d.asset}<br>` +
                `Type: ${d.type.charAt(0).toUpperCase() + d.type.slice(1)}<br>` +
                `Time deviation: ${d.timeDeviation} mins<br>` +
                `Date: ${d.date.toLocaleDateString()}<br>` +
                `${d.profitable ? 'Profitable' : 'Unprofitable'}`
            );
        }
        
        // Create colors based on metric
        let colors;
        if (this.efficiencyMetric === 'percent') {
            // For percentage, higher is better
            colors = values.map(v => {
                const normalizedValue = v / 100;
                return this.getColorFromScale(normalizedValue, this.options.colorScaleEfficiency);
            });
        } else {
            // For pips and time, lower is better
            const maxValue = Math.max(...values);
            colors = values.map(v => {
                const normalizedValue = maxValue > 0 ? 1 - (v / maxValue) : 0;
                return this.getColorFromScale(normalizedValue, this.options.colorScaleEfficiency);
            });
        }
        
        // Create markers for profitable vs unprofitable
        const symbols = this.timingEfficiencyData.map(d => d.profitable ? 'circle' : 'x');
        const symbolSizes = this.timingEfficiencyData.map(d => d.profitable ? 10 : 12);
        
        // Create trace for bar chart
        const trace = {
            type: 'bar',
            x: tradeIds,
            y: values,
            text: tooltips,
            hoverinfo: 'text',
            marker: {
                color: colors,
                line: {
                    color: 'rgba(var(--border-rgb), 0.5)',
                    width: 1
                }
            }
        };
        
        // Add a horizontal line for target efficiency if using percent
        let shapes = [];
        if (this.efficiencyMetric === 'percent') {
            shapes.push({
                type: 'line',
                x0: -0.5,
                x1: tradeIds.length - 0.5,
                y0: 80,
                y1: 80,
                line: {
                    color: 'rgba(var(--success-rgb), 0.5)',
                    width: 2,
                    dash: 'dash'
                }
            });
        }
        
        // Different titles based on metric
        let title, yaxis;
        if (this.efficiencyMetric === 'percent') {
            title = 'Entry/Exit Timing Efficiency (% of Optimal)';
            yaxis = { title: 'Efficiency (%)', range: [0, 105] };
        } else if (this.efficiencyMetric === 'pips') {
            title = 'Entry/Exit Timing Efficiency (Pips from Optimal)';
            yaxis = { title: 'Pips from Optimal' };
        } else {
            title = 'Entry/Exit Timing Efficiency (Time Deviation)';
            yaxis = { title: 'Minutes from Optimal Time' };
        }
        
        // Layout configuration
        const layout = {
            title: title,
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Trades (Most Recent First)',
                showgrid: false,
                linecolor: 'var(--border-light)'
            },
            yaxis: {
                ...yaxis,
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)'
            },
            shapes: shapes
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.timingEfficiencyElementId, [trace], layout, config);
        this.timingEfficiencyChart = document.getElementById(this.options.timingEfficiencyElementId);
    }
    
    renderSlippageChart() {
        const element = document.getElementById(this.options.slippageChartElementId);
        if (!element || !this.slippageData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        let traces = [];
        let layout = {};
        
        // Switch based on selected view
        if (this.slippageView === 'heatmap') {
            // Heatmap visualization
            traces = [{
                type: 'heatmap',
                x: this.slippageData.heatmap.x,
                y: this.slippageData.heatmap.y,
                z: this.slippageData.heatmap.z,
                colorscale: [
                    [0, 'rgba(var(--success-rgb), 0.7)'],
                    [0.5, 'rgba(var(--warning-rgb), 0.7)'],
                    [1, 'rgba(var(--danger-rgb), 0.7)']
                ],
                hoverongaps: false,
                showscale: true,
                colorbar: {
                    title: 'Slippage %',
                    titleside: 'right',
                    thickness: 10
                }
            }];
            
            layout = {
                title: `Slippage by ${this.getConditionTitleText()}`,
                margin: { l: 70, r: 60, t: 40, b: 70 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: this.getConditionLabel(),
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Asset',
                    linecolor: 'var(--border-light)'
                }
            };
        } else if (this.slippageView === 'scatter') {
            // Scatter plot visualization
            const data = this.slippageData.scatter;
            
            // Group by asset
            const assets = [...new Set(data.map(d => d.asset))];
            
            // Create a trace for each asset
            assets.forEach((asset, index) => {
                const assetData = data.filter(d => d.asset === asset);
                
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    name: asset,
                    x: assetData.map(d => d.condition),
                    y: assetData.map(d => d.slippage * 100), // Convert to percentage
                    marker: {
                        size: assetData.map(d => Math.max(10, d.volume / 5)),
                        sizemode: 'diameter',
                        sizeref: 2.0,
                        color: assetData.map(d => {
                            // Color based on slippage
                            const normalizedSlippage = d.slippage / 0.4; // Normalize to 0-1
                            return this.getColorFromScale(normalizedSlippage, this.options.colorScaleSlippage);
                        })
                    },
                    text: assetData.map(d => 
                        `${d.asset}<br>` +
                        `${this.getConditionLabel()}: ${d.condition}<br>` +
                        `Slippage: ${(d.slippage * 100).toFixed(2)}%<br>` +
                        `Volume: ${d.volume.toFixed(0)}`
                    ),
                    hoverinfo: 'text'
                });
            });
            
            layout = {
                title: `Slippage by ${this.getConditionTitleText()}`,
                margin: { l: 60, r: 20, t: 40, b: 70 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: this.getConditionLabel(),
                    showgrid: false,
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Slippage (%)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)'
                },
                legend: {
                    orientation: 'h',
                    xanchor: 'center',
                    x: 0.5,
                    y: 1.12
                }
            };
        } else if (this.slippageView === 'bar') {
            // Bar chart visualization
            traces = [{
                type: 'bar',
                x: this.slippageData.bar.categories,
                y: this.slippageData.bar.values.map(v => v * 100), // Convert to percentage
                marker: {
                    color: this.slippageData.bar.values.map(v => {
                        // Color based on slippage
                        const normalizedSlippage = v / 0.4; // Normalize to 0-1
                        return this.getColorFromScale(normalizedSlippage, this.options.colorScaleSlippage);
                    })
                },
                text: this.slippageData.bar.values.map(v => `${(v * 100).toFixed(2)}%`),
                textposition: 'auto',
                hovertemplate: '%{x}<br>Avg. Slippage: %{y:.2f}%<extra></extra>'
            }];
            
            layout = {
                title: `Average Slippage by ${this.getConditionTitleText()}`,
                margin: { l: 60, r: 20, t: 40, b: 70 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: this.getConditionLabel(),
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Average Slippage (%)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)'
                }
            };
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.slippageChartElementId, traces, layout, config);
        this.slippageChart = document.getElementById(this.options.slippageChartElementId);
    }
    
    renderMissedOpportunity() {
        const element = document.getElementById(this.options.missedOpportunityElementId);
        if (!element || !this.missedOpportunityData || this.missedOpportunityData.trades.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        const data = this.missedOpportunityData;
        
        // For visual clarity, limit to top 20 opportunities
        const topTrades = data.trades.slice(0, 20);
        
        // Bubble chart showing missed trades by impact and confidence
        const trace = {
            type: 'scatter',
            mode: 'markers',
            x: topTrades.map(d => d.impact),
            y: topTrades.map(d => d.confidence * 100), // Convert to percentage
            marker: {
                size: topTrades.map(d => Math.max(10, d.impact * 15)),
                sizemode: 'area',
                sizeref: 0.1,
                color: topTrades.map(d => {
                    // Color based on type
                    if (d.type === 'missed_signal') return 'rgba(var(--info-rgb), 0.7)';
                    if (d.type === 'late_entry') return 'rgba(var(--warning-rgb), 0.7)';
                    return 'rgba(var(--danger-rgb), 0.7)';
                }),
                line: {
                    color: 'rgba(var(--border-rgb), 0.5)',
                    width: 1
                }
            },
            text: topTrades.map(d => 
                `${d.asset}<br>` +
                `Type: ${this.formatMissedOpportunityType(d.type)}<br>` +
                `Impact: ${d.impact.toFixed(2)}%<br>` +
                `Confidence: ${(d.confidence * 100).toFixed(0)}%<br>` +
                `Reason: ${d.reason}<br>` +
                `Date: ${d.date.toLocaleDateString()}`
            ),
            hoverinfo: 'text'
        };
        
        // Add a small bar chart to show missed opportunity by type
        const categoryNames = Object.keys(data.stats.categories).map(k => this.formatMissedOpportunityType(k));
        const categoryValues = Object.values(data.stats.categories);
        
        const barTrace = {
            type: 'bar',
            x: categoryNames,
            y: categoryValues,
            marker: {
                color: [
                    'rgba(var(--info-rgb), 0.7)',
                    'rgba(var(--warning-rgb), 0.7)',
                    'rgba(var(--danger-rgb), 0.7)'
                ]
            },
            xaxis: 'x2',
            yaxis: 'y2',
            showlegend: false
        };
        
        // Layout configuration with subplot for the bar chart
        const layout = {
            title: `Missed Opportunities (${this.opportunityType.charAt(0).toUpperCase() + this.opportunityType.slice(1)})`,
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Potential Impact (%)',
                domain: [0, 0.7],
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)'
            },
            yaxis: {
                title: 'Confidence (%)',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                range: [50, 100]
            },
            xaxis2: {
                domain: [0.75, 1],
                showticklabels: false,
                showgrid: false
            },
            yaxis2: {
                domain: [0, 1],
                showgrid: false,
                linecolor: 'var(--border-light)'
            },
            annotations: [
                {
                    x: 0.875,
                    y: 1.05,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Counts by Type',
                    showarrow: false,
                    font: {
                        size: 10
                    }
                },
                {
                    x: 0.5,
                    y: -0.1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Total potential improvement: ${data.stats.potentialValue}%`,
                    showarrow: false,
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
                    }
                }
            ]
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.missedOpportunityElementId, [trace, barTrace], layout, config);
        this.missedOpportunityChart = document.getElementById(this.options.missedOpportunityElementId);
    }
    
    renderWhatIfScenario() {
        const element = document.getElementById(this.options.whatIfElementId);
        if (!element || !this.whatIfData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        const data = this.whatIfData;
        
        // Create traces based on the metric
        let actualName, metricSuffix;
        if (data.metric === 'pnl') {
            actualName = 'Actual P&L';
            metricSuffix = '$';
        } else if (data.metric === 'roi') {
            actualName = 'Actual ROI';
            metricSuffix = '%';
        } else if (data.metric === 'drawdown') {
            actualName = 'Actual Drawdown';
            metricSuffix = '%';
        }
        
        // Format the scenario name
        const scenarioName = this.formatScenarioName(data.scenarioType);
        
        // Actual trace
        const actualTrace = {
            type: 'scatter',
            mode: 'lines',
            name: actualName,
            x: data.dates,
            y: data.actual,
            line: {
                color: 'rgba(var(--primary-rgb), 0.8)',
                width: 2
            },
            hovertemplate: '%{x|%b %d, %Y}<br>' + actualName + ': %{y:.2f}' + metricSuffix + '<extra></extra>'
        };
        
        // Scenario trace
        const scenarioTrace = {
            type: 'scatter',
            mode: 'lines',
            name: scenarioName,
            x: data.dates,
            y: data.scenario,
            line: {
                color: 'rgba(var(--success-rgb), 0.8)',
                width: 2
            },
            hovertemplate: '%{x|%b %d, %Y}<br>' + scenarioName + ': %{y:.2f}' + metricSuffix + '<extra></extra>'
        };
        
        // Fill the area between the traces
        const fillBetween = {
            type: 'scatter',
            mode: 'none',
            name: 'Improvement Area',
            x: data.dates.concat(data.dates.slice().reverse()),
            y: (() => {
                const isDrawdown = data.metric === 'drawdown';
                if (isDrawdown) {
                    // For drawdown, smaller is better, so fill top to bottom
                    return data.actual.concat(data.scenario.slice().reverse());
                } else {
                    // For P&L and ROI, bigger is better, so fill bottom to top
                    return data.scenario.concat(data.actual.slice().reverse());
                }
            })(),
            fill: 'toself',
            fillcolor: 'rgba(var(--success-rgb), 0.1)',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip'
        };
        
        // Layout configuration
        const layout = {
            title: `"What-If" Scenario: ${scenarioName}`,
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: '',
                showgrid: false,
                linecolor: 'var(--border-light)'
            },
            yaxis: {
                title: this.capitalizeFirst(data.metric),
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                ticksuffix: metricSuffix
            },
            legend: {
                orientation: 'h',
                xanchor: 'center',
                x: 0.5,
                y: 1.12
            },
            annotations: [
                {
                    x: 0.5,
                    y: -0.15,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Improvement: ${data.summary.improvement} | Confidence Level: ${data.summary.confidenceLevel} | Affected Trades: ${data.summary.tradeCount}`,
                    showarrow: false,
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
                    }
                }
            ]
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.whatIfElementId, [fillBetween, actualTrace, scenarioTrace], layout, config);
        this.whatIfChart = document.getElementById(this.options.whatIfElementId);
    }
    
    renderQualityTrend() {
        const element = document.getElementById(this.options.qualityTrendElementId);
        if (!element || !this.summaryMetrics || !this.summaryMetrics.qualityTrend) return;
        
        const trend = this.summaryMetrics.qualityTrend;
        
        // Create a simple sparkline
        const trace = {
            type: 'scatter',
            mode: 'lines',
            x: trend.map(d => d.date),
            y: trend.map(d => d.value * 100), // Convert to percentage
            line: {
                color: 'rgba(var(--primary-rgb), 0.8)',
                width: 2
            },
            hovertemplate: '%{x|%b %d}<br>Quality: %{y:.1f}%<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            margin: { l: 0, r: 0, t: 0, b: 0 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                showticklabels: false,
                showgrid: false,
                zeroline: false
            },
            yaxis: {
                showticklabels: false,
                showgrid: false,
                zeroline: false,
                range: [50, 90] // Fixed range for better visualization
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false,
            staticPlot: true
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.qualityTrendElementId, [trace], layout, config);
        this.qualityTrendChart = document.getElementById(this.options.qualityTrendElementId);
    }
    
    updateSummaryMetrics() {
        // Update summary metrics in the DOM
        if (!this.summaryMetrics) return;
        
        const metrics = this.summaryMetrics;
        
        // Update metric elements
        this.updateElementText('avg-entry-efficiency', metrics.avgEntryEfficiency);
        this.updateElementText('avg-exit-efficiency', metrics.avgExitEfficiency);
        this.updateElementText('avg-slippage', metrics.avgSlippage);
        this.updateElementText('execution-rate', metrics.executionRate);
        this.updateElementText('improvement-potential', metrics.improvementPotential);
        this.updateElementText('trade-count-value', metrics.tradeCount);
        
        // Update footer metrics
        this.updateElementText('best-asset', metrics.bestAsset);
        this.updateElementText('worst-condition', metrics.worstCondition);
        this.updateElementText('best-strategy', metrics.bestStrategy);
        this.updateElementText('trade-quality-last-updated', metrics.lastUpdated);
    }
    
    updateRecommendations() {
        // Update recommendations in the DOM
        const container = document.getElementById('trade-recommendations');
        if (!container || !this.recommendationData) return;
        
        // Clear existing content
        container.innerHTML = '';
        
        // Add recommendation items
        this.recommendationData.forEach(recommendation => {
            const item = document.createElement('div');
            item.className = `recommendation-item ${recommendation.priority}-priority`;
            
            const header = document.createElement('div');
            header.className = 'recommendation-header';
            
            const title = document.createElement('span');
            title.className = 'recommendation-title';
            title.textContent = recommendation.title;
            
            const priority = document.createElement('span');
            priority.className = 'recommendation-priority';
            priority.textContent = `${this.capitalizeFirst(recommendation.priority)} Impact`;
            
            header.appendChild(title);
            header.appendChild(priority);
            
            const description = document.createElement('div');
            description.className = 'recommendation-description';
            description.textContent = recommendation.description;
            
            item.appendChild(header);
            item.appendChild(description);
            
            container.appendChild(item);
        });
    }
    
    setupEventListeners() {
        // Strategy selector change
        const strategySelector = document.getElementById('trade-quality-strategy');
        if (strategySelector) {
            strategySelector.addEventListener('change', () => {
                this.strategy = strategySelector.value;
                this.refreshData();
            });
        }
        
        // Asset selector change
        const assetSelector = document.getElementById('trade-quality-asset');
        if (assetSelector) {
            assetSelector.addEventListener('change', () => {
                this.asset = assetSelector.value;
                this.refreshData();
            });
        }
        
        // Time range selector change
        const timeRangeSelector = document.getElementById('trade-quality-timerange');
        if (timeRangeSelector) {
            timeRangeSelector.addEventListener('change', () => {
                this.timeRange = timeRangeSelector.value;
                this.refreshData();
            });
        }
        
        // Trade type selector change
        const tradeTypeSelector = document.getElementById('trade-quality-type');
        if (tradeTypeSelector) {
            tradeTypeSelector.addEventListener('change', () => {
                this.tradeType = tradeTypeSelector.value;
                this.refreshData();
            });
        }
        
        // Efficiency metric selector change
        const efficiencyMetricSelector = document.getElementById('efficiency-metric');
        if (efficiencyMetricSelector) {
            efficiencyMetricSelector.addEventListener('change', () => {
                this.efficiencyMetric = efficiencyMetricSelector.value;
                this.renderTimingEfficiency();
            });
        }
        
        // Trade count selector change
        const tradeCountSelector = document.getElementById('trade-count');
        if (tradeCountSelector) {
            tradeCountSelector.addEventListener('change', () => {
                this.tradeCount = tradeCountSelector.value;
                this.fetchTimingEfficiencyData()
                    .then(() => {
                        this.renderTimingEfficiency();
                    });
            });
        }
        
        // Slippage view selector change
        const slippageViewSelector = document.getElementById('slippage-view');
        if (slippageViewSelector) {
            slippageViewSelector.addEventListener('change', () => {
                this.slippageView = slippageViewSelector.value;
                this.renderSlippageChart();
            });
        }
        
        // Market condition selector change
        const conditionSelector = document.getElementById('market-condition');
        if (conditionSelector) {
            conditionSelector.addEventListener('change', () => {
                this.marketCondition = conditionSelector.value;
                this.fetchSlippageData()
                    .then(() => {
                        this.renderSlippageChart();
                    });
            });
        }
        
        // Opportunity type selector change
        const opportunityTypeSelector = document.getElementById('opportunity-type');
        if (opportunityTypeSelector) {
            opportunityTypeSelector.addEventListener('change', () => {
                this.opportunityType = opportunityTypeSelector.value;
                this.fetchMissedOpportunityData()
                    .then(() => {
                        this.renderMissedOpportunity();
                    });
            });
        }
        
        // Opportunity threshold selector change
        const thresholdSelector = document.getElementById('opportunity-threshold');
        if (thresholdSelector) {
            thresholdSelector.addEventListener('change', () => {
                this.opportunityThreshold = thresholdSelector.value;
                this.fetchMissedOpportunityData()
                    .then(() => {
                        this.renderMissedOpportunity();
                    });
            });
        }
        
        // Scenario type selector change
        const scenarioTypeSelector = document.getElementById('scenario-type');
        if (scenarioTypeSelector) {
            scenarioTypeSelector.addEventListener('change', () => {
                this.scenarioType = scenarioTypeSelector.value;
                this.fetchWhatIfData()
                    .then(() => {
                        this.renderWhatIfScenario();
                    });
            });
        }
        
        // Impact metric selector change
        const impactMetricSelector = document.getElementById('impact-metric');
        if (impactMetricSelector) {
            impactMetricSelector.addEventListener('change', () => {
                this.impactMetric = impactMetricSelector.value;
                this.fetchWhatIfData()
                    .then(() => {
                        this.renderWhatIfScenario();
                    });
            });
        }
        
        // Settings button
        const settingsBtn = document.getElementById('trade-quality-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('trade-quality-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-trade-quality-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadTradeQualityData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-trade-quality-panel-btn');
        if (expandBtn) {
            expandBtn.addEventListener('click', () => {
                // Handle panel expansion (would be implemented in a real application)
                alert('Panel expansion will be implemented in the next phase');
            });
        }
        
        // Modal close buttons
        const closeButtons = document.querySelectorAll('[data-dismiss="modal"]');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                const modal = button.closest('.modal');
                if (modal) {
                    modal.style.display = 'none';
                }
            });
        });
        
        // Slider value display
        const slider = document.getElementById('missed-opportunity-detection');
        if (slider) {
            slider.addEventListener('input', function() {
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
            });
        }
        
        // Save settings button
        const saveSettingsBtn = document.getElementById('save-trade-quality-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                
                // Close the modal
                const modal = document.getElementById('trade-quality-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Learn more button
        const learnMoreBtn = document.getElementById('trade-quality-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Trade Quality Analysis documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-trade-quality-report');
        if (exportReportBtn) {
            exportReportBtn.addEventListener('click', () => {
                // This would generate a report
                alert('Report export functionality will be implemented in the next phase');
            });
        }
    }
    
    refreshData() {
        // Show loading overlays
        const elements = [
            this.options.timingEfficiencyElementId,
            this.options.slippageChartElementId,
            this.options.missedOpportunityElementId,
            this.options.whatIfElementId
        ];
        
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                const overlay = element.querySelector('.chart-overlay');
                if (overlay) {
                    overlay.textContent = 'Loading...';
                    overlay.style.display = 'flex';
                }
            }
        });
        
        // Fetch new data and update visualizations
        this.fetchData()
            .then(() => {
                this.renderTimingEfficiency();
                this.renderSlippageChart();
                this.renderMissedOpportunity();
                this.renderWhatIfScenario();
                this.renderQualityTrend();
                this.updateSummaryMetrics();
                this.updateRecommendations();
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                
                // Show error in overlays
                elements.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        const overlay = element.querySelector('.chart-overlay');
                        if (overlay) {
                            overlay.textContent = 'Error loading data';
                            overlay.style.display = 'flex';
                        }
                    }
                });
            });
    }
    
    downloadTradeQualityData() {
        // Create downloadable JSON file with trade quality data
        const data = {
            strategy: this.strategy,
            asset: this.asset,
            timeRange: this.timeRange,
            tradeType: this.tradeType,
            timingEfficiencyData: this.timingEfficiencyData,
            slippageData: this.slippageData,
            missedOpportunityData: this.missedOpportunityData,
            whatIfData: this.whatIfData,
            summaryMetrics: this.summaryMetrics,
            recommendationData: this.recommendationData
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `trade_quality_analysis_${this.strategy}_${this.asset}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    // Helper methods
    
    getTimeRangeDays(timeRange) {
        switch (timeRange) {
            case '1w': return 7;
            case '1m': return 30;
            case '3m': return 90;
            case '6m': return 180;
            case '1y': return 365;
            default: return 30;
        }
    }
    
    getRandomDaysAgo(timeRange) {
        const maxDays = this.getTimeRangeDays(timeRange);
        return Math.floor(Math.random() * maxDays);
    }
    
    getRandomStrategy() {
        if (this.strategy !== 'all') return this.strategy;
        
        const strategies = ['ml_strategy', 'sentiment_strategy', 'technical_strategy', 'meta_strategy'];
        return strategies[Math.floor(Math.random() * strategies.length)];
    }
    
    getRandomElement(array) {
        return array[Math.floor(Math.random() * array.length)];
    }
    
    generateMarketConditions() {
        let conditions = [];
        
        switch (this.marketCondition) {
            case 'volatility':
                conditions = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
                break;
            case 'volume':
                conditions = ['Very Thin', 'Below Avg', 'Average', 'Above Avg', 'Heavy'];
                break;
            case 'spread':
                conditions = ['<0.01%', '0.01-0.05%', '0.05-0.1%', '0.1-0.2%', '>0.2%'];
                break;
            case 'timeofday':
                conditions = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'];
                break;
            default:
                conditions = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
        }
        
        return conditions;
    }
    
    getConditionTitleText() {
        switch (this.marketCondition) {
            case 'volatility': return 'Market Volatility';
            case 'volume': return 'Trading Volume';
            case 'spread': return 'Bid-Ask Spread';
            case 'timeofday': return 'Time of Day (UTC)';
            default: return 'Market Conditions';
        }
    }
    
    getConditionLabel() {
        switch (this.marketCondition) {
            case 'volatility': return 'Volatility Level';
            case 'volume': return 'Volume Level';
            case 'spread': return 'Spread Range';
            case 'timeofday': return 'Time (UTC)';
            default: return 'Condition';
        }
    }
    
    formatMissedOpportunityType(type) {
        switch (type) {
            case 'missed_signal': return 'Missed Signal';
            case 'late_entry': return 'Late Entry';
            case 'early_exit': return 'Early Exit';
            default: return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    }
    
    formatScenarioName(scenarioType) {
        switch (scenarioType) {
            case 'optimal-entry': return 'Optimal Entries';
            case 'optimal-exit': return 'Optimal Exits';
            case 'zero-slippage': return 'Zero Slippage';
            case 'all-signals': return 'All Signals Executed';
            default: return scenarioType.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    }
    
    updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    getColorFromScale(value, scale) {
        // Get color from colorscale based on value (0-1)
        const colorscale = scale || this.options.colorScaleEfficiency;
        
        // Handle edge cases
        if (value <= colorscale[0][0]) return colorscale[0][1];
        if (value >= colorscale[colorscale.length - 1][0]) return colorscale[colorscale.length - 1][1];
        
        // Find the color segments the value falls between
        for (let i = 0; i < colorscale.length - 1; i++) {
            if (value >= colorscale[i][0] && value <= colorscale[i + 1][0]) {
                const startColor = colorscale[i][1];
                const endColor = colorscale[i + 1][1];
                
                // Normalize value to 0-1 range within this segment
                const segmentStart = colorscale[i][0];
                const segmentEnd = colorscale[i + 1][0];
                const normalizedValue = (value - segmentStart) / (segmentEnd - segmentStart);
                
                // Interpolate color
                return this.interpolateColor(startColor, endColor, normalizedValue);
            }
        }
        
        // Fallback
        return colorscale[0][1];
    }
    
    interpolateColor(startColor, endColor, ratio) {
        // Parse RGBA values
        const startRgba = this.parseRgba(startColor);
        const endRgba = this.parseRgba(endColor);
        
        // Interpolate
        const r = Math.round(startRgba.r + (endRgba.r - startRgba.r) * ratio);
        const g = Math.round(startRgba.g + (endRgba.g - startRgba.g) * ratio);
        const b = Math.round(startRgba.b + (endRgba.b - startRgba.b) * ratio);
        const a = startRgba.a + (endRgba.a - startRgba.a) * ratio;
        
        return `rgba(${r}, ${g}, ${b}, ${a})`;
    }
    
    parseRgba(color) {
        // Parse rgba color string
        const match = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/);
        if (match) {
            return {
                r: parseInt(match[1]),
                g: parseInt(match[2]),
                b: parseInt(match[3]),
                a: parseFloat(match[4])
            };
        }
        
        // Fallback
        return { r: 0, g: 0, b: 0, a: 1 };
    }
}

// Initialize the Trade Quality Analysis component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of TradeQualityAnalysis
    const tradeQualityAnalysis = new TradeQualityAnalysis();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});