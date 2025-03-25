/**
 * Market Regime Analysis Dashboard
 * 
 * This module provides functionality for the Market Regime Analysis Dashboard component
 * including regime probability heat calendar, regime transition matrix, regime-specific 
 * performance metrics, and regime detection confidence indicators.
 */

class MarketRegimeAnalysis {
    constructor(options = {}) {
        this.options = Object.assign({
            calendarElementId: 'regime-calendar',
            transitionElementId: 'regime-transition-matrix',
            performanceElementId: 'regime-performance',
            confidenceElementId: 'regime-confidence',
            regimeColors: {
                bullish: 'rgba(72, 187, 120, 0.8)',    // Success color
                bearish: 'rgba(239, 68, 68, 0.8)',     // Danger color
                ranging: 'rgba(246, 173, 85, 0.8)',    // Warning color
                volatile: 'rgba(66, 153, 225, 0.8)'    // Info color
            },
        }, options);
        
        this.currentAsset = document.getElementById('regime-asset')?.value || 'BTC-USD';
        this.currentTimeframe = document.getElementById('regime-timeframe')?.value || '1d';
        this.currentPeriod = document.getElementById('regime-period')?.value || '180';
        this.currentRegimeModel = document.getElementById('regime-type')?.value || 'ensemble';
        
        this.performanceTab = 'returns';
        this.calendarData = null;
        this.transitionData = null;
        this.performanceData = null;
        this.confidenceData = null;
        
        this.initialize();
    }
    
    initialize() {
        // Fetch data and initialize visualizations
        this.fetchData()
            .then(() => {
                this.initializeCharts();
                this.setupEventListeners();
                
                // Update feather icons if available
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            })
            .catch(error => {
                console.error('Error initializing Market Regime Analysis:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchCalendarData(),
            this.fetchTransitionData(),
            this.fetchPerformanceData(),
            this.fetchConfidenceData()
        ]);
    }
    
    fetchCalendarData() {
        // Mock data for the regime calendar heat map
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate 180 days (or as specified by currentPeriod) of regime data
                const days = parseInt(this.currentPeriod);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const data = [];
                const regimes = ['bullish', 'bearish', 'ranging', 'volatile'];
                
                // Generate regime probabilities for each day
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create regime probabilities
                    const probabilities = {};
                    
                    // Base probabilities affected by asset and timeframe
                    let baseProbabilities;
                    if (this.currentAsset === 'BTC-USD') {
                        // For this example, we assume BTC has been more bullish recently
                        baseProbabilities = [0.5, 0.2, 0.2, 0.1]; // [bullish, bearish, ranging, volatile]
                    } else if (this.currentAsset === 'ETH-USD') {
                        baseProbabilities = [0.4, 0.3, 0.2, 0.1];
                    } else if (this.currentAsset === 'SOL-USD') {
                        baseProbabilities = [0.3, 0.3, 0.1, 0.3];
                    } else if (this.currentAsset === 'market') {
                        baseProbabilities = [0.4, 0.2, 0.3, 0.1];
                    } else {
                        baseProbabilities = [0.35, 0.25, 0.25, 0.15];
                    }
                    
                    // Add some randomness and cycles
                    // Create a cycle effect - markets tend to rotate between regimes
                    const cyclePhase = i / (days / 4); // Complete 4 cycles over the period
                    const cycleEffect = Math.sin(cyclePhase * 2 * Math.PI);
                    
                    // Create a trend effect - more recent time periods have different probabilities
                    const trendEffect = i / days; // 0 to 1 from oldest to newest
                    
                    // Apply these effects to the probabilities
                    let adjustedProbabilities = [...baseProbabilities];
                    
                    // Cycle effect: shift probability between bullish and bearish
                    const cycleAdjustment = cycleEffect * 0.2; // Scale the effect
                    adjustedProbabilities[0] += cycleAdjustment; // Add to bullish
                    adjustedProbabilities[1] -= cycleAdjustment; // Subtract from bearish
                    
                    // Trend effect: gradually become more bullish in recent periods
                    const trendAdjustment = trendEffect * 0.15; // Scale the effect
                    adjustedProbabilities[0] += trendAdjustment; // Add to bullish
                    
                    // Add randomness
                    adjustedProbabilities = adjustedProbabilities.map(p => {
                        return p + (Math.random() - 0.5) * 0.1; // Add random fluctuation
                    });
                    
                    // Normalize probabilities to ensure they sum to 1
                    const sum = adjustedProbabilities.reduce((a, b) => a + b, 0);
                    adjustedProbabilities = adjustedProbabilities.map(p => p / sum);
                    
                    // Assign to regimes
                    regimes.forEach((regime, index) => {
                        probabilities[regime] = adjustedProbabilities[index];
                    });
                    
                    // Determine dominant regime (highest probability)
                    const dominantRegime = regimes[adjustedProbabilities.indexOf(Math.max(...adjustedProbabilities))];
                    
                    data.push({
                        date: date,
                        probabilities: probabilities,
                        dominantRegime: dominantRegime
                    });
                }
                
                this.calendarData = data;
                resolve(data);
            }, 300);
        });
    }
    
    fetchTransitionData() {
        // Mock data for the regime transition matrix
        return new Promise(resolve => {
            setTimeout(() => {
                const regimes = ['bullish', 'bearish', 'ranging', 'volatile'];
                const matrix = [];
                
                // Base transition probabilities
                const baseTransitions = [
                    [0.70, 0.10, 0.15, 0.05], // From bullish to others
                    [0.15, 0.65, 0.10, 0.10], // From bearish to others
                    [0.25, 0.15, 0.55, 0.05], // From ranging to others
                    [0.20, 0.30, 0.15, 0.35]  // From volatile to others
                ];
                
                // Adjust based on asset
                let assetFactor = 0;
                if (this.currentAsset === 'BTC-USD') {
                    assetFactor = 0.05; // More likely to stay in current regime
                } else if (this.currentAsset === 'SOL-USD') {
                    assetFactor = -0.05; // More likely to change regimes
                }
                
                // Create transition matrix with labels
                regimes.forEach((fromRegime, i) => {
                    const row = {
                        from: fromRegime,
                        to: {}
                    };
                    
                    regimes.forEach((toRegime, j) => {
                        // Apply asset adjustment to diagonal elements (staying in same regime)
                        let probability = baseTransitions[i][j];
                        if (i === j) {
                            probability += assetFactor;
                        } else {
                            // Distribute the adjustment proportionally to off-diagonal elements
                            probability -= assetFactor / (regimes.length - 1);
                        }
                        
                        // Ensure probabilities are valid
                        probability = Math.max(0.05, Math.min(0.95, probability));
                        row.to[toRegime] = probability;
                    });
                    
                    // Normalize row probabilities
                    const sum = Object.values(row.to).reduce((a, b) => a + b, 0);
                    Object.keys(row.to).forEach(key => {
                        row.to[key] = row.to[key] / sum;
                    });
                    
                    matrix.push(row);
                });
                
                this.transitionData = matrix;
                resolve(matrix);
            }, 400);
        });
    }
    
    fetchPerformanceData() {
        // Mock data for regime performance metrics
        return new Promise(resolve => {
            setTimeout(() => {
                const regimes = ['bullish', 'bearish', 'ranging', 'volatile'];
                const metrics = {
                    returns: {},
                    volatility: {},
                    sharpe: {},
                    drawdown: {}
                };
                
                // Base performance metrics for each regime
                const baseMetrics = {
                    // [returns, volatility, sharpe, max drawdown]
                    bullish: [0.15, 0.20, 0.75, 0.12],
                    bearish: [-0.20, 0.35, -0.57, 0.30],
                    ranging: [0.03, 0.12, 0.25, 0.08],
                    volatile: [-0.05, 0.40, -0.12, 0.22]
                };
                
                // Adjust based on asset
                const assetAdjustments = {
                    'BTC-USD': [0.05, 0.05, 0.1, 0.03],  // Higher returns, higher volatility
                    'ETH-USD': [0.08, 0.07, 0.15, 0.05], // Even higher returns and volatility
                    'SOL-USD': [0.10, 0.10, 0.2, 0.08],  // Highest returns and volatility
                    'market': [0, 0, 0, 0],              // Baseline
                    'portfolio': [-0.02, -0.05, 0.05, -0.02] // Lower volatility but also lower returns
                };
                
                const adjustment = assetAdjustments[this.currentAsset] || [0, 0, 0, 0];
                
                // Create performance data
                regimes.forEach(regime => {
                    const baseMetric = baseMetrics[regime];
                    
                    // Apply asset-specific adjustments
                    metrics.returns[regime] = baseMetric[0] + adjustment[0];
                    metrics.volatility[regime] = baseMetric[1] + adjustment[1];
                    metrics.sharpe[regime] = baseMetric[2] + adjustment[2];
                    metrics.drawdown[regime] = baseMetric[3] + adjustment[3];
                    
                    // Add some random variation
                    metrics.returns[regime] += (Math.random() - 0.5) * 0.05;
                    metrics.volatility[regime] += (Math.random() - 0.5) * 0.03;
                    metrics.sharpe[regime] += (Math.random() - 0.5) * 0.1;
                    metrics.drawdown[regime] += (Math.random() - 0.5) * 0.02;
                    
                    // Ensure values are reasonable
                    metrics.volatility[regime] = Math.max(0.05, metrics.volatility[regime]);
                    metrics.drawdown[regime] = Math.max(0.01, metrics.drawdown[regime]);
                });
                
                this.performanceData = metrics;
                resolve(metrics);
            }, 350);
        });
    }
    
    fetchConfidenceData() {
        // Mock data for regime detection confidence
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(this.currentPeriod);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const data = [];
                const models = ['hmm', 'volatility', 'trend', 'momentum', 'ensemble'];
                
                // Create confidence scores over time
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    const entry = {
                        date: date,
                        confidence: {}
                    };
                    
                    // Base confidence values that change over time with some pattern
                    const cyclePhase = i / (days / 3); // Complete 3 cycles over the period
                    const cycleEffect = (Math.sin(cyclePhase * 2 * Math.PI) + 1) / 2; // 0 to 1
                    
                    // Each model has different baseline confidence and different response to the cycle
                    const baseConfidence = {
                        hmm: 0.7 + cycleEffect * 0.15,
                        volatility: 0.65 + cycleEffect * 0.2,
                        trend: 0.6 + (1 - cycleEffect) * 0.25,
                        momentum: 0.55 + cycleEffect * 0.3,
                        ensemble: 0.75 + Math.min(cycleEffect, 1 - cycleEffect) * 0.15
                    };
                    
                    // Add some random noise to each model's confidence
                    models.forEach(model => {
                        entry.confidence[model] = baseConfidence[model] + (Math.random() - 0.5) * 0.1;
                        // Ensure confidence is between 0 and 1
                        entry.confidence[model] = Math.min(0.98, Math.max(0.4, entry.confidence[model]));
                    });
                    
                    data.push(entry);
                }
                
                this.confidenceData = data;
                resolve(data);
            }, 450);
        });
    }
    
    initializeCharts() {
        this.renderCalendarChart();
        this.renderTransitionMatrix();
        this.renderPerformanceChart();
        this.renderConfidenceChart();
    }
    
    renderCalendarChart() {
        const element = document.getElementById(this.options.calendarElementId);
        if (!element || !this.calendarData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for heatmap calendar
        const months = {};
        const data = [];
        
        // Group data by month and prepare z values
        this.calendarData.forEach(item => {
            const date = item.date;
            const month = date.toLocaleString('default', { month: 'short' }) + ' ' + date.getFullYear();
            
            if (!months[month]) {
                months[month] = {
                    weekdays: {},
                    z: []
                };
            }
            
            const weekday = date.getDay(); // 0 = Sunday, 6 = Saturday
            const dayOfMonth = date.getDate();
            
            // Track what weekday each date falls on
            if (!months[month].weekdays[weekday]) {
                months[month].weekdays[weekday] = [];
            }
            months[month].weekdays[weekday].push(dayOfMonth);
            
            // Get dominant regime and push it to data
            const regime = item.dominantRegime;
            const color = this.options.regimeColors[regime] || 'rgba(128, 128, 128, 0.8)';
            
            data.push({
                date: date.toISOString().split('T')[0],
                regime: regime,
                color: color,
                // Probabilities for hover information
                bullish: item.probabilities.bullish,
                bearish: item.probabilities.bearish,
                ranging: item.probabilities.ranging,
                volatile: item.probabilities.volatile
            });
        });
        
        // Create calendar heatmap
        const sortedMonths = Object.keys(months).sort((a, b) => {
            const dateA = new Date(a);
            const dateB = new Date(b);
            return dateA - dateB;
        });
        
        // Create trace for each month
        const traces = [];
        const xLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
        // Prepare data for each month
        sortedMonths.forEach(month => {
            const monthData = months[month];
            const z = Array(6).fill().map(() => Array(7).fill(null)); // 6 weeks, 7 days
            
            // Fill in the z array with the day of month
            // This creates a calendar layout for each month
            for (let weekday = 0; weekday < 7; weekday++) {
                if (monthData.weekdays[weekday]) {
                    const days = monthData.weekdays[weekday].sort((a, b) => a - b);
                    days.forEach((day, i) => {
                        const weekIndex = Math.floor(i / 1); // Adjust how many weeks to show
                        z[weekIndex][weekday] = day;
                    });
                }
            }
            
            // Transpose the z array to match the expected format
            const zTransposed = z[0].map((_, colIndex) => z.map(row => row[colIndex]));
            
            // Push trace for each month
            traces.push({
                name: month,
                x: xLabels,
                y: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                z: zTransposed,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgba(var(--bg-rgb), 0.8)'],
                    [1, 'rgba(var(--primary-rgb), 0.8)']
                ],
                // Custom hover template
                hovertemplate: '%{text}<extra></extra>',
                text: zTransposed.map((row, i) => 
                    row.map((value, j) => {
                        if (value === null) return '';
                        
                        // Find the corresponding data entry
                        const date = new Date(`${month.split(' ')[1]}-${month.split(' ')[0]}-${value}`);
                        const dateString = date.toISOString().split('T')[0];
                        const item = data.find(d => d.date === dateString);
                        
                        if (!item) return `${month} ${value}`;
                        
                        return `<b>${month} ${value}</b><br>` +
                               `Regime: ${item.regime.charAt(0).toUpperCase() + item.regime.slice(1)}<br>` +
                               `Bullish: ${Math.round(item.bullish * 100)}%<br>` +
                               `Bearish: ${Math.round(item.bearish * 100)}%<br>` +
                               `Ranging: ${Math.round(item.ranging * 100)}%<br>` +
                               `Volatile: ${Math.round(item.volatile * 100)}%`;
                    })
                ),
                showscale: false
            });
        });
        
        // Create a custom marker trace to show the regime colors
        const markerTrace = {
            type: 'scatter',
            mode: 'markers',
            x: [],
            y: [],
            marker: {
                size: 15,
                color: []
            },
            hoverinfo: 'skip',
            showlegend: false
        };
        
        // Add marker for each day with its regime color
        data.forEach(item => {
            const date = new Date(item.date);
            const month = date.toLocaleString('default', { month: 'short' }) + ' ' + date.getFullYear();
            const weekday = date.getDay();
            const dayOfMonth = date.getDate();
            
            // Convert this to the x,y coordinates in the heatmap
            const monthIndex = sortedMonths.indexOf(month);
            if (monthIndex !== -1) {
                markerTrace.x.push(weekday);
                markerTrace.y.push(Math.floor(dayOfMonth / 7));
                markerTrace.marker.color.push(item.color);
            }
        });
        
        // Layout configuration
        const layout = {
            title: 'Regime Calendar',
            grid: {rows: Math.min(3, sortedMonths.length), columns: Math.ceil(sortedMonths.length / 3)},
            xaxis: {
                showgrid: false,
                tickangle: -90
            },
            yaxis: {
                showgrid: false,
                autorange: 'reversed'
            },
            margin: {l: 20, r: 20, t: 40, b: 20},
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            hoverlabel: {
                bgcolor: 'var(--tooltip-bg)',
                bordercolor: 'var(--tooltip-border)',
                font: {
                    color: 'var(--tooltip-text)',
                    size: 11
                }
            },
            calendar: true
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.calendarElementId, traces, layout, config);
    }
    
    renderTransitionMatrix() {
        const element = document.getElementById(this.options.transitionElementId);
        if (!element || !this.transitionData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for heatmap
        const regimes = ['bullish', 'bearish', 'ranging', 'volatile'];
        const x = regimes.map(r => r.charAt(0).toUpperCase() + r.slice(1));
        const y = [...x]; // Same labels for y axis
        
        const z = regimes.map(fromRegime => {
            const row = this.transitionData.find(r => r.from === fromRegime);
            return regimes.map(toRegime => row ? row.to[toRegime] : 0);
        });
        
        // Text to display in cells
        const text = z.map(row => row.map(value => `${Math.round(value * 100)}%`));
        
        // Create heatmap trace
        const trace = {
            x: x,
            y: y,
            z: z,
            type: 'heatmap',
            colorscale: [
                [0, 'rgba(var(--bg-rgb), 0.8)'],
                [0.5, 'rgba(var(--primary-light-rgb), 0.8)'],
                [1, 'rgba(var(--primary-rgb), 0.9)']
            ],
            text: text,
            texttemplate: '%{text}',
            hovertemplate: 'From: %{y}<br>To: %{x}<br>Probability: %{z:.1%}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Regime Transition Probabilities',
            xaxis: {
                title: 'To Regime',
                showgrid: false
            },
            yaxis: {
                title: 'From Regime',
                showgrid: false,
                autorange: 'reversed'
            },
            annotations: [],
            margin: {l: 80, r: 20, t: 40, b: 60},
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            }
        };
        
        // Add text annotations
        for (let i = 0; i < y.length; i++) {
            for (let j = 0; j < x.length; j++) {
                const textColor = z[i][j] > 0.5 ? 'white' : 'var(--text)';
                layout.annotations.push({
                    x: x[j],
                    y: y[i],
                    text: text[i][j],
                    font: {
                        color: textColor
                    },
                    showarrow: false
                });
            }
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.transitionElementId, [trace], layout, config);
    }
    
    renderPerformanceChart() {
        const element = document.getElementById(this.options.performanceElementId);
        if (!element || !this.performanceData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Get the current tab data
        const metric = this.performanceTab; // 'returns', 'volatility', 'sharpe', or 'drawdown'
        const data = this.performanceData[metric];
        
        // Prepare data for bar chart
        const regimes = Object.keys(data);
        const values = regimes.map(regime => data[regime]);
        
        // Calculate colors based on metric
        const colors = regimes.map(regime => {
            const baseColor = this.options.regimeColors[regime] || 'rgba(128, 128, 128, 0.8)';
            return baseColor.replace('0.8', '0.7'); // Slightly more transparent
        });
        
        // Format labels based on metric
        let textTemplate = '';
        let title = '';
        let yaxisTitle = '';
        
        switch(metric) {
            case 'returns':
                textTemplate = '%{y:.1%}';
                title = 'Average Returns by Regime';
                yaxisTitle = 'Return';
                break;
            case 'volatility':
                textTemplate = '%{y:.1%}';
                title = 'Volatility by Regime';
                yaxisTitle = 'Volatility';
                break;
            case 'sharpe':
                textTemplate = '%{y:.2f}';
                title = 'Sharpe Ratio by Regime';
                yaxisTitle = 'Sharpe Ratio';
                break;
            case 'drawdown':
                textTemplate = '%{y:.1%}';
                title = 'Maximum Drawdown by Regime';
                yaxisTitle = 'Drawdown';
                break;
        }
        
        // Create bar chart trace
        const trace = {
            x: regimes.map(r => r.charAt(0).toUpperCase() + r.slice(1)),
            y: values,
            type: 'bar',
            marker: {
                color: colors
            },
            text: values.map(v => {
                if (metric === 'returns' || metric === 'sharpe') {
                    return v >= 0 ? '▲' : '▼';
                }
                return '';
            }),
            textposition: 'outside',
            hovertemplate: `${yaxisTitle}: ${textTemplate}<extra></extra>`
        };
        
        // Layout configuration
        const layout = {
            title: title,
            xaxis: {
                title: 'Market Regime',
                showgrid: false
            },
            yaxis: {
                title: yaxisTitle,
                showgrid: true,
                tickformat: metric === 'returns' || metric === 'volatility' || metric === 'drawdown' ? ',.0%' : '.2f'
            },
            margin: {l: 50, r: 20, t: 40, b: 60},
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.performanceElementId, [trace], layout, config);
    }
    
    renderConfidenceChart() {
        const element = document.getElementById(this.options.confidenceElementId);
        if (!element || !this.confidenceData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for line chart
        const dates = this.confidenceData.map(item => item.date);
        const models = Object.keys(this.confidenceData[0].confidence);
        
        // Define colors and names for each model
        const modelInfo = {
            hmm: { name: 'HMM', color: 'rgba(66, 153, 225, 0.8)' },
            volatility: { name: 'Volatility', color: 'rgba(239, 68, 68, 0.8)' },
            trend: { name: 'Trend', color: 'rgba(72, 187, 120, 0.8)' },
            momentum: { name: 'Momentum', color: 'rgba(246, 173, 85, 0.8)' },
            ensemble: { name: 'Ensemble', color: 'rgba(160, 174, 192, 0.8)' }
        };
        
        // Create line traces for each model
        const traces = models.map(model => {
            const values = this.confidenceData.map(item => item.confidence[model]);
            const info = modelInfo[model] || { name: model, color: 'rgba(128, 128, 128, 0.8)' };
            
            return {
                x: dates,
                y: values,
                type: 'scatter',
                mode: 'lines',
                name: info.name,
                line: {
                    color: info.color,
                    width: model === this.currentRegimeModel ? 3 : 1.5
                },
                hovertemplate: 'Date: %{x|%b %d, %Y}<br>Confidence: %{y:.1%}<extra>' + info.name + '</extra>'
            };
        });
        
        // Layout configuration
        const layout = {
            title: 'Model Confidence Over Time',
            xaxis: {
                title: 'Date',
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                title: 'Confidence Score',
                showgrid: true,
                range: [0.4, 1],
                tickformat: ',.0%'
            },
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            },
            margin: {l: 50, r: 20, t: 40, b: 80},
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            hovermode: 'x unified'
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.confidenceElementId, traces, layout, config);
    }
    
    setupEventListeners() {
        // Asset selector change
        const assetSelector = document.getElementById('regime-asset');
        if (assetSelector) {
            assetSelector.addEventListener('change', () => {
                this.currentAsset = assetSelector.value;
                this.refreshData();
            });
        }
        
        // Timeframe selector change
        const timeframeSelector = document.getElementById('regime-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.currentTimeframe = timeframeSelector.value;
                this.refreshData();
            });
        }
        
        // Period selector change
        const periodSelector = document.getElementById('regime-period');
        if (periodSelector) {
            periodSelector.addEventListener('change', () => {
                this.currentPeriod = periodSelector.value;
                this.refreshData();
            });
        }
        
        // Regime type selector change
        const regimeTypeSelector = document.getElementById('regime-type');
        if (regimeTypeSelector) {
            regimeTypeSelector.addEventListener('change', () => {
                this.currentRegimeModel = regimeTypeSelector.value;
                this.refreshData();
            });
        }
        
        // Performance metric tabs
        const regimeTabs = document.querySelectorAll('.regime-tab');
        regimeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                regimeTabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');
                // Update the current tab
                this.performanceTab = tab.dataset.tab;
                // Re-render the performance chart
                this.renderPerformanceChart();
            });
        });
        
        // Regime settings button
        const settingsBtn = document.getElementById('regime-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('regime-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-regime-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadRegimeData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-regime-panel-btn');
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
        
        // Modal save settings button
        const saveSettingsBtn = document.getElementById('save-regime-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                // Close the modal
                const modal = document.getElementById('regime-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Custom regimes checkbox
        const customRegimesCheckbox = document.getElementById('use-custom-regimes');
        const customRegimesTextarea = document.getElementById('custom-regimes');
        if (customRegimesCheckbox && customRegimesTextarea) {
            customRegimesCheckbox.addEventListener('change', () => {
                customRegimesTextarea.disabled = !customRegimesCheckbox.checked;
            });
        }
        
        // Slider value updates
        const sliders = document.querySelectorAll('.form-range');
        sliders.forEach(slider => {
            slider.addEventListener('input', function() {
                // Update the value display
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = `${this.value}%`;
                }
                
                // Enforce that all weights sum to 100%
                const allSliders = document.querySelectorAll('.form-range');
                const currentTotal = Array.from(allSliders).reduce((sum, s) => sum + parseInt(s.value), 0);
                
                if (currentTotal !== 100) {
                    // Adjust other sliders proportionally
                    const diff = 100 - currentTotal;
                    const otherSliders = Array.from(allSliders).filter(s => s !== slider);
                    const otherTotal = otherSliders.reduce((sum, s) => sum + parseInt(s.value), 0);
                    
                    if (otherTotal > 0) {
                        otherSliders.forEach(s => {
                            const proportion = parseInt(s.value) / otherTotal;
                            const adjustment = Math.round(diff * proportion);
                            const newValue = Math.max(0, parseInt(s.value) + adjustment);
                            s.value = newValue;
                            
                            // Update value display
                            const valueDisplay = s.nextElementSibling;
                            if (valueDisplay) {
                                valueDisplay.textContent = `${newValue}%`;
                            }
                        });
                    }
                }
            });
        });
        
        // Learn more button
        const learnMoreBtn = document.getElementById('regime-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Market Regime Analysis documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-regime-report');
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
            this.options.calendarElementId,
            this.options.transitionElementId,
            this.options.performanceElementId,
            this.options.confidenceElementId
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
        
        // Fetch new data and update charts
        this.fetchData()
            .then(() => {
                this.renderCalendarChart();
                this.renderTransitionMatrix();
                this.renderPerformanceChart();
                this.renderConfidenceChart();
                this.updateRegimeSummary();
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
    
    updateRegimeSummary() {
        // Update the current regime summary
        // In a real implementation, this would use actual data
        
        const regimeBadge = document.querySelector('.regime-badge');
        if (!regimeBadge || !this.calendarData) return;
        
        // Get the most recent regime
        const latestData = this.calendarData[this.calendarData.length - 1];
        const currentRegime = latestData.dominantRegime;
        
        // Update the badge
        regimeBadge.setAttribute('data-regime', currentRegime);
        
        // Update the icon
        const icon = regimeBadge.querySelector('i');
        if (icon) {
            let iconName = 'alert-circle';
            
            switch(currentRegime) {
                case 'bullish':
                    iconName = 'trending-up';
                    break;
                case 'bearish':
                    iconName = 'trending-down';
                    break;
                case 'ranging':
                    iconName = 'minus';
                    break;
                case 'volatile':
                    iconName = 'activity';
                    break;
            }
            
            icon.setAttribute('data-feather', iconName);
            if (typeof feather !== 'undefined') {
                feather.replace();
            }
        }
        
        // Update the text
        const span = regimeBadge.querySelector('span');
        if (span) {
            const confidence = this.confidenceData[this.confidenceData.length - 1].confidence[this.currentRegimeModel];
            const confidenceText = confidence > 0.8 ? 'High Confidence' : 
                                  confidence > 0.6 ? 'Medium Confidence' : 'Low Confidence';
            const regimeText = currentRegime.charAt(0).toUpperCase() + currentRegime.slice(1);
            span.textContent = `${regimeText} (${confidenceText})`;
        }
        
        // Update stats
        // We'll use mock data here but in a real app this would come from actual calculations
        const durationElem = document.querySelector('.stat-item:nth-child(1) .stat-value');
        if (durationElem) {
            // Random duration between 5 and 30 days
            const duration = Math.floor(Math.random() * 25) + 5;
            durationElem.textContent = `${duration} days`;
        }
        
        const avgDurationElem = document.querySelector('.stat-item:nth-child(2) .stat-value');
        if (avgDurationElem) {
            // Set a slightly different average duration based on regime
            let avgDuration;
            switch(currentRegime) {
                case 'bullish': avgDuration = 24; break;
                case 'bearish': avgDuration = 18; break;
                case 'ranging': avgDuration = 32; break;
                case 'volatile': avgDuration = 12; break;
                default: avgDuration = 20;
            }
            avgDurationElem.textContent = `${avgDuration} days`;
        }
        
        const returnElem = document.querySelector('.stat-item:nth-child(3) .stat-value');
        if (returnElem) {
            // Get the return value from performance data
            const returnValue = this.performanceData?.returns[currentRegime] || 0;
            const formattedReturn = (returnValue * 100).toFixed(1);
            returnElem.textContent = returnValue >= 0 ? `+${formattedReturn}%` : `${formattedReturn}%`;
            returnElem.className = 'stat-value ' + (returnValue >= 0 ? 'positive' : 'negative');
        }
        
        const volatilityElem = document.querySelector('.stat-item:nth-child(4) .stat-value');
        if (volatilityElem) {
            // Get the volatility value from performance data
            const volatilityValue = this.performanceData?.volatility[currentRegime] || 0;
            volatilityElem.textContent = `${(volatilityValue * 100).toFixed(1)}%`;
        }
    }
    
    downloadRegimeData() {
        // Create a download of the regime data
        if (!this.calendarData) return;
        
        // Prepare CSV content
        let csv = 'Date,DominantRegime,BullishProbability,BearishProbability,RangingProbability,VolatileProbability\n';
        
        this.calendarData.forEach(item => {
            const date = item.date.toISOString().split('T')[0];
            const regime = item.dominantRegime;
            const bullish = item.probabilities.bullish.toFixed(4);
            const bearish = item.probabilities.bearish.toFixed(4);
            const ranging = item.probabilities.ranging.toFixed(4);
            const volatile = item.probabilities.volatile.toFixed(4);
            
            csv += `${date},${regime},${bullish},${bearish},${ranging},${volatile}\n`;
        });
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `market_regime_data_${this.currentAsset}_${this.currentTimeframe}.csv`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
}

// Initialize the Market Regime Analysis component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of MarketRegimeAnalysis
    const marketRegimeAnalysis = new MarketRegimeAnalysis();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});