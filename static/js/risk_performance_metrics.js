/**
 * Risk Performance Metrics
 * 
 * This module provides visualizations of various risk-adjusted performance metrics
 * for portfolio and trading strategy evaluation.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts and metrics
    initRiskMetricsPanel();
    
    // Set up event listeners
    setupMetricsEventListeners();
    
    // Initialize Feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});

function initRiskMetricsPanel() {
    renderRatioEvolutionChart();
    renderRiskRewardChart();
    renderDrawdownAnalysisChart();
    renderOptimizationSurfaceChart();
    updateMetricValues();
}

function setupMetricsEventListeners() {
    // Time range change
    document.getElementById('metrics-time-range').addEventListener('change', function() {
        updateAllCharts();
    });
    
    // Strategy change
    document.getElementById('metrics-strategy').addEventListener('change', function() {
        updateAllCharts();
    });
    
    // Metric type change
    document.getElementById('metric-type').addEventListener('change', function() {
        renderRatioEvolutionChart();
    });
    
    // Refresh button
    document.getElementById('refresh-metrics-btn').addEventListener('click', function() {
        refreshAllMetrics();
    });
    
    // Expand panel button
    document.getElementById('expand-metrics-panel-btn').addEventListener('click', function() {
        expandMetricsPanel();
    });
}

function updateAllCharts() {
    renderRatioEvolutionChart();
    renderRiskRewardChart();
    renderDrawdownAnalysisChart();
    renderOptimizationSurfaceChart();
    updateMetricValues();
}

function refreshAllMetrics() {
    // Add loading overlays
    document.querySelectorAll('.chart-container, .chart-container-sm, .chart-container-md').forEach(container => {
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.textContent = 'Refreshing...';
        }
    });
    
    // Simulate API call delay
    setTimeout(() => {
        updateAllCharts();
    }, 800);
}

function updateMetricValues() {
    // Simulate getting new values
    document.getElementById('sharpe-value').textContent = (1.5 + Math.random() * 1).toFixed(2);
    document.getElementById('sortino-value').textContent = (1.8 + Math.random() * 1).toFixed(2);
    document.getElementById('calmar-value').textContent = (1.2 + Math.random() * 1).toFixed(2);
    document.getElementById('mar-value').textContent = (1.4 + Math.random() * 1).toFixed(2);
}

function renderRatioEvolutionChart() {
    const container = document.getElementById('ratio-evolution-chart');
    const metricType = document.getElementById('metric-type').value;
    const timeRange = document.getElementById('metrics-time-range').value;
    const strategy = document.getElementById('metrics-strategy').value;
    
    // Hide loading overlay
    const overlay = container.querySelector('.chart-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Generate mock data based on selected parameters
    const data = generateRatioEvolutionData(metricType, timeRange, strategy);
    
    // Define chart colors based on metric type
    let colors = {
        sharpe: 'rgb(66, 133, 244)',
        sortino: 'rgb(52, 168, 83)',
        calmar: 'rgb(251, 188, 5)',
        omega: 'rgb(234, 67, 53)',
        mar: 'rgb(171, 71, 188)'
    };
    
    // Create trace
    let traces = [{
        type: 'scatter',
        mode: 'lines',
        name: getRatioDisplayName(metricType),
        x: data.dates,
        y: data.values,
        line: {
            color: colors[metricType] || 'var(--primary)',
            width: 2
        }
    }];
    
    // Add market stress periods as shapes
    let shapes = [];
    if (data.stressPeriods && data.stressPeriods.length > 0) {
        data.stressPeriods.forEach((period, index) => {
            shapes.push({
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: period.start,
                x1: period.end,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(var(--danger-rgb), 0.15)',
                line: {
                    width: 0
                },
                layer: 'below'
            });
        });
    }
    
    // Define average line
    const avgValue = data.values.reduce((sum, val) => sum + val, 0) / data.values.length;
    traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Average',
        x: [data.dates[0], data.dates[data.dates.length-1]],
        y: [avgValue, avgValue],
        line: {
            color: 'rgba(var(--text-light-rgb), 0.5)',
            width: 1,
            dash: 'dash'
        }
    });
    
    // Set up layout
    const layout = {
        title: {
            text: `${getRatioDisplayName(metricType)} Evolution - ${getTimeRangeDisplay(timeRange)}`,
            font: {
                size: 14,
                color: 'var(--text-light)'
            }
        },
        shapes: shapes,
        annotations: data.events.map(event => ({
            x: event.date,
            y: event.value,
            xref: 'x',
            yref: 'y',
            text: event.label,
            showarrow: true,
            arrowhead: 3,
            arrowsize: 1,
            arrowwidth: 1,
            arrowcolor: 'var(--text-light)',
            ax: 0,
            ay: -30,
            bgcolor: 'rgba(var(--tooltip-bg-rgb), 0.8)',
            bordercolor: 'var(--tooltip-border)',
            borderwidth: 1,
            borderpad: 4,
            font: {
                color: 'var(--tooltip-text)',
                size: 10
            }
        })),
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
            title: getRatioDisplayName(metricType),
            showgrid: true,
            gridcolor: 'var(--border-light)',
            gridwidth: 1,
            zerolinecolor: 'var(--border-light)',
            autorange: true
        },
        margin: {
            l: 50,
            r: 30,
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
    
    // Configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d', 'toggleSpikelines'],
        displaylogo: false
    };
    
    // Render the chart
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot('ratio-evolution-chart', traces, layout, config);
    }
}

function renderRiskRewardChart() {
    const container = document.getElementById('risk-reward-chart');
    if (!container) return;
    
    // Hide loading overlay
    const overlay = container.querySelector('.chart-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Generate scatter plot data for different strategies
    const strategies = [
        { name: 'MA Crossover', x: 0.12, y: 0.18, z: 1.5 },
        { name: 'Sentiment-based', x: 0.15, y: 0.22, z: 1.45 },
        { name: 'ML Strategy', x: 0.18, y: 0.28, z: 1.55 },
        { name: 'Combined', x: 0.14, y: 0.24, z: 1.7 },
        { name: 'Benchmark', x: 0.2, y: 0.17, z: 0.85 }
    ];
    
    // Add some variation to each strategy's risk-reward data
    const data = strategies.map(strategy => {
        // Add small random variations to make the chart more realistic
        const riskVariation = (Math.random() * 0.05) - 0.025;
        const returnVariation = (Math.random() * 0.05) - 0.025;
        
        return {
            x: [Math.max(0.05, strategy.x + riskVariation)], // Risk (volatility)
            y: [Math.max(0.05, strategy.y + returnVariation)], // Return
            mode: 'markers',
            type: 'scatter',
            name: strategy.name,
            marker: {
                size: strategy.name === 'Benchmark' ? 10 : 14,
                opacity: strategy.name === 'Benchmark' ? 0.7 : 0.9,
                color: strategy.name === 'Benchmark' ? 'var(--text-light)' : 
                       strategy.name === 'MA Crossover' ? 'var(--primary)' :
                       strategy.name === 'Sentiment-based' ? 'var(--success)' :
                       strategy.name === 'ML Strategy' ? 'var(--info)' : 'var(--purple)',
                line: {
                    width: strategy.name === 'Benchmark' ? 1 : 2,
                    color: 'var(--card-bg)'
                }
            },
            hoverinfo: 'text',
            text: [`${strategy.name}<br>Return: ${(strategy.y * 100).toFixed(1)}%<br>Risk: ${(strategy.x * 100).toFixed(1)}%<br>Sharpe: ${strategy.z.toFixed(2)}`]
        };
    });
    
    // Add the efficient frontier line
    const efficientFrontier = {
        x: [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22],
        y: [0.1, 0.14, 0.17, 0.21, 0.24, 0.26, 0.27, 0.28],
        mode: 'lines',
        type: 'scatter',
        name: 'Efficient Frontier',
        line: {
            color: 'rgba(var(--warning-rgb), 0.7)',
            width: 2,
            dash: 'dash'
        }
    };
    
    data.push(efficientFrontier);
    
    // Layout for risk-reward chart
    const layout = {
        title: {
            text: 'Risk-Return Profile',
            font: {
                size: 12,
                color: 'var(--text-light)'
            }
        },
        showlegend: false,
        xaxis: {
            title: 'Risk (Volatility)',
            tickformat: '.0%',
            showgrid: true,
            gridcolor: 'var(--border-light)',
            zerolinecolor: 'var(--border-light)',
            range: [0.05, 0.25]
        },
        yaxis: {
            title: 'Return',
            tickformat: '.0%',
            showgrid: true,
            gridcolor: 'var(--border-light)',
            zerolinecolor: 'var(--border-light)',
            range: [0.05, 0.3]
        },
        margin: {
            l: 50,
            r: 20,
            t: 30,
            b: 40
        },
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
    
    // Render the chart
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot('risk-reward-chart', data, layout, config);
    }
}

function renderDrawdownAnalysisChart() {
    const container = document.getElementById('drawdown-analysis-chart');
    if (!container) return;
    
    // Hide loading overlay
    const overlay = container.querySelector('.chart-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Generate underwater (drawdown) chart data
    const strategy = document.getElementById('metrics-strategy').value;
    const timeRange = document.getElementById('metrics-time-range').value;
    
    // Generate dates based on time range
    const dates = generateDateRange(timeRange);
    
    // Generate drawdown data - underwater chart
    let drawdowns = [];
    let currentDrawdown = 0;
    
    for (let i = 0; i < dates.length; i++) {
        // Random walk for drawdown, with occasional recovery
        if (Math.random() > 0.7) {
            // Start or deepen drawdown
            currentDrawdown -= Math.random() * 0.02;
        } else if (currentDrawdown < -0.01 && Math.random() > 0.6) {
            // Recover a bit
            currentDrawdown += Math.random() * 0.015;
            // Ensure we don't go positive (drawdowns are negative)
            if (currentDrawdown > 0) currentDrawdown = 0;
        }
        
        // Maximum allowed drawdown for visual clarity
        if (currentDrawdown < -0.25) currentDrawdown = -0.25;
        
        drawdowns.push(currentDrawdown);
    }
    
    // Add data for trace
    const trace = {
        x: dates,
        y: drawdowns,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        name: 'Drawdown',
        line: {
            color: 'rgba(var(--danger-rgb), 0.8)',
            width: 1
        },
        fillcolor: 'rgba(var(--danger-rgb), 0.2)'
    };
    
    // Layout for drawdown chart
    const layout = {
        title: {
            text: 'Drawdown Analysis',
            font: {
                size: 12,
                color: 'var(--text-light)'
            }
        },
        showlegend: false,
        xaxis: {
            showgrid: true,
            gridcolor: 'var(--border-light)',
            zerolinecolor: 'var(--border-light)'
        },
        yaxis: {
            title: 'Drawdown',
            tickformat: '.0%',
            showgrid: true,
            gridcolor: 'var(--border-light)',
            zerolinecolor: 'var(--border-light)',
            range: [-0.25, 0.01]
        },
        margin: {
            l: 50,
            r: 20,
            t: 30,
            b: 40
        },
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
    
    // Render the chart
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot('drawdown-analysis-chart', [trace], layout, config);
    }
}

function renderOptimizationSurfaceChart() {
    const container = document.getElementById('optimization-surface-chart');
    if (!container) return;
    
    // Hide loading overlay
    const overlay = container.querySelector('.chart-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Generate 3D surface data for risk-reward optimization
    const gridSize = 20;
    const x = [], y = [], z = [];
    
    // Generate grid coordinates for risk and return
    for (let i = 0; i < gridSize; i++) {
        const riskValues = [];
        const returnValues = [];
        const sharpeValues = [];
        
        for (let j = 0; j < gridSize; j++) {
            // Risk values from 0.05 to 0.25
            const risk = 0.05 + (0.2 * i / (gridSize - 1));
            riskValues.push(risk);
            
            // Return values from 0.05 to 0.35
            const ret = 0.05 + (0.3 * j / (gridSize - 1));
            returnValues.push(ret);
            
            // Sharpe ratio with a curve that peaks at a certain risk-return combination
            // and falls off as you move away from that peak
            const optimalRisk = 0.14;
            const optimalReturn = 0.25;
            
            // Distance from optimal point in 2D space
            const distance = Math.sqrt(
                Math.pow(risk - optimalRisk, 2) + 
                Math.pow(ret - optimalReturn, 2)
            );
            
            // Sharpe ratio decreases with distance from optimal point
            // but also generally increases as return increases relative to risk
            let sharpe = (ret / risk) * Math.exp(-5 * distance);
            
            // Add some randomness to make it look more natural
            sharpe *= (0.9 + Math.random() * 0.2);
            
            // Scale to reasonable sharpe values
            sharpe *= 2;
            
            sharpeValues.push(sharpe);
        }
        
        x.push(riskValues);
        y.push(returnValues);
        z.push(sharpeValues);
    }
    
    // Create surface plot
    const data = [{
        type: 'surface',
        x: x,
        y: y,
        z: z,
        colorscale: [
            [0, 'rgba(var(--danger-rgb), 0.7)'],
            [0.33, 'rgba(var(--warning-rgb), 0.7)'],
            [0.66, 'rgba(var(--info-rgb), 0.7)'],
            [1, 'rgba(var(--success-rgb), 0.7)']
        ],
        showscale: true,
        colorbar: {
            title: 'Sharpe Ratio',
            titleside: 'right',
            tickfont: {
                color: 'var(--text-light)',
                size: 10
            }
        },
        contours: {
            x: {
                show: true,
                color: 'rgba(var(--border-light-rgb), 0.3)',
                width: 1
            },
            y: {
                show: true,
                color: 'rgba(var(--border-light-rgb), 0.3)',
                width: 1
            },
            z: {
                show: true,
                color: 'rgba(var(--border-light-rgb), 0.3)',
                width: 1
            }
        }
    }];
    
    // Add scatter points for the strategies
    const strategies = [
        { name: 'MA Crossover', risk: 0.12, return: 0.18, sharpe: 1.5 },
        { name: 'Sentiment-based', risk: 0.15, return: 0.22, sharpe: 1.45 },
        { name: 'ML Strategy', risk: 0.18, return: 0.28, sharpe: 1.55 },
        { name: 'Combined', risk: 0.14, return: 0.24, sharpe: 1.7 }
    ];
    
    // Layout for optimization surface chart
    const layout = {
        title: {
            text: 'Risk-Return Optimization Surface',
            font: {
                size: 14,
                color: 'var(--text-light)'
            }
        },
        autosize: true,
        scene: {
            xaxis: {
                title: 'Risk (Volatility)',
                tickformat: '.0%',
                gridcolor: 'var(--border-light)',
                zerolinecolor: 'var(--border-light)',
                showbackground: true,
                backgroundcolor: 'rgba(var(--card-bg-rgb), 0.3)'
            },
            yaxis: {
                title: 'Return',
                tickformat: '.0%',
                gridcolor: 'var(--border-light)',
                zerolinecolor: 'var(--border-light)',
                showbackground: true,
                backgroundcolor: 'rgba(var(--card-bg-rgb), 0.3)'
            },
            zaxis: {
                title: 'Sharpe Ratio',
                gridcolor: 'var(--border-light)',
                zerolinecolor: 'var(--border-light)',
                showbackground: true,
                backgroundcolor: 'rgba(var(--card-bg-rgb), 0.3)'
            },
            camera: {
                eye: {x: 1.5, y: -1.5, z: 0.8},
                center: {x: 0, y: 0, z: 0}
            }
        },
        margin: {
            l: 0,
            r: 0,
            t: 40,
            b: 0
        },
        paper_bgcolor: 'var(--card-bg)',
        plot_bgcolor: 'var(--card-bg)',
        font: {
            color: 'var(--text)',
            size: 11
        }
    };
    
    // Configuration options
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'],
        displaylogo: false
    };
    
    // Render the surface plot
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot('optimization-surface-chart', data, layout, config);
    }
}

function generateRatioEvolutionData(metricType, timeRange, strategy) {
    // Generate dates based on time range
    const dates = generateDateRange(timeRange);
    
    // Generate ratio values based on metric type and strategy
    let values = [];
    let baseValue = 0;
    let volatility = 0;
    
    // Set base value and volatility based on metric type
    switch (metricType) {
        case 'sharpe':
            baseValue = 1.6;
            volatility = 0.2;
            break;
        case 'sortino':
            baseValue = 2.0;
            volatility = 0.25;
            break;
        case 'calmar':
            baseValue = 1.4;
            volatility = 0.3;
            break;
        case 'omega':
            baseValue = 1.8;
            volatility = 0.22;
            break;
        case 'mar':
            baseValue = 1.5;
            volatility = 0.18;
            break;
        default:
            baseValue = 1.5;
            volatility = 0.2;
    }
    
    // Adjust base value based on strategy
    switch (strategy) {
        case 'ma-crossover':
            baseValue *= 1.0;
            break;
        case 'sentiment':
            baseValue *= 0.9;
            break;
        case 'ml-strategy':
            baseValue *= 1.1;
            break;
        case 'combined':
            baseValue *= 1.15;
            break;
        default:
            // No adjustment
    }
    
    // Generate ratio values with a trend and some noise
    let trendDirection = Math.random() > 0.5 ? 1 : -1;
    let trendStrength = Math.random() * 0.0005;
    let currentValue = baseValue;
    
    for (let i = 0; i < dates.length; i++) {
        // Add random noise
        const noise = (Math.random() - 0.5) * volatility;
        
        // Add trend component
        const trend = trendDirection * trendStrength * i;
        
        // Calculate value
        currentValue = baseValue + noise + trend;
        
        // Ensure the value stays positive and reasonable
        currentValue = Math.max(0.5, Math.min(currentValue, baseValue * 2));
        
        values.push(currentValue);
        
        // Occasionally change trend direction
        if (Math.random() > 0.95) {
            trendDirection *= -1;
        }
    }
    
    // Generate market stress periods
    const stressPeriods = [];
    if (dates.length > 30) {
        // Add 1-2 stress periods depending on the length of the time range
        const numStressPeriods = dates.length > 100 ? 2 : 1;
        
        for (let i = 0; i < numStressPeriods; i++) {
            const startIdx = Math.floor(Math.random() * (dates.length * 0.7)) + Math.floor(dates.length * 0.15);
            const endIdx = startIdx + Math.floor(Math.random() * 20) + 10;
            
            if (endIdx < dates.length) {
                stressPeriods.push({
                    start: dates[startIdx],
                    end: dates[endIdx]
                });
            }
        }
    }
    
    // Generate significant events
    const events = [];
    if (dates.length > 30) {
        // Add 1-3 events
        const numEvents = Math.floor(Math.random() * 3) + 1;
        
        for (let i = 0; i < numEvents; i++) {
            const idx = Math.floor(Math.random() * dates.length);
            const eventLabels = [
                'Portfolio Rebalance',
                'Strategy Update',
                'Market Crash',
                'Volatility Spike',
                'Economic Report'
            ];
            
            events.push({
                date: dates[idx],
                value: values[idx],
                label: eventLabels[Math.floor(Math.random() * eventLabels.length)]
            });
        }
    }
    
    return { dates, values, stressPeriods, events };
}

function generateDateRange(timeRange) {
    const dates = [];
    const now = new Date();
    let startDate;
    let interval;
    
    // Set start date and interval based on time range
    switch (timeRange) {
        case '1m':
            startDate = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
            interval = 24 * 60 * 60 * 1000; // 1 day
            break;
        case '3m':
            startDate = new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
            interval = 24 * 60 * 60 * 1000 * 2; // 2 days
            break;
        case '6m':
            startDate = new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());
            interval = 24 * 60 * 60 * 1000 * 3; // 3 days
            break;
        case '1y':
            startDate = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
            interval = 24 * 60 * 60 * 1000 * 7; // 1 week
            break;
        case '2y':
            startDate = new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
            interval = 24 * 60 * 60 * 1000 * 14; // 2 weeks
            break;
        case 'all':
            startDate = new Date(now.getFullYear() - 3, now.getMonth(), now.getDate());
            interval = 24 * 60 * 60 * 1000 * 21; // 3 weeks
            break;
        default:
            startDate = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
            interval = 24 * 60 * 60 * 1000 * 7; // 1 week
    }
    
    // Generate dates
    for (let d = startDate; d <= now; d = new Date(d.getTime() + interval)) {
        dates.push(new Date(d));
    }
    
    // Ensure we include the current date
    if (dates[dates.length - 1].getTime() !== now.getTime()) {
        dates.push(new Date(now));
    }
    
    return dates;
}

function getRatioDisplayName(metricType) {
    const names = {
        'sharpe': 'Sharpe Ratio',
        'sortino': 'Sortino Ratio',
        'calmar': 'Calmar Ratio',
        'omega': 'Omega Ratio',
        'mar': 'MAR Ratio'
    };
    
    return names[metricType] || 'Ratio';
}

function getTimeRangeDisplay(timeRange) {
    const names = {
        '1m': '1 Month',
        '3m': '3 Months',
        '6m': '6 Months',
        '1y': '1 Year',
        '2y': '2 Years',
        'all': 'All Time'
    };
    
    return names[timeRange] || timeRange;
}

function expandMetricsPanel() {
    const panel = document.querySelector('.risk-performance-panel');
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