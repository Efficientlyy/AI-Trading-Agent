/**
 * Portfolio Stress Testing - JS Module
 * 
 * This module handles all the functionality for the Portfolio Stress Testing panel,
 * including scenario simulation, visualization, and reporting of stress test results.
 */

class PortfolioStressTesting {
    constructor(options = {}) {
        this.options = Object.assign({
            containerSelector: '.stress-testing-panel',
            scenarioSelector: '#stress-scenario',
            portfolioSelector: '#stress-portfolio',
            severitySelector: '#stress-severity',
            runTestButton: '#run-stress-test-btn',
            saveScenarioButton: '#save-scenario-btn',
            exportResultsButton: '#export-results-btn',
            expandButton: '#expand-stress-panel-btn',
            customScenarioContainer: '#custom-scenario-builder',
            tabButtons: '.tab-button',
            scenarioDescription: '#scenario-description',
            valueImpactChart: '#value-impact-chart',
            assetImpactChart: '#asset-impact-chart',
            liquidityImpactChart: '#liquidity-impact-chart',
            correlationImpactChart: '#correlation-impact-chart',
            statusBadge: '#stress-test-status',
            timestamp: '#stress-test-timestamp'
        }, options);
        
        this.initialize();
    }
    
    initialize() {
        // Initialize tab navigation
        this.setupTabNavigation();
        
        // Update scenario description
        this.updateScenarioDescription();
        
        // Toggle custom scenario builder visibility
        this.toggleCustomScenarioBuilder();
        
        // Initialize range sliders
        this.initRangeSliders();
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        const self = this;
        
        // Run stress test button
        document.querySelector(this.options.runTestButton)?.addEventListener('click', function() {
            self.runStressTest();
        });
        
        // Scenario selector
        document.querySelector(this.options.scenarioSelector)?.addEventListener('change', function() {
            self.updateScenarioDescription();
            self.toggleCustomScenarioBuilder();
            self.resetCharts();
        });
        
        // Portfolio selector
        document.querySelector(this.options.portfolioSelector)?.addEventListener('change', function() {
            self.resetCharts();
        });
        
        // Severity selector
        document.querySelector(this.options.severitySelector)?.addEventListener('change', function() {
            self.resetCharts();
        });
        
        // Save scenario button
        document.querySelector(this.options.saveScenarioButton)?.addEventListener('click', function() {
            self.saveScenario();
        });
        
        // Export results button
        document.querySelector(this.options.exportResultsButton)?.addEventListener('click', function() {
            self.exportResults();
        });
        
        // Expand panel button
        document.querySelector(this.options.expandButton)?.addEventListener('click', function() {
            self.expandPanel();
        });
    }
    
    setupTabNavigation() {
        const tabButtons = document.querySelectorAll(this.options.tabButtons);
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons and content
                tabButtons.forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab') + '-tab';
                document.getElementById(tabId)?.classList.add('active');
            });
        });
    }
    
    toggleCustomScenarioBuilder() {
        const scenarioSelector = document.querySelector(this.options.scenarioSelector);
        const scenarioBuilder = document.querySelector(this.options.customScenarioContainer);
        
        if (scenarioSelector && scenarioBuilder) {
            const showBuilder = scenarioSelector.value === 'custom-scenario';
            scenarioBuilder.style.display = showBuilder ? 'block' : 'none';
        }
    }
    
    initRangeSliders() {
        // Set up all range sliders with their respective units
        this.initRangeSlider('market-drop', '%');
        this.initRangeSlider('volatility-increase', '%');
        this.initRangeSlider('correlation-shift', '%');
        this.initRangeSlider('liquidity-reduction', '%');
        this.initRangeSlider('spread-widening', '%');
        this.initRangeSlider('recovery-time', ' months');
    }
    
    initRangeSlider(id, unit) {
        const slider = document.getElementById(id);
        const output = document.getElementById(id + '-value');
        
        if (slider && output) {
            // Set initial value
            output.textContent = slider.value + unit;
            
            // Update value on slider change
            slider.addEventListener('input', function() {
                output.textContent = this.value + unit;
            });
        }
    }
    
    updateScenarioDescription() {
        const scenarioSelector = document.querySelector(this.options.scenarioSelector);
        const scenarioDescription = document.querySelector(this.options.scenarioDescription);
        
        if (!scenarioSelector || !scenarioDescription) return;
        
        const scenario = scenarioSelector.value;
        let description = '';
        
        switch (scenario) {
            case 'historical-2008':
                description = '2008 Financial Crisis scenario applies historical market movements from Sep-Dec 2008.';
                break;
            case 'historical-2020':
                description = 'March 2020 COVID Crash scenario applies market movements from Feb-Apr 2020.';
                break;
            case 'historical-2022':
                description = '2022 Crypto Winter scenario applies crypto market downturn conditions from Nov 2021-June 2022.';
                break;
            case 'custom-scenario':
                description = 'Custom scenario using user-defined parameters for stress testing.';
                break;
            case 'extreme-volatility':
                description = 'Extreme Volatility scenario simulates a period of heightened market volatility without directional bias.';
                break;
            case 'liquidity-shock':
                description = 'Liquidity Shock scenario simulates a sudden drop in market liquidity and widening of spreads.';
                break;
            case 'correlation-breakdown':
                description = 'Correlation Breakdown scenario simulates traditional correlations failing during market stress.';
                break;
            default:
                description = 'Stress test scenario applies simulated market conditions to analyze portfolio resilience.';
        }
        
        scenarioDescription.textContent = description;
    }
    
    runStressTest() {
        // Show loading state
        document.querySelectorAll('.chart-overlay').forEach(overlay => {
            overlay.style.display = 'flex';
            overlay.textContent = 'Running stress test simulation...';
        });
        
        // Update status
        const statusBadge = document.querySelector(this.options.statusBadge);
        if (statusBadge) {
            statusBadge.textContent = 'Running';
            statusBadge.style.backgroundColor = 'var(--warning)';
            statusBadge.style.color = 'white';
        }
        
        // Get configuration
        const scenarioElement = document.querySelector(this.options.scenarioSelector);
        const portfolioElement = document.querySelector(this.options.portfolioSelector);
        const severityElement = document.querySelector(this.options.severitySelector);
        
        if (!scenarioElement || !portfolioElement || !severityElement) return;
        
        const scenario = scenarioElement.value;
        const portfolio = portfolioElement.value;
        const severity = severityElement.value;
        
        // Simulate API call delay
        setTimeout(() => {
            // Generate and render results
            this.generateStressTestResults(scenario, portfolio, severity);
            
            // Update timestamp
            const timestamp = document.querySelector(this.options.timestamp);
            if (timestamp) {
                const now = new Date();
                timestamp.textContent = now.toLocaleTimeString() + ', ' + now.toLocaleDateString();
            }
            
            // Update status
            if (statusBadge) {
                statusBadge.textContent = 'Completed';
                statusBadge.style.backgroundColor = 'var(--success)';
                statusBadge.style.color = 'white';
            }
        }, 1500);
    }
    
    generateStressTestResults(scenario, portfolio, severity) {
        // Generate impact values
        this.generateImpactValues(scenario, severity);
        
        // Render value impact chart
        this.renderValueImpactChart(scenario, portfolio, severity);
        
        // Render asset impact chart
        this.renderAssetImpactChart(scenario, portfolio, severity);
        
        // Render liquidity impact chart
        this.renderLiquidityImpactChart(scenario, severity);
        
        // Render correlation impact chart
        this.renderCorrelationImpactChart(scenario, severity);
    }
    
    generateImpactValues(scenario, severity) {
        // Base values for moderate severity
        let drawdown = -32.8;
        let portfolioValue = -24.6;
        let var95 = -18.7;
        let recoveryTime = 4.3;
        
        // Adjust based on severity
        const severityFactors = {
            'mild': 0.6,
            'moderate': 1.0,
            'severe': 1.5,
            'extreme': 2.0
        };
        
        const factor = severityFactors[severity] || 1.0;
        drawdown *= factor;
        portfolioValue *= factor;
        var95 *= factor;
        recoveryTime *= factor;
        
        // Adjust further based on scenario
        if (scenario === 'liquidity-shock') {
            drawdown *= 1.1;
            portfolioValue *= 1.2;
        } else if (scenario === 'extreme-volatility') {
            drawdown *= 1.3;
            var95 *= 1.4;
        } else if (scenario === 'historical-2008') {
            drawdown *= 1.2;
            recoveryTime *= 1.5;
        }
        
        // Update DOM
        document.getElementById('impact-drawdown').textContent = drawdown.toFixed(1) + '%';
        document.getElementById('impact-portfolio-value').textContent = portfolioValue.toFixed(1) + '%';
        
        // Calculate remaining value
        const remainingValue = 100000 * (1 + portfolioValue/100);
        document.getElementById('impact-portfolio-value').nextElementSibling.textContent = '$' + remainingValue.toLocaleString(undefined, {maximumFractionDigits: 0}) + ' remaining';
        
        document.getElementById('impact-var').textContent = var95.toFixed(1) + '%';
        document.getElementById('impact-recovery').textContent = recoveryTime.toFixed(1) + ' months';
    }
    
    renderValueImpactChart(scenario, portfolio, severity) {
        const container = document.querySelector(this.options.valueImpactChart);
        if (!container) return;
        
        // Hide loading overlay
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Generate mock data
        const data = this.generateValueImpactData(scenario, portfolio, severity);
        
        // Define traces
        const traces = [
            {
                x: data.dates,
                y: data.stressedValues,
                type: 'scatter',
                mode: 'lines',
                name: 'Stressed Portfolio',
                line: {
                    color: 'rgba(var(--danger-rgb), 0.8)',
                    width: 2
                }
            },
            {
                x: data.dates,
                y: data.baselineValues,
                type: 'scatter',
                mode: 'lines',
                name: 'Baseline Portfolio',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.5)',
                    width: 1.5,
                    dash: 'dot'
                }
            }
        ];
        
        // Add recovery period if it exists
        if (data.recoveryValues && data.recoveryDates) {
            traces.push({
                x: data.recoveryDates,
                y: data.recoveryValues,
                type: 'scatter',
                mode: 'lines',
                name: 'Recovery Path',
                line: {
                    color: 'rgba(var(--success-rgb), 0.7)',
                    width: 2,
                    dash: 'dot'
                }
            });
        }
        
        // Define layout
        const layout = {
            title: {
                text: 'Portfolio Value Impact',
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
                title: 'Date',
                showgrid: true,
                gridcolor: 'var(--border-light)',
                gridwidth: 1,
                autorange: true
            },
            yaxis: {
                title: 'Portfolio Value ($)',
                showgrid: true,
                gridcolor: 'var(--border-light)',
                gridwidth: 1,
                zerolinecolor: 'var(--border-light)',
                autorange: true
            },
            shapes: [
                {
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: data.stressStart,
                    x1: data.stressEnd,
                    y0: 0,
                    y1: 1,
                    fillcolor: 'rgba(var(--danger-rgb), 0.1)',
                    line: {
                        width: 0
                    },
                    layer: 'below'
                }
            ],
            annotations: [
                {
                    x: data.lowestPoint.date,
                    y: data.lowestPoint.value,
                    xref: 'x',
                    yref: 'y',
                    text: `Max Drawdown: ${data.lowestPoint.drawdown}%`,
                    showarrow: true,
                    arrowhead: 3,
                    arrowsize: 1,
                    arrowwidth: 1,
                    arrowcolor: 'var(--danger)',
                    ax: -30,
                    ay: -30,
                    bgcolor: 'rgba(var(--tooltip-bg-rgb), 0.8)',
                    bordercolor: 'var(--tooltip-border)',
                    borderwidth: 1,
                    borderpad: 4,
                    font: {
                        color: 'var(--tooltip-text)',
                        size: 10
                    }
                }
            ],
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
            Plotly.newPlot(container.id, traces, layout, config);
        }
    }
    
    renderAssetImpactChart(scenario, portfolio, severity) {
        const container = document.querySelector(this.options.assetImpactChart);
        if (!container) return;
        
        // Hide loading overlay
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Generate mock data
        const data = this.generateAssetImpactData(scenario, portfolio, severity);
        
        // Define traces
        const traces = [];
        
        // Create a trace for each asset
        for (const asset of data.assets) {
            traces.push({
                x: data.impacts.map(impact => impact.change),
                y: [asset],
                name: asset,
                orientation: 'h',
                type: 'bar',
                marker: {
                    color: data.impacts.map(impact => {
                        const value = impact.assets[asset];
                        return value < 0 ? 'rgba(var(--danger-rgb), 0.7)' : 'rgba(var(--success-rgb), 0.7)';
                    })
                },
                text: data.impacts.map(impact => `${impact.assets[asset]}%`),
                textposition: 'auto',
                hoverinfo: 'x+text',
                width: 0.7
            });
        }
        
        // Define layout
        const layout = {
            title: {
                text: 'Impact by Asset Class',
                font: {
                    size: 14,
                    color: 'var(--text-light)'
                }
            },
            barmode: 'group',
            showlegend: false,
            xaxis: {
                title: 'Impact (%)',
                showgrid: true,
                gridcolor: 'var(--border-light)',
                gridwidth: 1,
                zeroline: true,
                zerolinecolor: 'var(--text-light)',
                zerolinewidth: 1
            },
            yaxis: {
                title: 'Asset',
                showgrid: false,
                autorange: 'reversed'
            },
            margin: {
                l: 120,
                r: 30,
                t: 40,
                b: 50
            },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 11
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
            Plotly.newPlot(container.id, traces, layout, config);
        }
    }
    
    renderLiquidityImpactChart(scenario, severity) {
        const container = document.querySelector(this.options.liquidityImpactChart);
        if (!container) return;
        
        // Hide loading overlay
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Generate mock data
        const data = this.generateLiquidityImpactData(scenario, severity);
        
        // Define traces for liquidity heatmap
        const assets = data.assets;
        const timepoints = data.timepoints;
        
        // Create z-values for heatmap (liquidity scores)
        const zValues = [];
        for (const asset of assets) {
            const row = [];
            for (const time of timepoints) {
                row.push(data.liquidityScores[asset][time]);
            }
            zValues.push(row);
        }
        
        // Define trace for heatmap
        const trace = {
            type: 'heatmap',
            z: zValues,
            x: timepoints,
            y: assets,
            colorscale: [
                [0, 'rgba(239, 68, 68, 0.7)'],      // Red for low liquidity
                [0.5, 'rgba(245, 158, 11, 0.7)'],   // Amber for medium liquidity
                [1, 'rgba(16, 185, 129, 0.7)']      // Green for high liquidity
            ],
            showscale: true,
            colorbar: {
                title: 'Liquidity',
                titleside: 'right'
            },
            hoverongaps: false
        };
        
        // Define layout
        const layout = {
            title: {
                text: 'Liquidity Heatmap During Stress Period',
                font: {
                    size: 14,
                    color: 'var(--text-light)'
                }
            },
            margin: {
                l: 120,
                r: 70,
                t: 40,
                b: 50
            },
            xaxis: {
                title: 'Time'
            },
            yaxis: {
                title: 'Asset'
            },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 11
            }
        };
        
        // Configuration
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        // Render the chart
        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot(container.id, [trace], layout, config);
        }
    }
    
    renderCorrelationImpactChart(scenario, severity) {
        const container = document.querySelector(this.options.correlationImpactChart);
        if (!container) return;
        
        // Hide loading overlay
        const overlay = container.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Generate mock data
        const data = this.generateCorrelationImpactData(scenario, severity);
        
        // Create traces for normal and stress correlation matrices
        const assets = data.assets;
        
        // Create data for subplots
        const trace1 = {
            type: 'heatmap',
            z: data.normalCorrelation,
            x: assets,
            y: assets,
            colorscale: [
                [0, 'rgba(239, 68, 68, 0.7)'],      // Red for negative correlation
                [0.5, 'rgba(243, 244, 246, 0.7)'],  // Light gray for no correlation
                [1, 'rgba(16, 185, 129, 0.7)']      // Green for positive correlation
            ],
            zmin: -1,
            zmax: 1,
            showscale: false,
            name: 'Normal Conditions',
            xaxis: 'x',
            yaxis: 'y'
        };
        
        const trace2 = {
            type: 'heatmap',
            z: data.stressCorrelation,
            x: assets,
            y: assets,
            colorscale: [
                [0, 'rgba(239, 68, 68, 0.7)'],      // Red for negative correlation
                [0.5, 'rgba(243, 244, 246, 0.7)'],  // Light gray for no correlation
                [1, 'rgba(16, 185, 129, 0.7)']      // Green for positive correlation
            ],
            zmin: -1,
            zmax: 1,
            showscale: true,
            colorbar: {
                title: 'Correlation',
                titleside: 'right',
                x: 1.1,
                y: 0.5
            },
            name: 'Stress Conditions',
            xaxis: 'x2',
            yaxis: 'y2'
        };
        
        // Define layout for side-by-side correlation matrices
        const layout = {
            title: {
                text: 'Correlation Matrix Comparison: Normal vs. Stress',
                font: {
                    size: 14,
                    color: 'var(--text-light)'
                }
            },
            grid: {rows: 1, columns: 2, pattern: 'independent'},
            annotations: [
                {
                    text: 'Normal Conditions',
                    x: 0.2,
                    y: 1.05,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: {
                        size: 12,
                        color: 'var(--text-light)'
                    }
                },
                {
                    text: 'Stress Conditions',
                    x: 0.8,
                    y: 1.05,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: {
                        size: 12,
                        color: 'var(--text-light)'
                    }
                }
            ],
            margin: {
                l: 70,
                r: 100,
                t: 60,
                b: 50
            },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 11
            }
        };
        
        // Configuration
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        // Render the chart
        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot(container.id, [trace1, trace2], layout, config);
        }
    }
    
    generateValueImpactData(scenario, portfolio, severity) {
        // Generate dates for simulation
        const now = new Date();
        const dates = [];
        
        // Create 6 months of baseline data (past)
        const startDate = new Date(now);
        startDate.setMonth(startDate.getMonth() - 6);
        
        // Create date range for baseline (6 months)
        for (let d = new Date(startDate); d <= now; d.setDate(d.getDate() + 5)) {
            dates.push(new Date(d));
        }
        
        // Define stress period (1 month from now)
        const stressStart = new Date(now);
        const stressEnd = new Date(now);
        stressEnd.setMonth(stressEnd.getMonth() + 1);
        
        // Continue date range for stress period
        for (let d = new Date(now); d <= stressEnd; d.setDate(d.getDate() + 5)) {
            if (d > now) { // Don't duplicate the current date
                dates.push(new Date(d));
            }
        }
        
        // Generate baseline values
        const baselineValues = [];
        let value = 100000; // Starting portfolio value
        
        // Baseline growth rate varies by portfolio
        let growthRates = {
            'current': 0.10,
            'optimized': 0.14,
            'balanced': 0.08,
            'aggressive': 0.16,
            'conservative': 0.06
        };
        
        const baseGrowthRate = growthRates[portfolio] || 0.1;
        
        // Generate baseline growth with some volatility
        for (let i = 0; i < dates.length; i++) {
            // Daily growth rate with randomness
            const dailyGrowth = (baseGrowthRate / 365) + ((Math.random() - 0.5) * 0.002);
            
            if (dates[i] <= now) {
                // Baseline period
                value *= (1 + dailyGrowth);
            } else {
                // Flat projection after current date
                value *= (1 + baseGrowthRate / 365);
            }
            
            baselineValues.push(value);
        }
        
        // Generate stressed values
        const stressedValues = [];
        value = baselineValues[dates.findIndex(d => d.getTime() === now.getTime())]; // Current value
        
        // Severity factors
        const severityFactors = {
            'mild': 0.6,
            'moderate': 1.0,
            'severe': 1.5,
            'extreme': 2.0
        };
        
        // Scenario factors
        const scenarioFactors = {
            'historical-2008': 1.2,
            'historical-2020': 1.0,
            'historical-2022': 1.1,
            'custom-scenario': 0.9,
            'extreme-volatility': 1.3,
            'liquidity-shock': 1.1,
            'correlation-breakdown': 1.15
        };
        
        // Calculate total factor
        const factor = (severityFactors[severity] || 1.0) * (scenarioFactors[scenario] || 1.0);
        
        // Maximum drawdown percentage
        const maxDrawdown = -0.25 * factor;
        
        // Copy baseline values until current date
        for (let i = 0; i < dates.length; i++) {
            if (dates[i] <= now) {
                stressedValues.push(baselineValues[i]);
            } else {
                break;
            }
        }
        
        // Start index for stress period
        const stressStartIdx = dates.findIndex(d => d.getTime() === now.getTime());
        
        // Calculate stress period values
        for (let i = stressStartIdx + 1; i < dates.length; i++) {
            // Progress through stress period (0 to 1)
            const progress = (i - stressStartIdx) / (dates.length - stressStartIdx);
            
            // Draw down following roughly a sigmoid curve for more realistic market behavior
            // Concentrates most of the drawdown in the middle of the period
            const sigmoid = 1 / (1 + Math.exp(-12 * (progress - 0.5)));
            const drawdownFactor = maxDrawdown * sigmoid;
            
            // Apply drawdown to original value
            const newValue = stressedValues[stressStartIdx] * (1 + drawdownFactor);
            stressedValues.push(newValue);
        }
        
        // Find lowest point
        let minValue = Number.MAX_VALUE;
        let minDate = null;
        let minIdx = -1;
        
        for (let i = 0; i < stressedValues.length; i++) {
            if (stressedValues[i] < minValue) {
                minValue = stressedValues[i];
                minDate = dates[i];
                minIdx = i;
            }
        }
        
        // Calculate drawdown percentage
        const startValue = stressedValues[stressStartIdx];
        const drawdownPct = ((minValue - startValue) / startValue) * 100;
        
        // Generate recovery path if scenario allows for it
        const recoveryValues = [];
        const recoveryDates = [];
        
        // Only some scenarios have recovery paths
        if (scenario !== 'historical-2022' && scenario !== 'correlation-breakdown') {
            // Recovery period (3 months after stress end)
            const recoveryEnd = new Date(stressEnd);
            recoveryEnd.setMonth(recoveryEnd.getMonth() + 3);
            
            // Add dates for recovery period
            for (let d = new Date(stressEnd); d <= recoveryEnd; d.setDate(d.getDate() + 5)) {
                if (d > stressEnd) { // Don't duplicate the stress end date
                    recoveryDates.push(new Date(d));
                }
            }
            
            // Starting point for recovery is the last stressed value
            let recValue = stressedValues[stressedValues.length - 1];
            
            // Recovery curve shape
            const recoveryCurve = scenario === 'historical-2008' ? 'l-shape' : 
                              (scenario === 'liquidity-shock' ? 'v-shape' : 'u-shape');
            
            // Generate recovery values based on curve shape
            for (let i = 0; i < recoveryDates.length; i++) {
                const progress = i / (recoveryDates.length - 1);
                let recoveryFactor;
                
                switch (recoveryCurve) {
                    case 'v-shape':
                        // Quick recovery
                        recoveryFactor = progress;
                        break;
                    case 'u-shape':
                        // Slow start, faster middle, slow end
                        recoveryFactor = Math.pow(progress, 2) * (3 - 2 * progress);
                        break;
                    case 'l-shape':
                        // Very slow recovery
                        recoveryFactor = Math.pow(progress, 3);
                        break;
                    default:
                        recoveryFactor = progress;
                }
                
                // Calculate recovery amount (recover 80% of the lost value)
                const lostValue = startValue - recValue;
                const recoveryAmount = lostValue * 0.8 * recoveryFactor;
                
                // Apply recovery
                recValue = recValue + recoveryAmount;
                recoveryValues.push(recValue);
            }
        }
        
        return {
            dates,
            baselineValues,
            stressedValues,
            stressStart,
            stressEnd,
            lowestPoint: {
                date: minDate,
                value: minValue,
                drawdown: drawdownPct.toFixed(1)
            },
            recoveryValues,
            recoveryDates
        };
    }
    
    generateAssetImpactData(scenario, portfolio, severity) {
        // Define assets based on portfolio
        let assets;
        
        switch (portfolio) {
            case 'current':
                assets = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'DOT', 'Cash'];
                break;
            case 'optimized':
                assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'AAVE', 'LINK', 'Cash'];
                break;
            case 'balanced':
                assets = ['BTC', 'ETH', 'USDC', 'XRP', 'ADA', 'DOT', 'Cash'];
                break;
            case 'aggressive':
                assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'APE', 'ARB', 'SNX'];
                break;
            case 'conservative':
                assets = ['BTC', 'ETH', 'USDC', 'USDT', 'DAI', 'Cash'];
                break;
            default:
                assets = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'DOT', 'Cash'];
        }
        
        // Define impact categories
        const impacts = [
            {
                change: 'During Crash',
                assets: {}
            },
            {
                change: 'Peak to Trough',
                assets: {}
            },
            {
                change: 'Recovery Phase',
                assets: {}
            }
        ];
        
        // Severity factors
        const severityFactors = {
            'mild': 0.6,
            'moderate': 1.0,
            'severe': 1.5,
            'extreme': 2.0
        };
        
        const factor = severityFactors[severity] || 1.0;
        
        // Generate impact percentages for each asset in each category
        for (const asset of assets) {
            // Base impact values by asset (these would come from historical data or models)
            let duringCrash, peakToTrough, recoveryPhase;
            
            switch (asset) {
                case 'BTC':
                    duringCrash = -18 * factor;
                    peakToTrough = -35 * factor;
                    recoveryPhase = 15 * factor;
                    break;
                case 'ETH':
                    duringCrash = -22 * factor;
                    peakToTrough = -40 * factor;
                    recoveryPhase = 18 * factor;
                    break;
                case 'SOL':
                    duringCrash = -30 * factor;
                    peakToTrough = -55 * factor;
                    recoveryPhase = 25 * factor;
                    break;
                case 'LINK':
                    duringCrash = -25 * factor;
                    peakToTrough = -45 * factor;
                    recoveryPhase = 20 * factor;
                    break;
                case 'MATIC':
                    duringCrash = -28 * factor;
                    peakToTrough = -50 * factor;
                    recoveryPhase = 22 * factor;
                    break;
                case 'DOT':
                    duringCrash = -26 * factor;
                    peakToTrough = -48 * factor;
                    recoveryPhase = 18 * factor;
                    break;
                case 'AVAX':
                    duringCrash = -32 * factor;
                    peakToTrough = -58 * factor;
                    recoveryPhase = 28 * factor;
                    break;
                case 'AAVE':
                    duringCrash = -35 * factor;
                    peakToTrough = -60 * factor;
                    recoveryPhase = 30 * factor;
                    break;
                case 'XRP':
                    duringCrash = -20 * factor;
                    peakToTrough = -38 * factor;
                    recoveryPhase = 15 * factor;
                    break;
                case 'ADA':
                    duringCrash = -24 * factor;
                    peakToTrough = -42 * factor;
                    recoveryPhase = 18 * factor;
                    break;
                case 'APE':
                    duringCrash = -40 * factor;
                    peakToTrough = -65 * factor;
                    recoveryPhase = 35 * factor;
                    break;
                case 'ARB':
                    duringCrash = -38 * factor;
                    peakToTrough = -62 * factor;
                    recoveryPhase = 32 * factor;
                    break;
                case 'SNX':
                    duringCrash = -42 * factor;
                    peakToTrough = -68 * factor;
                    recoveryPhase = 38 * factor;
                    break;
                case 'USDC':
                case 'USDT':
                case 'DAI':
                    // Stablecoins with smaller impacts
                    duringCrash = -2 * factor;
                    peakToTrough = -5 * factor;
                    recoveryPhase = 1 * factor;
                    break;
                case 'Cash':
                    // Cash is stable
                    duringCrash = 0;
                    peakToTrough = 0;
                    recoveryPhase = 0;
                    break;
                default:
                    duringCrash = -25 * factor;
                    peakToTrough = -45 * factor;
                    recoveryPhase = 20 * factor;
            }
            
            // Adjust for scenario
            if (scenario === 'historical-2008') {
                duringCrash *= 1.2;
                peakToTrough *= 1.1;
                recoveryPhase *= 0.8;
            } else if (scenario === 'liquidity-shock') {
                duringCrash *= 1.3;
                peakToTrough *= 1.2;
                recoveryPhase *= 1.1;
            } else if (scenario === 'correlation-breakdown') {
                // Less predictable impacts in a correlation breakdown
                duringCrash *= 0.8 + Math.random() * 0.4;
                peakToTrough *= 0.8 + Math.random() * 0.4;
                recoveryPhase *= 0.8 + Math.random() * 0.4;
            }
            
            // Round to one decimal place
            impacts[0].assets[asset] = Math.round(duringCrash * 10) / 10;
            impacts[1].assets[asset] = Math.round(peakToTrough * 10) / 10;
            impacts[2].assets[asset] = Math.round(recoveryPhase * 10) / 10;
        }
        
        return {
            assets,
            impacts
        };
    }
    
    generateLiquidityImpactData(scenario, severity) {
        // Define assets
        const assets = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'BNB', 'AVAX', 'XRP'];
        
        // Define timepoints
        const timepoints = ['Pre-stress', 'Early stress', 'Mid stress', 'Peak stress', 'Early recovery', 'Full recovery'];
        
        // Severity factors
        const severityFactors = {
            'mild': 0.6,
            'moderate': 1.0,
            'severe': 1.5,
            'extreme': 2.0
        };
        
        const factor = severityFactors[severity] || 1.0;
        
        // Initialize liquidity scores (0-100, higher is better)
        const liquidityScores = {};
        
        // Generate liquidity scores for each asset at each timepoint
        for (const asset of assets) {
            liquidityScores[asset] = {};
            
            // Base liquidity profile by asset
            let baseScore;
            
            switch (asset) {
                case 'BTC':
                case 'ETH':
                    baseScore = 90; // Highest liquidity
                    break;
                case 'BNB':
                case 'XRP':
                    baseScore = 80;
                    break;
                case 'SOL':
                case 'AVAX':
                    baseScore = 70;
                    break;
                case 'LINK':
                case 'MATIC':
                    baseScore = 60;
                    break;
                default:
                    baseScore = 50;
            }
            
            // Generate liquidity evolution during stress
            for (const time of timepoints) {
                let timeMultiplier;
                
                switch (time) {
                    case 'Pre-stress':
                        timeMultiplier = 1.0;
                        break;
                    case 'Early stress':
                        timeMultiplier = 0.8;
                        break;
                    case 'Mid stress':
                        timeMultiplier = 0.6;
                        break;
                    case 'Peak stress':
                        timeMultiplier = 0.4;
                        break;
                    case 'Early recovery':
                        timeMultiplier = 0.7;
                        break;
                    case 'Full recovery':
                        timeMultiplier = 0.9;
                        break;
                    default:
                        timeMultiplier = 1.0;
                }
                
                // Adjust for scenario
                if (scenario === 'liquidity-shock') {
                    timeMultiplier *= 0.7; // Much worse liquidity in liquidity shock
                } else if (scenario === 'extreme-volatility') {
                    timeMultiplier *= 0.85; // Somewhat worse in high volatility
                }
                
                // Apply severity factor
                const adjustedMultiplier = 1 - ((1 - timeMultiplier) * factor);
                
                // Calculate final score with some noise
                const noise = Math.random() * 10 - 5;
                let finalScore = baseScore * adjustedMultiplier + noise;
                
                // Ensure score stays in range 0-100
                finalScore = Math.max(0, Math.min(100, finalScore));
                
                // Assign to data structure
                liquidityScores[asset][time] = Math.round(finalScore);
            }
        }
        
        return {
            assets,
            timepoints,
            liquidityScores
        };
    }
    
    generateCorrelationImpactData(scenario, severity) {
        // Define assets
        const assets = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'BNB', 'AVAX', 'XRP'];
        
        // Initialize correlation matrices
        const normalCorrelation = [];
        const stressCorrelation = [];
        
        // Severity factors
        const severityFactors = {
            'mild': 0.6,
            'moderate': 1.0,
            'severe': 1.5,
            'extreme': 2.0
        };
        
        const factor = severityFactors[severity] || 1.0;
        
        // Generate normal correlation matrix
        for (let i = 0; i < assets.length; i++) {
            const row = [];
            
            for (let j = 0; j < assets.length; j++) {
                if (i === j) {
                    // Diagonal is always 1 (self-correlation)
                    row.push(1);
                } else {
                    // Off-diagonal elements
                    // In crypto, most assets have moderate to high correlation
                    const baseCorr = 0.6 + (Math.random() * 0.3);
                    row.push(Math.round(baseCorr * 100) / 100);
                }
            }
            
            normalCorrelation.push(row);
        }
        
        // Generate stress correlation matrix based on scenario
        for (let i = 0; i < assets.length; i++) {
            const row = [];
            
            for (let j = 0; j < assets.length; j++) {
                if (i === j) {
                    // Diagonal is always 1 (self-correlation)
                    row.push(1);
                } else {
                    // Start with normal correlation
                    let baseCorr = normalCorrelation[i][j];
                    
                    if (scenario === 'correlation-breakdown') {
                        // In correlation breakdown, correlations become less predictable
                        // Some pairs become more correlated, others less
                        if (Math.random() > 0.5) {
                            // Increased correlation (toward 1)
                            baseCorr = baseCorr + (1 - baseCorr) * 0.6 * factor;
                        } else {
                            // Decreased correlation (toward 0 or negative)
                            baseCorr = baseCorr - (baseCorr + 0.5) * 0.6 * factor;
                        }
                    } else {
                        // In most stress scenarios, correlations increase
                        baseCorr = baseCorr + (1 - baseCorr) * 0.4 * factor;
                    }
                    
                    // Add some noise
                    baseCorr += (Math.random() * 0.2 - 0.1);
                    
                    // Ensure correlation stays in range [-1, 1]
                    baseCorr = Math.max(-1, Math.min(1, baseCorr));
                    
                    row.push(Math.round(baseCorr * 100) / 100);
                }
            }
            
            stressCorrelation.push(row);
        }
        
        return {
            assets,
            normalCorrelation,
            stressCorrelation
        };
    }
    
    resetCharts() {
        document.querySelectorAll('.chart-overlay').forEach(overlay => {
            overlay.style.display = 'flex';
            overlay.textContent = 'Select a scenario and run the stress test';
        });
        
        // Reset status badge
        const statusBadge = document.querySelector(this.options.statusBadge);
        if (statusBadge) {
            statusBadge.textContent = 'Ready';
            statusBadge.style.backgroundColor = '';
            statusBadge.style.color = '';
        }
    }
    
    saveScenario() {
        const scenarioSelector = document.querySelector(this.options.scenarioSelector);
        if (!scenarioSelector) return;
        
        const scenario = scenarioSelector.value;
        const name = prompt('Enter a name for this scenario:');
        
        if (name) {
            // In a real implementation, this would save to the backend
            alert(`Scenario "${name}" saved successfully.`);
        }
    }
    
    exportResults() {
        const scenarioSelector = document.querySelector(this.options.scenarioSelector);
        const portfolioSelector = document.querySelector(this.options.portfolioSelector);
        
        if (!scenarioSelector || !portfolioSelector) return;
        
        const scenario = scenarioSelector.value;
        const portfolio = portfolioSelector.value;
        
        // In a real implementation, this would generate and download a report
        alert(`Exporting stress test results for ${portfolio} portfolio under ${scenario} scenario.`);
    }
    
    expandPanel() {
        const panel = document.querySelector(this.options.containerSelector);
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

// Auto-initialize if the document is already loaded
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(() => {
        new PortfolioStressTesting();
    }, 1);
} else {
    document.addEventListener('DOMContentLoaded', () => {
        new PortfolioStressTesting();
    });
}