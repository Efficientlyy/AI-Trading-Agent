/**
 * Market Breadth Indicators
 * 
 * This module provides functionality for the Market Breadth Indicators component
 * including advance-decline metrics, sector heatmaps, breadth oscillators, and
 * market concentration analysis to show the underlying strength or weakness of the market.
 */

class MarketBreadthIndicators {
    constructor(options = {}) {
        this.options = Object.assign({
            primaryChartElementId: 'primary-breadth-chart',
            sectorHeatmapElementId: 'sector-heatmap',
            breadthOscillatorsElementId: 'breadth-oscillators',
            marketConcentrationElementId: 'market-concentration',
            colorScaleBreadth: [
                [0, 'rgba(239, 68, 68, 0.7)'],     // Bearish (red)
                [0.5, 'rgba(209, 213, 219, 0.7)'],  // Neutral (gray)
                [1, 'rgba(16, 185, 129, 0.7)']      // Bullish (green)
            ],
            colorScaleHeatmap: [
                [0, 'rgba(239, 68, 68, 0.9)'],     // Negative (red)
                [0.5, 'rgba(250, 204, 21, 0.9)'],  // Neutral (yellow)
                [1, 'rgba(16, 185, 129, 0.9)']      // Positive (green)
            ]
        }, options);
        
        this.marketType = document.getElementById('breadth-market')?.value || 'crypto';
        this.timeframe = document.getElementById('breadth-timeframe')?.value || '1d';
        this.period = document.getElementById('breadth-period')?.value || '90';
        this.indicator = document.getElementById('breadth-indicator')?.value || 'advance-decline';
        
        this.maType = document.getElementById('ma-type')?.value || 'sma';
        this.maPeriod = document.getElementById('ma-period')?.value || '20';
        this.heatmapView = document.getElementById('heatmap-view')?.value || 'treemap';
        this.heatmapTime = document.getElementById('heatmap-time')?.value || 'weekly';
        this.concentrationMetric = document.getElementById('concentration-metric')?.value || 'cap-weighted';
        this.activeOscillator = 'mcclellan';
        
        this.advanceDeclineData = [];
        this.sectorData = [];
        this.oscillatorData = {
            mcclellan: [],
            highLow: [],
            percentage: [],
            custom: []
        };
        this.concentrationData = [];
        
        this.primaryChart = null;
        this.sectorHeatmap = null;
        this.oscillatorsChart = null;
        this.concentrationChart = null;
        
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
                console.error('Error initializing Market Breadth Indicators:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchAdvanceDeclineData(),
            this.fetchSectorData(),
            this.fetchOscillatorData(),
            this.fetchConcentrationData()
        ]);
    }
    
    fetchAdvanceDeclineData() {
        // Mock data for advance-decline metrics
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(this.period);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const data = [];
                let advancingTotal = 0;
                let decliningTotal = 0;
                let advDecLine = 0;
                
                // Generate mock data with some trends and patterns
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create some patterns in the data
                    // Basic cycle with some randomness
                    const cycle = Math.sin(i * 2 * Math.PI / (days / 3)) * 20;
                    
                    // Add trend
                    const trend = (i / days) * 50;
                    
                    // Base values for advancing/declining that sum to our universe size
                    let baseAdvancing = 100 + cycle + trend + (Math.random() * 30 - 15);
                    let baseDeclining = 200 - baseAdvancing;
                    
                    // Add some randomness but ensure positive values
                    const advancing = Math.max(10, Math.round(baseAdvancing));
                    const declining = Math.max(10, Math.round(baseDeclining));
                    const unchanged = Math.round(Math.random() * 10);
                    
                    // Calculate cumulative advance-decline line
                    advancingTotal += advancing;
                    decliningTotal += declining;
                    advDecLine = advancingTotal - decliningTotal;
                    
                    // Generate new highs/lows with some correlation to advance/decline values
                    const newHighs = Math.max(0, Math.round((advancing / 200) * 40 + (Math.random() * 15 - 5)));
                    const newLows = Math.max(0, Math.round((declining / 200) * 30 + (Math.random() * 12 - 4)));
                    
                    data.push({
                        date: date,
                        advancing: advancing,
                        declining: declining,
                        unchanged: unchanged,
                        advDecLine: advDecLine,
                        newHighs: newHighs,
                        newLows: newLows,
                        advDecRatio: advancing / (advancing + declining),
                        netAdvance: advancing - declining
                    });
                }
                
                this.advanceDeclineData = data;
                resolve(data);
            }, 300);
        });
    }
    
    fetchSectorData() {
        // Mock data for sector performance heatmap
        return new Promise(resolve => {
            setTimeout(() => {
                let sectors = [];
                
                // Different sectors based on market type
                if (this.marketType === 'crypto') {
                    sectors = [
                        { name: 'DeFi', value: 15.2, change: 3.8, marketCap: 25, volume: 12 },
                        { name: 'Layer-1', value: 8.5, change: 1.2, marketCap: 40, volume: 22 },
                        { name: 'Layer-2', value: 12.3, change: 4.5, marketCap: 15, volume: 9 },
                        { name: 'Exchange', value: -3.2, change: -1.5, marketCap: 30, volume: 18 },
                        { name: 'Gaming', value: -5.8, change: -2.2, marketCap: 8, volume: 4 },
                        { name: 'Privacy', value: 2.5, change: 0.8, marketCap: 5, volume: 3 },
                        { name: 'NFT', value: -8.1, change: -3.5, marketCap: 3, volume: 2 },
                        { name: 'Infrastructure', value: 10.2, change: 2.7, marketCap: 18, volume: 10 },
                        { name: 'Web3', value: 7.8, change: 1.9, marketCap: 12, volume: 7 },
                        { name: 'Stablecoins', value: 0.2, change: 0.1, marketCap: 35, volume: 55 },
                        { name: 'Meme', value: -12.5, change: -5.2, marketCap: 2, volume: 6 },
                        { name: 'AI', value: 18.7, change: 6.4, marketCap: 7, volume: 8 }
                    ];
                } else if (this.marketType === 'US') {
                    sectors = [
                        { name: 'Technology', value: 12.3, change: 2.5, marketCap: 35, volume: 25 },
                        { name: 'Healthcare', value: 8.7, change: 1.8, marketCap: 15, volume: 12 },
                        { name: 'Financials', value: -2.1, change: -0.9, marketCap: 12, volume: 10 },
                        { name: 'Consumer Disc', value: 5.4, change: 1.2, marketCap: 10, volume: 8 },
                        { name: 'Industrials', value: 3.2, change: 0.7, marketCap: 9, volume: 7 },
                        { name: 'Energy', value: -5.8, change: -2.3, marketCap: 5, volume: 6 },
                        { name: 'Materials', value: 1.9, change: 0.4, marketCap: 4, volume: 3 },
                        { name: 'Utilities', value: -1.2, change: -0.3, marketCap: 3, volume: 2 },
                        { name: 'Real Estate', value: -4.5, change: -1.8, marketCap: 3, volume: 2 },
                        { name: 'Telecom', value: 6.8, change: 1.5, marketCap: 4, volume: 3 }
                    ];
                } else if (this.marketType === 'global') {
                    sectors = [
                        { name: 'North America', value: 9.2, change: 2.1, marketCap: 45, volume: 35 },
                        { name: 'Europe', value: 4.5, change: 1.2, marketCap: 20, volume: 15 },
                        { name: 'Asia Pacific', value: -3.8, change: -1.5, marketCap: 25, volume: 20 },
                        { name: 'Japan', value: 6.7, change: 1.8, marketCap: 8, volume: 7 },
                        { name: 'China', value: -7.2, change: -2.5, marketCap: 10, volume: 12 },
                        { name: 'Emerging', value: 2.8, change: 0.9, marketCap: 7, volume: 6 },
                        { name: 'Latin America', value: -1.9, change: -0.7, marketCap: 3, volume: 2 },
                        { name: 'Africa/ME', value: 3.5, change: 1.1, marketCap: 2, volume: 1 }
                    ];
                } else if (this.marketType === 'sectors') {
                    sectors = [
                        { name: 'Banking', value: 7.3, change: 1.8, marketCap: 12, volume: 9 },
                        { name: 'Insurance', value: 3.1, change: 0.6, marketCap: 8, volume: 6 },
                        { name: 'Software', value: 15.7, change: 3.8, marketCap: 18, volume: 14 },
                        { name: 'Hardware', value: 9.2, change: 2.5, marketCap: 14, volume: 10 },
                        { name: 'Biotech', value: 11.8, change: 3.2, marketCap: 7, volume: 5 },
                        { name: 'Pharma', value: 6.3, change: 1.5, marketCap: 10, volume: 7 },
                        { name: 'Retail', value: -2.5, change: -1.1, marketCap: 9, volume: 8 },
                        { name: 'Media', value: 1.8, change: 0.4, marketCap: 6, volume: 5 },
                        { name: 'Telecom', value: -1.2, change: -0.3, marketCap: 5, volume: 4 },
                        { name: 'Auto', value: 8.6, change: 2.1, marketCap: 6, volume: 5 },
                        { name: 'Airlines', value: -7.9, change: -2.8, marketCap: 2, volume: 2 },
                        { name: 'Hotels', value: -4.2, change: -1.7, marketCap: 3, volume: 2 },
                        { name: 'Food', value: 2.1, change: 0.5, marketCap: 4, volume: 3 },
                        { name: 'Construction', value: 4.7, change: 1.2, marketCap: 3, volume: 2 }
                    ];
                }
                
                // Add additional data points for trend analysis
                sectors.forEach(sector => {
                    // Generate a series of historical data points
                    const history = [];
                    for (let i = 0; i < 10; i++) {
                        // Create some continuity with the current value
                        const prevValue = i === 0 ? sector.value : history[i-1].value;
                        const change = (Math.random() * 3 - 1.5) + (sector.value > 0 ? 0.2 : -0.2);
                        history.push({
                            period: i,
                            value: prevValue + change,
                            date: new Date(new Date().setDate(new Date().getDate() - (10-i) * 7))
                        });
                    }
                    
                    // Add breadth metrics
                    sector.advancers = Math.round(50 + (sector.value / 2) + (Math.random() * 10 - 5));
                    sector.decliners = 100 - sector.advancers;
                    sector.history = history;
                    sector.trend = sector.value > 0 ? 
                        (sector.change > 0 ? 'strengthening' : 'weakening') : 
                        (sector.change > sector.value ? 'improving' : 'deteriorating');
                });
                
                this.sectorData = sectors;
                resolve(sectors);
            }, 350);
        });
    }
    
    fetchOscillatorData() {
        // Mock data for breadth oscillators
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(this.period);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const oscillatorData = {
                    mcclellan: [],
                    highLow: [],
                    percentage: [],
                    custom: []
                };
                
                // Generate McClellan Oscillator data
                let ema19 = 0;
                let ema39 = 0;
                
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create some cyclical patterns with noise
                    const cycle = Math.sin(i * 2 * Math.PI / (days / 2)) * 30;
                    const trend = (i < days/2) ? (i / (days/2)) * 20 : ((days - i) / (days/2)) * 20;
                    const noise = Math.random() * 15 - 7.5;
                    
                    // Calculate the net advance-decline for the day
                    const netAdvDec = cycle + trend + noise;
                    
                    // Simple EMA calculation (this is simplified)
                    ema19 = (i === 0) ? netAdvDec : (ema19 * 0.9 + netAdvDec * 0.1);
                    ema39 = (i === 0) ? netAdvDec : (ema39 * 0.95 + netAdvDec * 0.05);
                    
                    // McClellan Oscillator is the difference between the two EMAs
                    const mcClellan = ema19 - ema39;
                    
                    oscillatorData.mcclellan.push({
                        date: date,
                        value: mcClellan,
                        netAdvDec: netAdvDec,
                        ema19: ema19,
                        ema39: ema39
                    });
                }
                
                // Generate High-Low Index data
                let hlSum = 0;
                
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create pattern with more volatile swings
                    const cycle = Math.sin(i * 2 * Math.PI / (days / 4)) * 25;
                    const secondaryCycle = Math.cos(i * 2 * Math.PI / (days / 7)) * 15;
                    const trend = (i / days) * 10;
                    const noise = Math.random() * 10 - 5;
                    
                    // Calculate the number of new highs and new lows
                    const newHighs = Math.max(0, Math.round(25 + cycle + trend + noise));
                    const newLows = Math.max(0, Math.round(15 - cycle + secondaryCycle - trend + noise));
                    
                    // High-Low Index calculation
                    const highLowRatio = newHighs / (newHighs + newLows);
                    
                    // Smooth with a 10-day simple moving average
                    hlSum += highLowRatio;
                    if (i >= 10) {
                        hlSum -= oscillatorData.highLow[i-10].rawRatio;
                    }
                    const hlIndex = (i < 10) ? hlSum / (i+1) * 100 : hlSum / 10 * 100;
                    
                    oscillatorData.highLow.push({
                        date: date,
                        value: hlIndex,
                        newHighs: newHighs,
                        newLows: newLows,
                        rawRatio: highLowRatio
                    });
                }
                
                // Generate Percentage Above Moving Average data
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create pattern that fluctuates between extremes
                    const cycle = Math.sin(i * 2 * Math.PI / (days / 3)) * 25;
                    const trend = (i < days * 0.7) ? (i / (days * 0.7)) * 20 : ((days - i) / (days * 0.3)) * 30;
                    const noise = Math.random() * 8 - 4;
                    
                    // Percentage above 50-day MA
                    const pct50 = Math.min(100, Math.max(0, 50 + cycle + trend/2 + noise));
                    
                    // Percentage above 200-day MA
                    const pct200 = Math.min(100, Math.max(0, 60 + cycle/2 + noise));
                    
                    oscillatorData.percentage.push({
                        date: date,
                        pct50: pct50,
                        pct200: pct200
                    });
                }
                
                // Generate custom oscillator data (for example, Breadth Thrust)
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create pattern with sharp thrusts and slower declines
                    const baseValue = Math.sin(i * 2 * Math.PI / (days / 2)) * 0.3;
                    let thrust = 0.4 + baseValue;
                    
                    // Add occasional sharp thrusts
                    if (i % Math.floor(days/6) === 0) {
                        thrust += 0.2 + Math.random() * 0.2;
                    }
                    
                    // Ensure within bounds
                    thrust = Math.min(1, Math.max(0, thrust));
                    
                    oscillatorData.custom.push({
                        date: date,
                        thrust: thrust
                    });
                }
                
                this.oscillatorData = oscillatorData;
                resolve(oscillatorData);
            }, 400);
        });
    }
    
    fetchConcentrationData() {
        // Mock data for market concentration analysis
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(this.period);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const data = [];
                
                // Generate trending data with occasional divergences
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create patterns
                    const cycle = Math.sin(i * 2 * Math.PI / days) * 0.1;
                    const trend = (i / days) * 0.15; // Increasing concentration over time
                    
                    // Cap weighted vs equal weighted index performance
                    // Higher number means more concentration in top names
                    const capWeighted = 1 + trend + cycle + (Math.random() * 0.04 - 0.02);
                    const equalWeighted = 1 + (trend * 0.7) + cycle + (Math.random() * 0.03 - 0.015);
                    
                    // Calculate concentration ratio (how much cap weighted outperforms equal weighted)
                    // This is a simplified indicator of market concentration
                    const concentrationRatio = capWeighted / equalWeighted;
                    
                    // Top 10 dominance (percentage of market cap)
                    const top10Dominance = 0.45 + trend + (cycle/2) + (Math.random() * 0.02 - 0.01);
                    
                    // Gini coefficient (measure of inequality, higher = more concentrated)
                    const gini = 0.6 + trend + (cycle/3) + (Math.random() * 0.015 - 0.0075);
                    
                    data.push({
                        date: date,
                        capWeighted: capWeighted,
                        equalWeighted: equalWeighted,
                        concentrationRatio: concentrationRatio,
                        top10Dominance: top10Dominance,
                        gini: gini
                    });
                }
                
                this.concentrationData = data;
                resolve(data);
            }, 350);
        });
    }
    
    initializeVisualizations() {
        this.renderPrimaryIndicator();
        this.renderSectorHeatmap();
        this.renderBreadthOscillators();
        this.renderMarketConcentration();
        this.updateBreadthMetrics();
    }
    
    renderPrimaryIndicator() {
        const element = document.getElementById(this.options.primaryChartElementId);
        if (!element || !this.advanceDeclineData || this.advanceDeclineData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        const dates = this.advanceDeclineData.map(d => d.date);
        let yValues = [];
        let maValues = [];
        let title = '';
        
        // Switch data based on selected indicator
        if (this.indicator === 'advance-decline') {
            title = 'Advance-Decline Line';
            yValues = this.advanceDeclineData.map(d => d.advDecLine);
            
            // Calculate moving average
            if (this.maType !== 'none') {
                maValues = this.calculateMovingAverage(
                    yValues, 
                    parseInt(this.maPeriod), 
                    this.maType
                );
            }
        } else if (this.indicator === 'new-highs-lows') {
            title = 'New Highs-Lows Index';
            yValues = this.advanceDeclineData.map(d => d.newHighs - d.newLows);
            
            // Calculate moving average
            if (this.maType !== 'none') {
                maValues = this.calculateMovingAverage(
                    yValues, 
                    parseInt(this.maPeriod), 
                    this.maType
                );
            }
        } else if (this.indicator === 'mcclellan') {
            title = 'McClellan Oscillator';
            yValues = this.oscillatorData.mcclellan.map(d => d.value);
            
            // For oscillators, we don't need to show the MA
            maValues = [];
        } else if (this.indicator === 'concentration') {
            title = 'Market Concentration Ratio';
            yValues = this.concentrationData.map(d => (d.concentrationRatio - 1) * 100); // Convert to percentage
            
            // Calculate moving average
            if (this.maType !== 'none') {
                maValues = this.calculateMovingAverage(
                    yValues, 
                    parseInt(this.maPeriod), 
                    this.maType
                );
            }
        }
        
        // Create the primary indicator trace
        const mainTrace = {
            x: dates,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            name: title,
            line: {
                color: 'rgba(var(--primary-rgb), 0.8)',
                width: 2
            },
            hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}<extra></extra>'
        };
        
        // Create the moving average trace if needed
        const traces = [mainTrace];
        
        if (maValues.length > 0) {
            traces.push({
                x: dates.slice(parseInt(this.maPeriod) - 1),
                y: maValues,
                type: 'scatter',
                mode: 'lines',
                name: `${this.maType.toUpperCase()}-${this.maPeriod}`,
                line: {
                    color: 'rgba(var(--info-rgb), 0.8)',
                    width: 2,
                    dash: 'dot'
                },
                hovertemplate: '%{x|%b %d, %Y}<br>MA: %{y:.2f}<extra></extra>'
            });
        }
        
        // Add zero line for oscillators
        if (this.indicator === 'mcclellan' || this.indicator === 'concentration') {
            traces.push({
                x: [dates[0], dates[dates.length - 1]],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                name: 'Zero Line',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.5)',
                    width: 1,
                    dash: 'dash'
                },
                hoverinfo: 'none',
                showlegend: false
            });
        }
        
        // Add overbought/oversold levels for oscillators
        if (this.indicator === 'mcclellan') {
            traces.push({
                x: [dates[0], dates[dates.length - 1]],
                y: [70, 70],
                type: 'scatter',
                mode: 'lines',
                name: 'Overbought',
                line: {
                    color: 'rgba(var(--success-rgb), 0.4)',
                    width: 1,
                    dash: 'dot'
                },
                hoverinfo: 'none',
                showlegend: false
            });
            
            traces.push({
                x: [dates[0], dates[dates.length - 1]],
                y: [-70, -70],
                type: 'scatter',
                mode: 'lines',
                name: 'Oversold',
                line: {
                    color: 'rgba(var(--danger-rgb), 0.4)',
                    width: 1,
                    dash: 'dot'
                },
                hoverinfo: 'none',
                showlegend: false
            });
        }
        
        // Layout configuration
        const layout = {
            title: title,
            margin: { l: 40, r: 20, t: 40, b: 40 },
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
                title: '',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                yanchor: 'bottom',
                font: {
                    size: 10
                }
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.primaryChartElementId, traces, layout, config);
        this.primaryChart = document.getElementById(this.options.primaryChartElementId);
    }
    
    renderSectorHeatmap() {
        const element = document.getElementById(this.options.sectorHeatmapElementId);
        if (!element || !this.sectorData || this.sectorData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        let traces = [];
        let layout = {};
        
        // Switch between different visualizations based on view type
        if (this.heatmapView === 'treemap') {
            // Create treemap visualization
            traces = [{
                type: 'treemap',
                labels: this.sectorData.map(d => d.name),
                parents: this.sectorData.map(() => ''),
                values: this.sectorData.map(d => d.marketCap),
                marker: {
                    colorscale: this.options.colorScaleHeatmap,
                    colors: this.sectorData.map(d => d.value),
                    line: {
                        width: 1,
                        color: 'var(--border-light)'
                    }
                },
                textinfo: 'label+value+percent',
                hoverinfo: 'label+value+text',
                hovertext: this.sectorData.map(d => `Change: ${d.change.toFixed(2)}%<br>Adv: ${d.advancers} / Dec: ${d.decliners}`),
                textfont: {
                    size: 10,
                    color: 'var(--text-contrast)'
                }
            }];
            
            layout = {
                title: 'Sector Performance Treemap',
                margin: { l: 0, r: 0, t: 40, b: 0 },
                paper_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                }
            };
        } else if (this.heatmapView === 'heatmap') {
            // Create heatmap visualization (simplified for single period)
            traces = [{
                type: 'heatmap',
                z: [this.sectorData.map(d => d.value)],
                x: this.sectorData.map(d => d.name),
                y: [this.heatmapTime],
                colorscale: this.options.colorScaleHeatmap,
                showscale: true,
                hoverinfo: 'x+z',
                hovertemplate: 'Sector: %{x}<br>Performance: %{z:.2f}%<extra></extra>'
            }];
            
            layout = {
                title: 'Sector Performance Heatmap',
                margin: { l: 40, r: 60, t: 40, b: 60 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    tickangle: -45
                }
            };
        } else if (this.heatmapView === 'bubble') {
            // Create bubble chart visualization
            traces = [{
                type: 'scatter',
                mode: 'markers',
                x: this.sectorData.map(d => d.value),
                y: this.sectorData.map(d => d.change),
                text: this.sectorData.map(d => d.name),
                marker: {
                    color: this.sectorData.map(d => {
                        // Color based on value and change (green if both positive, red if both negative, else yellow)
                        if (d.value > 0 && d.change > 0) return 'rgba(var(--success-rgb), 0.7)';
                        if (d.value < 0 && d.change < 0) return 'rgba(var(--danger-rgb), 0.7)';
                        return 'rgba(var(--warning-rgb), 0.7)';
                    }),
                    size: this.sectorData.map(d => d.marketCap / 2 + 10),
                    sizemode: 'area',
                    sizeref: 0.1,
                    line: {
                        color: 'var(--border-light)',
                        width: 1
                    }
                },
                hovertemplate: '<b>%{text}</b><br>Performance: %{x:.2f}%<br>Change: %{y:.2f}%<br>Market Cap: %{marker.size}<extra></extra>'
            }];
            
            layout = {
                title: 'Sector Performance Bubble Chart',
                margin: { l: 40, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Performance (%)',
                    showgrid: true,
                    zeroline: true,
                    zerolinecolor: 'var(--border-light)',
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)'
                },
                yaxis: {
                    title: 'Change (%)',
                    showgrid: true,
                    zeroline: true,
                    zerolinecolor: 'var(--border-light)',
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)'
                }
            };
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.sectorHeatmapElementId, traces, layout, config);
        this.sectorHeatmap = document.getElementById(this.options.sectorHeatmapElementId);
    }
    
    renderBreadthOscillators() {
        const element = document.getElementById(this.options.breadthOscillatorsElementId);
        if (!element || !this.oscillatorData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        let data = [];
        let title = '';
        
        // Choose the data based on the active oscillator tab
        switch (this.activeOscillator) {
            case 'mcclellan':
                data = this.oscillatorData.mcclellan;
                title = 'McClellan Oscillator';
                break;
            case 'high-low':
                data = this.oscillatorData.highLow;
                title = 'High-Low Index';
                break;
            case 'percentage':
                data = this.oscillatorData.percentage;
                title = 'Percentage Above Moving Averages';
                break;
            case 'custom':
                data = this.oscillatorData.custom;
                title = 'Breadth Thrust Indicator';
                break;
            default:
                data = this.oscillatorData.mcclellan;
                title = 'McClellan Oscillator';
        }
        
        let traces = [];
        
        if (this.activeOscillator === 'mcclellan') {
            // McClellan Oscillator chart
            const dates = data.map(d => d.date);
            const values = data.map(d => d.value);
            
            traces = [
                {
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'McClellan Oscillator',
                    line: {
                        color: 'rgba(var(--primary-rgb), 0.8)',
                        width: 2
                    },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(var(--primary-rgb), 0.1)',
                    hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}<extra></extra>'
                },
                {
                    x: [dates[0], dates[dates.length - 1]],
                    y: [0, 0],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Zero Line',
                    line: {
                        color: 'rgba(var(--text-light-rgb), 0.5)',
                        width: 1,
                        dash: 'dash'
                    },
                    hoverinfo: 'none',
                    showlegend: false
                }
            ];
            
        } else if (this.activeOscillator === 'high-low') {
            // High-Low Index chart
            const dates = data.map(d => d.date);
            const values = data.map(d => d.value);
            
            traces = [
                {
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'High-Low Index',
                    line: {
                        color: 'rgba(var(--info-rgb), 0.8)',
                        width: 2
                    },
                    hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}<extra></extra>'
                },
                {
                    x: [dates[0], dates[dates.length - 1]],
                    y: [50, 50],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Neutral Line',
                    line: {
                        color: 'rgba(var(--text-light-rgb), 0.5)',
                        width: 1,
                        dash: 'dash'
                    },
                    hoverinfo: 'none',
                    showlegend: false
                }
            ];
            
        } else if (this.activeOscillator === 'percentage') {
            // Percentage Above Moving Averages chart
            const dates = data.map(d => d.date);
            const pct50Values = data.map(d => d.pct50);
            const pct200Values = data.map(d => d.pct200);
            
            traces = [
                {
                    x: dates,
                    y: pct50Values,
                    type: 'scatter',
                    mode: 'lines',
                    name: '% Above 50-day MA',
                    line: {
                        color: 'rgba(var(--primary-rgb), 0.8)',
                        width: 2
                    },
                    hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}%<extra></extra>'
                },
                {
                    x: dates,
                    y: pct200Values,
                    type: 'scatter',
                    mode: 'lines',
                    name: '% Above 200-day MA',
                    line: {
                        color: 'rgba(var(--secondary-rgb), 0.8)',
                        width: 2
                    },
                    hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}%<extra></extra>'
                },
                {
                    x: [dates[0], dates[dates.length - 1]],
                    y: [50, 50],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Neutral Line',
                    line: {
                        color: 'rgba(var(--text-light-rgb), 0.5)',
                        width: 1,
                        dash: 'dash'
                    },
                    hoverinfo: 'none',
                    showlegend: false
                }
            ];
            
        } else if (this.activeOscillator === 'custom') {
            // Breadth Thrust Indicator chart
            const dates = data.map(d => d.date);
            const values = data.map(d => d.thrust);
            
            traces = [
                {
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Breadth Thrust',
                    line: {
                        color: 'rgba(var(--success-rgb), 0.8)',
                        width: 2
                    },
                    hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}<extra></extra>'
                },
                {
                    x: [dates[0], dates[dates.length - 1]],
                    y: [0.4, 0.4],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Neutral',
                    line: {
                        color: 'rgba(var(--warning-rgb), 0.5)',
                        width: 1,
                        dash: 'dash'
                    },
                    hoverinfo: 'none',
                    showlegend: false
                },
                {
                    x: [dates[0], dates[dates.length - 1]],
                    y: [0.615, 0.615],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Thrust Signal',
                    line: {
                        color: 'rgba(var(--success-rgb), 0.5)',
                        width: 1,
                        dash: 'dash'
                    },
                    hoverinfo: 'none',
                    showlegend: false
                }
            ];
        }
        
        // Layout configuration
        const layout = {
            title: title,
            margin: { l: 40, r: 20, t: 40, b: 40 },
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
                title: '',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                yanchor: 'bottom',
                font: {
                    size: 10
                }
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.breadthOscillatorsElementId, traces, layout, config);
        this.oscillatorsChart = document.getElementById(this.options.breadthOscillatorsElementId);
    }
    
    renderMarketConcentration() {
        const element = document.getElementById(this.options.marketConcentrationElementId);
        if (!element || !this.concentrationData || this.concentrationData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        const dates = this.concentrationData.map(d => d.date);
        let yValues = [];
        let title = '';
        
        // Switch data based on selected concentration metric
        if (this.concentrationMetric === 'cap-weighted') {
            title = 'Cap-Weighted vs Equal-Weighted Ratio';
            yValues = this.concentrationData.map(d => (d.concentrationRatio - 1) * 100); // Convert to percentage diff
        } else if (this.concentrationMetric === 'equal-weighted') {
            title = 'Equal-Weighted Performance';
            // Normalize to percentage change
            const baseValue = this.concentrationData[0].equalWeighted;
            yValues = this.concentrationData.map(d => ((d.equalWeighted / baseValue) - 1) * 100);
        } else if (this.concentrationMetric === 'top-10') {
            title = 'Top 10 Dominance (% of Market Cap)';
            yValues = this.concentrationData.map(d => d.top10Dominance * 100);
        } else if (this.concentrationMetric === 'gini') {
            title = 'Gini Coefficient (Market Concentration)';
            yValues = this.concentrationData.map(d => d.gini);
        }
        
        // Create the primary trace
        const mainTrace = {
            x: dates,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            name: title,
            line: {
                color: 'rgba(var(--info-rgb), 0.8)',
                width: 2
            },
            hovertemplate: '%{x|%b %d, %Y}<br>Value: %{y:.2f}%<extra></extra>'
        };
        
        const traces = [mainTrace];
        
        // Add zero line for ratio charts
        if (this.concentrationMetric === 'cap-weighted') {
            traces.push({
                x: [dates[0], dates[dates.length - 1]],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                name: 'Equal Performance',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.5)',
                    width: 1,
                    dash: 'dash'
                },
                hoverinfo: 'none',
                showlegend: false
            });
        }
        
        // Layout configuration
        const layout = {
            title: title,
            margin: { l: 40, r: 20, t: 40, b: 40 },
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
                title: '',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.marketConcentrationElementId, traces, layout, config);
        this.concentrationChart = document.getElementById(this.options.marketConcentrationElementId);
    }
    
    updateBreadthMetrics() {
        // Update various breadth metrics in the UI
        if (!this.advanceDeclineData || this.advanceDeclineData.length === 0) return;
        
        // Get the most recent data point
        const latestData = this.advanceDeclineData[this.advanceDeclineData.length - 1];
        
        // Update advancing/declining values
        const advancingElement = document.getElementById('advancing-value');
        if (advancingElement) {
            advancingElement.textContent = latestData.advancing;
        }
        
        const decliningElement = document.getElementById('declining-value');
        if (decliningElement) {
            decliningElement.textContent = latestData.declining;
        }
        
        // Update new highs/lows values
        const newHighsElement = document.getElementById('new-highs-value');
        if (newHighsElement) {
            newHighsElement.textContent = latestData.newHighs;
        }
        
        const newLowsElement = document.getElementById('new-lows-value');
        if (newLowsElement) {
            newLowsElement.textContent = latestData.newLows;
        }
        
        // Calculate and update the breadth status and gauge
        this.updateBreadthGauge();
        
        // Update footer metrics
        this.updateFooterMetrics();
    }
    
    updateBreadthGauge() {
        if (!this.advanceDeclineData || this.advanceDeclineData.length === 0) return;
        
        // Calculate overall breadth strength (0-100)
        let breadthStrength = 50; // Neutral by default
        
        // Get the most recent data
        const latestData = this.advanceDeclineData[this.advanceDeclineData.length - 1];
        
        // Factor 1: Advance-Decline Ratio (0-40 points)
        const adRatio = latestData.advancing / (latestData.advancing + latestData.declining);
        let adScore = 0;
        if (adRatio > 0.5) {
            // Advancing > Declining (bullish)
            adScore = 20 + (adRatio - 0.5) * 40;
        } else {
            // Declining > Advancing (bearish)
            adScore = 20 - (0.5 - adRatio) * 40;
        }
        
        // Factor 2: New Highs vs New Lows (0-20 points)
        const hlRatio = latestData.newHighs / (latestData.newHighs + latestData.newLows || 1);
        let hlScore = 0;
        if (hlRatio > 0.5) {
            // More new highs (bullish)
            hlScore = 10 + (hlRatio - 0.5) * 20;
        } else {
            // More new lows (bearish)
            hlScore = 10 - (0.5 - hlRatio) * 20;
        }
        
        // Factor 3: Recent Trend in AD Line (0-30 points)
        const recentPeriod = Math.min(20, this.advanceDeclineData.length);
        const recentAdvDec = this.advanceDeclineData.slice(-recentPeriod);
        const firstAdvDec = recentAdvDec[0].advDecLine;
        const lastAdvDec = recentAdvDec[recentAdvDec.length - 1].advDecLine;
        const advDecTrend = lastAdvDec - firstAdvDec;
        
        let trendScore = 0;
        if (advDecTrend > 0) {
            // Uptrend (bullish)
            trendScore = 15 + Math.min(15, (advDecTrend / 200) * 15);
        } else {
            // Downtrend (bearish)
            trendScore = 15 - Math.min(15, (Math.abs(advDecTrend) / 200) * 15);
        }
        
        // Factor 4: Oscillator value (0-10 points)
        let oscScore = 0;
        if (this.oscillatorData && this.oscillatorData.mcclellan.length > 0) {
            const latestOsc = this.oscillatorData.mcclellan[this.oscillatorData.mcclellan.length - 1];
            if (latestOsc.value > 0) {
                // Positive oscillator (bullish)
                oscScore = 5 + Math.min(5, (latestOsc.value / 100) * 5);
            } else {
                // Negative oscillator (bearish)
                oscScore = 5 - Math.min(5, (Math.abs(latestOsc.value) / 100) * 5);
            }
        }
        
        // Combine all factors
        breadthStrength = Math.round(adScore + hlScore + trendScore + oscScore);
        
        // Ensure it's within 0-100 range
        breadthStrength = Math.max(0, Math.min(100, breadthStrength));
        
        // Update gauge fill
        const gaugeFill = document.getElementById('breadth-gauge-fill');
        if (gaugeFill) {
            gaugeFill.style.width = `${breadthStrength}%`;
        }
        
        // Update gauge marker
        const gaugeMarker = document.getElementById('breadth-gauge-marker');
        if (gaugeMarker) {
            gaugeMarker.style.left = `${breadthStrength}%`;
        }
        
        // Update breadth status text
        const breadthStatus = document.getElementById('breadth-status');
        if (breadthStatus) {
            let statusText = '';
            if (breadthStrength >= 80) {
                statusText = 'Strongly Bullish';
                breadthStatus.className = 'status-value positive';
            } else if (breadthStrength >= 60) {
                statusText = 'Moderately Bullish';
                breadthStatus.className = 'status-value positive';
            } else if (breadthStrength >= 45) {
                statusText = 'Neutral-Bullish';
                breadthStatus.className = 'status-value neutral';
            } else if (breadthStrength >= 35) {
                statusText = 'Neutral';
                breadthStatus.className = 'status-value neutral';
            } else if (breadthStrength >= 20) {
                statusText = 'Neutral-Bearish';
                breadthStatus.className = 'status-value negative';
            } else if (breadthStrength >= 0) {
                statusText = 'Moderately Bearish';
                breadthStatus.className = 'status-value negative';
            } else {
                statusText = 'Strongly Bearish';
                breadthStatus.className = 'status-value negative';
            }
            
            breadthStatus.textContent = statusText;
        }
    }
    
    updateFooterMetrics() {
        // Update footer metrics
        
        // AD Line Trend
        const adLineTrendElement = document.getElementById('adline-trend');
        if (adLineTrendElement && this.advanceDeclineData.length >= 10) {
            const recentData = this.advanceDeclineData.slice(-10);
            const trend = recentData[recentData.length - 1].advDecLine - recentData[0].advDecLine;
            
            if (trend > 50) {
                adLineTrendElement.textContent = 'Strong Uptrend';
                adLineTrendElement.className = 'metric-value positive';
            } else if (trend > 0) {
                adLineTrendElement.textContent = 'Uptrend';
                adLineTrendElement.className = 'metric-value positive';
            } else if (trend > -50) {
                adLineTrendElement.textContent = 'Downtrend';
                adLineTrendElement.className = 'metric-value negative';
            } else {
                adLineTrendElement.textContent = 'Strong Downtrend';
                adLineTrendElement.className = 'metric-value negative';
            }
        }
        
        // Breadth Thrust
        const breadthThrustElement = document.getElementById('breadth-thrust');
        if (breadthThrustElement && this.oscillatorData.custom.length > 0) {
            const latestThrust = this.oscillatorData.custom[this.oscillatorData.custom.length - 1].thrust;
            
            if (latestThrust >= 0.615) {
                breadthThrustElement.textContent = 'Thrust Signal';
                breadthThrustElement.className = 'metric-value positive';
            } else if (latestThrust >= 0.5) {
                breadthThrustElement.textContent = 'Bullish';
                breadthThrustElement.className = 'metric-value positive';
            } else if (latestThrust >= 0.4) {
                breadthThrustElement.textContent = 'Neutral';
                breadthThrustElement.className = 'metric-value neutral';
            } else {
                breadthThrustElement.textContent = 'Weak';
                breadthThrustElement.className = 'metric-value negative';
            }
        }
        
        // Sector Dispersion
        const sectorDispersionElement = document.getElementById('sector-dispersion');
        if (sectorDispersionElement && this.sectorData.length > 0) {
            // Calculate standard deviation of sector performances
            const performances = this.sectorData.map(d => d.value);
            const avg = performances.reduce((a, b) => a + b, 0) / performances.length;
            const variance = performances.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / performances.length;
            const stdDev = Math.sqrt(variance);
            
            if (stdDev > 10) {
                sectorDispersionElement.textContent = 'High';
                sectorDispersionElement.className = 'metric-value negative';
            } else if (stdDev > 5) {
                sectorDispersionElement.textContent = 'Medium';
                sectorDispersionElement.className = 'metric-value neutral';
            } else {
                sectorDispersionElement.textContent = 'Low';
                sectorDispersionElement.className = 'metric-value positive';
            }
        }
        
        // Last Updated
        const lastUpdatedElement = document.getElementById('breadth-last-updated');
        if (lastUpdatedElement) {
            const minutes = Math.floor(Math.random() * 30);
            lastUpdatedElement.textContent = `${minutes} minutes ago`;
        }
    }
    
    setupEventListeners() {
        // Market type selector change
        const marketSelector = document.getElementById('breadth-market');
        if (marketSelector) {
            marketSelector.addEventListener('change', () => {
                this.marketType = marketSelector.value;
                this.refreshData();
            });
        }
        
        // Timeframe selector change
        const timeframeSelector = document.getElementById('breadth-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }
        
        // Period selector change
        const periodSelector = document.getElementById('breadth-period');
        if (periodSelector) {
            periodSelector.addEventListener('change', () => {
                this.period = periodSelector.value;
                this.refreshData();
            });
        }
        
        // Primary indicator selector change
        const indicatorSelector = document.getElementById('breadth-indicator');
        if (indicatorSelector) {
            indicatorSelector.addEventListener('change', () => {
                this.indicator = indicatorSelector.value;
                this.renderPrimaryIndicator();
            });
        }
        
        // Moving average type change
        const maTypeSelector = document.getElementById('ma-type');
        if (maTypeSelector) {
            maTypeSelector.addEventListener('change', () => {
                this.maType = maTypeSelector.value;
                this.renderPrimaryIndicator();
            });
        }
        
        // Moving average period change
        const maPeriodSelector = document.getElementById('ma-period');
        if (maPeriodSelector) {
            maPeriodSelector.addEventListener('change', () => {
                this.maPeriod = maPeriodSelector.value;
                this.renderPrimaryIndicator();
            });
        }
        
        // Heatmap view type change
        const heatmapViewSelector = document.getElementById('heatmap-view');
        if (heatmapViewSelector) {
            heatmapViewSelector.addEventListener('change', () => {
                this.heatmapView = heatmapViewSelector.value;
                this.renderSectorHeatmap();
            });
        }
        
        // Heatmap time period change
        const heatmapTimeSelector = document.getElementById('heatmap-time');
        if (heatmapTimeSelector) {
            heatmapTimeSelector.addEventListener('change', () => {
                this.heatmapTime = heatmapTimeSelector.value;
                this.renderSectorHeatmap();
            });
        }
        
        // Concentration metric change
        const concentrationMetricSelector = document.getElementById('concentration-metric');
        if (concentrationMetricSelector) {
            concentrationMetricSelector.addEventListener('change', () => {
                this.concentrationMetric = concentrationMetricSelector.value;
                this.renderMarketConcentration();
            });
        }
        
        // Oscillator tab change
        const oscillatorTabs = document.querySelectorAll('.oscillator-tab');
        oscillatorTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                oscillatorTabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Update active oscillator
                this.activeOscillator = tab.dataset.oscillator;
                
                // Re-render oscillators chart
                this.renderBreadthOscillators();
            });
        });
        
        // Breadth settings button
        const settingsBtn = document.getElementById('breadth-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('breadth-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-breadth-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadBreadthData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-breadth-panel-btn');
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
        
        // Custom sectors toggle
        const customSectorsToggle = document.getElementById('use-custom-sectors');
        const customSectorsTextarea = document.getElementById('custom-sectors');
        if (customSectorsToggle && customSectorsTextarea) {
            customSectorsToggle.addEventListener('change', () => {
                customSectorsTextarea.disabled = !customSectorsToggle.checked;
            });
        }
        
        // Breadth smoothing slider
        const smoothingSlider = document.getElementById('breadth-smoothing');
        if (smoothingSlider) {
            smoothingSlider.addEventListener('input', function() {
                // Update the value display
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
            });
        }
        
        // Save settings button
        const saveSettingsBtn = document.getElementById('save-breadth-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                
                // Close the modal
                const modal = document.getElementById('breadth-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Learn more button
        const learnMoreBtn = document.getElementById('breadth-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Market Breadth Indicators documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-breadth-report');
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
            this.options.primaryChartElementId,
            this.options.sectorHeatmapElementId,
            this.options.breadthOscillatorsElementId,
            this.options.marketConcentrationElementId
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
                this.renderPrimaryIndicator();
                this.renderSectorHeatmap();
                this.renderBreadthOscillators();
                this.renderMarketConcentration();
                this.updateBreadthMetrics();
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
    
    downloadBreadthData() {
        // Create downloadable CSV file with breadth metrics
        if (!this.advanceDeclineData || this.advanceDeclineData.length === 0) return;
        
        // Prepare CSV content
        let csv = 'Date,Advancing,Declining,Unchanged,ADLine,NewHighs,NewLows,AdvDecRatio,NetAdvance\n';
        
        this.advanceDeclineData.forEach(d => {
            const date = d.date.toISOString().split('T')[0];
            csv += `${date},${d.advancing},${d.declining},${d.unchanged},${d.advDecLine},${d.newHighs},${d.newLows},${d.advDecRatio.toFixed(4)},${d.netAdvance}\n`;
        });
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `market_breadth_${this.marketType}_${this.timeframe}.csv`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    calculateMovingAverage(data, period, type) {
        if (!data || data.length < period) return [];
        
        const result = [];
        
        if (type === 'sma') {
            // Simple Moving Average
            for (let i = period - 1; i < data.length; i++) {
                let sum = 0;
                for (let j = 0; j < period; j++) {
                    sum += data[i - j];
                }
                result.push(sum / period);
            }
        } else if (type === 'ema') {
            // Exponential Moving Average
            const multiplier = 2 / (period + 1);
            let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
            
            result.push(ema);
            
            for (let i = period; i < data.length; i++) {
                ema = (data[i] - ema) * multiplier + ema;
                result.push(ema);
            }
        } else if (type === 'wma') {
            // Weighted Moving Average
            for (let i = period - 1; i < data.length; i++) {
                let sum = 0;
                let weightSum = 0;
                
                for (let j = 0; j < period; j++) {
                    const weight = period - j;
                    sum += data[i - j] * weight;
                    weightSum += weight;
                }
                
                result.push(sum / weightSum);
            }
        }
        
        return result;
    }
}

// Initialize the Market Breadth Indicators component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of MarketBreadthIndicators
    const breadthIndicators = new MarketBreadthIndicators();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});