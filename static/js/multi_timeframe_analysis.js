/**
 * Multi-timeframe Analysis Panel
 * 
 * This script provides the functionality for the multi-timeframe chart panel
 * in the AI Trading Agent dashboard.
 */

class MultiTimeframeAnalysis {
    constructor(options = {}) {
        this.options = Object.assign({
            // Default options
            defaultSymbol: 'BTC-USD',
            defaultTimeframes: ['1h', '4h', '1d'],
            defaultChartType: 'candle',
            apiEndpoint: '/api/market-data',
            chartIds: ['chart-top', 'chart-middle', 'chart-bottom'],
            syncCharts: false,
            indicators: [
                { type: 'sma', period: 20, color: '#4a69bd' },
                { type: 'ema', period: 50, color: '#f39c12' }
            ]
        }, options);
        
        // State management
        this.state = {
            symbol: this.options.defaultSymbol,
            timeframes: [...this.options.defaultTimeframes],
            chartType: this.options.defaultChartType,
            indicators: [...this.options.indicators],
            chartsInitialized: false,
            syncEnabled: this.options.syncCharts
        };
        
        // Chart instances
        this.charts = {
            top: null,
            middle: null,
            bottom: null
        };
        
        // Socket connection for real-time updates
        this.socket = null;
    }
    
    /**
     * Initialize the multi-timeframe panel
     */
    init() {
        console.log('Initializing multi-timeframe analysis panel');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initCharts();
        
        // Set up socket connection for real-time updates
        this.setupSocketConnection();
        
        return this;
    }
    
    /**
     * Set up event listeners for controls
     */
    setupEventListeners() {
        // Symbol selection
        const symbolSelect = document.getElementById('chart-symbol');
        if (symbolSelect) {
            symbolSelect.value = this.state.symbol;
            symbolSelect.addEventListener('change', (e) => {
                this.state.symbol = e.target.value;
                this.updateCharts();
            });
        }
        
        // Timeframe selections
        const timeframeSelects = [
            document.getElementById('timeframe-top'),
            document.getElementById('timeframe-middle'),
            document.getElementById('timeframe-bottom')
        ];
        
        timeframeSelects.forEach((select, index) => {
            if (select) {
                select.value = this.state.timeframes[index];
                select.addEventListener('change', (e) => {
                    this.state.timeframes[index] = e.target.value;
                    this.updateChart(index);
                });
            }
        });
        
        // Chart type selection
        const chartTypeSelect = document.getElementById('chart-type');
        if (chartTypeSelect) {
            chartTypeSelect.value = this.state.chartType;
            chartTypeSelect.addEventListener('change', (e) => {
                this.state.chartType = e.target.value;
                this.updateCharts();
            });
        }
        
        // Sync charts button
        const syncButton = document.getElementById('sync-charts-btn');
        if (syncButton) {
            syncButton.addEventListener('click', () => {
                this.state.syncEnabled = !this.state.syncEnabled;
                this.toggleChartSync();
                syncButton.classList.toggle('active', this.state.syncEnabled);
            });
        }
        
        // Add indicator buttons
        const addIndicatorButtons = [
            document.getElementById('add-indicator-btn'),
            document.getElementById('add-indicator-inline')
        ];
        
        addIndicatorButtons.forEach(button => {
            if (button) {
                button.addEventListener('click', () => {
                    this.showIndicatorModal();
                });
            }
        });
        
        // Add indicator confirmation
        const addIndicatorConfirm = document.getElementById('add-indicator-confirm');
        if (addIndicatorConfirm) {
            addIndicatorConfirm.addEventListener('click', () => {
                this.addIndicator();
            });
        }
        
        // Close modal buttons
        const closeModalButtons = document.querySelectorAll('[data-dismiss="modal"]');
        closeModalButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.hideIndicatorModal();
            });
        });
        
        // Remove indicator buttons
        this.setupRemoveIndicatorListeners();
        
        // Expand panel button
        const expandButton = document.getElementById('expand-panel-btn');
        if (expandButton) {
            expandButton.addEventListener('click', () => {
                this.toggleFullscreen();
            });
        }
    }
    
    /**
     * Initialize charts with default settings
     */
    initCharts() {
        if (this.state.chartsInitialized) {
            return;
        }
        
        this.options.chartIds.forEach((chartId, index) => {
            this.createChart(
                chartId,
                this.state.symbol,
                this.state.timeframes[index],
                this.state.chartType
            );
        });
        
        this.state.chartsInitialized = true;
    }
    
    /**
     * Create a chart with the specified settings
     */
    createChart(chartId, symbol, timeframe, chartType) {
        const chartElement = document.getElementById(chartId);
        if (!chartElement) {
            console.error(`Chart element with ID ${chartId} not found`);
            return;
        }
        
        // Show loading indicator
        this.showChartLoading(chartId);
        
        // Fetch data and create chart
        this.fetchMarketData(symbol, timeframe)
            .then(data => {
                // Hide loading indicator
                this.hideChartLoading(chartId);
                
                // Create chart
                const chartIndex = this.getChartIndex(chartId);
                if (chartIndex !== -1) {
                    const chartKey = ['top', 'middle', 'bottom'][chartIndex];
                    this.charts[chartKey] = this.renderChart(chartId, data, chartType, timeframe);
                }
            })
            .catch(error => {
                console.error(`Error creating chart: ${error}`);
                this.showChartError(chartId, 'Failed to load chart data');
            });
    }
    
    /**
     * Fetch market data for the specified symbol and timeframe
     */
    fetchMarketData(symbol, timeframe) {
        // In a production environment, this would make an API call
        // For now, we'll use mock data
        return new Promise((resolve) => {
            setTimeout(() => {
                const data = this.generateMockData(symbol, timeframe);
                resolve(data);
            }, 500);
        });
    }
    
    /**
     * Generate mock OHLCV data for demonstration
     */
    generateMockData(symbol, timeframe) {
        const now = new Date();
        const data = {
            time: [],
            open: [],
            high: [],
            low: [],
            close: [],
            volume: []
        };
        
        // Set time interval based on timeframe
        let interval;
        let bars = 100;
        
        switch(timeframe) {
            case '1m': interval = 60 * 1000; break;
            case '5m': interval = 5 * 60 * 1000; break;
            case '15m': interval = 15 * 60 * 1000; break;
            case '1h': interval = 60 * 60 * 1000; break;
            case '4h': interval = 4 * 60 * 60 * 1000; break;
            case '1d': interval = 24 * 60 * 60 * 1000; bars = 60; break;
            case '1w': interval = 7 * 24 * 60 * 60 * 1000; bars = 30; break;
            default: interval = 60 * 60 * 1000;
        }
        
        // Generate price based on symbol
        let basePrice;
        let volatility;
        
        switch(symbol) {
            case 'BTC-USD': basePrice = 50000; volatility = 0.02; break;
            case 'ETH-USD': basePrice = 3000; volatility = 0.025; break;
            case 'SOL-USD': basePrice = 100; volatility = 0.03; break;
            case 'ADA-USD': basePrice = 1.2; volatility = 0.035; break;
            case 'DOT-USD': basePrice = 25; volatility = 0.03; break;
            default: basePrice = 1000; volatility = 0.02;
        }
        
        // Generate OHLC data
        let currentPrice = basePrice;
        for (let i = bars; i >= 0; i--) {
            const time = new Date(now.getTime() - i * interval);
            
            // Random price movement
            const change = (Math.random() - 0.5) * 2 * volatility * currentPrice;
            const open = currentPrice;
            currentPrice = open + change;
            
            // Determine high and low
            const rangeFactor = volatility * open * 0.5;
            const high = Math.max(open, currentPrice) + Math.random() * rangeFactor;
            const low = Math.min(open, currentPrice) - Math.random() * rangeFactor;
            
            // Generate volume
            const volume = Math.floor(Math.random() * basePrice * 10) + basePrice;
            
            // Add data point
            data.time.push(time);
            data.open.push(open);
            data.high.push(high);
            data.low.push(low);
            data.close.push(currentPrice);
            data.volume.push(volume);
        }
        
        return data;
    }
    
    /**
     * Render a chart with the specified data
     */
    renderChart(chartId, data, chartType, timeframe) {
        const element = document.getElementById(chartId);
        
        // Prepare traces
        const traces = [];
        
        // Main price chart trace
        if (chartType === 'candle') {
            traces.push({
                type: 'candlestick',
                x: data.time,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                increasing: {line: {color: 'var(--success)'}},
                decreasing: {line: {color: 'var(--danger)'}},
                name: 'Price'
            });
        } else if (chartType === 'ohlc') {
            traces.push({
                type: 'ohlc',
                x: data.time,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                increasing: {line: {color: 'var(--success)'}},
                decreasing: {line: {color: 'var(--danger)'}},
                name: 'Price'
            });
        } else if (chartType === 'line') {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: data.time,
                y: data.close,
                line: {color: 'var(--primary)'},
                name: 'Price'
            });
        } else if (chartType === 'area') {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: data.time,
                y: data.close,
                fill: 'tozeroy',
                fillcolor: 'rgba(var(--primary-rgb), 0.1)',
                line: {color: 'var(--primary)'},
                name: 'Price'
            });
        } else if (chartType === 'bar') {
            traces.push({
                type: 'bar',
                x: data.time,
                y: data.close,
                marker: {
                    color: data.close.map((close, i) => {
                        return i > 0 ? (close >= data.close[i-1] ? 'var(--success)' : 'var(--danger)') : 'var(--primary)';
                    })
                },
                name: 'Price'
            });
        }
        
        // Add volume in a subplot for candlestick/ohlc charts
        if (chartType === 'candle' || chartType === 'ohlc') {
            traces.push({
                type: 'bar',
                x: data.time,
                y: data.volume,
                marker: {
                    color: data.close.map((close, i) => {
                        return i > 0 ? (close >= data.close[i-1] ? 'rgba(var(--success-rgb), 0.3)' : 'rgba(var(--danger-rgb), 0.3)') : 'rgba(var(--primary-rgb), 0.3)';
                    })
                },
                name: 'Volume',
                yaxis: 'y2'
            });
        }
        
        // Add indicators
        this.state.indicators.forEach(indicator => {
            const indicatorTrace = this.createIndicatorTrace(data, indicator);
            if (indicatorTrace) {
                traces.push(indicatorTrace);
            }
        });
        
        // Create layout
        const layout = this.createChartLayout(chartId, data, timeframe, chartType);
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'resetScale2d', 'toggleSpikelines'],
            displaylogo: false,
            scrollZoom: true
        };
        
        // Render the chart with Plotly
        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot(chartId, traces, layout, config);
            
            // Add event listeners for synchronized zooming
            if (this.state.syncEnabled) {
                element.on('plotly_relayout', (eventdata) => {
                    this.synchronizeCharts(chartId, eventdata);
                });
            }
            
            // Return the chartId for reference
            return chartId;
        } else {
            console.error('Plotly is not available');
            return null;
        }
    }
    
    /**
     * Create a layout object for the chart
     */
    createChartLayout(chartId, data, timeframe, chartType) {
        const chartNames = {
            'chart-top': 'Top',
            'chart-middle': 'Middle',
            'chart-bottom': 'Bottom'
        };
        
        const layout = {
            title: {
                text: `${this.state.symbol} - ${timeframe} - ${chartNames[chartId] || ''}`,
                font: {
                    size: 12,
                    color: 'var(--text-light)'
                }
            },
            showlegend: false,
            xaxis: {
                type: 'date',
                rangeslider: {visible: false},
                title: timeframe
            },
            yaxis: {
                title: 'Price',
                side: 'right',
                autorange: true,
                showgrid: true,
                zerolinecolor: 'var(--border-light)',
                tickformat: '.2f',
                hoverformat: '.2f'
            },
            grid: {rows: 1, columns: 1, pattern: 'independent'},
            margin: {l: 40, r: 40, t: 30, b: 20},
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            hovermode: 'x',
            hoverlabel: {
                bgcolor: 'var(--tooltip-bg)',
                bordercolor: 'var(--tooltip-border)',
                font: {
                    color: 'var(--tooltip-text)',
                    size: 11
                }
            }
        };
        
        // Add volume subplot for candlestick/ohlc
        if (chartType === 'candle' || chartType === 'ohlc') {
            layout.grid = {rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom'};
            layout.yaxis.domain = [0.2, 1];
            layout.yaxis2 = {
                title: 'Volume',
                domain: [0, 0.15],
                showgrid: false
            };
        }
        
        return layout;
    }
    
    /**
     * Create a trace for an indicator
     */
    createIndicatorTrace(data, indicator) {
        switch (indicator.type) {
            case 'sma':
                return this.createSMATrace(data, indicator.period, indicator.color);
            case 'ema':
                return this.createEMATrace(data, indicator.period, indicator.color);
            case 'rsi':
                return this.createRSITrace(data, indicator.period, indicator.color);
            case 'bollinger':
                return this.createBollingerTrace(data, indicator.period, indicator.color);
            default:
                console.warn(`Indicator type ${indicator.type} not implemented`);
                return null;
        }
    }
    
    /**
     * Create a Simple Moving Average trace
     */
    createSMATrace(data, period, color) {
        const sma = this.calculateSMA(data.close, period);
        return {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period - 1),
            y: sma,
            line: {
                color: color || 'var(--info)',
                width: 1.5,
                dash: 'solid'
            },
            name: `SMA(${period})`
        };
    }
    
    /**
     * Create an Exponential Moving Average trace
     */
    createEMATrace(data, period, color) {
        const ema = this.calculateEMA(data.close, period);
        return {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period - 1),
            y: ema,
            line: {
                color: color || 'var(--warning)',
                width: 1.5,
                dash: 'dot'
            },
            name: `EMA(${period})`
        };
    }
    
    /**
     * Create a Relative Strength Index trace
     */
    createRSITrace(data, period, color) {
        const rsi = this.calculateRSI(data.close, period);
        return {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period),
            y: rsi,
            line: {
                color: color || 'var(--danger)',
                width: 1.5
            },
            yaxis: 'y3',
            name: `RSI(${period})`
        };
    }
    
    /**
     * Create Bollinger Bands traces
     */
    createBollingerTrace(data, period, color) {
        const { middle, upper, lower } = this.calculateBollingerBands(data.close, period);
        
        // Middle band
        const middleBand = {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period - 1),
            y: middle,
            line: {
                color: color || 'var(--primary)',
                width: 1.5
            },
            name: `BB Middle(${period})`
        };
        
        // Upper band
        const upperBand = {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period - 1),
            y: upper,
            line: {
                color: color || 'var(--success)',
                width: 1,
                dash: 'dash'
            },
            name: `BB Upper(${period})`
        };
        
        // Lower band
        const lowerBand = {
            type: 'scatter',
            mode: 'lines',
            x: data.time.slice(period - 1),
            y: lower,
            line: {
                color: color || 'var(--danger)',
                width: 1,
                dash: 'dash'
            },
            fill: 'tonexty',
            fillcolor: 'rgba(var(--primary-rgb), 0.05)',
            name: `BB Lower(${period})`
        };
        
        return [middleBand, upperBand, lowerBand];
    }
    
    /**
     * Calculate Simple Moving Average
     */
    calculateSMA(data, period) {
        const sma = [];
        
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j];
            }
            sma.push(sum / period);
        }
        
        return sma;
    }
    
    /**
     * Calculate Exponential Moving Average
     */
    calculateEMA(data, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        // First EMA is SMA
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += data[i];
        }
        ema.push(sum / period);
        
        // Calculate EMA
        for (let i = period; i < data.length; i++) {
            ema.push((data[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1]);
        }
        
        return ema;
    }
    
    /**
     * Calculate Relative Strength Index
     */
    calculateRSI(data, period) {
        const rsi = [];
        const gains = [];
        const losses = [];
        
        // Calculate gains and losses
        for (let i = 1; i < data.length; i++) {
            const change = data[i] - data[i - 1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? -change : 0);
        }
        
        // Calculate first average gain and average loss
        let avgGain = 0;
        let avgLoss = 0;
        
        for (let i = 0; i < period; i++) {
            avgGain += gains[i];
            avgLoss += losses[i];
        }
        
        avgGain /= period;
        avgLoss /= period;
        
        // Calculate first RSI
        let rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
        rsi.push(100 - (100 / (1 + rs)));
        
        // Calculate rest of RSI
        for (let i = period; i < data.length - 1; i++) {
            avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
            avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
            
            rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
            rsi.push(100 - (100 / (1 + rs)));
        }
        
        return rsi;
    }
    
    /**
     * Calculate Bollinger Bands
     */
    calculateBollingerBands(data, period, multiplier = 2) {
        const middle = this.calculateSMA(data, period);
        const upper = [];
        const lower = [];
        
        // Calculate standard deviation and bands
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += Math.pow(data[i - j] - middle[i - (period - 1)], 2);
            }
            const stdDev = Math.sqrt(sum / period);
            
            upper.push(middle[i - (period - 1)] + (multiplier * stdDev));
            lower.push(middle[i - (period - 1)] - (multiplier * stdDev));
        }
        
        return { middle, upper, lower };
    }
    
    /**
     * Show loading indicator for a chart
     */
    showChartLoading(chartId) {
        const element = document.getElementById(chartId);
        if (!element) return;
        
        let overlay = element.querySelector('.chart-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'chart-overlay';
            overlay.textContent = 'Loading...';
            element.appendChild(overlay);
        } else {
            overlay.textContent = 'Loading...';
            overlay.style.display = 'flex';
        }
    }
    
    /**
     * Hide loading indicator for a chart
     */
    hideChartLoading(chartId) {
        const element = document.getElementById(chartId);
        if (!element) return;
        
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    /**
     * Show error message for a chart
     */
    showChartError(chartId, message) {
        const element = document.getElementById(chartId);
        if (!element) return;
        
        let overlay = element.querySelector('.chart-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'chart-overlay';
            element.appendChild(overlay);
        }
        
        overlay.textContent = message;
        overlay.style.display = 'flex';
        overlay.classList.add('error');
    }
    
    /**
     * Get chart index from chart ID
     */
    getChartIndex(chartId) {
        return this.options.chartIds.indexOf(chartId);
    }
    
    /**
     * Update all charts
     */
    updateCharts() {
        this.options.chartIds.forEach((chartId, index) => {
            this.updateChart(index);
        });
    }
    
    /**
     * Update a specific chart
     */
    updateChart(index) {
        const chartId = this.options.chartIds[index];
        this.createChart(
            chartId,
            this.state.symbol,
            this.state.timeframes[index],
            this.state.chartType
        );
    }
    
    /**
     * Toggle chart synchronization
     */
    toggleChartSync() {
        if (this.state.syncEnabled) {
            console.log('Enabling chart synchronization');
            // Add sync logic
        } else {
            console.log('Disabling chart synchronization');
            // Remove sync logic
        }
    }
    
    /**
     * Synchronize charts when one is zoomed or panned
     */
    synchronizeCharts(sourceChartId, eventdata) {
        if (!this.state.syncEnabled) return;
        
        const xaxis = eventdata['xaxis.range'] || eventdata['xaxis.range[0]'] ? [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']] : null;
        
        if (xaxis) {
            this.options.chartIds.forEach(chartId => {
                if (chartId !== sourceChartId) {
                    const update = {
                        'xaxis.range': xaxis
                    };
                    
                    Plotly.relayout(chartId, update);
                }
            });
        }
    }
    
    /**
     * Show indicator modal
     */
    showIndicatorModal() {
        const modal = document.getElementById('indicator-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }
    
    /**
     * Hide indicator modal
     */
    hideIndicatorModal() {
        const modal = document.getElementById('indicator-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    /**
     * Add indicator based on modal inputs
     */
    addIndicator() {
        const type = document.getElementById('indicator-type').value;
        const period = parseInt(document.getElementById('indicator-period').value);
        const color = document.getElementById('indicator-color').value;
        
        const applyTop = document.getElementById('apply-top').checked;
        const applyMiddle = document.getElementById('apply-middle').checked;
        const applyBottom = document.getElementById('apply-bottom').checked;
        
        const newIndicator = { type, period, color };
        
        // Add to state
        this.state.indicators.push(newIndicator);
        
        // Add indicator badge to UI
        this.addIndicatorBadge(newIndicator);
        
        // Update charts
        this.updateCharts();
        
        // Hide modal
        this.hideIndicatorModal();
    }
    
    /**
     * Add indicator badge to UI
     */
    addIndicatorBadge(indicator) {
        const badgesContainer = document.querySelector('.indicator-badges');
        if (!badgesContainer) return;
        
        const indicatorName = indicator.type.toUpperCase();
        
        const badge = document.createElement('span');
        badge.className = 'indicator-badge';
        badge.dataset.indicator = indicator.type;
        badge.dataset.period = indicator.period;
        badge.innerHTML = `
            ${indicatorName} <span class="badge">${indicator.period}</span>
            <button class="btn-close" data-tooltip="Remove indicator" data-position="top">×</button>
        `;
        
        // Add before the "Add Indicator" button
        const addButton = document.getElementById('add-indicator-inline');
        if (addButton) {
            badgesContainer.insertBefore(badge, addButton);
        } else {
            badgesContainer.appendChild(badge);
        }
        
        // Add remove event listener
        const removeButton = badge.querySelector('.btn-close');
        if (removeButton) {
            removeButton.addEventListener('click', () => {
                this.removeIndicator(indicator);
                badge.remove();
            });
        }
    }
    
    /**
     * Remove indicator from state and charts
     */
    removeIndicator(indicator) {
        // Remove from state
        const index = this.state.indicators.findIndex(i => 
            i.type === indicator.type && i.period === indicator.period
        );
        
        if (index !== -1) {
            this.state.indicators.splice(index, 1);
            
            // Update charts
            this.updateCharts();
        }
    }
    
    /**
     * Setup event listeners for removing indicators
     */
    setupRemoveIndicatorListeners() {
        const removeButtons = document.querySelectorAll('.indicator-badge .btn-close');
        removeButtons.forEach(button => {
            const badge = button.closest('.indicator-badge');
            if (badge) {
                button.addEventListener('click', () => {
                    const type = badge.dataset.indicator;
                    const period = parseInt(badge.dataset.period);
                    
                    this.removeIndicator({type, period});
                    badge.remove();
                });
            }
        });
    }
    
    /**
     * Toggle fullscreen mode for the panel
     */
    toggleFullscreen() {
        const panel = document.querySelector('.multi-timeframe-panel');
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
    
    /**
     * Set up socket connection for real-time updates
     */
    setupSocketConnection() {
        // In a production environment, this would connect to a WebSocket server
        // For now, we'll simulate real-time updates with setInterval
        setInterval(() => {
            // Only update if we have charts and they're visible
            if (document.visibilityState === 'visible' && this.state.chartsInitialized) {
                this.updateLatestPrices();
            }
        }, 5000);
    }
    
    /**
     * Update latest prices for real-time data
     */
    updateLatestPrices() {
        // In a production environment, this would process WebSocket messages
        // For now, we'll simulate price updates
        this.options.chartIds.forEach((chartId, index) => {
            const chart = document.getElementById(chartId);
            if (!chart || !chart.data || !chart.data[0]) return;
            
            // Generate new price data
            const lastPrice = chart.data[0].close[chart.data[0].close.length - 1];
            const changeFactor = Math.random() * 0.01 - 0.005; // -0.5% to +0.5%
            const newPrice = lastPrice * (1 + changeFactor);
            
            // Update data
            const update = {
                'close[0]': newPrice
            };
            
            // Plotly.restyle(chartId, update, [0]);
        });
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the multi-timeframe analysis panel
    const multiTimeframeAnalysis = new MultiTimeframeAnalysis();
    multiTimeframeAnalysis.init();
    
    // Make it available globally for debugging
    window.multiTimeframeAnalysis = multiTimeframeAnalysis;
});