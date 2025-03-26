/**
 * Advanced Order Flow Visualization
 * 
 * This module provides visualizations for market depth, order book data,
 * trade flow imbalance, and large order detection to help traders understand
 * real-time order flow dynamics.
 */

class AdvancedOrderFlow {
    constructor(options = {}) {
        this.options = Object.assign({
            depthChartElementId: 'market-depth-chart',
            orderBookHeatmapElementId: 'orderbook-heatmap',
            tradeFlowElementId: 'trade-flow-chart',
            largeOrdersElementId: 'large-orders-chart',
            updateInterval: 5000, // milliseconds
            maxHistoryPoints: 50,
            depthLevels: 10,
            colorScaleBid: [
                [0, 'rgba(16, 185, 129, 0.1)'],  // Light green
                [1, 'rgba(16, 185, 129, 0.9)']   // Dark green
            ],
            colorScaleAsk: [
                [0, 'rgba(239, 68, 68, 0.1)'],   // Light red
                [1, 'rgba(239, 68, 68, 0.9)']    // Dark red
            ],
            defaultSymbol: 'BTC-USD'
        }, options);

        // State management
        this.symbol = document.getElementById('order-flow-symbol')?.value || this.options.defaultSymbol;
        this.timeframe = document.getElementById('order-flow-timeframe')?.value || '1m';
        this.depthVisualization = document.getElementById('depth-visualization-type')?.value || 'heatmap';
        this.tradeFlowMetric = document.getElementById('trade-flow-metric')?.value || 'volume';
        this.aggregationLevel = document.getElementById('aggregation-level')?.value || 'default';

        // Data containers
        this.orderBookData = {
            bids: [],
            asks: [],
            timestamp: null
        };
        this.tradeFlowHistory = [];
        this.largeOrdersData = [];
        this.imbalanceHistory = [];

        // Visualization objects
        this.depthChart = null;
        this.orderBookHeatmap = null;
        this.tradeFlowChart = null;
        this.largeOrdersChart = null;

        // Flags
        this.isRealtime = true;
        this.isInitialized = false;
        this.updateTimer = null;

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
                console.error('Error initializing Advanced Order Flow:', error);
                this.showError('Failed to initialize order flow visualizations');
            });
    }

    fetchData() {
        // In a real implementation, this would fetch from an API
        return Promise.all([
            this.fetchOrderBookData(),
            this.fetchTradeFlowData(),
            this.fetchLargeOrdersData()
        ]);
    }

    fetchOrderBookData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock order book data
                const mockOrderBook = this.generateMockOrderBook();
                this.orderBookData = mockOrderBook;
                resolve(mockOrderBook);
            }, 300);
        });
    }

    fetchTradeFlowData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock trade flow data
                const mockTradeFlow = this.generateMockTradeFlow();
                this.tradeFlowHistory = mockTradeFlow;
                
                // Calculate imbalance from trade flow
                this.calculateImbalance();
                
                resolve(mockTradeFlow);
            }, 350);
        });
    }

    fetchLargeOrdersData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock large orders data
                const mockLargeOrders = this.generateMockLargeOrders();
                this.largeOrdersData = mockLargeOrders;
                resolve(mockLargeOrders);
            }, 250);
        });
    }

    initializeCharts() {
        this.renderDepthVisualization();
        this.renderOrderBookHeatmap();
        this.renderTradeFlowChart();
        this.renderLargeOrdersChart();
    }

    setupEventListeners() {
        // Symbol selector change
        const symbolSelector = document.getElementById('order-flow-symbol');
        if (symbolSelector) {
            symbolSelector.addEventListener('change', () => {
                this.symbol = symbolSelector.value;
                this.refreshData();
            });
        }

        // Timeframe selector change
        const timeframeSelector = document.getElementById('order-flow-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }

        // Depth visualization type change
        const depthVisualizationSelector = document.getElementById('depth-visualization-type');
        if (depthVisualizationSelector) {
            depthVisualizationSelector.addEventListener('change', () => {
                this.depthVisualization = depthVisualizationSelector.value;
                this.renderDepthVisualization();
            });
        }

        // Trade flow metric change
        const tradeFlowMetricSelector = document.getElementById('trade-flow-metric');
        if (tradeFlowMetricSelector) {
            tradeFlowMetricSelector.addEventListener('change', () => {
                this.tradeFlowMetric = tradeFlowMetricSelector.value;
                this.renderTradeFlowChart();
            });
        }

        // Aggregation level change
        const aggregationLevelSelector = document.getElementById('aggregation-level');
        if (aggregationLevelSelector) {
            aggregationLevelSelector.addEventListener('change', () => {
                this.aggregationLevel = aggregationLevelSelector.value;
                this.refreshData();
            });
        }

        // Realtime toggle
        const realtimeToggle = document.getElementById('realtime-toggle');
        if (realtimeToggle) {
            realtimeToggle.addEventListener('change', () => {
                this.isRealtime = realtimeToggle.checked;
                if (this.isRealtime) {
                    this.startRealtimeUpdates();
                } else {
                    this.stopRealtimeUpdates();
                }
            });
        }

        // Refresh button
        const refreshButton = document.getElementById('refresh-order-flow');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshData();
            });
        }

        // Export data button
        const exportButton = document.getElementById('export-order-flow-data');
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
            this.updateRealtimeData();
        }, this.options.updateInterval);
    }

    stopRealtimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    updateRealtimeData() {
        // Simulate real-time data updates
        this.updateOrderBookData();
        this.updateTradeFlowData();
        this.updateLargeOrdersData();
        
        // Update visualizations
        this.renderDepthVisualization();
        this.renderOrderBookHeatmap();
        this.renderTradeFlowChart();
        this.renderLargeOrdersChart();
    }

    refreshData() {
        // Show loading state
        this.showLoading();
        
        // Fetch new data
        this.fetchData()
            .then(() => {
                // Update visualizations
                this.renderDepthVisualization();
                this.renderOrderBookHeatmap();
                this.renderTradeFlowChart();
                this.renderLargeOrdersChart();
                
                // Hide loading state
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                this.showError('Failed to refresh order flow data');
            });
    }

    renderDepthVisualization() {
        const element = document.getElementById(this.options.depthChartElementId);
        if (!element || !this.orderBookData) return;

        // Clear loading states
        this.hideLoading(element);

        // Extract data
        const { bids, asks } = this.orderBookData;
        
        // Prepare data based on visualization type
        if (this.depthVisualization === 'heatmap') {
            this.renderDepthHeatmap(element, bids, asks);
        } else {
            this.renderDepthChart(element, bids, asks);
        }
    }

    renderDepthHeatmap(element, bids, asks) {
        // Create price levels for heatmap
        const bidPrices = bids.map(bid => bid.price).sort((a, b) => b - a); // Descending
        const askPrices = asks.map(ask => ask.price).sort((a, b) => a - b); // Ascending
        
        // Calculate midpoint
        const midPrice = (bidPrices[0] + askPrices[0]) / 2;
        
        // Prepare data for heatmap
        const heatmapData = [];
        
        // Add bid levels
        bids.forEach(bid => {
            const normalizedSize = Math.min(1, bid.size / this.getMaxOrderSize(bids));
            const pctFromMid = ((midPrice - bid.price) / midPrice) * 100;
            
            heatmapData.push({
                price: bid.price,
                size: bid.size,
                type: 'bid',
                pctFromMid: pctFromMid,
                normalizedSize: normalizedSize
            });
        });
        
        // Add ask levels
        asks.forEach(ask => {
            const normalizedSize = Math.min(1, ask.size / this.getMaxOrderSize(asks));
            const pctFromMid = ((ask.price - midPrice) / midPrice) * 100;
            
            heatmapData.push({
                price: ask.price,
                size: ask.size,
                type: 'ask',
                pctFromMid: pctFromMid,
                normalizedSize: normalizedSize
            });
        });
        
        // Sort by price
        heatmapData.sort((a, b) => a.price - b.price);
        
        // Create price labels
        const tickValues = heatmapData.map(d => d.price);
        const tickLabels = heatmapData.map(d => d.price.toFixed(2));
        
        // Create heatmap trace
        const trace = {
            y: heatmapData.map(d => d.price),
            x: ['Order Size'],
            z: [heatmapData.map(d => d.size)],
            type: 'heatmap',
            colorscale: heatmapData.map((d, i) => {
                // Determine color based on order type
                if (d.type === 'bid') {
                    return [i / heatmapData.length, this.getColorFromScale(d.normalizedSize, this.options.colorScaleBid)];
                } else {
                    return [i / heatmapData.length, this.getColorFromScale(d.normalizedSize, this.options.colorScaleAsk)];
                }
            }),
            showscale: false,
            hoverongaps: false,
            hovertemplate: 'Price: %{y}<br>Size: %{z}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Market Depth Heatmap',
            margin: { l: 60, r: 20, t: 30, b: 30 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            yaxis: {
                title: 'Price',
                tickvals: tickValues,
                ticktext: tickLabels,
                tickmode: 'array',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            xaxis: {
                showticklabels: false,
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            shapes: [
                // Midpoint line
                {
                    type: 'line',
                    x0: -0.5,
                    x1: 1.5,
                    y0: midPrice,
                    y1: midPrice,
                    line: {
                        color: 'rgba(var(--warning-rgb), 0.7)',
                        width: 2,
                        dash: 'dash'
                    }
                }
            ],
            annotations: [
                // Midpoint label
                {
                    x: 1,
                    y: midPrice,
                    xref: 'x',
                    yref: 'y',
                    text: 'Mid: ' + midPrice.toFixed(2),
                    showarrow: false,
                    font: {
                        color: 'var(--warning)',
                        size: 10
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
        Plotly.newPlot(this.options.depthChartElementId, [trace], layout, config);
        this.depthChart = document.getElementById(this.options.depthChartElementId);
    }

    renderDepthChart(element, bids, asks) {
        // Prepare cumulative depth data
        const bidsCumulative = this.calculateCumulativeDepth(bids, 'bid');
        const asksCumulative = this.calculateCumulativeDepth(asks, 'ask');
        
        // Calculate midpoint price
        const midPrice = (bids[0]?.price || 0 + asks[0]?.price || 0) / 2;
        
        // Bid trace
        const bidTrace = {
            x: bidsCumulative.map(d => d.price),
            y: bidsCumulative.map(d => d.cumulativeSize),
            type: 'scatter',
            mode: 'lines',
            name: 'Bids',
            fill: 'tozeroy',
            line: {
                color: 'rgba(16, 185, 129, 1)',
                width: 2
            },
            fillcolor: 'rgba(16, 185, 129, 0.2)',
            hovertemplate: 'Price: %{x}<br>Cumulative Size: %{y}<extra></extra>'
        };
        
        // Ask trace
        const askTrace = {
            x: asksCumulative.map(d => d.price),
            y: asksCumulative.map(d => d.cumulativeSize),
            type: 'scatter',
            mode: 'lines',
            name: 'Asks',
            fill: 'tozeroy',
            line: {
                color: 'rgba(239, 68, 68, 1)',
                width: 2
            },
            fillcolor: 'rgba(239, 68, 68, 0.2)',
            hovertemplate: 'Price: %{x}<br>Cumulative Size: %{y}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Market Depth Chart',
            margin: { l: 60, r: 20, t: 30, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Price',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Cumulative Size',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            legend: {
                orientation: 'h',
                xanchor: 'center',
                x: 0.5,
                y: -0.15
            },
            shapes: [
                // Midpoint line
                {
                    type: 'line',
                    x0: midPrice,
                    x1: midPrice,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {
                        color: 'rgba(var(--warning-rgb), 0.7)',
                        width: 2,
                        dash: 'dash'
                    }
                }
            ],
            annotations: [
                // Midpoint label
                {
                    x: midPrice,
                    y: 1,
                    xref: 'x',
                    yref: 'paper',
                    text: 'Mid: ' + midPrice.toFixed(2),
                    showarrow: false,
                    yshift: 10,
                    font: {
                        color: 'var(--warning)',
                        size: 10
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
        Plotly.newPlot(this.options.depthChartElementId, [bidTrace, askTrace], layout, config);
        this.depthChart = document.getElementById(this.options.depthChartElementId);
    }

    renderOrderBookHeatmap() {
        const element = document.getElementById(this.options.orderBookHeatmapElementId);
        if (!element || !this.orderBookData) return;

        // Clear loading states
        this.hideLoading(element);

        // Extract data
        const { bids, asks } = this.orderBookData;
        
        // Create orderbook heatmap data
        const bidPrices = bids.slice(0, this.options.depthLevels).map(b => b.price.toFixed(2)).reverse();
        const askPrices = asks.slice(0, this.options.depthLevels).map(a => a.price.toFixed(2));
        
        // Combine all prices for y-axis
        const allPrices = [...bidPrices, ...askPrices];
        
        // Create size values with proper normalization
        const bidSizes = bids.slice(0, this.options.depthLevels).map(b => b.size).reverse();
        const askSizes = asks.slice(0, this.options.depthLevels).map(a => a.size);
        
        const maxBidSize = Math.max(...bidSizes);
        const maxAskSize = Math.max(...askSizes);
        const maxSize = Math.max(maxBidSize, maxAskSize);
        
        // Normalize sizes
        const normalizedBidSizes = bidSizes.map(size => size / maxSize);
        const normalizedAskSizes = askSizes.map(size => size / maxSize);
        
        // Create color values
        const bidColors = normalizedBidSizes.map(size => this.getColorFromScale(size, this.options.colorScaleBid));
        const askColors = normalizedAskSizes.map(size => this.getColorFromScale(size, this.options.colorScaleAsk));
        
        // Create trace for bids
        const bidTrace = {
            x: Array(bidPrices.length).fill('Bid'),
            y: bidPrices,
            z: bidSizes,
            type: 'heatmap',
            colorscale: bidColors.map((color, i) => [i / bidColors.length, color]),
            showscale: false,
            hovertemplate: 'Price: %{y}<br>Size: %{z}<extra></extra>'
        };
        
        // Create trace for asks
        const askTrace = {
            x: Array(askPrices.length).fill('Ask'),
            y: askPrices,
            z: askSizes,
            type: 'heatmap',
            colorscale: askColors.map((color, i) => [i / askColors.length, color]),
            showscale: false,
            hovertemplate: 'Price: %{y}<br>Size: %{z}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Order Book Heatmap',
            margin: { l: 60, r: 20, t: 30, b: 30 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            yaxis: {
                title: 'Price',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false,
                tickangle: 0
            },
            xaxis: {
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false,
                tickangle: 0
            },
            annotations: [
                // Spread annotation
                {
                    x: 0.5,
                    y: 0.5,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Spread: ' + (asks[0]?.price - bids[0]?.price).toFixed(2),
                    showarrow: false,
                    font: {
                        color: 'var(--text-light)',
                        size: 10
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
        Plotly.newPlot(this.options.orderBookHeatmapElementId, [bidTrace, askTrace], layout, config);
        this.orderBookHeatmap = document.getElementById(this.options.orderBookHeatmapElementId);
    }

    renderTradeFlowChart() {
        const element = document.getElementById(this.options.tradeFlowElementId);
        if (!element || !this.tradeFlowHistory) return;

        // Clear loading states
        this.hideLoading(element);

        // Prepare data based on selected metric
        let xValues, yValues, barColors, title;
        
        if (this.tradeFlowMetric === 'volume') {
            // Buy/sell volume over time
            xValues = this.tradeFlowHistory.map(d => d.timestamp);
            const buyVolumeValues = this.tradeFlowHistory.map(d => d.buyVolume);
            const sellVolumeValues = this.tradeFlowHistory.map(d => d.sellVolume);
            
            // Create buy volume trace
            const buyTrace = {
                x: xValues,
                y: buyVolumeValues,
                type: 'bar',
                name: 'Buy Volume',
                marker: {
                    color: 'rgba(16, 185, 129, 0.7)'
                },
                hovertemplate: 'Time: %{x}<br>Buy Volume: %{y}<extra></extra>'
            };
            
            // Create sell volume trace
            const sellTrace = {
                x: xValues,
                y: sellVolumeValues.map(v => -v), // Negate for display
                type: 'bar',
                name: 'Sell Volume',
                marker: {
                    color: 'rgba(239, 68, 68, 0.7)'
                },
                hovertemplate: 'Time: %{x}<br>Sell Volume: %{y}<extra></extra>'
            };
            
            // Layout configuration
            const layout = {
                title: 'Buy/Sell Volume Flow',
                barmode: 'relative',
                margin: { l: 60, r: 20, t: 30, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Time',
                    showgrid: false,
                    linecolor: 'var(--border-light)',
                    type: 'date',
                    tickformat: '%H:%M:%S',
                    zeroline: false
                },
                yaxis: {
                    title: 'Volume',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)',
                    zeroline: false
                },
                legend: {
                    orientation: 'h',
                    xanchor: 'center',
                    x: 0.5,
                    y: 1.15
                }
            };
            
            // Configuration options
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            // Render with Plotly
            Plotly.newPlot(this.options.tradeFlowElementId, [buyTrace, sellTrace], layout, config);
            this.tradeFlowChart = document.getElementById(this.options.tradeFlowElementId);
            
        } else if (this.tradeFlowMetric === 'imbalance') {
            // Trade flow imbalance over time
            xValues = this.imbalanceHistory.map(d => d.timestamp);
            yValues = this.imbalanceHistory.map(d => d.imbalance * 100); // Convert to percentage
            
            // Determine colors based on imbalance value
            barColors = yValues.map(value => {
                if (value > 0) return 'rgba(16, 185, 129, 0.7)'; // Green for buy imbalance
                return 'rgba(239, 68, 68, 0.7)'; // Red for sell imbalance
            });
            
            // Create bar trace
            const trace = {
                x: xValues,
                y: yValues,
                type: 'bar',
                marker: {
                    color: barColors
                },
                hovertemplate: 'Time: %{x}<br>Imbalance: %{y:.1f}%<extra></extra>'
            };
            
            // Layout configuration
            const layout = {
                title: 'Trade Flow Imbalance',
                margin: { l: 60, r: 20, t: 30, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Time',
                    showgrid: false,
                    linecolor: 'var(--border-light)',
                    type: 'date',
                    tickformat: '%H:%M:%S',
                    zeroline: false
                },
                yaxis: {
                    title: 'Imbalance (%)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)',
                    zeroline: true,
                    zerolinecolor: 'var(--border-light)',
                    zerolinewidth: 1,
                    ticksuffix: '%'
                },
                annotations: [
                    // Average imbalance annotation
                    {
                        x: 1,
                        y: 1,
                        xref: 'paper',
                        yref: 'paper',
                        text: 'Avg. Imbalance: ' + (this.calculateAverageImbalance() * 100).toFixed(1) + '%',
                        showarrow: false,
                        xshift: -5,
                        yshift: -5,
                        xanchor: 'right',
                        yanchor: 'top',
                        font: {
                            color: 'var(--text-light)',
                            size: 10
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
            Plotly.newPlot(this.options.tradeFlowElementId, [trace], layout, config);
            this.tradeFlowChart = document.getElementById(this.options.tradeFlowElementId);
            
        } else if (this.tradeFlowMetric === 'trades') {
            // Trade count over time
            xValues = this.tradeFlowHistory.map(d => d.timestamp);
            const buyTradeValues = this.tradeFlowHistory.map(d => d.buyTrades);
            const sellTradeValues = this.tradeFlowHistory.map(d => d.sellTrades);
            
            // Create buy trades trace
            const buyTrace = {
                x: xValues,
                y: buyTradeValues,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Buy Trades',
                line: {
                    color: 'rgba(16, 185, 129, 1)',
                    width: 2
                },
                marker: {
                    color: 'rgba(16, 185, 129, 1)',
                    size: 6
                },
                hovertemplate: 'Time: %{x}<br>Buy Trades: %{y}<extra></extra>'
            };
            
            // Create sell trades trace
            const sellTrace = {
                x: xValues,
                y: sellTradeValues,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sell Trades',
                line: {
                    color: 'rgba(239, 68, 68, 1)',
                    width: 2
                },
                marker: {
                    color: 'rgba(239, 68, 68, 1)',
                    size: 6
                },
                hovertemplate: 'Time: %{x}<br>Sell Trades: %{y}<extra></extra>'
            };
            
            // Layout configuration
            const layout = {
                title: 'Trade Count Flow',
                margin: { l: 60, r: 20, t: 30, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Time',
                    showgrid: false,
                    linecolor: 'var(--border-light)',
                    type: 'date',
                    tickformat: '%H:%M:%S',
                    zeroline: false
                },
                yaxis: {
                    title: 'Number of Trades',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)',
                    zeroline: false
                },
                legend: {
                    orientation: 'h',
                    xanchor: 'center',
                    x: 0.5,
                    y: 1.15
                }
            };
            
            // Configuration options
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            // Render with Plotly
            Plotly.newPlot(this.options.tradeFlowElementId, [buyTrace, sellTrace], layout, config);
            this.tradeFlowChart = document.getElementById(this.options.tradeFlowElementId);
        }
    }

    renderLargeOrdersChart() {
        const element = document.getElementById(this.options.largeOrdersElementId);
        if (!element || !this.largeOrdersData) return;

        // Clear loading states
        this.hideLoading(element);

        // Extract large order data
        const xValues = this.largeOrdersData.map(d => d.timestamp);
        const yValues = this.largeOrdersData.map(d => d.price);
        const sizes = this.largeOrdersData.map(d => d.size);
        const types = this.largeOrdersData.map(d => d.type);
        
        // Calculate market price from order book
        const midPrice = (this.orderBookData.bids[0]?.price + this.orderBookData.asks[0]?.price) / 2;
        
        // Create colors and symbols based on order type
        const colors = types.map(type => {
            if (type === 'bid') return 'rgba(16, 185, 129, 0.7)';
            return 'rgba(239, 68, 68, 0.7)';
        });
        
        const symbols = types.map(type => {
            if (type === 'bid') return 'circle';
            return 'square';
        });
        
        // Normalize sizes for bubble plot
        const normalizedSizes = sizes.map(size => {
            // Adjust the scale for better visualization
            return Math.max(10, Math.min(40, size / 20));
        });
        
        // Create bubble trace
        const trace = {
            x: xValues,
            y: yValues,
            mode: 'markers',
            marker: {
                size: normalizedSizes,
                color: colors,
                symbol: symbols,
                line: {
                    color: 'rgba(var(--border-rgb), 0.5)',
                    width: 1
                }
            },
            text: this.largeOrdersData.map(d => 
                `Type: ${d.type === 'bid' ? 'Buy' : 'Sell'}<br>` +
                `Price: ${d.price.toFixed(2)}<br>` +
                `Size: ${d.size.toFixed(2)}<br>` +
                `Time: ${new Date(d.timestamp).toLocaleTimeString()}`
            ),
            hoverinfo: 'text'
        };
        
        // Layout configuration
        const layout = {
            title: 'Large Order Detection',
            margin: { l: 60, r: 20, t: 30, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                showgrid: false,
                linecolor: 'var(--border-light)',
                type: 'date',
                tickformat: '%H:%M:%S',
                zeroline: false
            },
            yaxis: {
                title: 'Price',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            shapes: [
                // Market price line
                {
                    type: 'line',
                    x0: xValues[0],
                    x1: xValues[xValues.length - 1],
                    y0: midPrice,
                    y1: midPrice,
                    line: {
                        color: 'rgba(var(--warning-rgb), 0.7)',
                        width: 1,
                        dash: 'dash'
                    }
                }
            ],
            annotations: [
                // Legend for buy orders
                {
                    x: 0.02,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Buy Orders',
                    showarrow: false,
                    font: {
                        color: 'rgba(16, 185, 129, 1)',
                        size: 10
                    },
                    bgcolor: 'rgba(var(--card-bg-rgb), 0.7)',
                    borderpad: 2
                },
                // Legend for sell orders
                {
                    x: 0.02,
                    y: 0.93,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Sell Orders',
                    showarrow: false,
                    font: {
                        color: 'rgba(239, 68, 68, 1)',
                        size: 10
                    },
                    bgcolor: 'rgba(var(--card-bg-rgb), 0.7)',
                    borderpad: 2
                },
                // Market price label
                {
                    x: 0.98,
                    y: midPrice,
                    xref: 'paper',
                    yref: 'y',
                    text: 'Market: ' + midPrice.toFixed(2),
                    showarrow: false,
                    font: {
                        color: 'var(--warning)',
                        size: 10
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
        Plotly.newPlot(this.options.largeOrdersElementId, [trace], layout, config);
        this.largeOrdersChart = document.getElementById(this.options.largeOrdersElementId);
    }

    // Data simulation and update functions
    generateMockOrderBook() {
        // Generate a realistic order book with reasonable price levels
        const basePrice = this.getBasePrice();
        const bids = [];
        const asks = [];
        
        // Generate bid levels (below market price)
        for (let i = 0; i < 20; i++) {
            // Prices decrease as we go down the book
            const priceOffset = (Math.random() * 0.01 + 0.01) * i;
            const price = basePrice * (1 - priceOffset);
            
            // Sizes tend to increase as price decreases
            const baseSize = 0.5 + Math.random() * 2;
            const sizeMultiplier = 1 + i * 0.2;
            const size = baseSize * sizeMultiplier * (1 + Math.random() * 0.5);
            
            bids.push({
                price: price,
                size: size
            });
        }
        
        // Sort bids by price (highest first)
        bids.sort((a, b) => b.price - a.price);
        
        // Generate ask levels (above market price)
        for (let i = 0; i < 20; i++) {
            // Prices increase as we go up the book
            const priceOffset = (Math.random() * 0.01 + 0.01) * i;
            const price = basePrice * (1 + priceOffset);
            
            // Sizes tend to increase as price increases
            const baseSize = 0.5 + Math.random() * 2;
            const sizeMultiplier = 1 + i * 0.2;
            const size = baseSize * sizeMultiplier * (1 + Math.random() * 0.5);
            
            asks.push({
                price: price,
                size: size
            });
        }
        
        // Sort asks by price (lowest first)
        asks.sort((a, b) => a.price - b.price);
        
        return {
            bids: bids,
            asks: asks,
            timestamp: new Date()
        };
    }

    generateMockTradeFlow() {
        // Generate the last N points of trade flow history
        const points = this.options.maxHistoryPoints;
        const history = [];
        
        // Generate initial point
        const now = new Date();
        let buyVolume = 50 + Math.random() * 50;
        let sellVolume = 40 + Math.random() * 60;
        
        // Add historical points
        for (let i = points - 1; i >= 0; i--) {
            const timestamp = new Date(now.getTime() - (i * 60000)); // One minute intervals
            
            // Add some randomness but maintain a trend
            buyVolume = Math.max(10, buyVolume + (Math.random() * 20 - 10));
            sellVolume = Math.max(10, sellVolume + (Math.random() * 20 - 10));
            
            // Add seasonality
            const seasonalFactor = Math.sin(i / 5) * 10;
            buyVolume += seasonalFactor;
            sellVolume -= seasonalFactor;
            
            // Add trade counts
            const buyTrades = Math.floor(buyVolume / 5);
            const sellTrades = Math.floor(sellVolume / 5);
            
            history.push({
                timestamp: timestamp,
                buyVolume: buyVolume,
                sellVolume: sellVolume,
                buyTrades: buyTrades,
                sellTrades: sellTrades
            });
        }
        
        return history;
    }

    generateMockLargeOrders() {
        // Generate large orders that appeared recently
        const orders = [];
        const now = new Date();
        const basePrice = this.getBasePrice();
        
        // How many large orders to generate
        const orderCount = 8 + Math.floor(Math.random() * 6);
        
        for (let i = 0; i < orderCount; i++) {
            // Random timestamp within the last hour
            const minutesAgo = Math.random() * 60;
            const timestamp = new Date(now.getTime() - minutesAgo * 60000);
            
            // Order type (bid or ask)
            const type = Math.random() > 0.5 ? 'bid' : 'ask';
            
            // Price (near basePrice)
            const priceOffset = (Math.random() * 0.02) * (type === 'bid' ? -1 : 1);
            const price = basePrice * (1 + priceOffset);
            
            // Size (larger than regular orders)
            const size = 10 + Math.random() * 40;
            
            // Add to orders
            orders.push({
                timestamp: timestamp,
                type: type,
                price: price,
                size: size
            });
        }
        
        // Sort by timestamp (oldest first)
        orders.sort((a, b) => a.timestamp - b.timestamp);
        
        return orders;
    }

    updateOrderBookData() {
        // Simulate changes to the order book
        const { bids, asks } = this.orderBookData;
        
        // Adjust existing bids and asks
        this.orderBookData.bids = bids.map(bid => {
            // Small random price and size adjustments
            const priceChange = bid.price * (Math.random() * 0.002 - 0.001);
            const sizeChange = bid.size * (Math.random() * 0.1 - 0.05);
            
            return {
                price: bid.price + priceChange,
                size: Math.max(0.1, bid.size + sizeChange)
            };
        });
        
        this.orderBookData.asks = asks.map(ask => {
            // Small random price and size adjustments
            const priceChange = ask.price * (Math.random() * 0.002 - 0.001);
            const sizeChange = ask.size * (Math.random() * 0.1 - 0.05);
            
            return {
                price: ask.price + priceChange,
                size: Math.max(0.1, ask.size + sizeChange)
            };
        });
        
        // Sort bids and asks to maintain order
        this.orderBookData.bids.sort((a, b) => b.price - a.price);
        this.orderBookData.asks.sort((a, b) => a.price - b.price);
        
        // Update timestamp
        this.orderBookData.timestamp = new Date();
    }

    updateTradeFlowData() {
        // Remove the oldest point
        if (this.tradeFlowHistory.length >= this.options.maxHistoryPoints) {
            this.tradeFlowHistory.shift();
        }
        
        // Get the latest point
        const lastPoint = this.tradeFlowHistory[this.tradeFlowHistory.length - 1];
        
        // Create a new point based on the last point
        const timestamp = new Date();
        const buyVolume = Math.max(10, lastPoint.buyVolume + (Math.random() * 20 - 10));
        const sellVolume = Math.max(10, lastPoint.sellVolume + (Math.random() * 20 - 10));
        const buyTrades = Math.floor(buyVolume / 5);
        const sellTrades = Math.floor(sellVolume / 5);
        
        // Add new point
        this.tradeFlowHistory.push({
            timestamp: timestamp,
            buyVolume: buyVolume,
            sellVolume: sellVolume,
            buyTrades: buyTrades,
            sellTrades: sellTrades
        });
        
        // Update imbalance calculation
        this.calculateImbalance();
    }

    updateLargeOrdersData() {
        // Update timestamps to move orders forward in time
        this.largeOrdersData = this.largeOrdersData.map(order => {
            // Move timestamp forward
            const newTimestamp = new Date(order.timestamp.getTime() + 60000);
            
            // If order is too old, regenerate it
            if (newTimestamp > new Date()) {
                const basePrice = this.getBasePrice();
                const type = Math.random() > 0.5 ? 'bid' : 'ask';
                const priceOffset = (Math.random() * 0.02) * (type === 'bid' ? -1 : 1);
                const price = basePrice * (1 + priceOffset);
                const size = 10 + Math.random() * 40;
                
                return {
                    timestamp: new Date(Date.now() - Math.random() * 3600000), // Within the last hour
                    type: type,
                    price: price,
                    size: size
                };
            }
            
            return {
                ...order,
                timestamp: newTimestamp
            };
        });
        
        // Sort by timestamp
        this.largeOrdersData.sort((a, b) => a.timestamp - b.timestamp);
        
        // Randomly add a new large order (10% chance)
        if (Math.random() < 0.1) {
            const basePrice = this.getBasePrice();
            const type = Math.random() > 0.5 ? 'bid' : 'ask';
            const priceOffset = (Math.random() * 0.02) * (type === 'bid' ? -1 : 1);
            const price = basePrice * (1 + priceOffset);
            const size = 10 + Math.random() * 40;
            
            this.largeOrdersData.push({
                timestamp: new Date(),
                type: type,
                price: price,
                size: size
            });
        }
    }

    calculateImbalance() {
        // Calculate imbalance from trade flow history
        this.imbalanceHistory = this.tradeFlowHistory.map(point => {
            const totalVolume = point.buyVolume + point.sellVolume;
            const imbalance = totalVolume > 0 ? (point.buyVolume - point.sellVolume) / totalVolume : 0;
            
            return {
                timestamp: point.timestamp,
                imbalance: imbalance
            };
        });
    }

    calculateAverageImbalance() {
        // Calculate the average imbalance from the history
        if (this.imbalanceHistory.length === 0) return 0;
        
        const sum = this.imbalanceHistory.reduce((total, point) => total + point.imbalance, 0);
        return sum / this.imbalanceHistory.length;
    }

    calculateCumulativeDepth(orders, type) {
        // Calculate cumulative depth for order book visualization
        let cumulative = [];
        let cumulativeSize = 0;
        
        if (type === 'bid') {
            // For bids, price decreases as we go deeper into the book
            // Start from highest bid
            orders.forEach(order => {
                cumulativeSize += order.size;
                cumulative.push({
                    price: order.price,
                    cumulativeSize: cumulativeSize
                });
            });
        } else {
            // For asks, price increases as we go deeper into the book
            // Start from lowest ask
            orders.forEach(order => {
                cumulativeSize += order.size;
                cumulative.push({
                    price: order.price,
                    cumulativeSize: cumulativeSize
                });
            });
        }
        
        return cumulative;
    }

    // Helper functions
    getBasePrice() {
        // Get a realistic base price for the selected symbol
        switch (this.symbol) {
            case 'BTC-USD':
                return 30000 + Math.random() * 5000;
            case 'ETH-USD':
                return 2000 + Math.random() * 400;
            case 'SOL-USD':
                return 80 + Math.random() * 20;
            case 'BNB-USD':
                return 300 + Math.random() * 50;
            default:
                return 1000 + Math.random() * 200;
        }
    }

    getMaxOrderSize(orders) {
        // Get the maximum order size from the list
        return Math.max(...orders.map(order => order.size));
    }

    showLoading(element) {
        // Show loading spinner on the charts
        const elements = element ? [element] : [
            document.getElementById(this.options.depthChartElementId),
            document.getElementById(this.options.orderBookHeatmapElementId),
            document.getElementById(this.options.tradeFlowElementId),
            document.getElementById(this.options.largeOrdersElementId)
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
        // Hide loading spinner
        const elements = element ? [element] : [
            document.getElementById(this.options.depthChartElementId),
            document.getElementById(this.options.orderBookHeatmapElementId),
            document.getElementById(this.options.tradeFlowElementId),
            document.getElementById(this.options.largeOrdersElementId)
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
        // Show error message on the charts
        const elements = element ? [element] : [
            document.getElementById(this.options.depthChartElementId),
            document.getElementById(this.options.orderBookHeatmapElementId),
            document.getElementById(this.options.tradeFlowElementId),
            document.getElementById(this.options.largeOrdersElementId)
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
            symbol: this.symbol,
            timestamp: new Date(),
            orderBook: this.orderBookData,
            tradeFlow: this.tradeFlowHistory,
            imbalance: this.imbalanceHistory,
            largeOrders: this.largeOrdersData
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `order_flow_${this.symbol}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    getColorFromScale(value, scale) {
        // Get color from a colorscale based on value (0-1)
        const colorscale = scale || this.options.colorScaleBid;
        
        // Handle edge cases
        if (value <= colorscale[0][0]) return colorscale[0][1];
        if (value >= colorscale[colorscale.length - 1][0]) return colorscale[colorscale.length - 1][1];
        
        // Find the color segments the value falls between
        for (let i = 0; i < colorscale.length - 1; i++) {
            if (value >= colorscale[i][0] && value <= colorscale[i + 1][0]) {
                const color1 = this.parseRgba(colorscale[i][1]);
                const color2 = this.parseRgba(colorscale[i + 1][1]);
                
                // Normalize value to 0-1 range within this segment
                const segmentSize = colorscale[i + 1][0] - colorscale[i][0];
                const normalizedValue = (value - colorscale[i][0]) / segmentSize;
                
                // Interpolate colors
                const r = Math.round(color1.r + normalizedValue * (color2.r - color1.r));
                const g = Math.round(color1.g + normalizedValue * (color2.g - color1.g));
                const b = Math.round(color1.b + normalizedValue * (color2.b - color1.b));
                const a = color1.a + normalizedValue * (color2.a - color1.a);
                
                return `rgba(${r}, ${g}, ${b}, ${a})`;
            }
        }
        
        // Fallback
        return colorscale[0][1];
    }

    parseRgba(color) {
        // Parse rgba color string to object
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of AdvancedOrderFlow
    const orderFlow = new AdvancedOrderFlow();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});