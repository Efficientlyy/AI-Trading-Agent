/**
 * Execution Quality Analytics
 * 
 * This module provides functionality for the Execution Quality Analytics component,
 * showing fill rates, market impact, execution algorithm performance comparisons,
 * and venue analytics to help users optimize their execution strategies.
 */

class ExecutionQualityAnalysis {
    constructor(options = {}) {
        this.options = Object.assign({
            fillRateElementId: 'fill-rate-chart',
            marketImpactElementId: 'market-impact-chart',
            algoPerformanceElementId: 'algo-performance-chart',
            venueAnalyticsElementId: 'venue-analytics-chart',
            executionTrendElementId: 'execution-trend-chart',
            colorScaleFillRate: [
                [0, 'rgba(239, 68, 68, 0.7)'],     // Poor fill rate (red)
                [0.5, 'rgba(234, 179, 8, 0.7)'],   // Medium fill rate (yellow)
                [1, 'rgba(16, 185, 129, 0.7)']     // Good fill rate (green)
            ],
            colorScaleMarketImpact: [
                [0, 'rgba(16, 185, 129, 0.7)'],    // Low impact (green)
                [0.5, 'rgba(234, 179, 8, 0.7)'],   // Medium impact (yellow)
                [1, 'rgba(239, 68, 68, 0.7)']      // High impact (red)
            ]
        }, options);
        
        // Initialize from UI control values
        this.strategy = document.getElementById('execution-quality-strategy')?.value || 'all';
        this.asset = document.getElementById('execution-quality-asset')?.value || 'all';
        this.timeRange = document.getElementById('execution-quality-timerange')?.value || '1m';
        this.orderType = document.getElementById('execution-quality-type')?.value || 'all';
        
        // Initialize chart-specific settings
        this.fillRateMetric = document.getElementById('fill-rate-metric')?.value || 'percentage';
        this.fillRateGroup = document.getElementById('fill-rate-group')?.value || 'asset';
        this.impactView = document.getElementById('impact-view')?.value || 'price';
        this.orderSize = document.getElementById('order-size')?.value || 'all';
        this.algoMetric = document.getElementById('algo-metric')?.value || 'vs_arrival';
        this.algoChartType = document.getElementById('algo-chart-type')?.value || 'bar';
        this.venueMetric = document.getElementById('venue-metric')?.value || 'fees';
        this.venueView = document.getElementById('venue-view')?.value || 'comparison';
        
        // Initialize data containers
        this.fillRateData = {};
        this.marketImpactData = {};
        this.algoPerformanceData = {};
        this.venueAnalyticsData = {};
        this.summaryMetrics = {};
        this.recommendationData = [];
        
        // Initialize chart objects
        this.fillRateChart = null;
        this.marketImpactChart = null;
        this.algoPerformanceChart = null;
        this.venueAnalyticsChart = null;
        this.executionTrendChart = null;
        
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
                console.error('Error initializing Execution Quality Analysis:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchFillRateData(),
            this.fetchMarketImpactData(),
            this.fetchAlgoPerformanceData(),
            this.fetchVenueAnalyticsData(),
            this.fetchSummaryMetrics(),
            this.fetchRecommendations()
        ]);
    }
    
    fetchFillRateData() {
        // Mock data for fill rate analysis
        return new Promise(resolve => {
            setTimeout(() => {
                const data = {
                    byAsset: {
                        labels: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'],
                        percentage: [96.8, 94.2, 92.7, 95.3],
                        time: [0.9, 1.2, 1.4, 1.1],
                        partial: [12, 18, 22, 14]
                    },
                    byVenue: {
                        labels: ['Exchange A', 'Exchange B', 'Exchange C', 'Exchange D'],
                        percentage: [97.3, 95.1, 93.8, 94.5],
                        time: [0.8, 1.1, 1.5, 1.3],
                        partial: [10, 15, 20, 18]
                    },
                    byStrategy: {
                        labels: ['TWAP', 'VWAP', 'Iceberg', 'Smart Order Routing'],
                        percentage: [94.5, 95.8, 93.2, 97.1],
                        time: [1.2, 1.0, 1.5, 0.9],
                        partial: [15, 12, 18, 10]
                    },
                    byTime: {
                        labels: ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00'],
                        percentage: [93.1, 94.8, 96.5, 95.2, 94.7, 92.9],
                        time: [1.5, 1.2, 0.8, 1.0, 1.1, 1.3],
                        partial: [20, 15, 10, 12, 14, 18]
                    }
                };
                
                // Filter based on strategy and asset if needed
                if (this.strategy !== 'all') {
                    // Apply filtering logic here in a real implementation
                }
                
                if (this.asset !== 'all') {
                    // Apply filtering logic here in a real implementation
                }
                
                this.fillRateData = data;
                resolve(data);
            }, 300);
        });
    }
    
    fetchMarketImpactData() {
        // Mock data for market impact analysis
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate random market impact data
                const assets = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'];
                const sizes = ['Small', 'Medium', 'Large'];
                const times = [];
                
                // Generate time points for the last 30 days
                const now = new Date();
                for (let i = 29; i >= 0; i--) {
                    const date = new Date(now);
                    date.setDate(date.getDate() - i);
                    times.push(date);
                }
                
                // Generate price impact data
                const priceImpactBySize = {};
                sizes.forEach(size => {
                    priceImpactBySize[size] = {
                        x: times,
                        y: times.map(() => {
                            let base = 0;
                            if (size === 'Small') base = 0.05 + (Math.random() * 0.05);
                            else if (size === 'Medium') base = 0.15 + (Math.random() * 0.1);
                            else if (size === 'Large') base = 0.25 + (Math.random() * 0.15);
                            
                            return base + (Math.random() * 0.05); // Add some randomness
                        }),
                        name: size
                    };
                });
                
                // Generate spread widening data
                const spreadWideningBySize = {};
                sizes.forEach(size => {
                    spreadWideningBySize[size] = {
                        x: times,
                        y: times.map(() => {
                            let base = 0;
                            if (size === 'Small') base = 0.02 + (Math.random() * 0.03);
                            else if (size === 'Medium') base = 0.05 + (Math.random() * 0.05);
                            else if (size === 'Large') base = 0.1 + (Math.random() * 0.1);
                            
                            return base + (Math.random() * 0.02); // Add some randomness
                        }),
                        name: size
                    };
                });
                
                // Generate order book depth impact
                const depthImpactBySize = {};
                sizes.forEach(size => {
                    depthImpactBySize[size] = {
                        x: times,
                        y: times.map(() => {
                            let base = 0;
                            if (size === 'Small') base = 0.5 + (Math.random() * 0.5);
                            else if (size === 'Medium') base = 1.0 + (Math.random() * 1.0);
                            else if (size === 'Large') base = 2.0 + (Math.random() * 2.0);
                            
                            return base + (Math.random() * 0.5); // Add some randomness
                        }),
                        name: size
                    };
                });
                
                // Heatmap data for different assets and order sizes
                const heatmapData = {
                    x: sizes,
                    y: assets,
                    price: [
                        [0.08, 0.18, 0.32], // BTC-USD
                        [0.12, 0.22, 0.38], // ETH-USD
                        [0.15, 0.28, 0.42], // SOL-USD
                        [0.10, 0.20, 0.35]  // BNB-USD
                    ],
                    spread: [
                        [0.03, 0.08, 0.15], // BTC-USD
                        [0.04, 0.10, 0.18], // ETH-USD
                        [0.05, 0.12, 0.20], // SOL-USD
                        [0.03, 0.09, 0.16]  // BNB-USD
                    ],
                    depth: [
                        [0.8, 1.5, 2.5], // BTC-USD
                        [1.0, 1.8, 3.0], // ETH-USD
                        [1.1, 2.0, 3.2], // SOL-USD
                        [0.9, 1.7, 2.8]  // BNB-USD
                    ]
                };
                
                // Compile all data
                const data = {
                    priceImpact: priceImpactBySize,
                    spreadWidening: spreadWideningBySize,
                    depthImpact: depthImpactBySize,
                    heatmap: heatmapData
                };
                
                this.marketImpactData = data;
                resolve(data);
            }, 350);
        });
    }
    
    fetchAlgoPerformanceData() {
        // Mock data for algorithm performance comparison
        return new Promise(resolve => {
            setTimeout(() => {
                const algorithms = ['TWAP', 'VWAP', 'Iceberg', 'Smart Order Routing'];
                
                // Performance vs arrival price (bps)
                const vsArrival = {
                    values: [15, 22, 8, 26],
                    uncertainty: [5, 8, 3, 10],
                    rawData: algorithms.map(algo => {
                        return {
                            algorithm: algo,
                            samples: Array(20).fill(0).map(() => {
                                let base = 0;
                                if (algo === 'TWAP') base = 15;
                                else if (algo === 'VWAP') base = 22;
                                else if (algo === 'Iceberg') base = 8;
                                else if (algo === 'Smart Order Routing') base = 26;
                                
                                return base + (Math.random() * 10 - 5); // +/- 5 bps variation
                            })
                        };
                    })
                };
                
                // Performance vs VWAP (bps)
                const vsVwap = {
                    values: [8, 0, 4, 12],
                    uncertainty: [4, 0, 2, 6],
                    rawData: algorithms.map(algo => {
                        return {
                            algorithm: algo,
                            samples: Array(20).fill(0).map(() => {
                                let base = 0;
                                if (algo === 'TWAP') base = 8;
                                else if (algo === 'VWAP') base = 0;
                                else if (algo === 'Iceberg') base = 4;
                                else if (algo === 'Smart Order Routing') base = 12;
                                
                                return base + (Math.random() * 8 - 4); // +/- 4 bps variation
                            })
                        };
                    })
                };
                
                // Performance vs TWAP (bps)
                const vsTwap = {
                    values: [0, 7, -3, 11],
                    uncertainty: [0, 3, 2, 5],
                    rawData: algorithms.map(algo => {
                        return {
                            algorithm: algo,
                            samples: Array(20).fill(0).map(() => {
                                let base = 0;
                                if (algo === 'TWAP') base = 0;
                                else if (algo === 'VWAP') base = 7;
                                else if (algo === 'Iceberg') base = -3;
                                else if (algo === 'Smart Order Routing') base = 11;
                                
                                return base + (Math.random() * 6 - 3); // +/- 3 bps variation
                            })
                        };
                    })
                };
                
                // Implementation shortfall (bps)
                const implShortfall = {
                    values: [25, 18, 32, 15],
                    uncertainty: [8, 6, 10, 5],
                    rawData: algorithms.map(algo => {
                        return {
                            algorithm: algo,
                            samples: Array(20).fill(0).map(() => {
                                let base = 0;
                                if (algo === 'TWAP') base = 25;
                                else if (algo === 'VWAP') base = 18;
                                else if (algo === 'Iceberg') base = 32;
                                else if (algo === 'Smart Order Routing') base = 15;
                                
                                return base + (Math.random() * 10 - 5); // +/- 5 bps variation
                            })
                        };
                    })
                };
                
                const data = {
                    algorithms: algorithms,
                    vsArrival: vsArrival,
                    vsVwap: vsVwap,
                    vsTwap: vsTwap,
                    implShortfall: implShortfall
                };
                
                this.algoPerformanceData = data;
                resolve(data);
            }, 400);
        });
    }
    
    fetchVenueAnalyticsData() {
        // Mock data for venue analytics
        return new Promise(resolve => {
            setTimeout(() => {
                const venues = ['Exchange A', 'Exchange B', 'Exchange C', 'Exchange D'];
                
                // Fees & costs (bps)
                const fees = {
                    maker: [5, 8, 12, 7],
                    taker: [15, 18, 22, 16],
                    average: [10, 13, 17, 12]
                };
                
                // Latency (ms)
                const latency = {
                    median: [45, 78, 120, 62],
                    p95: [85, 125, 180, 110],
                    p99: [120, 180, 250, 150]
                };
                
                // Reliability (%)
                const reliability = {
                    uptime: [99.98, 99.95, 99.9, 99.97],
                    orderSuccess: [99.8, 99.7, 99.5, 99.75],
                    apiAvailability: [99.9, 99.85, 99.8, 99.88]
                };
                
                // Combined score (0-100)
                const combinedScore = {
                    overall: [92, 88, 82, 90],
                    costScore: [95, 90, 80, 93],
                    latencyScore: [90, 80, 70, 85],
                    reliabilityScore: [95, 92, 88, 94]
                };
                
                // Time series data for trend analysis
                const trendData = {};
                const days = 30;
                const dates = [];
                
                // Generate dates
                const now = new Date();
                for (let i = days - 1; i >= 0; i--) {
                    const date = new Date(now);
                    date.setDate(date.getDate() - i);
                    dates.push(date);
                }
                
                // Generate trend data for each venue
                venues.forEach((venue, index) => {
                    trendData[venue] = {
                        dates: dates,
                        fees: dates.map(() => {
                            const base = fees.average[index];
                            return base * (0.95 + Math.random() * 0.1); // +/- 5% variance
                        }),
                        latency: dates.map(() => {
                            const base = latency.median[index];
                            return base * (0.9 + Math.random() * 0.2); // +/- 10% variance
                        }),
                        reliability: dates.map(() => {
                            const base = 100 - (100 - reliability.uptime[index]) * (0.8 + Math.random() * 0.4); // Variance in reliability
                            return base;
                        }),
                        score: dates.map(() => {
                            const base = combinedScore.overall[index];
                            return base * (0.97 + Math.random() * 0.06); // +/- 3% variance
                        })
                    };
                });
                
                const data = {
                    venues: venues,
                    fees: fees,
                    latency: latency,
                    reliability: reliability,
                    combinedScore: combinedScore,
                    trend: trendData
                };
                
                this.venueAnalyticsData = data;
                resolve(data);
            }, 350);
        });
    }
    
    fetchSummaryMetrics() {
        // Mock data for summary metrics
        return new Promise(resolve => {
            setTimeout(() => {
                const metrics = {
                    avgFillRate: '96.8%',
                    avgFillTime: '1.2s',
                    avgPriceImpact: '0.18%',
                    fulfilledOrders: '94.2%',
                    bestAlgorithm: 'VWAP (+0.12%)',
                    executionCost: '0.32%',
                    bestVenue: 'Exchange A (97.3% fills)',
                    bestTimeWindow: '09:00-11:00 UTC',
                    costSavingsPotential: '18 bps',
                    lastUpdated: Math.floor(Math.random() * 10) + ' minutes ago',
                    
                    // Quality trend data for mini chart
                    qualityTrend: []
                };
                
                // Generate trend data (30 days)
                for (let i = 0; i < 30; i++) {
                    metrics.qualityTrend.push({
                        date: new Date(new Date().setDate(new Date().getDate() - 29 + i)),
                        value: 0.92 + Math.cos(i / 5) * 0.03 + (Math.random() * 0.02) // Oscillating pattern with noise
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
                        title: 'Optimize Order Sizes for BTC/USD',
                        priority: 'high',
                        description: 'Orders over 1.5 BTC show 2.8x higher market impact. Consider splitting large orders into smaller chunks of 0.75-1 BTC max to minimize price impact.'
                    },
                    {
                        title: 'Switch from TWAP to VWAP During High Volatility',
                        priority: 'medium',
                        description: 'VWAP algorithm outperforms TWAP by 22 bps during high volatility periods. Consider switching default algorithm when volatility exceeds 1.5x normal levels.'
                    },
                    {
                        title: 'Optimize Exchange Selection',
                        priority: 'low',
                        description: 'Using Exchange B for orders under $10K could save approximately 6 bps in fees with equivalent fill rates and execution quality.'
                    }
                ];
                
                // Add more recommendations based on settings
                if (this.strategy === 'iceberg') {
                    recommendations.push({
                        title: 'Adjust Iceberg Order Parameters',
                        priority: 'medium',
                        description: 'Current iceberg displayed quantity (10%) is too low for effective execution. Consider increasing to 15-20% to improve fill rates by estimated 8%.'
                    });
                }
                
                if (this.orderType === 'algo') {
                    recommendations.push({
                        title: 'Reduce Algo Execution Window',
                        priority: 'medium',
                        description: 'Algo orders with >30 minute execution windows showing 15% higher slippage. Consider reducing max execution time to 20 minutes when market volatility is low.'
                    });
                }
                
                this.recommendationData = recommendations;
                resolve(recommendations);
            }, 300);
        });
    }
    
    initializeVisualizations() {
        this.renderFillRate();
        this.renderMarketImpact();
        this.renderAlgoPerformance();
        this.renderVenueAnalytics();
        this.renderQualityTrend();
        this.updateSummaryMetrics();
        this.updateRecommendations();
    }
    
    renderFillRate() {
        const element = document.getElementById(this.options.fillRateElementId);
        if (!element || !this.fillRateData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Determine which data to display based on the group by selection
        let data;
        switch (this.fillRateGroup) {
            case 'asset':
                data = this.fillRateData.byAsset;
                break;
            case 'venue':
                data = this.fillRateData.byVenue;
                break;
            case 'strategy':
                data = this.fillRateData.byStrategy;
                break;
            case 'time':
                data = this.fillRateData.byTime;
                break;
            default:
                data = this.fillRateData.byAsset;
        }
        
        // Determine which metric to display
        let values, yAxisTitle;
        switch (this.fillRateMetric) {
            case 'percentage':
                values = data.percentage;
                yAxisTitle = 'Fill Rate (%)';
                break;
            case 'time':
                values = data.time;
                yAxisTitle = 'Fill Time (seconds)';
                break;
            case 'partial':
                values = data.partial;
                yAxisTitle = 'Partial Fills (%)';
                break;
            default:
                values = data.percentage;
                yAxisTitle = 'Fill Rate (%)';
        }
        
        // Create colors based on the metric
        let colors;
        if (this.fillRateMetric === 'percentage') {
            // For percentage, higher is better
            colors = values.map(v => {
                const normalizedValue = v / 100;
                return this.getColorFromScale(normalizedValue, this.options.colorScaleFillRate);
            });
        } else {
            // For time and partial fills, lower is better
            const maxValue = Math.max(...values);
            colors = values.map(v => {
                const normalizedValue = 1 - (v / maxValue);
                return this.getColorFromScale(normalizedValue, this.options.colorScaleFillRate);
            });
        }
        
        // Create trace for bar chart
        const trace = {
            type: 'bar',
            x: data.labels,
            y: values,
            text: values.map(v => {
                if (this.fillRateMetric === 'percentage') return v.toFixed(1) + '%';
                else if (this.fillRateMetric === 'time') return v.toFixed(1) + 's';
                else return v.toFixed(1) + '%';
            }),
            textposition: 'auto',
            hoverinfo: 'x+y',
            marker: {
                color: colors,
                line: {
                    color: 'rgba(var(--border-rgb), 0.5)',
                    width: 1
                }
            }
        };
        
        // Layout configuration
        const layout = {
            title: `Fill Rate Analysis by ${this.capitalizeFirst(this.fillRateGroup)}`,
            margin: { l: 60, r: 20, t: 40, b: 80 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: this.capitalizeFirst(this.fillRateGroup),
                showgrid: false,
                linecolor: 'var(--border-light)',
                tickangle: -45
            },
            yaxis: {
                title: yAxisTitle,
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)'
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.fillRateElementId, [trace], layout, config);
        this.fillRateChart = document.getElementById(this.options.fillRateElementId);
    }
    
    renderMarketImpact() {
        const element = document.getElementById(this.options.marketImpactElementId);
        if (!element || !this.marketImpactData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        let traces = [];
        let layout = {};
        
        // Determine which data to display based on the impact view selection
        let data;
        let title;
        let zValuesKey;
        
        switch (this.impactView) {
            case 'price':
                data = this.marketImpactData.priceImpact;
                title = 'Price Impact by Order Size';
                zValuesKey = 'price';
                break;
            case 'spread':
                data = this.marketImpactData.spreadWidening;
                title = 'Spread Widening by Order Size';
                zValuesKey = 'spread';
                break;
            case 'depth':
                data = this.marketImpactData.depthImpact;
                title = 'Order Book Depth Impact by Order Size';
                zValuesKey = 'depth';
                break;
            default:
                data = this.marketImpactData.priceImpact;
                title = 'Price Impact by Order Size';
                zValuesKey = 'price';
        }
        
        // Filter by order size if specified
        let filteredData = data;
        if (this.orderSize !== 'all') {
            filteredData = {};
            filteredData[this.orderSize] = data[this.orderSize];
        }
        
        // Create a heatmap view
        const heatmapTrace = {
            type: 'heatmap',
            x: this.marketImpactData.heatmap.x,
            y: this.marketImpactData.heatmap.y,
            z: this.marketImpactData.heatmap[zValuesKey],
            colorscale: [
                [0, 'rgba(var(--success-rgb), 0.7)'],
                [0.5, 'rgba(var(--warning-rgb), 0.7)'],
                [1, 'rgba(var(--danger-rgb), 0.7)']
            ],
            hoverongaps: false,
            showscale: true,
            colorbar: {
                title: this.impactView === 'price' ? 'Impact %' : 
                       this.impactView === 'spread' ? 'Widening %' : 'Depth Impact %',
                titleside: 'right',
                thickness: 10
            }
        };
        
        traces = [heatmapTrace];
        
        layout = {
            title: `${title} by Asset`,
            margin: { l: 70, r: 60, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Order Size',
                linecolor: 'var(--border-light)'
            },
            yaxis: {
                title: 'Asset',
                linecolor: 'var(--border-light)'
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.marketImpactElementId, traces, layout, config);
        this.marketImpactChart = document.getElementById(this.options.marketImpactElementId);
    }
    
    renderAlgoPerformance() {
        const element = document.getElementById(this.options.algoPerformanceElementId);
        if (!element || !this.algoPerformanceData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Determine which data to display based on the algo metric selection
        let data;
        let title;
        
        switch (this.algoMetric) {
            case 'vs_arrival':
                data = this.algoPerformanceData.vsArrival;
                title = 'Algorithm Performance vs. Arrival Price';
                break;
            case 'vs_vwap':
                data = this.algoPerformanceData.vsVwap;
                title = 'Algorithm Performance vs. VWAP';
                break;
            case 'vs_twap':
                data = this.algoPerformanceData.vsTwap;
                title = 'Algorithm Performance vs. TWAP';
                break;
            case 'cost':
                data = this.algoPerformanceData.implShortfall;
                title = 'Implementation Shortfall by Algorithm';
                break;
            default:
                data = this.algoPerformanceData.vsArrival;
                title = 'Algorithm Performance vs. Arrival Price';
        }
        
        let traces = [];
        let layout = {};
        
        // Create visualization based on chart type
        if (this.algoChartType === 'bar') {
            // Bar chart with error bars
            traces = [{
                type: 'bar',
                x: this.algoPerformanceData.algorithms,
                y: data.values,
                error_y: {
                    type: 'data',
                    array: data.uncertainty,
                    visible: true,
                    color: 'rgba(var(--text-light-rgb), 0.5)'
                },
                marker: {
                    color: data.values.map(v => {
                        if (this.algoMetric === 'cost') {
                            // For cost, lower is better
                            const maxValue = Math.max(...data.values);
                            const normalizedValue = 1 - (v / maxValue);
                            return this.getColorFromScale(normalizedValue, this.options.colorScaleFillRate);
                        } else {
                            // For performance, higher is better
                            const maxValue = Math.max(...data.values.map(Math.abs));
                            const normalizedValue = (v + maxValue) / (2 * maxValue);
                            return this.getColorFromScale(normalizedValue, this.options.colorScaleFillRate);
                        }
                    })
                },
                text: data.values.map(v => `${v.toFixed(1)} bps`),
                textposition: 'auto',
                hoverinfo: 'x+y'
            }];
            
            layout = {
                title: title,
                margin: { l: 60, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Algorithm',
                    showgrid: false,
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Performance (bps)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)'
                }
            };
        } else if (this.algoChartType === 'box') {
            // Box plot showing distribution
            traces = data.rawData.map(item => ({
                type: 'box',
                y: item.samples,
                name: item.algorithm,
                boxpoints: 'outliers',
                boxmean: true,
                marker: {
                    color: 'rgba(var(--primary-rgb), 0.5)',
                    size: 4
                },
                line: {
                    width: 1
                }
            }));
            
            layout = {
                title: title,
                margin: { l: 60, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Algorithm',
                    showgrid: false,
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Performance (bps)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)',
                    zeroline: true,
                    zerolinecolor: 'var(--border-light)',
                    zerolinewidth: 1
                }
            };
        } else if (this.algoChartType === 'scatter') {
            // Scatter plot showing all data points
            traces = data.rawData.map(item => ({
                type: 'scatter',
                mode: 'markers',
                x: Array(item.samples.length).fill(item.algorithm),
                y: item.samples,
                name: item.algorithm,
                marker: {
                    size: 8,
                    opacity: 0.7,
                    color: this.getAlgorithmColor(item.algorithm)
                }
            }));
            
            layout = {
                title: title,
                margin: { l: 60, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Algorithm',
                    showgrid: false,
                    linecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: 'Performance (bps)',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    linecolor: 'var(--border-light)',
                    zeroline: true,
                    zerolinecolor: 'var(--border-light)',
                    zerolinewidth: 1
                }
            };
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.algoPerformanceElementId, traces, layout, config);
        this.algoPerformanceChart = document.getElementById(this.options.algoPerformanceElementId);
    }
    
    renderVenueAnalytics() {
        const element = document.getElementById(this.options.venueAnalyticsElementId);
        if (!element || !this.venueAnalyticsData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Determine which data to display based on the venue metric selection
        let data, traces, layout, title, yAxisTitle;
        const venues = this.venueAnalyticsData.venues;
        
        // Choose the appropriate data based on metric
        switch (this.venueMetric) {
            case 'fees':
                data = this.venueAnalyticsData.fees;
                title = 'Exchange Fee Comparison';
                yAxisTitle = 'Fee Rate (bps)';
                break;
            case 'latency':
                data = this.venueAnalyticsData.latency;
                title = 'Exchange Latency Comparison';
                yAxisTitle = 'Latency (ms)';
                break;
            case 'reliability':
                data = this.venueAnalyticsData.reliability;
                title = 'Exchange Reliability Metrics';
                yAxisTitle = 'Reliability (%)';
                break;
            case 'combined':
                data = this.venueAnalyticsData.combinedScore;
                title = 'Exchange Combined Performance Score';
                yAxisTitle = 'Score (0-100)';
                break;
            default:
                data = this.venueAnalyticsData.fees;
                title = 'Exchange Fee Comparison';
                yAxisTitle = 'Fee Rate (bps)';
        }
        
        // Choose the appropriate visualization based on view
        switch (this.venueView) {
            case 'comparison':
                // Bar chart comparison
                if (this.venueMetric === 'fees') {
                    // For fees, create grouped bar chart with maker/taker
                    traces = [
                        {
                            type: 'bar',
                            name: 'Maker',
                            x: venues,
                            y: data.maker,
                            marker: { color: 'rgba(var(--success-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'Taker',
                            x: venues,
                            y: data.taker,
                            marker: { color: 'rgba(var(--info-rgb), 0.7)' }
                        }
                    ];
                } else if (this.venueMetric === 'latency') {
                    // For latency, create grouped bar chart with median/p95/p99
                    traces = [
                        {
                            type: 'bar',
                            name: 'Median',
                            x: venues,
                            y: data.median,
                            marker: { color: 'rgba(var(--success-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'P95',
                            x: venues,
                            y: data.p95,
                            marker: { color: 'rgba(var(--warning-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'P99',
                            x: venues,
                            y: data.p99,
                            marker: { color: 'rgba(var(--danger-rgb), 0.7)' }
                        }
                    ];
                } else if (this.venueMetric === 'reliability') {
                    // For reliability, create grouped bar chart with uptime/orderSuccess/apiAvailability
                    traces = [
                        {
                            type: 'bar',
                            name: 'Uptime',
                            x: venues,
                            y: data.uptime,
                            marker: { color: 'rgba(var(--success-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'Order Success',
                            x: venues,
                            y: data.orderSuccess,
                            marker: { color: 'rgba(var(--info-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'API Availability',
                            x: venues,
                            y: data.apiAvailability,
                            marker: { color: 'rgba(var(--warning-rgb), 0.7)' }
                        }
                    ];
                } else {
                    // For combined score, create grouped bar chart with overall/components
                    traces = [
                        {
                            type: 'bar',
                            name: 'Overall',
                            x: venues,
                            y: data.overall,
                            marker: { color: 'rgba(var(--primary-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'Cost',
                            x: venues,
                            y: data.costScore,
                            marker: { color: 'rgba(var(--success-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'Latency',
                            x: venues,
                            y: data.latencyScore,
                            marker: { color: 'rgba(var(--info-rgb), 0.7)' }
                        },
                        {
                            type: 'bar',
                            name: 'Reliability',
                            x: venues,
                            y: data.reliabilityScore,
                            marker: { color: 'rgba(var(--warning-rgb), 0.7)' }
                        }
                    ];
                }
                
                layout = {
                    title: title,
                    margin: { l: 60, r: 20, t: 40, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    xaxis: {
                        title: 'Exchange',
                        showgrid: false,
                        linecolor: 'var(--border-light)'
                    },
                    yaxis: {
                        title: yAxisTitle,
                        showgrid: true,
                        gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                        linecolor: 'var(--border-light)'
                    },
                    barmode: 'group',
                    legend: {
                        orientation: 'h',
                        xanchor: 'center',
                        x: 0.5,
                        y: 1.05
                    }
                };
                break;
                
            case 'ranking':
                // Spider/radar chart for venue ranking
                let categories, values;
                
                if (this.venueMetric === 'combined') {
                    categories = ['Overall', 'Cost', 'Latency', 'Reliability'];
                    values = venues.map((venue, i) => [
                        data.overall[i], 
                        data.costScore[i], 
                        data.latencyScore[i], 
                        data.reliabilityScore[i]
                    ]);
                } else if (this.venueMetric === 'fees') {
                    categories = ['Maker Fee', 'Taker Fee', 'Average Fee'];
                    values = venues.map((venue, i) => [
                        100 - data.maker[i], // Invert for better visualization
                        100 - data.taker[i], 
                        100 - data.average[i]
                    ]);
                } else if (this.venueMetric === 'latency') {
                    categories = ['Median Latency', 'P95 Latency', 'P99 Latency'];
                    // Normalize to 0-100 score where lower is better
                    const maxMedian = Math.max(...data.median);
                    const maxP95 = Math.max(...data.p95);
                    const maxP99 = Math.max(...data.p99);
                    
                    values = venues.map((venue, i) => [
                        100 - (data.median[i] / maxMedian * 100),
                        100 - (data.p95[i] / maxP95 * 100),
                        100 - (data.p99[i] / maxP99 * 100)
                    ]);
                } else {
                    categories = ['Uptime', 'Order Success', 'API Availability'];
                    values = venues.map((venue, i) => [
                        data.uptime[i],
                        data.orderSuccess[i],
                        data.apiAvailability[i]
                    ]);
                }
                
                traces = venues.map((venue, i) => ({
                    type: 'scatterpolar',
                    r: values[i],
                    theta: categories,
                    fill: 'toself',
                    name: venue
                }));
                
                layout = {
                    title: `Venue Ranking - ${this.capitalizeFirst(this.venueMetric)}`,
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    polar: {
                        radialaxis: {
                            visible: true,
                            range: [0, 100]
                        }
                    },
                    showlegend: true,
                    legend: {
                        orientation: 'h',
                        xanchor: 'center',
                        x: 0.5,
                        y: 1.05
                    }
                };
                break;
                
            case 'trend':
                // Line chart showing metric trend over time
                const trendData = this.venueAnalyticsData.trend;
                let metricKey;
                
                switch (this.venueMetric) {
                    case 'fees':
                        metricKey = 'fees';
                        break;
                    case 'latency':
                        metricKey = 'latency';
                        break;
                    case 'reliability':
                        metricKey = 'reliability';
                        break;
                    case 'combined':
                        metricKey = 'score';
                        break;
                    default:
                        metricKey = 'score';
                }
                
                traces = venues.map(venue => ({
                    type: 'scatter',
                    mode: 'lines',
                    name: venue,
                    x: trendData[venue].dates,
                    y: trendData[venue][metricKey],
                    line: {
                        width: 2
                    }
                }));
                
                layout = {
                    title: `${this.capitalizeFirst(this.venueMetric)} Trend Over Time`,
                    margin: { l: 60, r: 20, t: 40, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    xaxis: {
                        title: 'Date',
                        showgrid: false,
                        linecolor: 'var(--border-light)'
                    },
                    yaxis: {
                        title: yAxisTitle,
                        showgrid: true,
                        gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                        linecolor: 'var(--border-light)'
                    },
                    legend: {
                        orientation: 'h',
                        xanchor: 'center',
                        x: 0.5,
                        y: 1.05
                    }
                };
                break;
                
            default:
                // Default to comparison view
                traces = [{
                    type: 'bar',
                    x: venues,
                    y: this.venueMetric === 'fees' ? data.average :
                       this.venueMetric === 'latency' ? data.median :
                       this.venueMetric === 'reliability' ? data.uptime :
                       data.overall,
                    marker: {
                        color: venues.map((v, i) => `rgba(${50 + i * 50}, ${100 + i * 30}, ${150 + i * 30}, 0.7)`)
                    }
                }];
                
                layout = {
                    title: title,
                    margin: { l: 60, r: 20, t: 40, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: {
                        color: 'var(--text)',
                        size: 10
                    },
                    xaxis: {
                        title: 'Exchange',
                        showgrid: false,
                        linecolor: 'var(--border-light)'
                    },
                    yaxis: {
                        title: yAxisTitle,
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
        Plotly.newPlot(this.options.venueAnalyticsElementId, traces, layout, config);
        this.venueAnalyticsChart = document.getElementById(this.options.venueAnalyticsElementId);
    }
    
    renderQualityTrend() {
        const element = document.getElementById(this.options.executionTrendElementId);
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
                range: [90, 98] // Fixed range for better visualization
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false,
            staticPlot: true
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.executionTrendElementId, [trace], layout, config);
        this.executionTrendChart = document.getElementById(this.options.executionTrendElementId);
    }
    
    updateSummaryMetrics() {
        // Update summary metrics in the DOM
        if (!this.summaryMetrics) return;
        
        const metrics = this.summaryMetrics;
        
        // Update metric elements
        this.updateElementText('avg-fill-rate', metrics.avgFillRate);
        this.updateElementText('avg-fill-time', metrics.avgFillTime);
        this.updateElementText('avg-price-impact', metrics.avgPriceImpact);
        this.updateElementText('fulfilled-orders', metrics.fulfilledOrders);
        this.updateElementText('best-algorithm', metrics.bestAlgorithm);
        this.updateElementText('execution-cost', metrics.executionCost);
        
        // Update footer metrics
        this.updateElementText('best-venue', metrics.bestVenue);
        this.updateElementText('best-time-window', metrics.bestTimeWindow);
        this.updateElementText('cost-savings-potential', metrics.costSavingsPotential);
        this.updateElementText('execution-quality-last-updated', metrics.lastUpdated);
    }
    
    updateRecommendations() {
        // Update recommendations in the DOM
        const container = document.getElementById('execution-recommendations');
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
        const strategySelector = document.getElementById('execution-quality-strategy');
        if (strategySelector) {
            strategySelector.addEventListener('change', () => {
                this.strategy = strategySelector.value;
                this.refreshData();
            });
        }
        
        // Asset selector change
        const assetSelector = document.getElementById('execution-quality-asset');
        if (assetSelector) {
            assetSelector.addEventListener('change', () => {
                this.asset = assetSelector.value;
                this.refreshData();
            });
        }
        
        // Time range selector change
        const timeRangeSelector = document.getElementById('execution-quality-timerange');
        if (timeRangeSelector) {
            timeRangeSelector.addEventListener('change', () => {
                this.timeRange = timeRangeSelector.value;
                this.refreshData();
            });
        }
        
        // Order type selector change
        const orderTypeSelector = document.getElementById('execution-quality-type');
        if (orderTypeSelector) {
            orderTypeSelector.addEventListener('change', () => {
                this.orderType = orderTypeSelector.value;
                this.refreshData();
            });
        }
        
        // Fill rate metric selector change
        const fillRateMetricSelector = document.getElementById('fill-rate-metric');
        if (fillRateMetricSelector) {
            fillRateMetricSelector.addEventListener('change', () => {
                this.fillRateMetric = fillRateMetricSelector.value;
                this.renderFillRate();
            });
        }
        
        // Fill rate group selector change
        const fillRateGroupSelector = document.getElementById('fill-rate-group');
        if (fillRateGroupSelector) {
            fillRateGroupSelector.addEventListener('change', () => {
                this.fillRateGroup = fillRateGroupSelector.value;
                this.renderFillRate();
            });
        }
        
        // Market impact view selector change
        const impactViewSelector = document.getElementById('impact-view');
        if (impactViewSelector) {
            impactViewSelector.addEventListener('change', () => {
                this.impactView = impactViewSelector.value;
                this.renderMarketImpact();
            });
        }
        
        // Order size selector change
        const orderSizeSelector = document.getElementById('order-size');
        if (orderSizeSelector) {
            orderSizeSelector.addEventListener('change', () => {
                this.orderSize = orderSizeSelector.value;
                this.renderMarketImpact();
            });
        }
        
        // Algorithm metric selector change
        const algoMetricSelector = document.getElementById('algo-metric');
        if (algoMetricSelector) {
            algoMetricSelector.addEventListener('change', () => {
                this.algoMetric = algoMetricSelector.value;
                this.renderAlgoPerformance();
            });
        }
        
        // Algorithm chart type selector change
        const algoChartTypeSelector = document.getElementById('algo-chart-type');
        if (algoChartTypeSelector) {
            algoChartTypeSelector.addEventListener('change', () => {
                this.algoChartType = algoChartTypeSelector.value;
                this.renderAlgoPerformance();
            });
        }
        
        // Venue metric selector change
        const venueMetricSelector = document.getElementById('venue-metric');
        if (venueMetricSelector) {
            venueMetricSelector.addEventListener('change', () => {
                this.venueMetric = venueMetricSelector.value;
                this.renderVenueAnalytics();
            });
        }
        
        // Venue view selector change
        const venueViewSelector = document.getElementById('venue-view');
        if (venueViewSelector) {
            venueViewSelector.addEventListener('change', () => {
                this.venueView = venueViewSelector.value;
                this.renderVenueAnalytics();
            });
        }
        
        // Settings button
        const settingsBtn = document.getElementById('execution-quality-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('execution-quality-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-execution-quality-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadExecutionQualityData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-execution-quality-panel-btn');
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
        
        // Sliders value display
        document.querySelectorAll('.form-range').forEach(slider => {
            slider.addEventListener('input', function() {
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
            });
        });
        
        // Save settings button
        const saveSettingsBtn = document.getElementById('save-execution-quality-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                
                // Close the modal
                const modal = document.getElementById('execution-quality-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Learn more button
        const learnMoreBtn = document.getElementById('execution-quality-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Execution Quality Analytics documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-execution-quality-report');
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
            this.options.fillRateElementId,
            this.options.marketImpactElementId,
            this.options.algoPerformanceElementId,
            this.options.venueAnalyticsElementId
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
                this.renderFillRate();
                this.renderMarketImpact();
                this.renderAlgoPerformance();
                this.renderVenueAnalytics();
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
    
    downloadExecutionQualityData() {
        // Create downloadable JSON file with execution quality data
        const data = {
            strategy: this.strategy,
            asset: this.asset,
            timeRange: this.timeRange,
            orderType: this.orderType,
            fillRateData: this.fillRateData,
            marketImpactData: this.marketImpactData,
            algoPerformanceData: this.algoPerformanceData,
            venueAnalyticsData: this.venueAnalyticsData,
            summaryMetrics: this.summaryMetrics,
            recommendationData: this.recommendationData
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `execution_quality_analysis_${this.strategy}_${this.asset}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    // Helper methods
    
    getAlgorithmColor(algorithm) {
        // Return consistent color for each algorithm
        switch (algorithm) {
            case 'TWAP':
                return 'rgba(var(--primary-rgb), 0.7)';
            case 'VWAP':
                return 'rgba(var(--success-rgb), 0.7)';
            case 'Iceberg':
                return 'rgba(var(--warning-rgb), 0.7)';
            case 'Smart Order Routing':
                return 'rgba(var(--info-rgb), 0.7)';
            default:
                return 'rgba(var(--text-rgb), 0.7)';
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
        const colorscale = scale || this.options.colorScaleFillRate;
        
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

// Initialize the Execution Quality Analysis component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of ExecutionQualityAnalysis
    const executionQualityAnalysis = new ExecutionQualityAnalysis();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});