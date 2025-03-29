/**
 * Sentiment Spike Analysis
 * 
 * This module provides visualization and analysis of sentiment spikes from
 * social media, news sources, and their correlation with price action.
 */

class SentimentSpikeAnalysis {
    constructor(options = {}) {
        this.options = Object.assign({
            socialMediaChartElementId: 'social-media-sentiment-chart',
            newsChartElementId: 'news-sentiment-chart',
            divergenceChartElementId: 'sentiment-divergence-chart',
            correlationMapElementId: 'sentiment-correlation-map',
            updateInterval: 10000, // milliseconds
            historyLength: 100,    // data points to show
            spikeThreshold: 2.0,   // standard deviations
            colorScalePositive: [
                [0, 'rgba(229, 231, 235, 0.5)'],   // Light gray
                [1, 'rgba(16, 185, 129, 0.7)']     // Green
            ],
            colorScaleNegative: [
                [0, 'rgba(229, 231, 235, 0.5)'],   // Light gray
                [1, 'rgba(239, 68, 68, 0.7)']      // Red
            ],
            colorScaleNeutral: [
                [0, 'rgba(229, 231, 235, 0.5)'],   // Light gray
                [1, 'rgba(59, 130, 246, 0.7)']     // Blue
            ],
            defaultSymbol: 'BTC-USD',
            defaultTimeframe: '1d'
        }, options);

        // State management
        this.symbol = document.getElementById('sentiment-symbol')?.value || this.options.defaultSymbol;
        this.timeframe = document.getElementById('sentiment-timeframe')?.value || this.options.defaultTimeframe;
        this.spikeThreshold = document.getElementById('sentiment-spike-threshold')?.value || this.options.spikeThreshold;
        this.sentimentSource = document.getElementById('sentiment-source')?.value || 'all';
        this.alertsEnabled = document.getElementById('sentiment-alerts-enabled')?.checked || true;

        // Data containers
        this.socialMediaData = {
            twitter: [],
            reddit: [],
            telegram: [],
            discord: []
        };
        this.newsData = {
            mainstream: [],
            crypto: [],
            blogs: []
        };
        this.priceData = [];
        this.spikes = {
            social: [],
            news: []
        };
        this.divergenceEvents = [];
        this.correlationData = [];

        // Chart objects
        this.socialMediaChart = null;
        this.newsChart = null;
        this.divergenceChart = null;
        this.correlationMap = null;

        // Flags
        this.isInitialized = false;
        this.isRealtime = true;
        this.updateTimer = null;

        // Initialize
        this.initialize();
    }

    initialize() {
        // Fetch initial data and set up charts
        this.fetchData()
            .then(() => {
                this.detectSentimentSpikes();
                this.detectDivergenceEvents();
                this.calculateCorrelations();
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
                console.error('Error initializing Sentiment Spike Analysis:', error);
                this.showError('Failed to initialize sentiment analysis');
            });
    }

    fetchData() {
        // In a real implementation, this would fetch from an API
        return Promise.all([
            this.fetchSocialMediaData(),
            this.fetchNewsData(),
            this.fetchPriceData()
        ]);
    }

    fetchSocialMediaData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock social media sentiment data
                this.socialMediaData = {
                    twitter: this.generateMockSentimentData('twitter'),
                    reddit: this.generateMockSentimentData('reddit'),
                    telegram: this.generateMockSentimentData('telegram'),
                    discord: this.generateMockSentimentData('discord')
                };
                
                resolve(this.socialMediaData);
            }, 300);
        });
    }

    fetchNewsData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock news sentiment data
                this.newsData = {
                    mainstream: this.generateMockSentimentData('mainstream', 0.6),
                    crypto: this.generateMockSentimentData('crypto', 0.8),
                    blogs: this.generateMockSentimentData('blogs', 0.7)
                };
                
                resolve(this.newsData);
            }, 250);
        });
    }

    fetchPriceData() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate mock price data
                this.priceData = this.generateMockPriceData();
                resolve(this.priceData);
            }, 200);
        });
    }

    detectSentimentSpikes() {
        // Reset spikes arrays
        this.spikes = {
            social: [],
            news: []
        };

        // Detect spikes in social media sentiment
        Object.entries(this.socialMediaData).forEach(([platform, data]) => {
            const spikes = this.detectSpikes(data, this.spikeThreshold);
            spikes.forEach(spike => {
                spike.platform = platform;
                spike.type = 'social';
            });
            this.spikes.social.push(...spikes);
        });

        // Detect spikes in news sentiment
        Object.entries(this.newsData).forEach(([source, data]) => {
            const spikes = this.detectSpikes(data, this.spikeThreshold);
            spikes.forEach(spike => {
                spike.source = source;
                spike.type = 'news';
            });
            this.spikes.news.push(...spikes);
        });

        // Sort all spikes by timestamp
        this.spikes.social.sort((a, b) => a.timestamp - b.timestamp);
        this.spikes.news.sort((a, b) => a.timestamp - b.timestamp);
    }

    detectSpikes(data, threshold) {
        const spikes = [];
        
        // Need at least some data points for spike detection
        if (data.length < 10) return spikes;
        
        // Calculate rolling mean and standard deviation for sentiment
        const windowSize = 10;
        
        for (let i = windowSize; i < data.length; i++) {
            // Get sentiment window
            const sentimentWindow = data.slice(i - windowSize, i).map(d => d.sentiment);
            
            // Calculate mean and standard deviation
            const mean = sentimentWindow.reduce((sum, val) => sum + val, 0) / windowSize;
            const stdDev = Math.sqrt(
                sentimentWindow.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / windowSize
            );
            
            // Get current sentiment
            const currentSentiment = data[i].sentiment;
            
            // Calculate z-score (how many standard deviations from mean)
            const zScore = stdDev > 0 ? Math.abs((currentSentiment - mean) / stdDev) : 0;
            
            // Check if spike based on threshold
            if (zScore > threshold) {
                // Determine direction (positive or negative spike)
                const direction = currentSentiment > mean ? 'positive' : 'negative';
                
                // Add to spikes
                spikes.push({
                    timestamp: data[i].timestamp,
                    sentiment: currentSentiment,
                    zScore: zScore,
                    direction: direction,
                    baseline: mean,
                    magnitude: Math.abs(currentSentiment - mean) / (Math.abs(mean) || 1) // Percentage change
                });
            }
        }
        
        return spikes;
    }

    detectDivergenceEvents() {
        this.divergenceEvents = [];
        
        // Need both price and sentiment data
        if (this.priceData.length < 10) return;
        
        // Get all spikes (social and news)
        const allSpikes = [...this.spikes.social, ...this.spikes.news];
        
        // Sort by timestamp
        allSpikes.sort((a, b) => a.timestamp - b.timestamp);
        
        // Look for price-sentiment divergence events
        allSpikes.forEach(spike => {
            // Find closest price point
            const pricePoint = this.findClosestPricePoint(spike.timestamp);
            if (!pricePoint) return;
            
            // Calculate price change around the spike (looking ahead)
            const priceChangeWindow = 12; // Look 12 hours ahead
            const futurePricePoint = this.findFuturePricePoint(spike.timestamp, priceChangeWindow);
            
            if (futurePricePoint) {
                const priceChange = (futurePricePoint.price - pricePoint.price) / pricePoint.price;
                
                // Check for divergence (sentiment and price going in opposite directions)
                const sentimentDirection = spike.direction === 'positive' ? 1 : -1;
                const priceDirection = priceChange >= 0 ? 1 : -1;
                
                if (sentimentDirection !== priceDirection && Math.abs(priceChange) > 0.005) {
                    this.divergenceEvents.push({
                        timestamp: spike.timestamp,
                        type: spike.type,
                        platform: spike.platform || spike.source,
                        sentiment: spike.sentiment,
                        sentimentDirection: spike.direction,
                        priceChange: priceChange,
                        initialPrice: pricePoint.price,
                        finalPrice: futurePricePoint.price,
                        divergenceMagnitude: Math.abs(priceChange) * Math.abs(spike.sentiment)
                    });
                }
            }
        });
        
        // Sort divergence events by timestamp
        this.divergenceEvents.sort((a, b) => b.timestamp - a.timestamp);
    }

    calculateCorrelations() {
        this.correlationData = [];
        
        // Define sources for correlation
        const sources = [
            { id: 'twitter', type: 'social', name: 'Twitter' },
            { id: 'reddit', type: 'social', name: 'Reddit' },
            { id: 'telegram', type: 'social', name: 'Telegram' },
            { id: 'discord', type: 'social', name: 'Discord' },
            { id: 'mainstream', type: 'news', name: 'Mainstream Media' },
            { id: 'crypto', type: 'news', name: 'Crypto News' },
            { id: 'blogs', type: 'news', name: 'Blogs' },
            { id: 'price', type: 'market', name: 'Price' }
        ];
        
        // Create correlation matrix (upper triangular)
        for (let i = 0; i < sources.length; i++) {
            for (let j = i + 1; j < sources.length; j++) {
                const source1 = sources[i];
                const source2 = sources[j];
                
                // Get data for correlation
                let data1, data2;
                
                if (source1.type === 'social') {
                    data1 = this.socialMediaData[source1.id];
                } else if (source1.type === 'news') {
                    data1 = this.newsData[source1.id];
                } else if (source1.type === 'market') {
                    data1 = this.priceData;
                }
                
                if (source2.type === 'social') {
                    data2 = this.socialMediaData[source2.id];
                } else if (source2.type === 'news') {
                    data2 = this.newsData[source2.id];
                } else if (source2.type === 'market') {
                    data2 = this.priceData;
                }
                
                // Calculate correlation between the time series
                const correlation = this.calculateTimeSeriesCorrelation(
                    data1, data2, 
                    source1.type === 'market' ? 'price' : 'sentiment',
                    source2.type === 'market' ? 'price' : 'sentiment'
                );
                
                this.correlationData.push({
                    source1: source1.name,
                    source2: source2.name,
                    correlation: correlation,
                    type1: source1.type,
                    type2: source2.type
                });
            }
        }
        
        // Sort by correlation strength (absolute value)
        this.correlationData.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
    }

    calculateTimeSeriesCorrelation(series1, series2, field1 = 'sentiment', field2 = 'sentiment') {
        // Need enough data points
        if (series1.length < 10 || series2.length < 10) return 0;
        
        // Align time series to common timepoints
        const alignedSeries = this.alignTimeSeries(series1, series2, field1, field2);
        
        // Need enough aligned points
        if (alignedSeries.length < 5) return 0;
        
        // Calculate correlation coefficient (Pearson)
        const n = alignedSeries.length;
        
        // Calculate means
        const mean1 = alignedSeries.reduce((sum, point) => sum + point.value1, 0) / n;
        const mean2 = alignedSeries.reduce((sum, point) => sum + point.value2, 0) / n;
        
        // Calculate covariance and variances
        let covariance = 0;
        let variance1 = 0;
        let variance2 = 0;
        
        alignedSeries.forEach(point => {
            const diff1 = point.value1 - mean1;
            const diff2 = point.value2 - mean2;
            
            covariance += diff1 * diff2;
            variance1 += diff1 * diff1;
            variance2 += diff2 * diff2;
        });
        
        // Avoid division by zero
        if (variance1 === 0 || variance2 === 0) return 0;
        
        // Calculate Pearson correlation
        return covariance / (Math.sqrt(variance1) * Math.sqrt(variance2));
    }

    alignTimeSeries(series1, series2, field1 = 'sentiment', field2 = 'sentiment') {
        const alignedSeries = [];
        const tolerance = 15 * 60 * 1000; // 15 minutes tolerance for time matching
        
        // For each point in series1, find closest point in series2
        series1.forEach(point1 => {
            const closestPoint = series2.reduce((closest, point2) => {
                const currentDiff = Math.abs(point1.timestamp - point2.timestamp);
                const closestDiff = Math.abs(point1.timestamp - closest.timestamp);
                return currentDiff < closestDiff ? point2 : closest;
            }, series2[0]);
            
            // Only add if within tolerance
            const timeDiff = Math.abs(point1.timestamp - closestPoint.timestamp);
            if (timeDiff <= tolerance) {
                alignedSeries.push({
                    timestamp: point1.timestamp,
                    value1: point1[field1],
                    value2: closestPoint[field2]
                });
            }
        });
        
        return alignedSeries;
    }

    initializeCharts() {
        this.renderSocialMediaChart();
        this.renderNewsChart();
        this.renderDivergenceChart();
        this.renderCorrelationMap();
        this.updateSourcesTable();
        this.updateSpikesTable();
        this.updateDivergenceTable();
    }

    setupEventListeners() {
        // Symbol selector change
        const symbolSelector = document.getElementById('sentiment-symbol');
        if (symbolSelector) {
            symbolSelector.addEventListener('change', () => {
                this.symbol = symbolSelector.value;
                this.refreshData();
            });
        }

        // Timeframe selector change
        const timeframeSelector = document.getElementById('sentiment-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }

        // Sentiment source selector change
        const sourceSelector = document.getElementById('sentiment-source');
        if (sourceSelector) {
            sourceSelector.addEventListener('change', () => {
                this.sentimentSource = sourceSelector.value;
                this.renderSocialMediaChart();
                this.renderNewsChart();
            });
        }

        // Spike threshold change
        const thresholdSlider = document.getElementById('sentiment-spike-threshold');
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', () => {
                this.spikeThreshold = parseFloat(thresholdSlider.value);
                this.detectSentimentSpikes();
                this.detectDivergenceEvents();
                this.renderSocialMediaChart();
                this.renderNewsChart();
                this.renderDivergenceChart();
                this.updateSpikesTable();
                this.updateDivergenceTable();
                
                // Update the threshold display
                const thresholdDisplay = document.getElementById('threshold-value');
                if (thresholdDisplay) {
                    thresholdDisplay.textContent = this.spikeThreshold.toFixed(1);
                }
            });
        }

        // Alerts toggle
        const alertsToggle = document.getElementById('sentiment-alerts-enabled');
        if (alertsToggle) {
            alertsToggle.addEventListener('change', () => {
                this.alertsEnabled = alertsToggle.checked;
            });
        }

        // Realtime toggle
        const realtimeToggle = document.getElementById('sentiment-realtime-toggle');
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
        const refreshButton = document.getElementById('refresh-sentiment');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshData();
            });
        }

        // Export data button
        const exportButton = document.getElementById('export-sentiment-data');
        if (exportButton) {
            exportButton.addEventListener('click', () => {
                this.exportData();
            });
        }

        // Source info buttons
        document.querySelectorAll('.source-info-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const source = e.target.closest('tr').dataset.source;
                this.showSourceDetails(source);
            });
        });
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
        // Update with new data points
        this.addNewDataPoints();
        
        // Detect spikes and divergences
        this.detectSentimentSpikes();
        this.detectDivergenceEvents();
        this.calculateCorrelations();
        
        // Update visualizations
        this.renderSocialMediaChart();
        this.renderNewsChart();
        this.renderDivergenceChart();
        this.renderCorrelationMap();
        this.updateSourcesTable();
        this.updateSpikesTable();
        this.updateDivergenceTable();
        
        // Check for new spikes to show notifications
        if (this.alertsEnabled) {
            this.checkForNewSpikes();
        }
    }

    refreshData() {
        // Show loading state
        this.showLoading();
        
        // Fetch new data
        this.fetchData()
            .then(() => {
                this.detectSentimentSpikes();
                this.detectDivergenceEvents();
                this.calculateCorrelations();
                this.renderSocialMediaChart();
                this.renderNewsChart();
                this.renderDivergenceChart();
                this.renderCorrelationMap();
                this.updateSourcesTable();
                this.updateSpikesTable();
                this.updateDivergenceTable();
                
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                this.showError('Failed to refresh sentiment data');
            });
    }

    renderSocialMediaChart() {
        const element = document.getElementById(this.options.socialMediaChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Prepare data for the chart
        const traces = [];
        
        // Get platforms based on selected source
        let platforms;
        if (this.sentimentSource === 'all' || this.sentimentSource === 'social') {
            platforms = Object.keys(this.socialMediaData);
        } else if (this.socialMediaData[this.sentimentSource]) {
            platforms = [this.sentimentSource];
        } else {
            platforms = [];
        }
        
        // Create a trace for each platform
        platforms.forEach(platform => {
            const data = this.socialMediaData[platform];
            
            traces.push({
                x: data.map(d => d.timestamp),
                y: data.map(d => d.sentiment),
                type: 'scatter',
                mode: 'lines',
                name: this.capitalizeFirst(platform),
                line: {
                    width: 2
                },
                hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<extra>' + this.capitalizeFirst(platform) + '</extra>'
            });
        });
        
        // Add spike markers
        const relevantSpikes = this.spikes.social.filter(spike => 
            platforms.includes(spike.platform)
        );
        
        if (relevantSpikes.length > 0) {
            // Positive spikes
            const positiveSpikes = relevantSpikes.filter(s => s.direction === 'positive');
            if (positiveSpikes.length > 0) {
                traces.push({
                    x: positiveSpikes.map(s => s.timestamp),
                    y: positiveSpikes.map(s => s.sentiment),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Positive Spikes',
                    marker: {
                        color: 'rgba(16, 185, 129, 0.9)',
                        size: positiveSpikes.map(s => 8 + (s.zScore * 1.5)),
                        line: {
                            color: 'white',
                            width: 1
                        }
                    },
                    hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<br>Z-Score: %{marker.size:.1f}<extra>Positive Spike</extra>'
                });
            }
            
            // Negative spikes
            const negativeSpikes = relevantSpikes.filter(s => s.direction === 'negative');
            if (negativeSpikes.length > 0) {
                traces.push({
                    x: negativeSpikes.map(s => s.timestamp),
                    y: negativeSpikes.map(s => s.sentiment),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Negative Spikes',
                    marker: {
                        color: 'rgba(239, 68, 68, 0.9)',
                        size: negativeSpikes.map(s => 8 + (s.zScore * 1.5)),
                        line: {
                            color: 'white',
                            width: 1
                        }
                    },
                    hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<br>Z-Score: %{marker.size:.1f}<extra>Negative Spike</extra>'
                });
            }
        }
        
        // Layout configuration
        const layout = {
            title: 'Social Media Sentiment Analysis',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Sentiment Score (-1 to 1)',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1,
                range: [-1, 1]
            },
            legend: {
                orientation: 'h',
                y: 1.15,
                x: 0.5,
                xanchor: 'center'
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Spike Threshold: ${this.spikeThreshold.toFixed(1)}σ`,
                    showarrow: false,
                    xanchor: 'right',
                    yanchor: 'top',
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
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
        Plotly.newPlot(this.options.socialMediaChartElementId, traces, layout, config);
        this.socialMediaChart = document.getElementById(this.options.socialMediaChartElementId);
    }

    renderNewsChart() {
        const element = document.getElementById(this.options.newsChartElementId);
        if (!element) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Prepare data for the chart
        const traces = [];
        
        // Get sources based on selected source
        let sources;
        if (this.sentimentSource === 'all' || this.sentimentSource === 'news') {
            sources = Object.keys(this.newsData);
        } else if (this.newsData[this.sentimentSource]) {
            sources = [this.sentimentSource];
        } else {
            sources = [];
        }
        
        // Create a trace for each news source
        sources.forEach(source => {
            const data = this.newsData[source];
            
            traces.push({
                x: data.map(d => d.timestamp),
                y: data.map(d => d.sentiment),
                type: 'scatter',
                mode: 'lines',
                name: this.formatSourceName(source),
                line: {
                    width: 2
                },
                hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<extra>' + this.formatSourceName(source) + '</extra>'
            });
        });
        
        // Add spike markers
        const relevantSpikes = this.spikes.news.filter(spike => 
            sources.includes(spike.source)
        );
        
        if (relevantSpikes.length > 0) {
            // Positive spikes
            const positiveSpikes = relevantSpikes.filter(s => s.direction === 'positive');
            if (positiveSpikes.length > 0) {
                traces.push({
                    x: positiveSpikes.map(s => s.timestamp),
                    y: positiveSpikes.map(s => s.sentiment),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Positive Spikes',
                    marker: {
                        color: 'rgba(16, 185, 129, 0.9)',
                        size: positiveSpikes.map(s => 8 + (s.zScore * 1.5)),
                        line: {
                            color: 'white',
                            width: 1
                        }
                    },
                    hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<br>Z-Score: %{marker.size:.1f}<extra>Positive Spike</extra>'
                });
            }
            
            // Negative spikes
            const negativeSpikes = relevantSpikes.filter(s => s.direction === 'negative');
            if (negativeSpikes.length > 0) {
                traces.push({
                    x: negativeSpikes.map(s => s.timestamp),
                    y: negativeSpikes.map(s => s.sentiment),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Negative Spikes',
                    marker: {
                        color: 'rgba(239, 68, 68, 0.9)',
                        size: negativeSpikes.map(s => 8 + (s.zScore * 1.5)),
                        line: {
                            color: 'white',
                            width: 1
                        }
                    },
                    hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<br>Z-Score: %{marker.size:.1f}<extra>Negative Spike</extra>'
                });
            }
        }
        
        // Layout configuration
        const layout = {
            title: 'News Sentiment Analysis',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Sentiment Score (-1 to 1)',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: true,
                zerolinecolor: 'var(--border-light)',
                zerolinewidth: 1,
                range: [-1, 1]
            },
            legend: {
                orientation: 'h',
                y: 1.15,
                x: 0.5,
                xanchor: 'center'
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Spike Threshold: ${this.spikeThreshold.toFixed(1)}σ`,
                    showarrow: false,
                    xanchor: 'right',
                    yanchor: 'top',
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
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
        Plotly.newPlot(this.options.newsChartElementId, traces, layout, config);
        this.newsChart = document.getElementById(this.options.newsChartElementId);
    }

    renderDivergenceChart() {
        const element = document.getElementById(this.options.divergenceChartElementId);
        if (!element || !this.priceData || this.priceData.length === 0) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Prepare price data
        const priceTrace = {
            x: this.priceData.map(d => d.timestamp),
            y: this.priceData.map(d => d.price),
            type: 'scatter',
            mode: 'lines',
            name: 'Price',
            yaxis: 'y',
            line: {
                color: 'rgba(59, 130, 246, 0.8)',
                width: 2
            },
            hovertemplate: '%{x}<br>Price: %{y:.2f}<extra>Price</extra>'
        };
        
        // Create aggregated sentiment trace
        // We'll weight social media and news equally for a combined sentiment
        const combinedSentiment = this.calculateCombinedSentiment();
        
        const sentimentTrace = {
            x: combinedSentiment.map(d => d.timestamp),
            y: combinedSentiment.map(d => d.sentiment),
            type: 'scatter',
            mode: 'lines',
            name: 'Combined Sentiment',
            yaxis: 'y2',
            line: {
                color: 'rgba(124, 58, 237, 0.8)',
                width: 2,
                dash: 'dot'
            },
            hovertemplate: '%{x}<br>Sentiment: %{y:.2f}<extra>Combined Sentiment</extra>'
        };
        
        // Add divergence markers
        const divergenceTrace = {
            x: this.divergenceEvents.map(d => d.timestamp),
            y: this.divergenceEvents.map(d => {
                // Find price at this timestamp
                const pricePoint = this.findClosestPricePoint(d.timestamp);
                return pricePoint ? pricePoint.price : null;
            }).filter(p => p !== null),
            type: 'scatter',
            mode: 'markers',
            name: 'Divergence Events',
            yaxis: 'y',
            marker: {
                symbol: 'circle',
                color: 'rgba(249, 115, 22, 0.9)',
                size: this.divergenceEvents.map(d => 10 + (d.divergenceMagnitude * 10)),
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '%{x}<br>Price: %{y:.2f}<br>Sentiment: ' + 
                this.divergenceEvents.map(d => d.sentiment.toFixed(2)) + 
                '<br>Divergence: ' + this.divergenceEvents.map(d => d.divergenceMagnitude.toFixed(2)) + 
                '<extra>Divergence</extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Sentiment vs. Price Divergence',
            margin: { l: 60, r: 60, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Time',
                type: 'date',
                showgrid: false,
                linecolor: 'var(--border-light)',
                zeroline: false
            },
            yaxis: {
                title: 'Price',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)',
                zeroline: false,
                side: 'left'
            },
            yaxis2: {
                title: 'Sentiment',
                titlefont: { color: 'rgba(124, 58, 237, 0.8)' },
                tickfont: { color: 'rgba(124, 58, 237, 0.8)' },
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                range: [-1, 1],
                zeroline: true,
                zerolinecolor: 'rgba(124, 58, 237, 0.3)',
                zerolinewidth: 1
            },
            legend: {
                orientation: 'h',
                y: 1.15,
                x: 0.5,
                xanchor: 'center'
            },
            annotations: [
                {
                    x: 1,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Divergence Events: ${this.divergenceEvents.length}`,
                    showarrow: false,
                    xanchor: 'right',
                    yanchor: 'top',
                    font: {
                        size: 10,
                        color: 'var(--text-light)'
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
        Plotly.newPlot(this.options.divergenceChartElementId, [priceTrace, sentimentTrace, divergenceTrace], layout, config);
        this.divergenceChart = document.getElementById(this.options.divergenceChartElementId);
    }

    renderCorrelationMap() {
        const element = document.getElementById(this.options.correlationMapElementId);
        if (!element || !this.correlationData || this.correlationData.length === 0) return;

        // Clear any loading overlay
        this.hideLoading(element);

        // Extract nodes and edges from correlation data
        const nodes = new Set();
        this.correlationData.forEach(item => {
            nodes.add(item.source1);
            nodes.add(item.source2);
        });
        
        // Create node data
        const nodeData = Array.from(nodes).map(name => {
            let group;
            if (name.includes('Twitter') || name.includes('Reddit') || 
                name.includes('Telegram') || name.includes('Discord')) {
                group = 'social';
            } else if (name.includes('Media') || name.includes('News') || name.includes('Blogs')) {
                group = 'news';
            } else {
                group = 'market';
            }
            
            return { id: name, group: group };
        });
        
        // Create edge data
        const edgeData = this.correlationData.map(item => ({
            from: item.source1,
            to: item.source2,
            value: Math.abs(item.correlation),
            title: `${item.source1} ↔ ${item.source2}: ${(item.correlation).toFixed(2)}`,
            color: item.correlation > 0 ? 
                `rgba(16, 185, 129, ${Math.abs(item.correlation)})` : 
                `rgba(239, 68, 68, ${Math.abs(item.correlation)})`
        }));
        
        // Filter edges by correlation strength
        const filteredEdges = edgeData.filter(edge => Math.abs(edge.value) >= 0.3);
        
        // Create network graph using vis.js if available
        if (typeof vis !== 'undefined') {
            // Create nodes and edges
            const nodes = new vis.DataSet(nodeData);
            const edges = new vis.DataSet(filteredEdges);
            
            // Create network
            const container = element;
            const data = { nodes, edges };
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 16,
                    font: {
                        size: 12,
                        color: 'var(--text)'
                    },
                    borderWidth: 2,
                    shadow: true,
                    color: {
                        border: 'var(--border)',
                        background: 'var(--bg-light)'
                    }
                },
                edges: {
                    width: 2,
                    selectionWidth: 4,
                    smooth: {
                        type: 'continuous'
                    }
                },
                physics: {
                    stabilization: true,
                    barnesHut: {
                        gravitationalConstant: -80,
                        springLength: 100,
                        springConstant: 0.03
                    }
                },
                groups: {
                    social: {
                        color: { background: 'rgba(59, 130, 246, 0.7)', border: 'rgba(59, 130, 246, 1)' },
                        shape: 'dot'
                    },
                    news: {
                        color: { background: 'rgba(16, 185, 129, 0.7)', border: 'rgba(16, 185, 129, 1)' },
                        shape: 'diamond'
                    },
                    market: {
                        color: { background: 'rgba(249, 115, 22, 0.7)', border: 'rgba(249, 115, 22, 1)' },
                        shape: 'square'
                    }
                }
            };
            
            // Initialize network
            new vis.Network(container, data, options);
            
        } else {
            // Fallback to a heatmap if vis.js is not available
            const sources = Array.from(nodes);
            
            // Create correlation matrix
            const correlationMatrix = [];
            sources.forEach(source1 => {
                const row = [];
                sources.forEach(source2 => {
                    if (source1 === source2) {
                        row.push(1); // Self-correlation is always 1
                    } else {
                        // Find correlation between these sources
                        const correlation = this.correlationData.find(
                            c => (c.source1 === source1 && c.source2 === source2) ||
                                 (c.source1 === source2 && c.source2 === source1)
                        );
                        
                        row.push(correlation ? correlation.correlation : 0);
                    }
                });
                correlationMatrix.push(row);
            });
            
            // Create heatmap trace
            const trace = {
                z: correlationMatrix,
                x: sources,
                y: sources,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgba(239, 68, 68, 1)'],      // Strong negative (red)
                    [0.4, 'rgba(229, 231, 235, 0.5)'], // Weak correlation (light gray)
                    [0.6, 'rgba(229, 231, 235, 0.5)'], // Weak correlation (light gray)
                    [1, 'rgba(16, 185, 129, 1)']      // Strong positive (green)
                ],
                zmin: -1,
                zmax: 1,
                hovertemplate: '%{y} ↔ %{x}<br>Correlation: %{z:.2f}<extra></extra>'
            };
            
            // Layout configuration
            const layout = {
                title: 'Cross-Source Sentiment Correlation',
                margin: { l: 120, r: 20, t: 40, b: 120 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    tickangle: 45
                }
            };
            
            // Configuration options
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            // Render with Plotly
            Plotly.newPlot(this.options.correlationMapElementId, [trace], layout, config);
            this.correlationMap = document.getElementById(this.options.correlationMapElementId);
        }
    }

    updateSourcesTable() {
        const tableBody = document.getElementById('sentiment-sources-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Calculate metrics for each source
        const sourceMetrics = this.calculateSourceMetrics();
        
        // Sort by volume
        sourceMetrics.sort((a, b) => b.volume - a.volume);
        
        // Add rows for each source
        sourceMetrics.forEach(source => {
            const row = document.createElement('tr');
            row.dataset.source = source.id;
            
            // Source name
            const nameCell = document.createElement('td');
            nameCell.textContent = source.name;
            
            // Volume
            const volumeCell = document.createElement('td');
            volumeCell.textContent = source.volume.toLocaleString();
            
            // Average sentiment
            const sentimentCell = document.createElement('td');
            sentimentCell.textContent = source.avgSentiment.toFixed(2);
            sentimentCell.className = source.avgSentiment > 0.1 ? 'positive' : 
                                    source.avgSentiment < -0.1 ? 'negative' : 'neutral';
            
            // Spike frequency
            const spikeCell = document.createElement('td');
            spikeCell.textContent = source.spikeFrequency.toFixed(2) + '/day';
            
            // Trend
            const trendCell = document.createElement('td');
            const trend = source.trend;
            trendCell.innerHTML = trend > 0.1 ? 
                '<span class="trend-up">↑</span>' : 
                trend < -0.1 ? 
                '<span class="trend-down">↓</span>' : 
                '<span class="trend-neutral">→</span>';
            
            // Price correlation
            const corrCell = document.createElement('td');
            corrCell.textContent = source.priceCorrelation.toFixed(2);
            corrCell.className = source.priceCorrelation > 0.3 ? 'positive' : 
                               source.priceCorrelation < -0.3 ? 'negative' : 'neutral';
            
            // Info button
            const infoCell = document.createElement('td');
            infoCell.innerHTML = '<button class="btn btn-icon source-info-btn"><i data-feather="info"></i></button>';
            
            // Add all cells to the row
            row.appendChild(nameCell);
            row.appendChild(volumeCell);
            row.appendChild(sentimentCell);
            row.appendChild(spikeCell);
            row.appendChild(trendCell);
            row.appendChild(corrCell);
            row.appendChild(infoCell);
            
            // Add row to the table
            tableBody.appendChild(row);
        });
        
        // Reinitialize feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    updateSpikesTable() {
        const tableBody = document.getElementById('sentiment-spikes-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Combine social and news spikes
        const allSpikes = [...this.spikes.social, ...this.spikes.news];
        
        // Sort by timestamp (most recent first)
        allSpikes.sort((a, b) => b.timestamp - a.timestamp);
        
        // Take the most recent 10 spikes
        const recentSpikes = allSpikes.slice(0, 10);
        
        // Add rows for each spike
        recentSpikes.forEach(spike => {
            const row = document.createElement('tr');
            
            // Timestamp
            const timeCell = document.createElement('td');
            timeCell.textContent = new Date(spike.timestamp).toLocaleString();
            
            // Source
            const sourceCell = document.createElement('td');
            sourceCell.textContent = this.formatSourceName(spike.platform || spike.source);
            
            // Type
            const typeCell = document.createElement('td');
            typeCell.textContent = spike.type === 'social' ? 'Social Media' : 'News';
            
            // Direction
            const directionCell = document.createElement('td');
            const direction = spike.direction === 'positive' ? 
                '<span class="positive">Positive</span>' : 
                '<span class="negative">Negative</span>';
            directionCell.innerHTML = direction;
            
            // Z-Score
            const zScoreCell = document.createElement('td');
            zScoreCell.textContent = spike.zScore.toFixed(2);
            
            // Magnitude
            const magnitudeCell = document.createElement('td');
            magnitudeCell.textContent = (spike.magnitude * 100).toFixed(1) + '%';
            
            // Add all cells to the row
            row.appendChild(timeCell);
            row.appendChild(sourceCell);
            row.appendChild(typeCell);
            row.appendChild(directionCell);
            row.appendChild(zScoreCell);
            row.appendChild(magnitudeCell);
            
            // Add row to the table
            tableBody.appendChild(row);
        });
    }

    updateDivergenceTable() {
        const tableBody = document.getElementById('sentiment-divergence-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Take the most recent 5 divergence events
        const recentDivergences = this.divergenceEvents.slice(0, 5);
        
        // Add rows for each divergence event
        recentDivergences.forEach(event => {
            const row = document.createElement('tr');
            
            // Timestamp
            const timeCell = document.createElement('td');
            timeCell.textContent = new Date(event.timestamp).toLocaleString();
            
            // Source
            const sourceCell = document.createElement('td');
            sourceCell.textContent = this.formatSourceName(event.platform);
            
            // Sentiment
            const sentimentCell = document.createElement('td');
            sentimentCell.textContent = event.sentiment.toFixed(2);
            sentimentCell.className = event.sentimentDirection === 'positive' ? 'positive' : 'negative';
            
            // Price change
            const priceChangeCell = document.createElement('td');
            priceChangeCell.textContent = (event.priceChange * 100).toFixed(2) + '%';
            priceChangeCell.className = event.priceChange > 0 ? 'positive' : 'negative';
            
            // Price values
            const priceValuesCell = document.createElement('td');
            priceValuesCell.textContent = `${event.initialPrice.toFixed(2)} → ${event.finalPrice.toFixed(2)}`;
            
            // Divergence magnitude
            const magnitudeCell = document.createElement('td');
            magnitudeCell.textContent = event.divergenceMagnitude.toFixed(3);
            
            // Add all cells to the row
            row.appendChild(timeCell);
            row.appendChild(sourceCell);
            row.appendChild(sentimentCell);
            row.appendChild(priceChangeCell);
            row.appendChild(priceValuesCell);
            row.appendChild(magnitudeCell);
            
            // Add row to the table
            tableBody.appendChild(row);
        });
    }

    showSourceDetails(sourceId) {
        alert(`Source details for ${sourceId} will be implemented in the next phase.`);
    }

    // Data generation and calculation functions
    generateMockSentimentData(source, volatility = 1.0) {
        // Generate realistic sentiment data for the given source
        const dataPoints = 200;
        const data = [];
        
        // Determine base sentiment bias
        let bias;
        switch (source) {
            case 'twitter':
                bias = 0.1;  // Slightly positive
                break;
            case 'reddit':
                bias = -0.05; // Slightly negative
                break;
            case 'telegram':
                bias = 0.2;  // More positive
                break;
            case 'discord':
                bias = 0;    // Neutral
                break;
            case 'mainstream':
                bias = -0.1; // Slightly negative
                break;
            case 'crypto':
                bias = 0.15; // Positive
                break;
            case 'blogs':
                bias = 0.05; // Slightly positive
                break;
            default:
                bias = 0;
        }
        
        // Generate data points
        const now = new Date();
        let sentiment = bias + (Math.random() * 0.2 - 0.1);
        
        for (let i = 0; i < dataPoints; i++) {
            // Random walk for sentiment
            const randomChange = (Math.random() * 0.1 - 0.05) * volatility;
            sentiment += randomChange;
            
            // Add trend based on time
            const trend = Math.sin(i / 40) * 0.1;
            sentiment += trend;
            
            // Keep sentiment within bounds
            sentiment = Math.max(-1, Math.min(1, sentiment));
            
            // Add occasional sentiment spikes
            if (Math.random() < 0.02) {
                const spikeDirection = Math.random() > 0.5 ? 1 : -1;
                const spikeMagnitude = (0.2 + Math.random() * 0.4) * volatility;
                sentiment += spikeDirection * spikeMagnitude;
                sentiment = Math.max(-1, Math.min(1, sentiment)); // Ensure bounds
            }
            
            // Add mean reversion
            sentiment = sentiment * 0.95 + bias * 0.05;
            
            // Calculate timestamp (going backward from now)
            const timestamp = new Date(now.getTime() - ((dataPoints - i) * 15 * 60000)); // 15 minute intervals
            
            // Add to data array with volume
            data.push({
                timestamp: timestamp,
                sentiment: sentiment,
                volume: this.getSourceVolume(source, i)
            });
        }
        
        return data;
    }

    generateMockPriceData() {
        // Generate realistic price data
        const dataPoints = 200;
        const data = [];
        
        // Base price varies by symbol
        let basePrice;
        switch (this.symbol) {
            case 'BTC-USD':
                basePrice = 30000;
                break;
            case 'ETH-USD':
                basePrice = 2000;
                break;
            case 'SOL-USD':
                basePrice = 100;
                break;
            case 'BNB-USD':
                basePrice = 300;
                break;
            default:
                basePrice = 1000;
        }
        
        // Start at the base price
        let price = basePrice;
        
        // Generate data points
        const now = new Date();
        for (let i = 0; i < dataPoints; i++) {
            // Basic random walk
            const randomMove = (Math.random() - 0.5) * (basePrice * 0.005);
            
            // Add some trends
            const trend = Math.sin(i / 20) * (basePrice * 0.01);
            
            // Add occasional jumps
            let jump = 0;
            if (Math.random() < 0.01) {
                jump = (Math.random() > 0.5 ? 1 : -1) * (basePrice * 0.01);
            }
            
            // Update price
            price = price + randomMove + trend + jump;
            
            // Ensure price doesn't go negative
            price = Math.max(price, basePrice * 0.5);
            
            // Calculate timestamp (going backward from now)
            const timestamp = new Date(now.getTime() - ((dataPoints - i) * 15 * 60000)); // 15 minute intervals
            
            // Add to data array
            data.push({
                timestamp: timestamp,
                price: price
            });
        }
        
        return data;
    }

    getSourceVolume(source, timepoint) {
        // Generate realistic volume data for the given source
        let baseVolume;
        
        // Base volume depends on the source
        switch (source) {
            case 'twitter':
                baseVolume = 5000 + Math.random() * 2000;
                break;
            case 'reddit':
                baseVolume = 3000 + Math.random() * 1500;
                break;
            case 'telegram':
                baseVolume = 1000 + Math.random() * 800;
                break;
            case 'discord':
                baseVolume = 800 + Math.random() * 600;
                break;
            case 'mainstream':
                baseVolume = 500 + Math.random() * 300;
                break;
            case 'crypto':
                baseVolume = 1200 + Math.random() * 600;
                break;
            case 'blogs':
                baseVolume = 300 + Math.random() * 200;
                break;
            default:
                baseVolume = 1000 + Math.random() * 500;
        }
        
        // Add daily cycle
        const cycle = Math.sin(timepoint / 24 * Math.PI) * 0.3 + 0.7;
        
        // Add randomness
        const randomFactor = 0.8 + Math.random() * 0.4;
        
        return Math.round(baseVolume * cycle * randomFactor);
    }

    calculateCombinedSentiment() {
        // Calculate a combined sentiment score from all sources
        const combinedData = [];
        
        // Get timestamps from price data (as reference)
        const priceTimestamps = this.priceData.map(d => d.timestamp);
        
        // For each timestamp, calculate weighted average of sentiment from all sources
        priceTimestamps.forEach(timestamp => {
            // Find closest sentiment readings from each source
            let totalSentiment = 0;
            let totalWeight = 0;
            
            // Process social media sources
            Object.values(this.socialMediaData).forEach(data => {
                const closest = this.findClosestSentimentPoint(data, timestamp);
                if (closest) {
                    totalSentiment += closest.sentiment * 0.5; // Weight of 0.5 for social
                    totalWeight += 0.5;
                }
            });
            
            // Process news sources
            Object.values(this.newsData).forEach(data => {
                const closest = this.findClosestSentimentPoint(data, timestamp);
                if (closest) {
                    totalSentiment += closest.sentiment * 0.5; // Weight of 0.5 for news
                    totalWeight += 0.5;
                }
            });
            
            // Calculate weighted average
            const avgSentiment = totalWeight > 0 ? totalSentiment / totalWeight : 0;
            
            // Add to combined data
            combinedData.push({
                timestamp: timestamp,
                sentiment: avgSentiment
            });
        });
        
        return combinedData;
    }

    calculateSourceMetrics() {
        const metrics = [];
        
        // Process social media sources
        Object.entries(this.socialMediaData).forEach(([platform, data]) => {
            metrics.push(this.calculateMetricsForSource(platform, data, 'social'));
        });
        
        // Process news sources
        Object.entries(this.newsData).forEach(([source, data]) => {
            metrics.push(this.calculateMetricsForSource(source, data, 'news'));
        });
        
        return metrics;
    }

    calculateMetricsForSource(sourceId, data, type) {
        // Skip if not enough data
        if (data.length < 10) {
            return {
                id: sourceId,
                name: this.formatSourceName(sourceId),
                type: type,
                volume: 0,
                avgSentiment: 0,
                spikeFrequency: 0,
                trend: 0,
                priceCorrelation: 0
            };
        }
        
        // Calculate total volume
        const volume = data.reduce((sum, point) => sum + (point.volume || 0), 0);
        
        // Calculate average sentiment
        const avgSentiment = data.reduce((sum, point) => sum + point.sentiment, 0) / data.length;
        
        // Calculate spike frequency (spikes per day)
        let spikeCount = 0;
        if (type === 'social') {
            spikeCount = this.spikes.social.filter(spike => spike.platform === sourceId).length;
        } else {
            spikeCount = this.spikes.news.filter(spike => spike.source === sourceId).length;
        }
        
        // Assuming data covers about 2 days worth
        const spikeFrequency = spikeCount / 2;
        
        // Calculate trend (recent sentiment vs older sentiment)
        const midpoint = Math.floor(data.length / 2);
        const recentData = data.slice(midpoint);
        const olderData = data.slice(0, midpoint);
        
        const recentAvg = recentData.reduce((sum, point) => sum + point.sentiment, 0) / recentData.length;
        const olderAvg = olderData.reduce((sum, point) => sum + point.sentiment, 0) / olderData.length;
        
        const trend = recentAvg - olderAvg;
        
        // Calculate price correlation
        let priceCorrelation = 0;
        if (this.priceData && this.priceData.length > 0) {
            priceCorrelation = this.calculateTimeSeriesCorrelation(
                data, this.priceData, 
                'sentiment', 'price'
            );
        }
        
        return {
            id: sourceId,
            name: this.formatSourceName(sourceId),
            type: type,
            volume: volume,
            avgSentiment: avgSentiment,
            spikeFrequency: spikeFrequency,
            trend: trend,
            priceCorrelation: priceCorrelation
        };
    }

    addNewDataPoints() {
        // Add a new data point to each data series
        const now = new Date();
        
        // Update social media data
        Object.keys(this.socialMediaData).forEach(platform => {
            const data = this.socialMediaData[platform];
            if (data.length === 0) return;
            
            // Get the last point
            const lastPoint = data[data.length - 1];
            
            // Random change in sentiment
            let newSentiment = lastPoint.sentiment + (Math.random() * 0.1 - 0.05);
            
            // Keep in bounds
            newSentiment = Math.max(-1, Math.min(1, newSentiment));
            
            // Random volume
            const newVolume = this.getSourceVolume(platform, data.length);
            
            // Add the new point
            this.socialMediaData[platform].push({
                timestamp: now,
                sentiment: newSentiment,
                volume: newVolume
            });
            
            // Remove the oldest point if needed
            if (this.socialMediaData[platform].length > this.options.historyLength) {
                this.socialMediaData[platform].shift();
            }
        });
        
        // Update news data
        Object.keys(this.newsData).forEach(source => {
            const data = this.newsData[source];
            if (data.length === 0) return;
            
            // Get the last point
            const lastPoint = data[data.length - 1];
            
            // Random change in sentiment
            let newSentiment = lastPoint.sentiment + (Math.random() * 0.08 - 0.04);
            
            // Keep in bounds
            newSentiment = Math.max(-1, Math.min(1, newSentiment));
            
            // Random volume
            const newVolume = this.getSourceVolume(source, data.length);
            
            // Add the new point
            this.newsData[source].push({
                timestamp: now,
                sentiment: newSentiment,
                volume: newVolume
            });
            
            // Remove the oldest point if needed
            if (this.newsData[source].length > this.options.historyLength) {
                this.newsData[source].shift();
            }
        });
        
        // Update price data
        if (this.priceData.length > 0) {
            const lastPoint = this.priceData[this.priceData.length - 1];
            const randomChange = (Math.random() - 0.5) * (lastPoint.price * 0.003);
            const newPrice = lastPoint.price + randomChange;
            
            this.priceData.push({
                timestamp: now,
                price: newPrice
            });
            
            // Remove the oldest point if needed
            if (this.priceData.length > this.options.historyLength) {
                this.priceData.shift();
            }
        }
    }

    findClosestPricePoint(timestamp) {
        if (!this.priceData || this.priceData.length === 0) return null;
        
        return this.priceData.reduce((closest, current) => {
            const currentDiff = Math.abs(timestamp - current.timestamp);
            const closestDiff = Math.abs(timestamp - closest.timestamp);
            return currentDiff < closestDiff ? current : closest;
        }, this.priceData[0]);
    }

    findFuturePricePoint(timestamp, hoursAhead) {
        if (!this.priceData || this.priceData.length === 0) return null;
        
        // Calculate target timestamp
        const targetTime = new Date(timestamp.getTime() + (hoursAhead * 60 * 60 * 1000));
        
        // Find the closest future point
        const futurePoints = this.priceData.filter(p => p.timestamp >= targetTime);
        if (futurePoints.length === 0) return null;
        
        return futurePoints.reduce((closest, current) => {
            const currentDiff = Math.abs(targetTime - current.timestamp);
            const closestDiff = Math.abs(targetTime - closest.timestamp);
            return currentDiff < closestDiff ? current : closest;
        }, futurePoints[0]);
    }

    findClosestSentimentPoint(data, timestamp) {
        if (!data || data.length === 0) return null;
        
        return data.reduce((closest, current) => {
            const currentDiff = Math.abs(timestamp - current.timestamp);
            const closestDiff = Math.abs(timestamp - closest.timestamp);
            return currentDiff < closestDiff ? current : closest;
        }, data[0]);
    }

    checkForNewSpikes() {
        // Get all spikes
        const allSpikes = [
            ...this.spikes.social,
            ...this.spikes.news
        ];
        
        // Check for spikes in the last minute
        const now = new Date();
        const recentSpikes = allSpikes.filter(spike => 
            (now - spike.timestamp) < 60000 // Within the last minute
        );
        
        // Show notifications for recent spikes
        recentSpikes.forEach(spike => {
            this.showSpikeNotification(spike);
        });
        
        // Check for divergence events in the last minute
        const recentDivergences = this.divergenceEvents.filter(event => 
            (now - event.timestamp) < 60000 // Within the last minute
        );
        
        // Show notifications for recent divergences
        recentDivergences.forEach(event => {
            this.showDivergenceNotification(event);
        });
    }

    showSpikeNotification(spike) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `sentiment-notification ${spike.direction}`;
        
        // Create notification content
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">${this.capitalizeFirst(spike.direction)} Sentiment Spike</span>
                <span class="notification-close">&times;</span>
            </div>
            <div class="notification-body">
                <div class="notification-source">${this.formatSourceName(spike.platform || spike.source)}</div>
                <div class="notification-info">
                    Sentiment: ${spike.sentiment.toFixed(2)} | Z-Score: ${spike.zScore.toFixed(1)}
                </div>
                <div class="notification-time">${new Date(spike.timestamp).toLocaleTimeString()}</div>
            </div>
        `;
        
        // Add to notifications area
        const notificationsArea = document.querySelector('.sentiment-notifications');
        if (notificationsArea) {
            notificationsArea.appendChild(notification);
            
            // Add close handler
            const closeButton = notification.querySelector('.notification-close');
            if (closeButton) {
                closeButton.addEventListener('click', () => {
                    notification.remove();
                });
            }
            
            // Auto-remove after 10 seconds
            setTimeout(() => {
                notification.classList.add('fade-out');
                setTimeout(() => {
                    notification.remove();
                }, 500);
            }, 10000);
        }
    }

    showDivergenceNotification(event) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'sentiment-notification divergence';
        
        // Create notification content
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">Sentiment-Price Divergence</span>
                <span class="notification-close">&times;</span>
            </div>
            <div class="notification-body">
                <div class="notification-source">${this.formatSourceName(event.platform)}</div>
                <div class="notification-info">
                    Sentiment: ${event.sentiment.toFixed(2)} (${event.sentimentDirection})
                </div>
                <div class="notification-info">
                    Price: ${(event.priceChange * 100).toFixed(2)}%
                </div>
                <div class="notification-time">${new Date(event.timestamp).toLocaleTimeString()}</div>
            </div>
        `;
        
        // Add to notifications area
        const notificationsArea = document.querySelector('.sentiment-notifications');
        if (notificationsArea) {
            notificationsArea.appendChild(notification);
            
            // Add close handler
            const closeButton = notification.querySelector('.notification-close');
            if (closeButton) {
                closeButton.addEventListener('click', () => {
                    notification.remove();
                });
            }
            
            // Auto-remove after 10 seconds
            setTimeout(() => {
                notification.classList.add('fade-out');
                setTimeout(() => {
                    notification.remove();
                }, 500);
            }, 10000);
        }
    }

    // Helper functions
    formatSourceName(source) {
        if (!source) return 'Unknown';
        
        // Special formatting for some sources
        switch (source) {
            case 'twitter': return 'Twitter';
            case 'reddit': return 'Reddit';
            case 'telegram': return 'Telegram';
            case 'discord': return 'Discord';
            case 'mainstream': return 'Mainstream Media';
            case 'crypto': return 'Crypto News';
            case 'blogs': return 'Blogs';
            default: return this.capitalizeFirst(source);
        }
    }

    capitalizeFirst(string) {
        if (!string) return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    showLoading(element) {
        // Show loading overlay on charts
        const elements = element ? [element] : [
            document.getElementById(this.options.socialMediaChartElementId),
            document.getElementById(this.options.newsChartElementId),
            document.getElementById(this.options.divergenceChartElementId),
            document.getElementById(this.options.correlationMapElementId)
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
        // Hide loading overlay
        const elements = element ? [element] : [
            document.getElementById(this.options.socialMediaChartElementId),
            document.getElementById(this.options.newsChartElementId),
            document.getElementById(this.options.divergenceChartElementId),
            document.getElementById(this.options.correlationMapElementId)
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
        // Show error message on charts
        const elements = element ? [element] : [
            document.getElementById(this.options.socialMediaChartElementId),
            document.getElementById(this.options.newsChartElementId),
            document.getElementById(this.options.divergenceChartElementId),
            document.getElementById(this.options.correlationMapElementId)
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
            timeframe: this.timeframe,
            timestamp: new Date(),
            socialMediaData: this.socialMediaData,
            newsData: this.newsData,
            priceData: this.priceData,
            spikes: this.spikes,
            divergenceEvents: this.divergenceEvents,
            correlationData: this.correlationData
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `sentiment_analysis_${this.symbol}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of SentimentSpikeAnalysis
    const sentimentAnalysis = new SentimentSpikeAnalysis();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});