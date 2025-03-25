/**
 * Cross-Asset Correlation Networks
 * 
 * This module provides functionality for the Cross-Asset Correlation Networks component
 * including interactive network graphs, correlation matrices, correlation change analysis,
 * and statistical insights about asset relationships.
 */

class CrossAssetCorrelation {
    constructor(options = {}) {
        this.options = Object.assign({
            networkElementId: 'correlation-network',
            matrixElementId: 'correlation-matrix',
            changesElementId: 'correlation-changes',
            analysisElementId: 'correlation-analysis',
            colorScale: [
                [0, 'rgba(239, 68, 68, 0.7)'],     // Negative correlation (red)
                [0.5, 'rgba(209, 213, 219, 0.7)'],  // No correlation (gray)
                [1, 'rgba(16, 185, 129, 0.7)']      // Positive correlation (green)
            ],
            thresholdValue: 0.6,
            minNodeSize: 10,
            maxNodeSize: 25
        }, options);
        
        this.assetClass = document.getElementById('asset-class')?.value || 'crypto';
        this.timeframe = document.getElementById('correlation-timeframe')?.value || '1d';
        this.period = document.getElementById('correlation-period')?.value || '90';
        this.metric = document.getElementById('correlation-metric')?.value || 'pearson';
        
        this.viewType = 'force';
        this.sortType = 'cluster';
        this.currentPair = 'BTC-ETH';
        this.analysisTab = 'stats';
        
        this.assets = [];
        this.correlationMatrix = [];
        this.timeSeriesData = [];
        this.clusterData = [];
        
        this.networkPlot = null;
        this.matrixPlot = null;
        this.changesPlot = null;
        this.analysisPlot = null;
        
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
                console.error('Error initializing Cross-Asset Correlation:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchAssets(),
            this.fetchCorrelationMatrix(),
            this.fetchTimeSeriesData(),
            this.fetchClusterData()
        ]);
    }
    
    fetchAssets() {
        // Mock data for assets based on asset class
        return new Promise(resolve => {
            setTimeout(() => {
                let assetsList = [];
                
                switch(this.assetClass) {
                    case 'crypto':
                        assetsList = [
                            { id: 'BTC', name: 'Bitcoin', marketCap: 1000 },
                            { id: 'ETH', name: 'Ethereum', marketCap: 500 },
                            { id: 'SOL', name: 'Solana', marketCap: 200 },
                            { id: 'BNB', name: 'Binance Coin', marketCap: 180 },
                            { id: 'XRP', name: 'Ripple', marketCap: 150 },
                            { id: 'ADA', name: 'Cardano', marketCap: 120 },
                            { id: 'DOGE', name: 'Dogecoin', marketCap: 100 },
                            { id: 'DOT', name: 'Polkadot', marketCap: 90 },
                            { id: 'AVAX', name: 'Avalanche', marketCap: 85 },
                            { id: 'LINK', name: 'Chainlink', marketCap: 80 }
                        ];
                        break;
                    case 'stock':
                        assetsList = [
                            { id: 'SPX', name: 'S&P 500', marketCap: 1000 },
                            { id: 'NDX', name: 'Nasdaq 100', marketCap: 900 },
                            { id: 'DJI', name: 'Dow Jones', marketCap: 850 },
                            { id: 'RUT', name: 'Russell 2000', marketCap: 700 },
                            { id: 'FTSE', name: 'FTSE 100', marketCap: 650 },
                            { id: 'DAX', name: 'DAX', marketCap: 600 },
                            { id: 'N225', name: 'Nikkei 225', marketCap: 550 },
                            { id: 'HSI', name: 'Hang Seng', marketCap: 500 },
                            { id: 'SSEC', name: 'Shanghai Composite', marketCap: 450 },
                            { id: 'BOVESPA', name: 'Bovespa', marketCap: 400 }
                        ];
                        break;
                    case 'forex':
                        assetsList = [
                            { id: 'EURUSD', name: 'EUR/USD', marketCap: 1000 },
                            { id: 'USDJPY', name: 'USD/JPY', marketCap: 900 },
                            { id: 'GBPUSD', name: 'GBP/USD', marketCap: 800 },
                            { id: 'AUDUSD', name: 'AUD/USD', marketCap: 700 },
                            { id: 'USDCAD', name: 'USD/CAD', marketCap: 650 },
                            { id: 'USDCHF', name: 'USD/CHF', marketCap: 600 },
                            { id: 'EURGBP', name: 'EUR/GBP', marketCap: 550 },
                            { id: 'USDRUB', name: 'USD/RUB', marketCap: 500 },
                            { id: 'USDCNY', name: 'USD/CNY', marketCap: 450 },
                            { id: 'USDINR', name: 'USD/INR', marketCap: 400 }
                        ];
                        break;
                    case 'commodity':
                        assetsList = [
                            { id: 'GOLD', name: 'Gold', marketCap: 1000 },
                            { id: 'SILVER', name: 'Silver', marketCap: 800 },
                            { id: 'CRUDE', name: 'Crude Oil', marketCap: 900 },
                            { id: 'BRENT', name: 'Brent Oil', marketCap: 850 },
                            { id: 'NATGAS', name: 'Natural Gas', marketCap: 700 },
                            { id: 'COPPER', name: 'Copper', marketCap: 650 },
                            { id: 'PLATINUM', name: 'Platinum', marketCap: 600 },
                            { id: 'ALUMINUM', name: 'Aluminum', marketCap: 550 },
                            { id: 'WHEAT', name: 'Wheat', marketCap: 500 },
                            { id: 'CORN', name: 'Corn', marketCap: 450 }
                        ];
                        break;
                    case 'mixed':
                        assetsList = [
                            { id: 'BTC', name: 'Bitcoin', marketCap: 1000 },
                            { id: 'ETH', name: 'Ethereum', marketCap: 500 },
                            { id: 'SPX', name: 'S&P 500', marketCap: 1000 },
                            { id: 'NDX', name: 'Nasdaq 100', marketCap: 900 },
                            { id: 'GOLD', name: 'Gold', marketCap: 800 },
                            { id: 'CRUDE', name: 'Crude Oil', marketCap: 750 },
                            { id: 'EURUSD', name: 'EUR/USD', marketCap: 700 },
                            { id: 'USDJPY', name: 'USD/JPY', marketCap: 650 },
                            { id: 'USDX', name: 'Dollar Index', marketCap: 600 },
                            { id: 'TNX', name: '10Y Treasury', marketCap: 550 }
                        ];
                        break;
                }
                
                this.assets = assetsList;
                resolve(assetsList);
            }, 300);
        });
    }
    
    fetchCorrelationMatrix() {
        // Mock data for correlation matrix
        return new Promise(resolve => {
            setTimeout(() => {
                if (!this.assets || this.assets.length === 0) {
                    resolve([]);
                    return;
                }
                
                const n = this.assets.length;
                const matrix = [];
                
                // Generate random correlation matrix with some patterns based on asset class
                for (let i = 0; i < n; i++) {
                    const row = [];
                    for (let j = 0; j < n; j++) {
                        if (i === j) {
                            // Self-correlation is always 1
                            row.push(1);
                        } else if (j < i) {
                            // Copy symmetric values
                            row.push(matrix[j][i]);
                        } else {
                            // Generate correlation with some patterns
                            let baseCor;
                            
                            // Different patterns for different asset classes
                            if (this.assetClass === 'crypto') {
                                // Cryptos tend to be highly correlated
                                baseCor = 0.7 + (Math.random() * 0.3 - 0.15);
                            } else if (this.assetClass === 'stock') {
                                // Stocks have moderate to high correlation
                                baseCor = 0.6 + (Math.random() * 0.4 - 0.2);
                            } else if (this.assetClass === 'forex') {
                                // Forex pairs can be positively or negatively correlated
                                baseCor = Math.random() * 1.6 - 0.8;
                            } else if (this.assetClass === 'commodity') {
                                // Commodities have varying correlation
                                baseCor = 0.3 + (Math.random() * 0.6 - 0.3);
                            } else if (this.assetClass === 'mixed') {
                                // Mixed assets have lower correlation
                                baseCor = 0.2 + (Math.random() * 0.8 - 0.4);
                            }
                            
                            // Ensure correlation is between -1 and 1
                            baseCor = Math.max(-0.95, Math.min(0.95, baseCor));
                            row.push(baseCor);
                        }
                    }
                    matrix.push(row);
                }
                
                this.correlationMatrix = matrix;
                resolve(matrix);
            }, 400);
        });
    }
    
    fetchTimeSeriesData() {
        // Mock data for time series of correlations
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(this.period);
                const startDate = new Date();
                startDate.setDate(startDate.getDate() - days);
                
                const data = [];
                
                // Generate sliding correlation data with some pattern and cycles
                for (let i = 0; i < days; i++) {
                    const date = new Date(startDate);
                    date.setDate(date.getDate() + i);
                    
                    // Create sinusoidal pattern for correlation with some noise
                    const cycle = Math.sin(i * 2 * Math.PI / (days / 3)) * 0.3;
                    const trend = i / days * 0.2; // Slight upward trend
                    
                    // Base correlations for different pairs
                    const baseCor = {
                        'BTC-ETH': 0.8,
                        'BTC-SOL': 0.7,
                        'ETH-SOL': 0.75,
                        'BTC-GOLD': 0.1,
                        'BTC-SPX': 0.4
                    };
                    
                    const pairData = {};
                    
                    for (const pair in baseCor) {
                        // Apply cycle, trend and noise to base correlation
                        let correlation = baseCor[pair] + cycle + trend + (Math.random() * 0.1 - 0.05);
                        
                        // Ensure correlation is between -1 and 1
                        correlation = Math.max(-1, Math.min(1, correlation));
                        
                        pairData[pair] = correlation;
                    }
                    
                    data.push({
                        date: date,
                        correlations: pairData
                    });
                }
                
                this.timeSeriesData = data;
                resolve(data);
            }, 350);
        });
    }
    
    fetchClusterData() {
        // Mock data for cluster analysis
        return new Promise(resolve => {
            setTimeout(() => {
                if (!this.assets || this.assets.length === 0) {
                    resolve({});
                    return;
                }
                
                // Simple clustering based on correlation matrix
                const clusters = [];
                const threshold = 0.7; // Threshold for clustering
                const visited = new Set();
                
                // Find clusters using simple threshold-based approach
                for (let i = 0; i < this.assets.length; i++) {
                    if (visited.has(i)) continue;
                    
                    const cluster = [i];
                    visited.add(i);
                    
                    for (let j = 0; j < this.assets.length; j++) {
                        if (i === j || visited.has(j)) continue;
                        
                        if (this.correlationMatrix[i][j] >= threshold) {
                            cluster.push(j);
                            visited.add(j);
                        }
                    }
                    
                    clusters.push({
                        assets: cluster.map(idx => this.assets[idx].id),
                        avgCorrelation: this.calculateClusterAvgCorrelation(cluster)
                    });
                }
                
                // Create some statistics
                const stats = {
                    clusters: clusters,
                    clusterCount: clusters.length,
                    avgCorrelation: this.calculateAvgCorrelation(),
                    highestCorrelatedPair: this.findHighestCorrelatedPair(),
                    lowestCorrelatedPair: this.findLowestCorrelatedPair(),
                    outliers: this.findOutliers()
                };
                
                this.clusterData = stats;
                resolve(stats);
            }, 450);
        });
    }
    
    calculateClusterAvgCorrelation(cluster) {
        // Calculate average correlation within a cluster
        if (cluster.length <= 1) return 1;
        
        let sum = 0;
        let count = 0;
        
        for (let i = 0; i < cluster.length; i++) {
            for (let j = i + 1; j < cluster.length; j++) {
                sum += this.correlationMatrix[cluster[i]][cluster[j]];
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0;
    }
    
    calculateAvgCorrelation() {
        // Calculate average correlation across all pairs
        if (!this.correlationMatrix || this.correlationMatrix.length === 0) return 0;
        
        let sum = 0;
        let count = 0;
        
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            for (let j = i + 1; j < this.correlationMatrix[i].length; j++) {
                sum += this.correlationMatrix[i][j];
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0;
    }
    
    findHighestCorrelatedPair() {
        // Find the pair with highest correlation
        if (!this.correlationMatrix || this.correlationMatrix.length === 0) return null;
        
        let highestCorr = -1;
        let pair = [0, 0];
        
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            for (let j = i + 1; j < this.correlationMatrix[i].length; j++) {
                if (this.correlationMatrix[i][j] > highestCorr) {
                    highestCorr = this.correlationMatrix[i][j];
                    pair = [i, j];
                }
            }
        }
        
        return {
            assets: [this.assets[pair[0]].id, this.assets[pair[1]].id],
            correlation: highestCorr
        };
    }
    
    findLowestCorrelatedPair() {
        // Find the pair with lowest correlation
        if (!this.correlationMatrix || this.correlationMatrix.length === 0) return null;
        
        let lowestCorr = 2; // Start higher than maximum possible correlation
        let pair = [0, 0];
        
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            for (let j = i + 1; j < this.correlationMatrix[i].length; j++) {
                if (this.correlationMatrix[i][j] < lowestCorr) {
                    lowestCorr = this.correlationMatrix[i][j];
                    pair = [i, j];
                }
            }
        }
        
        return {
            assets: [this.assets[pair[0]].id, this.assets[pair[1]].id],
            correlation: lowestCorr
        };
    }
    
    findOutliers() {
        // Find assets that are outliers (have low correlation with most others)
        if (!this.correlationMatrix || this.correlationMatrix.length === 0) return [];
        
        const outliers = [];
        const threshold = 0.3; // Threshold for identifying outliers
        
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            let avgCorr = 0;
            for (let j = 0; j < this.correlationMatrix[i].length; j++) {
                if (i !== j) {
                    avgCorr += this.correlationMatrix[i][j];
                }
            }
            avgCorr /= (this.correlationMatrix.length - 1);
            
            if (avgCorr < threshold) {
                outliers.push({
                    asset: this.assets[i].id,
                    avgCorrelation: avgCorr
                });
            }
        }
        
        return outliers;
    }
    
    initializeVisualizations() {
        this.renderNetworkGraph();
        this.renderCorrelationMatrix();
        this.renderCorrelationChanges();
        this.renderCorrelationAnalysis();
        this.updateMetrics();
    }
    
    renderNetworkGraph() {
        const element = document.getElementById(this.options.networkElementId);
        if (!element || !this.assets || !this.correlationMatrix) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for network graph
        const nodes = [];
        const edges = [];
        const threshold = this.options.thresholdValue;
        
        // Create nodes
        this.assets.forEach((asset, index) => {
            // Scale node size based on market cap
            const maxMarketCap = Math.max(...this.assets.map(a => a.marketCap));
            const normalizedSize = (asset.marketCap / maxMarketCap) * 
                                  (this.options.maxNodeSize - this.options.minNodeSize) + 
                                  this.options.minNodeSize;
            
            nodes.push({
                id: index,
                label: asset.id,
                size: normalizedSize,
                title: `${asset.name} (${asset.id})`
            });
        });
        
        // Create edges for correlations above threshold
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            for (let j = i + 1; j < this.correlationMatrix[i].length; j++) {
                const correlation = this.correlationMatrix[i][j];
                const absCorrelation = Math.abs(correlation);
                
                if (absCorrelation >= threshold) {
                    // Normalize correlation to 0-1 range for width
                    const width = (absCorrelation - threshold) / (1 - threshold) * 5 + 1;
                    
                    // Map correlation to color
                    const colorIndex = (correlation + 1) / 2; // Map from [-1,1] to [0,1]
                    
                    edges.push({
                        from: i,
                        to: j,
                        value: absCorrelation,
                        width: width,
                        correlation: correlation,
                        title: `${this.assets[i].id} - ${this.assets[j].id}: ${correlation.toFixed(2)}`,
                        color: this.getColorFromScale(colorIndex)
                    });
                }
            }
        }
        
        // Create network layout data
        const layout = {
            title: 'Asset Correlation Network',
            showlegend: false,
            margin: { l: 20, r: 20, t: 40, b: 20 },
            hovermode: 'closest',
            annotations: [],
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            }
        };
        
        let positions = [];
        
        // Calculate node positions based on view type
        if (this.viewType === 'force') {
            // Simple force-directed layout simulation
            const simulation = this.forceSimulation(nodes, edges);
            positions = simulation.positions;
        } else if (this.viewType === 'circular') {
            // Circular layout
            positions = this.circularLayout(nodes);
        } else if (this.viewType === 'cluster') {
            // Clustered layout
            positions = this.clusteredLayout(nodes, edges);
        }
        
        // Create node traces
        const nodeTrace = {
            x: positions.map(p => p.x),
            y: positions.map(p => p.y),
            mode: 'markers+text',
            marker: {
                size: nodes.map(n => n.size),
                color: 'rgba(var(--primary-rgb), 0.8)',
                line: { width: 1, color: 'var(--border-light)' }
            },
            text: nodes.map(n => n.label),
            textposition: 'middle center',
            textfont: {
                size: 9,
                color: 'white'
            },
            hoverinfo: 'text',
            hovertext: nodes.map(n => n.title),
            type: 'scatter'
        };
        
        // Create edge traces
        const edgeTraces = [];
        
        edges.forEach(edge => {
            const source = positions[edge.from];
            const target = positions[edge.to];
            
            edgeTraces.push({
                x: [source.x, target.x],
                y: [source.y, target.y],
                mode: 'lines',
                line: {
                    width: edge.width,
                    color: edge.color
                },
                hoverinfo: 'text',
                hovertext: edge.title,
                type: 'scatter'
            });
        });
        
        // Combine all traces
        const traces = [...edgeTraces, nodeTrace];
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.networkElementId, traces, layout, config);
        this.networkPlot = document.getElementById(this.options.networkElementId);
    }
    
    forceSimulation(nodes, edges) {
        // Very simplified force-directed layout algorithm
        const positions = [];
        const iterations = 100;
        const repulsionForce = 100;
        const attractionForce = 0.05;
        
        // Initialize random positions
        for (let i = 0; i < nodes.length; i++) {
            positions.push({
                x: Math.random() * 2 - 1,
                y: Math.random() * 2 - 1
            });
        }
        
        // Run simulation
        for (let iter = 0; iter < iterations; iter++) {
            // Calculate forces
            const forces = [];
            for (let i = 0; i < nodes.length; i++) {
                forces.push({ x: 0, y: 0 });
            }
            
            // Repulsion forces (node-node)
            for (let i = 0; i < nodes.length; i++) {
                for (let j = 0; j < nodes.length; j++) {
                    if (i === j) continue;
                    
                    const dx = positions[j].x - positions[i].x;
                    const dy = positions[j].y - positions[i].y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 0.1;
                    
                    const force = repulsionForce / (distance * distance);
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    
                    forces[i].x -= fx;
                    forces[i].y -= fy;
                }
            }
            
            // Attraction forces (edges)
            for (const edge of edges) {
                const i = edge.from;
                const j = edge.to;
                
                const dx = positions[j].x - positions[i].x;
                const dy = positions[j].y - positions[i].y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 0.1;
                
                const force = distance * attractionForce * edge.value;
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                
                forces[i].x += fx;
                forces[i].y += fy;
                forces[j].x -= fx;
                forces[j].y -= fy;
            }
            
            // Apply forces
            for (let i = 0; i < nodes.length; i++) {
                positions[i].x += forces[i].x * 0.1;
                positions[i].y += forces[i].y * 0.1;
                
                // Constrain to circle
                const dist = Math.sqrt(positions[i].x * positions[i].x + positions[i].y * positions[i].y);
                if (dist > 1) {
                    positions[i].x /= dist;
                    positions[i].y /= dist;
                }
            }
        }
        
        return { positions };
    }
    
    circularLayout(nodes) {
        // Simple circular layout
        const positions = [];
        const radius = 0.8;
        const angle = (2 * Math.PI) / nodes.length;
        
        for (let i = 0; i < nodes.length; i++) {
            positions.push({
                x: radius * Math.cos(i * angle),
                y: radius * Math.sin(i * angle)
            });
        }
        
        return positions;
    }
    
    clusteredLayout(nodes, edges) {
        // Clustered layout based on correlation
        const positions = [];
        const clusters = [];
        const visited = new Set();
        const threshold = 0.7;
        
        // Find clusters
        for (let i = 0; i < nodes.length; i++) {
            if (visited.has(i)) continue;
            
            const cluster = [i];
            visited.add(i);
            
            for (let j = 0; j < nodes.length; j++) {
                if (i === j || visited.has(j)) continue;
                
                if (this.correlationMatrix[i][j] >= threshold) {
                    cluster.push(j);
                    visited.add(j);
                }
            }
            
            clusters.push(cluster);
        }
        
        // Calculate positions for each cluster
        const clusterRadius = 0.8;
        const clusterAngle = (2 * Math.PI) / clusters.length;
        
        for (let c = 0; c < clusters.length; c++) {
            const cluster = clusters[c];
            const cx = clusterRadius * Math.cos(c * clusterAngle);
            const cy = clusterRadius * Math.sin(c * clusterAngle);
            
            // Position nodes within cluster
            const nodeRadius = 0.15;
            const nodeAngle = (2 * Math.PI) / cluster.length;
            
            for (let i = 0; i < cluster.length; i++) {
                const nodeIndex = cluster[i];
                positions[nodeIndex] = {
                    x: cx + nodeRadius * Math.cos(i * nodeAngle),
                    y: cy + nodeRadius * Math.sin(i * nodeAngle)
                };
            }
        }
        
        // Handle any unpositioned nodes
        for (let i = 0; i < nodes.length; i++) {
            if (!positions[i]) {
                positions[i] = {
                    x: Math.random() * 0.5 - 0.25,
                    y: Math.random() * 0.5 - 0.25
                };
            }
        }
        
        return positions;
    }
    
    renderCorrelationMatrix() {
        const element = document.getElementById(this.options.matrixElementId);
        if (!element || !this.assets || !this.correlationMatrix) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for heatmap
        let indices = [...Array(this.assets.length).keys()];
        
        // Apply sorting if needed
        if (this.sortType === 'cluster') {
            indices = this.getClusterSortedIndices();
        } else if (this.sortType === 'value') {
            indices = this.getValueSortedIndices();
        }
        
        // Labels for heatmap
        const labels = indices.map(i => this.assets[i].id);
        
        // Create sorted correlation matrix
        const zValues = [];
        for (let i = 0; i < indices.length; i++) {
            const row = [];
            for (let j = 0; j < indices.length; j++) {
                row.push(this.correlationMatrix[indices[i]][indices[j]]);
            }
            zValues.push(row);
        }
        
        // Create heatmap trace
        const trace = {
            x: labels,
            y: labels,
            z: zValues,
            type: 'heatmap',
            colorscale: this.options.colorScale,
            zmin: -1,
            zmax: 1,
            hovertemplate: 'X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Correlation Matrix',
            margin: { l: 50, r: 20, t: 40, b: 50 },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: '',
                showgrid: false,
                tickangle: -45
            },
            yaxis: {
                title: '',
                showgrid: false
            }
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.matrixElementId, [trace], layout, config);
        this.matrixPlot = document.getElementById(this.options.matrixElementId);
    }
    
    getClusterSortedIndices() {
        // Sort indices based on clustering
        const n = this.assets.length;
        const distanceMatrix = Array(n).fill().map(() => Array(n).fill(0));
        
        // Convert correlation matrix to distance matrix
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                // Convert correlation (-1 to 1) to distance (0 to 2)
                distanceMatrix[i][j] = 1 - this.correlationMatrix[i][j];
            }
        }
        
        // Simple hierarchical clustering
        const clusters = [];
        const visited = new Set();
        
        for (let i = 0; i < n; i++) {
            if (visited.has(i)) continue;
            
            const cluster = [i];
            visited.add(i);
            
            // Find nodes close to this one
            for (let j = 0; j < n; j++) {
                if (i === j || visited.has(j)) continue;
                
                if (distanceMatrix[i][j] < 0.4) { // Distance threshold
                    cluster.push(j);
                    visited.add(j);
                }
            }
            
            clusters.push(cluster);
        }
        
        // Flatten clusters
        return clusters.flat();
    }
    
    getValueSortedIndices() {
        // Sort indices based on average correlation value
        const n = this.assets.length;
        const avgCorrelations = [];
        
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    sum += this.correlationMatrix[i][j];
                }
            }
            avgCorrelations.push({ index: i, value: sum / (n - 1) });
        }
        
        // Sort by average correlation
        avgCorrelations.sort((a, b) => b.value - a.value);
        
        return avgCorrelations.map(item => item.index);
    }
    
    renderCorrelationChanges() {
        const element = document.getElementById(this.options.changesElementId);
        if (!element || !this.timeSeriesData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Prepare data for line chart
        const dates = this.timeSeriesData.map(item => item.date);
        let correlations = [];
        
        // Get correlation data for current pair
        if (this.currentPair) {
            correlations = this.timeSeriesData.map(item => item.correlations[this.currentPair] || 0);
        }
        
        // Create line chart trace
        const trace = {
            x: dates,
            y: correlations,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: 'rgba(var(--primary-rgb), 0.8)',
                width: 2
            },
            name: this.currentPair,
            hovertemplate: '%{x|%b %d, %Y}<br>Correlation: %{y:.2f}<extra></extra>'
        };
        
        // Add reference lines at 0, 0.5, and -0.5
        const refLines = [
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.5)',
                    width: 1,
                    dash: 'dash'
                },
                hoverinfo: 'none',
                showlegend: false
            },
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [0.5, 0.5],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.3)',
                    width: 1,
                    dash: 'dot'
                },
                hoverinfo: 'none',
                showlegend: false
            },
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [-0.5, -0.5],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'rgba(var(--text-light-rgb), 0.3)',
                    width: 1,
                    dash: 'dot'
                },
                hoverinfo: 'none',
                showlegend: false
            }
        ];
        
        // Layout configuration
        const layout = {
            title: `${this.currentPair} Correlation Over Time`,
            margin: { l: 40, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: '',
                showgrid: false
            },
            yaxis: {
                title: 'Correlation',
                range: [-1, 1],
                showgrid: true,
                zeroline: false,
                gridcolor: 'rgba(var(--border-light-rgb), 0.3)'
            },
            showlegend: false
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.changesElementId, [trace, ...refLines], layout, config);
        this.changesPlot = document.getElementById(this.options.changesElementId);
    }
    
    renderCorrelationAnalysis() {
        const element = document.getElementById(this.options.analysisElementId);
        if (!element || !this.clusterData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Render different content based on current tab
        if (this.analysisTab === 'stats') {
            this.renderCorrelationStats(element);
        } else if (this.analysisTab === 'clusters') {
            this.renderClusterAnalysis(element);
        } else if (this.analysisTab === 'outliers') {
            this.renderOutlierAnalysis(element);
        }
    }
    
    renderCorrelationStats(element) {
        // Create simple stats view
        const avgCorr = this.clusterData.avgCorrelation;
        const highest = this.clusterData.highestCorrelatedPair;
        const lowest = this.clusterData.lowestCorrelatedPair;
        
        // Calculate correlation distribution
        const bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1];
        const distribution = Array(bins.length - 1).fill(0);
        
        for (let i = 0; i < this.correlationMatrix.length; i++) {
            for (let j = i + 1; j < this.correlationMatrix[i].length; j++) {
                const corr = this.correlationMatrix[i][j];
                for (let k = 0; k < bins.length - 1; k++) {
                    if (corr >= bins[k] && corr < bins[k + 1]) {
                        distribution[k]++;
                        break;
                    }
                }
            }
        }
        
        // Calculate bin labels
        const binLabels = [];
        for (let i = 0; i < bins.length - 1; i++) {
            binLabels.push(`${bins[i].toFixed(1)} to ${bins[i+1].toFixed(1)}`);
        }
        
        // Create histogram trace
        const trace = {
            x: binLabels,
            y: distribution,
            type: 'bar',
            marker: {
                color: 'rgba(var(--primary-rgb), 0.7)',
                line: {
                    color: 'rgba(var(--primary-rgb), 1)',
                    width: 1
                }
            }
        };
        
        // Layout configuration
        const layout = {
            title: 'Correlation Distribution',
            margin: { l: 40, r: 20, t: 40, b: 80 },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Correlation Range',
                tickangle: -45
            },
            yaxis: {
                title: 'Number of Pairs'
            },
            annotations: [
                {
                    x: 0.5,
                    y: 1.1,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Average: ${avgCorr.toFixed(2)} | Highest: ${highest.assets.join('-')} (${highest.correlation.toFixed(2)}) | Lowest: ${lowest.assets.join('-')} (${lowest.correlation.toFixed(2)})`,
                    showarrow: false,
                    font: {
                        size: 10
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
        Plotly.newPlot(this.options.analysisElementId, [trace], layout, config);
        this.analysisPlot = document.getElementById(this.options.analysisElementId);
    }
    
    renderClusterAnalysis(element) {
        // Render cluster analysis
        const clusters = this.clusterData.clusters;
        
        // Create bar chart for clusters
        const clusterSizes = clusters.map(c => c.assets.length);
        const clusterLabels = clusters.map((c, i) => `Cluster ${i+1}`);
        const clusterCorrelations = clusters.map(c => c.avgCorrelation);
        
        // Create traces
        const sizeTrace = {
            x: clusterLabels,
            y: clusterSizes,
            type: 'bar',
            name: 'Size',
            marker: {
                color: 'rgba(var(--primary-rgb), 0.7)'
            },
            hovertemplate: 'Cluster: %{x}<br>Size: %{y}<extra></extra>'
        };
        
        const corrTrace = {
            x: clusterLabels,
            y: clusterCorrelations,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Avg. Correlation',
            yaxis: 'y2',
            marker: {
                color: 'rgba(var(--info-rgb), 0.8)'
            },
            line: {
                color: 'rgba(var(--info-rgb), 0.8)'
            },
            hovertemplate: 'Cluster: %{x}<br>Avg. Correlation: %{y:.2f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Cluster Analysis',
            margin: { l: 40, r: 40, t: 40, b: 40 },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: ''
            },
            yaxis: {
                title: 'Cluster Size',
                side: 'left'
            },
            yaxis2: {
                title: 'Avg. Correlation',
                side: 'right',
                overlaying: 'y',
                range: [0, 1]
            },
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            },
            annotations: clusters.map((cluster, i) => ({
                x: clusterLabels[i],
                y: clusterSizes[i] + 0.5,
                text: cluster.assets.join(', '),
                showarrow: false,
                font: {
                    size: 8
                }
            }))
        };
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.analysisElementId, [sizeTrace, corrTrace], layout, config);
        this.analysisPlot = document.getElementById(this.options.analysisElementId);
    }
    
    renderOutlierAnalysis(element) {
        // Render outlier analysis
        const outliers = this.clusterData.outliers;
        
        if (outliers.length === 0) {
            // No outliers
            element.innerHTML = `
                <div class="no-outliers">
                    <p>No significant outliers detected in the current asset set.</p>
                </div>
            `;
            return;
        }
        
        // Create sorted list of all assets by average correlation
        const allAssets = [];
        for (let i = 0; i < this.assets.length; i++) {
            let avgCorr = 0;
            for (let j = 0; j < this.correlationMatrix[i].length; j++) {
                if (i !== j) {
                    avgCorr += this.correlationMatrix[i][j];
                }
            }
            avgCorr /= (this.correlationMatrix.length - 1);
            
            allAssets.push({
                id: this.assets[i].id,
                avgCorrelation: avgCorr
            });
        }
        
        allAssets.sort((a, b) => a.avgCorrelation - b.avgCorrelation);
        
        // Create horizontal bar chart
        const assetIds = allAssets.map(a => a.id);
        const correlations = allAssets.map(a => a.avgCorrelation);
        
        // Create color array with outliers highlighted
        const colors = correlations.map(corr => {
            if (corr < 0.3) {
                return 'rgba(var(--danger-rgb), 0.7)';
            } else if (corr > 0.7) {
                return 'rgba(var(--success-rgb), 0.7)';
            } else {
                return 'rgba(var(--primary-rgb), 0.7)';
            }
        });
        
        // Create trace
        const trace = {
            y: assetIds,
            x: correlations,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: colors
            },
            hovertemplate: 'Asset: %{y}<br>Avg. Correlation: %{x:.2f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: 'Assets by Average Correlation',
            margin: { l: 60, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'var(--card-bg)',
            plot_bgcolor: 'var(--card-bg)',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Average Correlation',
                range: [-1, 1]
            },
            yaxis: {
                title: '',
                automargin: true
            },
            annotations: [
                {
                    x: -0.9,
                    y: -0.1,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Low correlation assets may provide diversification benefits',
                    showarrow: false,
                    font: {
                        size: 9,
                        color: 'rgba(var(--danger-rgb), 0.9)'
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
        Plotly.newPlot(this.options.analysisElementId, [trace], layout, config);
        this.analysisPlot = document.getElementById(this.options.analysisElementId);
    }
    
    updateMetrics() {
        // Update metrics in the footer
        if (this.clusterData) {
            // Average correlation
            const avgCorrelationElement = document.getElementById('avg-correlation');
            if (avgCorrelationElement) {
                avgCorrelationElement.textContent = this.clusterData.avgCorrelation.toFixed(2);
            }
            
            // Cluster count
            const clusterCountElement = document.getElementById('cluster-count');
            if (clusterCountElement) {
                clusterCountElement.textContent = this.clusterData.clusterCount;
            }
            
            // Correlation stability (mock data for now)
            const stabilityElement = document.getElementById('correlation-stability');
            if (stabilityElement) {
                const stability = this.calculateStability();
                
                if (stability > 0.8) {
                    stabilityElement.textContent = 'High';
                    stabilityElement.className = 'metric-value positive';
                } else if (stability > 0.5) {
                    stabilityElement.textContent = 'Medium';
                    stabilityElement.className = 'metric-value neutral';
                } else {
                    stabilityElement.textContent = 'Low';
                    stabilityElement.className = 'metric-value negative';
                }
            }
            
            // Last updated timestamp
            const lastUpdatedElement = document.getElementById('last-updated');
            if (lastUpdatedElement) {
                const minutes = Math.floor(Math.random() * 60);
                lastUpdatedElement.textContent = `${minutes} minutes ago`;
            }
        }
    }
    
    calculateStability() {
        // Calculate a mock stability score
        if (!this.timeSeriesData || this.timeSeriesData.length < 2) return 0.5;
        
        let sum = 0;
        let count = 0;
        
        // Calculate average variance of correlations over time
        for (const pair in this.timeSeriesData[0].correlations) {
            const values = this.timeSeriesData.map(d => d.correlations[pair]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            
            // Calculate variance
            const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
            
            sum += 1 - Math.min(variance, 1); // Convert variance to stability (lower variance = higher stability)
            count++;
        }
        
        return count > 0 ? sum / count : 0.5;
    }
    
    setupEventListeners() {
        // Asset class selector change
        const assetClassSelector = document.getElementById('asset-class');
        if (assetClassSelector) {
            assetClassSelector.addEventListener('change', () => {
                this.assetClass = assetClassSelector.value;
                this.refreshData();
            });
        }
        
        // Timeframe selector change
        const timeframeSelector = document.getElementById('correlation-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }
        
        // Period selector change
        const periodSelector = document.getElementById('correlation-period');
        if (periodSelector) {
            periodSelector.addEventListener('change', () => {
                this.period = periodSelector.value;
                this.refreshData();
            });
        }
        
        // Correlation metric change
        const metricSelector = document.getElementById('correlation-metric');
        if (metricSelector) {
            metricSelector.addEventListener('change', () => {
                this.metric = metricSelector.value;
                this.refreshData();
            });
        }
        
        // Correlation threshold change
        const thresholdSlider = document.getElementById('correlation-threshold');
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', () => {
                const thresholdValue = parseFloat(thresholdSlider.value);
                this.options.thresholdValue = thresholdValue;
                
                // Update threshold value display
                const thresholdDisplay = document.getElementById('threshold-value');
                if (thresholdDisplay) {
                    thresholdDisplay.textContent = thresholdValue.toFixed(2);
                }
                
                // Re-render network graph with new threshold
                this.renderNetworkGraph();
            });
        }
        
        // Network view type change
        const viewOptions = document.querySelectorAll('.view-option');
        viewOptions.forEach(option => {
            option.addEventListener('click', () => {
                // Remove active class from all options
                viewOptions.forEach(opt => opt.classList.remove('active'));
                // Add active class to clicked option
                option.classList.add('active');
                
                // Update view type
                this.viewType = option.dataset.view;
                
                // Re-render network graph with new view type
                this.renderNetworkGraph();
            });
        });
        
        // Matrix sort change
        const matrixSortSelector = document.getElementById('matrix-sort');
        if (matrixSortSelector) {
            matrixSortSelector.addEventListener('change', () => {
                this.sortType = matrixSortSelector.value;
                this.renderCorrelationMatrix();
            });
        }
        
        // Asset pair change
        const pairSelector = document.getElementById('asset-pair');
        if (pairSelector) {
            pairSelector.addEventListener('change', () => {
                this.currentPair = pairSelector.value;
                this.renderCorrelationChanges();
            });
        }
        
        // Analysis tab change
        const analysisTabs = document.querySelectorAll('.analysis-tab');
        analysisTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                analysisTabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Update analysis tab
                this.analysisTab = tab.dataset.tab;
                
                // Re-render analysis with new tab
                this.renderCorrelationAnalysis();
            });
        });
        
        // Correlation settings button
        const settingsBtn = document.getElementById('correlation-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('correlation-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-correlation-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadCorrelationData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-correlation-panel-btn');
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
        
        // Advanced settings toggle
        const advancedToggle = document.getElementById('use-advanced-settings');
        const advancedContainer = document.getElementById('advanced-settings');
        if (advancedToggle && advancedContainer) {
            advancedToggle.addEventListener('change', () => {
                advancedContainer.style.display = advancedToggle.checked ? 'block' : 'none';
            });
        }
        
        // Save settings button
        const saveSettingsBtn = document.getElementById('save-correlation-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                
                // Close the modal
                const modal = document.getElementById('correlation-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Slider value updates
        const minThresholdSlider = document.getElementById('min-threshold');
        if (minThresholdSlider) {
            minThresholdSlider.addEventListener('input', function() {
                // Update the value display
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
            });
        }
        
        // Learn more button
        const learnMoreBtn = document.getElementById('correlation-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Cross-Asset Correlation documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-correlation-report');
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
            this.options.networkElementId,
            this.options.matrixElementId,
            this.options.changesElementId,
            this.options.analysisElementId
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
                this.renderNetworkGraph();
                this.renderCorrelationMatrix();
                this.renderCorrelationChanges();
                this.renderCorrelationAnalysis();
                this.updateMetrics();
                this.updateAssetPairOptions();
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
    
    updateAssetPairOptions() {
        // Update asset pair dropdown options based on current assets
        const pairSelector = document.getElementById('asset-pair');
        if (pairSelector && this.assets && this.assets.length >= 2) {
            // Clear existing options
            pairSelector.innerHTML = '';
            
            // Create options for all possible pairs
            for (let i = 0; i < this.assets.length; i++) {
                for (let j = i + 1; j < this.assets.length; j++) {
                    const asset1 = this.assets[i].id;
                    const asset2 = this.assets[j].id;
                    const pairId = `${asset1}-${asset2}`;
                    
                    const option = document.createElement('option');
                    option.value = pairId;
                    option.textContent = pairId;
                    
                    // Set as selected if it matches current pair
                    if (pairId === this.currentPair) {
                        option.selected = true;
                    }
                    
                    pairSelector.appendChild(option);
                }
            }
            
            // If current pair is not in the list, select the first one
            if (!pairSelector.value) {
                this.currentPair = pairSelector.options[0].value;
                pairSelector.value = this.currentPair;
            }
        }
    }
    
    downloadCorrelationData() {
        // Create downloadable CSV file with correlation matrix
        if (!this.assets || !this.correlationMatrix) return;
        
        // Prepare CSV content
        let csv = 'Asset,' + this.assets.map(a => a.id).join(',') + '\n';
        
        for (let i = 0; i < this.assets.length; i++) {
            csv += this.assets[i].id;
            for (let j = 0; j < this.correlationMatrix[i].length; j++) {
                csv += ',' + this.correlationMatrix[i][j].toFixed(4);
            }
            csv += '\n';
        }
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `correlation_matrix_${this.assetClass}_${this.timeframe}.csv`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    getColorFromScale(value) {
        // Get color from colorscale based on value (0-1)
        const colorscale = this.options.colorScale;
        
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

// Initialize the Cross-Asset Correlation component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of CrossAssetCorrelation
    const correlationAnalysis = new CrossAssetCorrelation();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});