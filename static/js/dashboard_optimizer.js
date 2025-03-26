/**
 * Dashboard Optimizer
 * 
 * This module provides performance optimization utilities for the dashboard
 * including efficient rendering, data caching, and resource management.
 */

class DashboardOptimizer {
    constructor(options = {}) {
        this.options = Object.assign({
            enableCache: true,
            cacheLifetime: 60000, // Default cache lifetime (60 seconds)
            enableLazyLoading: true,
            enableChunkedRendering: true,
            chunkSize: 50, // Number of items per chunk for chunked rendering
            chunkDelay: 1, // Milliseconds delay between chunks
            memoryLimit: 50 * 1024 * 1024, // 50MB rough estimate for memory limit
            debounceTime: 250, // Debounce time for window resize events
            throttleTime: 100, // Throttle time for rapid events
        }, options);
        
        // Initialize caches
        this.cache = {};
        this.cacheTimestamps = {};
        this.cacheExpirations = {
            'market-data': 5000,      // 5 seconds
            'portfolio-data': 10000,  // 10 seconds
            'system-health': 5000,    // 5 seconds
            'risk-metrics': 30000,    // 30 seconds
            'performance-data': 30000 // 30 seconds
        };
        
        // Track loaded components
        this.loadedComponents = {};
        
        // Memory management
        this.cachedObjectSizes = {};
        this.totalMemoryUsed = 0;
        
        // Bind methods
        this.getData = this.getData.bind(this);
        this.renderChunked = this.renderChunked.bind(this);
        this.lazyLoadComponent = this.lazyLoadComponent.bind(this);
        this.getFromCache = this.getFromCache.bind(this);
        this.saveToCache = this.saveToCache.bind(this);
    }
    
    /**
     * Efficient data fetching with caching
     * @param {string} dataType - The type of data to fetch
     * @param {object} params - Parameters for the data request
     * @param {boolean} forceRefresh - Force a refresh of cached data
     * @returns {Promise<any>} The requested data
     */
    async getData(dataType, params = {}, forceRefresh = false) {
        // Create a cache key from data type and parameters
        const cacheKey = `${dataType}:${JSON.stringify(params)}`;
        
        // Check if caching is enabled and data is in cache
        if (this.options.enableCache && !forceRefresh) {
            const cachedData = this.getFromCache(cacheKey);
            if (cachedData) {
                return cachedData;
            }
        }
        
        // Data wasn't cached or refresh was forced, fetch from source
        try {
            // Simulate API fetch - in real implementation this would be a fetch() call
            const data = await this.fetchDataFromSource(dataType, params);
            
            // Cache the fetched data if caching is enabled
            if (this.options.enableCache) {
                this.saveToCache(cacheKey, data, this.cacheExpirations[dataType] || this.options.cacheLifetime);
            }
            
            return data;
        } catch (error) {
            console.error(`Error fetching ${dataType} data:`, error);
            throw error;
        }
    }
    
    /**
     * Fetch data from source (API, mock data, etc.)
     * @param {string} dataType - The type of data to fetch
     * @param {object} params - Parameters for the data request
     * @returns {Promise<any>} The fetched data
     */
    async fetchDataFromSource(dataType, params) {
        // In a real implementation, this would call the appropriate API
        // For demonstration, we'll just simulate a network request
        return new Promise((resolve) => {
            setTimeout(() => {
                // Generate appropriate mock data based on data type
                let data;
                switch (dataType) {
                    case 'market-data':
                        data = this.generateMockMarketData(params);
                        break;
                    case 'portfolio-data':
                        data = this.generateMockPortfolioData(params);
                        break;
                    case 'risk-metrics':
                        data = this.generateMockRiskMetricsData(params);
                        break;
                    case 'performance-data':
                        data = this.generateMockPerformanceData(params);
                        break;
                    case 'system-health':
                        data = this.generateMockSystemHealthData(params);
                        break;
                    default:
                        data = { message: 'Unknown data type' };
                }
                resolve(data);
            }, 300); // Simulate network delay
        });
    }
    
    /**
     * Get data from cache if it exists and hasn't expired
     * @param {string} key - Cache key
     * @returns {any|null} The cached data or null if not found/expired
     */
    getFromCache(key) {
        // Check if key exists in cache
        if (key in this.cache) {
            const timestamp = this.cacheTimestamps[key] || 0;
            const expiration = this.cacheExpirations[key.split(':')[0]] || this.options.cacheLifetime;
            
            // Check if cache is still valid
            if (Date.now() - timestamp < expiration) {
                return this.cache[key];
            }
        }
        return null;
    }
    
    /**
     * Save data to cache with size tracking for memory management
     * @param {string} key - Cache key
     * @param {any} data - Data to cache
     * @param {number} expiration - Cache expiration time in ms
     */
    saveToCache(key, data, expiration) {
        // Store the data in cache
        this.cache[key] = data;
        this.cacheTimestamps[key] = Date.now();
        
        // Set expiration time based on data type or default
        const dataType = key.split(':')[0];
        this.cacheExpirations[dataType] = expiration;
        
        // Estimate object size and track memory usage
        const estimatedSize = this.estimateObjectSize(data);
        this.cachedObjectSizes[key] = estimatedSize;
        this.totalMemoryUsed += estimatedSize;
        
        // Perform cache cleanup if memory limit is approached
        if (this.totalMemoryUsed > this.options.memoryLimit * 0.8) {
            this.cleanupCache();
        }
    }
    
    /**
     * Clean up cache when memory limits are approached
     * Removes oldest and least recently used items first
     */
    cleanupCache() {
        // Sort cache keys by timestamp (oldest first)
        const sortedKeys = Object.keys(this.cacheTimestamps)
            .sort((a, b) => this.cacheTimestamps[a] - this.cacheTimestamps[b]);
        
        // Remove oldest items until we're below 70% of memory limit
        let keysRemoved = 0;
        for (const key of sortedKeys) {
            if (this.totalMemoryUsed < this.options.memoryLimit * 0.7) {
                break;
            }
            
            // Free up memory
            this.totalMemoryUsed -= (this.cachedObjectSizes[key] || 0);
            delete this.cache[key];
            delete this.cacheTimestamps[key];
            delete this.cachedObjectSizes[key];
            keysRemoved++;
        }
        
        console.log(`Cache cleanup performed: ${keysRemoved} items removed. Memory usage: ${Math.round(this.totalMemoryUsed / 1024 / 1024)}MB`);
    }
    
    /**
     * Estimate size of an object in bytes (rough approximation)
     * @param {any} object - The object to measure
     * @returns {number} Estimated size in bytes
     */
    estimateObjectSize(object) {
        try {
            // Use JSON.stringify for a rough size estimate
            const json = JSON.stringify(object);
            return json ? json.length * 2 : 0; // Approximation: 2 bytes per character
        } catch (e) {
            // If the object can't be stringified, make a conservative estimate
            return 10000; // 10KB default size
        }
    }
    
    /**
     * Render large datasets in chunks to avoid blocking the main thread
     * @param {string} containerId - Container element ID
     * @param {Array} items - Array of items to render
     * @param {Function} renderItemFn - Function that renders a single item
     * @param {number} chunkSize - Number of items per chunk
     * @param {number} delay - Delay between chunks in ms
     * @returns {Promise<void>} Promise that resolves when all chunks are rendered
     */
    renderChunked(containerId, items, renderItemFn, chunkSize = null, delay = null) {
        return new Promise((resolve) => {
            // Use provided or default chunk size
            const itemsPerChunk = chunkSize || this.options.chunkSize;
            const chunkDelay = delay || this.options.chunkDelay;
            
            const container = document.getElementById(containerId);
            if (!container) {
                console.error(`Container ${containerId} not found`);
                resolve();
                return;
            }
            
            // Create a document fragment for efficient DOM manipulation
            const fragment = document.createDocumentFragment();
            
            // Function to render a chunk of items
            const renderChunk = (startIndex) => {
                // Get the current chunk
                const endIndex = Math.min(startIndex + itemsPerChunk, items.length);
                const chunk = items.slice(startIndex, endIndex);
                
                // Render each item in the chunk
                chunk.forEach(item => {
                    const renderedItem = renderItemFn(item);
                    if (renderedItem) {
                        fragment.appendChild(renderedItem);
                    }
                });
                
                // If this is the last chunk, append the fragment and resolve
                if (endIndex >= items.length) {
                    container.appendChild(fragment);
                    resolve();
                    return;
                }
                
                // If there are more chunks, schedule the next one
                setTimeout(() => {
                    // Before processing the next chunk, append the current fragment
                    container.appendChild(fragment);
                    
                    // Allow browser to render and process events
                    requestAnimationFrame(() => {
                        renderChunk(endIndex);
                    });
                }, chunkDelay);
            };
            
            // Start rendering from the first chunk
            renderChunk(0);
        });
    }
    
    /**
     * Lazy load a component when it becomes visible
     * @param {string} componentId - DOM element ID of the component
     * @param {Function} loadFn - Function to call when component should be loaded
     */
    lazyLoadComponent(componentId, loadFn) {
        // Skip if lazy loading is disabled
        if (!this.options.enableLazyLoading) {
            loadFn();
            this.loadedComponents[componentId] = true;
            return;
        }
        
        // Check if component already loaded
        if (this.loadedComponents[componentId]) {
            return;
        }
        
        const element = document.getElementById(componentId);
        if (!element) {
            console.error(`Component element ${componentId} not found`);
            return;
        }
        
        // Use Intersection Observer for lazy loading when available
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        // Component is visible, load it
                        loadFn();
                        this.loadedComponents[componentId] = true;
                        
                        // Stop observing once loaded
                        observer.unobserve(element);
                    }
                });
            }, {
                root: null, // viewport
                rootMargin: '100px', // Load 100px before it becomes visible
                threshold: 0.1 // Load when at least 10% is visible
            });
            
            // Start observing the element
            observer.observe(element);
        } else {
            // Fallback for browsers without IntersectionObserver
            loadFn();
            this.loadedComponents[componentId] = true;
        }
    }
    
    /**
     * Throttle a function to limit how often it can be called
     * @param {Function} fn - Function to throttle
     * @param {number} delay - Throttle delay in ms
     * @returns {Function} Throttled function
     */
    throttle(fn, delay = null) {
        const throttleDelay = delay || this.options.throttleTime;
        let lastCall = 0;
        
        return function(...args) {
            const now = Date.now();
            if (now - lastCall >= throttleDelay) {
                lastCall = now;
                return fn.apply(this, args);
            }
        };
    }
    
    /**
     * Debounce a function to delay execution until after a pause
     * @param {Function} fn - Function to debounce
     * @param {number} delay - Debounce delay in ms
     * @returns {Function} Debounced function
     */
    debounce(fn, delay = null) {
        const debounceDelay = delay || this.options.debounceTime;
        let timeout;
        
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                fn.apply(this, args);
            }, debounceDelay);
        };
    }
    
    /**
     * Optimize charts for performance
     * @param {object} chart - Chart object (Plotly chart)
     */
    optimizeChart(chart) {
        // Skip if no chart or Plotly not available
        if (!chart || !window.Plotly) return;
        
        // Set chart configurations for performance
        const config = {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'resetScale2d'],
            toImageButtonOptions: {
                format: 'png',
                width: 800,
                height: 600
            },
            queueLength: 2, // Limit queue size for better performance
            showSendToCloud: false,
            plotGlPixelRatio: 1 // Lower resolution for WebGL charts
        };
        
        // Apply optimized config
        Plotly.update(chart, {}, {}, config);
    }
    
    /**
     * Generate mock market data for testing
     * @param {object} params - Request parameters
     * @returns {object} Mock market data
     */
    generateMockMarketData(params) {
        const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD'];
        const data = {};
        
        symbols.forEach(symbol => {
            data[symbol] = {
                price: 10000 + Math.random() * 40000,
                volume: Math.random() * 1000000,
                change: (Math.random() * 10) - 5,
                high: 11000 + Math.random() * 40000,
                low: 9000 + Math.random() * 40000
            };
        });
        
        return {
            timestamp: Date.now(),
            data: data
        };
    }
    
    /**
     * Generate mock portfolio data for testing
     * @param {object} params - Request parameters
     * @returns {object} Mock portfolio data
     */
    generateMockPortfolioData(params) {
        const positions = [
            { symbol: 'BTC-USD', quantity: 1.25, value: 45000, unrealizedPnl: 2500 },
            { symbol: 'ETH-USD', quantity: 10.5, value: 32000, unrealizedPnl: -1200 },
            { symbol: 'SOL-USD', quantity: 50, value: 5000, unrealizedPnl: 800 },
            { symbol: 'AVAX-USD', quantity: 25, value: 2500, unrealizedPnl: 350 }
        ];
        
        return {
            timestamp: Date.now(),
            totalValue: 84500,
            totalPnl: 2450,
            positions: positions
        };
    }
    
    /**
     * Generate mock risk metrics data for testing
     * @param {object} params - Request parameters
     * @returns {object} Mock risk metrics data
     */
    generateMockRiskMetricsData(params) {
        const strategies = ['Trend Following', 'Mean Reversion', 'Statistical Arbitrage', 'Sentiment Based'];
        const ratios = {};
        
        strategies.forEach(strategy => {
            ratios[strategy] = {
                sharpe: 0.8 + Math.random() * 1.5,
                sortino: 1.2 + Math.random() * 1.8,
                calmar: 0.5 + Math.random() * 1.2,
                maxDrawdown: 5 + Math.random() * 20
            };
        });
        
        return {
            timestamp: Date.now(),
            portfolio: {
                sharpe: 1.2 + Math.random() * 0.8,
                sortino: 1.5 + Math.random() * 1.0,
                calmar: 0.7 + Math.random() * 0.6,
                maxDrawdown: 8 + Math.random() * 10
            },
            strategies: ratios
        };
    }
    
    /**
     * Generate mock performance data for testing
     * @param {object} params - Request parameters
     * @returns {object} Mock performance data
     */
    generateMockPerformanceData(params) {
        const timeframe = params.timeframe || '1m';
        const points = this.getTimeframePoints(timeframe);
        
        // Generate time series data
        const dates = [];
        const returns = [];
        const equity = [];
        let currentEquity = 10000;
        
        const now = new Date();
        for (let i = 0; i < points; i++) {
            const date = new Date(now);
            date.setDate(now.getDate() - (points - i));
            dates.push(date.toISOString().split('T')[0]);
            
            const dailyReturn = (Math.random() * 2 - 0.5) / 100; // -0.5% to 1.5%
            returns.push(dailyReturn);
            
            currentEquity *= (1 + dailyReturn);
            equity.push(currentEquity);
        }
        
        return {
            timestamp: Date.now(),
            timeframe: timeframe,
            equity: equity,
            dates: dates,
            returns: returns,
            sharpeRatio: 1.2 + Math.random() * 0.8,
            winRate: 55 + Math.random() * 15,
            profitFactor: 1.1 + Math.random() * 0.5
        };
    }
    
    /**
     * Generate mock system health data for testing
     * @param {object} params - Request parameters
     * @returns {object} Mock system health data
     */
    generateMockSystemHealthData(params) {
        return {
            timestamp: Date.now(),
            status: 'operational',
            cpu: 25 + Math.random() * 30,
            memory: 40 + Math.random() * 20,
            disk: 65 + Math.random() * 15,
            components: {
                dataCollection: { status: 'operational', latency: 50 + Math.random() * 100 },
                analysis: { status: 'operational', latency: 80 + Math.random() * 150 },
                execution: { status: 'operational', latency: 30 + Math.random() * 70 },
                dashboard: { status: 'operational', latency: 20 + Math.random() * 50 },
            }
        };
    }
    
    /**
     * Get number of data points based on timeframe
     * @param {string} timeframe - Timeframe code ('1d', '1w', '1m', etc.)
     * @returns {number} Number of data points
     */
    getTimeframePoints(timeframe) {
        switch (timeframe) {
            case '1d': return 24;
            case '1w': return 7;
            case '1m': return 30;
            case '3m': return 90;
            case '6m': return 180;
            case '1y': return 365;
            default: return 30;
        }
    }
}

// Usage example:
/*
const optimizer = new DashboardOptimizer();

// Efficient data loading
async function loadDashboardData() {
    try {
        // Get data efficiently with caching
        const marketData = await optimizer.getData('market-data', { symbols: ['BTC-USD', 'ETH-USD'] });
        const riskData = await optimizer.getData('risk-metrics', { timeframe: '1m' });
        
        // Use data for charts, etc.
        // ...
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Chunked rendering for large datasets
function renderTradesTable(trades) {
    const renderTrade = (trade) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${trade.id}</td>
            <td>${trade.symbol}</td>
            <td>${trade.price.toFixed(2)}</td>
            <td>${trade.quantity.toFixed(4)}</td>
            <td>${trade.side}</td>
        `;
        return row;
    };
    
    // Render trades in chunks
    optimizer.renderChunked('trades-table-body', trades, renderTrade);
}

// Lazy loading components
optimizer.lazyLoadComponent('risk-metrics-panel', () => {
    // Load and initialize risk metrics component
    const riskMetrics = new RiskAdjustedMetrics();
});
*/

// Export the optimizer class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardOptimizer;
} else if (typeof window !== 'undefined') {
    window.DashboardOptimizer = DashboardOptimizer;
}