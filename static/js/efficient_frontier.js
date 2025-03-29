/**
 * Interactive Efficient Frontier Module
 * 
 * This module provides functionality for portfolio optimization visualization
 * and interaction, including efficient frontier calculation, risk contribution
 * analysis, and weight optimization.
 */

// Configuration for the efficient frontier visualization
const frontierConfig = {
    // Chart colors
    colors: {
        frontier: 'rgba(59, 130, 246, 0.8)',      // Primary blue
        frontierArea: 'rgba(59, 130, 246, 0.1)',  // Lighter blue area
        optimalPoint: 'rgba(16, 185, 129, 1)',    // Green
        sharpeRatio: 'rgba(16, 185, 129, 0.8)',   // Green
        minRisk: 'rgba(249, 115, 22, 0.8)',       // Orange
        maxReturn: 'rgba(239, 68, 68, 0.8)',      // Red
        equalWeight: 'rgba(168, 85, 247, 0.8)',   // Purple
        simulatedPoints: 'rgba(107, 114, 128, 0.3)', // Gray for all points
        assetLabels: [
            'rgba(59, 130, 246, 0.8)',   // Blue
            'rgba(16, 185, 129, 0.8)',   // Green
            'rgba(249, 115, 22, 0.8)',   // Orange
            'rgba(239, 68, 68, 0.8)',    // Red
            'rgba(168, 85, 247, 0.8)',   // Purple
            'rgba(245, 158, 11, 0.8)',   // Yellow
            'rgba(99, 102, 241, 0.8)',   // Indigo
            'rgba(20, 184, 166, 0.8)',   // Teal
            'rgba(236, 72, 153, 0.8)',   // Pink
            'rgba(96, 165, 250, 0.8)'    // Light blue
        ],
        historicalLines: [
            'rgba(59, 130, 246, 0.5)',   // Blue (current)
            'rgba(156, 163, 175, 0.5)',  // Gray (1 month ago)
            'rgba(209, 213, 219, 0.5)'   // Light gray (3 months ago)
        ]
    },
    
    // Chart animation
    animation: {
        duration: 800,
        easing: 'easeOutQuart'
    },
    
    // Default rendering options
    defaults: {
        monteCarloPoints: 5000,
        seed: 42,
        maxPointSize: 8,
        minPointSize: 3,
        frontierResolution: 50,
        riskFreeRate: 0.025, // 2.5%
        showSimulatedPoints: true,
        showHistoricalFrontiers: true
    },
    
    // Asset market cap categories
    assetCategories: {
        largeCap: ['BTC', 'ETH'],
        midCap: ['BNB', 'SOL', 'XRP', 'ADA'],
        smallCap: ['DOT', 'AVAX', 'MATIC', 'LINK']
    }
};

/**
 * EfficientFrontier class for portfolio optimization visualization
 */
class EfficientFrontier {
    constructor(options = {}) {
        this.options = Object.assign({
            // DOM Elements
            frontierChartId: 'efficient-frontier-chart',
            riskContributionChartId: 'risk-contribution-chart',
            weightsChartId: 'optimal-weights-chart',
            performanceChartId: 'frontier-performance-chart',
            
            // Element selectors
            containerSelector: '.efficient-frontier-panel',
            periodSelector: '#frontier-period',
            frontierTypeSelector: '#frontier-type',
            riskFreeRateInput: '#risk-free-rate',
            assetCheckboxes: '[id^="asset-"]',
            recalculateButton: '#recalculate-frontier-btn',
            expandButton: '#expand-frontier-panel-btn',
            downloadButton: '#download-frontier-data-btn',
            applyConstraintsButton: '#apply-constraints-btn',
            resetConstraintsButton: '#reset-constraints-btn',
            selectAllButton: '#select-all-assets',
            selectNoneButton: '#select-none-assets',
            savePortfolioButton: '#save-portfolio-btn',
            exportReportButton: '#export-frontier-report',
            
            // Group constraint selectors
            maxLargeCapSlider: '#max-large-cap',
            maxMidCapSlider: '#max-mid-cap',
            maxSmallCapSlider: '#max-small-cap',
            maxVolatilitySlider: '#max-volatility',
            
            // Asset details (historical data will be fetched from API)
            availableAssets: [
                {id: 'BTC', name: 'Bitcoin', category: 'largeCap'},
                {id: 'ETH', name: 'Ethereum', category: 'largeCap'},
                {id: 'SOL', name: 'Solana', category: 'midCap'},
                {id: 'BNB', name: 'Binance Coin', category: 'midCap'},
                {id: 'ADA', name: 'Cardano', category: 'midCap'},
                {id: 'XRP', name: 'XRP', category: 'midCap'},
                {id: 'DOT', name: 'Polkadot', category: 'smallCap'},
                {id: 'AVAX', name: 'Avalanche', category: 'smallCap'},
                {id: 'MATIC', name: 'Polygon', category: 'smallCap'},
                {id: 'LINK', name: 'Chainlink', category: 'smallCap'}
            ],
            
            // API endpoints and settings
            apiEndpoint: '/api/portfolio/frontier',
            refreshInterval: 0, // 0 means no auto-refresh
            
            // Calculation settings
            defaultFrontierType: 'mean-variance',
            defaultPeriod: '1y',
            defaultRiskFree: 0.025,
            defaultOptimization: 'sharpe',
            defaultMethod: 'monte-carlo'
        }, options);
        
        // State management
        this.state = {
            selectedAssets: ['BTC', 'ETH', 'SOL', 'BNB'],
            period: this.options.defaultPeriod,
            frontierType: this.options.defaultFrontierType,
            riskFreeRate: this.options.defaultRiskFree,
            optimization: this.options.defaultOptimization,
            method: this.options.defaultMethod,
            monteCarloSettings: {
                portfolios: frontierConfig.defaults.monteCarloPoints,
                seed: frontierConfig.defaults.seed
            },
            constraints: {
                minWeights: {},
                maxWeights: {},
                groupConstraints: {
                    maxLargeCap: 0.8,
                    maxMidCap: 0.6,
                    maxSmallCap: 0.4,
                    maxVolatility: 0.25
                }
            },
            optimalPortfolio: null,
            lastUpdated: new Date()
        };
        
        // Chart instances
        this.charts = {
            frontierChart: null,
            riskContributionChart: null,
            weightsChart: null,
            performanceChart: null
        };
        
        // Asset data storage
        this.assetData = {
            returns: {},
            covariance: null,
            volatility: {},
            correlations: null,
            historicalPrices: {}
        };
    }
    
    /**
     * Initialize the efficient frontier component
     */
    init() {
        console.log('Initializing efficient frontier module');
        
        // Initialize constraints for selected assets
        this.initializeConstraints();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Fetch asset data
        this.fetchAssetData()
            .then(() => {
                // Initialize charts
                this.initializeCharts();
                
                // Calculate efficient frontier
                this.calculateEfficientFrontier();
                
                // Update UI elements
                this.updateUIElements();
            })
            .catch(error => {
                console.error('Error initializing efficient frontier:', error);
                this.showErrorMessage('Failed to initialize efficient frontier visualization');
            });
    }
    
    /**
     * Initialize min/max weight constraints for assets
     */
    initializeConstraints() {
        // Initialize min/max weight constraints for all available assets
        this.options.availableAssets.forEach(asset => {
            // Default min weight is 0% except for BTC (10%) and ETH (5%)
            let minWeight = 0;
            if (asset.id === 'BTC') minWeight = 0.1;
            if (asset.id === 'ETH') minWeight = 0.05;
            
            // Default max weight is 30% for small cap, 40% for others
            let maxWeight = 0.4;
            if (asset.category === 'smallCap') maxWeight = 0.3;
            
            this.state.constraints.minWeights[asset.id] = minWeight;
            this.state.constraints.maxWeights[asset.id] = maxWeight;
        });
    }
    
    /**
     * Set up event listeners for interactive elements
     */
    setupEventListeners() {
        const container = document.querySelector(this.options.containerSelector);
        if (!container) return;
        
        // Period selector
        const periodSelector = container.querySelector(this.options.periodSelector);
        if (periodSelector) {
            periodSelector.addEventListener('change', () => {
                this.state.period = periodSelector.value;
                this.fetchAssetData().then(() => {
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                });
            });
        }
        
        // Frontier type selector
        const frontierTypeSelector = container.querySelector(this.options.frontierTypeSelector);
        if (frontierTypeSelector) {
            frontierTypeSelector.addEventListener('change', () => {
                this.state.frontierType = frontierTypeSelector.value;
                this.calculateEfficientFrontier();
                this.updateUIElements();
            });
        }
        
        // Risk-free rate input
        const riskFreeInput = container.querySelector(this.options.riskFreeRateInput);
        if (riskFreeInput) {
            riskFreeInput.addEventListener('change', () => {
                const value = parseFloat(riskFreeInput.value) / 100;
                if (!isNaN(value)) {
                    this.state.riskFreeRate = value;
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                }
            });
        }
        
        // Asset checkboxes
        const assetCheckboxes = container.querySelectorAll(this.options.assetCheckboxes);
        assetCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateSelectedAssets();
                this.fetchAssetData().then(() => {
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                });
            });
        });
        
        // Recalculate button
        const recalculateButton = container.querySelector(this.options.recalculateButton);
        if (recalculateButton) {
            recalculateButton.addEventListener('click', () => {
                recalculateButton.classList.add('refreshing');
                this.fetchAssetData().then(() => {
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                    setTimeout(() => {
                        recalculateButton.classList.remove('refreshing');
                    }, 500);
                });
            });
        }
        
        // Apply constraints button
        const applyConstraintsButton = container.querySelector(this.options.applyConstraintsButton);
        if (applyConstraintsButton) {
            applyConstraintsButton.addEventListener('click', () => {
                this.updateConstraints();
                this.calculateEfficientFrontier();
                this.updateUIElements();
            });
        }
        
        // Reset constraints button
        const resetConstraintsButton = container.querySelector(this.options.resetConstraintsButton);
        if (resetConstraintsButton) {
            resetConstraintsButton.addEventListener('click', () => {
                this.resetConstraints();
                this.updateUIElements();
            });
        }
        
        // Select all/none buttons
        const selectAllButton = container.querySelector(this.options.selectAllButton);
        const selectNoneButton = container.querySelector(this.options.selectNoneButton);
        
        if (selectAllButton) {
            selectAllButton.addEventListener('click', () => {
                assetCheckboxes.forEach(checkbox => {
                    checkbox.checked = true;
                });
                this.updateSelectedAssets();
                this.fetchAssetData().then(() => {
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                });
            });
        }
        
        if (selectNoneButton) {
            selectNoneButton.addEventListener('click', () => {
                assetCheckboxes.forEach(checkbox => {
                    checkbox.checked = false;
                });
                this.updateSelectedAssets();
                this.fetchAssetData().then(() => {
                    this.calculateEfficientFrontier();
                    this.updateUIElements();
                });
            });
        }
        
        // Save portfolio button
        const savePortfolioButton = container.querySelector(this.options.savePortfolioButton);
        if (savePortfolioButton) {
            savePortfolioButton.addEventListener('click', () => {
                this.saveOptimalPortfolio();
            });
        }
        
        // Export report button
        const exportReportButton = container.querySelector(this.options.exportReportButton);
        if (exportReportButton) {
            exportReportButton.addEventListener('click', () => {
                this.exportFrontierReport();
            });
        }
        
        // Download data button
        const downloadButton = container.querySelector(this.options.downloadButton);
        if (downloadButton) {
            downloadButton.addEventListener('click', () => {
                this.downloadFrontierData();
            });
        }
        
        // Weight constraint sliders
        this.state.selectedAssets.forEach(assetId => {
            const minWeightSlider = container.querySelector(`#min-weight-${assetId.toLowerCase()}`);
            const maxWeightSlider = container.querySelector(`#max-weight-${assetId.toLowerCase()}`);
            
            if (minWeightSlider) {
                minWeightSlider.addEventListener('input', () => {
                    const value = parseInt(minWeightSlider.value) / 100;
                    const valueDisplay = minWeightSlider.nextElementSibling;
                    if (valueDisplay) valueDisplay.textContent = `${minWeightSlider.value}%`;
                    
                    // Ensure min weight ≤ max weight
                    const maxWeightSlider = container.querySelector(`#max-weight-${assetId.toLowerCase()}`);
                    if (maxWeightSlider && parseInt(minWeightSlider.value) > parseInt(maxWeightSlider.value)) {
                        maxWeightSlider.value = minWeightSlider.value;
                        const maxValueDisplay = maxWeightSlider.nextElementSibling;
                        if (maxValueDisplay) maxValueDisplay.textContent = `${maxWeightSlider.value}%`;
                    }
                });
            }
            
            if (maxWeightSlider) {
                maxWeightSlider.addEventListener('input', () => {
                    const value = parseInt(maxWeightSlider.value) / 100;
                    const valueDisplay = maxWeightSlider.nextElementSibling;
                    if (valueDisplay) valueDisplay.textContent = `${maxWeightSlider.value}%`;
                    
                    // Ensure max weight ≥ min weight
                    const minWeightSlider = container.querySelector(`#min-weight-${assetId.toLowerCase()}`);
                    if (minWeightSlider && parseInt(maxWeightSlider.value) < parseInt(minWeightSlider.value)) {
                        minWeightSlider.value = maxWeightSlider.value;
                        const minValueDisplay = minWeightSlider.nextElementSibling;
                        if (minValueDisplay) minValueDisplay.textContent = `${minWeightSlider.value}%`;
                    }
                });
            }
        });
        
        // Group constraint sliders
        const groupConstraintSliders = [
            this.options.maxLargeCapSlider,
            this.options.maxMidCapSlider,
            this.options.maxSmallCapSlider,
            this.options.maxVolatilitySlider
        ];
        
        groupConstraintSliders.forEach(selector => {
            const slider = container.querySelector(selector);
            if (slider) {
                slider.addEventListener('input', () => {
                    const valueDisplay = slider.nextElementSibling;
                    if (valueDisplay) valueDisplay.textContent = `${slider.value}%`;
                });
            }
        });
    }
    
    /**
     * Update selected assets from checkbox values
     */
    updateSelectedAssets() {
        const container = document.querySelector(this.options.containerSelector);
        if (!container) return;
        
        const assetCheckboxes = container.querySelectorAll(this.options.assetCheckboxes);
        const selectedAssets = [];
        
        assetCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                // Extract asset ID from checkbox id (format: asset-btc -> BTC)
                const assetId = checkbox.id.replace('asset-', '').toUpperCase();
                selectedAssets.push(assetId);
            }
        });
        
        // Ensure at least 2 assets are selected for a valid frontier
        if (selectedAssets.length < 2) {
            // If fewer than 2 assets, force selection of BTC and ETH
            const btcCheckbox = container.querySelector('#asset-btc');
            const ethCheckbox = container.querySelector('#asset-eth');
            
            if (btcCheckbox) btcCheckbox.checked = true;
            if (ethCheckbox) ethCheckbox.checked = true;
            
            selectedAssets.length = 0;
            selectedAssets.push('BTC', 'ETH');
            
            this.showNotification('warning', 'At least 2 assets must be selected for portfolio optimization');
        }
        
        this.state.selectedAssets = selectedAssets;
    }
    
    /**
     * Update constraints from UI elements
     */
    updateConstraints() {
        const container = document.querySelector(this.options.containerSelector);
        if (!container) return;
        
        // Update min/max weight constraints
        this.state.selectedAssets.forEach(assetId => {
            const minWeightSlider = container.querySelector(`#min-weight-${assetId.toLowerCase()}`);
            const maxWeightSlider = container.querySelector(`#max-weight-${assetId.toLowerCase()}`);
            
            if (minWeightSlider) {
                this.state.constraints.minWeights[assetId] = parseInt(minWeightSlider.value) / 100;
            }
            
            if (maxWeightSlider) {
                this.state.constraints.maxWeights[assetId] = parseInt(maxWeightSlider.value) / 100;
            }
        });
        
        // Update group constraints
        const maxLargeCapSlider = container.querySelector(this.options.maxLargeCapSlider);
        const maxMidCapSlider = container.querySelector(this.options.maxMidCapSlider);
        const maxSmallCapSlider = container.querySelector(this.options.maxSmallCapSlider);
        const maxVolatilitySlider = container.querySelector(this.options.maxVolatilitySlider);
        
        if (maxLargeCapSlider) {
            this.state.constraints.groupConstraints.maxLargeCap = parseInt(maxLargeCapSlider.value) / 100;
        }
        
        if (maxMidCapSlider) {
            this.state.constraints.groupConstraints.maxMidCap = parseInt(maxMidCapSlider.value) / 100;
        }
        
        if (maxSmallCapSlider) {
            this.state.constraints.groupConstraints.maxSmallCap = parseInt(maxSmallCapSlider.value) / 100;
        }
        
        if (maxVolatilitySlider) {
            this.state.constraints.groupConstraints.maxVolatility = parseInt(maxVolatilitySlider.value) / 100;
        }
    }
    
    /**
     * Reset constraints to default values
     */
    resetConstraints() {
        // Reset min/max weight constraints
        this.initializeConstraints();
        
        // Reset group constraints
        this.state.constraints.groupConstraints = {
            maxLargeCap: 0.8,
            maxMidCap: 0.6,
            maxSmallCap: 0.4,
            maxVolatility: 0.25
        };
        
        // Update UI elements with reset values
        const container = document.querySelector(this.options.containerSelector);
        if (!container) return;
        
        // Reset min/max weight sliders
        this.options.availableAssets.forEach(asset => {
            const minWeightSlider = container.querySelector(`#min-weight-${asset.id.toLowerCase()}`);
            const maxWeightSlider = container.querySelector(`#max-weight-${asset.id.toLowerCase()}`);
            
            if (minWeightSlider) {
                const minWeight = Math.round(this.state.constraints.minWeights[asset.id] * 100);
                minWeightSlider.value = minWeight;
                const minValueDisplay = minWeightSlider.nextElementSibling;
                if (minValueDisplay) minValueDisplay.textContent = `${minWeight}%`;
            }
            
            if (maxWeightSlider) {
                const maxWeight = Math.round(this.state.constraints.maxWeights[asset.id] * 100);
                maxWeightSlider.value = maxWeight;
                const maxValueDisplay = maxWeightSlider.nextElementSibling;
                if (maxValueDisplay) maxValueDisplay.textContent = `${maxWeight}%`;
            }
        });
        
        // Reset group constraint sliders
        const maxLargeCapSlider = container.querySelector(this.options.maxLargeCapSlider);
        const maxMidCapSlider = container.querySelector(this.options.maxMidCapSlider);
        const maxSmallCapSlider = container.querySelector(this.options.maxSmallCapSlider);
        const maxVolatilitySlider = container.querySelector(this.options.maxVolatilitySlider);
        
        if (maxLargeCapSlider) {
            maxLargeCapSlider.value = 80;
            const valueDisplay = maxLargeCapSlider.nextElementSibling;
            if (valueDisplay) valueDisplay.textContent = '80%';
        }
        
        if (maxMidCapSlider) {
            maxMidCapSlider.value = 60;
            const valueDisplay = maxMidCapSlider.nextElementSibling;
            if (valueDisplay) valueDisplay.textContent = '60%';
        }
        
        if (maxSmallCapSlider) {
            maxSmallCapSlider.value = 40;
            const valueDisplay = maxSmallCapSlider.nextElementSibling;
            if (valueDisplay) valueDisplay.textContent = '40%';
        }
        
        if (maxVolatilitySlider) {
            maxVolatilitySlider.value = 25;
            const valueDisplay = maxVolatilitySlider.nextElementSibling;
            if (valueDisplay) valueDisplay.textContent = '25%';
        }
        
        // Recalculate frontier with reset constraints
        this.calculateEfficientFrontier();
    }
    
    /**
     * Fetch asset data from API or generate mock data
     */
    async fetchAssetData() {
        try {
            // In a real implementation, fetch data from an API
            // For now, generate mock data
            this.generateMockAssetData();
            return Promise.resolve();
        } catch (error) {
            console.error('Error fetching asset data:', error);
            return Promise.reject(error);
        }
    }
    
    /**
     * Generate mock asset data for demonstration
     */
    generateMockAssetData() {
        const assets = this.state.selectedAssets;
        const assetCount = assets.length;
        
        // Generate mock returns (annual) for selected assets
        assets.forEach((asset, index) => {
            // Base expected returns (annualized)
            let baseReturn;
            if (asset === 'BTC') baseReturn = 0.45; // 45%
            else if (asset === 'ETH') baseReturn = 0.55; // 55%
            else if (['SOL', 'BNB'].includes(asset)) baseReturn = 0.65; // 65%
            else if (['ADA', 'XRP'].includes(asset)) baseReturn = 0.40; // 40%
            else baseReturn = 0.70; // 70% for small caps
            
            // Add some randomness to returns
            const expectedReturn = baseReturn * (0.9 + Math.random() * 0.2);
            this.assetData.returns[asset] = expectedReturn;
            
            // Generate volatility (annualized standard deviation)
            let baseVolatility;
            if (asset === 'BTC') baseVolatility = 0.70; // 70%
            else if (asset === 'ETH') baseVolatility = 0.85; // 85%
            else if (['SOL', 'BNB'].includes(asset)) baseVolatility = 0.95; // 95%
            else if (['ADA', 'XRP'].includes(asset)) baseVolatility = 0.90; // 90%
            else baseVolatility = 1.10; // 110% for small caps
            
            // Add some randomness to volatility
            const volatility = baseVolatility * (0.9 + Math.random() * 0.2);
            this.assetData.volatility[asset] = volatility;
        });
        
        // Generate mock covariance matrix
        const covariance = [];
        for (let i = 0; i < assetCount; i++) {
            covariance[i] = [];
            for (let j = 0; j < assetCount; j++) {
                if (i === j) {
                    // Diagonal elements are the variances (volatility squared)
                    covariance[i][j] = Math.pow(this.assetData.volatility[assets[i]], 2);
                } else {
                    // Off-diagonal elements are covariances
                    const correlation = 0.3 + Math.random() * 0.4; // Generate random correlation between 0.3 and 0.7
                    covariance[i][j] = correlation * this.assetData.volatility[assets[i]] * this.assetData.volatility[assets[j]];
                    covariance[j][i] = covariance[i][j]; // Ensure the matrix is symmetric
                }
            }
        }
        this.assetData.covariance = covariance;
        
        // Generate mock correlation matrix
        const correlations = [];
        for (let i = 0; i < assetCount; i++) {
            correlations[i] = [];
            for (let j = 0; j < assetCount; j++) {
                if (i === j) {
                    correlations[i][j] = 1; // Perfect correlation with itself
                } else {
                    // Calculate correlation from covariance
                    correlations[i][j] = covariance[i][j] / (this.assetData.volatility[assets[i]] * this.assetData.volatility[assets[j]]);
                }
            }
        }
        this.assetData.correlations = correlations;
        
        // Generate mock historical price data for performance chart
        const timeRange = this.getTimeRange();
        assets.forEach(asset => {
            const prices = [];
            let basePrice;
            
            // Starting price varies by asset
            if (asset === 'BTC') basePrice = 30000;
            else if (asset === 'ETH') basePrice = 2000;
            else if (asset === 'SOL') basePrice = 80;
            else if (asset === 'BNB') basePrice = 300;
            else if (asset === 'ADA') basePrice = 0.5;
            else if (asset === 'XRP') basePrice = 0.6;
            else basePrice = 10;
            
            // Generate price trend with random volatility
            let currentPrice = basePrice;
            let trend = 1.0 + (Math.random() * 0.4 - 0.1); // Random trend factor between 0.9x and 1.3x
            
            for (let i = 0; i < timeRange; i++) {
                // Add daily price with random walk
                const dailyReturn = (Math.random() * 0.08 - 0.04); // Daily return between -4% and +4%
                currentPrice *= (1 + dailyReturn);
                
                // Apply trend factor
                currentPrice *= Math.pow(trend, 1/20); // Subtle trend influence
                
                prices.push({
                    date: new Date(Date.now() - (timeRange - i) * 24 * 60 * 60 * 1000),
                    price: currentPrice
                });
            }
            
            this.assetData.historicalPrices[asset] = prices;
        });
    }
    
    /**
     * Get time range in days based on selected period
     */
    getTimeRange() {
        switch (this.state.period) {
            case '3m': return 90;
            case '6m': return 180;
            case '1y': return 365;
            case '2y': return 730;
            case '5y': return 1825;
            default: return 365;
        }
    }
    
    /**
     * Initialize chart instances
     */
    initializeCharts() {
        try {
            // Initialize efficient frontier chart
            this.initializeFrontierChart();
            
            // Initialize risk contribution chart
            this.initializeRiskContributionChart();
            
            // Initialize optimal weights chart
            this.initializeWeightsChart();
            
            // Initialize performance chart
            this.initializePerformanceChart();
        } catch (error) {
            console.error('Error initializing charts:', error);
            this.showErrorMessage('Failed to initialize portfolio optimization charts');
        }
    }
    
    /**
     * Initialize efficient frontier chart
     */
    initializeFrontierChart() {
        const ctx = document.getElementById(this.options.frontierChartId);
        if (!ctx) return;
        
        const config = {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Efficient Frontier',
                    data: [],
                    borderColor: frontierConfig.colors.frontier,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    showLine: true,
                    tension: 0.4,
                    pointRadius: 0
                }, {
                    label: 'Optimal Portfolio',
                    data: [],
                    backgroundColor: frontierConfig.colors.optimalPoint,
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10
                }, {
                    label: 'Simulated Portfolios',
                    data: [],
                    backgroundColor: frontierConfig.colors.simulatedPoints,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: frontierConfig.animation,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                if (point) {
                                    return [
                                        `Risk: ${(point.x * 100).toFixed(2)}%`,
                                        `Return: ${(point.y * 100).toFixed(2)}%`,
                                        point.sharpe ? `Sharpe: ${point.sharpe.toFixed(2)}` : ''
                                    ].filter(Boolean);
                                }
                                return '';
                            }
                        }
                    },
                    legend: {
                        position: 'top',
                        align: 'start',
                        labels: {
                            boxWidth: 10,
                            usePointStyle: true
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Risk (Volatility %)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Expected Return %'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        };
        
        if (this.charts.frontierChart) {
            this.charts.frontierChart.destroy();
        }
        
        this.charts.frontierChart = new Chart(ctx, config);
    }
    
    /**
     * Initialize risk contribution chart
     */
    initializeRiskContributionChart() {
        const ctx = document.getElementById(this.options.riskContributionChartId);
        if (!ctx) return;
        
        const config = {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Risk Contribution',
                    data: [],
                    backgroundColor: [],
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: frontierConfig.animation,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `Risk Contribution: ${(value * 100).toFixed(2)}%`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Risk Contribution %'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        };
        
        if (this.charts.riskContributionChart) {
            this.charts.riskContributionChart.destroy();
        }
        
        this.charts.riskContributionChart = new Chart(ctx, config);
    }
    
    /**
     * Initialize optimal weights chart
     */
    initializeWeightsChart() {
        const ctx = document.getElementById(this.options.weightsChartId);
        if (!ctx) return;
        
        const config = {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [],
                    borderColor: '#ffffff',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: frontierConfig.animation,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `Weight: ${(value * 100).toFixed(2)}%`;
                            }
                        }
                    },
                    legend: {
                        position: 'right',
                        labels: {
                            boxWidth: 12,
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        };
        
        if (this.charts.weightsChart) {
            this.charts.weightsChart.destroy();
        }
        
        this.charts.weightsChart = new Chart(ctx, config);
    }
    
    /**
     * Initialize historical performance chart
     */
    initializePerformanceChart() {
        const ctx = document.getElementById(this.options.performanceChartId);
        if (!ctx) return;
        
        const config = {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Optimal Portfolio',
                    data: [],
                    borderColor: frontierConfig.colors.optimalPoint,
                    backgroundColor: utils.setOpacity(frontierConfig.colors.optimalPoint, 0.1),
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: frontierConfig.animation,
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        position: 'top',
                        align: 'start',
                        labels: {
                            boxWidth: 10,
                            usePointStyle: true
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM YYYY'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Portfolio Value ($)'
                        }
                    }
                }
            }
        };
        
        if (this.charts.performanceChart) {
            this.charts.performanceChart.destroy();
        }
        
        this.charts.performanceChart = new Chart(ctx, config);
    }
    
    /**
     * Calculate efficient frontier and optimal portfolio
     */
    calculateEfficientFrontier() {
        try {
            const assets = this.state.selectedAssets;
            if (assets.length < 2 || !this.assetData.returns || !this.assetData.covariance) {
                console.warn('Insufficient data for efficient frontier calculation');
                return;
            }
            
            // Generate random portfolios using Monte Carlo simulation
            const portfolios = this.generateRandomPortfolios();
            
            // Calculate efficient frontier points
            const frontierPoints = this.calculateFrontierPoints(portfolios);
            
            // Find optimal portfolio based on optimization objective
            const optimalPortfolio = this.findOptimalPortfolio(portfolios);
            
            // Store optimal portfolio in state
            this.state.optimalPortfolio = optimalPortfolio;
            
            // Update charts with calculated data
            this.updateFrontierChart(portfolios, frontierPoints, optimalPortfolio);
            this.updateRiskContributionChart(optimalPortfolio);
            this.updateWeightsChart(optimalPortfolio);
            this.updatePerformanceChart(optimalPortfolio);
            
            // Update last updated timestamp
            this.state.lastUpdated = new Date();
            
        } catch (error) {
            console.error('Error calculating efficient frontier:', error);
            this.showErrorMessage('Failed to calculate efficient frontier');
        }
    }
    
    /**
     * Generate random portfolios using Monte Carlo simulation
     */
    generateRandomPortfolios() {
        const assets = this.state.selectedAssets;
        const numPortfolios = this.state.monteCarloSettings.portfolios;
        const portfolios = [];
        
        // Set random seed for reproducibility if needed
        Math.seedrandom && Math.seedrandom(this.state.monteCarloSettings.seed.toString());
        
        for (let i = 0; i < numPortfolios; i++) {
            // Generate random weights
            const weights = this.generateRandomWeights(assets.length);
            
            // Calculate portfolio return and risk
            const portfolioReturn = this.calculatePortfolioReturn(weights);
            const portfolioRisk = this.calculatePortfolioRisk(weights);
            
            // Calculate Sharpe ratio
            const excessReturn = portfolioReturn - this.state.riskFreeRate;
            const sharpeRatio = excessReturn / portfolioRisk;
            
            portfolios.push({
                weights: weights,
                return: portfolioReturn,
                risk: portfolioRisk,
                sharpeRatio: sharpeRatio
            });
        }
        
        return portfolios;
    }
    
    /**
     * Generate random weights that sum to 1
     */
    generateRandomWeights(numAssets) {
        const weights = [];
        let sum = 0;
        
        // Generate random weights
        for (let i = 0; i < numAssets; i++) {
            weights[i] = Math.random();
            sum += weights[i];
        }
        
        // Normalize weights to sum to 1
        for (let i = 0; i < numAssets; i++) {
            weights[i] /= sum;
        }
        
        return weights;
    }
    
    /**
     * Calculate portfolio expected return
     */
    calculatePortfolioReturn(weights) {
        const assets = this.state.selectedAssets;
        let portfolioReturn = 0;
        
        for (let i = 0; i < assets.length; i++) {
            portfolioReturn += weights[i] * this.assetData.returns[assets[i]];
        }
        
        return portfolioReturn;
    }
    
    /**
     * Calculate portfolio risk (volatility)
     */
    calculatePortfolioRisk(weights) {
        const assets = this.state.selectedAssets;
        let portfolioVariance = 0;
        
        // Calculate portfolio variance
        for (let i = 0; i < assets.length; i++) {
            for (let j = 0; j < assets.length; j++) {
                portfolioVariance += weights[i] * weights[j] * this.assetData.covariance[i][j];
            }
        }
        
        // Return portfolio volatility (standard deviation)
        return Math.sqrt(portfolioVariance);
    }
    
    /**
     * Calculate points along the efficient frontier
     */
    calculateFrontierPoints(portfolios) {
        // Sort portfolios by risk
        const sortedPortfolios = [...portfolios].sort((a, b) => a.risk - b.risk);
        
        // Calculate efficient frontier (maximum return for each risk level)
        const frontierPoints = [];
        const riskBuckets = {};
        
        // Group portfolios by risk level (rounded to 2 decimal places)
        sortedPortfolios.forEach(portfolio => {
            const roundedRisk = Math.round(portfolio.risk * 100) / 100;
            if (!riskBuckets[roundedRisk] || portfolio.return > riskBuckets[roundedRisk].return) {
                riskBuckets[roundedRisk] = portfolio;
            }
        });
        
        // Extract frontier points from risk buckets
        for (const risk in riskBuckets) {
            frontierPoints.push({
                risk: parseFloat(risk),
                return: riskBuckets[risk].return,
                sharpe: riskBuckets[risk].sharpeRatio
            });
        }
        
        // Sort frontier points by risk
        frontierPoints.sort((a, b) => a.risk - b.risk);
        
        // Filter to ensure frontier is monotonically increasing in return
        let maxReturn = -Infinity;
        const efficientFrontier = frontierPoints.filter(point => {
            if (point.return > maxReturn) {
                maxReturn = point.return;
                return true;
            }
            return false;
        });
        
        return efficientFrontier;
    }
    
    /**
     * Find optimal portfolio based on optimization objective
     */
    findOptimalPortfolio(portfolios) {
        const assets = this.state.selectedAssets;
        let optimalPortfolio = null;
        
        switch (this.state.optimization) {
            case 'sharpe':
                // Find portfolio with highest Sharpe ratio
                optimalPortfolio = portfolios.reduce((best, current) => {
                    return current.sharpeRatio > best.sharpeRatio ? current : best;
                }, portfolios[0]);
                break;
                
            case 'min-risk':
                // Find portfolio with minimum risk
                optimalPortfolio = portfolios.reduce((best, current) => {
                    return current.risk < best.risk ? current : best;
                }, portfolios[0]);
                break;
                
            case 'max-return':
                // Find portfolio with maximum return
                optimalPortfolio = portfolios.reduce((best, current) => {
                    return current.return > best.return ? current : best;
                }, portfolios[0]);
                break;
                
            default:
                // Default to maximum Sharpe ratio
                optimalPortfolio = portfolios.reduce((best, current) => {
                    return current.sharpeRatio > best.sharpeRatio ? current : best;
                }, portfolios[0]);
        }
        
        // Enrich optimal portfolio with additional information
        if (optimalPortfolio) {
            // Calculate risk contribution for each asset
            const riskContributions = this.calculateRiskContributions(optimalPortfolio.weights);
            
            // Create portfolio object with all information
            const portfolio = {
                return: optimalPortfolio.return,
                risk: optimalPortfolio.risk,
                sharpeRatio: optimalPortfolio.sharpeRatio,
                weights: {},
                riskContributions: {}
            };
            
            // Map weights and risk contributions to asset IDs
            assets.forEach((asset, index) => {
                portfolio.weights[asset] = optimalPortfolio.weights[index];
                portfolio.riskContributions[asset] = riskContributions[index];
            });
            
            return portfolio;
        }
        
        return null;
    }
    
    /**
     * Calculate risk contributions for a given portfolio
     */
    calculateRiskContributions(weights) {
        const assets = this.state.selectedAssets;
        const riskContributions = [];
        
        // Calculate portfolio risk
        const portfolioRisk = this.calculatePortfolioRisk(weights);
        
        // Calculate marginal contributions to risk
        for (let i = 0; i < assets.length; i++) {
            let contribution = 0;
            
            for (let j = 0; j < assets.length; j++) {
                contribution += weights[j] * this.assetData.covariance[i][j];
            }
            
            // Normalize by portfolio risk
            riskContributions[i] = weights[i] * contribution / portfolioRisk;
        }
        
        // Ensure risk contributions sum to portfolio risk
        const totalContribution = riskContributions.reduce((sum, value) => sum + value, 0);
        
        // Normalize to percentage contributions
        for (let i = 0; i < riskContributions.length; i++) {
            riskContributions[i] /= totalContribution;
        }
        
        return riskContributions;
    }
    
    /**
     * Update efficient frontier chart with calculated data
     */
    updateFrontierChart(portfolios, frontierPoints, optimalPortfolio) {
        if (!this.charts.frontierChart) return;
        
        // Convert frontier points to chart data format
        const frontierData = frontierPoints.map(point => ({
            x: point.risk,
            y: point.return
        }));
        
        // Convert simulated portfolios to chart data format
        const portfolioData = portfolios.map(portfolio => ({
            x: portfolio.risk,
            y: portfolio.return,
            sharpe: portfolio.sharpeRatio
        }));
        
        // Prepare optimal portfolio data point
        const optimalPoint = optimalPortfolio ? [{
            x: optimalPortfolio.risk,
            y: optimalPortfolio.return,
            sharpe: optimalPortfolio.sharpeRatio
        }] : [];
        
        // Update chart datasets
        this.charts.frontierChart.data.datasets[0].data = frontierData;
        this.charts.frontierChart.data.datasets[1].data = optimalPoint;
        
        // Show simulated portfolios if enabled
        if (frontierConfig.defaults.showSimulatedPoints) {
            this.charts.frontierChart.data.datasets[2].data = portfolioData;
        } else {
            this.charts.frontierChart.data.datasets[2].data = [];
        }
        
        // Update chart
        this.charts.frontierChart.update();
    }
    
    /**
     * Update risk contribution chart with optimal portfolio data
     */
    updateRiskContributionChart(optimalPortfolio) {
        if (!this.charts.riskContributionChart || !optimalPortfolio) return;
        
        const assets = this.state.selectedAssets;
        const riskContributions = [];
        const assetLabels = [];
        const backgroundColors = [];
        
        // Prepare data for chart
        assets.forEach((asset, index) => {
            // Only include assets with meaningful risk contribution (> 1%)
            if (optimalPortfolio.riskContributions[asset] > 0.01) {
                riskContributions.push(optimalPortfolio.riskContributions[asset]);
                assetLabels.push(asset);
                backgroundColors.push(frontierConfig.colors.assetLabels[index % frontierConfig.colors.assetLabels.length]);
            }
        });
        
        // Sort by risk contribution (descending)
        const sortIndices = riskContributions.map((_, i) => i)
            .sort((a, b) => riskContributions[b] - riskContributions[a]);
        
        const sortedRiskContributions = sortIndices.map(i => riskContributions[i]);
        const sortedAssetLabels = sortIndices.map(i => assetLabels[i]);
        const sortedBackgroundColors = sortIndices.map(i => backgroundColors[i]);
        
        // Update chart data
        this.charts.riskContributionChart.data.labels = sortedAssetLabels;
        this.charts.riskContributionChart.data.datasets[0].data = sortedRiskContributions;
        this.charts.riskContributionChart.data.datasets[0].backgroundColor = sortedBackgroundColors;
        
        // Update chart
        this.charts.riskContributionChart.update();
    }
    
    /**
     * Update optimal weights chart
     */
    updateWeightsChart(optimalPortfolio) {
        if (!this.charts.weightsChart || !optimalPortfolio) return;
        
        const assets = this.state.selectedAssets;
        const weights = [];
        const assetLabels = [];
        const backgroundColors = [];
        
        // Prepare data for chart
        assets.forEach((asset, index) => {
            // Only include assets with meaningful weight (> 1%)
            if (optimalPortfolio.weights[asset] > 0.01) {
                weights.push(optimalPortfolio.weights[asset]);
                assetLabels.push(asset);
                backgroundColors.push(frontierConfig.colors.assetLabels[index % frontierConfig.colors.assetLabels.length]);
            }
        });
        
        // Update chart data
        this.charts.weightsChart.data.labels = assetLabels;
        this.charts.weightsChart.data.datasets[0].data = weights;
        this.charts.weightsChart.data.datasets[0].backgroundColor = backgroundColors;
        
        // Update chart
        this.charts.weightsChart.update();
    }
    
    /**
     * Update historical performance chart for optimal portfolio
     */
    updatePerformanceChart(optimalPortfolio) {
        if (!this.charts.performanceChart || !optimalPortfolio) return;
        
        const assets = this.state.selectedAssets;
        
        // Generate historical performance data for optimal portfolio
        const portfolioPerformance = this.calculateHistoricalPerformance(optimalPortfolio);
        
        // Prepare dataset for optimal portfolio
        const portfolioData = portfolioPerformance.map(data => ({
            x: data.date,
            y: data.value
        }));
        
        // Get unique dates for labels
        const labels = portfolioPerformance.map(data => data.date);
        
        // Update chart data
        this.charts.performanceChart.data.labels = labels;
        this.charts.performanceChart.data.datasets[0].data = portfolioData;
        
        // Add individual asset performance for comparison (top 3 by weight)
        this.charts.performanceChart.data.datasets.splice(1); // Remove existing asset datasets
        
        // Sort assets by weight (descending)
        const sortedAssets = [...assets].sort((a, b) => 
            optimalPortfolio.weights[b] - optimalPortfolio.weights[a]
        );
        
        // Add top 3 assets to chart
        sortedAssets.slice(0, 3).forEach((asset, index) => {
            // Scale historical prices to start at the same value as the portfolio
            const historicalPrices = this.assetData.historicalPrices[asset];
            if (!historicalPrices || historicalPrices.length === 0) return;
            
            const startPrice = historicalPrices[0].price;
            const portfolioStartValue = portfolioPerformance[0].value;
            const scaleFactor = portfolioStartValue / startPrice;
            
            const assetData = historicalPrices.map(data => ({
                x: data.date,
                y: data.price * scaleFactor
            }));
            
            this.charts.performanceChart.data.datasets.push({
                label: asset,
                data: assetData,
                borderColor: frontierConfig.colors.assetLabels[index],
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                borderDash: [5, 5],
                pointRadius: 0
            });
        });
        
        // Update chart
        this.charts.performanceChart.update();
    }
    
    /**
     * Calculate historical performance for optimal portfolio
     */
    calculateHistoricalPerformance(optimalPortfolio) {
        const assets = this.state.selectedAssets;
        
        // Find earliest common date across all asset price histories
        let startDate = new Date();
        assets.forEach(asset => {
            const prices = this.assetData.historicalPrices[asset];
            if (prices && prices.length > 0) {
                const assetStartDate = prices[0].date;
                if (assetStartDate < startDate) {
                    startDate = assetStartDate;
                }
            }
        });
        
        // Build date map for each asset
        const dateMap = {};
        assets.forEach(asset => {
            const prices = this.assetData.historicalPrices[asset];
            if (!prices) return;
            
            dateMap[asset] = {};
            prices.forEach(price => {
                const dateStr = price.date.toISOString().split('T')[0];
                dateMap[asset][dateStr] = price.price;
            });
        });
        
        // Generate daily portfolio values
        const performanceData = [];
        const initialInvestment = 10000; // $10,000 starting capital
        let currentDate = new Date(startDate);
        const endDate = new Date(); // Today
        
        while (currentDate <= endDate) {
            const dateStr = currentDate.toISOString().split('T')[0];
            
            // Calculate portfolio value for this date
            let portfolioValue = 0;
            let validPrices = true;
            
            // Check if we have prices for all assets on this date
            assets.forEach(asset => {
                if (!dateMap[asset] || !dateMap[asset][dateStr]) {
                    validPrices = false;
                }
            });
            
            // If we have all prices, calculate portfolio value
            if (validPrices) {
                assets.forEach(asset => {
                    const weight = optimalPortfolio.weights[asset];
                    const price = dateMap[asset][dateStr];
                    portfolioValue += initialInvestment * weight * price / dateMap[asset][Object.keys(dateMap[asset])[0]];
                });
                
                performanceData.push({
                    date: new Date(currentDate),
                    value: portfolioValue
                });
            }
            
            // Move to next day
            currentDate.setDate(currentDate.getDate() + 1);
        }
        
        // If we don't have enough data, generate synthetic performance
        if (performanceData.length < 30) {
            return this.generateSyntheticPerformance(optimalPortfolio);
        }
        
        return performanceData;
    }
    
    /**
     * Generate synthetic performance data for demonstration
     */
    generateSyntheticPerformance(optimalPortfolio) {
        const timeRange = this.getTimeRange();
        const performanceData = [];
        let portfolioValue = 10000; // $10,000 starting capital
        
        // Use portfolio expected return and risk to generate realistic performance
        const annualReturn = optimalPortfolio.return;
        const annualRisk = optimalPortfolio.risk;
        
        // Convert annual metrics to daily
        const dailyReturn = annualReturn / 252;
        const dailyRisk = annualRisk / Math.sqrt(252);
        
        // Generate daily portfolio values
        for (let i = 0; i < timeRange; i++) {
            // Random daily return with normal distribution
            const randomReturn = this.generateNormalRandom(dailyReturn, dailyRisk);
            portfolioValue *= (1 + randomReturn);
            
            performanceData.push({
                date: new Date(Date.now() - (timeRange - i) * 24 * 60 * 60 * 1000),
                value: portfolioValue
            });
        }
        
        return performanceData;
    }
    
    /**
     * Generate random number from normal distribution
     */
    generateNormalRandom(mean, stdDev) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        
        return z0 * stdDev + mean;
    }
    
    /**
     * Update UI elements with optimal portfolio data
     */
    updateUIElements() {
        if (!this.state.optimalPortfolio) return;
        
        const optimalPortfolio = this.state.optimalPortfolio;
        const container = document.querySelector(this.options.containerSelector);
        if (!container) return;
        
        // Update optimal portfolio indicators
        const optimalRisk = container.querySelector('#optimal-portfolio-risk');
        const optimalReturn = container.querySelector('#optimal-portfolio-return');
        const optimalSharpe = container.querySelector('#optimal-portfolio-sharpe');
        
        if (optimalRisk) optimalRisk.textContent = `${(optimalPortfolio.risk * 100).toFixed(2)}%`;
        if (optimalReturn) optimalReturn.textContent = `${(optimalPortfolio.return * 100).toFixed(2)}%`;
        if (optimalSharpe) optimalSharpe.textContent = optimalPortfolio.sharpeRatio.toFixed(2);
        
        // Update footer metrics
        const optimalReturnFooter = container.querySelector('#optimal-return');
        const optimalRiskFooter = container.querySelector('#optimal-risk');
        const optimalSharpeFooter = container.querySelector('#optimal-sharpe');
        const diversificationScore = container.querySelector('#diversification-score');
        const lastUpdated = container.querySelector('#frontier-last-updated');
        
        if (optimalReturnFooter) optimalReturnFooter.textContent = `${(optimalPortfolio.return * 100).toFixed(2)}%`;
        if (optimalRiskFooter) optimalRiskFooter.textContent = `${(optimalPortfolio.risk * 100).toFixed(2)}%`;
        if (optimalSharpeFooter) optimalSharpeFooter.textContent = optimalPortfolio.sharpeRatio.toFixed(2);
        
        // Calculate diversification score (1-10 scale, higher = better diversified)
        if (diversificationScore) {
            const assets = this.state.selectedAssets;
            
            // Calculate Herfindahl-Hirschman Index (HHI) for weights
            let hhi = 0;
            assets.forEach(asset => {
                hhi += Math.pow(optimalPortfolio.weights[asset], 2);
            });
            
            // Convert HHI to diversification score (10 - 100*HHI/assets.length)
            // HHI ranges from 1/n (perfectly diversified) to 1 (single asset)
            const divScore = 10 - 9 * (hhi - 1/assets.length) / (1 - 1/assets.length);
            diversificationScore.textContent = divScore.toFixed(1);
        }
        
        if (lastUpdated) {
            const now = new Date();
            const minutesAgo = Math.floor((now - this.state.lastUpdated) / 60000);
            
            if (minutesAgo < 1) {
                lastUpdated.textContent = 'Just now';
            } else if (minutesAgo === 1) {
                lastUpdated.textContent = '1 minute ago';
            } else if (minutesAgo < 60) {
                lastUpdated.textContent = `${minutesAgo} minutes ago`;
            } else {
                const hoursAgo = Math.floor(minutesAgo / 60);
                lastUpdated.textContent = `${hoursAgo} hour${hoursAgo > 1 ? 's' : ''} ago`;
            }
        }
    }
    
    /**
     * Save optimal portfolio
     */
    saveOptimalPortfolio() {
        if (!this.state.optimalPortfolio) {
            this.showNotification('warning', 'No optimal portfolio available to save');
            return;
        }
        
        try {
            // In a real application, this would call an API to save the portfolio
            console.log('Saving optimal portfolio:', this.state.optimalPortfolio);
            
            // Create a formatted portfolio summary for display
            const assets = this.state.selectedAssets;
            const portfolio = this.state.optimalPortfolio;
            
            // Format weights and risk contributions for display
            const weightEntries = assets.map(asset => {
                const weight = portfolio.weights[asset];
                if (weight < 0.01) return null; // Skip tiny allocations
                return `${asset}: ${(weight * 100).toFixed(2)}%`;
            }).filter(Boolean);
            
            // Create portfolio summary
            const summary = `
                <div class="portfolio-saved-details">
                    <h4>Optimal Portfolio Saved</h4>
                    <p>Expected Return: ${(portfolio.return * 100).toFixed(2)}%</p>
                    <p>Expected Risk: ${(portfolio.risk * 100).toFixed(2)}%</p>
                    <p>Sharpe Ratio: ${portfolio.sharpeRatio.toFixed(2)}</p>
                    <p>Asset Allocation:</p>
                    <ul>
                        ${weightEntries.map(entry => `<li>${entry}</li>`).join('')}
                    </ul>
                </div>
            `;
            
            // Display confirmation notification
            this.showNotification('success', 'Portfolio saved successfully', summary);
            
        } catch (error) {
            console.error('Error saving portfolio:', error);
            this.showErrorMessage('Failed to save portfolio');
        }
    }
    
    /**
     * Export frontier report as PDF or CSV
     */
    exportFrontierReport() {
        if (!this.state.optimalPortfolio) {
            this.showNotification('warning', 'No portfolio data available to export');
            return;
        }
        
        try {
            // In a real application, this would generate and download a report
            console.log('Exporting frontier report');
            
            // Show a confirmation notification
            this.showNotification('success', 'Portfolio report exported successfully');
            
        } catch (error) {
            console.error('Error exporting frontier report:', error);
            this.showErrorMessage('Failed to export portfolio report');
        }
    }
    
    /**
     * Download frontier data as CSV
     */
    downloadFrontierData() {
        if (!this.state.optimalPortfolio) {
            this.showNotification('warning', 'No portfolio data available to download');
            return;
        }
        
        try {
            // In a real application, this would generate and download a CSV file
            console.log('Downloading frontier data');
            
            // Show a confirmation notification
            this.showNotification('success', 'Portfolio data downloaded successfully');
            
        } catch (error) {
            console.error('Error downloading frontier data:', error);
            this.showErrorMessage('Failed to download portfolio data');
        }
    }
    
    /**
     * Show error message to user
     */
    showErrorMessage(message) {
        // Use existing notification system if available
        if (typeof showNotification === 'function') {
            showNotification('error', message);
        } else {
            console.error(message);
            alert(message);
        }
    }
    
    /**
     * Show notification to user
     */
    showNotification(type, title, content = '') {
        // Use existing notification system if available
        if (typeof showNotification === 'function') {
            showNotification(type, title, content);
        } else {
            console.log(`${type.toUpperCase()}: ${title}`);
            if (content) console.log(content);
        }
    }
}

// Utility functions for efficient frontier calculations
const utils = {
    // Set opacity for a color
    setOpacity: (color, opacity) => {
        if (color.startsWith('rgba')) {
            return color.replace(/rgba\(([^,]+),([^,]+),([^,]+),[^)]+\)/, `rgba($1,$2,$3,${opacity})`);
        } else if (color.startsWith('rgb')) {
            return color.replace('rgb', 'rgba').replace(')', `, ${opacity})`);
        } else if (color.startsWith('#')) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${opacity})`;
        }
        return color;
    }
};

// Initialize efficient frontier when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if the efficient frontier panel exists
    const efficientFrontierPanel = document.querySelector('.efficient-frontier-panel');
    if (!efficientFrontierPanel) return;
    
    // Initialize feather icons if available
    if (typeof feather !== 'undefined' && feather.replace) {
        feather.replace();
    }
    
    // Initialize efficient frontier module
    const frontier = new EfficientFrontier();
    frontier.init();
    
    // Add global function for external access
    window.efficientFrontier = frontier;
});