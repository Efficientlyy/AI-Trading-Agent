/**
 * Algorithm Decision Explainability
 * 
 * This module provides functionality for the Algorithm Decision Explainability component,
 * showing how trading algorithms make decisions through factor contribution analysis,
 * feature importance, decision paths, and prediction vs. actual comparisons.
 */

class AlgorithmExplainability {
    constructor(options = {}) {
        this.options = Object.assign({
            factorContributionElementId: 'factor-contribution-chart',
            featureImportanceElementId: 'feature-importance-chart',
            decisionPathElementId: 'decision-path-chart',
            predictionActualElementId: 'prediction-actual-chart',
            colorScaleContribution: [
                [0, 'rgba(239, 68, 68, 0.7)'],     // Negative factor (red)
                [0.5, 'rgba(209, 213, 219, 0.7)'],  // Neutral (gray)
                [1, 'rgba(16, 185, 129, 0.7)']      // Positive factor (green)
            ],
            colorScaleImportance: [
                [0, 'rgba(59, 130, 246, 0.9)'],    // Low importance (blue)
                [1, 'rgba(239, 68, 68, 0.9)']      // High importance (red)
            ]
        }, options);
        
        this.algorithm = document.getElementById('explainability-algorithm')?.value || 'ml_strategy';
        this.timeframe = document.getElementById('explainability-timeframe')?.value || '1d';
        this.signal = document.getElementById('explainability-signal')?.value || 'last';
        this.datetime = document.getElementById('explainability-datetime')?.value || '';
        
        this.contributionView = document.getElementById('contribution-view')?.value || 'waterfall';
        this.normalize = document.getElementById('normalize-factors')?.value || 'percentage';
        this.importanceType = document.getElementById('importance-type')?.value || 'global';
        this.featureCount = document.getElementById('feature-count')?.value || '10';
        this.pathVisualization = document.getElementById('path-visualization')?.value || 'tree';
        this.pathDetail = document.getElementById('path-detail')?.value || 'medium';
        this.predictionMetric = document.getElementById('prediction-metric')?.value || 'price';
        this.comparisonPeriod = document.getElementById('comparison-period')?.value || '3d';
        
        this.factorData = [];
        this.featureImportanceData = [];
        this.decisionPathData = [];
        this.predictionActualData = [];
        this.decisionDetails = {};
        
        this.factorChart = null;
        this.importanceChart = null;
        this.pathChart = null;
        this.predictionChart = null;
        
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
                console.error('Error initializing Algorithm Explainability:', error);
            });
    }
    
    fetchData() {
        // In a real implementation, this would fetch data from an API
        return Promise.all([
            this.fetchFactorData(),
            this.fetchFeatureImportanceData(),
            this.fetchDecisionPathData(),
            this.fetchPredictionActualData(),
            this.fetchDecisionDetails()
        ]);
    }
    
    fetchFactorData() {
        // Mock data for factor contribution
        return new Promise(resolve => {
            setTimeout(() => {
                let factors = [];
                
                // Generate different factors based on algorithm
                if (this.algorithm === 'ml_strategy') {
                    factors = [
                        { name: 'Technical Indicators', contribution: 0.28, description: 'Moving averages, RSI, MACD' },
                        { name: 'Volume Profile', contribution: 0.15, description: 'Volume patterns and anomalies' },
                        { name: 'Market Sentiment', contribution: 0.45, description: 'Social media and news sentiment' },
                        { name: 'On-chain Metrics', contribution: -0.08, description: 'Blockchain transaction data' },
                        { name: 'Market Regime', contribution: 0.32, description: 'Current market conditions' },
                        { name: 'Momentum Oscillators', contribution: -0.12, description: 'Overbought/oversold indicators' },
                        { name: 'Baseline Signal', contribution: 0.08, description: 'Base prediction value' }
                    ];
                } else if (this.algorithm === 'sentiment_strategy') {
                    factors = [
                        { name: 'Social Media Sentiment', contribution: 0.52, description: 'Twitter, Reddit, Discord' },
                        { name: 'News Sentiment', contribution: 0.18, description: 'Crypto news articles' },
                        { name: 'Market Events', contribution: 0.15, description: 'Known market events' },
                        { name: 'Sentiment Momentum', contribution: 0.22, description: 'Change in sentiment over time' },
                        { name: 'Outlier Detection', contribution: -0.05, description: 'Unusual sentiment patterns' },
                        { name: 'Technical Confirmation', contribution: 0.12, description: 'Technical indicator confirmation' },
                        { name: 'Baseline Signal', contribution: 0.05, description: 'Base prediction value' }
                    ];
                } else if (this.algorithm === 'meta_strategy') {
                    factors = [
                        { name: 'Technical Strategy', contribution: 0.35, description: 'Technical analysis signals' },
                        { name: 'ML Strategy', contribution: 0.28, description: 'Machine learning predictions' },
                        { name: 'Sentiment Strategy', contribution: 0.22, description: 'Sentiment-based signals' },
                        { name: 'Order Flow Strategy', contribution: -0.08, description: 'Order book dynamics' },
                        { name: 'Volatility Strategy', contribution: 0.15, description: 'Volatility-based signals' },
                        { name: 'Strategy Correlation', contribution: -0.12, description: 'Correlation penalty' },
                        { name: 'Baseline Signal', contribution: 0.08, description: 'Base prediction value' }
                    ];
                } else if (this.algorithm === 'technical_strategy') {
                    factors = [
                        { name: 'Moving Averages', contribution: 0.45, description: '50/200 day crossover' },
                        { name: 'RSI', contribution: 0.28, description: 'Relative Strength Index' },
                        { name: 'MACD', contribution: 0.18, description: 'Moving Average Convergence Divergence' },
                        { name: 'Bollinger Bands', contribution: -0.05, description: 'Price volatility bands' },
                        { name: 'Fibonacci Levels', contribution: 0.12, description: 'Fibonacci retracement' },
                        { name: 'Chart Patterns', contribution: 0.22, description: 'Identified chart patterns' },
                        { name: 'Volume Profile', contribution: 0.15, description: 'Volume analysis' }
                    ];
                }
                
                this.factorData = factors;
                resolve(factors);
            }, 300);
        });
    }
    
    fetchFeatureImportanceData() {
        // Mock data for feature importance
        return new Promise(resolve => {
            setTimeout(() => {
                let features = [];
                
                // Generate different features based on algorithm
                if (this.algorithm === 'ml_strategy') {
                    features = [
                        { name: 'Price_MA_50', importance: 0.18 },
                        { name: 'Volume_MA_20', importance: 0.15 },
                        { name: 'RSI_14', importance: 0.12 },
                        { name: 'Social_Sentiment_Score', importance: 0.11 },
                        { name: 'MACD_Signal', importance: 0.09 },
                        { name: 'Volatility_20', importance: 0.08 },
                        { name: 'News_Sentiment_Score', importance: 0.07 },
                        { name: 'Market_Regime', importance: 0.06 },
                        { name: 'OBV_Change', importance: 0.05 },
                        { name: 'Funding_Rate', importance: 0.04 },
                        { name: 'Correlation_SPX', importance: 0.03 },
                        { name: 'On_Chain_Activity', importance: 0.02 }
                    ];
                } else if (this.algorithm === 'sentiment_strategy') {
                    features = [
                        { name: 'Twitter_Sentiment', importance: 0.22 },
                        { name: 'Reddit_Sentiment', importance: 0.18 },
                        { name: 'News_Headline_Score', importance: 0.15 },
                        { name: 'Discord_Activity', importance: 0.12 },
                        { name: 'Github_Activity', importance: 0.10 },
                        { name: 'Sentiment_Change_Rate', importance: 0.08 },
                        { name: 'Telegram_Sentiment', importance: 0.05 },
                        { name: 'Google_Trends', importance: 0.04 },
                        { name: 'Influencer_Score', importance: 0.03 },
                        { name: 'Event_Proximity', importance: 0.03 }
                    ];
                } else if (this.algorithm === 'meta_strategy') {
                    features = [
                        { name: 'Technical_Signal', importance: 0.20 },
                        { name: 'ML_Prediction', importance: 0.18 },
                        { name: 'Sentiment_Score', importance: 0.15 },
                        { name: 'Regime_State', importance: 0.12 },
                        { name: 'Volatility_Signal', importance: 0.10 },
                        { name: 'Order_Flow_Signal', importance: 0.08 },
                        { name: 'Correlation_Factor', importance: 0.07 },
                        { name: 'Performance_Weight', importance: 0.05 },
                        { name: 'Drawdown_Protection', importance: 0.03 },
                        { name: 'Strategy_Consistency', importance: 0.02 }
                    ];
                } else if (this.algorithm === 'technical_strategy') {
                    features = [
                        { name: 'MA_Crossover_50_200', importance: 0.25 },
                        { name: 'RSI_14', importance: 0.18 },
                        { name: 'MACD_Signal', importance: 0.15 },
                        { name: 'Volume_Change', importance: 0.12 },
                        { name: 'Bollinger_Position', importance: 0.10 },
                        { name: 'ATR_14', importance: 0.08 },
                        { name: 'Fibonacci_Level', importance: 0.05 },
                        { name: 'Support_Resistance', importance: 0.03 },
                        { name: 'Pattern_Recognition', importance: 0.02 },
                        { name: 'ADX_14', importance: 0.02 }
                    ];
                }
                
                this.featureImportanceData = features;
                resolve(features);
            }, 350);
        });
    }
    
    fetchDecisionPathData() {
        // Mock data for decision path
        return new Promise(resolve => {
            setTimeout(() => {
                let decisionPath = {};
                
                // Generate different decision paths based on algorithm
                if (this.algorithm === 'ml_strategy') {
                    // For ML strategy, create a simple decision tree with nodes and links
                    decisionPath = {
                        tree: {
                            nodes: [
                                { id: 0, label: 'Start', level: 0, type: 'root' },
                                { id: 1, label: 'Social Sentiment > 0.6?', level: 1, type: 'decision' },
                                { id: 2, label: 'Technical Score > 0.5?', level: 2, type: 'decision' },
                                { id: 3, label: 'Volume Increasing?', level: 3, type: 'decision' },
                                { id: 4, label: 'Market Regime Bullish?', level: 2, type: 'decision' },
                                { id: 5, label: 'Buy Signal', level: 3, type: 'output', result: 'buy' },
                                { id: 6, label: 'Weak Buy Signal', level: 4, type: 'output', result: 'weak_buy' },
                                { id: 7, label: 'Hold Signal', level: 3, type: 'output', result: 'hold' },
                                { id: 8, label: 'Sell Signal', level: 4, type: 'output', result: 'sell' }
                            ],
                            links: [
                                { source: 0, target: 1, label: 'Start', value: 1 },
                                { source: 1, target: 2, label: 'Yes', value: 0.8 },
                                { source: 1, target: 4, label: 'No', value: 0.2 },
                                { source: 2, target: 3, label: 'Yes', value: 0.7 },
                                { source: 2, target: 7, label: 'No', value: 0.3 },
                                { source: 3, target: 5, label: 'Yes', value: 0.9 },
                                { source: 3, target: 6, label: 'No', value: 0.1 },
                                { source: 4, target: 7, label: 'Yes', value: 0.6 },
                                { source: 4, target: 8, label: 'No', value: 0.4 }
                            ],
                            activePath: [0, 1, 2, 3, 5] // The actual path taken for this decision
                        },
                        sankey: {
                            nodes: [
                                { name: 'Input' },
                                { name: 'Technical Layer' },
                                { name: 'Sentiment Layer' },
                                { name: 'Market Conditions' },
                                { name: 'Signal Generation' },
                                { name: 'Buy' },
                                { name: 'Hold' },
                                { name: 'Sell' }
                            ],
                            links: [
                                { source: 0, target: 1, value: 0.4 },
                                { source: 0, target: 2, value: 0.4 },
                                { source: 0, target: 3, value: 0.2 },
                                { source: 1, target: 4, value: 0.3 },
                                { source: 2, target: 4, value: 0.35 },
                                { source: 3, target: 4, value: 0.15 },
                                { source: 4, target: 5, value: 0.5 },
                                { source: 4, target: 6, value: 0.3 },
                                { source: 4, target: 7, value: 0.2 }
                            ]
                        }
                    };
                } else if (this.algorithm === 'sentiment_strategy') {
                    // For sentiment strategy, create a different tree structure
                    decisionPath = {
                        tree: {
                            nodes: [
                                { id: 0, label: 'Start', level: 0, type: 'root' },
                                { id: 1, label: 'Social Media Sentiment', level: 1, type: 'process' },
                                { id: 2, label: 'News Sentiment', level: 1, type: 'process' },
                                { id: 3, label: 'Combined Sentiment > 0.7?', level: 2, type: 'decision' },
                                { id: 4, label: 'Sentiment Change Positive?', level: 3, type: 'decision' },
                                { id: 5, label: 'Technical Confirmation?', level: 3, type: 'decision' },
                                { id: 6, label: 'Strong Buy Signal', level: 4, type: 'output', result: 'strong_buy' },
                                { id: 7, label: 'Buy Signal', level: 4, type: 'output', result: 'buy' },
                                { id: 8, label: 'Hold Signal', level: 3, type: 'output', result: 'hold' },
                                { id: 9, label: 'Sell Signal', level: 4, type: 'output', result: 'sell' }
                            ],
                            links: [
                                { source: 0, target: 1, label: 'Start', value: 0.6 },
                                { source: 0, target: 2, label: 'Start', value: 0.4 },
                                { source: 1, target: 3, label: 'Process', value: 0.6 },
                                { source: 2, target: 3, label: 'Process', value: 0.4 },
                                { source: 3, target: 4, label: 'Yes', value: 0.8 },
                                { source: 3, target: 5, label: 'No', value: 0.2 },
                                { source: 4, target: 6, label: 'Yes', value: 0.7 },
                                { source: 4, target: 7, label: 'No', value: 0.3 },
                                { source: 5, target: 8, label: 'Yes', value: 0.6 },
                                { source: 5, target: 9, label: 'No', value: 0.4 }
                            ],
                            activePath: [0, 1, 3, 4, 6] // The actual path taken for this decision
                        },
                        sankey: {
                            nodes: [
                                { name: 'Social Media' },
                                { name: 'News' },
                                { name: 'Sentiment Analysis' },
                                { name: 'Momentum Analysis' },
                                { name: 'Signal Generation' },
                                { name: 'Buy' },
                                { name: 'Hold' },
                                { name: 'Sell' }
                            ],
                            links: [
                                { source: 0, target: 2, value: 0.6 },
                                { source: 1, target: 2, value: 0.4 },
                                { source: 2, target: 3, value: 0.7 },
                                { source: 2, target: 4, value: 0.3 },
                                { source: 3, target: 4, value: 0.7 },
                                { source: 4, target: 5, value: 0.6 },
                                { source: 4, target: 6, value: 0.3 },
                                { source: 4, target: 7, value: 0.1 }
                            ]
                        }
                    };
                } else {
                    // Default tree structure for other algorithms
                    decisionPath = {
                        tree: {
                            nodes: [
                                { id: 0, label: 'Start', level: 0, type: 'root' },
                                { id: 1, label: 'Process Signals', level: 1, type: 'process' },
                                { id: 2, label: 'Signal Strength > 0.7?', level: 2, type: 'decision' },
                                { id: 3, label: 'Market Conditions Favorable?', level: 3, type: 'decision' },
                                { id: 4, label: 'Risk Check Passed?', level: 3, type: 'decision' },
                                { id: 5, label: 'Strong Buy Signal', level: 4, type: 'output', result: 'strong_buy' },
                                { id: 6, label: 'Buy Signal', level: 4, type: 'output', result: 'buy' },
                                { id: 7, label: 'Hold Signal', level: 4, type: 'output', result: 'hold' },
                                { id: 8, label: 'Sell Signal', level: 4, type: 'output', result: 'sell' }
                            ],
                            links: [
                                { source: 0, target: 1, label: 'Start', value: 1 },
                                { source: 1, target: 2, label: 'Process', value: 1 },
                                { source: 2, target: 3, label: 'Yes', value: 0.7 },
                                { source: 2, target: 4, label: 'No', value: 0.3 },
                                { source: 3, target: 5, label: 'Yes', value: 0.8 },
                                { source: 3, target: 6, label: 'No', value: 0.2 },
                                { source: 4, target: 7, label: 'Yes', value: 0.6 },
                                { source: 4, target: 8, label: 'No', value: 0.4 }
                            ],
                            activePath: [0, 1, 2, 3, 5] // The actual path taken for this decision
                        },
                        sankey: {
                            nodes: [
                                { name: 'Input Signals' },
                                { name: 'Signal Processing' },
                                { name: 'Risk Assessment' },
                                { name: 'Signal Generation' },
                                { name: 'Buy' },
                                { name: 'Hold' },
                                { name: 'Sell' }
                            ],
                            links: [
                                { source: 0, target: 1, value: 1.0 },
                                { source: 1, target: 2, value: 0.6 },
                                { source: 1, target: 3, value: 0.4 },
                                { source: 2, target: 3, value: 0.6 },
                                { source: 3, target: 4, value: 0.5 },
                                { source: 3, target: 5, value: 0.3 },
                                { source: 3, target: 6, value: 0.2 }
                            ]
                        }
                    };
                }
                
                this.decisionPathData = decisionPath;
                resolve(decisionPath);
            }, 400);
        });
    }
    
    fetchPredictionActualData() {
        // Mock data for prediction vs. actual
        return new Promise(resolve => {
            setTimeout(() => {
                // Generate dates for the comparison period
                const days = this.comparisonPeriod === '1d' ? 1 : 
                             this.comparisonPeriod === '3d' ? 3 : 
                             this.comparisonPeriod === '1w' ? 7 : 30;
                
                const currentDate = new Date();
                const dates = [];
                const actualValues = [];
                const predictedValues = [];
                const upperBound = [];
                const lowerBound = [];
                
                // Generate some realistic price movement data
                let baseValue = 50000;
                let lastActual = baseValue;
                let trend = Math.random() < 0.6 ? 1 : -1; // 60% chance of uptrend
                
                for (let i = 0; i <= days; i++) {
                    const date = new Date(currentDate);
                    date.setDate(date.getDate() - days + i);
                    dates.push(date);
                    
                    if (i === 0) {
                        // Starting point
                        actualValues.push(baseValue);
                        predictedValues.push(null); // No prediction for starting point
                        upperBound.push(null);
                        lowerBound.push(null);
                    } else {
                        // Generate actual value with some randomness and trend
                        const volatility = baseValue * 0.01; // 1% volatility
                        const change = (Math.random() * volatility * 2 - volatility) + (trend * volatility * 0.5);
                        lastActual = lastActual + change;
                        actualValues.push(lastActual);
                        
                        if (i === 1) {
                            // First prediction point
                            const predictedChange = trend * volatility * 1.2; // Slightly optimistic/pessimistic
                            const predicted = baseValue + predictedChange;
                            predictedValues.push(predicted);
                            upperBound.push(predicted * 1.02); // 2% upper bound
                            lowerBound.push(predicted * 0.98); // 2% lower bound
                        } else {
                            // No more predictions after first point
                            predictedValues.push(null);
                            upperBound.push(null);
                            lowerBound.push(null);
                        }
                    }
                }
                
                // Add a prediction for the future
                const futureDays = 3;
                for (let i = 1; i <= futureDays; i++) {
                    const date = new Date(currentDate);
                    date.setDate(date.getDate() + i);
                    dates.push(date);
                    
                    // No actual values for future dates
                    actualValues.push(null);
                    
                    // Generate predicted values with increasing uncertainty
                    const predictedChange = trend * baseValue * 0.01 * i;
                    const predicted = lastActual + predictedChange;
                    predictedValues.push(predicted);
                    upperBound.push(predicted * (1 + 0.02 * i)); // Increasing upper bound
                    lowerBound.push(predicted * (1 - 0.02 * i)); // Increasing lower bound
                }
                
                const predictionData = {
                    dates: dates,
                    actual: actualValues,
                    predicted: predictedValues,
                    upperBound: upperBound,
                    lowerBound: lowerBound,
                    metric: this.predictionMetric,
                    period: this.comparisonPeriod
                };
                
                this.predictionActualData = predictionData;
                resolve(predictionData);
            }, 350);
        });
    }
    
    fetchDecisionDetails() {
        // Mock data for decision details
        return new Promise(resolve => {
            setTimeout(() => {
                const details = {
                    algorithm: 'ML Strategy v2.3',
                    decisionMethod: 'Ensemble (GBM+LSTM)',
                    featuresUsed: 42,
                    type: 'buy', // buy, sell, hold
                    asset: 'BTC/USD',
                    timestamp: '2025-03-25 14:30:00',
                    confidence: 0.78,
                    signalStrength: '8.2/10',
                    expectedReturn: '+3.8%',
                    historicalAccuracy: 0.76,
                    signalFrequency: 'Rare (92nd percentile)',
                    explanation: 'This buy signal was primarily driven by positive sentiment indicators (+45% contribution) and favorable technical patterns (+28%). Market regime recognition identified a bullish trend continuation pattern with medium confidence. The sentiment analysis detected positive social media momentum paired with neutral news coverage. Key technical signals include a breakout above the 50-day moving average with increasing volume.\n\nNote that some conflicting signals were detected: short-term momentum oscillators show slight bearish divergence (-12%), and on-chain metrics indicate potential selling pressure from miners (-8%).'
                };
                
                this.decisionDetails = details;
                resolve(details);
            }, 250);
        });
    }
    
    initializeVisualizations() {
        this.renderFactorContribution();
        this.renderFeatureImportance();
        this.renderDecisionPath();
        this.renderPredictionVsActual();
        this.updateDecisionSummary();
    }
    
    renderFactorContribution() {
        const element = document.getElementById(this.options.factorContributionElementId);
        if (!element || !this.factorData || this.factorData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Sort factors for better visualization
        const sortedFactors = [...this.factorData].sort((a, b) => b.contribution - a.contribution);
        
        let traces = [];
        let layout = {};
        
        // Switch based on selected view
        if (this.contributionView === 'waterfall') {
            // Waterfall chart showing how each factor contributes to the final decision
            const baseValue = 0.5; // Start with a neutral value
            let measure = [];
            let labels = [];
            let values = [];
            let text = [];
            let runningTotal = baseValue;
            
            // Add initial value
            measure.push('absolute');
            labels.push('Base');
            values.push(baseValue);
            text.push('Base value: 0.5');
            
            // Add each factor as a relative contribution
            sortedFactors.forEach(factor => {
                measure.push('relative');
                labels.push(factor.name);
                values.push(factor.contribution);
                text.push(`${factor.name}: ${(factor.contribution * 100).toFixed(1)}%`);
                runningTotal += factor.contribution;
            });
            
            // Add final value
            measure.push('total');
            labels.push('Final Signal');
            values.push(runningTotal);
            text.push(`Final signal: ${(runningTotal * 100).toFixed(1)}%`);
            
            // Create trace
            traces = [{
                type: 'waterfall',
                measure: measure,
                x: labels,
                y: values,
                text: text,
                textposition: 'outside',
                connector: {
                    line: {
                        color: 'rgb(var(--text-light-rgb))'
                    }
                },
                increasing: {
                    marker: { color: 'rgba(var(--success-rgb), 0.7)' }
                },
                decreasing: {
                    marker: { color: 'rgba(var(--danger-rgb), 0.7)' }
                },
                totals: {
                    marker: { color: 'rgba(var(--primary-rgb), 0.9)' }
                }
            }];
            
            layout = {
                title: 'Factor Contribution Analysis',
                margin: { l: 60, r: 20, t: 40, b: 80 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
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
                    title: 'Contribution',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    zerolinecolor: 'var(--border-light)'
                },
                showlegend: false
            };
        } else if (this.contributionView === 'horizontal') {
            // Horizontal bar chart
            const y = sortedFactors.map(d => d.name);
            const x = sortedFactors.map(d => d.contribution);
            
            traces = [{
                type: 'bar',
                orientation: 'h',
                x: x,
                y: y,
                text: x.map(v => (v * 100).toFixed(1) + '%'),
                textposition: 'auto',
                marker: {
                    color: x.map(value => {
                        if (value > 0) return 'rgba(var(--success-rgb), 0.7)';
                        return 'rgba(var(--danger-rgb), 0.7)';
                    })
                },
                hovertemplate: '%{y}: %{x:.3f}<extra></extra>'
            }];
            
            layout = {
                title: 'Factor Contribution Analysis',
                margin: { l: 120, r: 60, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {
                    color: 'var(--text)',
                    size: 10
                },
                xaxis: {
                    title: 'Contribution',
                    showgrid: true,
                    gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                    zerolinecolor: 'var(--border-light)'
                },
                yaxis: {
                    title: '',
                    showgrid: false
                },
                showlegend: false
            };
        } else if (this.contributionView === 'radar') {
            // Radar chart for factor visualization
            const theta = sortedFactors.map(d => d.name);
            const r = sortedFactors.map(d => Math.max(0, d.contribution + 0.5)); // Adjust to ensure positive values
            
            traces = [{
                type: 'scatterpolar',
                r: r,
                theta: theta,
                fill: 'toself',
                fillcolor: 'rgba(var(--primary-rgb), 0.2)',
                line: {
                    color: 'rgba(var(--primary-rgb), 0.8)'
                },
                hovertemplate: '%{theta}: %{r:.3f}<extra></extra>'
            }];
            
            layout = {
                title: 'Factor Contribution Analysis',
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
                        range: [0, 1.5]
                    },
                    angularaxis: {
                        direction: 'clockwise'
                    }
                },
                showlegend: false
            };
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.factorContributionElementId, traces, layout, config);
        this.factorChart = document.getElementById(this.options.factorContributionElementId);
    }
    
    renderFeatureImportance() {
        const element = document.getElementById(this.options.featureImportanceElementId);
        if (!element || !this.featureImportanceData || this.featureImportanceData.length === 0) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Limit features based on selection
        let featuresCount = parseInt(this.featureCount);
        if (isNaN(featuresCount) || this.featureCount === 'all') {
            featuresCount = this.featureImportanceData.length;
        }
        
        // Sort features by importance
        const sortedFeatures = [...this.featureImportanceData]
            .sort((a, b) => b.importance - a.importance)
            .slice(0, featuresCount);
        
        // Create horizontal bar chart for feature importance
        const y = sortedFeatures.map(d => d.name);
        const x = sortedFeatures.map(d => d.importance);
        
        const trace = {
            type: 'bar',
            orientation: 'h',
            x: x,
            y: y,
            marker: {
                color: x.map(value => {
                    const normalizedValue = value / Math.max(...x);
                    return this.getColorFromScale(normalizedValue, this.options.colorScaleImportance);
                })
            },
            hovertemplate: '%{y}: %{x:.3f}<extra></extra>'
        };
        
        // Layout configuration
        const layout = {
            title: `Feature Importance (${this.importanceType.charAt(0).toUpperCase() + this.importanceType.slice(1)})`,
            margin: { l: 150, r: 40, t: 40, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: 'var(--text)',
                size: 10
            },
            xaxis: {
                title: 'Importance',
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)'
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
        Plotly.newPlot(this.options.featureImportanceElementId, [trace], layout, config);
        this.importanceChart = document.getElementById(this.options.featureImportanceElementId);
    }
    
    renderDecisionPath() {
        const element = document.getElementById(this.options.decisionPathElementId);
        if (!element || !this.decisionPathData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        let traces = [];
        let layout = {};
        
        // Switch based on selected visualization type
        if (this.pathVisualization === 'tree') {
            // Tree visualization
            const { nodes, links, activePath } = this.decisionPathData.tree;
            
            // Calculate node positions based on their level
            const levelCounts = {};
            nodes.forEach(node => {
                levelCounts[node.level] = (levelCounts[node.level] || 0) + 1;
            });
            
            const nodePositions = {};
            nodes.forEach(node => {
                const levelWidth = levelCounts[node.level];
                const xStep = 1 / (levelWidth + 1);
                const positionsAtLevel = nodePositions[node.level] || [];
                
                const x = (positionsAtLevel.length + 1) * xStep;
                const y = 1 - (node.level / (Math.max(...Object.keys(levelCounts).map(k => parseInt(k))) + 1));
                
                nodePositions[node.level] = [...positionsAtLevel, x];
                node.x = x;
                node.y = y;
            });
            
            // Create node trace
            const nodeTrace = {
                x: nodes.map(node => node.x),
                y: nodes.map(node => node.y),
                mode: 'markers+text',
                marker: {
                    size: 20,
                    color: nodes.map(node => {
                        if (activePath && activePath.includes(node.id)) {
                            return 'rgba(var(--primary-rgb), 0.8)';
                        }
                        if (node.type === 'output') {
                            if (node.result === 'buy' || node.result === 'strong_buy') {
                                return 'rgba(var(--success-rgb), 0.8)';
                            } else if (node.result === 'sell') {
                                return 'rgba(var(--danger-rgb), 0.8)';
                            } else {
                                return 'rgba(var(--warning-rgb), 0.8)';
                            }
                        }
                        return 'rgba(var(--info-rgb), 0.8)';
                    }),
                    line: {
                        color: 'var(--border)',
                        width: 1
                    }
                },
                text: nodes.map(node => node.label),
                textposition: 'bottom center',
                hoverinfo: 'text',
                hovertext: nodes.map(node => `${node.label} (ID: ${node.id})`),
                type: 'scatter'
            };
            
            // Create edge traces
            const edgeTraces = [];
            
            links.forEach(link => {
                const sourceNode = nodes.find(n => n.id === link.source);
                const targetNode = nodes.find(n => n.id === link.target);
                
                if (!sourceNode || !targetNode) return;
                
                const isActivePath = activePath && 
                                     activePath.includes(sourceNode.id) && 
                                     activePath.includes(targetNode.id) &&
                                     activePath.indexOf(targetNode.id) === activePath.indexOf(sourceNode.id) + 1;
                
                edgeTraces.push({
                    x: [sourceNode.x, targetNode.x],
                    y: [sourceNode.y, targetNode.y],
                    mode: 'lines',
                    line: {
                        width: isActivePath ? 3 : 1,
                        color: isActivePath ? 'rgba(var(--primary-rgb), 0.8)' : 'rgba(var(--text-light-rgb), 0.5)'
                    },
                    hoverinfo: 'text',
                    hovertext: `${sourceNode.label} → ${targetNode.label} (${link.label})`,
                    type: 'scatter'
                });
                
                // Add edge labels
                edgeTraces.push({
                    x: [(sourceNode.x + targetNode.x) / 2],
                    y: [(sourceNode.y + targetNode.y) / 2],
                    mode: 'text',
                    text: [link.label],
                    textposition: 'middle center',
                    textfont: {
                        size: 10,
                        color: 'var(--text-light)'
                    },
                    hoverinfo: 'none',
                    type: 'scatter'
                });
            });
            
            // Combine all traces
            traces = [...edgeTraces, nodeTrace];
            
            layout = {
                title: 'Decision Path',
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { l: 20, r: 20, t: 40, b: 20 },
                showlegend: false,
                hovermode: 'closest',
                xaxis: {
                    showgrid: false,
                    zeroline: false,
                    showticklabels: false,
                    range: [0, 1]
                },
                yaxis: {
                    showgrid: false,
                    zeroline: false,
                    showticklabels: false,
                    range: [0, 1]
                },
                font: {
                    color: 'var(--text)',
                    size: 10
                }
            };
        } else if (this.pathVisualization === 'sankey' || this.pathVisualization === 'flow') {
            // Sankey diagram for flow visualization
            const { nodes, links } = this.decisionPathData.sankey;
            
            traces = [{
                type: 'sankey',
                orientation: 'h',
                node: {
                    pad: 15,
                    thickness: 20,
                    line: {
                        color: 'var(--border)',
                        width: 0.5
                    },
                    label: nodes.map(n => n.name),
                    color: nodes.map((n, i) => {
                        if (i === 0) return 'rgba(var(--info-rgb), 0.8)';
                        if (i === nodes.length - 1 || i === nodes.length - 2 || i === nodes.length - 3) {
                            if (n.name.toLowerCase().includes('buy')) {
                                return 'rgba(var(--success-rgb), 0.8)';
                            } else if (n.name.toLowerCase().includes('sell')) {
                                return 'rgba(var(--danger-rgb), 0.8)';
                            } else {
                                return 'rgba(var(--warning-rgb), 0.8)';
                            }
                        }
                        return 'rgba(var(--primary-rgb), 0.8)';
                    })
                },
                link: {
                    source: links.map(l => l.source),
                    target: links.map(l => l.target),
                    value: links.map(l => l.value),
                    color: links.map(l => 'rgba(var(--text-light-rgb), 0.2)')
                }
            }];
            
            layout = {
                title: 'Decision Flow',
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { l: 20, r: 20, t: 40, b: 20 },
                font: {
                    color: 'var(--text)',
                    size: 10
                }
            };
        }
        
        // Configuration options
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render with Plotly
        Plotly.newPlot(this.options.decisionPathElementId, traces, layout, config);
        this.pathChart = document.getElementById(this.options.decisionPathElementId);
    }
    
    renderPredictionVsActual() {
        const element = document.getElementById(this.options.predictionActualElementId);
        if (!element || !this.predictionActualData) return;
        
        // Clear any loading overlay
        const overlay = element.querySelector('.chart-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        const { dates, actual, predicted, upperBound, lowerBound, metric } = this.predictionActualData;
        
        // Create trace for actual values
        const actualTrace = {
            x: dates,
            y: actual,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Actual',
            line: {
                color: 'rgba(var(--primary-rgb), 0.8)',
                width: 2
            },
            marker: {
                size: 6,
                color: 'rgba(var(--primary-rgb), 0.8)'
            },
            hovertemplate: '%{x|%b %d, %Y}<br>Actual: %{y:.2f}<extra></extra>'
        };
        
        // Create trace for predicted values
        const predictedTrace = {
            x: dates,
            y: predicted,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Predicted',
            line: {
                color: 'rgba(var(--info-rgb), 0.8)',
                width: 2,
                dash: 'dash'
            },
            marker: {
                size: 6,
                color: 'rgba(var(--info-rgb), 0.8)'
            },
            hovertemplate: '%{x|%b %d, %Y}<br>Predicted: %{y:.2f}<extra></extra>'
        };
        
        // Create trace for confidence interval
        const upperTrace = {
            x: dates,
            y: upperBound,
            type: 'scatter',
            mode: 'lines',
            name: 'Upper Bound',
            line: {
                color: 'rgba(var(--info-rgb), 0.1)',
                width: 0
            },
            showlegend: false,
            hoverinfo: 'skip'
        };
        
        const lowerTrace = {
            x: dates,
            y: lowerBound,
            type: 'scatter',
            mode: 'lines',
            name: 'Lower Bound',
            fill: 'tonexty',
            fillcolor: 'rgba(var(--info-rgb), 0.1)',
            line: {
                color: 'rgba(var(--info-rgb), 0.1)',
                width: 0
            },
            showlegend: false,
            hoverinfo: 'skip'
        };
        
        // Add a vertical line at current date
        const currentDate = new Date();
        const nowLine = {
            x: [currentDate, currentDate],
            y: [Math.min(...actual.filter(v => v !== null)) * 0.99, Math.max(...actual.filter(v => v !== null)) * 1.01],
            type: 'scatter',
            mode: 'lines',
            name: 'Now',
            line: {
                color: 'rgba(var(--warning-rgb), 0.8)',
                width: 2,
                dash: 'dot'
            },
            hoverinfo: 'none'
        };
        
        // Layout configuration
        const layout = {
            title: `${metric.charAt(0).toUpperCase() + metric.slice(1)} Prediction vs. Actual`,
            margin: { l: 60, r: 20, t: 40, b: 40 },
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
                title: metric.charAt(0).toUpperCase() + metric.slice(1),
                showgrid: true,
                gridcolor: 'rgba(var(--border-light-rgb), 0.2)',
                linecolor: 'var(--border-light)'
            },
            legend: {
                orientation: 'h',
                xanchor: 'center',
                x: 0.5,
                y: 1.12
            },
            annotations: [
                {
                    x: dates[dates.length - 1],
                    y: predicted[predicted.length - 1],
                    text: 'Forecast',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 1,
                    arrowcolor: 'rgba(var(--info-rgb), 0.8)',
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
        Plotly.newPlot(this.options.predictionActualElementId, [lowerTrace, upperTrace, actualTrace, predictedTrace, nowLine], layout, config);
        this.predictionChart = document.getElementById(this.options.predictionActualElementId);
    }
    
    updateDecisionSummary() {
        // Update decision summary with decision details
        if (!this.decisionDetails) return;
        
        // Update decision header
        const decisionType = document.querySelector('.decision-type');
        if (decisionType) {
            decisionType.textContent = this.decisionDetails.type.toUpperCase() + ' SIGNAL';
            decisionType.className = `decision-type ${this.decisionDetails.type.toLowerCase()}`;
        }
        
        const decisionAsset = document.querySelector('.decision-asset');
        if (decisionAsset) {
            decisionAsset.textContent = this.decisionDetails.asset;
        }
        
        const decisionTimestamp = document.querySelector('.decision-timestamp');
        if (decisionTimestamp) {
            decisionTimestamp.textContent = this.decisionDetails.timestamp;
        }
        
        // Update confidence meter
        const confidenceFill = document.querySelector('.confidence-fill');
        if (confidenceFill) {
            confidenceFill.style.width = `${this.decisionDetails.confidence * 100}%`;
        }
        
        const confidenceValue = document.querySelector('.confidence-value');
        if (confidenceValue) {
            confidenceValue.textContent = `${Math.round(this.decisionDetails.confidence * 100)}%`;
        }
        
        // Update explanation text
        const explanationElement = document.getElementById('decision-explanation');
        if (explanationElement) {
            explanationElement.innerHTML = `<p>${this.decisionDetails.explanation.replace('\n\n', '</p><p>')}</p>`;
        }
        
        // Update key metrics
        const signalStrengthElement = document.getElementById('signal-strength');
        if (signalStrengthElement) {
            signalStrengthElement.textContent = this.decisionDetails.signalStrength;
        }
        
        const expectedReturnElement = document.getElementById('expected-return');
        if (expectedReturnElement) {
            expectedReturnElement.textContent = this.decisionDetails.expectedReturn;
        }
        
        const historicalAccuracyElement = document.getElementById('historical-accuracy');
        if (historicalAccuracyElement) {
            historicalAccuracyElement.textContent = `${Math.round(this.decisionDetails.historicalAccuracy * 100)}%`;
        }
        
        const signalFrequencyElement = document.getElementById('signal-frequency');
        if (signalFrequencyElement) {
            signalFrequencyElement.textContent = this.decisionDetails.signalFrequency;
        }
        
        // Update footer metrics
        const algorithmNameElement = document.getElementById('algorithm-name');
        if (algorithmNameElement) {
            algorithmNameElement.textContent = this.decisionDetails.algorithm;
        }
        
        const decisionMethodElement = document.getElementById('decision-method');
        if (decisionMethodElement) {
            decisionMethodElement.textContent = this.decisionDetails.decisionMethod;
        }
        
        const featuresUsedElement = document.getElementById('features-used');
        if (featuresUsedElement) {
            featuresUsedElement.textContent = this.decisionDetails.featuresUsed;
        }
        
        const lastUpdatedElement = document.getElementById('explainability-last-updated');
        if (lastUpdatedElement) {
            lastUpdatedElement.textContent = '5 minutes ago';
        }
    }
    
    setupEventListeners() {
        // Algorithm selector change
        const algorithmSelector = document.getElementById('explainability-algorithm');
        if (algorithmSelector) {
            algorithmSelector.addEventListener('change', () => {
                this.algorithm = algorithmSelector.value;
                this.refreshData();
            });
        }
        
        // Timeframe selector change
        const timeframeSelector = document.getElementById('explainability-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', () => {
                this.timeframe = timeframeSelector.value;
                this.refreshData();
            });
        }
        
        // Signal selector change
        const signalSelector = document.getElementById('explainability-signal');
        if (signalSelector) {
            signalSelector.addEventListener('change', () => {
                this.signal = signalSelector.value;
                
                // Show/hide datetime picker for specific date/time
                const datetimeContainer = document.getElementById('specific-datetime-container');
                if (datetimeContainer) {
                    datetimeContainer.style.display = this.signal === 'specific' ? 'block' : 'none';
                }
                
                this.refreshData();
            });
        }
        
        // Datetime picker change
        const datetimePicker = document.getElementById('explainability-datetime');
        if (datetimePicker) {
            datetimePicker.addEventListener('change', () => {
                this.datetime = datetimePicker.value;
                if (this.signal === 'specific') {
                    this.refreshData();
                }
            });
        }
        
        // Contribution view selector change
        const contributionViewSelector = document.getElementById('contribution-view');
        if (contributionViewSelector) {
            contributionViewSelector.addEventListener('change', () => {
                this.contributionView = contributionViewSelector.value;
                this.renderFactorContribution();
            });
        }
        
        // Normalize selector change
        const normalizeSelector = document.getElementById('normalize-factors');
        if (normalizeSelector) {
            normalizeSelector.addEventListener('change', () => {
                this.normalize = normalizeSelector.value;
                this.renderFactorContribution();
            });
        }
        
        // Importance type selector change
        const importanceTypeSelector = document.getElementById('importance-type');
        if (importanceTypeSelector) {
            importanceTypeSelector.addEventListener('change', () => {
                this.importanceType = importanceTypeSelector.value;
                this.fetchFeatureImportanceData()
                    .then(() => {
                        this.renderFeatureImportance();
                    });
            });
        }
        
        // Feature count selector change
        const featureCountSelector = document.getElementById('feature-count');
        if (featureCountSelector) {
            featureCountSelector.addEventListener('change', () => {
                this.featureCount = featureCountSelector.value;
                this.renderFeatureImportance();
            });
        }
        
        // Path visualization selector change
        const pathVisualizationSelector = document.getElementById('path-visualization');
        if (pathVisualizationSelector) {
            pathVisualizationSelector.addEventListener('change', () => {
                this.pathVisualization = pathVisualizationSelector.value;
                this.renderDecisionPath();
            });
        }
        
        // Path detail selector change
        const pathDetailSelector = document.getElementById('path-detail');
        if (pathDetailSelector) {
            pathDetailSelector.addEventListener('change', () => {
                this.pathDetail = pathDetailSelector.value;
                this.renderDecisionPath();
            });
        }
        
        // Prediction metric selector change
        const predictionMetricSelector = document.getElementById('prediction-metric');
        if (predictionMetricSelector) {
            predictionMetricSelector.addEventListener('change', () => {
                this.predictionMetric = predictionMetricSelector.value;
                this.fetchPredictionActualData()
                    .then(() => {
                        this.renderPredictionVsActual();
                    });
            });
        }
        
        // Comparison period selector change
        const comparisonPeriodSelector = document.getElementById('comparison-period');
        if (comparisonPeriodSelector) {
            comparisonPeriodSelector.addEventListener('change', () => {
                this.comparisonPeriod = comparisonPeriodSelector.value;
                this.fetchPredictionActualData()
                    .then(() => {
                        this.renderPredictionVsActual();
                    });
            });
        }
        
        // Explainability settings button
        const settingsBtn = document.getElementById('explainability-settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                // Show the settings modal
                const modal = document.getElementById('explainability-settings-modal');
                if (modal) {
                    modal.style.display = 'block';
                }
            });
        }
        
        // Download data button
        const downloadBtn = document.getElementById('download-explainability-data-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadExplainabilityData();
            });
        }
        
        // Expand panel button
        const expandBtn = document.getElementById('expand-explainability-panel-btn');
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
        
        // Benchmark models toggle
        const benchmarkToggle = document.getElementById('use-benchmark-models');
        const benchmarkSelection = document.getElementById('benchmark-selection');
        if (benchmarkToggle && benchmarkSelection) {
            benchmarkToggle.addEventListener('change', () => {
                benchmarkSelection.style.display = benchmarkToggle.checked ? 'block' : 'none';
            });
        }
        
        // Explanation depth slider
        const depthSlider = document.getElementById('explanation-depth');
        if (depthSlider) {
            depthSlider.addEventListener('input', function() {
                // Update the value display
                const valueDisplay = this.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
            });
        }
        
        // Save settings button
        const saveSettingsBtn = document.getElementById('save-explainability-settings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => {
                // In a real implementation, this would save the settings
                alert('Settings saved successfully');
                
                // Close the modal
                const modal = document.getElementById('explainability-settings-modal');
                if (modal) {
                    modal.style.display = 'none';
                }
                
                // Refresh data with new settings
                this.refreshData();
            });
        }
        
        // Learn more button
        const learnMoreBtn = document.getElementById('explainability-learn-more');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', () => {
                // This would open documentation or a tutorial
                alert('Algorithm Explainability documentation will be available in the next phase');
            });
        }
        
        // Export report button
        const exportReportBtn = document.getElementById('export-explainability-report');
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
            this.options.factorContributionElementId,
            this.options.featureImportanceElementId,
            this.options.decisionPathElementId,
            this.options.predictionActualElementId
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
                this.renderFactorContribution();
                this.renderFeatureImportance();
                this.renderDecisionPath();
                this.renderPredictionVsActual();
                this.updateDecisionSummary();
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
    
    downloadExplainabilityData() {
        // Create downloadable JSON file with explainability data
        const data = {
            algorithm: this.algorithm,
            timeframe: this.timeframe,
            signal: this.signal,
            datetime: this.datetime,
            factorData: this.factorData,
            featureImportanceData: this.featureImportanceData,
            decisionPathData: this.decisionPathData,
            predictionActualData: this.predictionActualData,
            decisionDetails: this.decisionDetails
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `algorithm_explainability_${this.algorithm}_${new Date().toISOString().split('T')[0]}.json`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    
    getColorFromScale(value, scale) {
        // Get color from colorscale based on value (0-1)
        const colorscale = scale || this.options.colorScaleContribution;
        
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

// Initialize the Algorithm Explainability component when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create instance of AlgorithmExplainability
    const algorithmExplainability = new AlgorithmExplainability();
    
    // Initialize Feather icons if available
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});