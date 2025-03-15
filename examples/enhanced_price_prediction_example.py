"""Example usage of enhanced price prediction components.

This example demonstrates:
1. Data generation and preprocessing
2. Feature extraction and engineering
3. Model training with cross-validation
4. Real-time prediction simulation
5. Performance analysis with multiple metrics
6. Handling different market scenarios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from src.ml.models.enhanced_price_prediction_v2 import (
    TechnicalFeatures,
    FeatureExtractor,
    ModelPredictor,
    FeatureVector
)
from src.ml.models.model_training import ModelTrainer, ModelValidator

def generate_market_scenario(
    scenario: str,
    n_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate different market scenario data.
    
    Scenarios:
    - trending_up: Strong upward trend with noise
    - trending_down: Strong downward trend with noise
    - sideways: Range-bound market with noise
    - volatile: High volatility with no clear trend
    - random_walk: Random walk with drift
    """
    if scenario == "trending_up":
        trend = np.linspace(0, 1, n_samples)
        noise = np.random.normal(0, 0.02, n_samples)
        returns = trend + noise
        
    elif scenario == "trending_down":
        trend = np.linspace(0, -1, n_samples)
        noise = np.random.normal(0, 0.02, n_samples)
        returns = trend + noise
        
    elif scenario == "sideways":
        returns = np.random.normal(0, 0.01, n_samples)
        
    elif scenario == "volatile":
        returns = np.random.normal(0, 0.05, n_samples)
        
    else:  # random_walk
        returns = np.random.normal(0.0002, 0.02, n_samples)  # Small positive drift
    
    # Generate prices from returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate labels (1 for up, 0 for down)
    labels = np.where(returns > 0, 1.0, 0.0)
    
    return prices.astype(np.float64), labels.astype(np.float64)

def generate_market_sentiment(
    trend: str = "neutral",
    volatility: float = 0.2
) -> Dict[str, float]:
    """Generate market sentiment data based on trend.
    
    Args:
        trend: One of 'bullish', 'bearish', or 'neutral'
        volatility: Amount of random variation in sentiment
    """
    base_sentiment = {
        "bullish": 0.7,
        "bearish": 0.3,
        "neutral": 0.5
    }.get(trend, 0.5)
    
    return {
        "social_sentiment": np.clip(
            base_sentiment + np.random.normal(0, volatility),
            0, 1
        ),
        "news_sentiment": np.clip(
            base_sentiment + np.random.normal(0, volatility),
            0, 1
        ),
        "order_flow_sentiment": np.clip(
            base_sentiment + np.random.normal(0, volatility),
            0, 1
        ),
        "fear_greed_index": np.clip(
            base_sentiment * 100 + np.random.normal(0, volatility * 100),
            0, 100
        )
    }

def generate_market_conditions(
    liquidity: str = "normal",
    volatility: str = "normal"
) -> Dict[str, float]:
    """Generate market condition data based on specified parameters.
    
    Args:
        liquidity: One of 'low', 'normal', or 'high'
        volatility: One of 'low', 'normal', or 'high'
    """
    liquidity_levels = {
        "low": 0.3,
        "normal": 0.6,
        "high": 0.9
    }
    
    volatility_levels = {
        "low": 0.1,
        "normal": 0.3,
        "high": 0.5
    }
    
    base_liquidity = liquidity_levels.get(liquidity, 0.6)
    base_volatility = volatility_levels.get(volatility, 0.3)
    
    return {
        "liquidity_score": np.clip(
            base_liquidity + np.random.normal(0, 0.1),
            0, 1
        ),
        "volatility": np.clip(
            base_volatility + np.random.normal(0, 0.05),
            0, 1
        ),
        "correlation_score": np.clip(
            np.random.normal(0, 0.3),
            -1, 1
        )
    }

def evaluate_scenario(
    scenario: str,
    market_sentiment: str,
    liquidity: str,
    volatility: str,
    n_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate model performance on a specific market scenario.
    
    Args:
        scenario: Market price scenario
        market_sentiment: Overall market sentiment
        liquidity: Market liquidity condition
        volatility: Market volatility condition
        n_samples: Number of samples to generate
    
    Returns:
        Dictionary containing performance metrics
    """
    # Generate scenario data
    prices, labels = generate_market_scenario(scenario, n_samples)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    trainer = ModelTrainer(model_type="random_forest", n_splits=5)
    validator = ModelValidator()
    
    # Extract features
    feature_list = []
    for i in range(len(prices) - 100):
        window = prices[i:i+100]
        features = feature_extractor.extract_features(
            window,
            generate_market_sentiment(market_sentiment),
            generate_market_conditions(liquidity, volatility)
        )
        combined = feature_extractor.combine_features(features)
        feature_list.append(combined)
    
    features = np.array(feature_list)
    train_labels = labels[100:]
    
    # Train and validate
    metrics_list = trainer.train_and_validate(features, train_labels)
    
    # Make predictions on test set
    predictor = ModelPredictor(trainer.model, trainer.scaler)
    test_predictions = []
    
    for feature in features[-100:]:
        pred = predictor.predict(feature)
        test_predictions.append(pred["direction"])
    
    test_predictions = np.array(test_predictions, dtype=np.float64)
    test_returns = np.diff(prices[-101:]) / prices[-101:-1]
    
    # Calculate performance metrics
    avg_metrics = {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics_list])),
        "precision": float(np.mean([m["precision"] for m in metrics_list])),
        "recall": float(np.mean([m["recall"] for m in metrics_list])),
        "f1_score": float(np.mean([m["f1_score"] for m in metrics_list]))
    }
    
    trading_metrics = {
        "profit_factor": float(validator.calculate_profit_factor(test_predictions, test_returns)),
        "sharpe_ratio": float(validator.calculate_sharpe_ratio(test_predictions, test_returns)),
        "max_drawdown": float(validator.calculate_max_drawdown(test_predictions, test_returns))
    }
    
    return {**avg_metrics, **trading_metrics}

def main():
    """Main example function demonstrating different market scenarios."""
    print("Enhanced Price Prediction Example")
    print("--------------------------------")
    
    # Define scenarios to test
    scenarios = [
        # Basic Market Conditions
        {
            "name": "Strong Bull Market",
            "price_scenario": "trending_up",
            "sentiment": "bullish",
            "liquidity": "high",
            "volatility": "low"
        },
        {
            "name": "Strong Bear Market",
            "price_scenario": "trending_down",
            "sentiment": "bearish",
            "liquidity": "low",
            "volatility": "high"
        },
        {
            "name": "Sideways Market",
            "price_scenario": "sideways",
            "sentiment": "neutral",
            "liquidity": "normal",
            "volatility": "normal"
        },
        
        # Mixed Conditions
        {
            "name": "Bull Market with High Volatility",
            "price_scenario": "trending_up",
            "sentiment": "bullish",
            "liquidity": "normal",
            "volatility": "high"
        },
        {
            "name": "Bear Market with Low Volatility",
            "price_scenario": "trending_down",
            "sentiment": "bearish",
            "liquidity": "normal",
            "volatility": "low"
        },
        
        # Sentiment-Price Divergence
        {
            "name": "Bearish Sentiment in Bull Market",
            "price_scenario": "trending_up",
            "sentiment": "bearish",
            "liquidity": "normal",
            "volatility": "normal"
        },
        {
            "name": "Bullish Sentiment in Bear Market",
            "price_scenario": "trending_down",
            "sentiment": "bullish",
            "liquidity": "normal",
            "volatility": "normal"
        },
        
        # Liquidity Scenarios
        {
            "name": "High Volatility Low Liquidity",
            "price_scenario": "volatile",
            "sentiment": "neutral",
            "liquidity": "low",
            "volatility": "high"
        },
        {
            "name": "Low Volatility High Liquidity",
            "price_scenario": "sideways",
            "sentiment": "neutral",
            "liquidity": "high",
            "volatility": "low"
        },
        
        # Market Regime Changes
        {
            "name": "Transition to High Volatility",
            "price_scenario": "volatile",
            "sentiment": "bearish",
            "liquidity": "low",
            "volatility": "high"
        },
        {
            "name": "Recovery from Bear Market",
            "price_scenario": "trending_up",
            "sentiment": "neutral",
            "liquidity": "normal",
            "volatility": "high"
        },
        
        # Random Walk Variations
        {
            "name": "Efficient Market",
            "price_scenario": "random_walk",
            "sentiment": "neutral",
            "liquidity": "high",
            "volatility": "low"
        },
        {
            "name": "Inefficient Market",
            "price_scenario": "random_walk",
            "sentiment": "neutral",
            "liquidity": "low",
            "volatility": "high"
        }
    ]
    
    # Evaluate each scenario
    results = []
    for scenario in scenarios:
        print(f"\nEvaluating {scenario['name']}...")
        metrics = evaluate_scenario(
            scenario["price_scenario"],
            scenario["sentiment"],
            scenario["liquidity"],
            scenario["volatility"]
        )
        
        results.append({
            "scenario": scenario["name"],
            **metrics
        })
    
    # Display results as a table
    df = pd.DataFrame(results)
    df = df.set_index("scenario")
    
    print("\nResults Summary:")
    print("---------------")
    print(df.round(4))
    
    # Find best and worst scenarios
    best_scenario = df["sharpe_ratio"].idxmax()
    worst_scenario = df["sharpe_ratio"].idxmin()
    
    print("\nBest Performing Scenario:", best_scenario)
    print("Worst Performing Scenario:", worst_scenario)
    
    # Additional Analysis
    print("\nScenario Analysis:")
    print("------------------")
    
    # Analyze performance by market condition
    bull_mask = df.index.str.contains("Bull")
    bear_mask = df.index.str.contains("Bear")
    neutral_mask = ~(bull_mask | bear_mask)
    
    print("\nAverage Metrics by Market Condition:")
    print("Bull Markets:   Sharpe={:.4f}, Profit Factor={:.4f}".format(
        df[bull_mask]["sharpe_ratio"].mean(),
        df[bull_mask]["profit_factor"].mean()
    ))
    print("Bear Markets:   Sharpe={:.4f}, Profit Factor={:.4f}".format(
        df[bear_mask]["sharpe_ratio"].mean(),
        df[bear_mask]["profit_factor"].mean()
    ))
    print("Neutral/Other: Sharpe={:.4f}, Profit Factor={:.4f}".format(
        df[neutral_mask]["sharpe_ratio"].mean(),
        df[neutral_mask]["profit_factor"].mean()
    ))
    
    # Analyze impact of volatility
    high_vol_mask = df.index.str.contains("High Volatility")
    low_vol_mask = df.index.str.contains("Low Volatility")
    
    print("\nAverage Metrics by Volatility:")
    print("High Volatility: Sharpe={:.4f}, Max Drawdown={:.4f}".format(
        df[high_vol_mask]["sharpe_ratio"].mean(),
        df[high_vol_mask]["max_drawdown"].mean()
    ))
    print("Low Volatility:  Sharpe={:.4f}, Max Drawdown={:.4f}".format(
        df[low_vol_mask]["sharpe_ratio"].mean(),
        df[low_vol_mask]["max_drawdown"].mean()
    ))
    
    # Find most consistent scenario
    df["sharpe_std"] = df.groupby(level=0)["sharpe_ratio"].transform("std")
    most_consistent = df["sharpe_std"].idxmin()
    
    print("\nMost Consistent Scenario:", most_consistent)
    print("(Lowest Sharpe Ratio Standard Deviation)")

if __name__ == "__main__":
    main() 