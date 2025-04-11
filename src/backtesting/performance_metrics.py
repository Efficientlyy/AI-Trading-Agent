import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any


def calculate_performance_metrics(portfolio_history: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """
    Calculates key performance metrics from portfolio value history.

    Args:
        portfolio_history: DataFrame with 'timestamp' (index) and 'value' columns.
        risk_free_rate: The annual risk-free rate for Sharpe Ratio calculation.

    Returns:
        Dictionary containing calculated performance metrics.
    """
    if portfolio_history.empty or len(portfolio_history) < 2:
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'annualized_volatility_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'error': 'Insufficient data for calculation'
        }

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(portfolio_history.index):
        portfolio_history.index = pd.to_datetime(portfolio_history.index)

    # Calculate daily returns (or returns based on data frequency)
    portfolio_history['returns'] = portfolio_history['value'].pct_change().fillna(0)

    # --- Total Return ---
    total_return = (portfolio_history['value'].iloc[-1] / portfolio_history['value'].iloc[0]) - 1

    # --- Annualized Return ---
    time_delta = portfolio_history.index[-1] - portfolio_history.index[0]
    years = time_delta.days / 365.25
    # Avoid division by zero if duration is very short
    annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0.0

    # --- Annualized Volatility ---
    # Assuming daily data, multiply by sqrt(252) trading days
    # More robust: detect frequency or require it as input
    returns_std = portfolio_history['returns'].std()
    # Check for NaN standard deviation (can happen with constant returns)
    annualized_volatility = returns_std * np.sqrt(252) if pd.notna(returns_std) and years > 0 else 0.0

    # --- Sharpe Ratio ---
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0.0

    # --- Max Drawdown ---
    portfolio_history['cumulative_max'] = portfolio_history['value'].cummax()
    portfolio_history['drawdown'] = (portfolio_history['value'] / portfolio_history['cumulative_max']) - 1
    max_drawdown = portfolio_history['drawdown'].min()

    metrics = {
        'total_return_pct': total_return * 100,
        'annualized_return_pct': annualized_return * 100,
        'annualized_volatility_pct': annualized_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'start_date': portfolio_history.index[0].strftime('%Y-%m-%d'),
        'end_date': portfolio_history.index[-1].strftime('%Y-%m-%d'),
        'duration_years': round(years, 2)
    }

    return metrics


# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=252 * 2, freq='B') # 2 years of business days
    initial_value = 10000
    returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))
    values = initial_value * (1 + returns).cumprod()
    sample_history = pd.DataFrame({'value': values}, index=dates)

    print("Sample Portfolio History:")
    print(sample_history.head())

    calculated_metrics = calculate_performance_metrics(sample_history)

    print("\nCalculated Metrics:")
    for key, value in calculated_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
