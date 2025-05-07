"""
Sentiment-aware Backtesting Framework.

This module extends the backtesting framework to incorporate sentiment data
and adds sentiment-specific performance metrics.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats

from ai_trading_agent.backtesting.backtester import Backtester
from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignalProcessor
from ai_trading_agent.signal_processing.signal_aggregator import TradingSignal, SignalDirection
from ai_trading_agent.signal_processing.regime import MarketRegimeDetector, MarketRegime
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Position, Portfolio, Trade

logger = logging.getLogger(__name__)


class SentimentBacktester(Backtester):
    """
    Extended backtesting framework that incorporates sentiment data.
    
    This class adds sentiment-specific functionality to the base Backtester,
    including loading and processing sentiment data, tracking sentiment-based
    signals, and calculating sentiment-specific performance metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment backtester.
        
        Args:
            config: Configuration dictionary for the backtester
        """
        super().__init__(config or {})
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(config=self.config.get("sentiment_analyzer", {}))
        
        # Initialize sentiment signal processor
        self.signal_processor = SentimentSignalProcessor(
            threshold=float(self.config.get("sentiment_threshold", 0.2)),
            window_size=int(self.config.get("window_size", 3)),
            sentiment_weight=float(self.config.get("sentiment_weight", 0.4)),
            min_confidence=float(self.config.get("min_confidence", 0.6)),
            enable_regime_detection=bool(self.config.get("enable_regime_detection", True))
        )
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector(
            volatility_window=int(self.config.get("volatility_window", 20)),
            trend_window=int(self.config.get("trend_window", 50)),
            volatility_threshold=float(self.config.get("volatility_threshold", 0.015)),
            trend_threshold=float(self.config.get("trend_threshold", 0.6)),
            range_threshold=float(self.config.get("range_threshold", 0.3))
        )
        
        # Sentiment data storage
        self.sentiment_data = {}  # symbol -> DataFrame
        
        # Sentiment signal storage
        self.sentiment_signals = {}  # symbol -> List[TradingSignal]
        
        # Sentiment-trade correlation tracking
        self.sentiment_trade_correlations = []
        
        # Sentiment regime tracking
        self.sentiment_regimes = {}  # timestamp -> regime
        
        # Additional metrics
        self.sentiment_metrics = {}
    
    def load_sentiment_data(self, symbol: str, sentiment_data: pd.DataFrame) -> None:
        """
        Load sentiment data for a symbol.
        
        Args:
            symbol: Symbol to load data for
            sentiment_data: DataFrame containing sentiment data
        """
        if sentiment_data.empty:
            logger.warning(f"Empty sentiment data provided for {symbol}")
            return
        
        # Ensure data has required columns
        required_columns = ['timestamp', 'compound_score']
        if not all(col in sentiment_data.columns for col in required_columns):
            logger.error(f"Sentiment data for {symbol} missing required columns: {required_columns}")
            return
        
        # Ensure timestamp is datetime
        sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
        
        # Sort by timestamp
        sentiment_data = sentiment_data.sort_values('timestamp')
        
        # Store data
        self.sentiment_data[symbol] = sentiment_data
        
        logger.info(f"Loaded sentiment data for {symbol}: {len(sentiment_data)} records")
    
    def process_sentiment_signals(self, symbol: str) -> List[TradingSignal]:
        """
        Process sentiment data into trading signals.
        
        Args:
            symbol: Symbol to process signals for
            
        Returns:
            List of TradingSignal objects
        """
        if symbol not in self.sentiment_data:
            logger.warning(f"No sentiment data available for {symbol}")
            return []
        
        if symbol not in self.price_data:
            logger.warning(f"No price data available for {symbol}")
            return []
        
        sentiment_df = self.sentiment_data[symbol]
        price_df = self.price_data[symbol]
        
        # Convert sentiment data to Series for processing
        sentiment_series = pd.Series(
            sentiment_df['compound_score'].values,
            index=pd.DatetimeIndex(sentiment_df['timestamp'])
        )
        
        # Process sentiment data into signals
        signals = self.signal_processor.process_sentiment_data(
            symbol=symbol,
            historical_sentiment=sentiment_series,
            timeframe=self.config.get("timeframe", "1d"),
            price_data=price_df
        )
        
        # Convert to TradingSignal objects
        trading_signals = [TradingSignal.from_sentiment_signal(signal) for signal in signals]
        
        # Store signals
        self.sentiment_signals[symbol] = trading_signals
        
        logger.info(f"Processed {len(trading_signals)} sentiment signals for {symbol}")
        
        return trading_signals
    
    def detect_sentiment_regimes(self, symbol: str) -> Dict[pd.Timestamp, str]:
        """
        Detect sentiment regimes for a symbol.
        
        Args:
            symbol: Symbol to detect regimes for
            
        Returns:
            Dictionary mapping timestamps to regime names
        """
        if symbol not in self.sentiment_data:
            logger.warning(f"No sentiment data available for {symbol}")
            return {}
        
        sentiment_df = self.sentiment_data[symbol]
        
        # Ensure we have enough data
        if len(sentiment_df) < 10:
            logger.warning(f"Not enough sentiment data for {symbol} to detect regimes")
            return {}
        
        # Calculate rolling statistics
        window = min(14, len(sentiment_df) // 2)
        
        # Resample to regular intervals if needed
        if len(sentiment_df) > 0:
            sentiment_df = sentiment_df.set_index('timestamp')
            # Check if index is already sorted
            if not sentiment_df.index.is_monotonic_increasing:
                sentiment_df = sentiment_df.sort_index()
            
            # Calculate rolling mean and standard deviation
            sentiment_df['rolling_mean'] = sentiment_df['compound_score'].rolling(window=window).mean()
            sentiment_df['rolling_std'] = sentiment_df['compound_score'].rolling(window=window).std()
            
            # Calculate z-scores
            sentiment_df['z_score'] = (sentiment_df['compound_score'] - sentiment_df['rolling_mean']) / sentiment_df['rolling_std'].replace(0, 0.001)
            
            # Define regimes
            regimes = {}
            for idx, row in sentiment_df.iterrows():
                if pd.isna(row['rolling_mean']) or pd.isna(row['rolling_std']):
                    continue
                    
                if row['rolling_mean'] > 0.3:
                    if row['rolling_std'] > 0.3:
                        regime = "volatile_bullish"
                    else:
                        regime = "stable_bullish"
                elif row['rolling_mean'] < -0.3:
                    if row['rolling_std'] > 0.3:
                        regime = "volatile_bearish"
                    else:
                        regime = "stable_bearish"
                else:
                    if row['rolling_std'] > 0.3:
                        regime = "volatile_neutral"
                    else:
                        regime = "stable_neutral"
                
                regimes[idx] = regime
            
            # Store regimes
            self.sentiment_regimes[symbol] = regimes
            
            logger.info(f"Detected sentiment regimes for {symbol}: {len(regimes)} periods")
            
            return regimes
        
        return {}
    
    def run_backtest_with_sentiment(
        self, 
        strategy: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run a backtest with sentiment data.
        
        Args:
            strategy: Strategy function that takes (timestamp, prices, portfolio, sentiment_signals)
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            commission: Commission rate for trades
            
        Returns:
            Dictionary of backtest results
        """
        # Process sentiment signals for all symbols
        all_sentiment_signals = {}
        for symbol in self.price_data.keys():
            if symbol in self.sentiment_data:
                signals = self.process_sentiment_signals(symbol)
                all_sentiment_signals[symbol] = signals
        
        # Detect sentiment regimes
        for symbol in self.sentiment_data.keys():
            self.detect_sentiment_regimes(symbol)
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            cash=Decimal(str(initial_capital)),
            positions={},
            orders=[],
            trades=[]
        )
        
        # Get all timestamps from price data
        all_timestamps = []
        for symbol, df in self.price_data.items():
            all_timestamps.extend(df.index.tolist())
        
        # Sort and deduplicate timestamps
        all_timestamps = sorted(set(all_timestamps))
        
        # Filter timestamps by date range
        if start_date:
            all_timestamps = [ts for ts in all_timestamps if ts >= pd.Timestamp(start_date)]
        if end_date:
            all_timestamps = [ts for ts in all_timestamps if ts <= pd.Timestamp(end_date)]
        
        # Track equity curve
        equity_curve = []
        
        # Track sentiment correlation
        sentiment_values = []
        returns = []
        
        # Run backtest
        for timestamp in all_timestamps:
            # Get current prices
            current_prices = {}
            for symbol, df in self.price_data.items():
                if timestamp in df.index:
                    current_prices[symbol] = Decimal(str(df.loc[timestamp, 'close']))
            
            # Get sentiment signals active at this timestamp
            current_sentiment_signals = {}
            for symbol, signals in all_sentiment_signals.items():
                # Filter signals active at this timestamp
                active_signals = [
                    signal for signal in signals 
                    if pd.Timestamp(signal.timestamp) <= timestamp and 
                    (not signal.expiry or pd.Timestamp(signal.expiry) > timestamp)
                ]
                if active_signals:
                    current_sentiment_signals[symbol] = active_signals
            
            # Update portfolio value
            portfolio_value_before = self.calculate_portfolio_value(current_prices)
            
            # Execute strategy
            orders = strategy(timestamp, current_prices, self.portfolio, current_sentiment_signals)
            
            # Process orders
            if orders:
                for order in orders:
                    self.execute_order(order, current_prices, commission)
            
            # Update portfolio value
            portfolio_value_after = self.calculate_portfolio_value(current_prices)
            
            # Calculate return
            if portfolio_value_before > 0:
                daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
            else:
                daily_return = 0
            
            # Track equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': float(portfolio_value_after)
            })
            
            # Track sentiment correlation
            for symbol, signals in current_sentiment_signals.items():
                if signals:
                    # Use average sentiment score
                    avg_sentiment = np.mean([
                        signal.strength * (1 if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else -1)
                        for signal in signals
                    ])
                    sentiment_values.append(avg_sentiment)
                    returns.append(daily_return)
                    
                    # Track correlation between sentiment and subsequent returns
                    self.sentiment_trade_correlations.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'sentiment': avg_sentiment,
                        'return': daily_return
                    })
        
        # Calculate sentiment-specific metrics
        self.calculate_sentiment_metrics()
        
        # Calculate overall performance metrics
        metrics = self.calculate_performance_metrics(equity_curve)
        
        # Add sentiment metrics
        metrics.update(self.sentiment_metrics)
        
        return metrics
    
    def calculate_sentiment_metrics(self) -> None:
        """
        Calculate sentiment-specific performance metrics.
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate sentiment-return correlation
        if self.sentiment_trade_correlations:
            df = pd.DataFrame(self.sentiment_trade_correlations)
            
            if len(df) > 5:  # Need enough data for meaningful correlation
                # Calculate correlation
                correlation, p_value = stats.pearsonr(df['sentiment'], df['return'])
                
                metrics['sentiment_return_correlation'] = correlation
                metrics['sentiment_return_correlation_p_value'] = p_value
                
                # Calculate returns by sentiment regime
                df['sentiment_regime'] = pd.cut(
                    df['sentiment'],
                    bins=[-1.0, -0.5, -0.2, 0.2, 0.5, 1.0],
                    labels=['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
                )
                
                # Calculate average returns by regime
                regime_returns = df.groupby('sentiment_regime')['return'].mean().to_dict()
                metrics['returns_by_sentiment_regime'] = regime_returns
                
                # Calculate hit rate by sentiment strength
                df['profitable'] = df['return'] > 0
                
                # Group by sentiment direction
                df['sentiment_direction'] = np.where(df['sentiment'] > 0, 'bullish', 'bearish')
                
                # Calculate hit rate for bullish sentiment
                bullish_df = df[df['sentiment_direction'] == 'bullish']
                if len(bullish_df) > 0:
                    bullish_hit_rate = bullish_df[bullish_df['profitable']].shape[0] / bullish_df.shape[0]
                    metrics['bullish_sentiment_hit_rate'] = bullish_hit_rate
                
                # Calculate hit rate for bearish sentiment
                bearish_df = df[df['sentiment_direction'] == 'bearish']
                if len(bearish_df) > 0:
                    bearish_hit_rate = bearish_df[~bearish_df['profitable']].shape[0] / bearish_df.shape[0]
                    metrics['bearish_sentiment_hit_rate'] = bearish_hit_rate
                
                # Calculate average return by sentiment strength
                df['sentiment_strength'] = df['sentiment'].abs()
                df['sentiment_strength_bin'] = pd.cut(
                    df['sentiment_strength'],
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
                )
                
                strength_returns = df.groupby('sentiment_strength_bin')['return'].mean().to_dict()
                metrics['returns_by_sentiment_strength'] = strength_returns
        
        # Store metrics
        self.sentiment_metrics = metrics
    
    def plot_sentiment_vs_price(self, symbol: str, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot sentiment data alongside price data.
        
        Args:
            symbol: Symbol to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if symbol not in self.sentiment_data or symbol not in self.price_data:
            logger.warning(f"Missing data for {symbol}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data available for {symbol}", ha='center', va='center')
            return fig
        
        sentiment_df = self.sentiment_data[symbol].copy()
        price_df = self.price_data[symbol].copy()
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot price
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='tab:blue')
        ax1.plot(price_df.index, price_df['close'], color='tab:blue', label='Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sentiment Score', color='tab:red')
        
        # Plot sentiment
        sentiment_df = sentiment_df.set_index('timestamp')
        ax2.scatter(sentiment_df.index, sentiment_df['compound_score'], color='tab:red', alpha=0.5, label='Sentiment')
        
        # Plot sentiment moving average
        if len(sentiment_df) > 5:
            window = min(5, len(sentiment_df) // 2)
            sentiment_df['ma'] = sentiment_df['compound_score'].rolling(window=window).mean()
            ax2.plot(sentiment_df.index, sentiment_df['ma'], color='tab:red', linestyle='--', label=f'Sentiment MA({window})')
        
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Add signals if available
        if symbol in self.sentiment_signals:
            signals = self.sentiment_signals[symbol]
            
            for signal in signals:
                if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                    marker = '^'
                    color = 'green'
                    y_pos = price_df['low'].min() * 0.98
                else:
                    marker = 'v'
                    color = 'red'
                    y_pos = price_df['high'].max() * 1.02
                
                ax1.scatter(
                    pd.Timestamp(signal.timestamp), 
                    y_pos,
                    marker=marker, 
                    s=100, 
                    color=color, 
                    alpha=0.7
                )
        
        # Add title and legend
        plt.title(f'{symbol} Price and Sentiment')
        fig.tight_layout()
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return fig
    
    def plot_sentiment_return_correlation(self, figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot correlation between sentiment and returns.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.sentiment_trade_correlations:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No sentiment-return correlation data available", ha='center', va='center')
            return fig
        
        df = pd.DataFrame(self.sentiment_trade_correlations)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        sns.scatterplot(
            data=df,
            x='sentiment',
            y='return',
            hue='symbol' if len(df['symbol'].unique()) <= 5 else None,
            alpha=0.6,
            ax=ax
        )
        
        # Add regression line
        sns.regplot(
            data=df,
            x='sentiment',
            y='return',
            scatter=False,
            line_kws={'color': 'red'},
            ax=ax
        )
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(df['sentiment'], df['return'])
        
        # Add correlation text
        ax.text(
            0.05, 0.95,
            f'Correlation: {correlation:.2f} (p={p_value:.3f})',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )
        
        # Add title and labels
        plt.title('Sentiment vs. Returns Correlation')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def plot_returns_by_sentiment_regime(self, figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot returns by sentiment regime.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.sentiment_metrics.get('returns_by_sentiment_regime'):
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No returns by sentiment regime data available", ha='center', va='center')
            return fig
        
        regime_returns = self.sentiment_metrics['returns_by_sentiment_regime']
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'regime': list(regime_returns.keys()),
            'return': list(regime_returns.values())
        })
        
        # Order regimes
        regime_order = ['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
        df['regime'] = pd.Categorical(df['regime'], categories=regime_order, ordered=True)
        df = df.sort_values('regime')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        bars = sns.barplot(
            data=df,
            x='regime',
            y='return',
            palette='RdYlGn',
            ax=ax
        )
        
        # Add value labels
        for i, bar in enumerate(bars.patches):
            value = df.iloc[i]['return']
            text = f'{value:.2%}'
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_height() + 0.002 if value >= 0 else bar.get_height() - 0.01
            ax.text(text_x, text_y, text, ha='center', va='bottom' if value >= 0 else 'top')
        
        # Add title and labels
        plt.title('Average Returns by Sentiment Regime')
        plt.xlabel('Sentiment Regime')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        return fig
    
    def generate_sentiment_report(self, output_dir: str) -> None:
        """
        Generate a comprehensive sentiment analysis report.
        
        Args:
            output_dir: Directory to save report files
        """
        import os
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sentiment vs price plots for each symbol
        for symbol in self.sentiment_data.keys():
            if symbol in self.price_data:
                fig = self.plot_sentiment_vs_price(symbol)
                fig.savefig(os.path.join(output_dir, f'{symbol}_sentiment_vs_price.png'))
                plt.close(fig)
        
        # Generate correlation plot
        fig = self.plot_sentiment_return_correlation()
        fig.savefig(os.path.join(output_dir, 'sentiment_return_correlation.png'))
        plt.close(fig)
        
        # Generate returns by regime plot
        fig = self.plot_returns_by_sentiment_regime()
        fig.savefig(os.path.join(output_dir, 'returns_by_sentiment_regime.png'))
        plt.close(fig)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-card {{ background: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                .metric-name {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Sentiment Backtest Report</h1>
            
            <h2>Performance Metrics</h2>
            <div class="metrics">
        """
        
        # Add overall metrics
        for key, value in self.sentiment_metrics.items():
            if isinstance(value, dict):
                continue
            
            # Format value
            if isinstance(value, float):
                if 'correlation' in key:
                    formatted_value = f"{value:.2f}"
                    class_name = "positive" if value > 0 else "negative"
                elif 'rate' in key:
                    formatted_value = f"{value:.2%}"
                    class_name = "positive" if value > 0.5 else "negative"
                else:
                    formatted_value = f"{value:.4f}"
                    class_name = ""
            else:
                formatted_value = str(value)
                class_name = ""
            
            html_content += f"""
            <div class="metric-card">
                <div class="metric-value {class_name}">{formatted_value}</div>
                <div class="metric-name">{key.replace('_', ' ').title()}</div>
            </div>
            """
        
        html_content += """
            </div>
            
            <h2>Returns by Sentiment Regime</h2>
            <img src="returns_by_sentiment_regime.png" alt="Returns by Sentiment Regime">
            
            <h2>Sentiment-Return Correlation</h2>
            <img src="sentiment_return_correlation.png" alt="Sentiment-Return Correlation">
            
            <h2>Sentiment vs Price Charts</h2>
        """
        
        # Add sentiment vs price charts
        for symbol in self.sentiment_data.keys():
            if symbol in self.price_data:
                html_content += f"""
                <h3>{symbol}</h3>
                <img src="{symbol}_sentiment_vs_price.png" alt="{symbol} Sentiment vs Price">
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML report
        with open(os.path.join(output_dir, 'sentiment_report.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated sentiment report in {output_dir}")
